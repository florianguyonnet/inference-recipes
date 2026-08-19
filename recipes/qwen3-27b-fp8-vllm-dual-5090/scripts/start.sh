#!/usr/bin/env bash
set -euo pipefail

# Start the vLLM server (Docker) for Qwen3.8-27B-FP8 at 512k context.
#
# Usage: ./scripts/start.sh [profile.env]   (default: profiles/fp8-full.env)
# Config layers: environment -> .env -> profile (the profile wins; put sweep
# overrides in a profile copy, not in the environment).
# First start downloads the checkpoint into the mounted HF cache (~23 GiB).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${PROJECT_DIR}/.env" ]]; then
    # shellcheck source=/dev/null
    set -a; source "${PROJECT_DIR}/.env"; set +a
fi

PROFILE="${1:-profiles/fp8-full.env}"
if [[ -f "${PROFILE}" ]]; then
    :
elif [[ -f "${PROJECT_DIR}/${PROFILE}" ]]; then
    PROFILE="${PROJECT_DIR}/${PROFILE}"
else
    echo "ERROR: profile not found: ${1:-<default>}"
    exit 1
fi
echo "==> Loading profile ${PROFILE}"
# shellcheck source=/dev/null
set -a; source "${PROFILE}"; set +a

IMAGE="${IMAGE:-vllm/vllm-openai:v0.27.1}"
NAME="${CONTAINER_NAME:-vllm-qwen-fp8}"
HOST_HF="${HF_HOME:-${PROJECT_DIR}/cache/huggingface}"
PORT="${VLLM_PORT:-8000}"
mkdir -p "${HOST_HF}"

# YaRN 512k overlay (relative symlinks inside the cache + patched config).
if [[ "${YARN_OVERLAY:-1}" == "1" ]]; then
    "${SCRIPT_DIR}/make_yarn_overlay.sh"
    MODEL_PATH="/root/.cache/huggingface/overlay-fp8-yarn512k"
else
    MODEL_PATH="${HF_REPO_ID:-Qwen/Qwen3.8-27B-FP8}"
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
    if docker ps --format '{{.Names}}' | grep -qx "${NAME}"; then
        echo "Container ${NAME} is already running."
        exit 0
    fi
    echo "==> Removing stopped container ${NAME}"
    docker rm "${NAME}" >/dev/null
fi

# This is a comparison recipe, not a service: refuse to start when another
# process already owns the cards. Without this the engine fails on "Free memory
# on device cuda:0 ... is less than desired GPU memory utilization" and, with a
# restart policy, loops on it. FORCE=1 to override.
if [[ "${FORCE:-0}" != "1" ]]; then
    BUSY="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | head -4)"
    if [[ -n "${BUSY}" ]]; then
        echo "ERROR: the GPUs are already in use:"
        echo "${BUSY}" | sed 's/^/       /'
        echo "       stop that server first (or FORCE=1 to try anyway)."
        exit 1
    fi
fi

ARGS=(
    --model "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME:-Qwen3.8-27B-FP8}"
    --host "${VLLM_HOST:-0.0.0.0}"
    --port "${PORT}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-2}"
    --kv-cache-dtype fp8_e4m3
    --attention-backend "${ATTENTION_BACKEND:-flashinfer}"
    --max-model-len "${MAX_MODEL_LEN:-524288}"
    --max-num-seqs "${MAX_NUM_SEQS:-16}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.94}"
    --safetensors-load-strategy prefetch
    --enable-prefix-caching
    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder
    --enable-auto-tool-choice
)
[[ -n "${KV_CACHE_MEMORY:-}" ]] && ARGS+=(--kv-cache-memory "${KV_CACHE_MEMORY}")
# QUANTIZATION stays unset: the unsloth checkpoint is compressed-tensors
# (NVFP4 W4A4 MLP + FP8 attention), auto-detected from config.json.
[[ -n "${QUANTIZATION:-}" ]] && ARGS+=(--quantization "${QUANTIZATION}")
[[ -n "${SPECULATIVE_CONFIG:-}" ]] && ARGS+=(--speculative-config "${SPECULATIVE_CONFIG}")
if [[ -n "${EXTRA_FLAGS:-}" ]]; then
    # shellcheck disable=SC2206
    ARGS+=(${EXTRA_FLAGS})
fi

echo "==> Starting ${NAME} (${IMAGE}) TP=${TENSOR_PARALLEL_SIZE:-2} on :${PORT}"
docker run -d \
    --name "${NAME}" \
    --network host \
    --ipc host \
    --health-cmd "python3 -c \"import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health', timeout=5)\"" \
    --health-interval 30s \
    --health-timeout 10s \
    --health-start-period 10m \
    --health-retries 3 \
    --gpus "${GPUS:-all}" \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_XET_HIGH_PERFORMANCE=1 \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
    -e NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
    -e VLLM_SKIP_P2P_CHECK="${VLLM_SKIP_P2P_CHECK:-1}" \
    -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    -e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}" \
    -e VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-1}" \
    -e FLASHINFER_EXTRA_CUDAFLAGS="${FLASHINFER_EXTRA_CUDAFLAGS:--DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK=1}" \
    -e MAX_JOBS="${MAX_JOBS:-4}" \
    -v "${HOST_HF}:/root/.cache/huggingface" \
    --entrypoint python3 \
    "${IMAGE}" \
    -m vllm.entrypoints.openai.api_server "${ARGS[@]}" \
    >/dev/null

# A few batched requests before the server takes traffic: vLLM JIT-compiles some
# Triton kernels on first use (batch_memcpy_kernel), which lands as a latency
# spike on a real request otherwise.
warmup() {
    local filler payload i
    filler="$(printf 'The quick brown fox jumps over the lazy dog. %.0s' $(seq 1 200))"
    payload="$(printf '{"model":"%s","messages":[{"role":"user","content":"%s Reply with OK."}],"max_tokens":32,"temperature":1.0}' \
        "${SERVED_MODEL_NAME:-Qwen3.8-27B-FP8}" "${filler}")"
    echo "==> Warming up (4 batched requests)..."
    for i in 1 2 3 4; do
        curl -fsS --max-time 180 "http://localhost:${PORT}/v1/chat/completions" \
            -H 'Content-Type: application/json' -d "${payload}" >/dev/null 2>&1 &
    done
    wait
}

echo "==> Container started. Waiting for http://localhost:${PORT}/health ..."
echo "    (warm start ~2.5 min, cold ~6 min; follow with: docker logs -f ${NAME})"
for i in $(seq 1 180); do
    if ! docker ps --format '{{.Names}}' | grep -qx "${NAME}"; then
        echo "ERROR: container died during startup. Last log lines:"
        docker logs --tail 40 "${NAME}" 2>&1
        exit 1
    fi
    if curl -fsS --max-time 5 "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "==> Server is UP on :${PORT} (model: ${SERVED_MODEL_NAME:-Qwen3.8-27B-FP8})"
        warmup
        exit 0
    fi
    sleep 5
done
echo "ERROR: server did not become healthy within 15 min"
docker logs --tail 40 "${NAME}" 2>&1
exit 1
