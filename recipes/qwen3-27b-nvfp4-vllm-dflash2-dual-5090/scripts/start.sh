#!/usr/bin/env bash
set -euo pipefail

# Start the vLLM + DFlash2 server (Docker) for Qwen3.8-27B-NVFP4 at 262k context.
#
# Usage: ./scripts/start.sh [profile.env]   (default: profiles/dflash2-tp2.env)
# Config layers: .env -> profile (the profile wins; put variants in a profile
# copy). Build the image first: ./scripts/build.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${PROJECT_DIR}/.env" ]]; then
    # shellcheck source=/dev/null
    set -a; source "${PROJECT_DIR}/.env"; set +a
fi

PROFILE="${1:-profiles/dflash2-tp2.env}"
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

IMAGE="${IMAGE:-vllm-dflash2:local}"
NAME="${CONTAINER_NAME:-vllm-dflash2}"
HOST_HF="${HF_HOME:-${PROJECT_DIR}/cache/huggingface}"
PORT="${VLLM_PORT:-8100}"
MODEL="${HF_REPO_ID:-Inferact/Qwen3.8-27B-NVFP4}"
mkdir -p "${HOST_HF}"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "ERROR: image ${IMAGE} not found. Run ./scripts/build.sh first."
    exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
    if docker ps --format '{{.Names}}' | grep -qx "${NAME}"; then
        echo "Container ${NAME} is already running."
        exit 0
    fi
    echo "==> Removing stopped container ${NAME}"
    docker rm "${NAME}" >/dev/null
fi

ARGS=(
    --model "${MODEL}"
    --served-model-name "${SERVED_MODEL_NAME:-Qwen3.8-27B-NVFP4}"
    --host "${VLLM_HOST:-0.0.0.0}"
    --port "${PORT}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-2}"
    --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}"
    --attention-backend "${ATTENTION_BACKEND:-flashinfer}"
    --max-model-len "${MAX_MODEL_LEN:-262144}"
    --max-num-seqs "${MAX_NUM_SEQS:-16}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.94}"
    --safetensors-load-strategy prefetch
    --enable-prefix-caching
    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder
    --enable-auto-tool-choice
)
if [[ "${SPEC:-on}" != "off" ]]; then
    # A profile can set SPECULATIVE_CONFIG to the full JSON instead.
    SPEC_JSON="${SPECULATIVE_CONFIG:-}"
    if [[ -z "${SPEC_JSON}" ]]; then
        SPEC_JSON="{\"method\": \"dflash\", \"model\": \"${DRAFT_REPO_ID:-incoai/Qwen3.8-27B-DFlash2}\", \"num_speculative_tokens\": ${NUM_SPECULATIVE_TOKENS:-7}}"
    fi
    ARGS+=(--speculative-config "${SPEC_JSON}")
fi
if [[ -n "${EXTRA_FLAGS:-}" ]]; then
    # shellcheck disable=SC2206
    ARGS+=(${EXTRA_FLAGS})
fi

echo "==> Starting ${NAME} (${IMAGE}) TP=${TENSOR_PARALLEL_SIZE:-2} spec=${SPEC:-on} on :${PORT}"
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
# Triton kernels on first use, which lands as a latency spike otherwise.
warmup() {
    local filler payload i
    filler="$(printf 'The quick brown fox jumps over the lazy dog. %.0s' $(seq 1 200))"
    payload="$(printf '{"model":"%s","messages":[{"role":"user","content":"%s Reply with OK."}],"max_tokens":32,"temperature":1.0}' \
        "${SERVED_MODEL_NAME:-Qwen3.8-27B-NVFP4}" "${filler}")"
    echo "==> Warming up (4 batched requests)..."
    for i in 1 2 3 4; do
        curl -fsS --max-time 180 "http://localhost:${PORT}/v1/chat/completions" \
            -H 'Content-Type: application/json' -d "${payload}" >/dev/null 2>&1 &
    done
    wait
}

echo "==> Container started. Waiting for http://localhost:${PORT}/health ..."
echo "    (follow with: docker logs -f ${NAME})"
for i in $(seq 1 240); do
    if ! docker ps --format '{{.Names}}' | grep -qx "${NAME}"; then
        echo "ERROR: container died during startup. Last log lines:"
        docker logs --tail 40 "${NAME}" 2>&1
        exit 1
    fi
    if curl -fsS --max-time 5 "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "==> Server is UP on :${PORT} (model: ${SERVED_MODEL_NAME:-Qwen3.8-27B-NVFP4})"
        warmup
        exit 0
    fi
    sleep 5
done
echo "ERROR: server did not become healthy within 20 min"
docker logs --tail 40 "${NAME}" 2>&1
exit 1
