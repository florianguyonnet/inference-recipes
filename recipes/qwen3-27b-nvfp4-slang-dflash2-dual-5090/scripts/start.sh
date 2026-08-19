#!/usr/bin/env bash
set -euo pipefail

# Start the SGLang + DFlash2 server (Docker) for Qwen3.8-27B-NVFP4.
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

IMAGE="${IMAGE:-sglang-dflash2:local}"
NAME="${CONTAINER_NAME:-sglang-dflash2}"
HOST_HF="${HF_HOME:-${PROJECT_DIR}/cache/huggingface}"
PORT="${SGLANG_PORT:-30100}"
mkdir -p "${HOST_HF}" "${HOME}/.triton"

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
    --model-path "${HF_REPO_ID:-Inferact/Qwen3.8-27B-NVFP4}"
    --served-model-name "${SERVED_MODEL_NAME:-Qwen3.8-27B-NVFP4}"
    --trust-remote-code
    --host "${SGLANG_HOST:-0.0.0.0}"
    --port "${PORT}"
    --tp "${TP:-2}"
    --mem-fraction-static "${MEM_FRACTION:-0.90}"
    --attention-backend "${ATTENTION_BACKEND:-flashinfer}"
    --chunked-prefill-size "${CHUNKED_PREFILL:-2048}"
    --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}"
    --mamba-ssm-dtype "${MAMBA_SSM_DTYPE:-bfloat16}"
    --mamba-full-memory-ratio "${MAMBA_FULL_MEMORY_RATIO:-8.26}"
    --context-length "${CONTEXT_LENGTH:-262144}"
    --mm-feature-transport "${MM_TRANSPORT:-cpu}"
    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder
    --sampling-defaults model
)
[[ -n "${MAMBA_CACHE_SIZE:-}" ]] && ARGS+=(--max-mamba-cache-size "${MAMBA_CACHE_SIZE}")
[[ -n "${MAX_RUNNING_REQUESTS:-}" ]] && ARGS+=(--max-running-requests "${MAX_RUNNING_REQUESTS}")

case "${SPEC:-dflash}" in
    dflash)
        ARGS+=(
            --speculative-algorithm DFLASH
            --speculative-draft-model-path "${DFLASH_DRAFT_ID:-incoai/Qwen3.8-27B-DFlash2}"
            --speculative-num-draft-tokens "${NUM_DRAFT_TOKENS:-8}"
            --speculative-draft-model-quantization unquant
            --speculative-draft-attention-backend "${DRAFT_ATTENTION_BACKEND:-flashinfer}"
            --linear-attn-verify-backend "${LINEAR_ATTN_VERIFY_BACKEND:-triton}"
        )
        ;;
    dspark)
        # Same image and same checkpoint as the dflash profile: the only
        # controlled way to compare the two drafters (see README).
        ARGS+=(
            --speculative-algorithm DSPARK
            --speculative-draft-model-path "${DSPARK_DRAFT_ID:-RadixArk/Qwen3.8-27B-DSpark}"
            --speculative-dspark-block-size "${DSPARK_BLOCK_SIZE:-7}"
            --speculative-draft-model-quantization unquant
            --speculative-draft-attention-backend "${DRAFT_ATTENTION_BACKEND:-flashinfer}"
            --linear-attn-verify-backend "${LINEAR_ATTN_VERIFY_BACKEND:-triton}"
        )
        ;;
    off|none|"") ;;
    *) echo "ERROR: unknown SPEC=${SPEC} (dflash|dspark|off)"; exit 1 ;;
esac

# Appended last so a sweep flag wins over everything above.
if [[ -n "${EXTRA_FLAGS:-}" ]]; then
    # shellcheck disable=SC2206
    ARGS+=(${EXTRA_FLAGS})
fi

echo "==> Starting ${NAME} (${IMAGE}) TP=${TP:-2} SPEC=${SPEC:-dflash} on :${PORT}"
docker run -d \
    --name "${NAME}" \
    --network host \
    --ipc host \
    --gpus "${GPUS:-all}" \
    --shm-size 32g \
    -e HF_HOME=/root/.cache/huggingface \
    -e TRITON_CACHE_DIR=/root/.triton \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
    -v "${HOST_HF}:/root/.cache/huggingface" \
    -v "${HOME}/.triton:/root/.triton" \
    -v "${PROJECT_DIR}/../../tools:/recipescripts:ro" \
    "${IMAGE}" \
    python3 -m sglang.launch_server "${ARGS[@]}" \
    >/dev/null

echo "==> Container started. Waiting for http://localhost:${PORT}/health ..."
echo "    (follow with: docker logs -f ${NAME})"
for i in $(seq 1 360); do
    if ! docker ps --format '{{.Names}}' | grep -qx "${NAME}"; then
        echo "ERROR: container died during startup. Last log lines:"
        docker logs --tail 40 "${NAME}" 2>&1
        exit 1
    fi
    if curl -fsS --max-time 5 "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "==> Server is UP on :${PORT} (model: ${SERVED_MODEL_NAME:-Qwen3.8-27B-NVFP4})"
        exit 0
    fi
    sleep 5
done
echo "ERROR: server did not become healthy within 30 min"
docker logs --tail 40 "${NAME}" 2>&1
exit 1
