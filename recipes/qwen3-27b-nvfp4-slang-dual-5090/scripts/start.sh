#!/usr/bin/env bash
set -euo pipefail

# Start the SGLang server (Docker) for Qwen3.8-27B-NVFP4.
#
# Usage: ./scripts/start.sh [profile.env]   (default: profiles/tp2-dspark.env)
# Config layers: environment -> .env -> profile (the profile wins; put sweep
# overrides in a profile copy, not in the environment).
# First start downloads the checkpoints into the mounted HF cache (~19 GB).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${PROJECT_DIR}/.env" ]]; then
    # shellcheck source=/dev/null
    set -a; source "${PROJECT_DIR}/.env"; set +a
fi

PROFILE="${1:-profiles/tp2-dspark.env}"
if [[ -f "${PROFILE}" ]]; then
    : # already a valid path
elif [[ -f "${PROJECT_DIR}/${PROFILE}" ]]; then
    PROFILE="${PROJECT_DIR}/${PROFILE}"
else
    echo "ERROR: profile not found: ${1:-<default>}"
    exit 1
fi
echo "==> Loading profile ${PROFILE}"
# shellcheck source=/dev/null
set -a; source "${PROFILE}"; set +a

IMAGE="${IMAGE:-lmsysorg/sglang:qwen38-27b}"
NAME="${CONTAINER_NAME:-sglang-qwen38}"
HOST_HF="${HF_HOME:-${PROJECT_DIR}/cache/huggingface}"
PORT="${SGLANG_PORT:-30000}"

mkdir -p "${HOST_HF}" "${HOME}/.triton"

if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
    if docker ps --format '{{.Names}}' | grep -qx "${NAME}"; then
        echo "Container ${NAME} is already running."
        exit 0
    fi
    echo "==> Removing stopped container ${NAME}"
    docker rm "${NAME}" >/dev/null
fi

# Optional generation_config overlay: patch default sampling params (e.g.
# temperature 0.6 instead of the checkpoint's 1.0) without touching the
# downloaded snapshot. Symlinks + patched JSON live in the mounted HF cache.
MODEL_PATH="${HF_REPO_ID:-RadixArk/Qwen3.8-27B-NVFP4}"
if [[ -n "${GEN_CONFIG_TEMP:-}" ]]; then
    SNAP_ROOT="${HOST_HF}/hub/models--$(echo "${MODEL_PATH}" | sed 's|/|--|g')/snapshots"
    SNAP="$(ls -d "${SNAP_ROOT}"/*/ 2>/dev/null | head -1)"
    OV="${HOST_HF}/overlay-genconfig-t${GEN_CONFIG_TEMP}"
    if [[ -n "${SNAP}" && -f "${SNAP}/generation_config.json" ]]; then
        mkdir -p "${OV}"
        rm -f "${OV}"/*
        # Relative symlinks: they must resolve identically on the host and in
        # the container (the cache is mounted at a different path inside).
        for f in "${SNAP}"/*; do
            ln -sfn "$(realpath --relative-to="${OV}" "${f}")" "${OV}/$(basename "${f}")"
        done
        rm -f "${OV}/generation_config.json"
        python3 - "${SNAP}/generation_config.json" "${OV}/generation_config.json" "${GEN_CONFIG_TEMP}" <<'PY'
import json, sys
src, dst, temp = sys.argv[1], sys.argv[2], float(sys.argv[3])
cfg = json.load(open(src))
cfg["temperature"] = temp
json.dump(cfg, open(dst, "w"), indent=2)
print(f"patched generation_config: temperature={temp}")
PY
        MODEL_PATH="/root/.cache/huggingface/overlay-genconfig-t${GEN_CONFIG_TEMP}"
        echo "==> Using generation_config overlay (temp=${GEN_CONFIG_TEMP}): ${MODEL_PATH}"
    else
        echo "WARNING: snapshot not found for overlay; using repo id as-is"
    fi
fi

ARGS=(
    --model-path "${MODEL_PATH}"
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
    --mamba-radix-cache-strategy "${MAMBA_STRATEGY:-extra_buffer_lazy}"
    --context-length "${CONTEXT_LENGTH:-262144}"
    --mm-feature-transport "${MM_TRANSPORT:-cpu}"
    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder
    --sampling-defaults model
)
[[ -n "${MAMBA_CACHE_SIZE:-}" ]] && ARGS+=(--max-mamba-cache-size "${MAMBA_CACHE_SIZE}")
[[ -n "${MAX_RUNNING_REQUESTS:-}" ]] && ARGS+=(--max-running-requests "${MAX_RUNNING_REQUESTS}")

case "${SPEC:-dspark}" in
    dspark)
        # DSpark recipe: separate BF16 drafter,
        # block 7 (verify width 8), GDN verify on triton.
        ARGS+=(
            --speculative-algorithm DSPARK
            --speculative-draft-model-path "${DSPARK_DRAFT_ID:-RadixArk/Qwen3.8-27B-DSpark}"
            --speculative-dspark-block-size "${DSPARK_BLOCK_SIZE:-7}"
            --speculative-draft-model-quantization unquant
            --speculative-draft-attention-backend "${DRAFT_ATTENTION_BACKEND:-flashinfer}"
            --linear-attn-verify-backend "${LINEAR_ATTN_VERIFY_BACKEND:-triton}"
            --min-free-slots-delay 1
        )
        ;;
    off|none|"") ;;
    *) echo "ERROR: unknown SPEC=${SPEC} (dspark|off)"; exit 1 ;;
esac

# Escape hatch for sweep/tuning, appended last so it wins over everything above:
# EXTRA_FLAGS="--foo bar --baz" is split and appended.
if [[ -n "${EXTRA_FLAGS:-}" ]]; then
    # shellcheck disable=SC2206
    ARGS+=(${EXTRA_FLAGS})
fi

echo "==> Starting ${NAME} (${IMAGE}) TP=${TP:-2} SPEC=${SPEC:-dspark} on :${PORT}"

DOCKER_EXTRA=()
# --privileged is only needed for zero-copy vision (cuda_ipc between host and
# container); the default cpu transport does not need it.
[[ "${MM_TRANSPORT:-cpu}" == "cuda_ipc" ]] && DOCKER_EXTRA+=(--privileged)

docker run -d \
    --name "${NAME}" \
    --network host \
    --ipc host \
    "${DOCKER_EXTRA[@]+"${DOCKER_EXTRA[@]}"}" \
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
echo "    (first start downloads ~19 GB of weights; follow with: docker logs -f ${NAME})"
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
