#!/usr/bin/env bash
set -euo pipefail

# Pre-download the checkpoints through the container (optional: start.sh
# downloads them automatically on first run). Useful to stage weights ahead
# of time or with an HF token.
#
# Usage: ./scripts/download_model.sh [repo_id ...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -f "${PROJECT_DIR}/.env" ]]; then
    # shellcheck source=/dev/null
    set -a; source "${PROJECT_DIR}/.env"; set +a
fi

IMAGE="${IMAGE:-lmsysorg/sglang:qwen38-27b}"
HOST_HF="${HF_HOME:-${PROJECT_DIR}/cache/huggingface}"
mkdir -p "${HOST_HF}"

if [[ $# -gt 0 ]]; then
    MODELS=("$@")
else
    MODELS=("${HF_REPO_ID:-RadixArk/Qwen3.8-27B-NVFP4}"
            "${DSPARK_DRAFT_ID:-RadixArk/Qwen3.8-27B-DSpark}")
fi

for M in "${MODELS[@]}"; do
    echo "==> Downloading ${M}..."
    docker run --rm \
        -e HF_HOME=/root/.cache/huggingface \
        -e HF_TOKEN="${HF_TOKEN:-}" \
        -e HF_XET_HIGH_PERFORMANCE=1 \
        -v "${HOST_HF}:/root/.cache/huggingface" \
        "${IMAGE}" \
        python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${M}'); print('done')"
done

echo "==> Models ready under ${HOST_HF}/hub/"
