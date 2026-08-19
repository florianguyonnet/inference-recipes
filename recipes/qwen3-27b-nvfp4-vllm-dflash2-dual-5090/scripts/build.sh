#!/usr/bin/env bash
set -euo pipefail

# Build the vLLM + DFlash2 image (nightly wheel + the two open DFlash2 PRs).
# Usage: ./scripts/build.sh [--pull]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -f "${PROJECT_DIR}/.env" ]]; then
    # shellcheck source=/dev/null
    set -a; source "${PROJECT_DIR}/.env"; set +a
fi

IMAGE="${IMAGE:-vllm-dflash2:local}"
echo "==> Building ${IMAGE} from ${PROJECT_DIR}/Dockerfile"
docker build "$@" -t "${IMAGE}" -f "${PROJECT_DIR}/Dockerfile" "${PROJECT_DIR}"
echo "==> Built ${IMAGE}"
