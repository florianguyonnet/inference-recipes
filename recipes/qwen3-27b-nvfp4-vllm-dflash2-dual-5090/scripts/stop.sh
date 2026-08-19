#!/usr/bin/env bash
set -euo pipefail

# Stop the container started by scripts/start.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -f "${PROJECT_DIR}/.env" ]]; then
    # shellcheck source=/dev/null
    set -a; source "${PROJECT_DIR}/.env"; set +a
fi
NAME="${CONTAINER_NAME:-vllm-dflash2}"

if docker ps --format '{{.Names}}' | grep -qx "${NAME}"; then
    echo "==> Stopping ${NAME}..."
    docker stop "${NAME}" >/dev/null
fi
if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
    docker rm "${NAME}" >/dev/null
fi
echo "Stopped."
