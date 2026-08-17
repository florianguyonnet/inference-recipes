#!/usr/bin/env bash
set -euo pipefail

# Set up the shared tools environment (lm-evaluation-harness + client deps).

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v uv &>/dev/null; then
    echo "ERROR: uv is not installed. Install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

uv venv --python 3.12 "${TOOLS_DIR}/.venv"
uv pip install --python "${TOOLS_DIR}/.venv/bin/python" \
    "lm-eval" \
    aiohttp tenacity

echo "==> Done. Tools venv at ${TOOLS_DIR}/.venv"
