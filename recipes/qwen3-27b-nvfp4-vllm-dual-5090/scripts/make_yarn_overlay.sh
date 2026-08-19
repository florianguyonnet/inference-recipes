#!/usr/bin/env bash
set -euo pipefail

# Build the YaRN 512k overlay inside the HF cache: relative symlinks to the
# downloaded snapshot + the patched config.json tracked in this recipe
# (YaRN rope_parameters, factor 2.0: 262k -> 512k context).
#
# The overlay lives INSIDE the cache (${HF_HOME}/overlay-yarn512k) and uses
# relative links, so it resolves identically on the host and in a container
# with the cache mounted at /root/.cache/huggingface.
#
# Usage: ./scripts/make_yarn_overlay.sh [repo_id]
# Run download_model.sh first. Safe to re-run.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_DIR}/.env"

if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck source=/dev/null
    set -a; source "${ENV_FILE}"; set +a
fi

HF_HOME="${HF_HOME:-${PROJECT_DIR}/cache/huggingface}"
REPO_ID="${1:-${HF_REPO_ID:-unsloth/Qwen3.8-27B-NVFP4}}"
OVERLAY_DIR="${HF_HOME}/overlay-yarn512k"
PATCHED_CONFIG="${PROJECT_DIR}/models/Qwen3.8-27B-NVFP4-yarn512k/config.json"

if [[ ! -f "${PATCHED_CONFIG}" ]]; then
    echo "ERROR: ${PATCHED_CONFIG} missing (the patched YaRN config is tracked in the recipe)."
    exit 1
fi

CACHE_NAME="models--$(echo "${REPO_ID}" | sed 's|/|--|g')"
REF="${HF_HOME}/hub/${CACHE_NAME}/refs/main"
if [[ -f "${REF}" ]]; then
    SNAPSHOT_DIR="${HF_HOME}/hub/${CACHE_NAME}/snapshots/$(cat "${REF}")"
else
    SNAPSHOT_DIR="$(ls -d "${HF_HOME}/hub/${CACHE_NAME}"/snapshots/*/ 2>/dev/null | head -1)"
fi
if [[ ! -d "${SNAPSHOT_DIR}" ]]; then
    echo "ERROR: ${REPO_ID} not found in ${HF_HOME}. Run scripts/download_model.sh first."
    exit 1
fi

# refs/main follows the Hub: anything that resolves the repo id against the Hub
# in this cache (a test run, a container fetching the repo id instead of the
# overlay) moves it to a newer revision whose weights may not be downloaded.
# Without this check the overlay is built from configs alone and vLLM crash-loops
# on "Cannot find any model weights".
if ! compgen -G "${SNAPSHOT_DIR}/*.safetensors" >/dev/null; then
    echo "ERROR: no *.safetensors in ${SNAPSHOT_DIR}"
    echo "       refs/main points at an incompletely downloaded revision."
    echo "       Fix: ./scripts/download_model.sh, or point refs/main back at a"
    echo "       complete snapshot under ${HF_HOME}/hub/${CACHE_NAME}/snapshots/."
    exit 1
fi

echo "==> Overlay : ${OVERLAY_DIR}"
echo "    Snapshot: ${SNAPSHOT_DIR}"

mkdir -p "${OVERLAY_DIR}"
rm -f "${OVERLAY_DIR}"/*

linked=0
for f in "${SNAPSHOT_DIR}"/*; do
    base="$(basename "${f}")"
    [[ "${base}" == "config.json" ]] && continue   # replaced by the patched YaRN config
    ln -sfn "$(realpath --relative-to="${OVERLAY_DIR}" "${f}")" "${OVERLAY_DIR}/${base}"
    linked=$((linked + 1))
done
cp "${PATCHED_CONFIG}" "${OVERLAY_DIR}/config.json"

# generation_config.json stays symlinked: its temperature 1.0 / top_p 0.95 /
# top_k 20 is the thinking-mode config the model card recommends, and vLLM
# applies it to requests that send no sampling params.

echo "==> Done: ${linked} files symlinked (relative), patched config.json copied."
