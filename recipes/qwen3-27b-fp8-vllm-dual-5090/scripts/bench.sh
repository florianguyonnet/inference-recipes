#!/usr/bin/env bash
set -euo pipefail

# Speed suite against the running server. The client is the shared
# tools/benchmark_agent.py, run in a throwaway container from the same image.
#
# Usage: ./scripts/bench.sh [label]
# Results: results/<label>/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOLS_DIR="$(cd "${PROJECT_DIR}/../../tools" && pwd)"
if [[ -f "${PROJECT_DIR}/.env" ]]; then
    # shellcheck source=/dev/null
    set -a; source "${PROJECT_DIR}/.env"; set +a
fi

IMAGE="${IMAGE:-vllm/vllm-openai:v0.27.1}"
PORT="${VLLM_PORT:-8001}"
URL="http://localhost:${PORT}"
MODEL="${SERVED_MODEL_NAME:-Qwen3.8-27B-FP8}"
LABEL="${1:-fp8}"
OUT="${PROJECT_DIR}/results/${LABEL}"
mkdir -p "${OUT}"

if ! curl -fsS --max-time 5 "${URL}/health" >/dev/null 2>&1; then
    echo "ERROR: no healthy server on ${URL}. Run scripts/start.sh first."
    exit 1
fi

# bench <name> <input-len> <output-len> <n-requests> <concurrency>
bench() {
    local name="$1"
    echo "==> [${LABEL}] ${name}"
    docker run --rm --network host -v "${TOOLS_DIR}:/tools:ro" \
        --entrypoint python3 "${IMAGE}" /tools/benchmark_agent.py \
        --url "${URL}" --model "${MODEL}" --input-lens "$2" \
        --output-len "$3" --num-requests "$4" --concurrency "$5" \
        | tee "${OUT}/${name}.txt"
}

# Prefill first, on a fresh server, before the prefix cache holds anything.
bench prefill-32k-cold 32000 64 1 1
bench prefill-128k-cold 128000 64 1 1
bench solo-2k 2048 256 3 1
bench solo-8k 8192 256 3 1
for c in 2 4 8 16; do
    bench "par${c}-8k" 8192 256 "${c}" "${c}"
done

echo "==> Done. Results in ${OUT}/"
