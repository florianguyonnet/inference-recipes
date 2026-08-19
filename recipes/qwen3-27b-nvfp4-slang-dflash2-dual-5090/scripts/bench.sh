#!/usr/bin/env bash
set -euo pipefail

# Speed suite against the running server. The client is the shared
# tools/benchmark_agent.py, mounted at /recipescripts by start.sh. Per scenario
# it also prints the mean acceptance length SGLang logged during that scenario.
#
# Usage: ./scripts/bench.sh [label]
# Results: results/<label>/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -f "${PROJECT_DIR}/.env" ]]; then
    # shellcheck source=/dev/null
    set -a; source "${PROJECT_DIR}/.env"; set +a
fi

NAME="${CONTAINER_NAME:-sglang-dflash2}"
PORT="${SGLANG_PORT:-30100}"
URL="http://localhost:${PORT}"
MODEL="${SERVED_MODEL_NAME:-Qwen3.8-27B-NVFP4}"
LABEL="${1:-dflash2}"
OUT="${PROJECT_DIR}/results/${LABEL}"
mkdir -p "${OUT}"

if ! curl -fsS --max-time 5 "${URL}/health" >/dev/null 2>&1; then
    echo "ERROR: no healthy server on ${URL}. Run scripts/start.sh first."
    exit 1
fi

accept_lines() { docker logs "${NAME}" 2>&1 | grep -c "accept len" || true; }
accept_since() {
    # grep exits 1 with no drafter (or before the first decode log line): under
    # pipefail that would abort the suite.
    { docker logs "${NAME}" 2>&1 | grep -oE "accept len: [0-9.]+" || true; } | tail -n +"$(($1 + 1))" \
        | awk '{s += $3; n++} END {if (n) printf "  acceptance length: %.2f over %d steps\n", s/n, n;
                                  else print "  acceptance length: n/a (no drafter)"}'
}

# bench <name> <input-len> <output-len> <n-requests> <concurrency>
bench() {
    local name="$1" ilen="$2" olen="$3" n="$4" c="$5" before
    echo "==> [${LABEL}] ${name}"
    before="$(accept_lines)"
    docker exec "${NAME}" python3 /recipescripts/benchmark_agent.py \
        --url "${URL}" --model "${MODEL}" --input-lens "${ilen}" \
        --output-len "${olen}" --num-requests "${n}" --concurrency "${c}" \
        | tee "${OUT}/${name}.txt"
    accept_since "${before}" | tee -a "${OUT}/${name}.txt"
}

bench prefill-32k-cold 32000 64 1 1
bench solo-2k 2048 256 3 1
bench solo-8k 8192 256 3 1
for c in 2 4 8; do
    bench "par${c}-8k" 8192 256 "${c}" "${c}"
done

echo "==> Done. Results in ${OUT}/"
