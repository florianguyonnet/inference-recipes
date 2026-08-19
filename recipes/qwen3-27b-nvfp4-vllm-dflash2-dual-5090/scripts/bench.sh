#!/usr/bin/env bash
set -euo pipefail

# Speed suite against the running server. The client is the shared
# tools/benchmark_agent.py, run in a throwaway container from the same image.
# Per scenario it also prints the DFlash2 acceptance length measured from the
# delta of vLLM's spec-decode counters over that scenario.
#
# Usage: ./scripts/bench.sh [label]        # label names the results dir
# Results: results/<label>/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOLS_DIR="$(cd "${PROJECT_DIR}/../../tools" && pwd)"
if [[ -f "${PROJECT_DIR}/.env" ]]; then
    # shellcheck source=/dev/null
    set -a; source "${PROJECT_DIR}/.env"; set +a
fi

IMAGE="${IMAGE:-vllm-dflash2:local}"
PORT="${VLLM_PORT:-8100}"
URL="http://localhost:${PORT}"
MODEL="${SERVED_MODEL_NAME:-Qwen3.8-27B-NVFP4}"
LABEL="${1:-dflash2}"
OUT="${PROJECT_DIR}/results/${LABEL}"
mkdir -p "${OUT}"

if ! curl -fsS --max-time 5 "${URL}/health" >/dev/null 2>&1; then
    echo "ERROR: no healthy server on ${URL}. Run scripts/start.sh first."
    exit 1
fi

# accepted+drafts counters; "0 0" when the server runs without a drafter.
counters() {
    curl -fsS --max-time 10 "${URL}/metrics" 2>/dev/null | awk '
        /^vllm:spec_decode_num_accepted_tokens_total/ {a += $2}
        /^vllm:spec_decode_num_drafts_total/          {d += $2}
        END {printf "%d %d\n", a, d}'
}

# bench <name> <input-len> <output-len> <n-requests> <concurrency>
bench() {
    local name="$1" ilen="$2" olen="$3" n="$4" c="$5" before after
    echo "==> [${LABEL}] ${name}"
    before="$(counters)"
    docker run --rm --network host -v "${TOOLS_DIR}:/tools:ro" \
        --entrypoint python3 "${IMAGE}" /tools/benchmark_agent.py \
        --url "${URL}" --model "${MODEL}" --input-lens "${ilen}" \
        --output-len "${olen}" --num-requests "${n}" --concurrency "${c}" \
        | tee "${OUT}/${name}.txt"
    after="$(counters)"
    python3 - "${before}" "${after}" <<'PY' | tee -a "${OUT}/${name}.txt"
import sys
a0, d0 = map(int, sys.argv[1].split())
a1, d1 = map(int, sys.argv[2].split())
if d1 > d0:
    print(f"  acceptance length: {(a1 - a0) / (d1 - d0) + 1:.2f} tokens/verify "
          f"({d1 - d0} drafts)")
else:
    print("  acceptance length: n/a (no drafter)")
PY
}

bench prefill-32k-cold 32000 64 1 1
bench solo-2k 2048 256 3 1
bench solo-8k 8192 256 3 1
for c in 2 4 8 16; do
    bench "par${c}-8k" 8192 256 "${c}" "${c}"
done

echo "==> Done. Results in ${OUT}/"
