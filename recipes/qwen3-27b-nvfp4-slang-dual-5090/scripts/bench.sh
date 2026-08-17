#!/usr/bin/env bash
set -euo pipefail

# Benchmark the running SGLang container. Client tools run inside the
# container (it has python + aiohttp); raw outputs land in results/<profile>/.
#
# Usage: ./scripts/bench.sh <profile-name>
# Example: ./scripts/bench.sh tp2-dspark

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -f "${PROJECT_DIR}/.env" ]]; then
    # shellcheck source=/dev/null
    set -a; source "${PROJECT_DIR}/.env"; set +a
fi

NAME="${CONTAINER_NAME:-sglang-qwen38}"
PORT="${SGLANG_PORT:-30000}"
URL="http://localhost:${PORT}"
MODEL="${SERVED_MODEL_NAME:-Qwen3.8-27B-NVFP4}"
PROFILE="${1:-bench}"
OUT="${PROJECT_DIR}/results/${PROFILE}"
mkdir -p "${OUT}"

if ! curl -fsS --max-time 5 "${URL}/health" >/dev/null 2>&1; then
    echo "ERROR: no healthy server on ${URL}. Run scripts/start.sh first."
    exit 1
fi

run_py() {
    docker exec "${NAME}" python3 "/recipescripts/$1" "${@:2}"
}

echo "==> [${PROFILE}] cold prefill probe (~27k tokens)"
run_py benchmark_agent.py --url "${URL}" --model "${MODEL}" \
    --input-lens 27000 --output-len 64 --num-requests 1 --concurrency 1 \
    | tee "${OUT}/prefill-27k-cold.txt"

echo "==> [${PROFILE}] solo decode (2k and 8k prompts)"
run_py benchmark_agent.py --url "${URL}" --model "${MODEL}" \
    --input-lens 2048 --output-len 256 --num-requests 3 --concurrency 1 \
    | tee "${OUT}/solo-2k.txt"
run_py benchmark_agent.py --url "${URL}" --model "${MODEL}" \
    --input-lens 8192 --output-len 256 --num-requests 3 --concurrency 1 \
    | tee "${OUT}/solo-8k.txt"

for c in 2 4 8; do
    echo "==> [${PROFILE}] ${c} parallel requests (8k prompts)"
    run_py benchmark_agent.py --url "${URL}" --model "${MODEL}" \
        --input-lens 8192 --output-len 256 --num-requests "${c}" --concurrency "${c}" \
        | tee "${OUT}/par${c}-8k.txt"
done

echo "==> [${PROFILE}] tool calling + vision"
run_py test_tool_call.py --url "${URL}" --model "${MODEL}" 2>&1 | tee "${OUT}/tool_call.txt" || true
run_py test_vision.py --url "${URL}" --model "${MODEL}" 2>&1 | tee "${OUT}/vision.txt" || true

echo "==> [${PROFILE}] server-side decode throughput (from container log):"
docker logs "${NAME}" 2>&1 | grep "gen throughput" | tail -400 \
    | sed -E 's/.*#running-req: ([0-9]+).*gen throughput \(token\/s\): ([0-9.]+).*/\1 \2/' \
    | awk '{c[$1]+=$2; n[$1]++} END {for (k in c) printf "  %s req: ~%d tok/s (%d samples)\n", k, c[k]/n[k], n[k]}' \
    | sort -n | tee "${OUT}/server_decode_tput.txt"
docker logs "${NAME}" 2>&1 | grep -oE "accept len: [0-9.]+" | tail -50 | tee "${OUT}/spec_accept.txt"

echo "==> Done. Results in ${OUT}/"
