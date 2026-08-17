#!/usr/bin/env bash
set -uo pipefail

# DSpark tuning sweep: restart the server with variant flags and probe
# solo decode throughput on fixed prompts (temp 0.6, the DSpark eval setting).
#
# Usage: ./scripts/sweep.sh            # runs the default sweep matrix
# Results: results/sweep/<config>.txt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT="${PROJECT_DIR}/results/sweep"
mkdir -p "${OUT}"

# name|profile|extra flags
CONFIGS=(
    "block7-baseline|profiles/tp2-dspark.env|"
    "block5|profiles/tp2-dspark.env|--speculative-dspark-block-size 5"
    "block6|profiles/tp2-dspark.env|--speculative-dspark-block-size 6"
    "attn-decode|profiles/tp2-dspark.env|--speculative-attention-mode decode"
    "tp1-block7|profiles/tp1-dspark.env|"
)

echo "config | tput_med | tput_max" | tee "${OUT}/summary.txt"
for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r name profile flags <<< "${entry}"
    echo ""
    echo "=== ${name} (profile ${profile}, extra: ${flags:-none}) ==="
    "${SCRIPT_DIR}/stop.sh" >/dev/null 2>&1 || true
    sleep 3
    if ! EXTRA_FLAGS="${flags}" "${SCRIPT_DIR}/start.sh" "${profile}" >/dev/null 2>&1; then
        echo "${name} | STARTUP FAILED" | tee -a "${OUT}/summary.txt"
        continue
    fi
    python3 "${PROJECT_DIR}/../../tools/probe_decode.py" | tee "${OUT}/${name}.txt"
    line=$(grep -oE "tput_med=[0-9.]+ tput_max=[0-9.]+" "${OUT}/${name}.txt" || echo "PROBE-FAIL")
    echo "${name} | ${line}" | sed 's/tput_med=//;s/tput_max=//' | tee -a "${OUT}/summary.txt"
done

"${SCRIPT_DIR}/stop.sh" >/dev/null 2>&1 || true
echo ""
echo "==> Sweep done:"
cat "${OUT}/summary.txt"
