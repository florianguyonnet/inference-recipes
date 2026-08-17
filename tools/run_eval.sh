#!/usr/bin/env bash
set -euo pipefail

# Standard evaluation suite for every recipe, via lm-evaluation-harness.
# Same protocol across recipes so scores are comparable:
#   - gsm8k_cot              (math word problems, CoT)
#   - mmlu_flan_cot_zeroshot (knowledge/reasoning, 57 subjects)
#   - aime25                 (competition math, 30 problems)
# Sampling: temperature=0.6, top_p=0.95, top_k=20 (Qwen reasoning settings;
# greedy on long generations is a known repetition-loop failure mode).
#
# We report flexible-extract only: strict-match is meaningless with
# --reasoning-parser qwen3 (the "####" marker lives in the stripped trace).
#
# Usage:
#   ./run_eval.sh [url] [model] [gsm8k_limit] [mmlu_limit]
# Defaults: http://localhost:30000 Qwen3.8-27B-NVFP4 1319(full) 1700
#
# Follow a running suite: tail -f tools/results/suite-*/<task>.log

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${TOOLS_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: ${TOOLS_DIR}/.venv missing. Run ./setup.sh first."
    exit 1
fi

URL="${1:-http://localhost:30000}"
MODEL="${2:-Qwen3.8-27B-NVFP4}"
GSM8K_LIMIT="${3:-1319}"
MMLU_LIMIT="${4:-1700}"

# timeout=900: a 32k-token AIME trace at ~130 tok/s solo exceeds lm-eval's
# 300s default and dies client-side.
MODEL_ARGS="model=${MODEL},base_url=${URL}/v1/chat/completions,tokenized_requests=False,num_concurrent=8,max_retries=3,timeout=900"
SAMPLING="temperature=0.6,top_p=0.95,top_k=20"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="${TOOLS_DIR}/results/suite-${STAMP}"
mkdir -p "${OUT}"

run_task() {
    local task="$1" limit="$2" max_tokens="$3"
    echo "==> ${task} (limit ${limit}, max_tokens ${max_tokens})"
    echo "    log: ${OUT}/${task}.log"
    "${PYTHON}" -m lm_eval \
        --model local-chat-completions \
        --model_args "${MODEL_ARGS}" \
        --tasks "${task}" \
        --limit "${limit}" \
        --gen_kwargs "max_tokens=${max_tokens},${SAMPLING}" \
        --apply_chat_template \
        --log_samples \
        --output_path "${OUT}" \
        > "${OUT}/${task}.log" 2>&1
}

# Micro-average the flexible-extract score straight from the samples files:
# lm-eval does not compute group aggregates when --limit is used.
aggregate() {
    "${PYTHON}" - "$1" <<'PY'
import json, glob, sys
n = correct = 0
for f in glob.glob(f"{sys.argv[1]}/**/samples_*.jsonl", recursive=True):
    task = f.split("samples_")[1].rsplit("_2", 1)[0]
    for line in open(f):
        d = json.loads(line)
        if d.get("filter") != "flexible-extract":
            continue
        n += 1
        correct += bool(d.get("exact_match"))
print(f"    flexible-extract micro-avg: {correct}/{n} = {correct/n:.4f}" if n else "    no samples found")
PY
}

run_task gsm8k_cot "${GSM8K_LIMIT}" 8192
aggregate "${OUT}"
run_task mmlu_flan_cot_zeroshot "${MMLU_LIMIT}" 8192
aggregate "${OUT}"
# AIME gets a long budget: competition-math thinking traces run 5-15k tokens.
run_task aime25 30 32768
aggregate "${OUT}"

echo ""
echo "==> Suite done. Logs + samples in ${OUT}/"
