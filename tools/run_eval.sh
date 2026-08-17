#!/usr/bin/env bash
set -euo pipefail

# Accuracy check for a recipe. Not a leaderboard run: it verifies a serving
# config did not damage the model, and separates "wrong answer" from "no answer
# inside the token budget".
#
#   smoke (default)  probe_ppl.py + gsm8k_cot_lite (200)          ~6 min
#   full             + mmlu_pro_lite (251) + aime25_boxed (30)    ~1 h
#
# Task configs are in tools/tasks/ (the upstream ones stop generation on
# "Q:"/"Question:", which a reasoning model trips over inside its own trace).
#
# Usage:
#   ./run_eval.sh [url] [model] [smoke|full]
# Defaults: http://localhost:8000 Qwen3.8-27B-NVFP4 smoke
#
# Follow a running suite: tail -f tools/results/suite-*/<task>.log

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${TOOLS_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: ${TOOLS_DIR}/.venv missing. Run ./setup.sh first."
    exit 1
fi

URL="${1:-http://localhost:8000}"
MODEL="${2:-Qwen3.8-27B-NVFP4}"
TIER="${3:-smoke}"

# timeout: a 64k-token AIME trace at ~45 tok/s per stream takes ~25 min, well
# past lm-eval's 300 s default.
MODEL_ARGS="model=${MODEL},base_url=${URL}/v1/chat/completions,tokenized_requests=False,num_concurrent=8,max_retries=3,timeout=3600"
# Thinking mode as the checkpoint card specifies it, same for every recipe so the
# scores stay comparable. Greedy loops on long generations.
SAMPLING="temperature=1.0,top_p=0.95,top_k=20,min_p=0"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="${TOOLS_DIR}/results/suite-${STAMP}"
mkdir -p "${OUT}"

run_task() {
    local task="$1" max_tokens="$2" limit="${3:-}"
    echo "==> ${task} (max_tokens ${max_tokens}${limit:+, limit ${limit}})"
    echo "    log: ${OUT}/${task}.log"
    "${PYTHON}" -m lm_eval \
        --model local-chat-completions \
        --model_args "${MODEL_ARGS}" \
        --tasks "${task}" \
        --include_path "${TOOLS_DIR}/tasks" \
        ${limit:+--limit "${limit}"} \
        --gen_kwargs "max_tokens=${max_tokens},${SAMPLING}" \
        --apply_chat_template \
        --log_samples \
        --output_path "${OUT}" \
        > "${OUT}/${task}.log" 2>&1
    report "${task}"
}

# Score, plus the number that says whether the score means anything: replies with
# no extractable answer are truncated or looping traces, not wrong answers.
report() {
    "${PYTHON}" - "${OUT}" "$1" <<'PY'
import glob, json, sys
out_dir, task = sys.argv[1], sys.argv[2]
n = correct = invalid = empty = 0
for f in glob.glob(f"{out_dir}/**/samples_{task}_*.jsonl", recursive=True):
    for line in open(f):
        d = json.loads(line)
        n += 1
        correct += bool(d.get("exact_match"))
        empty += not d["resps"][0][0].strip()
        invalid += "[invalid]" in d["filtered_resps"]
if not n:
    print("    no samples found")
else:
    print(f"    score {correct}/{n} = {correct / n:.4f}   "
          f"no answer extracted: {invalid} ({invalid / n:.1%})   empty reply: {empty}")
PY
}

echo "==> ${TIER} against ${URL} (${MODEL})"
echo "==> perplexity canary (fixed text, deterministic)"
"${PYTHON}" "${TOOLS_DIR}/probe_ppl.py" --url "${URL}" --model "${MODEL}" \
    2>/dev/null | tee "${OUT}/ppl.log" | tail -1

run_task gsm8k_cot_lite 8192 200
if [[ "${TIER}" == "full" ]]; then
    MMLU_PRO_STRIDE=48 run_task mmlu_pro_lite 16384
    run_task aime25_boxed 65536
fi

echo ""
echo "==> Done. Logs + samples in ${OUT}/"
