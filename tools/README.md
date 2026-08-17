# Shared benchmark & evaluation tools

Engine-agnostic clients that work against any OpenAI-compatible server (vLLM, SGLang, ...). Each recipe's bench harness calls into these.

## Setup

```bash
./setup.sh        # uv venv + lm-evaluation-harness, aiohttp
```

## Load / latency tools

| Tool | What it measures |
|---|---|
| `benchmark_agent.py` | TTFT / TPOT / throughput with `--concurrency` support |
| `probe_decode.py` | Solo decode tok/s on fixed prompts (client-side, spec-decode safe) |
| `probe_ppl.py` | Perplexity on fixed text — deterministic calibration canary |
| `test_tool_call.py` | OpenAI `tool_calls` schema validation |
| `test_vision.py` | Vision-tower smoke test (synthetic digit image) |

```bash
.venv/bin/python benchmark_agent.py --url http://localhost:8000 --model Qwen3.8-27B-NVFP4 \
    --input-lens 8192 --output-len 256 --num-requests 8 --concurrency 8
```

## Accuracy suite

```bash
./run_eval.sh [url] [model] [smoke|full]      # default: localhost:8000, smoke
```

| Tier | Runs | Size | Time |
|---|---|---|---|
| `smoke` | `probe_ppl.py`, `gsm8k_cot_lite` | 5.7k tokens, 200 q | ~6 min |
| `full` | + `mmlu_pro_lite`, `aime25_boxed` | 251 q, 30 problems | ~1 h |

Every task reports the score *and* the number of replies with no extractable
answer. The second number is the one that moves first when a serving config
damages the model: truncated or looping traces, not wrong answers.

Task configs live in `tasks/`: no stop strings, a `max_tokens` budget sized for a
thinking trace (8k gsm8k, 16k MMLU-Pro, 64k AIME), `\boxed{}` / `the answer is (X)`
extraction, non-matches marked `[invalid]`.

The upstream `gsm8k_cot`, `mmlu_flan_cot_zeroshot` and `aime25` configs are
written for completion-style models: they stop generation on `"Q:"` /
`"Question:"` and cap the budget at 2k tokens. With `--reasoning-parser` the
trace goes to the `reasoning` field, so a reply cut mid-thought reaches the
scorer as an empty string and counts as wrong. Measured on Qwen3.8-27B-NVFP4:
`mmlu_flan_cot_zeroshot` 0.6166 with 27% of replies stopped after 6-17 tokens,
`aime25` 0.60 with 8/30 traces over the 32k budget.

- Sampling: `temperature=1.0, top_p=0.95, top_k=20, min_p=0` (checkpoint card, thinking mode), same for every recipe. What a recipe pins as its own serving default is a separate decision.
- `MMLU_PRO_STRIDE` sets the MMLU-Pro subset (48 → 251 questions). The test split is grouped by category, so `--limit` alone only samples `business`.
- `probe_ppl.py` needs GPU headroom: `prompt_logprobs` allocates ~0.6 MB of fp32 logits per prompt token, and at `--gpu-memory-utilization 0.94` a 2.5k-token echo request OOMs the engine.
