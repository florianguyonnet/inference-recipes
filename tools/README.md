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
| `test_tool_call.py` | OpenAI `tool_calls` schema validation |
| `test_vision.py` | Vision-tower smoke test (synthetic digit image) |

```bash
.venv/bin/python benchmark_agent.py --url http://localhost:8000 --model Qwen3.8-27B-NVFP4 \
    --input-lens 8192 --output-len 256 --num-requests 8 --concurrency 8
```

## Accuracy suite

`run_eval.sh` runs the same protocol at the end of every recipe, so scores are comparable across engines/configs:

| Task | Metric | Size |
|---|---|---|
| `gsm8k_cot` | exact_match (flexible-extract) | full 1319 |
| `mmlu_flan_cot_zeroshot` | acc (flexible-extract) | 1700 |
| `aime25` | exact_match | full 30 |

```bash
./run_eval.sh [url] [model] [gsm8k_limit] [mmlu_limit]
```

Conventions:

- Sampling: `temperature=0.6, top_p=0.95, top_k=20` — the Qwen reasoning settings. Greedy on long generations is a known repetition-loop failure mode for reasoning models.
- `max_tokens` is long on purpose (8192; 32768 for AIME): Qwen3.8 reasons a lot and truncated traces score as wrong.
- **We report flexible-extract only.** strict-match is meaningless here by design: our servers run `--reasoning-parser qwen3`, so the thinking trace (where the `####` answer marker lives) is split off into `reasoning_content` and never reaches the scorer.
- A 8-concurrency full suite takes ~1 h against a dual-RTX-5090 server.

Reference on the SGLang + DSpark recipe (dual RTX 5090): gsm8k_cot flexible-extract **0.92** (50-example sanity subset).
