# SGLang + DSpark — Qwen3.8-27B-NVFP4 on dual RTX 5090

Serve `RadixArk/Qwen3.8-27B-NVFP4` (NVFP4 MLP + FP8 attention, FP8 KV) with DSpark speculative decoding on two RTX 5090, in Docker with the model-specific SGLang build. Vision tower and tool calling work.

Solo decode: 120-190 tok/s (content-dependent, peaks at 275 on structured output). 434 tok/s at 7 concurrent requests. Native 262k context.

## Requirements

- 2× NVIDIA RTX 5090 (32 GB, sm_120)
- Docker + nvidia-container-toolkit
- ~60 GB for the image, ~19 GB for the checkpoints

## Quick start

```bash
cp .env.example .env
./scripts/download_model.sh      # optional: pre-stage the checkpoints
./scripts/start.sh               # default profile: tp2-dspark
./scripts/bench.sh tp2-dspark    # load suite -> results/tp2-dspark/
```

OpenAI-compatible API on `http://<host>:30000/v1`, model id `Qwen3.8-27B-NVFP4`. `./scripts/stop.sh` to stop.

## Profiles

| Profile | GPUs | Spec decode |
|---|---|---|
| `tp2-dspark` (default) | 2 | DSpark drafter, block 7 |
| `tp1-dspark` | 1 | DSpark drafter, block 7 |
| `tp2-greedy` | 2 | off |

Pass as first argument: `./scripts/start.sh profiles/tp1-dspark.env`.

## Measured performance

TP=2, DSpark, fp8 KV, bf16 GDN states, 96-slot state pool (8 concurrent):

| Scenario | Throughput |
|---|---|
| Solo decode | 120-190 tok/s (peaks 275) |
| 7 parallel requests | 434 tok/s |
| 27k-token cold prefill | 5.4 s (~5k tok/s) |
| 2k prompt TTFT | 0.11-0.18 s |

Acceptance length: 2.1-3.3 tokens/verify on generic prompts; the drafter card reports 2.7-4.6 at temperature 0.6 with thinking enabled. Solo decode tracks content predictability: ~120 tok/s on free prose, 200-275 on code and structured output.

## Accuracy

Measured with the shared lm-evaluation-harness suite (`../../tools/run_eval.sh`), flexible-extract:

| Task | Score |
|---|---|
| gsm8k_cot (1319) | 0.9393 |
| mmlu_flan_cot_zeroshot (1531) | 0.6166 |
| aime25 (30) | 0.6667 |

## Configuration notes

- **GDN state pool sizing is the flag that matters.** Post-weight VRAM splits into a GDN state pool (concurrency ceiling) and the KV pool, divided by `--mamba-full-memory-ratio`. The default (0.9) over-provisions KV and clamps `max_running_requests` near zero on 32 GB cards. The profiles pin the pool: `--max-mamba-cache-size 96 --max-running-requests 8` (8 requests × S=4 slots + D=8 verify tokens). Do not combine a large pin with DSpark on one 32 GB card: speculative intermediates scale with slots × 8 and OOM the warmup.
- `--linear-attn-verify-backend triton` is required by the DSpark recipe.
- `GEN_CONFIG_TEMP=0.6`: the checkpoint's generation_config defaults to 1.0; the DSpark evaluation used 0.6. `start.sh` builds a generation_config overlay so clients that don't set a temperature get 0.6.
- `--mm-feature-transport cpu` is the default (works everywhere). `MM_TRANSPORT=cuda_ipc` enables zero-copy vision; it works here because the container runs `--privileged`.
- KV pool at TP=2 with the pinned state pool: ~467k tokens fp8 per rank. Context is the native 262,144.

## Files

```
.
├── .env.example                  # all knobs documented
├── cache/huggingface/            # checkpoints, mounted into the container (gitignored)
├── profiles/                     # tp1/tp2 × greedy/dspark
├── scripts/
│   ├── start.sh                  # docker run + health wait (profile as $1)
│   ├── stop.sh
│   ├── download_model.sh         # optional weight pre-staging via the container
│   ├── bench.sh                  # load suite (runs tools/ inside the container)
│   └── sweep.sh                  # restart-with-variant-flags + probe harness
├── results/                      # raw bench outputs (gitignored)
└── logs/                         # (gitignored)
```
