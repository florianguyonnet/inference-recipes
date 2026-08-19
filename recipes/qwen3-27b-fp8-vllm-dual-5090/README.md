# vLLM — Qwen3.8-27B-FP8 on dual RTX 5090, 512k context

Serve the official `Qwen/Qwen3.8-27B-FP8` at 512k context (YaRN 2.0 on the native
262k checkpoint) on two RTX 5090, with the `vllm/vllm-openai:v0.27.1` image. Same
shape as the NVFP4 recipe next door, so the two are directly comparable: this one
exists to answer whether the official FP8 checkpoint is worth its extra 3.6 GB
per card, and whether MTP speculative decoding — rejected on NVFP4 — pays off
here.

Short answer: FP8 costs 28-30% of solo decode against NVFP4 and 22% of the KV
pool. MTP with 3 draft tokens buys most of the decode back (95 tok/s solo,
within 1% of NVFP4), but not the pool, and only up to 4 concurrent requests.

## Requirements

- 2× NVIDIA RTX 5090 (32 GB, sm_120)
- Docker + nvidia-container-toolkit
- ~29 GiB for the checkpoint, ~15 GB for the image

## Quick start

```bash
./scripts/download_model.sh      # optional: pre-stage the checkpoint
./scripts/start.sh               # default: fp8-full (no drafter), :8001
./scripts/start.sh profiles/fp8-mtp3.env   # MTP, 3 draft tokens
./scripts/bench.sh fp8           # speed suite -> results/fp8/
```

OpenAI-compatible API on `http://<host>:8001/v1`, model id `Qwen3.8-27B-FP8`.
`./scripts/stop.sh` to stop. `start.sh` refuses to run while another process
holds the GPUs (`FORCE=1` overrides) — with five recipes on one box, the failure
it prevents is a confusing `Free memory on device cuda:0 ... is less than
desired GPU memory utilization`.

## Profiles

| Profile | Shape |
|---|---|
| `fp8-full` (default) | no drafter, util 0.94, `max-num-seqs 16` |
| `fp8-mtp3` | MTP k=3, util 0.94, `max-num-seqs 4` |

## Measured performance

`tools/benchmark_agent.py`, TP=2, 512k (YaRN), fp8 KV, 256 output tokens.
NVFP4 column is the sibling recipe (`../qwen3-27b-nvfp4-vllm-dual-5090`), same
box, same tool, no drafter.

| Scenario | NVFP4 (unsloth) | FP8, no drafter | FP8 + MTP k=3, seqs 4 |
|---|---|---|---|
| 1 req, 2k prompt | 96 tok/s | 68.8 (TPOT 14 ms) | **95.1** (9 ms) |
| 1 req, 8k prompt | 87 tok/s | 61.1 (14 ms) | 74.6 (9 ms) |
| 2 par, 8k prompt | 146 | 114 | 128.7 |
| 4 par, 8k prompt | 246 | 201 | 194.0 |
| 8 par, 8k prompt | 368 | 318 | 193.8 (queued) |
| 16 par, 8k prompt | 520 | 444 | 198.2 (queued) |
| Weights | 10.8 GiB/card | 14.41 GiB/card | 14.41 GiB/card |
| KV pool | 1,053,188 (2.04x) | 821,025 (1.57x) | 755,097 (1.44x) |
| 32k cold prefill | TTFT 5.3 s | 7.2 s | — |
| 128k cold prefill | — | 27.5 s (~4,650 tok/s) | — |

Prefill is ~26% slower than NVFP4 and the drafter does not touch it: MTP only
changes decode.

### MTP sweep

The number of draft tokens, at `max-num-seqs 16` unless stated. "died" means the
engine hit `CUDA error: an illegal memory access was encountered` in the
attention backend and the container exited.

| k | util | seqs | solo 2k | solo 8k | 2 par | 4 par | 8 par | 16 par | acceptance |
|---|---|---|---|---|---|---|---|---|---|
| off | 0.94 | 16 | 68.8 | 61.1 | 114 | 201 | 318 | 444 | — |
| 2 | 0.94 | 16 | 85.9 | 73.6 | — | 191.7 | 234.7 | 270.3 | 2.36-2.51 |
| **3** | **0.94** | **16** | **93.8** | **79.9** | — | **184.3** | **237.0** | **264.0** | **2.68-2.91** |
| 4 | 0.94 | 16 | — | 80.8 | 152.6 | 202.0 | 251.1 | 31.4 → died | 2.85-3.28 |
| 5 | 0.94 | 16 | 93.0 | 80.5 | — | 7.4 → died | died | died | 2.93-3.21 |
| 5 | 0.88 | 16 | 94.0 | 77.8 | 134.1 | 195.5 | 233.7 | 55.7 | 2.91-3.23 |
| **3** | **0.94** | **4** | **95.1** | **74.6** | **128.7** | **194.0** | **193.8** | **198.2** | **2.59-2.82** |
| 4 | 0.95 | 4 | 86.7 | 82.1 | 126.9 | 199.6 | 136.4 | died | 2.60-3.05 |
| 5 | 0.95 | 4 | 95.7 | 85.7 | 143.2 | 2.1 → died | died | died | 2.99-3.34 |
| 5 | 0.90 | 4 | 97.9 | 82.6 | 136.0 | 11.0 → died | died | died | 3.11-3.26 |

- **k=3 is the only setting that completed every scenario, twice.** k=4 and k=5
  post the best solo numbers and the best acceptance (up to 3.34) and then die
  under load. Lowering utilization moves the crash but does not remove it: k=5
  survived one configuration out of four. Treat it as a build bug, not a knob.
- **MTP works here, unlike on NVFP4.** The sibling recipe measured MTP at 55
  tok/s solo against 96 without, and rejected it: verifying speculative tokens
  on this hybrid GDN model rewinds recurrent state per engine step. On the FP8
  checkpoint the same mechanism nets +36% solo (68.8 → 93.8). The difference is
  worth knowing before dismissing MTP on a GDN model from someone else's
  measurement.
- **Acceptance does not explain the gain.** 2.7-2.9 accepted tokens per verify
  here against the ~2.4 the NVFP4 recipe measured — nearly the same. What
  changed is the cost side: FP8 decode is memory-bound enough that one extra
  verified token per step pays for the rewind.
- **`max-num-seqs 4` is a stability setting, not a preference.** At 16 the verify
  batch reaches the shape that collapses (k=4: 31 tok/s at 16 concurrent, then
  death). Capped at 4, throughput is flat past the cap — 194 at 4, 198 at 16 —
  because requests queue.

## Accuracy

Shared suite (`../../tools/run_eval.sh`), card sampling, on `fp8-mtp3`:

| Check | FP8 + MTP k=3 | NVFP4 (sibling recipe) |
|---|---|---|
| perplexity canary (5.7k tokens) | **ppl 19.26** | ppl 19.69 |
| gsm8k_cot_lite (200) | 0.9450 (2 no answer) | 0.9700 (1) |

Run: `../../tools/run_eval.sh http://localhost:8001 Qwen3.8-27B-FP8`

Perplexity is better on FP8, gsm8k is five items worse — within noise at n=200,
so this pair does not settle the quality question either way. The structural
difference is that this checkpoint leaves `lm_head` unquantized while the unsloth
NVFP4 one quantizes it to FP8; whether that matters for a given workload is not
something a 200-question smoke can show.

## Configuration notes

| Setting | Value | Why |
|---|---|---|
| `MAX_MODEL_LEN` | 524288 | 512k via YaRN 2.0 on the native 262k checkpoint |
| `GPU_MEMORY_UTILIZATION` | 0.94 | 0.95 was measured and it is where MTP k=4/k=5 die |
| `MAX_NUM_SEQS` | 16 without a drafter, 4 with MTP | see the sweep |
| `--kv-cache-dtype fp8_e4m3` | — | 512k does not fit otherwise |
| `--attention-backend flashinfer` | — | flash_attn rejects FP8 KV on this hybrid model |
| quantization | auto-detect | `quant_method: fp8`, `activation_scheme: dynamic` |

- **512k via YaRN**: `models/Qwen3.8-27B-FP8-yarn512k/config.json` sets
  `text_config.rope_parameters` to YaRN (`factor 2.0`,
  `original_max_position_embeddings 262144`), mrope and theta preserved.
  `start.sh` builds an overlay of relative symlinks plus that config inside the
  mounted cache, so the snapshot is untouched. YaRN is extrapolation: expect
  drift past 262k, the trained range.
- **The MTP weights ship with the checkpoint** (`mtp.safetensors`, 22 tensors,
  FP8 like the body), so no second download and no drafter repo.
- **This recipe has no restart policy.** It is a comparison recipe: a failed
  start should stay failed, not loop against whichever server owns the cards.

## Files

```
.
├── .env.example                  # optional: scripts run on their defaults
├── cache/huggingface/            # checkpoint + YaRN overlay, mounted (gitignored)
├── models/
│   └── Qwen3.8-27B-FP8-yarn512k/config.json   # patched YaRN config (tracked)
├── profiles/
│   ├── fp8-full.env              # default: no drafter
│   └── fp8-mtp3.env              # MTP k=3, max-num-seqs 4
├── scripts/
│   ├── start.sh                  # overlay build + GPU-busy guard + docker run
│   ├── stop.sh
│   ├── download_model.sh
│   ├── make_yarn_overlay.sh
│   └── bench.sh                  # speed suite
├── results/                      # raw bench outputs (gitignored)
└── README.md
```
