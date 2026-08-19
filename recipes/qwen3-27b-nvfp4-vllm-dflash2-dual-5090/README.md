# vLLM + DFlash2 — Qwen3.8-27B-NVFP4 on dual RTX 5090

Serve `Inferact/Qwen3.8-27B-NVFP4` with the `incoai/Qwen3.8-27B-DFlash2` drafter
on two RTX 5090, in Docker. DFlash2 shipped on 2026-08-18 and is in no released
vLLM, so the image is a pinned nightly wheel plus two open PRs (#52816, #52883)
applied as a patch — no source rebuild.

Solo decode 133-139 tok/s, 1.8x the same checkpoint without the drafter, TPOT
5-6 ms and still 9 ms under a 16-request burst. Native 262k context. Tool calling
and the vision tower work.

## Requirements

- 2× NVIDIA RTX 5090 (32 GB, sm_120)
- Docker + nvidia-container-toolkit
- ~25 GiB target + ~4 GiB drafter, ~32 GB for the image

## Quick start

```bash
cp .env.example .env
./scripts/build.sh               # nightly + DFlash2 PRs, ~30 s
./scripts/download_model.sh      # optional: pre-stage target + drafter
./scripts/start.sh               # default profile: dflash2-latency, :8000
./scripts/bench.sh dflash2-latency  # speed suite -> results/dflash2-latency/
```

OpenAI-compatible API on `http://<host>:8000/v1`, model id `Qwen3.8-27B-NVFP4`.
`./scripts/stop.sh` to stop. The profile passed to `start.sh` wins over `.env`.

## Profiles

| Profile | Shape | For |
|---|---|---|
| `dflash2-latency` (default) | util 0.92, `max-num-seqs 4`, :8000 | serving: latency first |
| `dflash2-tp2` | util 0.88, `max-num-seqs 16`, :8100 | aggregate throughput |
| `nospec-tp2` | no drafter, util 0.94, :8100 | the baseline the tables below compare against |

## Running it as a service

The container carries `--restart unless-stopped` (boot and crash) and a
`/health` healthcheck. Docker never acts on an unhealthy container, so two cron
entries cover the rest:

```cron
# Nightly restart at 04:00 — stop + start: health wait, then warmup.
0 4 * * * { date; cd <recipe> && ./scripts/stop.sh && ./scripts/start.sh; } >> ~/vllm.log 2>&1
# Restart a wedged engine that keeps the port open (unhealthy after ~90 s).
*/5 * * * * [ "$(docker inspect -f '{{.State.Health.Status}}' vllm-dflash2 2>/dev/null)" = unhealthy ] && { date; cd <recipe> && ./scripts/stop.sh && ./scripts/start.sh; } >> ~/vllm.log 2>&1
```

Startup is ~4.5 min, during which the port refuses connections.

## Measured performance

`tools/benchmark_agent.py`, TP=2, 262k context, fp8 KV, 256 output tokens, same
checkpoint and same image in every column. Concurrent requests share the prompt,
so they hit the prefix cache.

| Scenario | no drafter | DFlash2, `dflash2-latency` | DFlash2, `dflash2-tp2` |
|---|---|---|---|
| 1 req, 2k prompt | 82 tok/s (TPOT 11 ms) | **139** (6 ms) | 144 (5 ms) |
| 1 req, 8k prompt | 73 tok/s (11 ms) | **133** (5 ms) | 133 (5 ms) |
| 2 par, 8k prompt | 131 (14 ms) | **198** (7 ms) | 196 (7 ms) |
| 4 par, 8k prompt | 229 (14 ms) | **265** (8 ms) | 282 (7 ms) |
| 8 par, 8k prompt | 356 (16 ms) | 266 (9 ms) | 301 (15 ms) |
| 16 par, 8k prompt | 508 (21 ms) | 264 (9 ms) | 301 (27 ms) |
| KV pool (tokens) | 958,181 | 550,749 (2.10x) | 489,358 (1.87x) |
| 32k cold prefill | TTFT 6.5 s | ~7 s | 5.5 s |

`max-num-seqs 4` trades 12% of aggregate throughput above 4 concurrent for a
TPOT that does not move under load: 9 ms at 16 concurrent against 27 ms at 16
seqs. Requests beyond 4 queue instead of degrading. Both drafted shapes lose to
plain decoding once the batch is large — spec decode buys latency here, not peak
throughput.

Acceptance length: **3.2-4.1 tokens per verification**, ~3.5 typical at 7 draft
tokens, flat in concurrency. The DFlash2 card reports 4.80 mean on
`Qwen/Qwen3.8-27B`; the official FP8 target measures 3.0-3.6 (see the SGLang
recipe), so target quantization does not explain the gap.

`../qwen3-27b-nvfp4-slang-dflash2-dual-5090` runs the same checkpoint and the
same drafter at 187 tok/s solo and 624 at 8 concurrent — vLLM's DFlash2 path is
days old, SGLang's DFlash lineage is a year older. This recipe holds 262k with a
2.10x KV pool and keeps scaling to 16 concurrent; the SGLang one is pinned at 8
running requests with a 250k pool.

## Accuracy

Shared suite (`../../tools/run_eval.sh`), card sampling (temperature 1.0,
top_p 0.95, top_k 20), drafter enabled:

| Check | Score | No answer extracted |
|---|---|---|
| perplexity canary (5.7k tokens) | ppl 20.38 | — |
| gsm8k_cot_lite (200) | 0.9550 | 5 (2.5%) |

Run: `../../tools/run_eval.sh http://localhost:8000 Qwen3.8-27B-NVFP4`

The five empty replies are the thinking-trace length tail, not corruption:
replaying the first of them three times returns the right answer in 318, 440 and
1439 tokens with `finish_reason: stop`. Empty means the 8192-token budget went
into the trace, the failure mode `tools/README.md` documents.

## Configuration notes

| Setting | Value | Why |
|---|---|---|
| `HF_REPO_ID` | `Inferact/Qwen3.8-27B-NVFP4` | NVFP4 body with an **unquantized `lm_head`** |
| `GPU_MEMORY_UTILIZATION` | 0.92 | holds at `max-num-seqs 4`; 0.94 untried at this shape |
| `MAX_NUM_SEQS` | 4 | verify batch 4×8 instead of 16×8: flat TPOT, +13% KV pool |
| `NUM_SPECULATIVE_TOKENS` | 7 | the drafter's block size is 8 |
| `KV_CACHE_DTYPE` | `fp8_e4m3` | bf16 KV measured: same acceptance (3.47 vs 3.52), slower decode, and 262k stops fitting |
| `MAX_MODEL_LEN` | 262144 | native, no YaRN overlay here |

- **The checkpoint choice is forced by the drafter.** DFlash2's candidate
  selector runs top-K through the *target's* LM head and refuses a quantized
  one. Both checkpoints used by the sibling recipes quantize `lm_head`
  (unsloth: FP8, RadixArk: NVFP4), so neither can serve DFlash2 — startup fails
  with `DFlash2 requires an unquantized target LM head`.
- **Utilization is shape-dependent.** At `max-num-seqs 16`, 0.94 dies on the
  first GDN prefill: `expandable_segments: memory mapping failed with OOM ...
  (free: 3801088)` inside `chunk_gated_delta_rule`, warmup requests answer 500,
  the engine exits. 0.88 holds there. At `max-num-seqs 4` the verify batch is 4x
  smaller and 0.92 holds through warmup, a 32k prefill and a 16-request burst.
- **262k, not 512k.** The sibling recipe reaches 512k with a YaRN overlay. At a
  550,749-token pool, 512k would mean 1.05x concurrency: one full request, no
  margin.
- **Why the image is a patched nightly.** vLLM 0.27.1 has DFlash1 but not
  DFlash2, and DFlash2 also needs the V2 model runner (`use_v2_model_runner`
  forces it, as for DSpark). The Dockerfile drops the PR diff onto the pinned
  nightly wheel: pure Python, and `patch` fails the build if either side drifts.
  Replace the Dockerfile with the released image once #52816 and #52883 land.
- **A draft-length schedule does not fix the concurrent case.**
  `num_speculative_tokens_per_batch_size` with `[[1,4,7],[5,8,3],[9,256,1]]` is
  honoured (acceptance falls to 1.9 at 16 concurrent, as expected for k=1) but
  buys 2%: 318 tok/s at 8 concurrent and 308 at 16, against 301/301 for a fixed
  7. The drafter's per-step cost, not the draft length, is what a large batch
  cannot absorb.
- **Losslessness was not verified, and text comparison cannot verify it here.**
  This engine is not batch-invariant: without any drafter, 8 identical greedy
  requests sent together return 3 different completions, and a 9th sent alone
  returns a 4th.
- The drafter gets no multimodal embeddings (vLLM warns at startup); vision
  requests fall back to text-only drafting. The vision tower itself works.

## Files

```
.
├── Dockerfile                    # pinned vLLM nightly + DFlash2 PRs
├── .env.example                  # defaults (copy to .env); profiles override it
├── cache/huggingface/            # target + drafter, mounted (gitignored)
├── profiles/
│   ├── dflash2-latency.env       # default: latency shape, :8000
│   ├── dflash2-tp2.env           # throughput shape, :8100
│   └── nospec-tp2.env            # baseline, no drafter
├── scripts/
│   ├── build.sh                  # docker build
│   ├── start.sh                  # docker run + health wait + warmup (profile as $1)
│   ├── stop.sh
│   ├── download_model.sh         # optional weight pre-staging via the container
│   └── bench.sh                  # speed suite + acceptance length per scenario
├── results/                      # raw bench outputs (gitignored)
└── README.md
```
