# vLLM + DFlash2 — Qwen3.8-27B-NVFP4 on dual RTX 5090

Serve `Inferact/Qwen3.8-27B-NVFP4` with the `incoai/Qwen3.8-27B-DFlash2` drafter
on two RTX 5090, in Docker. DFlash2 landed on 2026-08-18 and is not in any
released vLLM, so the image is the pinned nightly wheel plus the two open PRs
(#52816, #52883), applied as a patch — no source rebuild.

**This is the recipe in production** (port 8000, 262k context, since
2026-08-19). Solo decode 131-139 tok/s, 1.8x the same checkpoint without the
drafter, TPOT 5-6 ms and it stays at 9 ms under a 16-request burst.

## Requirements

- 2× NVIDIA RTX 5090 (32 GB, sm_120)
- Docker + nvidia-container-toolkit
- ~25 GiB target + ~4 GiB drafter, ~32 GB for the image

## Quick start

```bash
cp .env.example .env
./scripts/build.sh               # nightly + DFlash2 PRs, ~30 s
./scripts/download_model.sh      # optional: pre-stage target + drafter
./scripts/start.sh               # default profile: dflash2-prod, :8000
./scripts/bench.sh dflash2-prod  # speed suite -> results/dflash2-prod/
```

OpenAI-compatible API on `http://<host>:8000/v1`, model id `Qwen3.8-27B-NVFP4`.
`./scripts/stop.sh` to stop. The profile passed to `start.sh` wins over `.env`.

## Profiles

| Profile | Shape | For |
|---|---|---|
| `dflash2-prod` (default) | util 0.92, `max-num-seqs 4`, :8000 | production: latency first |
| `dflash2-tp2` | util 0.88, `max-num-seqs 16`, :8100 | aggregate throughput, the reference bench shape |
| `nospec-tp2` | no drafter, util 0.94, :8100 | the baseline the tables below compare against |

## Running it as a service

The container carries `--restart unless-stopped` (boot and crash) and a
`/health` healthcheck. Docker never acts on an unhealthy container, so two cron
entries cover the rest:

```cron
# Nightly restart at 04:00 — stop + start: health wait, then warmup.
0 4 * * * { date; cd <recipe> && ./scripts/stop.sh && ./scripts/start.sh; } >> ~/vllm-prod.log 2>&1
# Restart a wedged engine that keeps the port open (unhealthy after ~90 s).
*/5 * * * * [ "$(docker inspect -f '{{.State.Health.Status}}' vllm-dflash2 2>/dev/null)" = unhealthy ] && { date; cd <recipe> && ./scripts/stop.sh && ./scripts/start.sh; } >> ~/vllm-prod.log 2>&1
```

Startup is ~4.5 min, during which the port refuses connections. Rollback is the
no-drafter 512k recipe next door (`../qwen3-27b-nvfp4-vllm-dual-5090`, container
`vllm-qwen`): stop this one first, both want the two GPUs.

## Measured performance

`tools/benchmark_agent.py`, TP=2, 262k context, fp8 KV, 256 output tokens, same
checkpoint and same image in every column. Concurrent requests share the prompt,
so they hit the prefix cache.

| Scenario | no drafter | DFlash2, prod shape | DFlash2, throughput shape |
|---|---|---|---|
| 1 req, 2k prompt | 82 tok/s (TPOT 11 ms) | **139** (6 ms) | 144 (5 ms) |
| 1 req, 8k prompt | 73 tok/s (11 ms) | **133** (5 ms) | 133 (5 ms) |
| 2 par, 8k prompt | 131 (14 ms) | **198** (7 ms) | 196 (7 ms) |
| 4 par, 8k prompt | 229 (14 ms) | **265** (8 ms) | 282 (7 ms) |
| 8 par, 8k prompt | 356 (16 ms) | 266 (9 ms) | 301 (15 ms) |
| 16 par, 8k prompt | 508 (21 ms) | 264 (9 ms) | 301 (27 ms) |
| KV pool (tokens) | 958,181 | **550,749** (2.10x) | 489,358 (1.87x) |
| 32k cold prefill | TTFT 6.5 s | ~7 s | 5.5 s |

The prod shape trades 12% of aggregate throughput above 4 concurrent for a TPOT
that does not move under load: 9 ms at 16 concurrent against 27 ms with
`max-num-seqs 16`. Requests beyond 4 queue instead of degrading. Both drafted
shapes lose to plain decoding once the batch is large — spec decode buys latency
here, not peak throughput.

Acceptance length: **3.2-4.1 tokens per verification**, ~3.5 typical at 7 draft
tokens, flat in concurrency. The DFlash2 card reports 4.80 mean on
`Qwen/Qwen3.8-27B`. The official FP8 target was measured (in the SGLang recipe,
same drafter) at 3.0-3.6, so target quantization is *not* the explanation for
the gap.

**SGLang runs this drafter faster.** Same checkpoint, same drafter, in
`../qwen3-27b-nvfp4-slang-dflash2-dual-5090`: 187 tok/s solo at 8k and 624 at 8
concurrent. vLLM's DFlash2 path is days old; SGLang's DFlash lineage is a year
older. This recipe keeps prod because it holds 262k with a 2.10x KV pool and
degrades gracefully past 4 concurrent — SGLang's shape is pinned at 8 running
requests with a 250k pool.

## Accuracy

Shared suite (`../../tools/run_eval.sh`), card sampling (temperature 1.0,
top_p 0.95, top_k 20), drafter enabled:

| Check | Score | No answer extracted |
|---|---|---|
| perplexity canary (5.7k tokens) | ppl 20.38 | — |
| gsm8k_cot_lite (200) | 0.9550 | 5 (2.5%) |

Run: `../../tools/run_eval.sh http://localhost:8000 Qwen3.8-27B-NVFP4`

The no-drafter 512k recipe scores ppl 19.69 and 0.9700 with 1 empty reply, on a
different checkpoint (unsloth). 191/200 against 194/200 is noise at n=200; the
five empty replies are the length tail, not corruption — replaying the first of
them three times returns the right answer in 318, 440 and 1439 tokens with
`finish_reason: stop`. Empty means the 8192-token budget went into the thinking
trace, the failure mode `tools/README.md` already documents.

Tool calling and the vision tower were verified on this build
(`tools/test_tool_call.py`, `tools/test_vision.py`).

## Configuration notes

| Setting | Value | Why |
|---|---|---|
| `HF_REPO_ID` | `Inferact/Qwen3.8-27B-NVFP4` | NVFP4 body with an **unquantized `lm_head`** |
| `GPU_MEMORY_UTILIZATION` | 0.92 | holds at `max-num-seqs 4`; 0.94 was not retried at this shape |
| `MAX_NUM_SEQS` | 4 | verify batch 4×8 instead of 16×8: flat TPOT, +13% KV pool |
| `NUM_SPECULATIVE_TOKENS` | 7 | the drafter's block size is 8 |
| `KV_CACHE_DTYPE` | `fp8_e4m3` | bf16 KV measured: same acceptance (3.47 vs 3.52), slower decode, and 262k no longer fits |
| `MAX_MODEL_LEN` | 262144 | native, no YaRN overlay in this recipe |

- **The checkpoint choice is forced by the drafter.** DFlash2's candidate
  selector runs top-K through the *target's* LM head and refuses a quantized
  one. Both checkpoints used by the sibling recipes quantize `lm_head`
  (unsloth: FP8, RadixArk: NVFP4), so neither can serve DFlash2 — the engine
  fails at startup with `DFlash2 requires an unquantized target LM head`.
- **Utilization is shape-dependent.** At `max-num-seqs 16`, 0.94 dies on the
  first GDN prefill — `expandable_segments: memory mapping failed with OOM ...
  (free: 3801088)` inside `chunk_gated_delta_rule`, the warmup requests answer
  500, the engine exits. 0.88 holds there. At `max-num-seqs 4` the verify batch
  is 4x smaller and 0.92 holds through the warmup, a 32k prefill and a
  16-request burst; 0.94 at that shape was not tried.
- **262k, not 512k.** The sibling recipe reaches 512k with a YaRN overlay. Here
  the pool is 550,749 tokens: 512k would mean 1.05x concurrency, one full
  request and no margin. Prod runs the native window.
- **Why the image is a patched nightly.** vLLM 0.27.1 has DFlash1 but no
  DFlash2, and DFlash2 also needs the V2 model runner (`use_v2_model_runner`
  forces it, as for DSpark). The Dockerfile drops the PR diff onto the pinned
  nightly wheel: the diff is pure Python, `patch` fails the build if either side
  drifts. Once #52816 and #52883 land in a release, replace the whole Dockerfile
  with the released image.
- **A draft-length schedule does not fix the concurrent case.**
  `num_speculative_tokens_per_batch_size` with `[[1,4,7],[5,8,3],[9,256,1]]` is
  honoured (acceptance falls to 1.9 at 16 concurrent, as expected for k=1) but
  buys 2%: 318 tok/s at 8 concurrent and 308 at 16, against 301/301 for a fixed
  7 and 356/508 with no drafter. The drafter's per-step cost, not the draft
  length, is what the large batch cannot absorb.
- **Losslessness was not verified, and cannot be by comparing text here.** This
  engine is not batch-invariant: without any drafter, 8 identical greedy
  requests sent together return 3 different completions, and a 9th sent alone
  returns a 4th. Spec-decode output diverging from no-spec output therefore says
  nothing about the accept rule.
- **Give every experiment its own HF cache.** Pointing a test container at
  another recipe's cache by repo id re-resolves `refs/main` against the Hub and
  can leave a snapshot without weights behind — that is how the sibling recipe's
  YaRN overlay was broken into a 37-restart crash loop on 2026-08-19.
- The drafter gets no multimodal embeddings (vLLM warns at startup); vision
  requests fall back to text-only drafting. The vision tower itself works.

## Files

```
.
├── Dockerfile                    # pinned vLLM nightly + DFlash2 PRs
├── .env.example                  # defaults (copy to .env); profiles override it
├── cache/huggingface/            # target + drafter, mounted (gitignored)
├── profiles/
│   ├── dflash2-prod.env          # default: production shape, :8000
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
