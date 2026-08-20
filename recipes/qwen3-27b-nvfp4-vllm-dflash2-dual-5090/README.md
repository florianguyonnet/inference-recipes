# vLLM + DFlash2 — Qwen3.8-27B-NVFP4 on dual RTX 5090, 512k context

Serve `unsloth/Qwen3.8-27B-NVFP4` at 512k (YaRN 2.0 on the native 262k
checkpoint) with the `incoai/Qwen3.8-27B-DFlash2` drafter, on two RTX 5090 in
Docker. DFlash2 shipped 2026-08-18 and is in no released vLLM, so the image is a
pinned nightly wheel plus the open PR #52816, with the target-LM-head guard
relaxed.

Solo decode 132 tok/s against 96 without the drafter, TPOT 5-6 ms, and it holds
under a burst because requests beyond 4 queue. Tool calling, vision and 268k
retrieval verified.

## Requirements

- 2× NVIDIA RTX 5090 (32 GB, sm_120)
- Docker + nvidia-container-toolkit
- ~22 GiB target + ~4 GiB drafter, ~32 GB for the image

## Quick start

```bash
cp .env.example .env
./scripts/build.sh               # nightly + PR #52816 + the guard patch, ~30 s
./scripts/download_model.sh      # optional: pre-stage target + drafter
./scripts/start.sh               # default profile: dflash2-512k, :8000
./scripts/bench.sh dflash2-512k  # speed suite -> results/dflash2-512k/
```

OpenAI-compatible API on `http://<host>:8000/v1`, model id `Qwen3.8-27B-NVFP4`.
`./scripts/stop.sh` to stop. Both overlays are (re)built on every start, so a
fresh clone reproduces the config from the two upstream repo ids.

`start.sh` refuses to start while another process holds the GPUs (`FORCE=1`
overrides) and defaults to no restart policy; the serving profile turns the
policy on. With several recipes on one box, that combination is what keeps a
failed experiment from looping against whatever is serving.

## Profiles

| Profile | Shape |
|---|---|
| `dflash2-512k` (default) | 512k YaRN, util 0.92, `max-num-seqs 4`, :8000, restarts |
| `dflash2-latency` | 262k native, util 0.92, `max-num-seqs 4`, :8000 |
| `dflash2-tp2` | 262k native, util 0.88, `max-num-seqs 16`, :8100 |
| `nospec-tp2` | no drafter, util 0.94, :8100 — the baseline below |

## Running it as a service

The serving profile sets `--restart unless-stopped` and the container carries a
`/health` healthcheck. Docker never acts on an unhealthy container, so two cron
entries cover the rest:

```cron
# Nightly restart at 04:00 — rebuilds both overlays, waits for /health, warms up.
0 4 * * * { date; cd <recipe> && ./scripts/stop.sh && ./scripts/start.sh; } >> ~/vllm.log 2>&1
# Restart a wedged engine that keeps the port open (unhealthy after ~90 s).
*/5 * * * * [ "$(docker inspect -f '{{.State.Health.Status}}' vllm-dflash2 2>/dev/null)" = unhealthy ] && { date; cd <recipe> && ./scripts/stop.sh && ./scripts/start.sh; } >> ~/vllm.log 2>&1
```

Startup is ~5 min, during which the port refuses connections.

## Measured performance

`tools/benchmark_agent.py`, TP=2, fp8 KV, 256 output tokens, same checkpoint and
image in every column. Concurrent requests share the prompt, so they hit the
prefix cache.

| Scenario | no drafter | DFlash2, 262k | DFlash2, 512k (default) |
|---|---|---|---|
| 1 req, 2k prompt | 96 tok/s | **155** | 132 |
| 1 req, 8k prompt | 87 tok/s | 110 | 110 |
| 2 par, 8k prompt | 146 | 201 | 176 |
| 4 par, 8k prompt | 246 | 267 | 271 |
| 8 par, 8k prompt | 368 | 265 (queued) | 268 (queued) |
| 16 par, 8k prompt | 520 | 266 (queued) | 264 (queued) |
| KV pool | 1,009,405 @262k | 612,845 (2.34x) | 648,991 (1.24x @512k) |

Acceptance length: 3.2-3.8 on bench prompts, 4.3-4.7 on chat and long-context
requests. `max-num-seqs 4` trades aggregate throughput above 4 concurrent for a
TPOT that does not move under load; the drafter loses to plain decoding once the
batch is large, so spec decode here buys latency, not peak throughput.

Long context, 267,946-token prompt (fact planted at 93% depth, retrieved):
109 s cold, ~2,450 tok/s prefill, acceptance 4.29.

## Accuracy

Shared suite (`../../tools/run_eval.sh`), card sampling, drafter enabled:

| Check | This recipe | Same checkpoint, no drafter |
|---|---|---|
| perplexity canary (5.7k tokens) | ppl 19.59 | 19.50 (262k), 19.69 (512k YaRN) |
| gsm8k_cot_lite (200) | 0.9600 | 0.9450 |

Run: `../../tools/run_eval.sh http://localhost:8000 Qwen3.8-27B-NVFP4`

The drafter costs nothing measurable: with it 0.9400, without it 0.9450 on the
same 262k shape, and the deterministic ppl canary is flat. gsm8k moved between
0.940 and 0.970 across runs of identical configs, so treat differences under
~3 points at n=200 as sampling noise and read the canary instead.

## Configuration notes

| Setting | Value | Why |
|---|---|---|
| `HF_REPO_ID` | `unsloth/Qwen3.8-27B-NVFP4` | quantized `lm_head`, served anyway — see below |
| `GPU_MEMORY_UTILIZATION` | 0.92 | 0.94 serves fine but dies on a `prompt_logprobs` request |
| `MAX_NUM_SEQS` | 4 | verify batch 4×8: flat TPOT, more KV pool |
| `NUM_SPECULATIVE_TOKENS` | 7 | the drafter's block size is 8 |
| `MAX_MODEL_LEN` | 524288 | YaRN 2.0 on the 262k checkpoint |
| `DRAFTER_OVERLAY` | 1 | mandatory past 262k — see below |

- **The quantized-LM-head guard is patched out.** DFlash2 refuses a quantized
  target `lm_head`, which rules out unsloth (FP8 head) and RadixArk (NVFP4).
  The candidate Top-K dispatches through `quant_method.apply()` either way, and
  a compressed-tensors FP8 head runs it: reported in vllm-project/vllm#52883 and
  reproduced here. The Dockerfile widens the isinstance check and fails the build
  loudly if that line moves upstream. Acceptance on this checkpoint reaches 4.7,
  the highest of the three NVFP4 checkpoints tried.
- **The drafter needs a bigger position table past 262k.** It ships
  `max_position_embeddings=262144`; a 357k prompt indexes its cos/sin cache out of
  bounds and the run dies on `CUDA error: device-side assert triggered` while
  building attention metadata. `make_drafter_overlay.sh` raises the table to
  524288 and touches nothing else — the drafter attends in a 2048-token sliding
  window, so relative geometry, and acceptance, are unchanged (4.29 measured at
  268k).
- **Utilization is bounded by `prompt_logprobs`, not by serving.** 0.94 passes
  warmup, a 268k prefill and a 16-request burst, and then the perplexity canary
  (5.7k tokens with prompt logprobs, ~0.6 MB of fp32 logits per token) OOMs the
  engine. 0.92 survives the whole toolchain for 30k tokens less pool.
- **The W4A16 drafter does not load.** `syvai/Qwen3.8-27B-DFlash2-W4A16` is well
  built — 1.19 GiB, and its `ignore` list keeps the candidate selector and the
  kernel projections in bf16 — but `qwen3_dflash.py` reads `qkv_proj.weight`
  unconditionally to precompute its fused KV buffers, and a pack-quantized layer
  has no `.weight`. Fails at load with `AttributeError`.
- **LMCache and any KV-connector offload are incompatible with this model.**
  Both integrations turn the hybrid KV cache manager off, and the engine then
  refuses to start: `ValueError: Failed to promote local KV cache specs to one
  unified type.` The model's GDN recurrent state cannot be unified with
  attention KV. Not a tuning problem — there is no flag that avoids it today.
- **A draft-length schedule does not help.**
  `num_speculative_tokens_per_batch_size` `[[1,4,7],[5,8,3],[9,256,1]]` is
  honoured (acceptance falls to 1.9 at 16 concurrent) and buys 2%.
- **Losslessness cannot be checked by comparing text here.** Without any
  drafter, 8 identical greedy requests sent together return 3 different
  completions: this engine is not batch-invariant to begin with.
- Give every experiment its own HF cache. Anything that resolves a repo id
  against the Hub in a shared cache moves `refs/main` to a revision whose weights
  are absent, and an overlay built from it has no `*.safetensors`; both overlay
  scripts refuse to build in that state.

## Files

```
.
├── Dockerfile                    # pinned vLLM nightly + PR #52816 + guard patch
├── .env.example                  # defaults (copy to .env); profiles override it
├── cache/huggingface/            # checkpoints + both overlays, mounted (gitignored)
├── models/
│   └── Qwen3.8-27B-NVFP4-yarn512k/config.json   # patched YaRN config (tracked)
├── profiles/
│   ├── dflash2-512k.env          # default: 512k, restarts, :8000
│   ├── dflash2-latency.env       # 262k native
│   ├── dflash2-tp2.env           # throughput shape, :8100
│   └── nospec-tp2.env            # baseline, no drafter
├── scripts/
│   ├── build.sh
│   ├── start.sh                  # overlays + GPU guard + docker run + warmup
│   ├── stop.sh
│   ├── download_model.sh
│   ├── make_yarn_overlay.sh      # target: 262k -> 512k YaRN
│   ├── make_drafter_overlay.sh   # drafter: position table -> 524288
│   └── bench.sh                  # speed suite + acceptance per scenario
├── results/                      # raw bench outputs (gitignored)
└── README.md
```
