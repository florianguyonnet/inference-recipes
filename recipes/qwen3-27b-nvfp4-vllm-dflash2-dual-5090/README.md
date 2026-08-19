# vLLM + DFlash2 — Qwen3.8-27B-NVFP4 on dual RTX 5090

Serve `Inferact/Qwen3.8-27B-NVFP4` with the `incoai/Qwen3.8-27B-DFlash2` drafter
on two RTX 5090, in Docker. DFlash2 landed on 2026-08-18 and is not in any
released vLLM, so the image is the pinned nightly wheel plus the two open PRs
(#52816, #52883), applied as a patch — no source rebuild.

Solo decode: 133-144 tok/s, **1.8x** the same checkpoint without the drafter.
The gain shrinks with load and turns negative past 4 concurrent requests: at 16
concurrent the drafter costs 40% of throughput. Speed only — no accuracy suite
was run.

## Requirements

- 2× NVIDIA RTX 5090 (32 GB, sm_120)
- Docker + nvidia-container-toolkit
- ~25 GiB target + ~4 GiB drafter, ~32 GB for the image

## Quick start

```bash
cp .env.example .env
./scripts/build.sh               # nightly + DFlash2 PRs, ~30 s
./scripts/download_model.sh      # optional: pre-stage target + drafter
./scripts/start.sh               # default profile: dflash2-tp2, :8100
./scripts/bench.sh dflash2       # speed suite -> results/dflash2/
```

OpenAI-compatible API on `http://<host>:8100/v1`, model id `Qwen3.8-27B-NVFP4`.
`./scripts/stop.sh` to stop. Port 8100 keeps it clear of the plain vLLM recipe
on 8000, but both need the two GPUs: run one at a time.

## Profiles

| Profile | Drafter | Notes |
|---|---|---|
| `dflash2-tp2` (default) | DFlash2, 7 draft tokens | `--gpu-memory-utilization 0.88` |
| `nospec-tp2` | off | the baseline the table below compares against |

## Measured performance

`tools/benchmark_agent.py`, TP=2, 262k context, fp8 KV, 256 output tokens, same
checkpoint and same image in both columns. Concurrent requests share the prompt,
so they hit the prefix cache.

| Scenario | no drafter | DFlash2 | |
|---|---|---|---|
| 1 req, 2k prompt | 82 tok/s (TPOT 11 ms) | **144 tok/s** (5 ms) | 1.76x |
| 1 req, 8k prompt | 73 tok/s (11 ms) | **133 tok/s** (5 ms) | 1.82x |
| 2 par, 8k prompt | 131 tok/s (14 ms) | **196 tok/s** (7 ms) | 1.49x |
| 4 par, 8k prompt | 229 tok/s (14 ms) | **282 tok/s** (7 ms) | 1.23x |
| 8 par, 8k prompt | **356 tok/s** (16 ms) | 301 tok/s (15 ms) | 0.85x |
| 16 par, 8k prompt | **508 tok/s** (21 ms) | 301 tok/s (27 ms) | 0.59x |
| 32k cold prefill | TTFT 6.5 s | TTFT 5.5 s | prefill is not drafted |

Acceptance length: **3.2-4.1 tokens per verification**, ~3.5 typical at 7 draft
tokens, flat in concurrency (3.5 solo, 3.3 at 8, 3.4 at 16). The DFlash2 card
reports 4.80 mean on `Qwen/Qwen3.8-27B`; the drafter was trained against the
bf16 target and runs here against an NVFP4 one, which is the likeliest reason
for the gap — untestable on 64 GB of VRAM, where the bf16 target does not fit.

KV pool: 489,358 tokens with the drafter (util 0.88), 958,181 without (0.94).
Startup ~4.5 min, of which ~70 s is CUDA-graph capture for target + drafter.

**SGLang runs this drafter faster.** Same checkpoint, same drafter, in
`../qwen3-27b-nvfp4-slang-dflash2-dual-5090`: 187 tok/s solo at 8k against 133
here, and 624 at 8 concurrent against 301 — where this recipe is already behind
its own no-drafter baseline. vLLM's DFlash2 path is days old; SGLang's DFlash
lineage is a year older. This recipe is worth running for the 262k context with
a full KV pool and the 16-concurrent shape, not for peak drafted throughput.

## Configuration notes

| Setting | Value | Why |
|---|---|---|
| `HF_REPO_ID` | `Inferact/Qwen3.8-27B-NVFP4` | NVFP4 body with an **unquantized `lm_head`** |
| `GPU_MEMORY_UTILIZATION` | 0.88 | 0.94 OOMs on the first GDN prefill |
| `NUM_SPECULATIVE_TOKENS` | 7 | the drafter's block size is 8 |
| `KV_CACHE_DTYPE` | `fp8_e4m3` | bf16 KV measured: same acceptance (3.47 vs 3.52), slower decode, and 262k no longer fits |
| `MAX_MODEL_LEN` | 262144 | native, no YaRN overlay in this recipe |

- **The checkpoint choice is forced by the drafter.** DFlash2's candidate
  selector runs top-K through the *target's* LM head and refuses a quantized
  one. Both checkpoints used by the sibling recipes quantize `lm_head`
  (unsloth: FP8, RadixArk: NVFP4), so neither can serve DFlash2 — the engine
  fails at startup with `DFlash2 requires an unquantized target LM head`.
  `Inferact/Qwen3.8-27B-NVFP4` keeps `lm_head` in bf16.
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
- The drafter gets no multimodal embeddings (vLLM warns at startup); vision
  requests fall back to text-only drafting.

## Files

```
.
├── Dockerfile                    # pinned vLLM nightly + DFlash2 PRs
├── .env.example                  # production defaults (copy to .env)
├── cache/huggingface/            # target + drafter, mounted (gitignored)
├── profiles/
│   ├── dflash2-tp2.env           # default
│   └── nospec-tp2.env            # baseline, no drafter
├── scripts/
│   ├── build.sh                  # docker build
│   ├── start.sh                  # docker run + health wait (profile as $1)
│   ├── stop.sh
│   ├── download_model.sh         # optional weight pre-staging via the container
│   └── bench.sh                  # speed suite + acceptance length per scenario
├── results/                      # raw bench outputs (gitignored)
└── README.md
```
