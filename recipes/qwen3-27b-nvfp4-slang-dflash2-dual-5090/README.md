# SGLang + DFlash2 — Qwen3.8-27B-NVFP4 on dual RTX 5090

Serve `Inferact/Qwen3.8-27B-NVFP4` with the `incoai/Qwen3.8-27B-DFlash2` drafter
on two RTX 5090, in Docker. DFlash2 merged into SGLang main on 2026-08-19, hours
after the last published nightly, so the image is that nightly with the merge
commit's python tree overlaid.

Solo decode: 178-187 tok/s, **2.0x** the same checkpoint without a drafter and
**+57%** over DSpark on the same image and checkpoint. 624 tok/s at 8 concurrent.
Speed only — no accuracy suite was run.

This is the fastest of the four recipes here at every concurrency measured.

## Requirements

- 2× NVIDIA RTX 5090 (32 GB, sm_120)
- Docker + nvidia-container-toolkit
- ~25 GiB target + ~4 GiB drafter, ~60 GB for the image

## Quick start

```bash
cp .env.example .env
./scripts/build.sh               # nightly + the DFlash2 merge commit, ~1 min
./scripts/download_model.sh      # optional: pre-stage target + drafter
./scripts/start.sh               # default profile: dflash2-tp2, :30100
./scripts/bench.sh dflash2       # speed suite -> results/dflash2/
```

OpenAI-compatible API on `http://<host>:30100/v1`, model id `Qwen3.8-27B-NVFP4`.
`./scripts/stop.sh` to stop. The port is clear of the other recipes, but all of
them want the two GPUs: run one at a time.

## Profiles

| Profile | Drafter | Notes |
|---|---|---|
| `dflash2-tp2` (default) | DFlash2, 8-token verify | `--mem-fraction-static 0.85` |
| `dspark-tp2` | DSpark, block 7 | same image and checkpoint, the comparison point |
| `nospec-tp2` | off | autoregressive baseline |

## Measured performance

`tools/benchmark_agent.py`, TP=2, 256 output tokens, fp8 KV, same image and same
checkpoint in all three columns. Concurrent requests share the prompt and the
suite runs twice, so these are warm-radix-cache numbers: decode throughput, not
prefill.

| Scenario | no drafter | DSpark | DFlash2 | DFlash2 vs none |
|---|---|---|---|---|
| 1 req, 2k prompt | 91 tok/s (TPOT 10 ms) | 113 (8 ms) | **178** (5 ms) | 1.95x |
| 1 req, 8k prompt | 92 tok/s (10 ms) | 119 (8 ms) | **187** (5 ms) | 2.03x |
| 2 par, 8k prompt | 172 (11 ms) | 182 (11 ms) | **279** (7 ms) | 1.62x |
| 4 par, 8k prompt | 328 (12 ms) | 341 (11 ms) | **529** (7 ms) | 1.61x |
| 8 par, 8k prompt | 572 (14 ms) | 341 (14 ms) | **624** (11 ms) | 1.09x |
| Acceptance length | — | 2.1-2.3 | **3.5-3.8** | |
| KV pool (tokens) | 655,721 | 284,593 | 250,460 | |

Cold 32k prefill, first request after a restart: 6.8-7.4 s TTFT (~4,600 tok/s) in
all three configs — prefill is not drafted, and the drafter costs nothing there.

`--max-running-requests 8` is the GDN state-pool pin inherited from the DSpark
recipe, so 16 concurrent is out of reach in this shape; that is where the vLLM
recipe's numbers still climb.

### Against the other recipes

Same drafter and same target checkpoint on vLLM (`../qwen3-27b-nvfp4-vllm-dflash2-dual-5090`):
133 tok/s solo at 8k and 301 at 8 concurrent, against 187 and 624 here. SGLang's
DFlash implementation is a year older than vLLM's and it shows: same weights,
same drafter, 1.4x solo and 2.1x at 8 concurrent.

DFlash2 also beats DSpark at every point measured, and unlike DSpark it stays
ahead of no-drafter serving at 8 concurrent (DSpark falls to 341 there, well
under the 572 of plain decoding).

## Configuration notes

| Setting | Value | Why |
|---|---|---|
| `HF_REPO_ID` | `Inferact/Qwen3.8-27B-NVFP4` | NVFP4 body with an **unquantized `lm_head`** |
| `MEM_FRACTION` | 0.85 | 0.90 leaves 0.66 GB: draft CUDA graph disabled, warmup OOMs |
| `NUM_DRAFT_TOKENS` | 8 | the drafter's block size; SGLang counts the verify width, vLLM counts 7 proposals |
| `MAMBA_CACHE_SIZE` / `MAX_RUNNING_REQUESTS` | 96 / 8 | GDN state pool pin, see the DSpark recipe |
| `KV_CACHE_DTYPE` | `fp8_e4m3` | no scaling factors in this checkpoint, so SGLang defaults them to 1.0 |

- **The checkpoint choice is forced by the drafter.** DFlash2's candidate
  selector runs top-K through the *target's* LM head and rejects a quantized
  one — SGLang has a unit test for that rejection. The `RadixArk` checkpoint the
  DSpark recipe serves quantizes `lm_head` to NVFP4, so it cannot serve DFlash2.
- **`--mem-fraction-static 0.90` starts but does not work.** The engine reports
  `Disable DFLASH draft cuda graph because only 0.66 GB GPU memory is available`,
  then the warmup request OOMs on a 96 MiB allocation and the server never
  becomes ready. 0.85 captures the draft graph (5 s) and leaves 1.78 GB.
- **Effective context is the KV pool, not `--context-length`.** With the drafter
  at 0.85 the pool is 250,460 tokens against a declared 262,144, so a single
  full-length request does not fit. Lower `CONTEXT_LENGTH` to 250k if you need
  the declared limit to be honest, or drop the drafter (655k pool).
- **The official FP8 target is not the better choice, and not for the reason we
  expected.** `Qwen/Qwen3.8-27B-FP8` also keeps `lm_head` unquantized, so it
  serves DFlash2 as well. Measured, same image, same drafter: acceptance
  3.0-3.6 — *no higher* than NVFP4's 3.5-3.8, so the gap against the 4.80 the
  model card reports is not the target's quantization. Throughput splits by
  regime: 142 tok/s solo at 8k (−24% vs NVFP4, more weight bytes to read) but
  721 at 8 concurrent (+16%), with the KV pool down to 136,845 tokens. Worth it
  only if 8-concurrent throughput is the whole workload.
- **Why the image is an overlaid nightly.** The dev image installs sglang
  editable from `/sgl-workspace/sglang`, so the Dockerfile drops the merge
  commit's python tree over it; the image's `sgl-kernel` build stays. Files
  deleted upstream since the nightly's commit are left behind — nothing in the
  newer tree imports them. Once a nightly ships c14312a6 or later, delete the
  Dockerfile and use it directly.

## Files

```
.
├── Dockerfile                    # SGLang nightly + the DFlash2 merge commit
├── .env.example                  # all knobs documented
├── cache/huggingface/            # target + drafter, mounted (gitignored)
├── profiles/
│   ├── dflash2-tp2.env           # default
│   ├── dspark-tp2.env            # same image/checkpoint, other drafter
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
