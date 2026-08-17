# vLLM — Qwen3.8-27B-NVFP4 on dual RTX 5090, 512k context

Serve `unsloth/Qwen3.8-27B-NVFP4` at 512k context (YaRN 2.0x on the native 262k checkpoint) on two RTX 5090, in Docker with the official `vllm/vllm-openai:v0.27.1` image. Tool calling works; MTP speculative decoding was evaluated and rejected (see configuration notes).

Solo decode: ~90 tok/s. 368 tok/s at 8 concurrent, 520 at 16. 512k context validated by a needle test (fact hidden at token ~269,800 of a 310k prompt retrieved correctly).

## Requirements

- 2× NVIDIA RTX 5090 (32 GB, sm_120)
- Docker + nvidia-container-toolkit
- ~23 GiB for the checkpoint, ~15 GB for the image

## Quick start

```bash
cp .env.example .env
./scripts/download_model.sh      # optional: pre-stage the checkpoint
./scripts/start.sh               # builds the YaRN overlay, starts on :8000
```

OpenAI-compatible API on `http://<host>:8000/v1`, model id `Qwen3.8-27B-NVFP4`. `./scripts/stop.sh` to stop.

## Profiles

| Profile | GPUs | Context |
|---|---|---|
| `qwen-full` (default) | 2 | 512k (YaRN) |

## Measured performance

Streaming benchmark (`tools/benchmark_agent.py` at the repo root), TP=2, 256 output tokens:

| Scenario | TTFT median | TPOT | Throughput |
|---|---|---|---|
| 1 req, 2k prompt | 0.08 s | 10 ms | ~96 tok/s |
| 1 req, 8k prompt | 0.27 s | 10 ms | ~87 tok/s |
| 2 par, 8k prompt | 0.40 s | 12 ms | ~146 tok/s |
| 4 par, 8k prompt | 0.98 s | 12 ms | ~246 tok/s |
| 8 par, 8k prompt | 1.9 s | 14 ms | ~368 tok/s |
| 16 par, 8k prompt | 2.8 s | 20 ms | ~520 tok/s |

Concurrent runs share the prompt, so they hit the prefix cache: cold, the same 8-request
batch is 299 tok/s (TTFT 3.2 s).

Cold prefill (unique prompts, TP=2): 32k tokens in 5.3 s (~6,200 tok/s); 310k in 124 s (~2,500 tok/s). A 500k prompt takes ~3 min before the first token.

## Accuracy

Shared suite (`tools/run_eval.sh`), card sampling (temperature 1.0, top_p 0.95, top_k 20):

| Check | Score | No answer extracted |
|---|---|---|
| perplexity canary (5.7k tokens) | ppl 19.69 | — |
| gsm8k_cot_lite (200) | 0.9700 | 1 (0.5%) |

Run: `../../tools/run_eval.sh http://localhost:8000 Qwen3.8-27B-NVFP4`

## Configuration notes

| Setting | Value | Why |
|---|---|---|
| `TENSOR_PARALLEL_SIZE` | 2 | 512k context does not fit on one 32 GB card |
| `MAX_MODEL_LEN` | 524288 | 512k via YaRN on the native 262k checkpoint |
| `MAX_NUM_SEQS` | 16 | 372 → 520 tok/s at 16 concurrent, −0.4% KV. 32 gives 635 tok/s at 32 concurrent for −1.2% KV |
| `MAX_NUM_BATCHED_TOKENS` | 8192 | 16384 was measured: same prefill (5.30 vs 5.32 s on 32k), −5% KV |
| `GPU_MEMORY_UTILIZATION` | 0.94 | vLLM 0.27 profiles CUDA-graph memory honestly; 0.94 ≈ 30.1 GB/card |
| `--kv-cache-dtype fp8_e4m3` | — | matches checkpoint calibration, 2x KV capacity |
| `--attention-backend flashinfer` | — | flash_attn rejects FP8 KV cache on this hybrid model |
| quantization | auto-detect | compressed-tensors checkpoint (NVFP4 W4A4 MLP + FP8 attention) |

Measured KV cache at startup (TP=2, util 0.94, seqs 16): **1,053,188 tokens → 2.04x concurrency at 524,288 tokens**. Worst-case concurrency is 2x at full 512k.

- **512k via YaRN**: `models/Qwen3.8-27B-NVFP4-yarn512k/config.json` sets `text_config.rope_parameters` to YaRN (`factor 2.0`, `original_max_position_embeddings 262144`, mrope/theta preserved). vLLM 0.27 dropped the `--rope-scaling` flag, so the patched config is the supported mechanism. `start.sh` builds the overlay: relative symlinks to the HF snapshot + this config, inside the mounted cache. YaRN is extrapolation: expect quality drift beyond 262k, the native trained range.
- **Startup time**: ~2.5 min warm, ~6 min cold. The container runs with `--restart unless-stopped`.
- **MTP rejected** (vLLM 0.27.1): good draft acceptance (~2.4 tokens/step), but verifying speculative tokens on this hybrid GDN model rewinds recurrent state at ~4x per engine step. Net: 55 tok/s solo vs 96 without. Revisit if vLLM optimizes GDN state handling for spec decode.
- **`--async-scheduling` rejected**: measured neutral (±0.3% on solo decode, 8/16/32 concurrent and 32k prefill).
- **`--enable-prefix-caching` is not redundant**: vLLM defaults it to *off* for this hybrid model, and enabling it switches the Mamba cache to `align` mode, which vLLM flags as experimental. `--enable-chunked-prefill` and `--dtype bfloat16` were dropped: both are already the defaults here.
- **`--kv-cache-memory`** (knob, unset by default): vLLM's profiler reports only ~2% more KV available at util 0.94, and taking it leaves no headroom — `prompt_logprobs`/`echo` requests then OOM the engine.
- **Startup warmup**: `start.sh` sends 4 batched requests once the server is healthy. vLLM otherwise JIT-compiles a Triton kernel during the first real request.
- **Adding another model**: `./scripts/download_model.sh <repo_id>`, then a profile with a different `HF_REPO_ID`/`VLLM_PORT`.

## Files

```
.
├── .env.example                  # production defaults (copy to .env)
├── cache/huggingface/            # checkpoints + YaRN overlay, mounted (gitignored)
├── models/
│   └── Qwen3.8-27B-NVFP4-yarn512k/config.json  # patched YaRN config (tracked)
├── profiles/
│   └── qwen-full.env             # explicit baseline profile (= .env defaults)
├── scripts/
│   ├── start.sh                  # overlay build + docker run + health wait (profile as $1)
│   ├── stop.sh
│   ├── download_model.sh         # optional weight pre-staging via the container
│   └── make_yarn_overlay.sh      # rebuild the 512k overlay inside the cache
└── README.md
```
