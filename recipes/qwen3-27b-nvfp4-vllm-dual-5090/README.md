# vLLM — Qwen3.8-27B-NVFP4 on dual RTX 5090, 512k context

Serve `unsloth/Qwen3.8-27B-NVFP4` at 512k context (YaRN 2.0x on the native 262k checkpoint) on two RTX 5090, in Docker with the official `vllm/vllm-openai:v0.27.1` image. Tool calling works; MTP speculative decoding was evaluated and rejected (see configuration notes).

Solo decode: ~96 tok/s. 368 tok/s at 8 concurrent requests. 512k context validated by a needle test (fact hidden at token ~269,800 of a 310k prompt retrieved correctly).

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

Cold prefill (unique prompts, TP=2): 27.4k tokens in 6.1 s (~4,500 tok/s); 310k in 124 s (~2,500 tok/s). A 500k prompt takes ~3 min before the first token.

## Accuracy

Measured with the shared lm-evaluation-harness suite (`tools/run_eval.sh`), flexible-extract:

| Task | Score |
|---|---|
| gsm8k_cot (1319) | pending |
| mmlu_flan_cot_zeroshot (1531) | pending |
| aime25 (30) | pending |

Run: `../../tools/run_eval.sh http://localhost:8000 Qwen3.8-27B-NVFP4`

## Configuration notes

| Setting | Value | Why |
|---|---|---|
| `TENSOR_PARALLEL_SIZE` | 2 | 512k context does not fit on one 32 GB card |
| `MAX_MODEL_LEN` | 524288 | 512k via YaRN on the native 262k checkpoint |
| `MAX_NUM_SEQS` | 8 | KV fits ~34× 30k-token requests; GDN state ~40 MB/card/seq |
| `MAX_NUM_BATCHED_TOKENS` | 8192 | chunked prefill without activation OOM |
| `GPU_MEMORY_UTILIZATION` | 0.94 | vLLM 0.27 profiles CUDA-graph memory honestly; 0.94 ≈ 30.1 GB/card |
| `--kv-cache-dtype fp8_e4m3` | — | matches checkpoint calibration, 2x KV capacity |
| `--attention-backend flashinfer` | — | flash_attn rejects FP8 KV cache on this hybrid model |
| quantization | auto-detect | compressed-tensors checkpoint (NVFP4 W4A4 MLP + FP8 attention) |

Measured KV cache at startup (TP=2, util 0.94): **1,062,413 tokens → 2.03x concurrency at 524,288 tokens**. Worst-case concurrency is 2x at full 512k.

- **512k via YaRN**: `models/Qwen3.8-27B-NVFP4-yarn512k/config.json` sets `text_config.rope_parameters` to YaRN (`factor 2.0`, `original_max_position_embeddings 262144`, mrope/theta preserved). vLLM 0.27 dropped the `--rope-scaling` flag, so the patched config is the supported mechanism. `start.sh` builds the overlay: relative symlinks to the HF snapshot + this config, inside the mounted cache. YaRN is extrapolation: expect quality drift beyond 262k, the native trained range.
- **Startup time**: ~2.5 min warm, ~6 min cold. The container runs with `--restart unless-stopped`.
- **MTP rejected** (vLLM 0.27.1): good draft acceptance (~2.4 tokens/step), but verifying speculative tokens on this hybrid GDN model rewinds recurrent state at ~4x per engine step. Net: 55 tok/s solo vs 96 without. Revisit if vLLM optimizes GDN state handling for spec decode.
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
