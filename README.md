# LLM inference recipes

Tested, reproducible setups for serving self-hosted models. Each recipe ships a container launcher, tuned configs, and measured performance.

## Recipes

| Recipe | Checkpoint | Engine | Solo decode | 7-8 concurrent | Context |
|---|---|---|---|---|---|
| [qwen3-27b-nvfp4-slang-dflash2-dual-5090](recipes/qwen3-27b-nvfp4-slang-dflash2-dual-5090/) | Qwen3.8-27B-NVFP4 (Inferact) | SGLang main + DFlash2 | 178-187 tok/s | 624 tok/s | 250k (KV pool) |
| [qwen3-27b-nvfp4-slang-dual-5090](recipes/qwen3-27b-nvfp4-slang-dual-5090/) | Qwen3.8-27B-NVFP4 (RadixArk) | SGLang + DSpark | 120-190 tok/s | 434 tok/s | 262k (native) |
| [qwen3-27b-nvfp4-vllm-dflash2-dual-5090](recipes/qwen3-27b-nvfp4-vllm-dflash2-dual-5090/) | Qwen3.8-27B-NVFP4 (Inferact) | vLLM nightly + DFlash2 | 133-139 tok/s | 266 tok/s | 262k (native) |
| [qwen3-27b-nvfp4-vllm-dual-5090](recipes/qwen3-27b-nvfp4-vllm-dual-5090/) | Qwen3.8-27B-NVFP4 (unsloth) | vLLM 0.27.1 | 96 tok/s | 368 tok/s | 512k (YaRN) |

Both DFlash2 recipes need a patched engine: the drafter shipped 2026-08-18, and
neither vLLM nor SGLang has it in a release. Their default profiles favour
latency, so the concurrent column is not their ceiling — see each README.

Hardware for both: 2× NVIDIA RTX 5090 (32 GB, sm_120), KVM VM, Docker + nvidia-container-toolkit.

## Usage

Each recipe is self-contained: copy its directory to the target machine and follow its README. Bench and evaluation clients are shared in `tools/` and work against any OpenAI-compatible server, so every recipe is measured with the same protocol.
