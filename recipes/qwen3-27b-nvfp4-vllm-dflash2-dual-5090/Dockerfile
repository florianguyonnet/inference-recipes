# vLLM with DFlash2 speculative decoding.
#
# No released vLLM has DFlash2: PR #52816 adds the architecture and #52883
# (stacked on it, so its diff carries both) fixes the LM-head guard a quantized
# target needs. Both are pure Python, so the PR diff goes onto the nightly wheel
# in place — no rebuild of the CUDA extensions.
#
# BASE is the nightly closest to the PR's base commit (9842d70, 31 commits
# behind). `patch` fails the build if either side drifts far enough to conflict.
ARG BASE=vllm/vllm-openai:nightly-5a4c8d99242e9e069b604d0e9b969e77f7dd501d
FROM ${BASE}

ARG PR=52883
ADD https://github.com/vllm-project/vllm/pull/${PR}.diff /tmp/dflash2.diff

# The diff also touches tests/, which the wheel does not ship: keep the vllm/
# files only. patch is in the base image, git is not.
RUN <<'SH'
set -eu
python3 - <<'PY'
import re
diff = open("/tmp/dflash2.diff").read()
keep = [f for f in re.split(r"(?m)^(?=diff --git )", diff) if f.startswith("diff --git a/vllm/")]
assert len(keep) >= 7, f"only {len(keep)} vllm/ files in the diff"
open("/tmp/dflash2-vllm.diff", "w").write("".join(keep))
PY
patch -p1 -d /usr/local/lib/python3.12/dist-packages --no-backup-if-mismatch -i /tmp/dflash2-vllm.diff
python3 -c "import vllm.model_executor.models.qwen3_dflash2, vllm.v1.worker.gpu.spec_decode.dflash2.speculator"
SH
