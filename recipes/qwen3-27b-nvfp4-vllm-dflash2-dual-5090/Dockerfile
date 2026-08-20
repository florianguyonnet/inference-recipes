ARG BASE=vllm/vllm-openai:nightly-5a4c8d99242e9e069b604d0e9b969e77f7dd501d
FROM ${BASE}

# PR #52816 head: DFlash2 architecture + the LM-head relaxation of #52883 +
# the candidate-selector compile-cache namespace fix (64c5b80).
ADD https://github.com/vllm-project/vllm/pull/52816.diff /tmp/dflash2.diff

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

# Accept a quantized target LM head. Reported working on compressed-tensors FP8
# (unsloth NVFP4) in vllm-project/vllm#52883: the candidate Top-K goes through
# quant_method.apply() either way, so the unquantized-only guard is stricter
# than the dispatch requires.
python3 - <<'PY'
p = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash2.py"
s = open(p).read()
old = "            (UnquantizedEmbeddingMethod, UnquantizedLinearMethod),\n"
new = "            (UnquantizedEmbeddingMethod, UnquantizedLinearMethod, object),\n"
assert s.count(old) == 1, "LM-head guard not found as expected"
open(p, "w").write(s.replace(old, new))
PY
python3 -c "import vllm.model_executor.models.qwen3_dflash2, vllm.v1.worker.gpu.spec_decode.dflash2.speculator"
SH
