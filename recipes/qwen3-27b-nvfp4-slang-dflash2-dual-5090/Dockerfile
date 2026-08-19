# SGLang with DFlash2 speculative decoding.
#
# DFlash2 merged into SGLang main on 2026-08-19 (c14312a6), hours after the last
# published nightly, so no image carries it. The dev image installs sglang
# editable from /sgl-workspace/sglang, so overlaying that python tree with the
# one from the merge commit is the whole patch — python only, the image's
# sgl-kernel build stays. Files deleted upstream since the image's commit are
# left behind; nothing in the newer tree imports them.
ARG BASE=lmsysorg/sglang:nightly-dev-cu13-20260818-c0b6474b
FROM ${BASE}

# c14312a6 = "[Spec] DFlash2: local convolution + candidate selector" (#35371).
# git fetch is not an option: the image's git has no GitHub credentials.
ARG SGLANG_COMMIT=c14312a66420b75ca9a11bf1817c4db1fa26b097
RUN curl -fsSL "https://codeload.github.com/sgl-project/sglang/tar.gz/${SGLANG_COMMIT}" -o /tmp/sglang.tgz \
 && tar xzf /tmp/sglang.tgz -C /tmp \
 && cp -a "/tmp/sglang-${SGLANG_COMMIT}/python/sglang/." /sgl-workspace/sglang/python/sglang/ \
 && rm -rf /tmp/sglang.tgz "/tmp/sglang-${SGLANG_COMMIT}" \
 && python3 -c "\
import inspect, sglang.srt.speculative.dflash_utils as d; \
src = inspect.getsource(d); \
assert 'selector_top_k' in src and 'conv_kernel_size' in src, 'DFlash2 draft config fields missing'"
