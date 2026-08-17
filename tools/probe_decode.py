#!/usr/bin/env python3
"""
Fixed-prompt decode probe for spec-decode sweeps.

Sends N fixed streaming chat prompts (temperature 0.6, the DSpark eval
setting) and measures decode speed client-side:
  decode_tput = completion_tokens / (last_token_time - first_token_time)
Token counts come from usage (stream_options.include_usage), so spec-decode
SSE batching does not distort them. Comparable across configs: prompts,
lengths and sampling never change.

Usage: probe_decode.py [--url http://localhost:30000] [--model NAME] [--tokens 512]
Prints: "tput_med=X tput_max=Y tokens_total=Z" or PROBE-FAIL.
"""

import argparse
import json
import statistics
import time
import urllib.request

PROMPTS = [
    "Explain in detail how speculative decoding works for LLM inference.",
    "Write a Python function that merges two sorted lists, then explain its complexity.",
    "Describe the process of photosynthesis step by step, for a biology student.",
]


def measure(url: str, model: str, prompt: str, tokens: int) -> float:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": tokens,
        "temperature": 0.6,
        "top_k": 20,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
        # Decode exactly `tokens` every time. Without it a run that stops early
        # is averaged over a shorter, warmer window, which is a large part of the
        # run-to-run swing that made earlier sweeps unreadable.
        "ignore_eos": True,
    }
    req = urllib.request.Request(
        f"{_args.url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft = None
    completion_tokens = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        buf = b""
        for chunk in resp:
            buf += chunk
            while b"\n\n" in buf:
                line, buf = buf.split(b"\n\n", 1)
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    break
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if ttft is None and data.get("choices"):
                    ttft = time.perf_counter()
                usage = data.get("usage")
                if usage and usage.get("completion_tokens"):
                    completion_tokens = usage["completion_tokens"]
    end = time.perf_counter()
    if ttft is None or completion_tokens < 8:
        return 0.0
    return completion_tokens / (end - ttft)


_args = None


def main() -> None:
    global _args
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:30000")
    p.add_argument("--model", default="Qwen3.8-27B-NVFP4")
    p.add_argument("--tokens", type=int, default=512)
    _args = p.parse_args()

    # warmup (also primes prefix cache/radix state)
    measure(_args.url, _args.model, "Say hello.", 16)

    rates = []
    for prompt in PROMPTS:
        r = measure(_args.url, _args.model, prompt, _args.tokens)
        rates.append(r)
        print(f"  prompt: {prompt[:40]}... -> {r:.1f} tok/s")

    rates = [r for r in rates if r > 0]
    if not rates:
        print("PROBE-FAIL")
        return
    print(f"tput_med={statistics.median(rates):.1f} tput_max={max(rates):.1f}")


if __name__ == "__main__":
    main()
