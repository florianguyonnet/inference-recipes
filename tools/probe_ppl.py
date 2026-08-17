#!/usr/bin/env python3
"""Perplexity on a fixed text — the cheap calibration canary.

Prefill-only passes with `echo=true, logprobs=1`: no sampling, no reasoning, no
answer extraction, so the number only moves when the weights, the KV cache dtype
or the rope config do. ~30 s per run, where an accuracy suite needs 20 min.

Compare runs of the SAME checkpoint (quantization, kv-cache-dtype, YaRN on/off,
vLLM vs SGLang). Absolute values are not comparable across checkpoints, and not
comparable to published wikitext numbers either: this scores short independent
chunks, not a sliding window.

    ./.venv/bin/python probe_ppl.py --url http://localhost:8000 --model Qwen3.8-27B-NVFP4

Rule of thumb: ±1% between engines is measurement drift, +10% or more means the
serving config is damaging the model.

Chunks stay small on purpose. `prompt_logprobs` materializes fp32 logits for
every prompt position (~0.6 MB/token at this vocab size), and a server at
`--gpu-memory-utilization 0.94` has ~200 MB of headroom: a 2.5k-token echo
request OOMs the engine and kills it (measured on vLLM 0.27.1).
"""

import argparse
import json
import math
import urllib.error
import urllib.request

WIKITEXT = ("Salesforce/wikitext", "wikitext-2-raw-v1", "test")


def load_chunks(chunk_chars: int, count: int) -> list[str]:
    """Deterministic prose chunks from wikitext-2 test (headings/blanks skipped)."""
    from datasets import load_dataset

    path, name, split = WIKITEXT
    text = "\n".join(
        line.strip() for line in load_dataset(path, name, split=split)["text"] if len(line.strip()) > 200
    )
    return [text[i * chunk_chars : (i + 1) * chunk_chars] for i in range(count)]


def chunk_nll(url: str, model: str, text: str) -> tuple[float, int]:
    body = {
        "model": model,
        "prompt": text,
        "max_tokens": 0,
        "echo": True,
        "logprobs": 1,
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"{url}/v1/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        resp = json.load(urllib.request.urlopen(req, timeout=300))
    except urllib.error.HTTPError as e:
        raise SystemExit(
            f"HTTP {e.code} on an echo request: {e.read().decode()[:200]}\n"
            "If the engine died: prompt_logprobs needs GPU headroom. Use a smaller "
            "--chunk-chars, or serve with --gpu-memory-utilization 0.90."
        ) from None
    logprobs = [lp for lp in resp["choices"][0]["logprobs"]["token_logprobs"] if lp is not None]
    if not logprobs:
        raise SystemExit("server returned no logprobs (needs echo+logprobs support)")
    return -sum(logprobs), len(logprobs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen3.8-27B-NVFP4")
    ap.add_argument("--chunk-chars", type=int, default=400, help="~100 tokens/request")
    ap.add_argument("--chunks", type=int, default=64)
    args = ap.parse_args()

    total_nll = 0.0
    total_tokens = 0
    for i, text in enumerate(load_chunks(args.chunk_chars, args.chunks), 1):
        nll, n = chunk_nll(args.url, args.model, text)
        total_nll += nll
        total_tokens += n
        if i % 16 == 0:
            print(f"    {i}/{args.chunks} chunks, ppl so far {math.exp(total_nll / total_tokens):.4f}")
    mean = total_nll / total_tokens
    print(f"==> {args.model}: {total_tokens} tokens, mean NLL {mean:.4f}, ppl {math.exp(mean):.4f}")


def demo() -> None:
    """Self-check: the aggregation is token-weighted, not chunk-averaged."""
    chunks = [(10.0, 5), (30.0, 15)]  # nll sums, token counts
    total = sum(n for n, _ in chunks) / sum(t for _, t in chunks)
    assert abs(total - 2.0) < 1e-9, total
    print("ok")


if __name__ == "__main__":
    main()
