#!/usr/bin/env python3
"""
Benchmark vLLM OpenAI-compatible server for agent workloads.

Measures:
- TTFT  : time to first token (streaming)
- TPOT  : time per output token
- Total : end-to-end latency

Usage:
    python scripts/benchmark_agent.py \
        --url http://localhost:8000 \
        --model Qwen3.8-27B-NVFP4 \
        --input-lens 512,2048,8192 \
        --output-len 256 \
        --num-requests 8 \
        --concurrency 2
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass

import aiohttp


@dataclass
class Result:
    ttft: float
    tpot: float
    total: float
    prompt_tokens: int
    completion_tokens: int


async def stream_completion(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> Result:
    t0 = time.perf_counter()
    ttft = None
    token_times = []

    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    async with session.post(
        f"{url}/v1/completions",
        json=body,
        timeout=aiohttp.ClientTimeout(total=600),
    ) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {text}")

        buffer = b""
        async for chunk in resp.content.iter_any():
            buffer += chunk
            while b"\n\n" in buffer:
                line, buffer = buffer.split(b"\n\n", 1)
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    break
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                token_times.append(now)

    if ttft is None:
        raise RuntimeError("No tokens received")

    total = time.perf_counter() - t0
    n_tokens = len(token_times)
    if n_tokens > 1:
        intervals = [
            token_times[i] - token_times[i - 1] for i in range(1, n_tokens)
        ]
        tpot = statistics.mean(intervals)
    else:
        tpot = 0.0

    return Result(
        ttft=ttft,
        tpot=tpot,
        total=total,
        prompt_tokens=0,
        completion_tokens=n_tokens,
    )


async def worker(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    results: list,
    index: int,
) -> None:
    try:
        r = await stream_completion(session, url, model, prompt, max_tokens)
        results[index] = r
    except Exception as exc:
        print(f"  Request {index} failed: {exc}")


async def run(
    url: str,
    model: str,
    input_len: int,
    output_len: int,
    num_requests: int,
    concurrency: int,
) -> None:
    prompt = "Explain the difference between concurrency and parallelism. " * (
        input_len // 10
    )

    async with aiohttp.ClientSession() as session:
        results: list = [None] * num_requests
        semaphore = asyncio.Semaphore(concurrency)

        async def sem_worker(idx: int) -> None:
            async with semaphore:
                await worker(session, url, model, prompt, output_len, results, idx)

        t0 = time.perf_counter()
        await asyncio.gather(*(sem_worker(i) for i in range(num_requests)))
        wall_time = time.perf_counter() - t0

    valid = [r for r in results if r is not None]
    if not valid:
        print("No successful requests")
        return

    ttfts = [r.ttft for r in valid]
    tpots = [r.tpot for r in valid]
    totals = [r.total for r in valid]
    tokens = [r.completion_tokens for r in valid]

    print()
    print(f"Input tokens (approx): {input_len}")
    print(f"Output tokens per req: {output_len}")
    print(f"Requests             : {num_requests}")
    print(f"Concurrency          : {concurrency}")
    print(f"Wall time            : {wall_time:.2f}s")
    print(f"{'Metric':<18}{'Mean':>10}{'Median':>10}{'p99':>10}")
    print("-" * 48)
    print(
        f"{'TTFT (s)':<18}{statistics.mean(ttfts):>10.3f}"
        f"{statistics.median(ttfts):>10.3f}{sorted(ttfts)[int(len(ttfts) * 0.99)]:>10.3f}"
    )
    print(
        f"{'TPOT (s)':<18}{statistics.mean(tpots):>10.3f}"
        f"{statistics.median(tpots):>10.3f}{sorted(tpots)[int(len(tpots) * 0.99)]:>10.3f}"
    )
    print(
        f"{'Total (s)':<18}{statistics.mean(totals):>10.3f}"
        f"{statistics.median(totals):>10.3f}{sorted(totals)[int(len(totals) * 0.99)]:>10.3f}"
    )
    print(f"{'Throughput (tok/s)':<18}{sum(tokens) / wall_time:>10.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-lens", default="512,2048,8192")
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    for input_len in [int(x) for x in args.input_lens.split(",")]:
        print(f"\n=== Input length ~{input_len} tokens ===")
        asyncio.run(
            run(
                url=args.url,
                model=args.model,
                input_len=input_len,
                output_len=args.output_len,
                num_requests=args.num_requests,
                concurrency=args.concurrency,
            )
        )


if __name__ == "__main__":
    main()
