#!/usr/bin/env python3
"""
Tight controlled decode benchmark against a running dflash-serve.

Reports prefill latency + decode tok/s for each call. Same prompt repeated to
measure warm vs cold (warm hits the prefix cache).

Usage:
    python bench_decode.py [--prompt-tokens 1000] [--max-tokens 100] [--n 3]

Default: 1000-token prompt, 100-token output, 3 sequential calls (1 cold + 2 warm).
"""
import argparse
import json
import os
import time
import urllib.request

BASE = os.environ.get("DFLASH_BASE_URL", "http://127.0.0.1:8000") + "/v1/chat/completions"


def _chat(model, messages, max_tokens, seed=42):
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
    }).encode()
    req = urllib.request.Request(
        BASE, data=body, headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as resp:
        payload = json.loads(resp.read().decode())
    elapsed = time.perf_counter() - start
    return payload, elapsed


def _build_prompt(target_tokens):
    """Build a prompt that tokenizes to roughly `target_tokens` tokens.
    Uses a sentence that's ~8 tokens, repeated."""
    sentence = "Modern compilers do significant inlining and dead-code elimination during release builds. "
    # ~8 tokens per sentence (rough estimate). Add a small instruction prefix.
    prefix = "You are a senior systems engineer. Answer the following question briefly. Background: "
    suffix = "\n\nQuestion: summarize the above background in one sentence."
    body_count = max(1, (target_tokens - 30) // 8)
    return prefix + (sentence * body_count) + suffix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt-tokens", type=int, default=1000)
    p.add_argument("--max-tokens", type=int, default=100)
    p.add_argument("--n", type=int, default=3, help="sequential calls")
    p.add_argument("--model", default="default", help="model name to send (server resolves)")
    args = p.parse_args()

    prompt_text = _build_prompt(args.prompt_tokens)
    messages = [{"role": "user", "content": prompt_text}]

    print(f"[bench] target_prompt_tokens={args.prompt_tokens}  max_tokens={args.max_tokens}  runs={args.n}")
    print(f"[bench] {'idx':>3} {'wall_s':>8} {'prompt':>7} {'cached':>7} {'gen':>5} {'decode_s':>9} {'tok/s':>7}")

    last_prefill_us = None
    for i in range(args.n):
        resp, wall = _chat(args.model, messages, args.max_tokens)
        usage = resp.get("usage", {}) or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        cached_tokens = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        # Approx decode time: total wall - prefill. We don't have prefill directly
        # via the API, but for cold runs prefill dominates the gap. For warm it's
        # tiny. Estimate decode_s = wall - 0 for warm, wall - (prefill estimate).
        # Simplest: report wall + tok/s computed from completion / (wall - small).
        # Better: decode_s = wall - max(0, prefill_estimate). We don't have it,
        # so use: tok/s = completion / wall is conservative; tok/s from end-to-end.
        # Instead, expose just completion/wall as the "end-to-end tok/s" and let
        # the user note prefill from server logs.
        decode_s_approx = max(0.001, wall)  # end-to-end; underestimates tok/s on cold
        toks_per_s = completion_tokens / decode_s_approx
        kind = "cold" if i == 0 else "warm"
        print(
            f"[bench] {i:>3} {wall:>8.2f} {prompt_tokens:>7} {cached_tokens:>7} "
            f"{completion_tokens:>5} {decode_s_approx:>9.2f} {toks_per_s:>7.1f}  ({kind})"
        )

    print("\nNote: tok/s above is END-TO-END (includes prefill). Cross-reference server")
    print("stderr for the line `[dflash] X tok/s | Y% accepted | Z tokens | T s | prompt: P tokens`")
    print("which reports DECODE-only tok/s (prefill excluded).")


if __name__ == "__main__":
    main()
