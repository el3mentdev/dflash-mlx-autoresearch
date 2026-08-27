#!/usr/bin/env python3
"""
Verify prefix-match cache correctness + timing against a running dflash-serve.

Phase A - smoke test (one invocation, one server):
    python test_prefix_cache.py --phase smoke

    Sends turn 1 then turn 2 where turn 2 extends turn 1. Expects usage.prompt_tokens_details.cached_tokens on turn 2 to cover >= 90% of turn 1's prompt.

Phase B - correctness check (two invocations, restart between):
    1. Start dflash-serve.
    2. python test_prefix_cache.py --phase warm     # seeds cache, saves turn 2 output.
    3. Restart dflash-serve (drops the in-process prefix cache).
    4. python test_prefix_cache.py --phase cold     # runs turn 2 from cold, saves output.
    5. python test_prefix_cache.py --phase compare  # prints token-match rate.

Correctness target: warm and cold turn-2 outputs match (ideally bit-identical at temp=0 / seed=42).
"""

import argparse
import json
import os
import time
import urllib.request

BASE = os.environ.get("DFLASH_BASE_URL", "http://127.0.0.1:8000") + "/v1/chat/completions"
MODEL = os.environ.get("DFLASH_MODEL", "default")  # dflash-serve resolves "default"
MAX_TOKENS = 300
SEED = 42

# Padding chosen so turn 1 prompt is large enough that prefill dominates.
# "We model requests as discrete events. " is ~8 tokens, so 200x => ~1600 tokens.
TURN1_USER = (
    "You are a senior engineer. Read the following spec carefully and then wait for questions. "
    "Spec: " + "We model requests as discrete events. " * 200
)
TURN1_ASSISTANT = (
    "Understood. The spec repeatedly states that we model requests as discrete events. "
    "Ready for questions."
)
TURN2_USER = "Now summarize what's in the spec in exactly three bullet points."


def _chat(messages, max_tokens=MAX_TOKENS, seed=SEED):
    body = json.dumps(
        {
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": seed,
        }
    ).encode()
    req = urllib.request.Request(
        BASE, data=body, headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as resp:
        payload = json.loads(resp.read().decode())
    elapsed = time.perf_counter() - start
    return payload, elapsed


def _turn2_messages():
    return [
        {"role": "user", "content": TURN1_USER},
        {"role": "assistant", "content": TURN1_ASSISTANT},
        {"role": "user", "content": TURN2_USER},
    ]


def _print_usage(tag, resp, elapsed):
    usage = resp.get("usage", {}) or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    print(
        f"  [{tag}] elapsed={elapsed:.2f}s "
        f"prompt_tokens={usage.get('prompt_tokens')} "
        f"cached_tokens={cached} "
        f"completion_tokens={usage.get('completion_tokens')}"
    )
    return usage, cached


def run_smoke():
    print("[smoke] turn 1 (cold)")
    r1, t1 = _chat([{"role": "user", "content": TURN1_USER}])
    usage1, _ = _print_usage("turn1", r1, t1)

    print("[smoke] turn 2 (should partial-hit on turn 1's tokens)")
    r2, t2 = _chat(_turn2_messages())
    _, cached2 = _print_usage("turn2", r2, t2)

    expected_hit = int((usage1.get("prompt_tokens") or 0) * 0.9)
    if cached2 >= expected_hit and expected_hit > 0:
        print(f"[smoke] PASS: turn 2 cached_tokens={cached2} >= 0.9 x turn1 prompt ({expected_hit})")
    else:
        print(
            f"[smoke] CHECK: cached_tokens={cached2}, expected >= {expected_hit}. "
            "Inspect server stderr for 'prefix cache HIT (prefix)'."
        )


def _save(path, resp, elapsed):
    msg = resp["choices"][0]["message"]
    out = {
        "content": msg.get("content", ""),
        "finish_reason": resp["choices"][0].get("finish_reason"),
        "usage": resp.get("usage", {}),
        "elapsed_s": elapsed,
    }
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  wrote {path}")


def run_warm():
    print("[warm] turn 1 (primes cache)")
    r1, t1 = _chat([{"role": "user", "content": TURN1_USER}])
    _print_usage("turn1", r1, t1)
    print("[warm] turn 2 (should partial-hit)")
    r2, t2 = _chat(_turn2_messages())
    _print_usage("turn2", r2, t2)
    _save("test_prefix_warm.json", r2, t2)


def run_cold():
    print("[cold] turn 2 only (server must have been restarted since any warm run)")
    r2, t2 = _chat(_turn2_messages())
    _print_usage("turn2", r2, t2)
    _save("test_prefix_cold.json", r2, t2)


def run_compare():
    with open("test_prefix_warm.json") as fh:
        warm = json.load(fh)
    with open("test_prefix_cold.json") as fh:
        cold = json.load(fh)
    w, c = warm["content"], cold["content"]
    print(f"  warm elapsed: {warm.get('elapsed_s'):.2f}s  cold elapsed: {cold.get('elapsed_s'):.2f}s")
    if w == c:
        print("[compare] EXACT MATCH")
        return
    shared = 0
    for a, b in zip(w, c):
        if a != b:
            break
        shared += 1
    longest = max(len(w), len(c))
    pct = (shared / longest * 100) if longest else 0.0
    print(f"[compare] prefix match: {shared}/{longest} chars ({pct:.1f}%)")
    print(f"  warm head: {w[:200]!r}")
    print(f"  cold head: {c[:200]!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["smoke", "warm", "cold", "compare"], default="smoke")
    args = p.parse_args()
    {
        "smoke": run_smoke,
        "warm": run_warm,
        "cold": run_cold,
        "compare": run_compare,
    }[args.phase]()


if __name__ == "__main__":
    main()
