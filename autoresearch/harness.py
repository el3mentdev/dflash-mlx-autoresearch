#!/usr/bin/env python3
"""
Fixed benchmark harness for autoresearch DFlash optimization.

**READ-ONLY for the agent.** The agent edits files in `dflash_mlx/` and runs
this harness via `uv run autoresearch/harness.py` (or the venv equivalent) to
measure impact. All randomness is seeded. All prompts, models, and budgets
are pinned. Output format matches `program.md`.

Exit code 0 always (crash detection is via missing lines in the output block).
"""

from __future__ import annotations

import json
import os
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

# ---------------- Pinned configuration (do not change) ----------------
# Model/data locations are overridable via env vars so the pinned benchmark
# itself (prompts, seeds, budgets) stays fixed across machines.
TARGET_MODEL = os.environ.get(
    "AUTORESEARCH_TARGET_MODEL", "TheCluster/Qwen3.5-27B-Heretic-MLX-4bit"
)
DRAFT_MODEL = os.environ.get("AUTORESEARCH_DRAFT_MODEL", "z-lab/Qwen3.5-27B-DFlash")
SPECBENCH_QUESTIONS = Path(
    os.environ.get(
        "AUTORESEARCH_SPECBENCH_QUESTIONS",
        str(
            Path(__file__).resolve().parents[2]
            / "Spec-Bench/data/spec_bench/question.jsonl"
        ),
    )
)
# Hand-picked balanced 10-prompt subset: 6 strong-tier + 4 weak-tier for
# regression detection across the category distribution.
PROMPT_QIDS: list[int] = [
    # strong tier (expect >=1.5x speedup)
    111,  # math
    449,  # math_reasoning
    108,  # reasoning
    146,  # stem
    314,  # summarization
    # medium tier
    485,  # rag
    # weak tier (regression detection)
    151,  # humanities
    336,  # qa
    93,   # roleplay
    239,  # translation
]
MAX_TOKENS = 256
BLOCK_TOKENS = 16
SEED = 42
# ---------------------------------------------------------------------


def load_prompts() -> list[dict]:
    if not SPECBENCH_QUESTIONS.exists():
        sys.exit(
            f"missing Spec-Bench data at {SPECBENCH_QUESTIONS} — "
            "clone https://github.com/hemingkx/Spec-Bench first"
        )
    by_qid: dict[int, dict] = {}
    with open(SPECBENCH_QUESTIONS) as fh:
        for line in fh:
            q = json.loads(line)
            by_qid[q["question_id"]] = q
    missing = [qid for qid in PROMPT_QIDS if qid not in by_qid]
    if missing:
        sys.exit(f"missing question ids in Spec-Bench: {missing}")
    return [by_qid[qid] for qid in PROMPT_QIDS]


def main() -> int:
    t_wall_start = time.time()
    random.seed(SEED)

    try:
        import mlx.core as mx

        mx.random.seed(SEED)
        from dflash_mlx.generate import get_stop_token_ids
        from dflash_mlx.runtime import (
            generate_baseline_once,
            generate_dflash_once,
            load_draft_bundle,
            load_target_bundle,
        )
    except Exception:
        traceback.print_exc()
        _print_summary(crashed=True)
        return 0

    prompts = load_prompts()

    try:
        target, tok, _ = load_target_bundle(TARGET_MODEL, lazy=True)
        draft, _ = load_draft_bundle(DRAFT_MODEL, lazy=True)
    except Exception:
        traceback.print_exc()
        _print_summary(crashed=True)
        return 0

    stops = get_stop_token_ids(tok)
    per_prompt = []
    peak_mem_gb = 0.0

    for i, q in enumerate(prompts):
        prompt = q["turns"][0]
        try:
            baseline = generate_baseline_once(
                target_model=target,
                tokenizer=tok,
                prompt=prompt,
                max_new_tokens=MAX_TOKENS,
                use_chat_template=True,
                stop_token_ids=stops,
            )
            dflash = generate_dflash_once(
                target_model=target,
                tokenizer=tok,
                draft_model=draft,
                prompt=prompt,
                max_new_tokens=MAX_TOKENS,
                use_chat_template=True,
                block_tokens=BLOCK_TOKENS,
                stop_token_ids=stops,
            )
        except Exception:
            print(f"[prompt {i} qid={q['question_id']}] CRASHED:")
            traceback.print_exc()
            _print_summary(crashed=True)
            return 0

        b_gen_us = max(1.0, baseline["elapsed_us"] - baseline.get("prefill_us", 0.0))
        d_gen_us = max(
            1.0,
            dflash["elapsed_us"]
            - dflash.get("phase_timings_us", {}).get("prefill", 0.0),
        )
        b_tps = baseline["generation_tokens"] / (b_gen_us / 1e6)
        d_tps = dflash["generation_tokens"] / (d_gen_us / 1e6)
        speedup = d_tps / b_tps if b_tps > 0 else 0.0
        accept = float(dflash.get("acceptance_ratio", 0.0))
        match = baseline["generated_token_ids"] == dflash["generated_token_ids"]

        per_prompt.append(
            {
                "qid": q["question_id"],
                "category": q["category"],
                "baseline_tps": b_tps,
                "dflash_tps": d_tps,
                "speedup": speedup,
                "acceptance": accept,
                "token_match": match,
            }
        )

        try:
            if hasattr(mx, "get_peak_memory"):
                peak_mem_gb = max(peak_mem_gb, mx.get_peak_memory() / 1e9)
        except Exception:
            pass

        print(
            f"[{i+1:>2}/{len(prompts)}] qid={q['question_id']} cat={q['category']:<14} "
            f"base={b_tps:5.1f} dflash={d_tps:5.1f} speedup={speedup:.2f} "
            f"accept={accept:.2%} match={int(match)}",
            flush=True,
        )

    wall = time.time() - t_wall_start
    _print_summary(
        per_prompt=per_prompt, wall_seconds=wall, peak_mem_gb=peak_mem_gb, crashed=False
    )
    return 0


def _print_summary(
    *,
    per_prompt: list[dict] | None = None,
    wall_seconds: float = 0.0,
    peak_mem_gb: float = 0.0,
    crashed: bool,
) -> None:
    print("\n---")
    if crashed or not per_prompt:
        print(f"crashed:          1")
        print(f"baseline_tps:     0.0")
        print(f"dflash_tps:       0.0")
        print(f"speedup:          0.000")
        print(f"acceptance:       0.000")
        print(f"token_match_rate: 0.000")
        print(f"peak_memory_gb:   0.0")
        print(f"wall_seconds:     {wall_seconds:.1f}")
        print(f"n_prompts:        0")
        return

    base_tps = statistics.median(r["baseline_tps"] for r in per_prompt)
    dflash_tps = statistics.median(r["dflash_tps"] for r in per_prompt)
    speedup = statistics.median(r["speedup"] for r in per_prompt)
    accept = statistics.median(r["acceptance"] for r in per_prompt)
    match_rate = sum(1 for r in per_prompt if r["token_match"]) / len(per_prompt)
    print(f"crashed:          0")
    print(f"baseline_tps:     {base_tps:.2f}")
    print(f"dflash_tps:       {dflash_tps:.2f}")
    print(f"speedup:          {speedup:.3f}")
    print(f"acceptance:       {accept:.3f}")
    print(f"token_match_rate: {match_rate:.3f}")
    print(f"peak_memory_gb:   {peak_mem_gb:.2f}")
    print(f"wall_seconds:     {wall_seconds:.1f}")
    print(f"n_prompts:        {len(per_prompt)}")


if __name__ == "__main__":
    raise SystemExit(main())
