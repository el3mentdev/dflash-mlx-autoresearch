# autoresearch: DFlash-MLX throughput optimization

Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
Goal: **maximize DFlash decoding throughput (tok/s) on Apple Silicon** without
sacrificing losslessness.

## Setup

1. Branch `autoresearch/<tag>` already exists (e.g. `autoresearch/apr15`). Work
   on that branch.
2. Read the in-scope files before you begin:
   - `README.md` — upstream repository context.
   - `autoresearch/harness.py` — **read-only**. Pinned benchmark: 10 Spec-Bench
     prompts, seeded, `MAX_TOKENS=256`, `BLOCK_TOKENS=16`.
   - `autoresearch/prompts.jsonl` — NOT used; harness loads from Spec-Bench path.
   - `dflash_mlx/runtime.py` — the verify/commit/rollback driver loop. **Editable.**
   - `dflash_mlx/kernels.py` — custom Metal kernels. **Editable.**
   - `dflash_mlx/recurrent_rollback_cache.py` — KV rollback cache. **Editable.**
3. **Do NOT edit**: `dflash_mlx/model.py` (draft architecture),
   `dflash_mlx/generate.py`, `dflash_mlx/serve.py`, `dflash_mlx/__init__.py`,
   `benchmark/`, the `autoresearch/` directory itself, `pyproject.toml`.
4. The Python environment at `~/.venvs/dflash-mlx/` has this repo installed in
   **editable** mode. Your edits take effect on the next harness run — no rebuild
   needed.
5. Verify `results.tsv` exists with just a header (`commit\tspeedup\tdflash_tps\ttoken_match\tstatus\tdescription`). The baseline will be recorded after the first run.

## Experimentation

Each experiment runs one harness invocation: `~/.venvs/dflash-mlx/bin/python autoresearch/harness.py > run.log 2>&1`.

The harness is deterministic (seeded). Target wall-time is **~3 minutes**. If a
run exceeds **10 minutes**, kill it, treat it as a failure, and revert.

**Primary metric**: `speedup` (DFlash tok/s ÷ baseline tok/s), higher is better.
**Guard rail**: `token_match_rate` must stay within **0.05** of the baseline
run's value. A change that increases speedup by making the output diverge from
the baseline greedy trajectory is not a real win — auto-discard.

**What you CAN do:**
- Modify `dflash_mlx/runtime.py` — the speculative decoding loop, verify/commit
  logic, block sizing heuristics, KV management calls.
- Modify `dflash_mlx/kernels.py` — the custom MLX Metal kernels
  (`tape_replay_kernel` and friends). Only if you actually know what you're
  doing at the Metal shader level; otherwise stay away.
- Modify `dflash_mlx/recurrent_rollback_cache.py` — cache allocation, eviction,
  pack/unpack formats for draft + target KV.

**What you CANNOT do:**
- Modify `autoresearch/harness.py`. It is the ground-truth benchmark.
- Modify the evaluation prompts or seeds.
- Modify `dflash_mlx/model.py` (changes the drafter architecture — off-limits
  because the drafter is a distilled z-lab checkpoint and swapping architecture
  would invalidate comparisons).
- Install new packages.

**Simplicity criterion**: same as upstream autoresearch. A small speedup gain
that adds ugly complexity is probably not worth it. Cleaner code at equal or
better speedup is a win.

## Output format

The harness prints a summary block at the end:

```
---
crashed:          0
baseline_tps:     23.7
dflash_tps:       37.0
speedup:          1.540
acceptance:       0.832
token_match_rate: 0.400
peak_memory_gb:   19.3
wall_seconds:     182.4
n_prompts:        10
```

Extract with:
```
grep -E "^(crashed|baseline_tps|dflash_tps|speedup|token_match_rate|peak_memory_gb):" run.log
```

Exit code is always 0. Crash is signaled by `crashed: 1` and zero metrics.

## Logging results

`results.tsv` (tab-separated, 6 columns):

```
commit	speedup	dflash_tps	token_match	status	description
```

Status values:
- `keep` — change kept, speedup improved, guard rails held
- `discard` — speedup did not improve OR token_match_rate dropped below baseline - 0.05
- `crash` — harness reported crashed: 1

Do NOT commit `results.tsv` — keep it untracked.

## Experiment loop

LOOP FOREVER:

1. Look at git state: current branch/commit.
2. Form a hypothesis and edit one or more files in `dflash_mlx/` to test it.
3. `git add -A && git commit -m "<short description of the change>"`
4. `~/.venvs/dflash-mlx/bin/python autoresearch/harness.py > run.log 2>&1`
5. `grep -E "^(crashed|baseline_tps|dflash_tps|speedup|token_match_rate):" run.log`
6. If the grep output is empty, the run didn't complete — read `tail -50 run.log`,
   fix if the issue is trivial, otherwise revert and move on.
7. Append a row to `results.tsv`.
8. Decision:
   - `speedup` improved AND `token_match_rate` ≥ baseline - 0.05 → **keep** (leave
     branch HEAD on the new commit).
   - `speedup` did not improve OR guard rail violated → **discard**
     (`git reset --hard HEAD~1`).
   - Harness crashed → **crash**; revert (`git reset --hard HEAD~1`).

## Hypothesis buckets

If stuck, here are directions that are worth trying on an MLX Apple-Silicon
runtime:

1. **Block size sweep** — the default `BLOCK_TOKENS=16` is pinned in the harness
   for consistency, but the runtime's internal verify batching can be retuned
   independently (`verify_chunk_tokens`, see `runtime.py`).
2. **KV cache packing** — `recurrent_rollback_cache.py` allocates fresh arrays
   for rollback snapshots. Reuse buffers or switch to in-place copies.
3. **Attention mask construction cost** — `build_suppress_token_mask` etc. run
   per-prompt; hoist them out of hot loops.
4. **Draft window / sliding attention** — the runtime calls
   `_resolve_draft_window()` on every generate; cache it.
5. **`mx.eval` batching** — reduce the number of `mx.eval` sync points in the
   draft/verify/commit phases.
6. **Kernel fusion** — combine sequential Metal operations in `kernels.py` where
   profitable (careful — wrong launches can hang Metal silently).

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human if you should
continue. The human expects you to continue working indefinitely until manually
stopped.
