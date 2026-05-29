---
name: running-eval-suite
description: Run all reference benchmarks under benchmarks/eval/ and refresh the reference-table cells in benchmark_*.py. Auto-detects host hardware (H200/H100/H800/...). For each row's matching hw entry: replace cells in place. For new hardware or a new workload tag: append a new row to the section's table. Local-Pipeline-Result tables are skipped. One slash command runs everything end-to-end and auto-commits.
---

# running-eval-suite

## Scope

A contributor on any sglang-omni dev container runs `/running-eval-suite`
to refresh every reference table cell in `benchmarks/eval/benchmark_*.py`
that corresponds to the host's hardware:

1. Auto-detect host hardware (`H200` / `H100` / `H800` / `A100` / ...).
2. Run every row declared in `models/<model>/config.yaml`.
3. For each row, after the client succeeds:
   - If the file has a row whose Source column contains
     `[<hw>, <workload>]` matching this entry → **replace** its cells.
   - Otherwise → **append** a new row at the end of that section's
     table.
4. `Local v1 Pipeline Result` and similar Local-* sections are
   **skipped entirely** — those are contributors' personal experiments,
   not refresh targets.
5. Commit `benchmarks/eval/` to a local commit. **No push.**

The skill never edits anything outside `benchmarks/eval/`. Cell rewrite
is exact-string replacement on the data row line; the audit trail is
plain `git show HEAD`.

## Models

```
.claude/skills/running-eval-suite/models/<model>/config.yaml
```

Today: `qwen3-omni` (covers all 6 benchmarks/eval/*.py — Qwen3-Omni
rows + S2-Pro TTS rows live in the same config; per-row `hf_model_id`
distinguishes the model launched per row).

## Prerequisites (the skill verifies, it does not create)

- A sglang-omni clone (`benchmarks/eval/` reachable from the working dir).
- A venv with `sglang` + `torch` importable. Default candidates (first existing wins):
  `/sgl-workspace/sglang-omni/omni-qwen3/bin/python`,
  `/github/home/calibration/qwen3/omni-qwen3/bin/python`,
  `/github/home/omni-qwen3/bin/python` (legacy). Override with
  `--venv-python <path>` or `$EVAL_VENV_PYTHON`.
- HF model + datasets cached under `HF_HOME=/github/home/.cache/huggingface`
  (see `auto_env` in `config.yaml`). precheck reports `✓` / `✗` and prints
  the exact `huggingface-cli download …` commands when something's missing.
- Free GPUs. The skill **never** kills another user's processes; if
  GPUs are busy precheck fails with the busy PID list and stops.
- `auto_env` from `config.yaml` is applied at startup (overrides shell) to
  match CI omni-setup: `OMNI_CI_HOME`, `UV_INDEX_URL`, torchinductor slice
  paths, etc. See `tune-ci-thresholds` skill for the full cache layout table.
- First-time venv on the repro host: use `.github/scripts/prepare_omni_venv.sh`
  and `.github/scripts/install_flashinfer_jit_cache.sh` with
  `OMNI_CI_HOME=/github/home/calibration/qwen3` (documented in tune-ci-thresholds).

If anything's off, `precheck` fails with an actionable message; fix it
yourself and retry.

## Invocation

- `/running-eval-suite` — runs everything end-to-end with defaults
  (model `qwen3-omni`, all benchmarks, 1 round, real apply,
  auto-commit).
- `/running-eval-suite --benchmarks mmsu,omni_seedtts` — run only those
  benchmark files. Use `omni_seedtts` / `qwen_seedtts` for Qwen3-Omni
  SeedTTS rows and `tts_seedtts` / `s2pro_seedtts` for S2-Pro rows.
- `/running-eval-suite --rounds 3` — multi-round; only round 1 is used
  for apply (multi-round aggregation is a follow-up).
- `/running-eval-suite --smoke 50` — `--max-samples 50` injected; no
  edits applied. For benchmarks whose full set is already ≤50, the
  smoke flag is effectively a no-op (they run their full set anyway).
- `/running-eval-suite --venv-python /usr/bin/python3` — override the
  default venv detection (also via `$EVAL_VENV_PYTHON`).
- `/running-eval-suite --exclude-ids s2pro-` — skip rows whose id
  contains a substring.
- `/running-eval-suite --retry-failed-from <run-dir>` — rerun rows that
  were not successfully edited in a previous run.
- `/running-eval-suite --gpu-pool 0,1,2` — restrict server and ASR GPU
  allocation to a specific free GPU pool.

I never call `AskUserQuestion`. Defaults handle the full run; optional
flags above are only for filtering, recovery, or explicit GPU placement.
Errors halt the run with a printed reason. This matches the
`tune-ci-thresholds` contract: type `/running-eval-suite` and walk away.

## Steps I follow

1. Apply defaults to the invocation: `model=qwen3-omni`,
   `benchmarks=all`, `rounds=1`, `smoke=null`. **No `AskUserQuestion`.**
   If the user passed unknown flags, stop with the parsing error.

2. Run precheck. The runner creates `<repo>/.eval-runs/<utc-ts>/` on
   first run; reuse the same `<run-dir>` for the rest of the
   invocation:
   ```
   python .claude/skills/running-eval-suite/runner.py \
     --model <M> precheck --benchmarks <names|all> \
     [--exclude-ids <substr>] [--gpu-pool <ids>] \
     --output-dir <run-dir>
   ```
   If exit code != 0, surface the `✗` lines and stop.

3. Run the benchmarks. The runner does the cell rewrites inline as
   each row finishes (no separate apply step):
   ```
   python .claude/skills/running-eval-suite/runner.py \
     --model <M> run --benchmarks <names|all> \
     --rounds <K> [--smoke <N>] \
     [--exclude-ids <substr>] [--gpu-pool <ids>] \
     --output-dir <run-dir>
   ```
   - For each row: launch server (process-group SIGTERM at end, never
     `pkill -f`), run client, capture `result.json`, then immediately
     edit the matching row in `benchmarks/eval/benchmark_*.py`
     (replace existing hw row OR append new hw row at end of the
     section's table; Local sections never touched).
   - Managed-router rows use `router, 2-worker, ...` Source workload
     tags. The first run on a hardware target appends new router-tagged
     rows next to the older direct/V1-pipeline rows; later runs replace
     those router rows in place.
   - Per-row state lands in `<run-dir>/run-state.json`. Stream the
     runner's stdout to the user so they can watch progress.
   - **If a row fails** (server boot timeout / client crash / locator
     mismatch / column missing / JSON path missing / etc.), that row's
     reason is recorded; **the next row continues**. No global halt.

4. After the run finishes, look at `run-state.json`'s summary block
   (also printed to stdout): replaced count + appended count + failed
   runs + failed edits. If anything failed, surface those lines.

5. Auto-commit the refresh:
   ```
   git -C <repo> add benchmarks/eval/
   git -C <repo> diff --cached --quiet && (echo "no changes — nothing to commit"; exit 0)
   git -C <repo> commit -m "[Docs] Refresh reference benchmarks via /running-eval-suite (<utc-ts>)"
   ```
   No `git push`. If `git diff --cached --quiet` says there's nothing
   to commit (every row failed or was skipped), skip the commit and
   print a one-liner.

6. Print three lines for the user:
   - `Review:   git show HEAD`
   - `Revert:   git reset --soft HEAD~1   # keeps the edits unstaged`
   - `Run dir:  <run-dir>` (logs / result.json per row)

## What I do not do

- **Ask the user any questions.** "Type the slash command and walk
  away" is the contract.
- Set up the container / venv / HF cache (covered by
  `docs/contributing/running-eval-suite.md`).
- Touch `Local v1 Pipeline Result` / `Local Speech Pipeline` tables
  inside `benchmark_*.py` — those are contributors' personal
  experiments.
- Kill other users' processes. Busy GPUs → precheck reports busy PID
  list → stop.
- Push. I commit locally so the user can `git show HEAD`,
  rebase/squash/amend, then push to their fork.
- Multi-round aggregation. `--rounds K` runs K rounds (useful for
  variance observation), but inline apply uses round 1 only. Cells
  whose existing Source tag references `n=K mean` are flagged in the
  preview but still updated; user can manually recompute the mean
  before pushing.

## Files

```
.claude/skills/running-eval-suite/
├── SKILL.md
├── runner.py                                # CLI: precheck / run
└── models/
    └── qwen3-omni/
        └── config.yaml                      # 31 rows across all 6 benchmark .py files
```

## Adding a new row

1. Identify the reference table row in some
   `benchmarks/eval/benchmark_*.py`. Note the **section header**
   above the table, the row's **Config** column substring, and the
   **workload tag** inside the row's Source column (the part inside
   the `[…]` brackets, **without** the leading hardware token —
   e.g. `router, 2-worker, full-set, c=8` from
   `[H200, router, 2-worker, full-set, c=8]`).
2. Append a new entry to `models/<model>/config.yaml`:
   ```yaml
   - id: <unique-handle>          # informational; not used to select rows
     file: "benchmarks/eval/benchmark_omni_<name>.py"
     hf_model_id: "Qwen/Qwen3-Omni-30B-A3B-Instruct"   # optional, overrides config-level
     locate:
       section_substring: "Accuracy (accuracy)"
       config_substring: "modalities=text "
       source_workload: "router, 2-worker, full-set, c=8"
     # Server launch is normally inherited from default_server_profile or
     # server_profile_by_hf_model_id. Set server_profile only when a row needs
     # a different managed-router topology.
     server_profile: qwen3_omni_colocated_router
     client: "python benchmarks/eval/benchmark_omni_<name>.py --model qwen3-omni --port {port} --output-dir {output_dir} ..."
     result_json: "<benchmark>_results.json"
     cells:
       <col1>: { path: "<dotted.path>", format: "{v}" }
       <col2>: { paths: { v: "<a>", t: "<b>" }, format: "{v}/{t}" }
   ```
3. Verify the locator matches the intended H200 row (or doesn't yet
   exist for new hw → will append) before committing the config:
   ```bash
   python - <<'EOF'
   import sys; sys.path.insert(0, '.claude/skills/running-eval-suite')
   import runner, yaml; from pathlib import Path
   cfg = yaml.safe_load(open('.claude/skills/running-eval-suite/models/qwen3-omni/config.yaml'))
   for row in cfg['rows']:
       text = Path(row['file']).read_text()
       rows = list(runner._find_table_rows_official(text))
       loc = row['locate']
       full = f"[H200, {loc['source_workload']}]"
       cands = [r for r in rows if loc['section_substring'] in (r.get('section_line') or '')
                and loc['config_substring'] in r['row_text']]
       matched = [r for r in cands if full in r['row_text']]
       print(f"{row['id']}: section_rows={len(cands)} h200_match={len(matched)}")
   EOF
   ```
4. Smoke-test with `--smoke 50 --benchmarks <name>` before
   committing.

Adding a whole new model = drop in `models/<new-model>/config.yaml`
mirroring `qwen3-omni/config.yaml`. No Python changes needed unless
the new benchmark client emits result JSON in a structure that needs
new helpers in `runner.py`.
