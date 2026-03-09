# Plan: Remove shuai-talker Pollution from s2-integration-Jing

## Problem Summary

Branch `s2-integration-Jing` merged `shuai-talker` (commit `d7c395e`) anticipating
that shuai-talker would land on main first. That merge is now cancelled, leaving
44 talker-related file changes polluting the branch. A simple `git revert` of the
merge commit fails due to heavy conflicts with subsequent commits.

## Branch Archaeology

```
main (91a69eb)
  └── e591da0 (merge-base)
        ├── [S2-pro commits: a270bc3 .. 6b3ce68]   ← YOUR work (11 commits)
        ├── d7c395e  Merge shuai-talker             ← THE PROBLEM
        ├── e6bd3c5  code cleanup, playground        ← mixed (S2-pro + talker refs)
        ├── 8d0d635..676d971  (remote branch)        ← cleanup commits (6 commits)
        ├── 98911a4  clear s1, code formatting       ← S2-pro cleanup
        └── 36b0b27  Merge remote (HEAD)             ← tip
```

## File Impact Analysis

| Category | Count | Action |
|----------|-------|--------|
| **S2-pro only** files (fishaudio_s2_pro/*, playground/*, benchmarks/*, serve/*, etc.) | 56 | Keep as-is from HEAD |
| **Talker-only** shared files (engines/omni/*, qwen3_omni/*, pipeline/*, executors/*, etc.) | 42 | Revert to main's version |
| **Both S2-pro + talker** shared files | 2 | Apply only S2-pro diff |

The 2 shared files needing manual merge are:
- `sglang_omni/client/client.py` — S2-pro added `sample_rate` propagation (small diff)
- `sglang_omni/config/schema.py` — S2-pro changed `relay_backend` default from `"nixl"` to `"shm"` (1-line diff)

## Recommended Strategy: New Branch from Main + Apply S2-pro Diff

This is the safest approach. It produces a clean branch with no talker history.

### Step 1: Create a clean branch from main

```bash
git checkout main
git pull origin main
git checkout -b s2-integration-Jing-clean
```

### Step 2: Copy the 56 S2-pro-only files from HEAD

These files have zero overlap with shuai-talker and can be copied directly:

```bash
git checkout s2-integration-Jing -- \
  benchmarks/eval_wer.py \
  benchmarks/profile_s2pro_sglang.py \
  docs/developer_reference/apiserver_design.md \
  docs/get_started/apiserver_quickstart.md \
  examples/configs/s2pro_tts.yaml \
  .github/workflows/publish-docs.yaml \
  playground/gradio/start.sh \
  playground/tts/app.py \
  playground/tts/start.sh \
  pyproject.toml \
  sglang_omni/client/types.py \
  sglang_omni/engines/omni/runtime/tokenizer.py \
  sglang_omni/models/fishaudio_s2_pro/config.py \
  sglang_omni/models/fishaudio_s2_pro/factory.py \
  sglang_omni/models/fishaudio_s2_pro/__init__.py \
  sglang_omni/models/fishaudio_s2_pro/io.py \
  sglang_omni/models/fishaudio_s2_pro/pipeline/engine_io.py \
  sglang_omni/models/fishaudio_s2_pro/pipeline/__init__.py \
  sglang_omni/models/fishaudio_s2_pro/pipeline/next_stage.py \
  sglang_omni/models/fishaudio_s2_pro/pipeline/stages.py \
  sglang_omni/models/fishaudio_s2_pro/pipeline/state_io.py \
  sglang_omni/models/fishaudio_s2_pro/README.md \
  sglang_omni/models/fishaudio_s2_pro/runtime/__init__.py \
  sglang_omni/models/fishaudio_s2_pro/runtime/s2pro_ar.py \
  sglang_omni/models/fishaudio_s2_pro/runtime/s2pro_sglang_ar.py \
  sglang_omni/models/fishaudio_s2_pro/sglang_model.py \
  sglang_omni/models/fishaudio_s2_pro_sglang.py \
  sglang_omni/models/fishaudio_s2_pro/tokenizer.py \
  sglang_omni/models/registry.py \
  sglang_omni/serve/launcher.py \
  sglang_omni/serve/openai_api.py \
  sglang_omni/serve/protocol.py
```

And the vendored fish_speech directory (all new files):

```bash
git checkout s2-integration-Jing -- \
  sglang_omni/models/fishaudio_s2_pro/fish_speech/
```

### Step 3: Apply the 2 small S2-pro diffs to shared files

**client.py** — Extract S2-pro-only changes from main → pre-merge diff:
```bash
git diff main 6b3ce68 -- sglang_omni/client/client.py | git apply
```

**schema.py** — Extract S2-pro-only changes from main → pre-merge diff:
```bash
git diff main 6b3ce68 -- sglang_omni/config/schema.py | git apply
```

### Step 4: Skip the 42 talker-only shared files

These remain at main's version — no action needed. They include:
- `sglang_omni/config/compiler.py`, `mp_runner.py`
- `sglang_omni/engines/ar/sglang_backend/*`
- `sglang_omni/engines/omni/*` (engine.py, factory.py, sglang_ar.py, etc.)
- `sglang_omni/executors/*`
- `sglang_omni/models/qwen3_omni/*`
- `sglang_omni/pipeline/*`
- `sglang_omni/proto/*`
- `docs/developer_reference/talker_decode_parity.md`, `docs/index.rst`
- `sglang_omni/vendor/sglang/server_args.py`

### Step 5: Verify

```bash
# Should show only S2-pro-related files
git diff --stat main

# Quick sanity: no qwen3_omni, no talker, no executors, no chunk/mailbox
git diff --name-only main | grep -E "(qwen3_omni|talker|executors|chunk/mailbox)"
# Expected: empty output

# Verify imports work
python -c "from sglang_omni.models.fishaudio_s2_pro import factory; print('OK')"
```

### Step 6: Commit and replace old branch

```bash
git add -A
git commit -m "feat(s2pro): S2-pro integration (clean, without shuai-talker)"

# Archive the polluted branch, then replace
git branch s2-integration-Jing-archived s2-integration-Jing
git branch -D s2-integration-Jing
git branch -m s2-integration-Jing-clean s2-integration-Jing
git push origin s2-integration-Jing --force-with-lease
```

## Why Not Other Approaches?

| Approach | Problem |
|----------|---------|
| `git revert -m1 d7c395e` | Heavy conflicts — post-merge commits modified the same files |
| `git rebase -i` dropping the merge | 3+ merge commits in history make this extremely error-prone |
| `git rebase --onto main` | Same issue — merge commits cause cascading conflicts |
| Manual file-by-file revert | 42 files to revert, many with interleaved changes from lint fixes |

The "new branch + selective copy" approach avoids all conflict resolution entirely.
The only manual work is applying 2 small, clean diffs (total ~25 lines changed).

## Risk Assessment

- **Low risk**: 56 S2-pro-only files are copied verbatim — no data loss possible
- **Low risk**: 42 talker-only files are simply left at main — no data loss
- **Minimal manual work**: Only 2 files need a targeted diff application
- **Reversible**: Old branch is archived as `s2-integration-Jing-archived`
- **Verify before force-push**: Run imports and basic tests first
