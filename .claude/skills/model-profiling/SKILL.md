---
name: model-profiling
description: Generate a bounded task plan (and the agent prompt for it) that runs the 5-layer profiling methodology in .claude/skills/model-profiling/METHODOLOGY.md against one model, stops for human confirmation before any GPU work starts, then delegates the actual run to a background agent, verifies its completion claim, and folds findings into docs/developer_reference/<model>_profile.md plus the methodology doc.
---

# model-profiling

## Scope

A contributor runs `/model-profiling <model>` to get a ready-to-review plan
for profiling one model with the layered methodology already validated on
Whisper-ASR, Qwen3-ASR, and moss_transcribe_diarize:

1. Resolve `<model>` to its benchmark entrypoint and existing docs.
2. Fill `PROMPT_TEMPLATE.md` into a bounded agent prompt for this model.
3. Print the plan and the prompt, then **stop and wait for confirmation**.
4. Only after the human confirms (or edits) the plan: launch a background
   agent via the `Agent` tool to actually run the GPU work.
5. When the agent reports back, verify its completion claim before trusting
   it, then route findings into the right docs.

This is **not** a fully autonomous runner like `running-eval-suite`. Layer 2
(which leaf frame to chase) and Layer 4 (which variable to A/B) require
judgment that this skill cannot make on its own — its job stops at producing
a plan a human can approve or redirect, matching how this methodology has
actually been executed by hand on all three models so far.

## Method reference

The 5-layer methodology itself lives in
`.claude/skills/model-profiling/METHODOLOGY.md` and is **not duplicated
here**:

- `§1` — environment / pre-check checklist (GPU selection, baseline env
  fingerprint, orphan process cleanup, CPU isolation).
- `§2` — Layer 1 (GPU busy ratio) through Layer 5 (functional regression).
- `§3` — checklist for the next executor / expected output shape.
- `§4` — tool and script index (py-spy, CPU pinning, nvidia-smi, DCGM, nsys).

This is currently English-only — no Chinese translation is tracked yet.
`docs/profiling_methodology_zh.md` exists locally as an untranslated draft
and is not part of this skill; if it's ever added to the repo, treat it as
a second document to keep in sync, not a replacement for the path above.

The methodology doc gets corrected as new models are profiled. Always read
the current version at invocation time — never assume the section numbers
or findings above are still accurate, and never copy its prose into this
skill or into the generated prompt; reference section names instead.

## Prerequisites (the skill verifies, it does not create)

- A sglang-omni clone with the target model's benchmark entrypoint reachable
  — check `benchmarks/eval/benchmark_*.py` first, then the model's own
  `sglang_omni/models/<model>/` or `benchmarks/` subtree.
- Free GPUs. Same rule as the other two skills: **never** kill another
  user's processes. If a GPU looks busy, report the PID list and stop — and
  do not treat an unattributable PID (`nvidia-smi` shows usage but `ps`
  shows nothing) as proof no one else is using the card; shared sandboxes
  can hide cross-tenant PIDs.
- `py-spy` installed and permitted to attach to the target process (Layer 2).
- Check whether `docs/developer_reference/<model>_profile.md` already
  exists — if it does, this is a resume/extend run, not a fresh one; read
  it first so the plan only covers the missing layers.

## Invocation

- `/model-profiling <model>` — resolve the model, draft the plan and the
  filled-in agent prompt, print both, and stop. No GPU work happens yet.
- `/model-profiling <model> --resume` — same, but explicitly scoped to only
  the layers missing from the existing `<model>_profile.md`.

Unlike `running-eval-suite`, this skill **does** pause for a human decision
before the GPU-touching step — that pause is the point, not an oversight.
A fresh run (no prior Layer 2 evidence) pauses **twice**: once before any
GPU work starts, and again after Layer 2 surfaces a candidate Layer 4
hypothesis, since no one — human or agent — can pick an A/B variable before
that evidence exists.

## Steps I follow

1. Locate the model's benchmark entrypoint and any existing
   `docs/developer_reference/<model>_profile.md`. Decide fresh vs. resume.
2. Run the cheap parts of the
   `.claude/skills/model-profiling/METHODOLOGY.md` §1 checklist
   myself (`nvidia-smi` free-GPU check, `py-spy --version`); leave the
   expensive/stateful checks (baseline env fingerprint, orphan cleanup) for
   the executing agent to do and record.
3. Check whether a concrete Layer 4 hypothesis already exists — from an
   existing `<model>_profile.md`'s Layer 2 findings, or from something
   already established earlier in this conversation. If not, this run
   cannot pick a Layer 4 A/B variable yet: scope the plan to **discovery
   only** (Layer 1, then 3 and/or 2 per
   `.claude/skills/model-profiling/METHODOLOGY.md` §2's routing rule, not a
   fixed order — see Method reference above) and
   leave Layer 4/5 for a second confirmation later.
4. Fill `PROMPT_TEMPLATE.md`'s placeholders for this model — discovery-only
   scope, or the full scope if a hypothesis is already confirmed — and
   print the resulting prompt plus a short plan summary.
5. **Stop.** Wait for the user to confirm, edit, or reject the plan.
6. On confirmation, launch a background agent (via the `Agent` tool, a
   fresh general-purpose agent — not a context-inheriting fork, since this
   skill must work the same way with or without prior conversation) with
   the confirmed prompt.
7. When a completion notification arrives, do not trust the `status`
   label alone — read the agent's own `result` text. If it describes
   unfinished work despite `status: completed`, resume it with
   `SendMessage` to the same agent and explicit continuation instructions,
   rather than assuming it will pick back up on its own.
8. **Second confirmation, discovery-only runs**: once Layer 2 (or the
   saturation branch that skips it) genuinely finishes, stop again before
   any Layer 4 work happens. Present the agent's candidate hypothesis to
   the user and wait for confirmation of which one, if any, to A/B — a
   fresh run has no basis to pick a Layer 4 variable until this evidence
   exists, so the agent is instructed to stop here too rather than deciding
   on its own. Only after confirmation, resume the same agent
   (`SendMessage`) with the approved Layer 4 (and Layer 5, if applicable)
   scope. Skip this step when the hypothesis was already confirmed before
   step 5 (e.g. a resume run).
9. Once genuinely done — discovery, and the approved experiment if one ran
   — independently verify before reporting: check the new/updated doc's
   section headers (`grep "^## "`) and `git status --short docs/` — don't
   just relay the agent's self-summary.
10. Route findings:
    - Model-specific numbers and tables → new or updated
      `docs/developer_reference/<model>_profile.md`, following the section
      shape of the most complete prior example (currently
      `moss_transcribe_diarize_profile.md`: baseline env, Layer 1/3, Layer 2,
      Layer 4, Layer 5, a findings-evidence-recommendation table, cleanup).
    - Cross-model, generalizable lessons (sampling bias, outlier fragility,
      tooling gotchas) → `.claude/skills/model-profiling/METHODOLOGY.md`.
    - Anything ambiguous between the two → ask the user, don't decide alone.
11. **Do not auto-commit.** Print `git status --short docs/` and let the
    user review and commit themselves — unlike `running-eval-suite`'s
    auto-commit, these docs carry judgment calls a human should sign off on.

## What I do not do

- Start GPU work without an explicit human confirmation of the plan.
- Duplicate the methodology's prose — reference
  `.claude/skills/model-profiling/METHODOLOGY.md` section names only.
- Auto-commit or push.
- Kill another user's processes, or assume an unattributable PID means no
  one else is using a GPU.
- Forward unverified, suspiciously specific technical claims (exact PIDs,
  numbers, timestamps) into an agent prompt or a report without
  independently checking them first — if they can't be verified, say so
  explicitly and ask the user where they came from instead of relaying or
  silently dropping them.
- Decide ambiguous output-routing calls (methodology doc vs. per-model doc)
  on my own — ask when it's not clear-cut.
- Let a discovery-only run proceed into Layer 4 without a second, explicit
  confirmation of the concrete hypothesis Layer 2 turned up.

## Files

```
.claude/skills/model-profiling/
├── SKILL.md
├── METHODOLOGY.md                   # the 5-layer methodology itself (English only)
└── PROMPT_TEMPLATE.md               # placeholder agent prompt filled in per model
```

## Adding a new model

1. Confirm a benchmark entrypoint exists (or note in the plan that one must
   be written first — that's out of scope for this skill).
2. No config file to add: this skill fills `PROMPT_TEMPLATE.md` fresh per
   invocation instead of reading a per-model registry, since Layer 2/4
   choices are model-specific judgment calls anyway.
3. After a model's first full run, its `docs/developer_reference/<model>_profile.md`
   becomes the new best example for Step 10's section shape if it's more
   complete than the current reference example.
