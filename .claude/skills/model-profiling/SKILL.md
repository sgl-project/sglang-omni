---
name: model-profiling
description: Generate a bounded task plan (and the agent prompt for it) that runs the 5-layer profiling methodology in .claude/skills/model-profiling/METHODOLOGY.md against one model, stops for human confirmation before any GPU work starts, then delegates the actual run to a background agent, verifies its completion claim, and routes findings into a gitignored .profiling-runs/<model>/ directory — durably tracked as a sub-issue under the tracking issue named in SKILL.md's "Result tracking" section — plus the methodology doc.
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

The methodology doc gets corrected as new models are profiled. Always read
the current version at invocation time — never assume the section numbers
or findings above are still accurate, and never copy its prose into this
skill or into the generated prompt; reference section names instead.

## Result tracking

Single source of truth for where results go — every "tracking issue"
mention elsewhere in this skill and in `PROMPT_TEMPLATE.md` refers here,
so if the tracking issue ever moves, update only this section.

- **Local working artifacts**: `.profiling-runs/<model>/` (gitignored) —
  `profile.md` plus every raw artifact its numbers cite (py-spy outputs,
  benchmark logs, nvidia-smi captures). Disposable by design: a fresh
  clone won't have it, and nothing under `.profiling-runs/` is ever
  committed. The only committable output of a run is a METHODOLOGY.md
  edit.
- **Durable record**: a sub-issue under GitHub issue **#1798** — the only
  place results survive an environment reset. A run is not durably
  recorded until its sub-issue exists. #1798 is the **tracking index
  only**: METHODOLOGY.md stays canonical for the methodology itself, and
  any methodology copy in the issue body is a historical snapshot.
- **Sub-issue summary shape** — self-contained enough that the main
  conclusions can be re-checked without access to the original host,
  because the local artifacts will not survive an environment reset:
  baseline fingerprint (including the repo commit SHA), per-layer
  conclusions with evidence strength graded by METHODOLOGY.md §3 item 8's
  rubric, the findings-evidence-recommendation table **inlined in full**
  along with each layer's headline numbers, and small raw artifacts
  (py-spy summaries, benchmark result JSON/logs up to a few MB) **attached
  to the sub-issue** rather than merely referenced. Only bulky artifacts
  (e.g. full server logs) stay local-only — name the host/directory they
  lived in so a reader knows what existed, but never let a conclusion rest
  solely on an artifact that only exists there.

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
- Check whether `.profiling-runs/<model>/profile.md` already exists — if it
  does, this is a resume/extend run, not a fresh one; read it first so the
  plan only covers the missing layers, and compare its recorded baseline
  fingerprint (commit SHA, GPU model, dependency pins) against the current
  environment — drift means the prior layers' evidence may be stale, which
  changes what "missing" means (the agent prompt re-checks this too). The
  path is gitignored, so a fresh clone won't have it even if the model was
  profiled before — in that case check the sub-issues under the tracking
  issue (see Result tracking) before scoping the plan as fresh.

## Invocation

- `/model-profiling <model>` — resolve the model, draft the plan and the
  filled-in agent prompt, print both, and stop. No GPU work happens yet.
- `/model-profiling <model> --resume` — same, but explicitly scoped to only
  the layers missing from the existing `.profiling-runs/<model>/profile.md`.

Unlike `running-eval-suite`, this skill **does** pause for a human decision
before the GPU-touching step — that pause is the point, not an oversight.
A fresh run (no prior Layer 2 evidence) pauses **twice**: once before any
GPU work starts, and again after Layer 2 surfaces a candidate Layer 4
hypothesis, since no one — human or agent — can pick an A/B variable before
that evidence exists.

## Steps I follow

1. Locate the model's benchmark entrypoint and any existing
   `.profiling-runs/<model>/profile.md` (and, if that's absent, any prior
   sub-issue under the tracking issue — see Result tracking). Decide fresh
   vs. resume.
2. Run the cheap parts of the
   `.claude/skills/model-profiling/METHODOLOGY.md` §1 checklist
   myself (`nvidia-smi` free-GPU check, `py-spy --version`); leave the
   expensive/stateful checks (baseline env fingerprint, orphan cleanup) for
   the executing agent to do and record.
3. Check whether a concrete Layer 4 hypothesis already exists — from an
   existing `.profiling-runs/<model>/profile.md`'s Layer 2 findings, or
   from something
   already established earlier in this conversation. If not, this run
   cannot pick a Layer 4 A/B variable yet: scope the plan to **discovery
   only** (Layer 1 first, then branch per
   `.claude/skills/model-profiling/METHODOLOGY.md` §2's routing rule — see
   Method reference above) and leave Layer 4/5 for a second confirmation
   later.
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
   step 5 (e.g. a resume run). If the agent took the saturation branch
   (GPU-kernel-bound, Layer 2 skipped), there is no hypothesis to confirm —
   the pause still happens, but what's presented is the saturation evidence
   and a choice: end the run there, or scope kernel-level analysis as a
   new plan. Don't let the absence of a hypothesis silently skip the pause.
9. Once genuinely done — discovery, and the approved experiment if one ran
   — independently verify before reporting, don't just relay the agent's
   self-summary: check the new/updated `.profiling-runs/<model>/profile.md`
   has a section per in-scope layer (`grep "^## "`), that the
   findings-evidence-recommendation table actually exists (headers alone
   don't prove content), that the raw artifacts the doc cites are actually
   present in `.profiling-runs/<model>/`, and — if the methodology doc was
   touched — `git status --short .claude/`.
10. Route findings:
    - Model-specific numbers and tables → new or updated
      `.profiling-runs/<model>/profile.md`, with the report shape
      METHODOLOGY.md §3 item 8 defines (baseline env including commit SHA,
      one section per layer entered with method/findings/rubric-graded
      evidence strength, a findings-evidence-recommendation table,
      cleanup), raw artifacts alongside it in the same directory. This
      stays out of the repo — see Result tracking.
    - Cross-model, generalizable lessons (sampling bias, outlier fragility,
      tooling gotchas) → `.claude/skills/model-profiling/METHODOLOGY.md`.
    - Anything ambiguous between the two → ask the user, don't decide alone.
11. **Do not auto-commit.** Nothing under `.profiling-runs/` is ever
    committed; the only committable output is a METHODOLOGY.md edit —
    print `git status --short .claude/` and let the user review and commit
    that themselves, since it carries judgment calls a human should sign
    off on.
12. **Close the loop on the durable record.** End the final report with a
    ready-to-post sub-issue summary (shape per Result tracking) and the
    explicit status line "tracking sub-issue not yet filed" — the run is
    not durably recorded until the user posts it (or explicitly declines,
    in which case note where the local artifacts live and that they won't
    survive an environment reset). Do not post the sub-issue yourself
    without the user's confirmation.

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
- Treat a run as durably recorded while only the gitignored local artifacts
  exist — the record isn't closed until the tracking sub-issue is filed or
  the user explicitly declines it (step 12).

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
3. The report shape comes from METHODOLOGY.md §3 item 8, not from a prior
   model's profile doc — those are gitignored working artifacts and may not
   exist in a given clone, so never point the agent prompt at one as an
   example.
