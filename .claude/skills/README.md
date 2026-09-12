# Agent skills

Skills are checked-in playbooks for long, error-prone maintenance jobs that we
would otherwise re-explain to a coding agent every time. Each subdirectory here
is one skill: a `SKILL.md` that tells the agent how to run the job, plus the
scripts and config the job needs.

Claude Code picks skills up automatically from `.claude/skills/` and exposes
each one as a slash command named after its directory. Nothing else in
`.claude/` is shared — `.gitignore` keeps `.claude/*` out of the repo and
re-includes only `.claude/skills/`, so local agent settings stay local.

These are maintainer tools, not part of the runtime. Nothing in this directory
is imported by `sglang_omni/`, and no CI job runs them.

## Available skills

| Skill | What it does | Who can run it |
|---|---|---|
| [`tune-ci-thresholds`](tune-ci-thresholds/SKILL.md) | Recalibrates the numeric CI gates for ASR, TTS and Qwen3-Omni: repeats each stage until it has enough clean observations, rejects rounds poisoned by host contention, and emits an apply plan. | Maintainers on the shared H100 CI host. Needs a host profile in [`hosts/`](tune-ci-thresholds/hosts/); today only `sglang-h100-ci` exists. |
| [`running-eval-suite`](running-eval-suite/SKILL.md) | Reruns every reference benchmark under `benchmarks/eval/` and rewrites the reference-table cells in `benchmark_*.py` for the hardware it detects. Commits locally, never pushes. | Any sglang-omni dev container with free GPUs and the `omni` venv. |

Both skills expect the CI-equivalent environment (the `omni` venv,
`HF_HOME` populated, `source .github/scripts/ci_env.sh`). Their prechecks
verify this and stop with an actionable message rather than fixing it for you.
Neither skill will ever kill another user's processes: busy GPUs are a hard
stop.

## Running one

Type the slash command in Claude Code from the repo root:

```
/tune-ci-thresholds
/running-eval-suite --benchmarks mmsu
```

Read the skill's `SKILL.md` first — both want a supervision terminal open
alongside the job, and `tune-ci-thresholds` additionally wants you to read its
`CONTRACT.md`, `AGENT-PRECHECK.md` and `OPERATIONS.md` before a calibration.

You can also drive the underlying tools directly, without an agent:

```bash
python .claude/skills/tune-ci-thresholds/tune.py --model omni precheck --output-dir "$RUN"
python .claude/skills/running-eval-suite/runner.py --model qwen3-omni precheck --output-dir "$RUN"
```

Run artifacts land in `.tune-runs/` and `.eval-runs/`, both gitignored.

## Adding a skill

Layout:

```
.claude/skills/<skill-name>/
├── SKILL.md          # required: frontmatter + the playbook
├── <tool>.py         # the actual work, runnable without an agent
└── models/ hosts/    # config, one file per model or host
```

`SKILL.md` frontmatter needs `name` (matching the directory) and
`description`. The description is the only thing an agent sees when deciding
whether to reach for the skill, so lead with the trigger, then the mechanism:

```yaml
---
name: tune-ci-thresholds
description: Use when a CI threshold needs recalibrating after a model, kernel or host change — recalibrates ASR/TTS/Qwen3-Omni gates from repeated clean observations and emits an apply plan for review.
---
```
