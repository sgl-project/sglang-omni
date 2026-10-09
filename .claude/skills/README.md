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
| [`running-eval-suite`](running-eval-suite/SKILL.md) | Reruns every reference benchmark under `benchmarks/eval/` and rewrites the reference-table cells in `benchmark_*.py` for the hardware it detects. Commits locally, never pushes. | Any sglang-omni dev container with free GPUs and the `omni` venv. |

The evaluation skill expects the CI-equivalent environment (the `omni` venv,
`HF_HOME` populated, `source .github/scripts/ci_env.sh`). Its precheck
verifies this and stops with an actionable message rather than fixing it for you.
The skill never kills another user's processes: busy GPUs are a hard
stop.

CI threshold calibration is maintained separately in the private
[`sglang-omni-calibration` repository](https://github.com/zhaochenyang20/sglang-omni-calibration/tree/main/skills/calibrate-h100-ci).
Maintainers with access can find `calibrate-h100-ci`, its tests, and calibration
instructions there.

## Running one

Type the slash command in Claude Code from the repo root:

```
/running-eval-suite --benchmarks mmsu
```

Read the skill's `SKILL.md` first and keep a supervision terminal open
alongside the job.

You can also drive the underlying tools directly, without an agent:

```bash
python .claude/skills/running-eval-suite/runner.py --model qwen3-omni precheck --output-dir "$RUN"
```

Run artifacts land in `.eval-runs/`, which is gitignored.

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
name: running-eval-suite
description: Run the reference benchmarks under benchmarks/eval/ and refresh their reference-table cells for the detected hardware.
---
```
