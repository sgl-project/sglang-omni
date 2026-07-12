# SGLang-Omni Global Stage Scheduler

This package implements the global CI queue proposed in
[issue #551](https://github.com/sgl-project/sglang-omni/issues/551). It compiles
the existing GitHub Actions YAML into a dependency graph and generates one
`workflow_dispatch` workflow per executable job. Only self-hosted jobs consume
the configured runner capacity.

## Scheduling Contract

- An open, non-draft PR must have `run-ci` to enter the queue.
- Pending jobs from a PR with `high-priority` rank before normal jobs.
- Dependencies and stable job order come from workflow YAML, not a separate
stage registry.
- A job is eligible only after all of its dependencies are terminal.
- GitHub-hosted jobs dispatch when eligible and do not consume GPU capacity.
- Self-hosted jobs are selected globally. The default capacity is one.
- Pending jobs are re-ranked from current labels on every selection.
- A new PR SHA makes pending, reserved, and dispatched work for older SHAs stale.

Selection order is:

1. Priority type (run-ci, high-priority).
2. Enqueue time.
3. PR number.
4. YAML declaration order.
5. Stable stage key.
6. Queue item ID.



## Components


| Component      | Responsibility                                                                                              |
| -------------- | ----------------------------------------------------------------------------------------------------------- |
| `compiler/`    | Parse YAML, flatten reusable calls, validate the DAG, rewrite cross-run references, and generate workflows. |
| `domain/`      | Queue models, pure selection policy, and state transitions.                                                 |
| `adapters/`    | SQLite persistence and GitHub API access.                                                                   |
| `app/`         | Webhook handling, orchestration, dispatch, and check updates.                                               |
| GitHub Actions | Evaluate generated job conditions and execute each generated job.                                           |




## Why Generate One Workflow per Job?

Starting an original multi-job workflow lets GitHub queue every ready
self-hosted job before the global scheduler can reconsider priority. A later
high-priority PR cannot move ahead of those jobs.

The alternative is for the scheduler to reproduce GitHub's job evaluator,
including `needs`, `if`, contexts, outputs, reusable workflows, matrices, and
cancellation semantics. This implementation instead generates a workflow that
contains exactly one executable job. The scheduler chooses when to dispatch it;
GitHub still evaluates and executes the resulting job.

Generated files are deterministic build artifacts. Contributors edit the source
workflow YAML, regenerate the files, and do not edit generated files directly.
Unsupported syntax fails generation instead of being approximated at runtime.

## DAG Compilation

The compiler reads configured roots from `.github/workflows/` and preserves job
declaration order. It normalizes `needs`, resolves local reusable-workflow calls,
and detects missing dependencies and cycles.

For example, `omni-ci.yaml` calls `test-asr-ci.yaml`:

```text
omni-ci::preflight
  -> omni-ci::setup
  -> omni-ci::pr-test                         (reusable workflow aggregate)
  -> omni-ci::asr-ci                          (reusable workflow aggregate)
       -> ...::stage-1-multi-speaker
       -> ...::stage-2-seedtts
```

Reusable calls are virtual aggregate nodes. Caller dependencies are attached to
the called workflow's root jobs. Downstream jobs wait for the called workflow's
terminal aggregate result.

Runner classification is derived from `runs-on`:

- `ubuntu-latest` and supported hosted labels: `github-hosted`.
- Labels containing `self-hosted`: `self-hosted`.
- Dynamic or unknown runner expressions: rejected.



## Generated Workflows

The generator writes machine-owned files under
`.github/workflows/generated/`:

```text
zz_generated_scheduler__omni-ci__setup.yaml
zz_generated_scheduler__omni-ci__asr-ci_test-asr-ci__stage-1-multi-speaker.yaml
```

Each generated workflow:

- Accepts queue identity, PR SHA, attempt, source hash, and
`scheduler_context` through `workflow_dispatch` inputs.
- Contains one `run-stage` job.
- Copies the source job's runner, timeout, container, services, permissions,
environment, defaults, outputs, and steps.
- Checks out trusted orchestration and the immutable PR SHA separately.
- Publishes declared job outputs as a dispatch-bound artifact.

GitHub does not execute workflow files from a subdirectory. Before production
dispatch, generated workflows must be published as flat files directly under
`.github/workflows/`. New generated workflow paths become available after they
reach the default branch.

## Cross-Run Conditions

Splitting jobs into separate workflow runs removes GitHub's native `needs`
context. The compiler rewrites supported references to the scheduler-provided
context while leaving GitHub to evaluate the expression.


| Source reference                   | Generated reference                                                             |
| ---------------------------------- | ------------------------------------------------------------------------------- |
| `needs.setup.result`               | `fromJSON(inputs.scheduler_context).needs['omni-ci::setup'].result`             |
| `needs.preflight.outputs.labels`   | `fromJSON(inputs.scheduler_context).needs['omni-ci::preflight'].outputs.labels` |
| `github.event.pull_request.number` | `fromJSON(inputs.scheduler_context).source_event.number`                        |


Rewriting applies recursively to expression-bearing job fields, including
`if`, `env`, `with`, outputs, and step definitions. Runtime values such as
`github.run_id` remain native to the generated run.

## Scheduler Context and Outputs

Before dispatch, the scheduler builds JSON containing the source PR state and
the latest direct dependency states:

```json
{
  "source_event": {
    "event_name": "pull_request",
    "number": 100,
    "head_sha": "abc123",
    "labels": ["run-ci", "high-priority"]
  },
  "needs": {
    "omni-ci::preflight": {
      "result": "success",
      "outputs": {"labels": "[\"run-ci\"]"}
    }
  }
}
```

Generated workflows write declared outputs to `scheduler-outputs.json` and
upload it under the immutable dispatch ID. On successful completion, the
scheduler:

1. Finds the artifact on the bound workflow run.
2. Enforces compressed, uncompressed, entry-count, name, and value limits.
3. Validates schema version `1` and declared output names.
4. Stores outputs by repository, SHA, stage, and attempt.
5. Includes those outputs in downstream `scheduler_context`.

Artifact values are untrusted strings. They cannot select workflow paths,
runner labels, commands, secrets, or checkout revisions.

## Queue Lifecycle

```text
pending -> reserved -> dispatched -> running
pending/reserved/dispatched/running -> cancelled | stale
running -> passed | failed | timed_out
pending -> skipped
```

Dispatch flow:

1. A PR event creates one queue attempt per executable DAG node.
2. Dependency-ready hosted jobs are reserved and dispatched immediately.
3. SQLite `BEGIN IMMEDIATE` selects and reserves at most one self-hosted job
  when capacity is one.
4. The scheduler builds `scheduler_context` and dispatches the generated
  workflow.
5. `workflow_job.in_progress` marks the item running.
6. `workflow_run.completed` records outputs and terminal status, releases
  capacity, and starts the next scheduling turn.

`/rerun-failed-ci` support is disabled by default through
`SCHEDULER_RERUN_ENABLED=false`. Legacy command handling remains active until
cutover.

## Supported YAML Syntax


| Feature                                                                                              | Status              |
| ---------------------------------------------------------------------------------------------------- | ------------------- |
| Static `needs`                                                                                       | Supported           |
| One level of local reusable workflows                                                                | Supported           |
| Supported `needs.*`, PR-event, `always()`, `cancelled()`, `contains()`, and `fromJSON()` expressions | Rewritten or copied |
| Static GitHub-hosted and self-hosted `runs-on`                                                       | Supported           |
| `strategy.matrix`                                                                                    | Rejected            |
| Dynamic `runs-on`                                                                                    | Rejected            |
| Nested reusable-workflow calls                                                                       | Rejected            |




## Contributor Workflow

```bash
cd ci-scheduler
python tools/workflow_gen.py
python tools/workflow_gen.py --check
pytest tests/ -v
```

Review source and generated workflow changes together. `--check` fails for
missing, stale, or manually edited generated files.

## Local Service

```bash
cd ci-scheduler
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
```

Docker:

```bash
docker compose up --build
```

The GitHub App subscribes to `pull_request`, `issue_comment`, `workflow_run`,
and `workflow_job`. It requires Actions read/write, Checks read/write, Contents
read, Issues read, Pull requests read, and Metadata read permissions. Configure
the same `GITHUB_WEBHOOK_SECRET` in GitHub and the service environment.

## Cutover

1. Keep the existing PR workflows active while validating scheduler decisions.
2. Run generated dispatches behind a canary label or repository.
3. Publish generated workflows directly under `.github/workflows/`.
4. Disable the legacy GPU triggers only after parity and capacity checks pass.
5. Route `/rerun-failed-ci` through the scheduler and enable
  `SCHEDULER_RERUN_ENABLED`.

