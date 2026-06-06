# CI Scheduler

This directory contains a GitHub App service for scheduling `sglang-omni`
self-hosted GPU CI stages through a global, stage-level priority queue.

The service is intentionally separate from the model/runtime code. It is a
control plane only: it receives GitHub webhooks, persists queue state in SQLite,
updates check runs, and dispatches exactly one GPU stage at a time through
GitHub Actions `workflow_dispatch`.

## Components

- `scheduler_app/main.py`: FastAPI webhook and callback app.
- `scheduler_app/store.py`: SQLite schema and atomic queue operations.
- `scheduler_app/scheduler.py`: scheduling orchestration and dispatch.
- `scheduler_app/github.py`: GitHub App auth, Checks API, and Actions API.
- `stages.json`: canonical stage registry and dependency graph.
- `tools/run_stage.py`: helper used by the dispatched GPU workflow.
- `.github/workflows/scheduler-gpu-stage.yaml`: internal workflow target that
  runs a single selected stage on `runs-on: [self-hosted]`.

## Local Run

Create a local env file:

```bash
cp .env.example .env
```

For local development you may set `GITHUB_TOKEN` to a token with the required
repo permissions. For production, use GitHub App credentials instead.

Start the service:

```bash
docker compose up --build
```

Health check:

```bash
curl http://localhost:8080/health
```

For GitHub webhooks to reach a local machine, expose port `8080` with a tunnel
such as `ngrok` or `cloudflared`, then configure the GitHub App webhook URL:

```text
https://<tunnel-host>/webhook
```

## GitHub App Configuration

Subscribe to these webhook events:

- Pull request
- Issue comment
- Workflow run

Repository permissions:

- Actions: read/write
- Checks: read/write
- Contents: read
- Issues: read
- Pull requests: read
- Metadata: read

Configure `GITHUB_WEBHOOK_SECRET` in the service and in the GitHub App webhook
settings.

## Runtime Flow

1. A PR receives `run-ci`.
2. The GitHub App receives a `pull_request.labeled` webhook.
3. The app snapshots PR state and inserts one queue item per stage for the
   current PR SHA.
4. The app creates one queued check run per stage.
5. The scheduler selects dispatchable work under a SQLite write transaction.
6. High-priority PRs sort ahead of normal PRs, but only after dependencies are
   satisfied.
7. The app marks one selected item `dispatched`, updates its check to
   `in_progress`, and calls `workflow_dispatch` for
   `.github/workflows/scheduler-gpu-stage.yaml`.
8. The internal workflow checks out the PR SHA and runs only the selected
   `stage_id` on a self-hosted GPU runner.
9. The workflow calls `/callbacks/stage` on start and completion. The app also
   listens to `workflow_run.completed` as a fallback.
10. Completion releases the scheduler slot and wakes the scheduler again.

## Current Integration Boundary

This is the scheduler-side implementation plus a first single-stage workflow
target. The existing PR-triggered GPU workflows still need to be disabled or
refactored before this should be enabled in production; otherwise the old
workflows can still enqueue self-hosted jobs directly.

The intended production switch is:

- keep non-GPU checks such as lint/docs/layout on normal PR triggers;
- route GPU CI through this scheduler only;
- replace native `/rerun-failed-ci` reruns with the app's queue-based rerun
  handling.
