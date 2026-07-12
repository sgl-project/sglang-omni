from __future__ import annotations

import hmac
import json
import re
from hashlib import sha256
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

from adapters.github_client import GitHubClient
from adapters.sqlite_store import SqliteQueueStore
from app.config import get_settings
from app.scheduler import CiScheduler, load_graphs
from domain.models import PullRequestState


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    root = Path(__file__).resolve().parents[1]
    return root / path


settings = get_settings()
store = SqliteQueueStore(_resolve_path(settings.database_path))
graphs = load_graphs(
    _resolve_path(settings.workflows_dir),
    roots=settings.scheduler_root_workflows,
)
github = GitHubClient(settings)
scheduler = CiScheduler(settings=settings, store=store, github=github, graphs=graphs)

app = FastAPI(title="sglang-omni CI Scheduler")


class StageCallback(BaseModel):
    queue_item_id: int
    status: str
    workflow_run_id: int | None = None


def _verify_signature(body: bytes, signature: str | None) -> None:
    if not settings.webhook_secret:
        return
    if not signature or not signature.startswith("sha256="):
        raise HTTPException(status_code=401, detail="Missing GitHub webhook signature")
    digest = hmac.new(settings.webhook_secret.encode(), body, sha256).hexdigest()
    expected = f"sha256={digest}"
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=401, detail="Invalid GitHub webhook signature")


def _repo_name(payload: dict[str, Any]) -> str:
    repo = payload.get("repository") or {}
    return repo.get("full_name") or settings.default_repo


def _installation_id(payload: dict[str, Any]) -> int | None:
    installation = payload.get("installation")
    if not installation:
        return None
    return int(installation["id"])


def _labels_from_pr(pr: dict[str, Any]) -> frozenset[str]:
    return frozenset(label["name"] for label in pr.get("labels", []))


def _pr_state_from_payload(payload: dict[str, Any]) -> PullRequestState:
    pr = payload["pull_request"]
    return PullRequestState(
        repo=_repo_name(payload),
        pr_number=int(pr["number"]),
        head_sha=pr["head"]["sha"],
        state=pr["state"],
        draft=bool(pr.get("draft", False)),
        labels=_labels_from_pr(pr),
        installation_id=_installation_id(payload),
    )


def _pr_state_from_api(repo: str, pr: dict[str, Any], installation_id: int | None) -> PullRequestState:
    return PullRequestState(
        repo=repo,
        pr_number=int(pr["number"]),
        head_sha=pr["head"]["sha"],
        state=pr["state"],
        draft=bool(pr.get("draft", False)),
        labels=_labels_from_pr(pr),
        installation_id=installation_id,
    )


def _commenter_can_rerun(payload: dict[str, Any]) -> bool:
    association = (payload.get("comment") or {}).get("author_association")
    return association in {"OWNER", "MEMBER", "COLLABORATOR"}


def _workflow_run_queue_item_id(payload: dict[str, Any]) -> int | None:
    run = payload.get("workflow_run") or {}
    text = " ".join(
        str(value or "")
        for value in (
            run.get("name"),
            run.get("display_title"),
            run.get("run_number"),
        )
    )
    match = re.search(r"(?:queue[_ -]?item[_ -]?id=|queue-item-)(\d+)", text)
    return int(match.group(1)) if match else None


def _status_from_workflow_conclusion(conclusion: str | None) -> str:
    if conclusion == "success":
        return "passed"
    if conclusion == "skipped":
        return "skipped"
    if conclusion == "timed_out":
        return "timed_out"
    if conclusion == "cancelled":
        return "cancelled"
    return "failed"


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/webhook")
async def github_webhook(
    request: Request,
    x_github_event: str = Header(alias="X-GitHub-Event"),
    x_hub_signature_256: str | None = Header(default=None, alias="X-Hub-Signature-256"),
) -> dict[str, Any]:
    body = await request.body()
    _verify_signature(body, x_hub_signature_256)
    payload = json.loads(body)

    if x_github_event == "pull_request":
        await _handle_pull_request(payload)
        return {"handled": "pull_request"}

    if x_github_event == "issue_comment":
        await _handle_issue_comment(payload)
        return {"handled": "issue_comment"}

    if x_github_event == "workflow_run":
        await _handle_workflow_run(payload)
        return {"handled": "workflow_run"}

    if x_github_event == "workflow_job":
        await _handle_workflow_job(payload)
        return {"handled": "workflow_job"}

    return {"ignored": x_github_event}


@app.post("/callbacks/stage")
async def stage_callback(
    callback: StageCallback,
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> dict[str, Any]:
    if settings.callback_token:
        expected = f"Bearer {settings.callback_token}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Invalid callback token")

    if callback.status == "running":
        await scheduler.mark_running(callback.queue_item_id, callback.workflow_run_id)
    elif callback.status in {"passed", "failed", "cancelled", "timed_out", "skipped"}:
        await scheduler.mark_terminal(
            callback.queue_item_id,
            callback.status,
            callback.workflow_run_id,
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported status: {callback.status}")
    return {"ok": True}


async def _handle_pull_request(payload: dict[str, Any]) -> None:
    action = payload.get("action")
    pr = _pr_state_from_payload(payload)

    if action == "unlabeled" and (payload.get("label") or {}).get("name") == settings.run_label:
        await scheduler.cancel_pr(pr)
        return

    if action in {
        "opened",
        "synchronize",
        "reopened",
        "ready_for_review",
        "converted_to_draft",
        "labeled",
        "unlabeled",
        "closed",
    }:
        await scheduler.enqueue_pr(pr)


async def _handle_issue_comment(payload: dict[str, Any]) -> None:
    issue = payload.get("issue") or {}
    if not issue.get("pull_request"):
        return

    body = ((payload.get("comment") or {}).get("body") or "").strip()
    first_line = body.splitlines()[0].strip() if body else ""
    if first_line != "/rerun-failed-ci":
        return
    if not _commenter_can_rerun(payload):
        raise HTTPException(status_code=403, detail="Commenter cannot rerun CI")

    repo = _repo_name(payload)
    installation_id = _installation_id(payload)
    pull = await github.get_pull(repo, int(issue["number"]), installation_id)
    pr = _pr_state_from_api(repo, pull, installation_id)
    await scheduler.rerun_failed(pr)


async def _handle_workflow_run(payload: dict[str, Any]) -> None:
    action = payload.get("action")
    run = payload.get("workflow_run") or {}
    if action != "completed":
        return
    queue_item_id = _workflow_run_queue_item_id(payload)
    if queue_item_id is None:
        return
    await scheduler.mark_terminal(
        queue_item_id,
        _status_from_workflow_conclusion(run.get("conclusion")),
        workflow_run_id=int(run["id"]) if run.get("id") else None,
    )


async def _handle_workflow_job(payload: dict[str, Any]) -> None:
    action = payload.get("action")
    job = payload.get("workflow_job") or {}
    if action != "in_progress":
        return
    run = payload.get("workflow_run") or {}
    text = str(run.get("name") or "")
    match = re.search(r"queue-item-(\d+)", text)
    if not match:
        return
    await scheduler.mark_running(int(match.group(1)), int(run["id"]) if run.get("id") else None)
