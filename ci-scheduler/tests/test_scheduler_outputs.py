from __future__ import annotations

import asyncio
import io
import json
import types
import zipfile
from pathlib import Path

import pytest

from adapters.github_client import GitHubClient, GitHubError
from adapters.sqlite_store import SqliteQueueStore
from app.scheduler import CiScheduler
from compiler.graph import compile_workflow
from domain.models import PullRequestState


REPO = "sgl-project/sglang-omni"
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"


def _artifact_zip(payload: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("scheduler-outputs.json", json.dumps(payload))
    return buffer.getvalue()


def test_github_client_validates_scheduler_output_artifact() -> None:
    client = GitHubClient(object())  # type: ignore[arg-type]

    async def fake_request(self, method, path, **kwargs):
        return {"artifacts": [{"id": 99, "name": "dispatch-1", "expired": False}]}

    async def fake_request_bytes(self, method, path, **kwargs):
        return _artifact_zip({"version": 1, "outputs": {"tts_ci_model": "higgs"}})

    client._request = types.MethodType(fake_request, client)  # type: ignore[method-assign]
    client._request_bytes = types.MethodType(fake_request_bytes, client)  # type: ignore[method-assign]

    outputs = asyncio.run(
        client.get_scheduler_outputs(
            repo=REPO,
            installation_id=1,
            workflow_run_id=10,
            dispatch_id="dispatch-1",
        )
    )
    assert outputs == {"tts_ci_model": "higgs"}


def test_github_client_rejects_wrong_output_schema() -> None:
    client = GitHubClient(object())  # type: ignore[arg-type]

    async def fake_request(self, method, path, **kwargs):
        return {"artifacts": [{"id": 99, "name": "dispatch-1", "expired": False}]}

    async def fake_request_bytes(self, method, path, **kwargs):
        return _artifact_zip({"version": 2, "outputs": {}})

    client._request = types.MethodType(fake_request, client)  # type: ignore[method-assign]
    client._request_bytes = types.MethodType(fake_request_bytes, client)  # type: ignore[method-assign]

    with pytest.raises(GitHubError, match="schema version"):
        asyncio.run(
            client.get_scheduler_outputs(
                repo=REPO,
                installation_id=1,
                workflow_run_id=10,
                dispatch_id="dispatch-1",
            )
        )


def test_scheduler_context_contains_stored_dependency_outputs(tmp_path: Path) -> None:
    graph = compile_workflow(WORKFLOWS_DIR / "omni-ci.yaml", workflows_dir=WORKFLOWS_DIR)
    store = SqliteQueueStore(tmp_path / "queue.sqlite3")
    pr = PullRequestState(
        repo=REPO,
        pr_number=100,
        head_sha="abc123",
        state="open",
        draft=False,
        labels=frozenset({"run-ci", "high-priority"}),
        installation_id=1,
    )
    items = store.enqueue_pr(pr, graph, run_label="run-ci")
    preflight = next(item for item in items if item.stage_key == "omni-ci::preflight")
    pick = next(item for item in items if item.stage_key == "omni-ci::pick-tts-model")
    store.store_outputs(
        repo=REPO,
        head_sha=pr.head_sha,
        stage_key=preflight.stage_key,
        attempt=preflight.attempt,
        outputs={"labels": '["run-ci", "high-priority"]'},
    )

    scheduler = CiScheduler(
        settings=object(),  # type: ignore[arg-type]
        store=store,
        github=object(),  # type: ignore[arg-type]
        graphs={"omni-ci": graph},
    )
    context = scheduler.build_scheduler_context(pick, graph).to_json_dict()

    assert context["source_event"]["number"] == 100
    assert context["source_event"]["head"]["sha"] == "abc123"
    assert context["needs"]["omni-ci::preflight"]["outputs"] == {
        "labels": '["run-ci", "high-priority"]'
    }
