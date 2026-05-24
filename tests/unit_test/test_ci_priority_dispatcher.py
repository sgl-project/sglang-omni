from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_dispatcher_module():
    repo_root = Path(__file__).resolve().parents[2]
    scripts_dir = repo_root / ".github" / "scripts"
    sys.path.insert(0, str(scripts_dir))
    module_path = scripts_dir / "ci_priority_dispatcher.py"
    spec = importlib.util.spec_from_file_location("ci_priority_dispatcher", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dispatcher = _load_dispatcher_module()


class FakeClient:
    def __init__(self):
        self.created = []
        self.updated = []

    def post(self, path, payload=None):
        created = {"id": 100 + len(self.created), **payload}
        self.created.append((path, payload))
        return created

    def patch(self, path, payload):
        self.updated.append((path, payload))
        return {"id": int(path.rsplit("/", 1)[-1]), **payload}


def test_stage_check_runs_for_sha_keeps_latest_check_per_name():
    class CheckClient:
        def paginate(self, path, params=None):
            assert path == "/commits/abc/check-runs"
            assert params == {"filter": "latest"}
            return iter(
                [
                    {
                        "id": 1,
                        "name": "stage",
                        "started_at": "2026-05-24T00:00:00Z",
                    },
                    {
                        "id": 2,
                        "name": "stage",
                        "started_at": "2026-05-24T00:01:00Z",
                    },
                ]
            )

    checks = dispatcher._stage_check_runs_for_sha(CheckClient(), "abc")

    assert checks["stage"]["id"] == 2


def test_upsert_creates_new_check_when_rerunning_completed_stage():
    client = FakeClient()
    existing = {"stage": {"id": 42, "name": "stage", "status": "completed"}}

    dispatcher._upsert_check_run(
        client,
        existing,
        {
            "name": "stage",
            "head_sha": "abc",
            "status": "in_progress",
        },
    )

    assert client.updated == []
    assert client.created == [
        (
            "/check-runs",
            {
                "name": "stage",
                "head_sha": "abc",
                "status": "in_progress",
            },
        )
    ]
    assert existing["stage"]["id"] == 100


def test_upsert_updates_non_completed_check_in_place():
    client = FakeClient()
    existing = {"stage": {"id": 42, "name": "stage", "status": "queued"}}

    dispatcher._upsert_check_run(
        client,
        existing,
        {
            "name": "stage",
            "head_sha": "abc",
            "status": "completed",
            "conclusion": "success",
        },
    )

    assert client.created == []
    assert client.updated == [
        (
            "/check-runs/42",
            {
                "name": "stage",
                "status": "completed",
                "conclusion": "success",
            },
        )
    ]
