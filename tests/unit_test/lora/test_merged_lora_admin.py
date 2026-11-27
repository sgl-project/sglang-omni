# SPDX-License-Identifier: Apache-2.0
"""Unit tests for merged LoRA admin control plane, schedulers, and HTTP routes."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient

from sglang_omni.client.client import Client
from sglang_omni.lora.ipc_wire import encode_inline_tensor
from sglang_omni.lora.merged_manager import MergedLoRAManager
from sglang_omni.lora.weight_transfer import (
    MergedLoRAWorkerSession,
    build_merged_lora_admin_handler,
)
from sglang_omni.model_runner import model_worker as model_worker_mod
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner
from sglang_omni.models.auk.dit import AuKDit
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto.admin import (
    ADMIN_ABORT_LORA_UPDATE,
    ADMIN_FINISH_LORA_UPDATE,
    ADMIN_RESTORE_BASE_WEIGHTS,
    ADMIN_START_LORA_UPDATE,
    ADMIN_UPDATE_LORA_WEIGHTS,
    AdminResponse,
    is_update_action,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.serve.openai_api import create_app


def test_admin_constants_registered_as_update_actions() -> None:
    for action in (
        ADMIN_START_LORA_UPDATE,
        ADMIN_UPDATE_LORA_WEIGHTS,
        ADMIN_FINISH_LORA_UPDATE,
        ADMIN_ABORT_LORA_UPDATE,
        ADMIN_RESTORE_BASE_WEIGHTS,
    ):
        assert is_update_action(action) is True


def test_simple_scheduler_default_and_merged_lora_admin_handler() -> None:
    plain = SimpleScheduler(lambda x: x)
    skipped = plain.admin(ADMIN_START_LORA_UPDATE, {"update_id": "u0"})
    assert skipped["success"] is True
    assert skipped["data"]["skipped"] is True
    assert skipped["data"]["unsupported"] is True

    torch.manual_seed(3)
    dit = AuKDit(
        dim=16,
        num_layers=1,
        num_single_layers=1,
        heads=2,
        dim_head=8,
        text_hidden_dim=8,
        latent_dim=4,
    )
    base_qkv = dit.transformer_blocks[0].attn.to_qkv.weight.detach().clone()
    manager = MergedLoRAManager(
        dit, architecture="AuKForConditionalGeneration", prefer_pinned=False
    )
    session = MergedLoRAWorkerSession(manager)
    scheduler = SimpleScheduler(
        lambda x: x,
        admin_handler=build_merged_lora_admin_handler(session),
    )

    lora_a = torch.randn(2, 16)
    lora_b = torch.randn(16, 2)
    r1 = scheduler.admin(ADMIN_START_LORA_UPDATE, {"update_id": "s1"})
    assert r1["success"] is True

    r2 = scheduler.admin(
        ADMIN_UPDATE_LORA_WEIGHTS,
        {
            "update_id": "s1",
            "adapter_id": "dit_lora",
            "target_modules": ["to_q"],
            "tensors": {
                "dit.transformer_blocks.0.attn.to_q.lora_A.weight": encode_inline_tensor(
                    lora_a
                ),
                "dit.transformer_blocks.0.attn.to_q.lora_B.weight": encode_inline_tensor(
                    lora_b
                ),
            },
        },
    )
    assert r2["success"] is True

    r3 = scheduler.admin(ADMIN_FINISH_LORA_UPDATE, {"update_id": "s1"})
    assert r3["success"] is True
    assert torch.allclose(
        dit.transformer_blocks[0].attn.to_qkv.weight[:16],
        base_qkv[:16] + (lora_b @ lora_a),
        atol=1e-5,
    )

    r4 = scheduler.admin(ADMIN_RESTORE_BASE_WEIGHTS, {})
    assert r4["success"] is True
    assert torch.equal(dit.transformer_blocks[0].attn.to_qkv.weight, base_qkv)

    unknown = scheduler.admin("unknown_action", {})
    assert unknown["data"]["unsupported"] is True


def test_model_runner_blocks_lora_updates_when_weights_shared() -> None:
    runner = SGLModelRunner.__new__(SGLModelRunner)
    runner.weight_share_config = SimpleNamespace(role="follower")
    runner.merged_lora_session = None

    ok, msg, _ = runner.start_lora_update("u1")
    assert ok is False
    assert "weight sharing" in msg

    ok, msg, _ = runner.update_lora_weights("u1", {})
    assert ok is False
    assert "weight sharing" in msg

    ok, msg, _ = runner.finish_lora_update("u1")
    assert ok is False
    assert "weight sharing" in msg

    ok, msg, _ = runner.restore_base_weights()
    assert ok is False
    assert "weight sharing" in msg

    # abort_lora_update is non-mutating when no session exists.
    ok, _, data = runner.abort_lora_update("u1")
    assert ok is True
    assert data["committed"] is False


def test_omni_scheduler_and_model_worker_lora_admin_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []
    flushed = []
    epochs = []
    version_overrides: list[tuple[str, dict[str, object]]] = []

    def fake_override(server_args: object, source: str, **fields: object) -> None:
        version_overrides.append((source, dict(fields)))
        for key, value in fields.items():
            setattr(server_args, key, value)

    monkeypatch.setattr(model_worker_mod, "override_server_args", fake_override)

    class FakeRunner:
        def start_lora_update(
            self, update_id: str
        ) -> tuple[bool, str, dict[str, object]]:
            calls.append(("start", update_id))
            return True, "started", {"update_id": update_id}

        def update_lora_weights(
            self, update_id: str, payload: dict[str, Any]
        ) -> tuple[bool, str, dict[str, object]]:
            calls.append(("update", update_id))
            return True, "staged", {"update_id": update_id}

        def finish_lora_update(
            self, update_id: str
        ) -> tuple[bool, str, dict[str, object]]:
            calls.append(("finish", update_id))
            return True, "committed", {"update_id": update_id}

        def abort_lora_update(
            self, update_id: str | None = None
        ) -> tuple[bool, str, dict[str, object]]:
            calls.append(("abort", update_id))
            return True, "aborted", {"aborted_update_id": update_id}

        def restore_base_weights(self) -> tuple[bool, str, dict[str, object]]:
            calls.append(("restore", None))
            return True, "restored", {"restored_params": 1}

    worker = ModelWorker.__new__(ModelWorker)
    worker.model_runner = FakeRunner()
    worker.server_args = SimpleNamespace(weight_version="0")

    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler.admin_lock = threading.Lock()
    scheduler.model_worker = worker
    scheduler.enable_overlap = False
    scheduler.async_pending = None
    scheduler._engine_paused = False  # noqa: leading-underscore
    scheduler.last_pause_mode = None
    scheduler.active_request_ids = lambda: []
    scheduler.abort_all_requests = lambda: []
    scheduler.advance_prompt_cache_epoch = lambda: epochs.append(1)
    scheduler.flush_cache_after_update = lambda: flushed.append(1) or True
    scheduler.empty_torch_cache = lambda: None

    res_start = scheduler.run_admin_action(ADMIN_START_LORA_UPDATE, {"update_id": "u1"})
    assert res_start["success"] is True

    res_stage = scheduler.run_admin_action(
        ADMIN_UPDATE_LORA_WEIGHTS,
        {
            "update_id": "u1",
            "adapter_id": "a1",
            "target_modules": ["to_q"],
            "tensors": {},
        },
    )
    assert res_stage["success"] is True

    res_finish = scheduler.run_admin_action(
        ADMIN_FINISH_LORA_UPDATE, {"update_id": "u1", "weight_version": "v1"}
    )
    assert res_finish["success"] is True
    assert worker.server_args.weight_version == "v1"
    assert len(epochs) == 1
    assert len(flushed) == 1

    res_abort = scheduler.run_admin_action(ADMIN_ABORT_LORA_UPDATE, {"update_id": "u2"})
    assert res_abort["success"] is True

    res_restore = scheduler.run_admin_action(
        ADMIN_RESTORE_BASE_WEIGHTS, {"weight_version": "base"}
    )
    assert res_restore["success"] is True
    assert worker.server_args.weight_version == "base"
    assert version_overrides == [
        ("sglang-omni-lora-update", {"weight_version": "v1"}),
        ("sglang-omni-lora-restore", {"weight_version": "base"}),
    ]
    assert len(epochs) == 2
    assert len(flushed) == 2
    assert [name for name, _ in calls] == [
        "start",
        "update",
        "finish",
        "abort",
        "restore",
    ]


@pytest.mark.asyncio
async def test_coordinator_client_and_http_lora_endpoints() -> None:
    recorded: list[tuple[str, dict[str, Any]]] = []

    coord = Coordinator.__new__(Coordinator)

    async def fake_admin(
        action: str,
        payload: dict[str, object] | None = None,
        *,
        stages: Any = None,
        timeout_s: float = 30.0,
    ) -> AdminResponse:
        recorded.append((action, dict(payload or {})))
        return AdminResponse(
            operation_id="op1",
            action=action,
            success=True,
            message="ok",
        )

    coord.admin = fake_admin
    client = Client.__new__(Client)
    client.coordinator = coord

    await client.start_lora_update({"update_id": "u1"})
    await client.update_lora_weights(
        {
            "update_id": "u1",
            "adapter_id": "a1",
            "target_modules": ["to_q"],
            "tensors": {},
        }
    )
    await client.finish_lora_update({"update_id": "u1"})
    await client.abort_lora_update({"update_id": "u1"})
    await client.restore_base_weights({})
    assert [a for a, _ in recorded] == [
        ADMIN_START_LORA_UPDATE,
        ADMIN_UPDATE_LORA_WEIGHTS,
        ADMIN_FINISH_LORA_UPDATE,
        ADMIN_ABORT_LORA_UPDATE,
        ADMIN_RESTORE_BASE_WEIGHTS,
    ]

    recorded.clear()
    app = create_app(
        client=client,
        model_name="test-model",
    )
    http = TestClient(app)
    assert http.post("/start_lora_update", json={"update_id": "u9"}).status_code == 200
    assert (
        http.post(
            "/update_lora_weights",
            json={
                "update_id": "u9",
                "adapter_id": "a9",
                "target_modules": ["to_q"],
                "tensors": {},
            },
        ).status_code
        == 200
    )
    assert http.post("/finish_lora_update", json={"update_id": "u9"}).status_code == 200
    assert http.post("/abort_lora_update", json={"update_id": "u9"}).status_code == 200
    assert http.post("/restore_base_weights", json={}).status_code == 200
    assert [a for a, _ in recorded] == [
        ADMIN_START_LORA_UPDATE,
        ADMIN_UPDATE_LORA_WEIGHTS,
        ADMIN_FINISH_LORA_UPDATE,
        ADMIN_ABORT_LORA_UPDATE,
        ADMIN_RESTORE_BASE_WEIGHTS,
    ]
