# SPDX-License-Identifier: Apache-2.0
from collections import deque
from threading import RLock
from types import SimpleNamespace

import pytest

from sglang_omni.proto.admin import (
    ADMIN_DESTROY_WEIGHTS_UPDATE_GROUP,
    ADMIN_INIT_WEIGHTS_UPDATE_GROUP,
    ADMIN_MODEL_INFO,
    ADMIN_UPDATE_WEIGHTS_FROM_DISK,
    ADMIN_UPDATE_WEIGHTS_FROM_DISTRIBUTED,
    ADMIN_UPDATE_WEIGHTS_FROM_TENSOR,
)
from sglang_omni.scheduling import omni_scheduler as module
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


@pytest.mark.parametrize("native_speculative", [False, True])
@pytest.mark.parametrize(
    ("action", "worker_method"),
    [
        (ADMIN_UPDATE_WEIGHTS_FROM_DISK, "update_weights_from_disk"),
        (ADMIN_UPDATE_WEIGHTS_FROM_TENSOR, "update_weights_from_tensor"),
        (ADMIN_UPDATE_WEIGHTS_FROM_DISTRIBUTED, "update_weights_from_distributed"),
        (ADMIN_INIT_WEIGHTS_UPDATE_GROUP, "init_weights_update_group"),
        (ADMIN_DESTROY_WEIGHTS_UPDATE_GROUP, "destroy_weights_update_group"),
    ],
)
def test_weight_admin_dispatch_rejects_native_speculation_only(
    native_speculative, action, worker_method
):
    calls = []
    scheduler = object.__new__(OmniScheduler)
    scheduler._native_speculative = native_speculative
    scheduler._running = False
    scheduler._scheduler_thread_id = None
    scheduler._admin_lock = RLock()
    scheduler._engine_paused = False
    scheduler._resolve_pending_async = lambda: None
    scheduler._active_request_ids = lambda: []
    scheduler._advance_prompt_cache_epoch = lambda: None
    scheduler.model_worker = SimpleNamespace(
        **{worker_method: lambda payload: calls.append(payload) or (True, "ok")}
    )
    payload = {"model_path": "/unused", "flush_cache": False}

    result = scheduler.admin(action, payload)

    assert scheduler._engine_paused is False
    if native_speculative:
        assert result["success"] is False
        assert "native speculative decoding" in result["error"]
        assert result["data"]["unsupported"] is True
        assert calls == []
    else:
        assert result["success"] is True
        assert calls == [payload]


@pytest.mark.parametrize("native_speculative", [False, True])
def test_model_info_reports_native_speculation_weight_updates_unsupported(
    monkeypatch, native_speculative
):
    scheduler = object.__new__(OmniScheduler)
    scheduler._native_speculative = native_speculative
    scheduler._running = False
    scheduler._scheduler_thread_id = None
    scheduler.model_worker = SimpleNamespace(
        model_info=lambda: {
            "supports_weight_update": True,
            "supports_weight_checker": True,
        }
    )
    scheduler._request_admission_lock = RLock()
    scheduler._pending_request_builds = {}
    scheduler._pending_request_admissions = {}
    scheduler._backlogged_request_build_payloads = deque()
    scheduler.waiting_queue = []
    scheduler.running_batch = SimpleNamespace(reqs=[])
    scheduler.tp_rank = 0
    scheduler.tp_size = 1
    scheduler._engine_paused = False
    scheduler.request_build_max_workers = 1
    scheduler.request_build_max_pending = 1
    scheduler._request_build_max_pending_observed = 0
    monkeypatch.setattr(
        module,
        "get_model",
        lambda: SimpleNamespace(model_path="/unused", load_format="auto"),
    )
    monkeypatch.setattr(
        module, "get_serving", lambda: SimpleNamespace(weight_version=None)
    )

    result = scheduler.admin(ADMIN_MODEL_INFO)

    assert result["success"] is True
    assert result["data"]["supports_weight_update"] is (not native_speculative)
    assert result["data"]["supports_weight_checker"] is True
