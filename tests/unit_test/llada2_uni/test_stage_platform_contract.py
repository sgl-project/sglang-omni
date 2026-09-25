# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sglang.srt.model_executor.cuda_graph_config import Backend

import sglang_omni.platforms as platforms
from sglang_omni.models.llada2_uni import stages
from sglang_omni.models.llada2_uni.config import IMAGE_STAGE, THINKER_STAGE, EntryClass
from tests.unit_test.fixtures.mini_checkpoint import write_mini_llama_checkpoint


class _FakePlatform:

    def __init__(
        self,
        *,
        dllm_backend: str | None,
        dllm_graph: bool,
        device_type: str = "xpu",
        decode_graph_backend: str | None = None,
    ) -> None:
        self.device_type = device_type
        self._dllm_backend = dllm_backend
        self._dllm_graph = dllm_graph
        self._decode_graph_backend = decode_graph_backend

    def get_dllm_attention_backend(self) -> str | None:
        return self._dllm_backend

    def enable_dllm_decode_graph(self) -> bool:
        return self._dllm_graph

    def get_decode_cuda_graph_backend(self) -> str | None:
        return self._decode_graph_backend


def _drive_thinker(
    monkeypatch: pytest.MonkeyPatch,
    platform: _FakePlatform,
    **factory_kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from sglang_omni.models.llada2_uni import bootstrap
    from sglang_omni.scheduling import sglang_backend

    build_kwargs: dict[str, Any] = {}
    scheduler_kwargs: dict[str, Any] = {}

    def fake_build(model_path, **kwargs):
        del model_path
        build_kwargs.update(kwargs)
        decode = kwargs.get("cuda_graph_backend_decode")
        if decode is None:
            decode = (
                Backend.DISABLED if kwargs.get("disable_cuda_graph") else Backend.FULL
            )
        return SimpleNamespace(
            attention_backend=kwargs.get("attention_backend"),
            dllm_algorithm=kwargs.get("dllm_algorithm"),
            mem_fraction_static=None,
            disable_cuda_graph=bool(kwargs.get("disable_cuda_graph", False)),
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(backend=decode),
                prefill=SimpleNamespace(backend=Backend.DISABLED),
            ),
        )

    def fake_scheduler(server_args, gpu_id, **kwargs):
        scheduler_kwargs.update(
            {"server_args": server_args, "gpu_id": gpu_id, **kwargs}
        )
        return SimpleNamespace()

    monkeypatch.setattr(platforms, "current_platform", platform)
    monkeypatch.setattr(sglang_backend, "build_sglang_server_args", fake_build)
    monkeypatch.setattr(bootstrap, "create_dllm_thinker_scheduler", fake_scheduler)

    stages.create_sglang_dllm_thinker_executor_from_config("unused", **factory_kwargs)
    return build_kwargs, scheduler_kwargs


def _write_mini_dllm_checkpoint(tmp_path: Path) -> str:
    return write_mini_llama_checkpoint(tmp_path, architectures=["LLaDA2MoeModelLM"])


def _drive_real_thinker(
    monkeypatch: pytest.MonkeyPatch,
    platform: _FakePlatform,
    model_path: str,
    **overrides: Any,
) -> dict[str, Any]:
    from sglang_omni.models.llada2_uni import bootstrap

    scheduler_kwargs: dict[str, Any] = {}

    def fake_scheduler(server_args, gpu_id, **kwargs):
        scheduler_kwargs.update(
            {"server_args": server_args, "gpu_id": gpu_id, **kwargs}
        )
        return SimpleNamespace()

    monkeypatch.setattr(platforms, "current_platform", platform)
    monkeypatch.setattr(bootstrap, "create_dllm_thinker_scheduler", fake_scheduler)

    stages.create_sglang_dllm_thinker_executor_from_config(
        model_path,
        device="cuda",
        gpu_id=0,
        max_seq_len=2048,
        server_args_overrides=dict(overrides),
    )
    return scheduler_kwargs


def test_no_stage_pins_a_device_in_the_pipeline_config() -> None:
    config = EntryClass(model_path="unused")

    assert config.stage_named(IMAGE_STAGE).factory.device is None
    assert config.stage_named(THINKER_STAGE).factory.device is None


def test_the_image_encoder_resolves_an_absent_device_to_its_placed_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.llada2_uni.components import image_encoder

    built: dict[str, Any] = {}

    class _Encoder:
        def __init__(self, *, model_path, device, dtype):
            del model_path, dtype
            built["device"] = device

    monkeypatch.setattr(image_encoder, "LLaDA2ImageEncoder", _Encoder)

    stages.create_image_encoder_executor("unused", device=None, gpu_id=1)

    live = platforms.current_platform.device_type
    assert built["device"] == ("cpu" if live == "cpu" else f"{live}:1")


def test_the_thinker_asks_for_the_backend_its_platform_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    named, _ = _drive_thinker(
        monkeypatch, _FakePlatform(dllm_backend="triton", dllm_graph=False)
    )
    assert named["attention_backend"] == "triton"

    unnamed, _ = _drive_thinker(
        monkeypatch, _FakePlatform(dllm_backend=None, dllm_graph=False)
    )
    assert unnamed["attention_backend"] == "flashinfer"


def test_a_platform_named_backend_refuses_the_capture_that_would_replace_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="flashinfer"):
        _drive_thinker(
            monkeypatch, _FakePlatform(dllm_backend="triton", dllm_graph=True)
        )


def test_the_platforms_sglang_covers_keep_its_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    named, _ = _drive_thinker(
        monkeypatch, _FakePlatform(dllm_backend=None, dllm_graph=True)
    )

    assert named["attention_backend"] == "flashinfer"
    assert "disable_cuda_graph" not in named


def test_an_operator_named_backend_does_not_escape_the_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="flashinfer"):
        _drive_thinker(
            monkeypatch,
            _FakePlatform(dllm_backend="triton", dllm_graph=True),
            server_args_overrides={"attention_backend": "intel_xpu"},
        )

    named, _ = _drive_thinker(
        monkeypatch,
        _FakePlatform(dllm_backend="triton", dllm_graph=True),
        server_args_overrides={
            "attention_backend": "intel_xpu",
            "disable_cuda_graph": True,
        },
    )
    assert named["attention_backend"] == "intel_xpu"


def test_graph_capture_follows_the_platform_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eager, _ = _drive_thinker(
        monkeypatch, _FakePlatform(dllm_backend=None, dllm_graph=False)
    )
    assert eager["disable_cuda_graph"] is True

    captured, _ = _drive_thinker(
        monkeypatch, _FakePlatform(dllm_backend=None, dllm_graph=True)
    )
    assert "disable_cuda_graph" not in captured


def test_a_stated_graph_switch_outranks_the_platform_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forced_off, _ = _drive_thinker(
        monkeypatch,
        _FakePlatform(dllm_backend="triton", dllm_graph=True),
        server_args_overrides={"disable_cuda_graph": True},
    )
    assert forced_off["disable_cuda_graph"] is True
    assert forced_off["attention_backend"] == "triton"

    forced_on, _ = _drive_thinker(
        monkeypatch,
        _FakePlatform(dllm_backend=None, dllm_graph=False),
        server_args_overrides={"disable_cuda_graph": False},
    )
    assert forced_on["disable_cuda_graph"] is False


def test_the_thinker_passes_its_tp_identity_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    named, scheduler = _drive_thinker(
        monkeypatch,
        _FakePlatform(dllm_backend="triton", dllm_graph=False),
        gpu_id=2,
        tp_rank=1,
        tp_size=2,
        nccl_port=29500,
    )

    assert named["tp_size"] == 2
    assert scheduler["gpu_id"] == 2
    assert scheduler["tp_rank"] == 1
    assert scheduler["nccl_port"] == 29500


def test_the_resolved_decode_backend_is_never_the_field_on_server_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sglang_omni.scheduling.generation_batch_policy import (
        get_decode_cuda_graph_backend,
    )

    scheduler = _drive_real_thinker(
        monkeypatch,
        _FakePlatform(dllm_backend="triton", dllm_graph=False, device_type="cuda"),
        _write_mini_dllm_checkpoint(tmp_path),
    )
    server_args = scheduler["server_args"]

    assert server_args._resolution_finished is True
    assert server_args.cuda_graph_config is None
    assert get_decode_cuda_graph_backend(server_args) == Backend.DISABLED


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"cuda_graph_backend_decode": "full"}, id="per-phase-backend"),
        pytest.param(
            {"cuda_graph_config": {"decode": {"backend": "full"}}}, id="nested-config"
        ),
    ],
)
def test_capture_turned_on_past_the_disable_switch_still_meets_the_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, overrides: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="flashinfer"):
        _drive_real_thinker(
            monkeypatch,
            _FakePlatform(dllm_backend="triton", dllm_graph=False, device_type="cuda"),
            _write_mini_dllm_checkpoint(tmp_path),
            **overrides,
        )


def test_a_platform_that_defaults_decode_capture_on_meets_the_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    platform = _FakePlatform(
        dllm_backend="triton",
        dllm_graph=True,
        device_type="cuda",
        decode_graph_backend=Backend.FULL,
    )
    with pytest.raises(ValueError, match="flashinfer"):
        _drive_real_thinker(
            monkeypatch, platform, _write_mini_dllm_checkpoint(tmp_path)
        )


def test_the_thinker_takes_the_memory_fraction_its_stage_was_placed_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.config.runtime import resolve_stage_factory_args

    config = EntryClass(model_path="unused")
    stage_cfg = config.stage_named(THINKER_STAGE)
    stage_cfg.gpu_memory_fraction = 0.8

    resolved = resolve_stage_factory_args(stage_cfg, config, gpu_id=0)
    assert resolved["total_gpu_memory_fraction"] == 0.8

    _, scheduler = _drive_thinker(
        monkeypatch,
        _FakePlatform(dllm_backend="triton", dllm_graph=False),
        total_gpu_memory_fraction=0.8,
    )
    assert scheduler["total_gpu_memory_fraction"] == 0.8


def test_the_thinker_refuses_a_tp_size_the_pipeline_did_not_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="tp_size"):
        _drive_thinker(
            monkeypatch,
            _FakePlatform(dllm_backend="triton", dllm_graph=False),
            tp_size=2,
            server_args_overrides={"tp_size": 4},
        )

    named, _ = _drive_thinker(
        monkeypatch,
        _FakePlatform(dllm_backend="triton", dllm_graph=False),
        tp_size=2,
        server_args_overrides={"tp_size": 2},
    )
    assert named["tp_size"] == 2


def test_the_dllm_scheduler_takes_its_round_cap_from_the_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.scheduling import dllm_scheduler as dllm_scheduler_module
    from sglang_omni.scheduling.dllm_scheduler import DllmScheduler

    monkeypatch.setattr(
        dllm_scheduler_module,
        "current_platform",
        SimpleNamespace(dllm_max_requests_per_round=lambda: 1),
    )

    scheduler = DllmScheduler(
        tp_worker=SimpleNamespace(),
        tree_cache=SimpleNamespace(),
        req_to_token_pool=SimpleNamespace(),
        token_to_kv_pool_allocator=SimpleNamespace(),
        server_args=SimpleNamespace(),
        model_config=SimpleNamespace(),
        dllm_config=SimpleNamespace(block_size=32),
        request_builder=lambda data: data,
        result_adapter=lambda data: data,
    )

    assert scheduler.max_requests_per_round == 1
    assert scheduler.requires_tp_work_fanout is True


def test_every_platform_answers_the_hooks_this_model_reads() -> None:
    from sglang_omni.platforms.cuda import CUDAOmniPlatform
    from sglang_omni.platforms.interface import OmniPlatform
    from sglang_omni.platforms.rocm import ROCMOmniPlatform
    from sglang_omni.platforms.xpu import XPUOmniPlatform

    expected = {
        OmniPlatform: (None, False, None),
        CUDAOmniPlatform: (None, False, None),
        ROCMOmniPlatform: (None, False, None),
        XPUOmniPlatform: ("triton", False, 1),
    }
    for platform_class, (backend, graph, cap) in expected.items():
        platform = platform_class()
        name = platform_class.__name__
        assert platform.get_dllm_attention_backend() == backend, name
        assert platform.enable_dllm_decode_graph() is graph, name
        assert platform.dllm_max_requests_per_round() == cap, name
