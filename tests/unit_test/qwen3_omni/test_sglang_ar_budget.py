# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import subprocess
import sys
from types import SimpleNamespace

import pytest
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

import sglang_omni.model_runner.sglang_model_runner as runner_mod
import sglang_omni.models.qwen3_omni.bootstrap as qwen_bootstrap
import sglang_omni.models.qwen3_omni.stages as qwen_stages
from sglang_omni.platforms import current_platform


def _configurator(*, total_gpu_memory_fraction: float | None):
    """Build the profiling surface without upstream's full field set.

    ``_OmniKVCacheConfigurator`` is a slots dataclass with ~25 required fields,
    so populate only the attributes ``_profile_available_bytes`` reads.
    """
    configurator = runner_mod._OmniKVCacheConfigurator.__new__(
        runner_mod._OmniKVCacheConfigurator
    )
    configurator.gpu_id = 0
    configurator.device = "cuda"
    configurator.server_args = SimpleNamespace(mem_fraction_static=0.9)
    configurator.total_gpu_memory_fraction = total_gpu_memory_fraction
    return configurator


def _patch_thinker_startup(monkeypatch) -> list[dict[str, object]]:
    scheduler_calls: list[dict[str, object]] = []

    def _fake_server_args_builder(model_path, context_length, **overrides):
        assert model_path == "dummy"
        assert context_length == 8192
        assert overrides["sampling_backend"] == "pytorch"
        return SimpleNamespace(
            mem_fraction_static=overrides["mem_fraction_static"],
            sampling_backend=overrides["sampling_backend"],
            max_running_requests=overrides["max_running_requests"],
            cuda_graph_max_bs=overrides["cuda_graph_max_bs"],
            cuda_graph_bs=overrides["cuda_graph_bs"],
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(
                    max_bs=overrides["cuda_graph_max_bs"],
                    bs=overrides["cuda_graph_bs"],
                ),
                prefill=SimpleNamespace(backend="disabled", bs=None, max_bs=None),
            ),
            disable_cuda_graph=overrides["disable_cuda_graph"],
            enable_torch_compile=overrides.get("enable_torch_compile", False),
            torch_compile_max_bs=overrides["torch_compile_max_bs"],
        )

    def _fake_create_thinker_scheduler(server_args, gpu_id, **kwargs):
        scheduler_calls.append(
            {
                "mem_fraction_static": server_args.mem_fraction_static,
                "sampling_backend": server_args.sampling_backend,
                "gpu_id": gpu_id,
                "total_gpu_memory_fraction": kwargs["total_gpu_memory_fraction"],
            }
        )
        return object()

    monkeypatch.setattr(
        qwen_stages,
        "build_sglang_server_args",
        _fake_server_args_builder,
    )
    monkeypatch.setattr(
        qwen_stages,
        "create_thinker_scheduler",
        _fake_create_thinker_scheduler,
    )
    monkeypatch.setattr(qwen_stages, "avail_gpu_mem", lambda gpu_id: 90.0)
    monkeypatch.setattr(
        qwen_stages,
        "get_process_gpu_memory_bytes",
        lambda gpu_id: None,
    )
    return scheduler_calls


def test_colocated_ar_budget_uses_stage_total_fraction(monkeypatch) -> None:
    configurator = _configurator(total_gpu_memory_fraction=0.4)
    monkeypatch.setattr(
        runner_mod,
        "get_process_gpu_memory_bytes",
        lambda gpu_id: 30 * 1024**3,
    )
    monkeypatch.setattr(
        runner_mod,
        "get_gpu_device_info",
        lambda gpu_id: SimpleNamespace(total_memory_bytes=100 * 1024**3),
    )

    available = configurator._profile_available_bytes(0)

    assert available == 10 * 1024**3


@pytest.mark.parametrize("process_memory", [None, 0])
def test_colocated_ar_budget_uses_stage_load_delta_when_process_memory_unavailable(
    monkeypatch,
    process_memory,
) -> None:
    configurator = _configurator(total_gpu_memory_fraction=0.4)
    monkeypatch.setattr(
        runner_mod,
        "get_process_gpu_memory_bytes",
        lambda gpu_id: process_memory,
    )
    monkeypatch.setattr(
        runner_mod,
        "get_gpu_device_info",
        lambda gpu_id: SimpleNamespace(total_memory_bytes=100 * 1024**3),
    )

    def _fake_stage_load_delta(self, pre_model_load_memory, total_memory):
        assert pre_model_load_memory == 95.0
        assert total_memory == 100 * 1024**3
        return 7 * 1024**3

    monkeypatch.setattr(
        runner_mod._OmniKVCacheConfigurator,
        "_profile_available_bytes_from_stage_load_delta",
        _fake_stage_load_delta,
    )

    assert configurator._profile_available_bytes(95.0) == 7 * 1024**3


def test_non_colocated_ar_delegates_to_upstream_available_bytes(
    monkeypatch,
) -> None:
    configurator = _configurator(total_gpu_memory_fraction=None)

    def _fake_upstream_profile(self, pre_model_load_memory):
        assert pre_model_load_memory == 123
        return 456

    monkeypatch.setattr(
        KVCacheConfigurator,
        "_profile_available_bytes",
        _fake_upstream_profile,
    )

    assert configurator._profile_available_bytes(123) == 456


def test_qwen_ar_factory_derives_mem_fraction_from_total_budget() -> None:
    overrides = {"disable_cuda_graph": False}

    contract = qwen_stages._apply_colocated_ar_memory_contract(
        overrides,
        stage_name="thinker",
        total_gpu_memory_fraction=0.78,
    )

    assert overrides["mem_fraction_static"] == 0.78
    assert contract.effective_total_gpu_memory_fraction == 0.78
    assert contract.applied_encoder_mem_reserve == 0.0


def test_qwen_colocated_thinker_reserve_reduces_effective_ar_budget() -> None:
    overrides = {"disable_cuda_graph": False}

    contract = qwen_stages._apply_colocated_ar_memory_contract(
        overrides,
        stage_name="thinker",
        total_gpu_memory_fraction=0.75,
        encoder_mem_reserve=0.05,
    )

    assert overrides["mem_fraction_static"] == 0.70
    assert contract.effective_total_gpu_memory_fraction == 0.70
    assert contract.applied_encoder_mem_reserve == 0.05


def test_qwen_colocated_ar_explicit_matching_mem_fraction_keeps_stage_budget() -> None:
    overrides = {"mem_fraction_static": 0.75}

    contract = qwen_stages._apply_colocated_ar_memory_contract(
        overrides,
        stage_name="thinker",
        total_gpu_memory_fraction=0.75,
    )

    assert overrides["mem_fraction_static"] == 0.75
    assert contract.effective_total_gpu_memory_fraction == 0.75
    assert contract.applied_encoder_mem_reserve == 0.0


def test_qwen_ar_factory_rejects_conflicting_memory_contract() -> None:
    with pytest.raises(ValueError, match="conflicting colocated memory contracts"):
        qwen_stages._apply_colocated_ar_memory_contract(
            {"mem_fraction_static": 0.7},
            stage_name="thinker",
            total_gpu_memory_fraction=0.78,
        )


def test_qwen_colocated_thinker_startup_threads_effective_budget(
    monkeypatch,
    caplog,
) -> None:
    scheduler_calls = _patch_thinker_startup(monkeypatch)

    with caplog.at_level(logging.INFO, logger=qwen_stages.logger.name):
        qwen_stages.create_sglang_thinker_executor_from_config(
            "dummy",
            total_gpu_memory_fraction=0.75,
            encoder_mem_reserve=0.05,
        )

    assert scheduler_calls == [
        {
            "mem_fraction_static": 0.70,
            "sampling_backend": "pytorch",
            "gpu_id": 0,
            "total_gpu_memory_fraction": 0.70,
        }
    ]
    assert "total_gpu_memory_fraction=0.75" in caplog.text
    assert "effective_total_gpu_memory_fraction=0.7" in caplog.text
    assert "encoder_mem_reserve=0.05" in caplog.text


def test_qwen_colocated_thinker_explicit_mem_fraction_skips_default_reserve(
    monkeypatch,
    caplog,
) -> None:
    scheduler_calls = _patch_thinker_startup(monkeypatch)

    with caplog.at_level(logging.INFO, logger=qwen_stages.logger.name):
        qwen_stages.create_sglang_thinker_executor_from_config(
            "dummy",
            server_args_overrides={"mem_fraction_static": 0.75},
            total_gpu_memory_fraction=0.75,
        )

    assert scheduler_calls == [
        {
            "mem_fraction_static": 0.75,
            "sampling_backend": "pytorch",
            "gpu_id": 0,
            "total_gpu_memory_fraction": 0.75,
        }
    ]
    assert "effective_total_gpu_memory_fraction=0.75" in caplog.text
    assert "encoder_mem_reserve=0.0" in caplog.text


@pytest.mark.parametrize(
    ("server_args_overrides", "expected_max_bs"),
    [
        (None, 64),
        ({"max_running_requests": 16}, 16),
    ],
)
def test_qwen_thinker_threads_explicit_generation_batch_policy(
    monkeypatch,
    server_args_overrides,
    expected_max_bs,
) -> None:
    build_calls: list[dict[str, object]] = []
    validation_calls: list[tuple[str, object]] = []
    validate_generation_batch_policy = qwen_stages.validate_generation_batch_policy

    def _fake_server_args_builder(model_path, context_length, **overrides):
        assert model_path == "dummy"
        assert context_length == 8192
        build_calls.append(dict(overrides))
        return SimpleNamespace(
            mem_fraction_static=0.85,
            max_running_requests=overrides["max_running_requests"],
            cuda_graph_max_bs=overrides["cuda_graph_max_bs"],
            cuda_graph_bs=overrides["cuda_graph_bs"],
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(
                    max_bs=overrides["cuda_graph_max_bs"],
                    bs=overrides["cuda_graph_bs"],
                ),
                prefill=SimpleNamespace(backend="disabled", bs=None, max_bs=None),
            ),
            disable_cuda_graph=overrides["disable_cuda_graph"],
            enable_torch_compile=overrides.get("enable_torch_compile", False),
            torch_compile_max_bs=overrides["torch_compile_max_bs"],
        )

    def _validate_generation_batch_policy(*, model_name, server_args):
        validation_calls.append((model_name, server_args))
        validate_generation_batch_policy(
            model_name=model_name,
            server_args=server_args,
        )

    monkeypatch.setattr(
        qwen_stages,
        "build_sglang_server_args",
        _fake_server_args_builder,
    )
    monkeypatch.setattr(
        qwen_stages,
        "validate_generation_batch_policy",
        _validate_generation_batch_policy,
    )
    monkeypatch.setattr(
        qwen_stages,
        "create_thinker_scheduler",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(qwen_stages, "avail_gpu_mem", lambda gpu_id: 90.0)
    monkeypatch.setattr(
        qwen_stages,
        "get_process_gpu_memory_bytes",
        lambda gpu_id: None,
    )

    qwen_stages.create_sglang_thinker_executor_from_config(
        "dummy",
        server_args_overrides=server_args_overrides,
    )

    assert build_calls[-1]["max_running_requests"] == expected_max_bs
    assert build_calls[-1]["cuda_graph_max_bs"] == expected_max_bs
    assert max(build_calls[-1]["cuda_graph_bs"]) == expected_max_bs
    assert build_calls[-1]["torch_compile_max_bs"] == expected_max_bs
    assert validation_calls[-1][0] == "Qwen3-Omni thinker"
    assert validation_calls[-1][1].max_running_requests == expected_max_bs


def test_qwen_talker_ar_threads_explicit_generation_batch_policy(monkeypatch) -> None:
    build_calls: list[dict[str, object]] = []
    scheduler_calls: list[dict[str, object]] = []

    def _fake_server_args_builder(model_path, context_length, **overrides):
        assert model_path == "dummy"
        assert context_length == 4096
        build_calls.append(dict(overrides))
        return SimpleNamespace(
            mem_fraction_static=0.55,
            sampling_backend=overrides["sampling_backend"],
            max_running_requests=overrides["max_running_requests"],
            cuda_graph_max_bs=overrides["cuda_graph_max_bs"],
            cuda_graph_bs=overrides["cuda_graph_bs"],
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(
                    max_bs=overrides["cuda_graph_max_bs"],
                    bs=overrides["cuda_graph_bs"],
                ),
                prefill=SimpleNamespace(backend="disabled", bs=None, max_bs=None),
            ),
            disable_cuda_graph=overrides["disable_cuda_graph"],
            enable_torch_compile=overrides.get("enable_torch_compile", False),
            torch_compile_max_bs=overrides["torch_compile_max_bs"],
        )

    def _fake_create_talker_scheduler(server_args, gpu_id, **kwargs):
        scheduler_calls.append(
            {
                "gpu_id": gpu_id,
                "sampling_backend": server_args.sampling_backend,
                "max_running_requests": server_args.max_running_requests,
                "cuda_graph_max_bs": server_args.cuda_graph_max_bs,
                "cuda_graph_bs": server_args.cuda_graph_bs,
                "torch_compile_max_bs": server_args.torch_compile_max_bs,
                "weight_prefix": kwargs["weight_prefix"],
            }
        )
        return object()

    monkeypatch.setattr(
        qwen_stages,
        "build_sglang_server_args",
        _fake_server_args_builder,
    )
    monkeypatch.setattr(
        qwen_bootstrap,
        "create_talker_scheduler",
        _fake_create_talker_scheduler,
    )
    monkeypatch.setattr(qwen_stages, "avail_gpu_mem", lambda gpu_id: 90.0)
    monkeypatch.setattr(
        qwen_stages,
        "get_process_gpu_memory_bytes",
        lambda gpu_id: None,
    )

    qwen_stages.create_talker_ar_executor_from_config("dummy")

    assert build_calls == [
        {
            "cuda_graph_bs": [1, 2, 4, 8, 12, 16, 24, 32],
            "cuda_graph_max_bs": 32,
            "disable_cuda_graph": not current_platform.enable_talker_graph(),
            "max_running_requests": 32,
            "sampling_backend": "pytorch",
            "torch_compile_max_bs": 32,
            "tp_size": 1,
        }
    ]
    assert scheduler_calls == [
        {
            "gpu_id": 0,
            "sampling_backend": "pytorch",
            "max_running_requests": 32,
            "cuda_graph_max_bs": 32,
            "cuda_graph_bs": [1, 2, 4, 8, 12, 16, 24, 32],
            "torch_compile_max_bs": 32,
            "weight_prefix": "talker.",
        }
    ]


def test_talker_ar_default_running_batch_width_is_32(monkeypatch) -> None:
    """talker_ar default max_running_requests is 32; a config override still wins."""
    captured: list[dict[str, object]] = []

    def _fake_builder(model_path, context_length, **overrides):
        captured.append(dict(overrides))
        return SimpleNamespace(
            mem_fraction_static=overrides.get("mem_fraction_static"),
            max_running_requests=overrides["max_running_requests"],
            cuda_graph_max_bs=overrides["cuda_graph_max_bs"],
            cuda_graph_bs=overrides["cuda_graph_bs"],
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(
                    max_bs=overrides["cuda_graph_max_bs"],
                    bs=overrides["cuda_graph_bs"],
                ),
                prefill=SimpleNamespace(backend="disabled", bs=None, max_bs=None),
            ),
            disable_cuda_graph=overrides["disable_cuda_graph"],
            enable_torch_compile=overrides.get("enable_torch_compile", False),
            torch_compile_max_bs=overrides["torch_compile_max_bs"],
        )

    monkeypatch.setattr(qwen_stages, "build_sglang_server_args", _fake_builder)
    monkeypatch.setattr(
        qwen_bootstrap, "create_talker_scheduler", lambda *a, **k: object()
    )
    monkeypatch.setattr(qwen_stages, "avail_gpu_mem", lambda gpu_id: 90.0)
    monkeypatch.setattr(
        qwen_stages, "get_process_gpu_memory_bytes", lambda gpu_id: None
    )

    qwen_stages.create_talker_ar_executor_from_config("dummy")
    assert captured[-1]["max_running_requests"] == 32

    qwen_stages.create_talker_ar_executor_from_config(
        "dummy", server_args_overrides={"max_running_requests": 8}
    )
    assert captured[-1]["max_running_requests"] == 8


def _thinker_overrides(monkeypatch, *, allows: bool, **kwargs) -> dict[str, object]:
    captured: list[dict[str, object]] = []
    _patch_thinker_startup(monkeypatch)
    inner = qwen_stages.build_sglang_server_args

    def _recording_builder(model_path, context_length, **overrides):
        captured.append(dict(overrides))
        return inner(model_path, context_length, **overrides)

    monkeypatch.setattr(qwen_stages, "build_sglang_server_args", _recording_builder)
    monkeypatch.setattr(current_platform, "enable_thinker_decode_graph", lambda: allows)
    qwen_stages.create_sglang_thinker_executor_from_config(
        "dummy", total_gpu_memory_fraction=0.75, **kwargs
    )
    return captured[-1]


@pytest.mark.parametrize("allows, expected", [(False, True), (True, None)])
def test_thinker_decode_graph_follows_the_platform_gate(
    monkeypatch, allows: bool, expected: bool | None
) -> None:
    overrides = _thinker_overrides(monkeypatch, allows=allows)

    assert overrides.get("disable_decode_cuda_graph") is expected
    # The gate reaches for the decode key only, never the global switch.
    assert overrides["disable_cuda_graph"] is False


def test_thinker_decode_graph_stays_available_to_an_explicit_override(
    monkeypatch,
) -> None:
    overrides = _thinker_overrides(
        monkeypatch,
        allows=False,
        server_args_overrides={"disable_decode_cuda_graph": False},
    )

    assert overrides["disable_decode_cuda_graph"] is False


def test_importing_the_stages_module_does_not_load_the_platform_layer() -> None:
    """Read in a subprocess: the suite loads the platform layer long before this."""
    probe = (
        "import sys;"
        "import sglang_omni.models.qwen3_omni.stages;"
        "print('LOADED' if 'sglang_omni.platforms' in sys.modules else 'ABSENT')"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == "ABSENT"


def _talker_overrides(monkeypatch, *, allows: bool, **kwargs) -> dict[str, object]:
    captured: list[dict[str, object]] = []

    def _recording_builder(model_path, context_length, **overrides):
        captured.append(dict(overrides))
        return SimpleNamespace(
            mem_fraction_static=overrides.get("mem_fraction_static"),
            sampling_backend=overrides["sampling_backend"],
            max_running_requests=overrides["max_running_requests"],
            cuda_graph_max_bs=overrides["cuda_graph_max_bs"],
            cuda_graph_bs=overrides["cuda_graph_bs"],
            torch_compile_max_bs=overrides["torch_compile_max_bs"],
            disable_cuda_graph=overrides["disable_cuda_graph"],
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(
                    max_bs=overrides["cuda_graph_max_bs"],
                    bs=overrides["cuda_graph_bs"],
                ),
                prefill=SimpleNamespace(backend="disabled", bs=None, max_bs=None),
            ),
            enable_torch_compile=overrides.get("enable_torch_compile", False),
        )

    monkeypatch.setattr(qwen_stages, "build_sglang_server_args", _recording_builder)
    monkeypatch.setattr(
        qwen_bootstrap, "create_talker_scheduler", lambda *a, **k: object()
    )
    monkeypatch.setattr(qwen_stages, "avail_gpu_mem", lambda gpu_id: 90.0)
    monkeypatch.setattr(
        qwen_stages, "get_process_gpu_memory_bytes", lambda gpu_id: None
    )
    monkeypatch.setattr(current_platform, "enable_talker_graph", lambda: allows)

    qwen_stages.create_talker_ar_executor_from_config("dummy", **kwargs)
    return captured[-1]


@pytest.mark.parametrize("allows, expected", [(False, True), (True, False)])
def test_talker_decode_graph_follows_the_platform_gate(
    monkeypatch, allows: bool, expected: bool
) -> None:
    overrides = _talker_overrides(monkeypatch, allows=allows)

    assert overrides["disable_cuda_graph"] is expected


def test_talker_graph_stays_available_to_an_explicit_override(monkeypatch) -> None:
    """The stage note promises engine.disable_cuda_graph wins."""
    overrides = _talker_overrides(
        monkeypatch,
        allows=False,
        server_args_overrides={"disable_cuda_graph": False},
    )

    assert overrides["disable_cuda_graph"] is False
