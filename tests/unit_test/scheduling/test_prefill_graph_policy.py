# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the breakable prefill CUDA graph policy and wiring."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest
from sglang.srt.server_args import ServerArgs

from sglang_omni.scheduling.generation_batch_policy import (
    build_default_prefill_cuda_graph_bs,
    build_generation_batch_overrides,
    validate_generation_batch_policy,
)
from sglang_omni.vendor.sglang.server_args import override_server_args


def _server_args(
    *,
    prefill_backend: str = "disabled",
    prefill_bs: Any = None,
    prefill_max_bs: Any = None,
    locked: frozenset = frozenset(),
    chunked_prefill_size: int | None = 8192,
    max_prefill_tokens: int = 16384,
    disable_cuda_graph: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        max_running_requests=4,
        disable_cuda_graph=disable_cuda_graph,
        enable_torch_compile=False,
        torch_compile_max_bs=None,
        chunked_prefill_size=chunked_prefill_size,
        max_prefill_tokens=max_prefill_tokens,
        tp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        lora_paths=None,
        enable_lora=None,
        moe_a2a_backend="none",
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(max_bs=4, bs=(1, 2, 4)),
            prefill=SimpleNamespace(
                backend=prefill_backend,
                bs=prefill_bs,
                max_bs=prefill_max_bs,
            ),
        ),
        _cuda_graph_config_locked=locked,
    )


_PREFILL_BS_LOCKED = frozenset({("prefill", "bs")})


def _validate(server_args: SimpleNamespace) -> None:
    validate_generation_batch_policy(
        model_name="Test TTS",
        server_args=server_args,
        model_buffer_bs=None,
    )


def test_prefill_policy_accepts_disabled_and_declared_breakable() -> None:
    _validate(_server_args())
    _validate(
        _server_args(
            prefill_backend="breakable",
            prefill_bs=(128, 256, 512),
            prefill_max_bs=512,
        )
    )


def test_breakable_requires_cuda_graphs_enabled() -> None:
    with pytest.raises(ValueError, match="require CUDA graphs enabled"):
        _validate(
            _server_args(
                prefill_backend="breakable",
                prefill_bs=(128,),
                disable_cuda_graph=True,
            )
        )


def test_non_breakable_prefill_backend_is_rejected() -> None:
    for backend in ("full", "tc_piecewise"):
        with pytest.raises(ValueError, match="must be 'breakable'"):
            _validate(
                _server_args(
                    prefill_backend=backend,
                    prefill_bs=(128,),
                )
            )


def test_breakable_accepts_a_derived_ladder(caplog) -> None:
    server_args = _server_args(
        prefill_backend="breakable",
        prefill_bs=build_default_prefill_cuda_graph_bs(8192),
        prefill_max_bs=8192,
        chunked_prefill_size=8192,
    )

    with caplog.at_level(logging.INFO):
        _validate(server_args)

    assert not caplog.records


def test_breakable_rejects_non_increasing_buckets() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        _validate(
            _server_args(
                prefill_backend="breakable",
                prefill_bs=(256, 128),
            )
        )


@pytest.mark.parametrize("buckets", [(128.0, 256), ("128", 256), (True, 256)])
def test_breakable_rejects_non_integral_bucket_types(
    buckets: tuple[Any, int],
) -> None:
    with pytest.raises(ValueError, match="sequence of positive integers"):
        _validate(
            _server_args(
                prefill_backend="breakable",
                prefill_bs=buckets,
            )
        )


def test_prefill_max_bs_above_the_top_bucket_is_accepted(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        _validate(
            _server_args(
                prefill_backend="breakable",
                prefill_bs=(128, 256),
                prefill_max_bs=512,
            )
        )

    assert not caplog.records


def test_buckets_above_the_chunk_warn_and_stay(caplog) -> None:
    server_args = _server_args(
        prefill_backend="breakable",
        prefill_bs=(4096, 12288),
        chunked_prefill_size=8192,
    )

    with caplog.at_level(logging.WARNING):
        _validate(server_args)

    assert server_args.cuda_graph_config.prefill.bs == (4096, 12288)
    assert "max=12288 exceeds chunked_prefill_size=8192" in caplog.text


@pytest.mark.parametrize(
    "server_args_kwargs",
    [
        {
            "prefill_bs": (4096, 8192),
            "prefill_max_bs": 8192,
            "chunked_prefill_size": 8192,
            "max_prefill_tokens": 4096,
        },
        {
            "prefill_bs": (16384, 32768),
            "prefill_max_bs": 32768,
            "chunked_prefill_size": -1,
            "max_prefill_tokens": 16384,
        },
        {
            "prefill_bs": (2048, 4096),
            "prefill_max_bs": 4096,
            "chunked_prefill_size": 0,
            "max_prefill_tokens": 2048,
        },
    ],
)
def test_only_the_chunk_bounds_a_prefill_forward(
    server_args_kwargs: dict[str, Any], caplog
) -> None:
    with caplog.at_level(logging.WARNING):
        _validate(_server_args(prefill_backend="breakable", **server_args_kwargs))

    assert not caplog.records


def test_top_bucket_equal_to_the_chunk_is_silent(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        _validate(
            _server_args(
                prefill_backend="breakable",
                prefill_bs=(2048, 4096),
                prefill_max_bs=8192,
                chunked_prefill_size=4096,
            )
        )

    assert not caplog.records


def test_empty_derived_ladder_passes_with_a_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        _validate(
            _server_args(
                prefill_backend="breakable",
                prefill_bs=[],
                prefill_max_bs=-1,
                chunked_prefill_size=-1,
            )
        )

    assert "require a positive prefill graph cap" in caplog.text


@pytest.mark.parametrize(
    ("incompatibility", "match"),
    [
        ({"attn_cp_size": 2}, "attn_cp_size"),
        ({"dcp_size": 2}, "dcp_size"),
        ({"lora_paths": ["adapter"]}, "LoRA"),
        ({"enable_lora": True}, "LoRA"),
        ({"moe_a2a_backend": "deepep"}, "MoE A2A"),
    ],
)
def test_breakable_rejects_sglang_incompatible_features(
    incompatibility: dict[str, Any], match: str
) -> None:
    server_args = _server_args(
        prefill_backend="breakable",
        prefill_bs=(128, 256, 512),
        prefill_max_bs=512,
    )
    override_server_args(server_args, "test", **incompatibility)

    with pytest.raises(ValueError, match=match):
        _validate(server_args)


def test_nested_prefill_bs_composes_as_operator_buckets() -> None:
    overrides = build_generation_batch_overrides(
        max_running_requests=4,
        server_args_overrides={
            "cuda_graph_config": {"prefill": {"backend": "breakable", "bs": [128, 256]}}
        },
    )

    assert overrides["cuda_graph_backend_prefill"] == "breakable"
    assert overrides["cuda_graph_bs_prefill"] == [128, 256]
    assert overrides["cuda_graph_max_bs_prefill"] == 256


def test_nested_prefill_max_bs_composes_as_a_flat_cap() -> None:
    overrides = build_generation_batch_overrides(
        max_running_requests=4,
        server_args_overrides={"cuda_graph_config": {"prefill": {"max_bs": 512}}},
    )

    assert overrides["cuda_graph_max_bs_prefill"] == 512
    assert "cuda_graph_bs_prefill" not in overrides


def test_breakable_prefill_cap_builds_the_shared_default_ladder() -> None:
    overrides = build_generation_batch_overrides(
        max_running_requests=4,
        server_args_overrides={
            "cuda_graph_backend_prefill": "breakable",
            "cuda_graph_max_bs_prefill": 512,
        },
    )

    assert overrides["cuda_graph_max_bs_prefill"] == 512
    assert overrides["cuda_graph_bs_prefill"] == build_default_prefill_cuda_graph_bs(
        512
    )


def test_prefill_cap_is_not_clamped_by_max_prefill_tokens() -> None:
    overrides = build_generation_batch_overrides(
        max_running_requests=4,
        server_args_overrides={
            "cuda_graph_backend_prefill": "breakable",
            "cuda_graph_max_bs_prefill": 2048,
            "max_prefill_tokens": 768,
        },
    )

    assert overrides["cuda_graph_max_bs_prefill"] == 2048
    assert overrides["cuda_graph_bs_prefill"][-1] == 2048


@pytest.mark.parametrize(
    ("server_args_overrides", "expected_cap"),
    [
        ({"chunked_prefill_size": 4096}, 4096),
        ({"chunked_prefill_size": 4592}, 4592),
        ({"chunked_prefill_size": 4096, "max_total_tokens": 1000}, 1000),
        ({"chunked_prefill_size": 4096, "max_total_tokens": 8192}, 4096),
        ({"cuda_graph_max_bs_prefill": 3000, "max_total_tokens": 1000}, 3000),
    ],
)
def test_explicit_prefill_caps_build_the_ladder_before_server_args(
    server_args_overrides: dict[str, Any], expected_cap: int
) -> None:
    overrides = build_generation_batch_overrides(
        max_running_requests=4,
        server_args_overrides={
            "cuda_graph_backend_prefill": "breakable",
            **server_args_overrides,
        },
    )

    assert overrides["cuda_graph_max_bs_prefill"] == expected_cap
    assert overrides["cuda_graph_bs_prefill"] == (
        build_default_prefill_cuda_graph_bs(expected_cap)
    )


@pytest.mark.parametrize(
    "server_args_overrides",
    [
        {},
        {"chunked_prefill_size": None},
        {"chunked_prefill_size": None, "max_total_tokens": 3000},
        {"chunked_prefill_size": 0},
        {"chunked_prefill_size": -1},
        {"cuda_graph_max_bs_prefill": 0},
        {"cuda_graph_backend_prefill": "disabled", "chunked_prefill_size": 4096},
    ],
)
def test_unset_or_disabled_prefill_caps_leave_the_ladder_to_sglang(
    server_args_overrides: dict[str, Any],
) -> None:
    overrides = build_generation_batch_overrides(
        max_running_requests=4,
        server_args_overrides={
            "cuda_graph_backend_prefill": "breakable",
            **server_args_overrides,
        },
    )

    assert "cuda_graph_bs_prefill" not in overrides


@pytest.mark.parametrize(("cap", "expected_top"), [(128, 128), (100, 100)])
def test_nested_prefill_max_bs_trims_a_stage_default_ladder(
    cap: int, expected_top: int
) -> None:
    overrides = build_generation_batch_overrides(
        max_running_requests=4,
        cuda_graph_bs_prefill=build_default_prefill_cuda_graph_bs(512),
        server_args_overrides={"cuda_graph_config": {"prefill": {"max_bs": cap}}},
    )

    assert overrides["cuda_graph_max_bs_prefill"] == cap
    assert overrides["cuda_graph_bs_prefill"][-1] == expected_top
    assert all(bucket <= cap for bucket in overrides["cuda_graph_bs_prefill"])


def test_operator_prefill_cap_below_its_own_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="below the declared cuda_graph_bs_prefill"):
        build_generation_batch_overrides(
            max_running_requests=4,
            server_args_overrides={
                "cuda_graph_bs_prefill": [128, 256],
                "cuda_graph_config": {"prefill": {"max_bs": 128}},
            },
        )


def test_operator_prefill_cap_above_its_list_is_kept() -> None:
    overrides = build_generation_batch_overrides(
        max_running_requests=4,
        server_args_overrides={
            "cuda_graph_bs_prefill": [128, 256],
            "cuda_graph_max_bs_prefill": 512,
        },
    )

    assert overrides["cuda_graph_bs_prefill"] == [128, 256]
    assert overrides["cuda_graph_max_bs_prefill"] == 512


def test_nested_disabled_backend_overrides_a_stage_default() -> None:
    overrides = build_generation_batch_overrides(
        max_running_requests=4,
        cuda_graph_backend_prefill="breakable",
        server_args_overrides={
            "cuda_graph_config": {"prefill": {"backend": "disabled"}}
        },
    )

    assert overrides["cuda_graph_backend_prefill"] == "disabled"


def test_conflicting_flat_and_nested_prefill_overrides_are_rejected() -> None:
    with pytest.raises(ValueError, match="Conflicting"):
        build_generation_batch_overrides(
            max_running_requests=4,
            server_args_overrides={
                "cuda_graph_bs_prefill": [128],
                "cuda_graph_config": {"prefill": {"bs": [128, 256]}},
            },
        )


def test_breakable_accepts_disabled_chunked_prefill(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        _validate(
            _server_args(
                prefill_backend="breakable",
                prefill_bs=(4096, 8192),
                prefill_max_bs=-1,
                chunked_prefill_size=-1,
            )
        )

    assert not caplog.records


def test_breakable_warns_on_padding_factor_gaps(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        _validate(
            _server_args(
                prefill_backend="breakable",
                prefill_bs=(64, 512),
            )
        )

    assert any("padding factor" in record.message for record in caplog.records)


def test_padding_gap_warning_requires_a_non_empty_eager_range(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        _validate(
            _server_args(
                prefill_backend="breakable",
                prefill_bs=(64, 129),
            )
        )
    assert not any("padding factor" in record.message for record in caplog.records)

    with caplog.at_level(logging.WARNING):
        _validate(
            _server_args(
                prefill_backend="breakable",
                prefill_bs=(64, 131),
            )
        )
    assert any("[(65, 65)]" in record.getMessage() for record in caplog.records)


def _sglang_prefill_ladder(max_bs: int) -> list[int]:
    unresolved = ServerArgs.__new__(ServerArgs)
    return ServerArgs._generate_prefill_cuda_graph_batch_sizes(unresolved, max_bs)


@pytest.mark.parametrize("cap", [256, 512, 2048, 4096, 8192, 16384])
def test_default_prefill_ladder_matches_the_sglang_generator_on_grid(
    cap: int,
) -> None:
    assert build_default_prefill_cuda_graph_bs(cap) == _sglang_prefill_ladder(cap)


@pytest.mark.parametrize("cap", [2, 100, 1000, 4592])
def test_default_prefill_ladder_appends_the_off_grid_cap_sglang_omits(
    cap: int,
) -> None:
    sglang_ladder = _sglang_prefill_ladder(cap)

    assert not sglang_ladder or sglang_ladder[-1] < cap
    assert build_default_prefill_cuda_graph_bs(cap) == [*sglang_ladder, cap]


def test_overrides_derive_prefill_max_bs_from_buckets() -> None:
    overrides = build_generation_batch_overrides(
        max_running_requests=4,
        server_args_overrides={
            "cuda_graph_backend_prefill": "breakable",
            "cuda_graph_bs_prefill": [128, 256],
        },
    )

    assert overrides["cuda_graph_backend_prefill"] == "breakable"
    assert overrides["cuda_graph_max_bs_prefill"] == 256

    explicit = build_generation_batch_overrides(
        max_running_requests=4,
        server_args_overrides={
            "cuda_graph_bs_prefill": [128, 256],
            "cuda_graph_max_bs_prefill": 256,
        },
    )

    assert explicit["cuda_graph_max_bs_prefill"] == 256


def test_disable_overrides_win_over_default_prefill_backend() -> None:
    stage_defaults = {
        "cuda_graph_backend_prefill": "breakable",
        "cuda_graph_bs_prefill": [128, 256],
    }

    for disable_key in ("disable_cuda_graph", "disable_prefill_cuda_graph"):
        overrides = build_generation_batch_overrides(
            max_running_requests=4,
            server_args_overrides={disable_key: True},
            **stage_defaults,
        )
        assert overrides["cuda_graph_backend_prefill"] == "disabled"
        assert "cuda_graph_bs_prefill" not in overrides
        assert "cuda_graph_max_bs_prefill" not in overrides

    explicit_disable = build_generation_batch_overrides(
        max_running_requests=4,
        server_args_overrides={"cuda_graph_backend_prefill": "disabled"},
        **stage_defaults,
    )
    assert explicit_disable["cuda_graph_backend_prefill"] == "disabled"

    explicit_conflict = build_generation_batch_overrides(
        max_running_requests=4,
        server_args_overrides={
            "disable_cuda_graph": True,
            "cuda_graph_backend_prefill": "breakable",
        },
        **stage_defaults,
    )
    assert explicit_conflict["cuda_graph_backend_prefill"] == "breakable"


def test_builder_wires_payload_slot_and_attestation(monkeypatch) -> None:
    from sglang_omni.scheduling import bootstrap, sglang_backend
    from sglang_omni.scheduling.engine_factory import TtsEngineBuilder
    from sglang_omni.utils import cuda_graph_batch_validator

    infra_kwargs_seen: list[dict[str, Any]] = []
    attest_calls: list[tuple[Any, Any]] = []
    backend_locks_seen: list[bool] = []

    def fake_build_sglang_server_args(checkpoint_dir, *, context_length, **overrides):
        del checkpoint_dir, context_length
        prefill_backend = overrides.get("cuda_graph_backend_prefill", "disabled")
        prefill_bs = overrides.get("cuda_graph_bs_prefill")
        locked = set(_PREFILL_BS_LOCKED if prefill_bs else ())
        if "cuda_graph_backend_prefill" in overrides:
            locked.add(("prefill", "backend"))
        return _server_args(
            prefill_backend=prefill_backend,
            prefill_bs=prefill_bs,
            prefill_max_bs=overrides.get("cuda_graph_max_bs_prefill"),
            locked=locked,
        )

    def fake_create_sglang_infrastructure(server_args, gpu_id, **kwargs):
        del gpu_id
        infra_kwargs_seen.append(dict(kwargs))
        backend_locks_seen.append(
            ("prefill", "backend") in server_args._cuda_graph_config_locked
        )
        model_runner = SimpleNamespace(
            model=SimpleNamespace(),
            init_cuda_graphs=lambda: None,
        )
        return (
            SimpleNamespace(
                model_runner=model_runner,
                model_config=SimpleNamespace(is_multimodal=False),
                enable_prefill_input_embeds=bool(
                    kwargs.get("enable_prefill_input_embeds")
                ),
            ),
            "tree_cache",
            "req_pool",
            "kv_pool",
            "model_config",
        )

    def fake_attest(model_runner, server_args) -> None:
        attest_calls.append((model_runner, server_args))

    monkeypatch.setattr(
        sglang_backend, "build_sglang_server_args", fake_build_sglang_server_args
    )
    monkeypatch.setattr(
        bootstrap, "create_sglang_infrastructure", fake_create_sglang_infrastructure
    )
    monkeypatch.setattr(
        sglang_backend, "SGLangOutputProcessor", lambda **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(
        cuda_graph_batch_validator, "attest_prefill_cuda_graphs", fake_attest
    )

    class PolicyBuilder(TtsEngineBuilder):
        model_name = "Test TTS"
        context_length = 123
        supports_breakable_prefill_cuda_graph = True

        def resolve_checkpoint(self, model_path: str) -> str:
            return model_path

        def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
            del dtype
            return {"max_running_requests": 4}

        def setup_model(self, **kwargs: Any) -> None:
            del kwargs

        def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
            del output_proc
            return model_worker

        def make_adapters(self, model: Any) -> tuple[Any, Any]:
            del model
            return object(), object()

        def make_scheduler(self, **kwargs: Any) -> Any:
            return SimpleNamespace(outbox=None, kwargs=kwargs)

    PolicyBuilder().build(
        "model",
        server_args_overrides={
            "cuda_graph_backend_prefill": "breakable",
            "cuda_graph_bs_prefill": [128, 256],
        },
    )

    assert infra_kwargs_seen[-1]["enable_prefill_input_embeds"] is True
    assert backend_locks_seen[-1] is True
    assert len(attest_calls) == 1

    PolicyBuilder().build("model")

    assert "enable_prefill_input_embeds" not in infra_kwargs_seen[-1]
    assert backend_locks_seen[-1] is False
    assert len(attest_calls) == 1


def test_builder_rejects_breakable_without_model_opt_in(monkeypatch) -> None:
    from sglang_omni.scheduling import sglang_backend
    from sglang_omni.scheduling.engine_factory import TtsEngineBuilder

    def fake_build_sglang_server_args(checkpoint_dir, *, context_length, **overrides):
        del checkpoint_dir, context_length
        return _server_args(
            prefill_backend="breakable",
            prefill_bs=overrides.get("cuda_graph_bs_prefill"),
            locked=_PREFILL_BS_LOCKED,
        )

    monkeypatch.setattr(
        sglang_backend, "build_sglang_server_args", fake_build_sglang_server_args
    )

    class NonAdoptingBuilder(TtsEngineBuilder):
        model_name = "Test TTS"
        context_length = 123

        def resolve_checkpoint(self, model_path: str) -> str:
            return model_path

        def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
            del dtype
            return {"max_running_requests": 4}

        def setup_model(self, **kwargs: Any) -> None:
            del kwargs

        def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
            del output_proc
            return model_worker

        def make_adapters(self, model: Any) -> tuple[Any, Any]:
            del model
            return object(), object()

    with pytest.raises(RuntimeError, match="has not adopted the breakable prefill"):
        NonAdoptingBuilder().build(
            "model",
            server_args_overrides={
                "cuda_graph_backend_prefill": "breakable",
                "cuda_graph_bs_prefill": [128, 256],
            },
        )


def test_raised_operator_cap_extends_a_stage_ladder_without_dropping_buckets() -> None:
    """A raised cap must grow a stage ladder, not replace it with the shared one.

    The stage ladders carry buckets the shared one does not, such as the
    Qwen3-TTS 1-token bucket, and losing them sends those shapes back to eager.
    """
    stage_ladder = [1, 2] + build_default_prefill_cuda_graph_bs(512)
    overrides = build_generation_batch_overrides(
        max_running_requests=4,
        cuda_graph_backend_prefill="breakable",
        cuda_graph_bs_prefill=list(stage_ladder),
        server_args_overrides={"cuda_graph_max_bs_prefill": 1024},
    )

    result = overrides["cuda_graph_bs_prefill"]
    assert overrides["cuda_graph_max_bs_prefill"] == 1024
    # Every stage bucket survives, including the ones the shared ladder lacks.
    assert set(stage_ladder) <= set(result)
    assert 1 in result and 2 in result
    assert max(result) == 1024
    assert result == sorted(result)
    assert len(result) == len(set(result))


def test_raised_operator_cap_leaves_an_operator_declared_ladder_alone() -> None:
    """When the operator declares both, they own the pair."""
    overrides = build_generation_batch_overrides(
        max_running_requests=4,
        server_args_overrides={
            "cuda_graph_backend_prefill": "breakable",
            "cuda_graph_bs_prefill": [128, 256],
            "cuda_graph_max_bs_prefill": 1024,
        },
    )

    assert overrides["cuda_graph_bs_prefill"] == [128, 256]
    assert overrides["cuda_graph_max_bs_prefill"] == 1024
