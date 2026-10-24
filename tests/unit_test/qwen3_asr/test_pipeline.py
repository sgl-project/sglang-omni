# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from sglang.srt.arg_groups.cuda_graph_hook import (
    generate_prefill_cuda_graph_batch_sizes,
)

import sglang_omni.model_runner.base as model_runner_base
import sglang_omni.models.qwen3_asr.engine_builder as qwen3_asr_builder
import sglang_omni.models.qwen3_asr.stages as qwen3_asr_stages
import sglang_omni.scheduling.bootstrap as bootstrap
import sglang_omni.scheduling.omni_scheduler as omni_scheduler
import sglang_omni.scheduling.sglang_backend as sglang_backend
import sglang_omni.utils.cuda_graph_batch_validator as cuda_graph_batch_validator
import sglang_omni.utils.device as device_utils
from sglang_omni.models.qwen3_asr import request_builders
from sglang_omni.models.qwen3_asr.config import Qwen3ASRPipelineConfig
from sglang_omni.models.qwen3_asr.stages import create_sglang_qwen3_asr_executor
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.scheduling.generation_batch_policy import (
    build_default_cuda_graph_bs,
    build_generation_batch_overrides,
    validate_generation_batch_policy,
)


@pytest.fixture(autouse=True)
def select_cpu_platform_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qwen3_asr_builder.current_platform, "is_cuda", lambda: False)


def sglang_prefill_ladder(max_bs: int) -> list[int]:
    return generate_prefill_cuda_graph_batch_sizes(max_bs)


def fake_server_args_builder(
    build_kwargs: dict[str, object],
    *,
    resolved_chunked_prefill_size: int = 8192,
):
    def build(model_path, context_length, **overrides):
        del model_path
        build_kwargs.update(overrides)
        flat = {
            key: value
            for key, value in overrides.items()
            if not key.startswith("cuda_graph_")
        }
        for name, default in (
            ("attn_cp_size", 1),
            ("dcp_size", 1),
            ("lora_paths", None),
            ("enable_lora", None),
            ("moe_a2a_backend", "none"),
            ("max_prefill_tokens", 16384),
            ("max_total_tokens", None),
            ("attention_backend", "fa3"),
            ("prefill_attention_backend", None),
            ("decode_attention_backend", None),
        ):
            flat.setdefault(name, default)
        if flat.get("chunked_prefill_size") is None:
            flat["chunked_prefill_size"] = resolved_chunked_prefill_size
        server_args = SimpleNamespace(context_length=context_length, **flat)
        prefill_bs = overrides.get("cuda_graph_bs_prefill")
        prefill_max_bs = overrides.get("cuda_graph_max_bs_prefill")
        locked = set()
        if "cuda_graph_backend_prefill" in overrides:
            locked.add(("prefill", "backend"))
        if prefill_bs is not None:
            locked.add(("prefill", "bs"))
        if prefill_max_bs is not None:
            locked.add(("prefill", "max_bs"))
        if prefill_max_bs is None:
            prefill_max_bs = server_args.chunked_prefill_size
            if server_args.max_total_tokens is not None:
                prefill_max_bs = min(prefill_max_bs, server_args.max_total_tokens)
        if prefill_bs is None:
            prefill_bs = sglang_prefill_ladder(prefill_max_bs)
        server_args.cuda_graph_config = SimpleNamespace(
            decode=SimpleNamespace(
                max_bs=overrides["cuda_graph_max_bs"],
                bs=overrides["cuda_graph_bs"],
            ),
            prefill=SimpleNamespace(
                backend=overrides.get("cuda_graph_backend_prefill", "disabled"),
                bs=prefill_bs,
                max_bs=prefill_max_bs,
            ),
        )
        server_args._cuda_graph_config_locked = (
            locked  # noqa: leading-underscore  # upstream name
        )
        return server_args

    return build


def make_engine_builder(
    *, mm_attention_backend: str | None = None, context_length: int = 1636
) -> qwen3_asr_builder.Qwen3ASREngineBuilder:
    builder = qwen3_asr_builder.Qwen3ASREngineBuilder(
        max_running_requests=64,
        max_new_tokens=128,
        enable_async_decode=True,
        async_decode_min_batch_size=2,
        mem_fraction_static=None,
        mm_embedding_cache_size_bytes=0,
        enable_torch_compile=False,
        torch_compile_max_bs=1,
        mm_attention_backend=mm_attention_backend,
        request_build_max_workers=8,
        request_build_max_pending=32,
        prefill_coalesce_requests=16,
        prefill_coalesce_wait_ms=24.0,
        prefill_coalesce_when_idle=True,
        prefill_coalesce_requires_pending_builds=True,
        prefill_coalesce_after_builds_during_decode=True,
        stream_emit_interval_s=0.05,
    )
    builder.context_length = context_length
    return builder


def test_qwen3_asr_engine_builder_binds_encode_wait_policy() -> None:
    builder = make_engine_builder()
    assert builder.should_wait_for_encode() is False

    builder.post_scheduler_setup(
        SimpleNamespace(request_build_queue_fits_workers=lambda: True),
        object(),
    )

    assert builder.should_wait_for_encode() is True


@pytest.mark.parametrize(
    ("sm_version", "expected_backend"),
    [(89, None), (100, "triton_attn"), (120, "triton_attn")],
)
def test_qwen3_asr_default_mm_attention_backend_by_sm(
    monkeypatch: pytest.MonkeyPatch,
    sm_version: int,
    expected_backend: str | None,
) -> None:
    queried_gpu_ids: list[int] = []
    monkeypatch.setattr(
        qwen3_asr_builder,
        "get_visible_gpu_sm_version",
        lambda gpu_id: queried_gpu_ids.append(gpu_id) or sm_version,
        raising=False,
    )
    builder = make_engine_builder()
    builder.gpu_id = 3

    defaults = builder.generation_defaults(dtype="bfloat16")

    assert queried_gpu_ids == [3]
    if expected_backend is None:
        assert "mm_attention_backend" not in defaults
    else:
        assert defaults["mm_attention_backend"] == expected_backend


def test_qwen3_asr_explicit_mm_attention_backend_overrides_sm_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qwen3_asr_builder,
        "get_visible_gpu_sm_version",
        lambda gpu_id: pytest.fail(f"unexpected SM lookup for GPU {gpu_id}"),
        raising=False,
    )
    builder = make_engine_builder(mm_attention_backend="fa3")
    builder.gpu_id = 0

    defaults = builder.generation_defaults(dtype="bfloat16")

    assert defaults["mm_attention_backend"] == "fa3"


@pytest.mark.parametrize(
    ("is_rocm", "gfx95_supported", "hip_version", "expected_backend"),
    [
        (True, True, (7, 2, 0), "triton"),
        (False, True, (7, 2, 0), None),
        (True, False, (7, 2, 0), None),
        (True, True, (7, 1, 0), None),
        (True, True, (7, 3, 0), None),
    ],
)
def test_qwen3_asr_defaults_prefill_to_triton_only_on_rocm_72_gfx95(
    monkeypatch: pytest.MonkeyPatch,
    is_rocm: bool,
    gfx95_supported: bool,
    hip_version: tuple[int, int, int],
    expected_backend: str | None,
) -> None:
    monkeypatch.setattr(
        qwen3_asr_builder.current_platform,
        "is_rocm",
        lambda: is_rocm,
    )
    monkeypatch.setattr(
        qwen3_asr_builder,
        "is_gfx95_supported",
        lambda: gfx95_supported,
    )
    monkeypatch.setattr(
        qwen3_asr_builder,
        "get_hip_version",
        lambda: hip_version,
    )

    defaults = make_engine_builder(mm_attention_backend="fa3").generation_defaults(
        dtype="bfloat16"
    )

    if expected_backend is None:
        assert "prefill_attention_backend" not in defaults
    else:
        assert defaults["prefill_attention_backend"] == expected_backend


def test_qwen3_asr_explicit_prefill_backend_overrides_rocm_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qwen3_asr_builder.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(qwen3_asr_builder, "is_gfx95_supported", lambda: True)
    monkeypatch.setattr(qwen3_asr_builder, "get_hip_version", lambda: (7, 2, 0))
    builder = make_engine_builder(mm_attention_backend="fa3")

    overrides = build_generation_batch_overrides(
        **builder.generation_defaults(dtype="bfloat16"),
        server_args_overrides={"prefill_attention_backend": "aiter"},
    )

    assert overrides["prefill_attention_backend"] == "aiter"


def test_qwen3_asr_torch_mps_uses_eager_native_profile() -> None:
    builder = make_engine_builder()
    builder.device = "mps"

    defaults = builder.generation_defaults(dtype="float16")

    assert defaults == {
        "max_running_requests": 1,
        "disable_cuda_graph": True,
        "disable_overlap_schedule": True,
        "disable_radix_cache": True,
        "enable_torch_compile": False,
        "context_length": 2048,
        "max_total_tokens": 2048,
        "max_prefill_tokens": 2048,
        "chunked_prefill_size": -1,
        "mem_fraction_static": None,
        "attention_backend": "torch_native",
        "mm_attention_backend": "sdpa",
        "sampling_backend": "pytorch",
        "dtype": "float16",
    }


def test_qwen3_asr_torch_mps_profile_overrides_typed_torch_compile_default() -> None:
    builder = make_engine_builder()
    builder.device = "mps"
    overrides = {"enable_torch_compile": True}

    defaults = builder.generation_defaults(dtype="auto")
    builder.adjust_overrides(overrides)

    assert defaults["enable_torch_compile"] is False
    assert overrides["enable_torch_compile"] is False


def test_qwen3_asr_config_uses_batched_stage_with_64_running_requests() -> None:
    config = Qwen3ASRPipelineConfig(model_path="Qwen/Qwen3-ASR-1.7B")

    assert config.entry_stage == "asr"
    assert [stage.name for stage in config.stages] == ["asr"]
    assert config.terminal_stages == ["asr"]
    assert config.gpu_placement == {"asr": 0}
    stage = config.stages[0]
    assert stage.factory_path.endswith("create_sglang_qwen3_asr_executor")
    assert stage.factory.device is None
    assert stage.engine.max_running_requests == 64
    assert stage.engine.enable_torch_compile is True
    assert stage.engine.torch_compile_max_bs == 2
    assert stage.factory.request_build_max_workers == 8
    assert stage.factory.request_build_max_pending == 32
    assert stage.factory.prefill_coalesce_requests == 16
    assert stage.factory.prefill_coalesce_wait_ms == 40
    assert stage.factory.prefill_coalesce_when_idle is True
    assert stage.factory.prefill_coalesce_requires_pending_builds is True
    assert stage.factory.prefill_coalesce_after_builds_during_decode is True
    assert stage.factory.enable_pre_lm_encoder is True
    assert stage.factory.pre_lm_cache_max_entries == 4096
    assert stage.factory.pre_lm_cache_size_bytes == 2 * 1024**3
    assert stage.factory.pre_lm_max_batch_size == 8
    assert stage.factory.pre_lm_max_batch_wait_ms == 0
    assert type(config).stage_config_cls("asr").engine_stage
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("Qwen3ASRForConditionalGeneration")
        is Qwen3ASRPipelineConfig
    )


@pytest.mark.parametrize(
    ("batch_size", "wait_ms", "match"),
    [
        (0, 4, "pre_lm_max_batch_size"),
        (8, -1, "pre_lm_max_batch_wait_ms"),
    ],
)
def test_qwen3_asr_stage_rejects_invalid_pre_lm_batch_knobs(
    batch_size: int, wait_ms: int, match: str
) -> None:
    # Validation runs before any model/tokenizer load.
    with pytest.raises(ValueError, match=match):
        create_sglang_qwen3_asr_executor(
            "dummy",
            pre_lm_max_batch_size=batch_size,
            pre_lm_max_batch_wait_ms=wait_ms,
        )


def patch_engine_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    want_cuda_graph: bool = False,
) -> SimpleNamespace:
    recorded = SimpleNamespace(
        build_kwargs={},
        adapter_kwargs={},
        memory_queries=[],
        infra_kwargs=[],
        attest_calls=[],
        graph_init_calls=[],
        encoder_service=SimpleNamespace(close=lambda: None),
        encoder_service_kwargs={},
        tokenizer=object(),
        stream_output_builder=object(),
        stream_builder_calls=[],
    )
    monkeypatch.setattr(
        device_utils,
        "resolve_device_spec",
        lambda device, gpu_id: f"cuda:{gpu_id or 0}",
    )

    monkeypatch.setattr(
        qwen3_asr_builder.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: recorded.tokenizer,
    )
    monkeypatch.setattr(
        qwen3_asr_builder.AutoFeatureExtractor,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(nb_max_frames=55072),
    )
    monkeypatch.setattr(
        qwen3_asr_builder,
        "get_visible_gpu_sm_version",
        lambda gpu_id: 89,
    )
    monkeypatch.setattr(
        qwen3_asr_builder,
        "get_process_gpu_memory_bytes",
        lambda gpu_id: recorded.memory_queries.append(gpu_id) or 0,
    )
    monkeypatch.setattr(qwen3_asr_builder, "init_mm_embedding_cache", lambda size: None)

    def make_encoder_service(*args, **kwargs):
        recorded.encoder_service_kwargs.update(kwargs)
        return recorded.encoder_service

    monkeypatch.setattr(
        qwen3_asr_builder,
        "Qwen3ASRPreLMEncoderService",
        make_encoder_service,
    )
    monkeypatch.setattr(
        qwen3_asr_builder,
        "build_cache_namespace",
        lambda *args, **kwargs: "testns",
    )
    monkeypatch.setattr(
        request_builders,
        "make_qwen3_asr_scheduler_adapters",
        lambda **kwargs: (recorded.adapter_kwargs.update(kwargs) or object(), object()),
    )
    monkeypatch.setattr(
        request_builders,
        "make_qwen3_asr_stream_output_builder",
        lambda **kwargs: recorded.stream_builder_calls.append(kwargs)
        or recorded.stream_output_builder,
    )
    monkeypatch.setattr(
        sglang_backend,
        "SGLangOutputProcessor",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        model_runner_base,
        "ModelRunner",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        omni_scheduler,
        "OmniScheduler",
        lambda **kwargs: SimpleNamespace(
            request_build_queue_fits_workers=lambda: False,
            **kwargs,
        ),
    )
    monkeypatch.setattr(
        sglang_backend,
        "build_sglang_server_args",
        fake_server_args_builder(recorded.build_kwargs),
    )

    def fake_create_infrastructure(server_args, gpu_id, **kwargs):
        recorded.server_args = server_args
        recorded.infra_kwargs.append(dict(kwargs))
        model_worker = SimpleNamespace(
            gpu_id=gpu_id,
            model_runner=SimpleNamespace(
                model=SimpleNamespace(init_encoder_graphs=lambda **kwargs: None),
                prefill_attention_backend_str="fa3",
                decode_attention_backend_str="fa3",
            ),
        )
        return want_cuda_graph, (
            model_worker,
            object(),
            object(),
            object(),
            object(),
        )

    monkeypatch.setattr(
        bootstrap,
        "create_sglang_infrastructure_defer_cuda_graph",
        fake_create_infrastructure,
    )
    monkeypatch.setattr(
        bootstrap,
        "init_sglang_cuda_graphs",
        lambda model_worker: recorded.graph_init_calls.append(model_worker),
    )
    monkeypatch.setattr(
        cuda_graph_batch_validator,
        "attest_prefill_cuda_graphs",
        lambda model_runner, *, operator_selected: recorded.attest_calls.append(
            (model_runner, operator_selected)
        ),
    )
    return recorded


def test_qwen3_asr_threads_explicit_cuda_graph_bs(monkeypatch, caplog) -> None:
    recorded = patch_engine_dependencies(monkeypatch)
    build_kwargs = recorded.build_kwargs

    with caplog.at_level("INFO", logger=qwen3_asr_builder.__name__):
        scheduler = qwen3_asr_stages.create_sglang_qwen3_asr_executor(
            "dummy",
            enable_async_decode=False,
            async_decode_min_batch_size=4,
            stream_emit_interval_s=0.125,
            server_args_overrides={"context_length": 2048},
        )

    assert build_kwargs["cuda_graph_max_bs"] == 64
    assert build_kwargs["cuda_graph_bs"] == [
        1,
        2,
        4,
        8,
        12,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
    ]
    assert "cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64]" in caplog.text
    assert "mm_attention_backend" not in build_kwargs
    assert recorded.memory_queries == [0, 0, 0, 0]
    assert recorded.adapter_kwargs["context_length"] == 2048
    assert scheduler.enable_async_decode is False
    assert scheduler.async_decode_min_batch_size == 4
    assert scheduler.prefill_coalesce_requests == 16
    assert scheduler.prefill_coalesce_wait_ms == 40.0
    assert scheduler.prefill_coalesce_when_idle is True
    assert scheduler.prefill_coalesce_requires_pending_builds is True
    assert scheduler.prefill_coalesce_after_builds_during_decode is True
    assert scheduler.shutdown_callback is recorded.encoder_service.close
    assert scheduler.stream_output_builder is recorded.stream_output_builder
    assert recorded.stream_builder_calls == [
        {"tokenizer": recorded.tokenizer, "min_emit_interval_s": 0.125}
    ]


def test_qwen3_asr_prefill_ladder_is_accepted_by_the_shared_policy(caplog) -> None:
    builder = make_engine_builder(mm_attention_backend="fa3")
    overrides = build_generation_batch_overrides(
        **builder.generation_defaults(dtype="bfloat16")
    )
    builder.adjust_overrides(overrides)
    assert overrides["cuda_graph_bs_prefill"] == sglang_prefill_ladder(4096)
    server_args = fake_server_args_builder({})("dummy", 1636, **overrides)

    with caplog.at_level(logging.WARNING):
        validate_generation_batch_policy(
            model_name="Qwen3-ASR", server_args=server_args
        )

    assert not any("padding factor" in record.message for record in caplog.records)


def test_qwen3_asr_build_initializes_and_attests_prefill_graphs(monkeypatch) -> None:
    recorded = patch_engine_dependencies(monkeypatch, want_cuda_graph=True)

    qwen3_asr_stages.create_sglang_qwen3_asr_executor("dummy")

    assert recorded.infra_kwargs[-1]["enable_prefill_input_embeds"] is True
    assert len(recorded.graph_init_calls) == 1
    assert len(recorded.attest_calls) == 1


@pytest.mark.parametrize("resolved_chunked_prefill_size", [2048, 8192])
def test_qwen3_asr_auto_chunked_prefill_uses_the_sglang_ladder(
    monkeypatch,
    caplog,
    resolved_chunked_prefill_size: int,
) -> None:
    recorded = patch_engine_dependencies(monkeypatch, want_cuda_graph=True)
    monkeypatch.setattr(
        sglang_backend,
        "build_sglang_server_args",
        fake_server_args_builder(
            recorded.build_kwargs,
            resolved_chunked_prefill_size=resolved_chunked_prefill_size,
        ),
    )

    with caplog.at_level(logging.INFO):
        scheduler = qwen3_asr_stages.create_sglang_qwen3_asr_executor(
            "dummy",
            server_args_overrides={"chunked_prefill_size": None},
        )

    assert scheduler is not None
    assert recorded.build_kwargs["chunked_prefill_size"] is None
    assert "cuda_graph_bs_prefill" not in recorded.build_kwargs
    assert "cuda_graph_max_bs_prefill" not in recorded.build_kwargs
    assert recorded.attest_calls[-1][1] is False
    prefill = recorded.server_args.cuda_graph_config.prefill
    assert list(prefill.bs) == sglang_prefill_ladder(resolved_chunked_prefill_size)
    assert prefill.max_bs == resolved_chunked_prefill_size
    assert (
        "Qwen3-ASR: chunked_prefill_size was unset, SGLang resolved "
        f"{resolved_chunked_prefill_size}, prefill CUDA graph cap "
        f"{resolved_chunked_prefill_size}"
    ) in caplog.text
    assert "cannot be scheduled" not in caplog.text


def test_qwen3_asr_disabled_chunked_prefill_derives_no_buckets(
    monkeypatch, caplog
) -> None:
    recorded = patch_engine_dependencies(monkeypatch, want_cuda_graph=True)

    with caplog.at_level(logging.WARNING):
        qwen3_asr_stages.create_sglang_qwen3_asr_executor(
            "dummy", server_args_overrides={"chunked_prefill_size": 0}
        )

    assert list(recorded.server_args.cuda_graph_config.prefill.bs) == []
    assert len(recorded.graph_init_calls) == 1
    assert "require a positive prefill graph cap" in caplog.text


def test_qwen3_asr_operator_buckets_above_the_chunk_start_with_a_warning(
    monkeypatch, caplog
) -> None:
    recorded = patch_engine_dependencies(monkeypatch, want_cuda_graph=True)

    with caplog.at_level(logging.WARNING):
        qwen3_asr_stages.create_sglang_qwen3_asr_executor(
            "dummy", server_args_overrides={"cuda_graph_bs_prefill": [1024, 8192]}
        )

    assert recorded.attest_calls[-1][1] is False
    assert recorded.build_kwargs["cuda_graph_bs_prefill"] == [1024, 8192]
    assert recorded.build_kwargs["cuda_graph_max_bs_prefill"] == 8192
    assert list(recorded.server_args.cuda_graph_config.prefill.bs) == [1024, 8192]
    assert "max=8192 exceeds chunked_prefill_size=4096" in caplog.text


@pytest.mark.parametrize(
    ("cap_override", "expected_cap"),
    [
        ({"chunked_prefill_size": 512}, 512),
        ({"chunked_prefill_size": 4592}, 4592),
        ({"cuda_graph_max_bs_prefill": 512}, 512),
        ({"max_total_tokens": 640}, 640),
        ({"max_total_tokens": 1000}, 1000),
    ],
)
def test_qwen3_asr_ladder_respects_deployment_cap_overrides(
    monkeypatch: pytest.MonkeyPatch,
    cap_override: dict[str, int],
    expected_cap: int,
) -> None:
    recorded = patch_engine_dependencies(monkeypatch, want_cuda_graph=True)

    qwen3_asr_stages.create_sglang_qwen3_asr_executor(
        "dummy", server_args_overrides=dict(cap_override)
    )

    assert len(recorded.attest_calls) == 1
    assert all(
        recorded.build_kwargs[key] == value for key, value in cap_override.items()
    )
    assert recorded.build_kwargs["cuda_graph_bs_prefill"][-1] == expected_cap
    assert max(recorded.build_kwargs["cuda_graph_bs_prefill"]) == expected_cap
    assert recorded.build_kwargs["cuda_graph_max_bs_prefill"] == expected_cap


@pytest.mark.parametrize(
    ("disable_override", "want_cuda_graph", "expected_graph_inits"),
    [
        ({"disable_cuda_graph": True}, False, 0),
        ({"cuda_graph_backend_prefill": "disabled"}, True, 1),
    ],
)
def test_qwen3_asr_disable_override_yields_no_prefill_ladder(
    monkeypatch: pytest.MonkeyPatch,
    disable_override: dict[str, object],
    want_cuda_graph: bool,
    expected_graph_inits: int,
) -> None:
    recorded = patch_engine_dependencies(monkeypatch, want_cuda_graph=want_cuda_graph)

    qwen3_asr_stages.create_sglang_qwen3_asr_executor(
        "dummy", server_args_overrides=dict(disable_override)
    )

    assert "cuda_graph_bs_prefill" not in recorded.build_kwargs
    assert len(recorded.graph_init_calls) == expected_graph_inits
    assert recorded.attest_calls == []


def test_qwen3_asr_nested_prefill_override_supersedes_the_derived_ladder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = patch_engine_dependencies(monkeypatch, want_cuda_graph=True)

    qwen3_asr_stages.create_sglang_qwen3_asr_executor(
        "dummy",
        server_args_overrides={
            "cuda_graph_config": {"prefill": {"backend": "breakable", "bs": [128, 256]}}
        },
    )

    assert recorded.attest_calls[-1][1] is True
    assert list(recorded.build_kwargs["cuda_graph_bs_prefill"]) == [128, 256]
    assert recorded.build_kwargs["cuda_graph_max_bs_prefill"] == 256


@pytest.mark.parametrize(
    "operator_batches", [None, [1, 8, 64], build_default_cuda_graph_bs(64)]
)
def test_qwen3_asr_cuda_decode_graphs_preserve_operator_buckets(
    monkeypatch: pytest.MonkeyPatch, operator_batches: list[int] | None
) -> None:
    """Exact-size defaults must not replace an operator's sparse capture grid."""
    monkeypatch.setattr(qwen3_asr_builder.current_platform, "is_cuda", lambda: True)
    builder = make_engine_builder(mm_attention_backend="fa3")
    options = {} if operator_batches is None else {"cuda_graph_bs": operator_batches}
    builder.server_args_overrides = options
    overrides = build_generation_batch_overrides(
        **builder.generation_defaults(dtype="bfloat16"), server_args_overrides=options
    )
    builder.adjust_overrides(overrides)
    expected_batches = (
        list(range(1, 65)) if operator_batches is None else operator_batches
    )
    assert overrides["cuda_graph_bs"] == expected_batches
    overrides["attention_backend"] = "fa3"
    overrides["prefill_attention_backend"] = None
    overrides["decode_attention_backend"] = None
    server_args = fake_server_args_builder({})("dummy", 1636, **overrides)
    validate_generation_batch_policy(
        model_name="Qwen3-ASR",
        server_args=server_args,
        allowed_prefill_backends=("full", "breakable"),
    )


def test_qwen3_asr_glue_graph_stage_override_is_preserved() -> None:
    """Operators must retain the eager metadata escape hatch at stage resolution."""
    config = Qwen3ASRPipelineConfig(model_path="Qwen/Qwen3-ASR-1.7B")
    config.stage_named("asr").env["SGLANG_ENABLE_METADATA_GLUE_GRAPH"] = "0"
    assert (
        config.resolved_stage_env_defaults("asr")["SGLANG_ENABLE_METADATA_GLUE_GRAPH"]
        == "0"
    )


@pytest.mark.parametrize("decode_backend", ["fa3", "flashinfer"])
def test_qwen3_asr_metadata_graph_rejects_host_planned_backends(
    monkeypatch: pytest.MonkeyPatch, decode_backend: str
) -> None:
    """FlashInfer host plans must not be frozen by successful metadata capture."""
    monkeypatch.setattr(qwen3_asr_builder.current_platform, "is_cuda", lambda: True)
    monkeypatch.setenv("SGLANG_ENABLE_METADATA_GLUE_GRAPH", "1")
    capture_enabled = []
    monkeypatch.setattr(
        qwen3_asr_builder.metadata_glue,
        "apply_thread_local_glue_capture",
        lambda: capture_enabled.append(True),
    )
    builder = make_engine_builder()
    model_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            prefill_attention_backend_str="fa3",
            decode_attention_backend_str=decode_backend,
        )
    )
    builder.setup_model(
        model_worker=model_worker,
        checkpoint_dir="dummy",
        device="cuda",
        gpu_id=0,
        server_args=None,
    )
    assert capture_enabled == ([True] if decode_backend == "fa3" else [])
    assert qwen3_asr_builder.envs.SGLANG_ENABLE_METADATA_GLUE_GRAPH.get() == (
        decode_backend == "fa3"
    )


def test_qwen3_asr_cuda_pipeline_builds_full_and_exact_size_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CUDA engine must pass full prefill and dense decode sizing to the runner."""
    recorded = patch_engine_dependencies(monkeypatch, want_cuda_graph=True)
    monkeypatch.setattr(qwen3_asr_builder.current_platform, "is_cuda", lambda: True)
    qwen3_asr_stages.create_sglang_qwen3_asr_executor("dummy")
    assert recorded.build_kwargs["cuda_graph_bs"] == list(range(1, 65))
    assert recorded.server_args.cuda_graph_config.prefill.backend == "full"
    assert (
        recorded.build_kwargs["cuda_graph_config"]["prefill"]["full_prefill_max_req"]
        == 32
    )
    assert len(recorded.graph_init_calls) == 1
    assert len(recorded.attest_calls) == 1


@pytest.mark.parametrize("prefill_override", [{}, {"full_prefill_max_req": 12}])
def test_decode_only_override_retains_full_prefill_capacity(
    monkeypatch: pytest.MonkeyPatch, prefill_override: dict[str, int]
) -> None:
    """A decode override must not silently reduce full-prefill admission."""
    monkeypatch.setattr(qwen3_asr_builder.current_platform, "is_cuda", lambda: True)
    builder = make_engine_builder(mm_attention_backend="fa3")
    builder.server_args_overrides = {
        "cuda_graph_config": {"decode": {"max_bs": 16}, "prefill": prefill_override}
    }
    overrides = build_generation_batch_overrides(
        **builder.generation_defaults(dtype="bfloat16"),
        server_args_overrides=builder.server_args_overrides,
    )
    builder.adjust_overrides(overrides)
    graph_config = overrides["cuda_graph_config"]
    assert graph_config["decode"]["max_bs"] == 16
    assert graph_config["prefill"]["full_prefill_max_req"] == prefill_override.get(
        "full_prefill_max_req", 32
    )
