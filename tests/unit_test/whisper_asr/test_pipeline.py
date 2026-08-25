# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import sys
from types import SimpleNamespace

import pytest

import sglang_omni.model_runner.base as model_runner_base
import sglang_omni.models.whisper_asr.stages as whisper_asr_stages
import sglang_omni.scheduling.bootstrap as bootstrap
import sglang_omni.scheduling.omni_scheduler as omni_scheduler
import sglang_omni.scheduling.sglang_backend as sglang_backend
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.models.whisper_asr import request_builders as whisper_request_builders
from sglang_omni.models.whisper_asr.config import WhisperASRPipelineConfig
from tests.unit_test.fakes import FakeServerArgs


def _encoder_graph_builder(**kwargs):
    from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder

    params = {
        "max_running_requests": 16,
        "max_new_tokens": 32,
        "mem_fraction_static": 0.2,
        "enable_encoder_cuda_graph": True,
    }
    params.update(kwargs)
    builder = WhisperASREngineBuilder(**params)
    builder.processor = SimpleNamespace(
        feature_extractor=SimpleNamespace(nb_max_frames=3000)
    )
    builder.encoder_token_count = 1500
    return builder


def test_whisper_stage_defaults() -> None:
    signature = inspect.signature(whisper_asr_stages.create_sglang_whisper_asr_executor)

    assert signature.parameters["max_running_requests"].default == 64
    assert signature.parameters["enable_encoder_cuda_graph"].default is False
    assert signature.parameters["encoder_graph_batch_buckets"].default is None
    assert signature.parameters["request_build_max_workers"].default == 8
    assert signature.parameters["enable_async_decode"].default is True
    assert signature.parameters["async_decode_min_batch_size"].default == 2
    assert signature.parameters["request_build_max_pending"].default == 16
    assert signature.parameters["prefill_coalesce_requests"].default == 2
    assert signature.parameters["prefill_coalesce_wait_ms"].default == 6.0
    assert signature.parameters["prefill_coalesce_when_idle"].default is True
    assert (
        signature.parameters["prefill_coalesce_requires_pending_builds"].default is True
    )
    assert (
        signature.parameters["prefill_coalesce_after_builds_during_decode"].default
        is False
    )
    assert signature.parameters["enable_pre_lm_encoder"].default is True
    assert signature.parameters["pre_lm_max_batch_size"].default == 8


def test_whisper_encoder_cuda_graph_setup_is_ordered_after_generation_graphs() -> None:
    calls: list[tuple[list[int], int]] = []
    builder = _encoder_graph_builder(max_running_requests=4)
    assert builder.encoder_graph_batch_buckets == (1, 2, 4, 8, 12, 16)
    model = SimpleNamespace(
        init_encoder_graphs=lambda buckets, feature_len: calls.append(
            (list(buckets), feature_len)
        )
    )

    builder.setup_model_resources(
        model,
        server_args=SimpleNamespace(max_prefill_tokens=4096, max_running_requests=4),
        generation_cuda_graph_enabled=True,
    )
    assert calls == [([1, 2, 4], 3000)]

    builder.setup_model_resources(
        model,
        server_args=SimpleNamespace(max_prefill_tokens=4096, max_running_requests=4),
        generation_cuda_graph_enabled=False,
    )
    assert calls == [([1, 2, 4], 3000)]


def test_whisper_default_encoder_graph_buckets_follow_prefill_without_pre_lm() -> None:
    calls: list[list[int]] = []
    builder = _encoder_graph_builder(
        enable_pre_lm_encoder=False,
        max_running_requests=32,
    )
    model = SimpleNamespace(
        init_encoder_graphs=lambda buckets, feature_len: calls.append(list(buckets))
    )

    builder.setup_model_resources(
        model,
        server_args=SimpleNamespace(
            max_prefill_tokens=6144,
            max_running_requests=32,
        ),
        generation_cuda_graph_enabled=True,
    )

    assert calls == [[1, 2, 4]]


@pytest.mark.parametrize(
    ("builder_kwargs", "max_prefill_tokens", "expected"),
    [
        ({"pre_lm_max_batch_size": 8}, 4096, [1, 4, 8]),
        ({"enable_pre_lm_encoder": False}, 8192, [1, 4]),
    ],
    ids=["pre_lm", "prefill_without_pre_lm"],
)
def test_whisper_encoder_cuda_graph_buckets_are_filtered(
    builder_kwargs: dict[str, object],
    max_prefill_tokens: int,
    expected: list[int],
) -> None:
    calls: list[list[int]] = []
    builder = _encoder_graph_builder(
        encoder_graph_batch_buckets=[8, 1, 4, 4, 16],
        **builder_kwargs,
    )
    model = SimpleNamespace(
        init_encoder_graphs=lambda buckets, feature_len: calls.append(list(buckets))
    )

    builder.setup_model_resources(
        model,
        server_args=SimpleNamespace(
            max_prefill_tokens=max_prefill_tokens,
            max_running_requests=16,
        ),
        generation_cuda_graph_enabled=True,
    )

    assert calls == [expected]


def test_whisper_disables_chunked_prefill_for_atomic_encoder_prefix() -> None:
    from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder

    builder = WhisperASREngineBuilder(
        max_running_requests=4,
        max_new_tokens=32,
        mem_fraction_static=0.2,
    )
    defaults = builder.generation_defaults(dtype="float16")

    assert defaults["max_prefill_tokens"] == 6144
    assert defaults["chunked_prefill_size"] == 0

    overrides = {"chunked_prefill_size": 0}
    builder.adjust_overrides(overrides)
    assert overrides["chunked_prefill_size"] == 0
    assert overrides["enable_custom_logit_processor"] is True

    with pytest.raises(ValueError, match="encoder prefix must be admitted atomically"):
        builder.adjust_overrides({"chunked_prefill_size": 4096})


def test_whisper_prefill_coalescing_defaults_are_forwarded() -> None:
    from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder

    builder = WhisperASREngineBuilder(
        max_running_requests=16,
        max_new_tokens=32,
        mem_fraction_static=0.2,
    )

    assert builder.extra_scheduler_kwargs() == {
        "enable_async_decode": True,
        "async_decode_min_batch_size": 2,
        "request_build_max_workers": 8,
        "request_build_max_pending": 16,
        "prefill_coalesce_requests": 2,
        "prefill_coalesce_wait_ms": 6.0,
        "prefill_coalesce_when_idle": True,
        "prefill_coalesce_requires_pending_builds": True,
        "prefill_coalesce_after_builds_during_decode": False,
    }


def test_whisper_rejects_invalid_pre_lm_batch_knobs() -> None:
    from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder

    with pytest.raises(ValueError, match="pre_lm_max_batch_size must be >= 1"):
        WhisperASREngineBuilder(
            max_running_requests=4,
            max_new_tokens=32,
            mem_fraction_static=0.2,
            pre_lm_max_batch_size=0,
        )
    with pytest.raises(ValueError, match="pre_lm_max_batch_wait_ms must be >= 0"):
        WhisperASREngineBuilder(
            max_running_requests=4,
            max_new_tokens=32,
            mem_fraction_static=0.2,
            pre_lm_max_batch_wait_ms=-1,
        )


def test_whisper_asr_config_uses_single_batched_stage() -> None:
    config = WhisperASRPipelineConfig(model_path="openai/whisper-large-v3")

    assert config.entry_stage == "asr"
    assert [stage.name for stage in config.stages] == ["asr"]
    assert config.terminal_stages == ["asr"]
    assert config.gpu_placement == {"asr": 0}
    stage = config.stages[0]
    assert stage.factory_path.endswith("create_sglang_whisper_asr_executor")
    assert stage.engine.max_running_requests == 64
    factory = stage.factory
    assert factory.device == "cuda:0"
    assert factory.enable_encoder_cuda_graph is True
    assert factory.request_build_max_workers == 8
    assert factory.enable_async_decode is True
    assert factory.async_decode_min_batch_size == 2
    assert factory.request_build_max_pending == 16
    assert factory.prefill_coalesce_requests == 2
    assert factory.prefill_coalesce_wait_ms == 6.0
    assert factory.prefill_coalesce_when_idle is True
    assert factory.prefill_coalesce_requires_pending_builds is True
    assert factory.prefill_coalesce_after_builds_during_decode is False
    assert factory.enable_pre_lm_encoder is True
    assert factory.pre_lm_cache_max_entries == 1024
    # None: the byte budget is derived from the entry count.
    assert factory.pre_lm_cache_size_bytes is None
    assert factory.pre_lm_max_batch_size == 8
    assert factory.pre_lm_max_batch_wait_ms == 0
    assert (factory.model_extra or {}).get("max_prefill_tokens") is None
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("WhisperForConditionalGeneration")
        is WhisperASRPipelineConfig
    )


def test_whisper_async_decode_dotted_overrides() -> None:
    from sglang_omni.config.manager import ConfigManager

    config = WhisperASRPipelineConfig(model_path="openai/whisper-base")

    forced_sync = ConfigManager(config).merge_config(
        [("asr.factory.enable_async_decode", "false")]
    )
    assert forced_sync.stages[0].factory.enable_async_decode is False

    forced_async = ConfigManager(config).merge_config(
        [
            ("asr.factory.enable_async_decode", "true"),
            ("asr.factory.async_decode_min_batch_size", "4"),
        ]
    )
    assert forced_async.stages[0].factory.enable_async_decode is True
    assert forced_async.stages[0].factory.async_decode_min_batch_size == 4


def test_whisper_asr_threads_explicit_cuda_graph_bs(monkeypatch) -> None:
    build_kwargs: dict[str, object] = {}
    scheduler_kwargs: dict[str, object] = {}
    fake_processor = SimpleNamespace(
        tokenizer=object(),
        feature_extractor=SimpleNamespace(nb_max_frames=3000),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoConfig=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: SimpleNamespace(
                    max_target_positions=448
                )
            ),
            AutoProcessor=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: fake_processor
            ),
            GenerationConfig=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: object()
            ),
        ),
    )
    monkeypatch.setattr(
        whisper_request_builders,
        "make_whisper_scheduler_adapters",
        lambda **kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        model_runner_base,
        "ModelRunner",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        sglang_backend,
        "SGLangOutputProcessor",
        lambda **kwargs: object(),
    )

    def _fake_scheduler(**kwargs):
        scheduler_kwargs.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(omni_scheduler, "OmniScheduler", _fake_scheduler)

    def _fake_server_args_builder(model_path, context_length, **overrides):
        build_kwargs["context_length"] = context_length
        build_kwargs.update(overrides)
        server_args = FakeServerArgs(**overrides)
        server_args.cuda_graph_config = SimpleNamespace(
            decode=SimpleNamespace(
                max_bs=overrides["cuda_graph_max_bs"],
                bs=overrides["cuda_graph_bs"],
            ),
            prefill=SimpleNamespace(backend="disabled", bs=None, max_bs=None),
        )
        return server_args

    def _fake_create_infrastructure(server_args, gpu_id, **kwargs):
        model_worker = SimpleNamespace(model_runner=SimpleNamespace(model=object()))
        return False, (
            model_worker,
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
        )

    monkeypatch.setattr(
        sglang_backend,
        "build_sglang_server_args",
        _fake_server_args_builder,
    )
    monkeypatch.setattr(
        bootstrap,
        "create_sglang_infrastructure_defer_cuda_graph",
        _fake_create_infrastructure,
    )

    whisper_asr_stages.create_sglang_whisper_asr_executor(
        "dummy",
        enable_pre_lm_encoder=False,
        enable_async_decode=False,
        async_decode_min_batch_size=4,
    )

    assert build_kwargs["cuda_graph_max_bs"] == 64
    assert build_kwargs["cuda_graph_bs"] == [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64]
    # note (jiannan-17): context_length = encoder_token_count + max_prev_tokens + max_new_tokens + 8
    assert build_kwargs["context_length"] == 1500 + 224 + 256 + 8
    assert build_kwargs["chunked_prefill_size"] == 0
    assert build_kwargs["enable_custom_logit_processor"] is True
    assert build_kwargs["max_prefill_tokens"] == 6144
    assert scheduler_kwargs["enable_async_decode"] is False
    assert scheduler_kwargs["async_decode_min_batch_size"] == 4
