# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import sys
import types
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset.seedtts import SampleInput
from benchmarks.tasks.tts import (
    MOSS_TTS_TOKEN_COUNT_AUTO,
    _build_tts_payload,
    _handle_raw_pcm_streaming_response,
    estimate_moss_tts_duration_tokens,
)
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.models.moss_tts.config import MossTTSPipelineConfig
from sglang_omni.models.moss_tts.payload_types import MossTTSState
from sglang_omni.models.moss_tts.request_builders import (
    _INF_DELAY,
    build_moss_tts_state,
    build_row_cache_key_ids,
    build_sglang_moss_tts_request,
    clear_moss_tts_preprocessing_context,
    preprocess_moss_tts_payload,
    set_moss_tts_preprocessing_context,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.types import RequestOutput
from tests.unit_test.fakes import FakeServerArgs


def install_fake_sglang(monkeypatch: pytest.MonkeyPatch) -> None:
    from sglang_omni.models.moss_tts import request_builders

    class FakeReq:
        def __init__(
            self,
            *,
            rid,
            origin_input_text,
            origin_input_ids,
            sampling_params,
            eos_token_ids=None,
            vocab_size=None,
            extra_key=None,
            **kwargs,
        ) -> None:
            del kwargs
            self.rid = rid
            self.origin_input_text = origin_input_text
            self.origin_input_ids = origin_input_ids
            self.sampling_params = sampling_params
            self.eos_token_ids = eos_token_ids
            self.vocab_size = vocab_size
            self.extra_key = extra_key
            self.output_ids = []
            self.prefix_indices = []
            self.extend_range = SimpleNamespace(length=len(origin_input_ids))

    class FakeSamplingParams:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

        def normalize(self, tokenizer) -> None:
            del tokenizer

        def verify(self, vocab_size) -> None:
            self.vocab_size = vocab_size

    monkeypatch.setattr(request_builders, "Req", FakeReq)
    monkeypatch.setattr(request_builders, "SamplingParams", FakeSamplingParams)


def make_payload(
    *,
    inputs,
    params: dict | None = None,
    tts_params: dict | None = None,
    request_id: str = "req-moss",
) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(
            inputs=inputs,
            params=params or {},
            metadata={"tts_params": tts_params or {}},
        ),
        data={},
    )


def test_moss_tts_config_and_registry_contracts() -> None:
    config = MossTTSPipelineConfig(model_path="model")
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "tts_engine",
        "vocoder",
    ]
    assert config.terminal_stages == ["vocoder"]
    assert config.gpu_placement == {
        "preprocessing": 0,
        "tts_engine": 0,
        "vocoder": 0,
    }
    assert {stage.process for stage in config.stages} == {"pipeline"}
    assert config.supports_uploaded_voice_references() is True
    tts_engine = next(stage for stage in config.stages if stage.name == "tts_engine")
    vocoder = next(stage for stage in config.stages if stage.name == "vocoder")
    assert tts_engine.stream_to == ["vocoder"]
    assert vocoder.can_accept_stream_before_payload is True
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("MossTTSDelayModel")
        is MossTTSPipelineConfig
    )
    assert MossTTSPipelineConfig.mem_fraction_role_to_stage() == {
        "talker": "tts_engine"
    }
    assert MossTTSPipelineConfig.talker_sglang_role_to_stage() == {
        "talker": "tts_engine"
    }
    preprocessing = next(
        stage for stage in config.stages if stage.name == "preprocessing"
    )
    vocoder = next(stage for stage in config.stages if stage.name == "vocoder")
    assert preprocessing.factory_args == {
        "dtype": "float32",
        "ref_audio_cache": True,
        "ref_audio_cache_max_items": 8192,
        "ref_audio_cache_max_bytes": 64 * 1024 * 1024,
    }
    assert vocoder.factory_args == {
        "dtype": "float32",
        "compute_dtype": "bfloat16",
    }


def test_moss_tts_production_config_resolves_codec_memory_policy() -> None:
    config = ConfigManager.from_file("examples/configs/moss_tts.yaml").config

    assert isinstance(config, MossTTSPipelineConfig)
    stages = {stage.name: stage for stage in config.stages}
    preprocessing_args = resolve_stage_factory_args(
        stages["preprocessing"], config, gpu_id=0
    )
    vocoder_args = resolve_stage_factory_args(stages["vocoder"], config, gpu_id=0)

    assert preprocessing_args == {
        "dtype": "float32",
        "ref_audio_cache": True,
        "ref_audio_cache_max_items": 8192,
        "ref_audio_cache_max_bytes": 64 * 1024 * 1024,
        "model_path": "OpenMOSS-Team/MOSS-TTS-v1.5",
        "gpu_id": 0,
    }
    assert vocoder_args == {
        "dtype": "float32",
        "compute_dtype": "bfloat16",
        "model_path": "OpenMOSS-Team/MOSS-TTS-v1.5",
        "gpu_id": 0,
    }


def test_moss_tts_32gb_config_bounds_runtime_memory() -> None:
    config = ConfigManager.from_file("examples/configs/moss_tts_32gb.yaml").config

    assert isinstance(config, MossTTSPipelineConfig)
    stages = {stage.name: stage for stage in config.stages}
    preprocessing_args = resolve_stage_factory_args(
        stages["preprocessing"], config, gpu_id=0
    )
    tts_engine_args = resolve_stage_factory_args(stages["tts_engine"], config, gpu_id=0)
    vocoder_args = resolve_stage_factory_args(stages["vocoder"], config, gpu_id=0)

    assert preprocessing_args["device"] == "cpu"
    assert preprocessing_args["dtype"] == "float32"
    assert preprocessing_args["max_concurrency"] == 1
    assert tts_engine_args["dtype"] == "bfloat16"
    assert tts_engine_args["server_args_overrides"] == {
        "max_running_requests": 1,
        "mem_fraction_static": 0.70,
        "cuda_graph_max_bs": 1,
    }
    assert vocoder_args["dtype"] == "bfloat16"
    assert vocoder_args["compute_dtype"] == "bfloat16"
    assert vocoder_args["max_batch_size"] == 1
    assert vocoder_args["max_batch_wait_ms"] == 2


def test_moss_tts_24gb_config_bounds_runtime_memory() -> None:
    config = ConfigManager.from_file("examples/configs/moss_tts_24gb.yaml").config

    assert isinstance(config, MossTTSPipelineConfig)
    stages = {stage.name: stage for stage in config.stages}
    preprocessing_args = resolve_stage_factory_args(
        stages["preprocessing"], config, gpu_id=0
    )
    tts_engine_args = resolve_stage_factory_args(stages["tts_engine"], config, gpu_id=0)
    vocoder_args = resolve_stage_factory_args(stages["vocoder"], config, gpu_id=0)

    assert preprocessing_args["device"] == "cpu"
    assert preprocessing_args["dtype"] == "float32"
    assert preprocessing_args["max_concurrency"] == 1
    assert tts_engine_args["dtype"] == "bfloat16"
    assert tts_engine_args["server_args_overrides"] == {
        "max_running_requests": 1,
        "max_total_tokens": 8192,
        "mem_fraction_static": 0.78,
        "cuda_graph_max_bs": 1,
    }
    assert vocoder_args["dtype"] == "bfloat16"
    assert vocoder_args["compute_dtype"] == "bfloat16"
    assert vocoder_args["max_batch_size"] == 1
    assert vocoder_args["max_batch_wait_ms"] == 2


def test_moss_tts_codec_runtime_overrides_take_precedence() -> None:
    config = MossTTSPipelineConfig(
        model_path="model",
        runtime_overrides={
            "preprocessing": {"device": "cuda:7", "dtype": "bfloat16"},
            "vocoder": {"device": "cpu", "dtype": "float32"},
        },
    )
    stages = {stage.name: stage for stage in config.stages}

    preprocessing_args = resolve_stage_factory_args(
        stages["preprocessing"], config, gpu_id=2
    )
    vocoder_args = resolve_stage_factory_args(stages["vocoder"], config, gpu_id=2)

    assert preprocessing_args["device"] == "cuda:7"
    assert preprocessing_args["dtype"] == "bfloat16"
    assert preprocessing_args["gpu_id"] == 2
    assert vocoder_args["device"] == "cpu"
    assert vocoder_args["dtype"] == "float32"
    assert vocoder_args["compute_dtype"] == "bfloat16"
    assert vocoder_args["gpu_id"] == 2


def test_moss_tts_config_merge_updates_reference_cache_factory_args() -> None:
    from sglang_omni.config.manager import ConfigManager

    config = MossTTSPipelineConfig(model_path="model")
    merged = ConfigManager(config).merge_config(
        {
            "stages.preprocessing.factory_args.ref_audio_cache": False,
            "stages.preprocessing.factory_args.ref_audio_cache_max_items": 17,
            "stages.preprocessing.factory_args.ref_audio_cache_max_bytes": 4096,
        }
    )
    preprocessing = next(
        stage for stage in merged.stages if stage.name == "preprocessing"
    )

    assert preprocessing.factory_args["ref_audio_cache"] is False
    assert preprocessing.factory_args["ref_audio_cache_max_items"] == 17
    assert preprocessing.factory_args["ref_audio_cache_max_bytes"] == 4096


def test_moss_tts_config_merge_updates_vocoder_factory_args() -> None:
    from sglang_omni.config.manager import ConfigManager

    config = MossTTSPipelineConfig(model_path="model")
    merged = ConfigManager(config).merge_config(
        {
            "stages.vocoder.factory_args.compute_dtype": "float32",
        }
    )
    vocoder = next(stage for stage in merged.stages if stage.name == "vocoder")

    assert vocoder.factory_args["compute_dtype"] == "float32"

    disabled = ConfigManager(config).merge_config(
        {"stages.vocoder.factory_args.compute_dtype": None}
    )
    vocoder = next(stage for stage in disabled.stages if stage.name == "vocoder")

    assert vocoder.factory_args["compute_dtype"] is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("float32", torch.float32),
        ("bfloat16", torch.bfloat16),
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16),
    ],
)
def test_moss_tts_resolves_compute_dtype(value, expected) -> None:
    from sglang_omni.models.moss_tts import stages

    assert stages._resolve_compute_dtype(value) is expected


@pytest.mark.parametrize(
    "value", ["fp16", "float16", "fp32", "bf16", "invalid", torch.float16]
)
def test_moss_tts_rejects_invalid_compute_dtype(value) -> None:
    from sglang_omni.models.moss_tts import stages

    with pytest.raises(ValueError, match="compute_dtype"):
        stages._resolve_compute_dtype(value)


def test_moss_tts_preprocessing_factory_receives_placement_gpu_id() -> None:
    from sglang_omni.config.manager import ConfigManager
    from sglang_omni.config.runtime import resolve_stage_factory_args

    config = ConfigManager(MossTTSPipelineConfig(model_path="model")).merge_config(
        {"stages.preprocessing.gpu": 2}
    )
    preprocessing = next(
        stage for stage in config.stages if stage.name == "preprocessing"
    )
    factory_args = resolve_stage_factory_args(
        preprocessing,
        config,
        gpu_id=2,
    )

    assert preprocessing.gpu == 2
    assert factory_args["gpu_id"] == 2
    assert "device" not in factory_args


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"ref_audio_cache_max_items": 0}, "ref_audio_cache_max_items"),
        ({"ref_audio_cache_max_bytes": 0}, "ref_audio_cache_max_bytes"),
    ],
)
def test_moss_tts_preprocessing_rejects_invalid_reference_cache_settings(
    monkeypatch: pytest.MonkeyPatch,
    kwargs,
    match,
) -> None:
    from sglang_omni.models.moss_tts import stages

    monkeypatch.setattr(
        stages,
        "_load_moss_processor",
        lambda *_args, **_kwargs: pytest.fail("validation must precede model loading"),
    )
    with pytest.raises(ValueError, match=match):
        stages.create_preprocessing_executor("model", **kwargs)


def test_moss_tts_engine_uses_auto_mem_fraction_by_default(monkeypatch) -> None:
    from sglang_omni.models.moss_tts import request_builders, stages
    from sglang_omni.scheduling import (
        bootstrap,
        engine_factory,
        omni_scheduler,
        sglang_backend,
    )

    captured: dict[str, object] = {"build_kwargs": []}

    def fake_build_sglang_server_args(model_path, context_length, **kwargs):
        captured["model_path"] = model_path
        captured["context_length"] = context_length
        captured["build_kwargs"].append(dict(kwargs))
        return FakeServerArgs(
            disable_cuda_graph=kwargs["disable_cuda_graph"],
            disable_overlap_schedule=False,
            max_running_requests=kwargs["max_running_requests"],
            cuda_graph_max_bs=kwargs["cuda_graph_max_bs"],
            cuda_graph_bs=kwargs["cuda_graph_bs"],
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(
                    max_bs=kwargs["cuda_graph_max_bs"],
                    bs=kwargs["cuda_graph_bs"],
                ),
                prefill=SimpleNamespace(backend="disabled", bs=None, max_bs=None),
            ),
            enable_torch_compile=kwargs["enable_torch_compile"],
            torch_compile_max_bs=kwargs.get("torch_compile_max_bs"),
        )

    def fake_create_sglang_infrastructure(
        server_args,
        gpu_id,
        *,
        model_arch_override=None,
        defer_cuda_graph_capture=False,
    ):
        captured["gpu_id"] = gpu_id
        captured["model_arch_override"] = model_arch_override
        captured["defer_cuda_graph_capture"] = defer_cuda_graph_capture

        def init_sampling_graphs(batch_sizes, *, disable_padding):
            captured.setdefault("sampling_graph_inits", []).append(
                (batch_sizes, disable_padding)
            )
            captured.setdefault("graph_init_order", []).append("sampling")

        def init_cuda_graphs():
            captured["graph_inits"] = int(captured.get("graph_inits", 0)) + 1
            captured.setdefault("graph_init_order", []).append("backbone")

        model = SimpleNamespace(init_sampling_graphs=init_sampling_graphs)
        model_runner = SimpleNamespace(
            model=model,
            decode_cuda_graph_runner=SimpleNamespace(
                capture_bs=tuple(server_args.cuda_graph_bs),
                disable_padding=False,
            ),
            init_cuda_graphs=init_cuda_graphs,
        )
        model_worker = SimpleNamespace(
            model_runner=model_runner,
            enable_prefill_input_embeds=False,
        )
        return (
            model_worker,
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
        )

    class FakeOutputProcessor:
        def __init__(self, **kwargs) -> None:
            captured["output_processor_kwargs"] = kwargs

    class FakeMossTTSModelRunner:
        def __init__(self, model_worker, output_proc) -> None:
            captured["model_runner_args"] = (model_worker, output_proc)

    class FakeOmniScheduler:
        def __init__(self, **kwargs) -> None:
            captured["scheduler_kwargs"] = kwargs

    fake_model_runner_module = types.ModuleType(
        "sglang_omni.models.moss_tts.model_runner"
    )
    fake_model_runner_module.MossTTSModelRunner = FakeMossTTSModelRunner

    monkeypatch.setattr(
        engine_factory, "_resolve_checkpoint", lambda model_path: model_path
    )
    monkeypatch.setattr(
        request_builders,
        "make_moss_tts_scheduler_adapters",
        lambda model: (lambda payload: payload, lambda data: data),
    )
    monkeypatch.setattr(
        sglang_backend,
        "build_sglang_server_args",
        fake_build_sglang_server_args,
    )
    monkeypatch.setattr(sglang_backend, "SGLangOutputProcessor", FakeOutputProcessor)
    monkeypatch.setattr(
        bootstrap,
        "create_sglang_infrastructure",
        fake_create_sglang_infrastructure,
    )
    monkeypatch.setattr(omni_scheduler, "OmniScheduler", FakeOmniScheduler)
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.moss_tts.model_runner",
        fake_model_runner_module,
    )

    stages.create_sglang_tts_engine_executor("OpenMOSS-Team/MOSS-TTS-v1.5")
    stages.create_sglang_tts_engine_executor(
        "OpenMOSS-Team/MOSS-TTS-v1.5",
        server_args_overrides={
            "enable_torch_compile": True,
            "mem_fraction_static": 0.61,
        },
    )

    default_kwargs, explicit_kwargs = captured["build_kwargs"]
    assert default_kwargs["cuda_graph_bs"] == [1, 2, 4, 8, 12, 16]
    assert default_kwargs["cuda_graph_max_bs"] == 16
    assert default_kwargs["enable_torch_compile"] is False
    assert "mem_fraction_static" not in default_kwargs
    assert explicit_kwargs["cuda_graph_bs"] == [1, 2, 4, 8, 12, 16]
    assert explicit_kwargs["cuda_graph_max_bs"] == 16
    assert explicit_kwargs["enable_torch_compile"] is True
    assert explicit_kwargs["mem_fraction_static"] == 0.61
    assert captured["context_length"] == 8192
    assert captured["model_arch_override"] == "MossTTSDelaySGLangModel"
    assert captured["defer_cuda_graph_capture"] is True
    assert captured["graph_inits"] == 2
    assert captured["sampling_graph_inits"] == [
        ([1, 2, 4, 8, 12, 16], False),
        ([1, 2, 4, 8, 12, 16], False),
    ]
    assert captured["graph_init_order"] == [
        "backbone",
        "sampling",
        "backbone",
        "sampling",
    ]


def test_moss_tts_talker_torch_compile_cli_override_targets_tts_engine() -> None:
    from sglang_omni.cli.serve import apply_torch_compile_cli_overrides

    config = MossTTSPipelineConfig(model_path="model")
    apply_torch_compile_cli_overrides(
        config,
        thinker_torch_compile="default",
        talker_torch_compile="on",
        thinker_torch_compile_max_bs=None,
        talker_torch_compile_max_bs=4,
    )

    tts_engine = next(stage for stage in config.stages if stage.name == "tts_engine")
    server_args_overrides = tts_engine.factory_args["server_args_overrides"]
    assert server_args_overrides["enable_torch_compile"] is True
    assert server_args_overrides["torch_compile_max_bs"] == 4


def test_moss_tts_vocoder_uses_batch_base_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from sglang_omni.models.moss_tts import stages

    decoded_segments: list[list[list[int]]] = []

    class FakeProcessor:
        model_config = SimpleNamespace(audio_pad_code=1024, sampling_rate=16000)
        audio_tokenizer = None

    class FakeAudioTokenizer:
        sample_rate = 16000

        def decode_codes(self, segments):
            decoded_segments.extend(segment.tolist() for segment in segments)
            offset = float(len(decoded_segments) * 10)
            return [torch.tensor([offset, offset + 1], dtype=torch.float32)]

    monkeypatch.setattr(
        stages,
        "_load_moss_processor",
        lambda *args, **kwargs: FakeProcessor(),
    )
    monkeypatch.setattr(
        stages,
        "load_moss_tts_audio_tokenizer",
        lambda *args, **kwargs: FakeAudioTokenizer(),
    )

    scheduler = stages.create_vocoder_executor(
        "model",
        device="cpu",
        max_batch_size=2,
        max_batch_wait_ms=4,
    )
    first = make_payload(inputs="first")
    first.data = MossTTSState(
        delayed_audio_codes=torch.tensor(
            [
                [1, 1024],
                [2, 3],
                [1024, 4],
                [1024, 1024],
            ],
            dtype=torch.long,
        ),
        prompt_tokens=3,
        completion_tokens=5,
    ).to_dict()
    second = make_payload(inputs="second")
    second.data = MossTTSState(
        delayed_audio_codes=torch.tensor(
            [
                [5, 1024],
                [6, 7],
                [1024, 8],
                [1024, 1024],
            ],
            dtype=torch.long,
        ),
    ).to_dict()

    results = asyncio.run(scheduler._batch_fn([first, second]))

    assert scheduler._max_batch_size == 2
    assert scheduler._max_batch_wait_s == pytest.approx(0.004)
    assert decoded_segments == [
        [[1, 3], [2, 4]],
        [[5, 7], [6, 8]],
    ]
    first_audio = np.frombuffer(results[0].data["audio_waveform"], dtype=np.float32)
    second_audio = np.frombuffer(results[1].data["audio_waveform"], dtype=np.float32)
    assert first_audio.tolist() == [10.0, 11.0]
    assert second_audio.tolist() == [20.0, 21.0]
    assert results[0].data["sample_rate"] == 16000
    assert results[0].data["modality"] == "audio"
    assert "delayed_audio_codes" not in results[0].data
    assert results[0].data["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
    }


def test_moss_tts_preprocessing_loads_separate_codec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.moss_tts import request_builders as rb
    from sglang_omni.models.moss_tts import stages

    processor = SimpleNamespace(
        audio_tokenizer=None,
        model_config=SimpleNamespace(
            n_vq=32,
            audio_tokenizer_name_or_path="codec-from-model-config",
        ),
    )
    codec = SimpleNamespace()
    loaded: list[tuple[str, str, str]] = []

    def load_codec(model_path, *, device, dtype):
        loaded.append((model_path, device, dtype))
        return codec

    monkeypatch.setattr(stages, "_load_moss_processor", lambda model_path: processor)
    monkeypatch.setattr(stages, "load_moss_tts_audio_tokenizer", load_codec)

    try:
        stages.create_preprocessing_executor(
            "model",
            device="cpu",
            ref_audio_cache=False,
        )
        context = rb._QUEUE.snapshot().context
        assert context is not None
        assert context.processor is processor
        assert context.processor.audio_tokenizer is None
        assert context.reference_encoder._audio_tokenizer is codec
        assert context.reference_encoder._n_vq == 32
        assert isinstance(context.reference_encoder, stages._BatchedReferenceEncoder)
    finally:
        rb.clear_moss_tts_preprocessing_context()

    assert loaded == [("codec-from-model-config", "cpu", "float32")]


def test_moss_tts_preprocessing_uses_placement_gpu_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.moss_tts import request_builders as rb
    from sglang_omni.models.moss_tts import stages

    processor = SimpleNamespace(
        audio_tokenizer=None,
        model_config=SimpleNamespace(
            n_vq=32,
            audio_tokenizer_name_or_path="codec",
        ),
    )
    codec = SimpleNamespace()
    loaded: list[tuple[str, str, str]] = []

    def load_codec(model_path, *, device, dtype):
        loaded.append((model_path, device, dtype))
        return codec

    monkeypatch.setattr(stages, "_load_moss_processor", lambda model_path: processor)
    monkeypatch.setattr(stages, "load_moss_tts_audio_tokenizer", load_codec)

    try:
        stages.create_preprocessing_executor(
            "model",
            gpu_id=2,
            ref_audio_cache=False,
        )
        context = rb._QUEUE.snapshot().context
        assert context is not None
        assert isinstance(context.reference_encoder, stages._BatchedReferenceEncoder)
    finally:
        rb.clear_moss_tts_preprocessing_context()

    assert loaded == [("codec", "cuda:2", "float32")]


def test_moss_tts_pathlike_reference_uses_separate_codec() -> None:
    from sglang_omni.models.moss_tts import request_builders as rb

    encoded = torch.tensor([[7, 8]], dtype=torch.long)
    encoded_paths: list[str] = []

    class FakeReferenceEncoder:
        def encode(self, path: str) -> torch.Tensor:
            encoded_paths.append(path)
            return encoded

    reference = rb._reference_for_processor(
        object(),
        Path("voice.wav"),
        FakeReferenceEncoder(),
    )

    assert reference is not None
    assert reference[0] is encoded
    assert encoded_paths == ["voice.wav"]


def test_moss_tts_preprocessing_reference_cache_toggles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.moss_tts import request_builders as rb
    from sglang_omni.models.moss_tts import stages

    processor = SimpleNamespace(
        audio_tokenizer=None,
        model_config=SimpleNamespace(
            n_vq=32,
            audio_tokenizer_name_or_path="codec",
        ),
    )
    codec = SimpleNamespace(sample_rate=24000, device="cpu", model=None)
    monkeypatch.setattr(stages, "_load_moss_processor", lambda model_path: processor)
    monkeypatch.setattr(stages, "load_moss_tts_audio_tokenizer", lambda *a, **k: codec)

    try:
        stages.create_preprocessing_executor(
            "model",
            device="cpu",
            ref_audio_cache=False,
        )
        assert isinstance(
            rb._QUEUE.snapshot().context.reference_encoder,
            stages._BatchedReferenceEncoder,
        )

        monkeypatch.setenv("MOSS_REF_AUDIO_CACHE", "0")
        stages.create_preprocessing_executor("model", device="cpu")
        assert isinstance(
            rb._QUEUE.snapshot().context.reference_encoder,
            stages._BatchedReferenceEncoder,
        )

        monkeypatch.delenv("MOSS_REF_AUDIO_CACHE")
        stages.create_preprocessing_executor("model", device="cpu")
        cached = rb._QUEUE.snapshot().context.reference_encoder
        assert isinstance(cached, stages._MossTTSReferenceEncoder)
        assert cached._service._cache.max_size == 8192
    finally:
        rb.clear_moss_tts_preprocessing_context()


def test_moss_tts_processor_load_preserves_codec_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers

    from sglang_omni.models.moss_tts import stages

    model_config = SimpleNamespace(audio_tokenizer_name_or_path=None)
    tokenizer = object()

    class FakeProcessor:
        @classmethod
        def get_processor_dict(cls, checkpoint_dir):
            assert checkpoint_dir == "model"
            return (
                {
                    "audio_tokenizer_name_or_path": "codec-from-processor-config",
                },
                {},
            )

        def __init__(self, *, tokenizer, audio_tokenizer, model_config):
            self.tokenizer = tokenizer
            self.audio_tokenizer = audio_tokenizer
            self.model_config = model_config

    monkeypatch.setattr(stages, "load_moss_processor_class", lambda _: FakeProcessor)
    monkeypatch.setattr(stages, "moss_transformers_processor_compat", nullcontext)
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: model_config,
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )

    processor = stages._load_moss_processor("model")

    assert processor.audio_tokenizer is None
    assert (
        stages._resolve_audio_tokenizer_model_path(processor, None)
        == "codec-from-processor-config"
    )


@pytest.mark.parametrize("configured_model_path", [None, ""])
def test_moss_tts_codec_path_uses_default_for_falsey_checkpoint_metadata(
    configured_model_path,
) -> None:
    from sglang_omni.models.moss_tts import stages

    processor = SimpleNamespace(
        model_config=SimpleNamespace(
            audio_tokenizer_name_or_path=configured_model_path,
        )
    )

    assert (
        stages._resolve_audio_tokenizer_model_path(processor, None)
        == stages.DEFAULT_MOSS_TTS_AUDIO_TOKENIZER
    )


def test_moss_tts_vocoder_honors_explicit_codec_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.moss_tts import stages

    processor = SimpleNamespace(
        audio_tokenizer=None,
        model_config=SimpleNamespace(audio_pad_code=1024, sampling_rate=24000),
    )
    codec = SimpleNamespace(sample_rate=24000)
    loaded: list[tuple[str, str, str]] = []

    def load_codec(model_path, *, device, dtype):
        loaded.append((model_path, device, dtype))
        return codec

    monkeypatch.setattr(stages, "_load_moss_processor", lambda model_path: processor)
    monkeypatch.setattr(stages, "load_moss_tts_audio_tokenizer", load_codec)

    stages.create_vocoder_executor(
        "model",
        device="cpu",
        gpu_id=2,
        codec_model_path="explicit-codec",
    )

    assert processor.audio_tokenizer is None
    assert loaded == [("explicit-codec", "cpu", "float32")]


def test_moss_tts_audio_tokenizer_preserves_processor_code_layout() -> None:
    from sglang_omni.models.moss_tts.audio_tokenizer import MossTTSAudioTokenizer

    class FakeCodec:
        config = SimpleNamespace(sampling_rate=24000)

        def __init__(self) -> None:
            self.prepared: list[torch.Tensor] = []
            self.decode_args = None

        def batch_encode(self, waveforms, num_quantizers):
            self.prepared = [wav.detach().clone() for wav in waveforms]
            assert num_quantizers == 2
            return SimpleNamespace(
                audio_codes=torch.tensor(
                    [
                        [[1, 2, 3], [4, 5, 0]],
                        [[6, 7, 8], [9, 10, 0]],
                    ],
                    dtype=torch.long,
                ),
                audio_codes_lengths=torch.tensor([3, 2]),
            )

        def decode(
            self,
            audio_codes,
            *,
            padding_mask,
            return_dict,
            chunk_duration,
        ):
            self.decode_args = (
                audio_codes.detach().clone(),
                padding_mask.detach().clone(),
                return_dict,
                chunk_duration,
            )
            return SimpleNamespace(
                audio=torch.tensor(
                    [
                        [[0.1, 0.2, 0.3]],
                        [[0.4, 0.5, 0.0]],
                    ],
                    dtype=torch.float32,
                ),
                audio_lengths=torch.tensor([3, 2]),
            )

    model = FakeCodec()
    tokenizer = MossTTSAudioTokenizer(model, device="cpu")
    encoded = tokenizer.encode_waveforms(
        [
            (torch.tensor([[0.1, 0.2], [0.3, 0.4]]), 24000),
            (torch.tensor([0.2, 0.4]), 24000),
        ],
        num_quantizers=2,
    )

    assert [tuple(codes.shape) for codes in encoded] == [(3, 2), (2, 2)]
    assert encoded[0].tolist() == [[1, 6], [2, 7], [3, 8]]
    assert encoded[1].tolist() == [[4, 9], [5, 10]]
    assert all(tuple(wav.shape) == (2,) for wav in model.prepared)

    decoded = tokenizer.decode_codes(encoded)

    assert [wav.tolist() for wav in decoded] == [
        pytest.approx([0.1, 0.2, 0.3]),
        pytest.approx([0.4, 0.5]),
    ]
    audio_codes, padding_mask, return_dict, chunk_duration = model.decode_args
    assert tuple(audio_codes.shape) == (2, 2, 3)
    assert padding_mask.tolist() == [[True, True, True], [True, True, False]]
    assert return_dict is True
    assert chunk_duration == 8


@pytest.mark.parametrize(
    ("device", "dtype", "expected"),
    [
        ("cuda:0", torch.bfloat16, [("cuda", torch.bfloat16)]),
        ("cuda:0", torch.float16, [("cuda", torch.float16)]),
        ("cuda:0", torch.float32, []),
        ("cpu", torch.bfloat16, []),
    ],
)
def test_moss_tts_audio_tokenizer_encode_autocast_matches_model_dtype(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    dtype: torch.dtype,
    expected: list[tuple[str, torch.dtype]],
) -> None:
    from sglang_omni.models.moss_tts import audio_tokenizer as audio_tokenizer_mod

    class FakeCodec(torch.nn.Module):
        config = SimpleNamespace(sampling_rate=24000)

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(1, dtype=dtype))

        def batch_encode(self, waveforms, num_quantizers):
            return SimpleNamespace(
                audio_codes=torch.ones(1, 1, 1, dtype=torch.long),
                audio_codes_lengths=torch.ones(1, dtype=torch.long),
            )

    calls: list[tuple[str, torch.dtype]] = []

    def fake_autocast(*, device_type: str, dtype: torch.dtype):
        calls.append((device_type, dtype))
        return nullcontext()

    monkeypatch.setattr(audio_tokenizer_mod.torch, "autocast", fake_autocast)
    tokenizer = audio_tokenizer_mod.MossTTSAudioTokenizer(FakeCodec(), device=device)
    monkeypatch.setattr(
        tokenizer,
        "_prepare_waveform",
        lambda wav, sample_rate: wav,
    )

    tokenizer.encode_waveforms([(torch.zeros(1), 24000)])

    assert calls == expected


@pytest.mark.parametrize(
    ("device", "dtype", "expected"),
    [
        ("cuda:0", torch.bfloat16, [("cuda", torch.bfloat16)]),
        ("cuda:0", torch.float16, [("cuda", torch.float16)]),
        ("cuda:0", torch.float32, []),
        ("cpu", torch.bfloat16, []),
    ],
)
def test_moss_tts_audio_tokenizer_autocast_matches_model_dtype(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    dtype: torch.dtype,
    expected: list[tuple[str, torch.dtype]],
) -> None:
    from sglang_omni.models.moss_tts import audio_tokenizer as audio_tokenizer_mod

    class FakeCodec(torch.nn.Module):
        config = SimpleNamespace(sampling_rate=24000)

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(1, dtype=dtype))

    calls: list[tuple[str, torch.dtype]] = []

    def fake_autocast(*, device_type: str, dtype: torch.dtype):
        calls.append((device_type, dtype))
        return nullcontext()

    monkeypatch.setattr(audio_tokenizer_mod.torch, "autocast", fake_autocast)
    tokenizer = audio_tokenizer_mod.MossTTSAudioTokenizer(FakeCodec(), device=device)

    with tokenizer._autocast():
        pass

    assert calls == expected


def test_moss_tts_maps_references_token_count_and_checkpoint_defaults() -> None:
    payload = make_payload(
        inputs={
            "text": "${token:120}hello [pause 0.5s] ni3 hao3 /hello/",
            "references": [{"audio_path": "voice.wav", "text": "reference"}],
        },
        params={"temperature": 0.8, "top_p": 0.8, "top_k": 30},
        tts_params={"language": "en"},
    )

    state = build_moss_tts_state(payload)

    # ${token:N} is stripped from the text (it becomes the processor's tokens=
    # field), while [pause Xs], pinyin, and IPA markup pass through unchanged.
    assert state.text == "hello [pause 0.5s] ni3 hao3 /hello/"
    assert "${token" not in state.text
    assert state.ref_audio == "voice.wav"
    assert state.ref_text == "reference"
    assert state.language == "en"
    assert state.token_count == 120
    assert state.generation_kwargs["max_new_tokens"] == 4096
    # Defaults follow the upstream checkpoint's generate() (sampling), not greedy.
    assert state.generation_kwargs["text_temperature"] == 1.5
    assert state.generation_kwargs["text_top_p"] == 1.0
    assert state.generation_kwargs["text_top_k"] == 50
    assert state.generation_kwargs["audio_temperature"] == 1.7
    assert state.generation_kwargs["audio_top_p"] == 0.8
    assert state.generation_kwargs["audio_top_k"] == 25
    assert state.generation_kwargs["audio_repetition_penalty"] == 1.0


def test_moss_tts_benchmark_auto_token_count_uses_openmoss_estimate() -> None:
    sample = SampleInput(
        sample_id="sample-1",
        ref_text="reference",
        ref_audio="ref.wav",
        target_text="hello world",
    )

    payload = _build_tts_payload(
        sample,
        "OpenMOSS-Team/MOSS-TTS-v1.5",
        token_count=MOSS_TTS_TOKEN_COUNT_AUTO,
    )

    assert payload["token_count"] == estimate_moss_tts_duration_tokens("hello world")
    assert payload["token_count"] == 32


def test_tts_benchmark_payload_supports_streaming_pcm_control() -> None:
    sample = SampleInput(
        sample_id="sample-1",
        ref_text="reference",
        ref_audio="ref.wav",
        target_text="hello world",
    )

    payload = _build_tts_payload(
        sample,
        "OpenMOSS-Team/MOSS-TTS-v1.5",
        stream=True,
        response_format="pcm",
        initial_codec_chunk_frames=1,
    )

    assert payload["stream"] is True
    assert payload["response_format"] == "pcm"
    assert payload["initial_codec_chunk_frames"] == 1


def test_tts_benchmark_raw_audio_transport_forces_pcm_payload() -> None:
    sample = SampleInput(
        sample_id="sample-1",
        ref_text="reference",
        ref_audio="ref.wav",
        target_text="hello world",
    )

    payload = _build_tts_payload(
        sample,
        "OpenMOSS-Team/MOSS-TTS-v1.5",
        stream=True,
        response_format="wav",
    )

    assert payload["response_format"] == "pcm"


def test_tts_benchmark_raw_pcm_uses_http_chunk_boundaries() -> None:
    class FakeContent:
        async def iter_chunks(self):
            yield b"abcd", False
            yield b"efghij", True
            yield b"klmnop", True

    response = SimpleNamespace(
        headers={
            "Content-Type": "audio/pcm",
            "x-sample-rate": "4",
            "x-channels": "1",
            "x-bit-depth": "16",
        },
        content=FakeContent(),
    )
    result = RequestResult(request_id="raw-pcm")

    asyncio.run(
        _handle_raw_pcm_streaming_response(
            response,
            result,
            start_time=0.0,
            save_audio_dir=None,
        )
    )

    assert result.is_success
    assert result.audio_chunk_count == 2
    assert result.first_audio_payload_bytes == 10
    assert result.audio_duration_s == pytest.approx(2.0)


def test_tts_benchmark_raw_pcm_rejects_sse_response() -> None:
    class FakeContent:
        async def iter_chunks(self):
            yield b"data: [DONE]\n\n", True

    response = SimpleNamespace(
        headers={"Content-Type": "text/event-stream"},
        content=FakeContent(),
    )
    result = RequestResult(request_id="raw-pcm")

    asyncio.run(
        _handle_raw_pcm_streaming_response(
            response,
            result,
            start_time=0.0,
            save_audio_dir=None,
        )
    )

    assert not result.is_success
    assert "audio/pcm" in result.error


def test_tts_benchmark_raw_pcm_rejects_partial_frame() -> None:
    class FakeContent:
        async def iter_chunks(self):
            yield b"abc", True

    response = SimpleNamespace(
        headers={
            "Content-Type": "audio/pcm",
            "x-sample-rate": "4",
            "x-channels": "1",
            "x-bit-depth": "16",
        },
        content=FakeContent(),
    )
    result = RequestResult(request_id="raw-pcm")

    asyncio.run(
        _handle_raw_pcm_streaming_response(
            response,
            result,
            start_time=0.0,
            save_audio_dir=None,
        )
    )

    assert not result.is_success
    assert "partial audio frame" in result.error


def test_moss_tts_preserves_explicit_standard_sampling_values() -> None:
    payload = make_payload(
        inputs="hello",
        params={"temperature": 0.7, "top_p": 0.9, "top_k": 40},
        tts_params={
            "explicit_generation_params": ["temperature", "top_p", "top_k"],
            "token_count": 42,
        },
    )

    state = build_moss_tts_state(payload)

    assert state.token_count == 42
    assert state.generation_kwargs["text_temperature"] == 0.7
    assert state.generation_kwargs["audio_temperature"] == 0.7
    assert state.generation_kwargs["text_top_p"] == 0.9
    assert state.generation_kwargs["audio_top_k"] == 40


def test_moss_row_cache_keys_are_content_based() -> None:
    rows = torch.tensor([[1, 1024, 1024], [2, 1024, 1024]], dtype=torch.long)
    same = rows.clone()
    different = torch.tensor([[1, 1024, 1024], [2, 1024, 1023]], dtype=torch.long)

    assert build_row_cache_key_ids(rows) == build_row_cache_key_ids(same)
    assert build_row_cache_key_ids(rows) != build_row_cache_key_ids(different)


def test_moss_preprocess_and_sglang_request_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang(monkeypatch)

    class FakeProcessor:
        def __init__(self) -> None:
            self.message_kwargs = None

        def build_user_message(self, **kwargs):
            self.message_kwargs = kwargs
            return {"role": "user", **kwargs}

        def __call__(self, conversations, mode):
            assert mode == "generation"
            assert conversations[0][0]["text"] == "hello"
            return {
                "input_ids": torch.tensor(
                    [
                        [
                            [1, 1024, 1024],
                            [151644, 1024, 1024],
                            [198, 1024, 1024],
                        ]
                    ],
                    dtype=torch.long,
                )
            }

    class FakeReferenceEncoder:
        def __init__(self) -> None:
            self.paths: list[str] = []

        def encode(self, path: str) -> torch.Tensor:
            self.paths.append(path)
            return torch.tensor([[7, 8]], dtype=torch.long)

    processor = FakeProcessor()
    reference_encoder = FakeReferenceEncoder()
    payload = make_payload(
        inputs={
            "text": "hello",
            "references": [{"audio_path": "voice.wav"}],
        },
        params={"max_new_tokens": 12},
        tts_params={"token_count": 80, "language": "en"},
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            vocab_size_list=[200000, 1025, 1025],
            im_end_token_id=151645,
            im_start_token_id=151644,
            audio_start_token_id=151652,
            audio_assistant_gen_slot_token_id=151656,
        )
    )

    try:
        set_moss_tts_preprocessing_context(
            processor=processor,
            reference_encoder=reference_encoder,
        )
        prepared_payload = preprocess_moss_tts_payload(payload)
        data = build_sglang_moss_tts_request(prepared_payload, model=model)
    finally:
        clear_moss_tts_preprocessing_context()

    assert processor.message_kwargs["tokens"] == 80
    assert processor.message_kwargs["language"] == "en"
    assert processor.message_kwargs["reference"][0].tolist() == [[7, 8]]
    assert reference_encoder.paths == ["voice.wav"]
    assert data.req._input_embeds_are_projected is True
    assert data.input_embeds_are_projected is True
    assert data.max_new_tokens == 12
    assert data.prompt_rows.shape == (3, 3)
    assert data.state.assistant_start_length == 0
    assert data.req.sampling_params.stop_token_ids == [151645]


def _moss_delay_model() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            vocab_size_list=[200000, 1025, 1025],
            im_end_token_id=151645,
            im_start_token_id=151644,
            audio_start_token_id=151652,
            audio_assistant_gen_slot_token_id=151656,
            audio_assistant_delay_slot_token_id=151662,
        )
    )


class _ConstantMossProcessor:
    def build_user_message(self, **kwargs):
        return {"role": "user", **kwargs}

    def __call__(self, conversations, mode):
        del conversations, mode
        return {
            "input_ids": torch.tensor(
                [
                    [
                        [1, 1024, 1024],
                        [151644, 1024, 1024],
                        [198, 1024, 1024],
                    ]
                ],
                dtype=torch.long,
            )
        }


def _build_moss_sglang_request(*, request_id: str = "req-moss"):
    payload = make_payload(inputs="hello", request_id=request_id)
    try:
        set_moss_tts_preprocessing_context(processor=_ConstantMossProcessor())
        prepared_payload = preprocess_moss_tts_payload(payload)
        return build_sglang_moss_tts_request(
            prepared_payload, model=_moss_delay_model()
        )
    finally:
        clear_moss_tts_preprocessing_context()


def test_moss_request_lifetime_extra_key_is_unique_and_survives_retract() -> None:
    # note (Richard Wang): same rid can recur so extra_key must differ
    first = _build_moss_sglang_request(request_id="shared-moss-id")
    second = _build_moss_sglang_request(request_id="shared-moss-id")

    assert first.req.rid == second.req.rid == "shared-moss-id"
    assert first.req.extra_key
    assert second.req.extra_key
    assert first.req.extra_key.startswith("moss_tts:")
    assert second.req.extra_key.startswith("moss_tts:")
    assert first.req.extra_key != second.req.extra_key
    assert list(first.req.origin_input_ids) == list(second.req.origin_input_ids)

    kept = first.req.extra_key
    first.req.reset_for_retract()
    assert first.req.extra_key == kept


def test_moss_lifetime_extra_key_isolates_delay_slot_generated_prefix() -> None:
    # note (Richard Wang): text channel keys can match while RVQ differs
    from array import array

    from sglang.srt.mem_cache.radix_cache import RadixKey

    delay_slot = int(_moss_delay_model().config.audio_assistant_delay_slot_token_id)
    row_a = torch.tensor([delay_slot, 11, 22], dtype=torch.long)
    row_b = torch.tensor([delay_slot, 99, 88], dtype=torch.long)
    assert int(row_a[0]) == int(row_b[0])
    assert not torch.equal(row_a[1:], row_b[1:])

    first = _build_moss_sglang_request(request_id="shared-moss-id")
    second = _build_moss_sglang_request(request_id="shared-moss-id")
    gen_ids = [delay_slot, delay_slot]
    fill_a = array("q", list(first.req.origin_input_ids) + gen_ids)
    fill_b = array("q", list(second.req.origin_input_ids) + gen_ids)
    assert fill_a == fill_b

    colliding = RadixKey(fill_a, extra_key=None)
    assert colliding.match(RadixKey(fill_b, extra_key=None)) == len(fill_a)

    with pytest.raises(ValueError, match="matching extra_key"):
        RadixKey(fill_a, first.req.extra_key).match(
            RadixKey(fill_b, second.req.extra_key)
        )


def test_moss_delay_runner_samples_audio_and_appends_feedback() -> None:
    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner

    cfg = SimpleNamespace(
        pad_token_id=0,
        audio_start_token_id=10,
        audio_end_token_id=11,
        audio_assistant_gen_slot_token_id=12,
        audio_assistant_delay_slot_token_id=13,
        audio_pad_code=4,
        im_end_token_id=14,
    )
    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.model = SimpleNamespace(
        config=cfg,
        hidden_size=3,
        device=torch.device("cpu"),
        _prepare_multi_modal_inputs=lambda rows: rows.to(torch.float32)[:, :3],
    )
    data = SimpleNamespace(
        audio_length=0,
        delayed_length=_INF_DELAY,
        is_audio=False,
        generation_steps=0,
        sampling_seed=0,
        text_temperature=0.0,
        text_top_p=1.0,
        text_top_k=-1,
        audio_temperature=0.0,
        audio_top_p=1.0,
        audio_top_k=-1,
        audio_repetition_penalty=1.0,
        prompt_rows=None,
        output_rows=[],
        pending_feedback_queue=[],
    )
    text_logits = torch.full((1, 20), -100.0)
    text_logits[0, cfg.audio_start_token_id] = 10.0
    audio0_logits = torch.tensor([[-1.0, 0.0, 5.0, 1.0, -100.0]])
    audio1_logits = torch.tensor([[-1.0, 6.0, 0.0, 1.0, -100.0]])

    rows = runner._sample_rows(
        [text_logits, audio0_logits, audio1_logits],
        [data],
        n_vq=2,
    )
    text_token = int(rows[0, 0].item())
    audio_tokens = rows[0, 1:]

    assert text_token == cfg.audio_start_token_id
    assert audio_tokens.tolist() == [cfg.audio_pad_code, cfg.audio_pad_code]
    assert data.is_audio is True
    assert data.audio_length == 1

    data.generation_steps = 1
    text_logits[0] = -100.0
    text_logits[0, cfg.audio_assistant_gen_slot_token_id] = 10.0
    rows = runner._sample_rows(
        [text_logits, audio0_logits, audio1_logits],
        [data],
        n_vq=2,
    )
    text_token = int(rows[0, 0].item())
    audio_tokens = rows[0, 1:]

    assert text_token == cfg.audio_assistant_gen_slot_token_id
    assert audio_tokens.tolist() == [2, cfg.audio_pad_code]
    assert data.audio_length == 2


def test_moss_delay_runner_restricts_text_only_while_audio_is_active() -> None:
    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner

    cfg = SimpleNamespace(
        pad_token_id=0,
        audio_start_token_id=10,
        audio_end_token_id=11,
        audio_assistant_gen_slot_token_id=12,
        audio_assistant_delay_slot_token_id=13,
        audio_pad_code=4,
        im_end_token_id=14,
    )
    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.model = SimpleNamespace(
        config=cfg,
        text_control_token_ids=torch.tensor([12, 13], dtype=torch.long),
    )
    data = SimpleNamespace(
        delay_state=torch.tensor([1, torch.iinfo(torch.int64).max, 1]),
        generation_steps=1,
        sampling_seed=0,
        text_temperature=0.0,
        text_top_p=1.0,
        text_top_k=-1,
        audio_temperature=0.0,
        audio_top_p=1.0,
        audio_top_k=-1,
        audio_repetition_penalty=1.0,
        prompt_rows=None,
        output_rows=[],
    )
    control_logits = torch.tensor([[5.0, -5.0]])
    audio0_logits = torch.tensor([[-1.0, 0.0, 5.0, 1.0, -100.0]])
    audio1_logits = torch.tensor([[-1.0, 6.0, 0.0, 1.0, -100.0]])

    rows = runner._sample_rows(
        [control_logits, audio0_logits, audio1_logits],
        [data],
        n_vq=2,
        is_audio=True,
    )
    assert int(rows[0, 0]) == cfg.audio_assistant_gen_slot_token_id

    # Once the Delay tail reaches n_vq, audio_end is forced. The next step must
    # return to the original full-vocabulary text path; it is not forced to EOS
    # inside the two-token sampler.
    data.delay_state = torch.tensor([1, 2, 1])
    data.generation_steps = 3
    rows = runner._sample_rows(
        [control_logits, audio0_logits, audio1_logits],
        [data],
        n_vq=2,
        is_audio=True,
    )
    assert int(rows[0, 0]) == cfg.audio_end_token_id


def test_moss_collect_step_uses_full_text_path_outside_audio() -> None:
    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner

    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.model = SimpleNamespace(
        device=torch.device("cpu"),
        _prepare_multi_modal_inputs=lambda rows: rows.to(torch.float32),
    )
    seen: dict[str, bool] = {}

    def channel_logits(result, forward_batch, *, is_audio=False):
        del result, forward_batch
        seen["head"] = is_audio
        return [torch.zeros(1, 20), torch.zeros(1, 5)]

    def sample_rows(channel_logits, datas, *, n_vq, is_audio=False):
        del channel_logits, datas, n_vq
        seen["sampler"] = is_audio
        return torch.tensor([[0, 4]], dtype=torch.long)

    runner._channel_logits_from_result = channel_logits
    runner._sample_rows = sample_rows
    result = SimpleNamespace(next_token_ids=None)
    schedule_batch = SimpleNamespace(output_ids=None)
    request = SimpleNamespace(data=SimpleNamespace(is_audio=False))

    runner._collect_moss_step(result, object(), schedule_batch, [request])

    assert seen == {"head": False, "sampler": False}


def test_moss_collect_step_mixed_batch_uses_full_text_path() -> None:
    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner

    cfg = SimpleNamespace(
        pad_token_id=0,
        audio_start_token_id=10,
        audio_end_token_id=11,
        audio_assistant_gen_slot_token_id=12,
        audio_assistant_delay_slot_token_id=13,
        audio_pad_code=4,
        im_end_token_id=14,
    )
    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.model = SimpleNamespace(
        config=cfg,
        device=torch.device("cpu"),
        _prepare_multi_modal_inputs=lambda rows: rows.to(torch.float32),
    )
    seen: dict[str, bool] = {}
    text_logits = torch.full((2, 20), -100.0)
    text_logits[0, 5] = 100.0
    text_logits[0, cfg.audio_assistant_gen_slot_token_id] = 10.0
    text_logits[0, cfg.audio_assistant_delay_slot_token_id] = 9.0
    text_logits[1, 5] = 10.0
    text_logits[1, cfg.audio_assistant_gen_slot_token_id] = 100.0
    audio_logits = torch.tensor(
        [
            [-1.0, 0.0, 5.0, 1.0, -100.0],
            [-1.0, 0.0, 1.0, 5.0, -100.0],
        ]
    )

    def channel_logits(result, forward_batch, *, is_audio=False):
        del result, forward_batch
        seen["head"] = is_audio
        return [text_logits, audio_logits]

    sample_rows = runner._sample_rows

    def sample_rows_spy(channel_logits, datas, *, n_vq, is_audio=False):
        seen["sampler"] = is_audio
        return sample_rows(channel_logits, datas, n_vq=n_vq, is_audio=is_audio)

    runner._channel_logits_from_result = channel_logits
    runner._sample_rows = sample_rows_spy

    def data(*, is_audio: bool) -> SimpleNamespace:
        return SimpleNamespace(
            delay_state=torch.tensor(
                [int(is_audio), torch.iinfo(torch.int64).max, int(is_audio)]
            ),
            generation_steps=1,
            sampling_seed=0,
            text_temperature=0.0,
            text_top_p=1.0,
            text_top_k=-1,
            audio_temperature=0.0,
            audio_top_p=1.0,
            audio_top_k=-1,
            audio_repetition_penalty=1.0,
            is_audio=is_audio,
        )

    requests = [
        SimpleNamespace(data=data(is_audio=True)),
        SimpleNamespace(data=data(is_audio=False)),
    ]
    result = SimpleNamespace(next_token_ids=None)
    schedule_batch = SimpleNamespace(output_ids=None)

    runner._collect_moss_step(result, object(), schedule_batch, requests)

    assert seen == {"head": False, "sampler": False}
    assert runner._pending_rows[:, 0].tolist() == [
        cfg.audio_assistant_gen_slot_token_id,
        5,
    ]


def test_moss_collect_step_requires_is_audio_state() -> None:
    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner

    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    request = SimpleNamespace(data=SimpleNamespace())

    with pytest.raises(AttributeError, match="is_audio"):
        runner._collect_moss_step(object(), object(), object(), [request])


def test_moss_prefill_forward_uses_prompt_row_embeds() -> None:
    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner

    class FakeModel:
        dtype = torch.float32
        hidden_size = 2

        def _prepare_multi_modal_inputs(self, rows):
            return rows.to(torch.float32)[:, :2]

    model = FakeModel()
    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.model = model
    prompt_rows = torch.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        dtype=torch.long,
    )
    sched_req = SimpleNamespace(
        data=SimpleNamespace(
            req=SimpleNamespace(
                extend_range=SimpleNamespace(length=2), prefix_indices=[0]
            ),
            prompt_rows=prompt_rows,
            output_rows=[],
        )
    )
    forward_batch = SimpleNamespace(
        input_ids=torch.tensor([123456, 123457], dtype=torch.long),
        positions=torch.arange(2),
        mrope_positions=None,
    )

    result = runner.custom_prefill_forward(forward_batch, object(), [sched_req])

    assert result is None
    assert torch.equal(forward_batch.input_ids, torch.tensor([123456, 123457]))
    assert torch.equal(
        forward_batch.input_embeds,
        torch.tensor([[4.0, 5.0], [7.0, 8.0]]),
    )


def _retract_runner(hidden_size: int = 2, decode_embedding=None):
    # note (Richard Wang): skips __init__ to test _build_prefill_input_embeds
    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner

    model = SimpleNamespace(
        dtype=torch.float32,
        hidden_size=hidden_size,
        _prepare_multi_modal_inputs=lambda rows: rows.to(torch.float32)[
            :, :hidden_size
        ],
    )
    if decode_embedding is not None:
        model._decode_input_embedding = decode_embedding
    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.model = model
    return runner


def _retract_sched_req(*, prompt_rows, output_rows, extend_len, feedback_queue):
    return SimpleNamespace(
        data=SimpleNamespace(
            req=SimpleNamespace(
                rid="a",
                extend_range=SimpleNamespace(length=extend_len),
                prefix_indices=[],
            ),
            prompt_rows=prompt_rows,
            output_rows=output_rows,
            pending_feedback_queue=feedback_queue,
        )
    )


def test_moss_reprefill_after_retract_concatenates_output_rows() -> None:
    prompt_rows = torch.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        dtype=torch.long,
    )
    generated = [
        torch.tensor([10, 11, 12], dtype=torch.long),
        torch.tensor([13, 14, 15], dtype=torch.long),
    ]
    sched_req = _retract_sched_req(
        prompt_rows=prompt_rows,
        output_rows=generated,
        extend_len=5,
        feedback_queue=deque(),
    )
    forward_batch = SimpleNamespace(input_ids=torch.zeros(5, dtype=torch.long))

    embeds = _retract_runner()._build_prefill_input_embeds(forward_batch, [sched_req])

    assert torch.equal(
        embeds,
        torch.tensor(
            [
                [1.0, 2.0],
                [4.0, 5.0],
                [7.0, 8.0],
                [10.0, 11.0],
                [13.0, 14.0],
            ]
        ),
    )


def test_moss_reprefill_without_generated_rows_fails_loudly() -> None:
    sched_req = _retract_sched_req(
        prompt_rows=torch.zeros((3, 3), dtype=torch.long),
        output_rows=[],
        extend_len=5,
        feedback_queue=[],
    )
    forward_batch = SimpleNamespace(input_ids=torch.zeros(5, dtype=torch.long))

    with pytest.raises(RuntimeError, match="prefill row mismatch"):
        _retract_runner()._build_prefill_input_embeds(forward_batch, [sched_req])


def test_moss_reprefill_mismatch_does_not_clear_feedback_queue() -> None:
    stranded = torch.full((2,), 7.5)
    queue = deque([stranded])
    sched_req = _retract_sched_req(
        prompt_rows=torch.tensor([[1, 2, 3]], dtype=torch.long),
        output_rows=[torch.tensor([4, 5, 6], dtype=torch.long)],
        extend_len=3,
        feedback_queue=queue,
    )
    forward_batch = SimpleNamespace(input_ids=torch.zeros(3, dtype=torch.long))

    with pytest.raises(RuntimeError, match="prefill row mismatch"):
        _retract_runner()._build_prefill_input_embeds(forward_batch, [sched_req])

    assert len(queue) == 1
    assert torch.equal(queue[0], stranded)


def test_moss_reprefill_discards_stranded_feedback() -> None:
    # note (Richard Wang): resume must consume the new frame, not the stale row
    embedding = torch.nn.Embedding(4, 3)
    runner = _retract_runner(hidden_size=3, decode_embedding=embedding)
    old_feedback = torch.full((3,), 1.0)
    new_feedback = torch.full((3,), 9.0)
    sched_req = _retract_sched_req(
        prompt_rows=torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
        output_rows=[torch.tensor([7, 8, 9], dtype=torch.long)],
        extend_len=3,
        feedback_queue=deque([old_feedback.clone()]),
    )
    data = sched_req.data
    prefill_batch = SimpleNamespace(input_ids=torch.zeros(3, dtype=torch.long))

    runner._build_prefill_input_embeds(prefill_batch, [sched_req])
    assert list(data.pending_feedback_queue) == []

    data.pending_feedback_queue.append(new_feedback)
    decode_batch = SimpleNamespace(input_ids=torch.tensor([99], dtype=torch.long))
    runner._write_decode_input_embedding(decode_batch, [sched_req])

    assert torch.equal(embedding.weight[0].detach(), new_feedback)
    assert list(data.pending_feedback_queue) == []


def test_moss_decode_feedback_uses_row_id_embedding() -> None:
    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner

    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    embedding = torch.nn.Embedding(4, 3)
    runner.model = SimpleNamespace(
        hidden_size=3,
        _decode_input_embedding=embedding,
    )
    forward_batch = SimpleNamespace(
        input_ids=torch.full((2,), 99, dtype=torch.long),
    )
    requests = [
        SimpleNamespace(data=SimpleNamespace(pending_feedback_queue=[torch.ones(3)])),
        SimpleNamespace(
            data=SimpleNamespace(pending_feedback_queue=[torch.full((3,), 2.0)])
        ),
    ]

    runner._write_decode_input_embedding(forward_batch, requests)

    assert forward_batch.input_ids.tolist() == [0, 1]
    assert torch.equal(embedding.weight[0].detach(), torch.ones(3))
    assert torch.equal(embedding.weight[1].detach(), torch.full((3,), 2.0))
    assert requests[0].data.pending_feedback_queue == []
    assert requests[1].data.pending_feedback_queue == []


def test_moss_channel_logits_fallback_uses_hidden_states() -> None:
    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner

    class FakeModel:
        def __init__(self) -> None:
            self.seen_hidden = None
            self.seen_forward_batch = None

        def compute_channel_logits(self, hidden_states, forward_batch):
            self.seen_hidden = hidden_states
            self.seen_forward_batch = forward_batch
            return [hidden_states + 1, hidden_states + 2]

    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.model = FakeModel()
    forward_batch = object()
    hidden = torch.arange(6, dtype=torch.float32).view(2, 1, 3)
    result = SimpleNamespace(
        logits_output=SimpleNamespace(
            customized_info=None,
            hidden_states=hidden,
        )
    )

    logits = runner._channel_logits_from_result(result, forward_batch)

    expected_hidden = hidden[:, -1, :]
    assert torch.equal(runner.model.seen_hidden, expected_hidden)
    assert runner.model.seen_forward_batch is forward_batch
    assert torch.equal(logits[0], expected_hidden + 1)
    assert torch.equal(logits[1], expected_hidden + 2)


def test_moss_forward_ignores_graph_mrope_placeholder() -> None:
    from sglang_omni.models.moss_tts.sglang_model import MossTTSDelaySGLangModel

    class FakeBackbone:
        def __init__(self) -> None:
            self.positions = None

        def __call__(
            self,
            *,
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors,
        ):
            del input_ids, forward_batch, pp_proxy_tensors
            self.positions = positions
            return input_embeds

    backbone = FakeBackbone()
    model = SimpleNamespace(
        pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
        model=backbone,
        _prepare_multi_modal_inputs=lambda input_ids: torch.ones(input_ids.shape[0], 3),
        _select_sample_hidden_states=lambda hidden_states, forward_batch: hidden_states,
    )
    positions = torch.arange(2, dtype=torch.long)
    forward_batch = SimpleNamespace(
        mrope_positions=torch.zeros((3, 2), dtype=torch.long),
        forward_mode=SimpleNamespace(is_decode=lambda: False),
    )

    MossTTSDelaySGLangModel.forward(
        model,
        input_ids=torch.arange(2, dtype=torch.long),
        positions=positions,
        forward_batch=forward_batch,
    )

    assert backbone.positions is positions


def test_moss_channel_logits_use_decode_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    from sglang_omni.models.moss_tts import sglang_model
    from sglang_omni.models.moss_tts.sglang_model import MossTTSDelaySGLangModel

    metadata = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        next_token_logits_buffer=object(),
    )
    monkeypatch.setattr(
        sglang_model.LogitsMetadata,
        "from_forward_batch",
        classmethod(lambda cls, forward_batch: metadata),
    )

    class FakeProcessor:
        def __init__(self) -> None:
            self.seen_metadata = None

        def __call__(self, input_ids, *, hidden_states, lm_head, logits_metadata):
            del input_ids, lm_head
            self.seen_metadata = logits_metadata
            return SimpleNamespace(next_token_logits=hidden_states)

    processor = FakeProcessor()
    model = SimpleNamespace(
        logits_processors=[processor],
        lm_heads=[object()],
    )
    hidden_states = torch.ones(2, 3)

    outputs = MossTTSDelaySGLangModel.compute_channel_outputs(
        model,
        hidden_states,
        forward_batch=object(),
    )

    assert outputs[0].next_token_logits is hidden_states
    assert processor.seen_metadata is metadata
    assert metadata.forward_mode is ForwardMode.DECODE
    assert metadata.next_token_logits_buffer is None


def test_moss_text_control_logits_select_from_full_processor_output() -> None:
    from sglang_omni.models.moss_tts.sglang_model import MossTTSDelaySGLangModel

    full_text_logits = torch.arange(20, dtype=torch.float32).view(1, 20)
    audio_logits = torch.tensor([[1.0, 2.0, 3.0]])
    model = SimpleNamespace(
        _text_control_token_ids=torch.tensor([12, 13], dtype=torch.long),
        compute_channel_outputs=lambda hidden_states, forward_batch: [
            SimpleNamespace(next_token_logits=full_text_logits),
            SimpleNamespace(next_token_logits=audio_logits),
        ],
    )

    logits = MossTTSDelaySGLangModel.compute_channel_logits(
        model,
        hidden_states=torch.ones(1, 4),
        forward_batch=object(),
        is_audio=True,
    )

    assert torch.equal(logits[0], full_text_logits[:, [12, 13]])
    assert logits[1] is audio_logits


def test_moss_post_process_outputs_skips_im_end() -> None:
    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner

    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.model = SimpleNamespace(
        config=SimpleNamespace(
            im_end_token_id=14,
            audio_start_token_id=10,
            audio_end_token_id=11,
        )
    )
    runner._pending_rows = torch.tensor([[12, 2, 4], [14, 4, 4]], dtype=torch.long)
    runner._pending_embeds = torch.ones((2, 3))
    requests = [
        SimpleNamespace(
            request_id="active",
            data=SimpleNamespace(output_rows=[], pending_feedback_queue=[]),
        ),
        SimpleNamespace(
            request_id="eos",
            data=SimpleNamespace(output_rows=[], pending_feedback_queue=[]),
        ),
    ]

    runner.post_process_outputs(
        object(),
        SimpleNamespace(requests=requests),
        {
            "active": RequestOutput("active", data=12),
            "eos": RequestOutput("eos", data=14),
        },
    )

    assert [row.tolist() for row in requests[0].data.output_rows] == [[12, 2, 4]]
    assert len(requests[0].data.pending_feedback_queue) == 1
    assert requests[1].data.output_rows == []
    assert requests[1].data.pending_feedback_queue == []


def test_moss_post_process_audio_end_restores_full_text_sampling() -> None:
    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner

    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.model = SimpleNamespace(
        config=SimpleNamespace(
            im_end_token_id=14,
            audio_start_token_id=10,
            audio_end_token_id=11,
        )
    )
    runner._pending_rows = torch.tensor([[11, 4, 4]], dtype=torch.long)
    runner._pending_embeds = torch.ones((1, 3))
    data = SimpleNamespace(
        output_rows=[],
        pending_feedback_queue=[],
        is_audio=True,
    )
    request = SimpleNamespace(request_id="audio-end", data=data)

    runner.post_process_outputs(
        object(),
        SimpleNamespace(requests=[request]),
        {"audio-end": RequestOutput("audio-end", data=11)},
    )

    assert data.is_audio is False
    assert [row.tolist() for row in data.output_rows] == [[11, 4, 4]]


def test_moss_audio_end_in_batch_uses_full_text_path_on_next_step() -> None:
    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner

    cfg = SimpleNamespace(
        im_end_token_id=14,
        audio_start_token_id=10,
        audio_end_token_id=11,
    )
    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.model = SimpleNamespace(
        config=cfg,
        device=torch.device("cpu"),
        _prepare_multi_modal_inputs=lambda rows: rows.to(torch.float32),
    )
    requests = [
        SimpleNamespace(
            request_id="audio-end",
            data=SimpleNamespace(
                is_audio=True,
                output_rows=[],
                pending_feedback_queue=[],
            ),
        ),
        SimpleNamespace(
            request_id="audio-active",
            data=SimpleNamespace(
                is_audio=True,
                output_rows=[],
                pending_feedback_queue=[],
            ),
        ),
    ]
    runner._pending_rows = torch.tensor([[11, 4], [12, 2]], dtype=torch.long)
    runner._pending_embeds = torch.ones((2, 2))

    runner.post_process_outputs(
        object(),
        SimpleNamespace(requests=requests),
        {
            "audio-end": RequestOutput("audio-end", data=11),
            "audio-active": RequestOutput("audio-active", data=12),
        },
    )

    assert requests[0].data.is_audio is False
    assert requests[1].data.is_audio is True

    seen: dict[str, bool] = {}

    def channel_logits(result, forward_batch, *, is_audio=False):
        del result, forward_batch
        seen["head"] = is_audio
        return [torch.zeros(2, 20), torch.zeros(2, 5)]

    def sample_rows(channel_logits, datas, *, n_vq, is_audio=False):
        del channel_logits, datas, n_vq
        seen["sampler"] = is_audio
        return torch.tensor([[5, 4], [12, 2]], dtype=torch.long)

    runner._channel_logits_from_result = channel_logits
    runner._sample_rows = sample_rows
    result = SimpleNamespace(next_token_ids=None)

    runner._collect_moss_step(
        result,
        object(),
        SimpleNamespace(output_ids=None),
        requests,
    )

    assert seen == {"head": False, "sampler": False}


def test_moss_sample_tokens_uses_per_row_top_k() -> None:
    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner

    # Identical logits for both rows; row 0 uses top_k=1 (locked to its argmax),
    # row 1 uses top_k=4. If the sampler reused row 0's params for the whole
    # batch (the datas[0] bug), row 1 would also be locked to the argmax.
    logits = torch.tensor(
        [
            [3.0, 2.0, 1.0, 0.0, float("-inf")],
            [3.0, 2.0, 1.0, 0.0, float("-inf")],
        ]
    )
    temperature = torch.tensor([2.0, 2.0])
    top_p = torch.tensor([1.0, 1.0])
    top_k = torch.tensor([1, 4])

    row0_vals: set[int] = set()
    row1_vals: set[int] = set()
    for seed in range(64):
        out = MossTTSModelRunner._sample_tokens(
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seeds=torch.tensor([seed, seed], dtype=torch.long),
            positions=torch.zeros(2, dtype=torch.long),
        )
        row0_vals.add(int(out[0]))
        row1_vals.add(int(out[1]))

    assert row0_vals == {0}  # top_k=1 -> always the argmax
    assert row1_vals - {0}  # top_k=4 -> reaches beyond the argmax
    assert row1_vals <= {0, 1, 2, 3}  # never the -inf-masked token


def test_moss_compact_candidate_sampler_maps_greedy_indices() -> None:
    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner

    token_ids = torch.tensor([12, 13], dtype=torch.long)
    sampled = MossTTSModelRunner._sample_tokens(
        torch.tensor([[1.0, 3.0], [4.0, 2.0]]),
        temperature=torch.zeros(2),
        top_p=torch.ones(2),
        top_k=torch.full((2,), 50, dtype=torch.long),
        seeds=torch.tensor([100, 101]),
        positions=torch.tensor([33, 66]),
        candidate_token_ids=token_ids,
    )

    assert sampled.tolist() == [13, 12]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA seeded sampler"
)
def test_moss_text_control_sampler_preserves_full_vocab_seed_columns() -> None:
    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner

    device = torch.device("cuda")
    rows = 16
    vocab = 32
    token_ids = torch.tensor([12, 13], dtype=torch.long, device=device)
    torch.manual_seed(0)
    selected_logits = torch.randn(rows, 2, device=device)
    full_logits = torch.full(
        (rows, vocab), float("-inf"), dtype=torch.float32, device=device
    )
    full_logits[:, token_ids] = selected_logits
    temperature = torch.full((rows,), 1.5, device=device)
    top_p = torch.tensor([1.0, 0.8] * (rows // 2), device=device)
    top_k = torch.tensor([50, 1] * (rows // 2), dtype=torch.long, device=device)
    seeds = torch.arange(rows, dtype=torch.long, device=device) + 100
    positions = torch.arange(rows, dtype=torch.long, device=device) * 33

    full = MossTTSModelRunner._sample_tokens(
        full_logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seeds=seeds,
        positions=positions,
    )
    restricted = MossTTSModelRunner._sample_tokens(
        selected_logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seeds=seeds,
        positions=positions,
        candidate_token_ids=token_ids,
    )

    assert torch.equal(restricted, full)


def test_moss_tts_rejects_invalid_token_count() -> None:
    for bad_text in ("${token:0}hi", "${token:-1}hi", "${token:abc}hi"):
        with pytest.raises(ValueError):
            build_moss_tts_state(make_payload(inputs={"text": bad_text}))
    with pytest.raises(ValueError):
        build_moss_tts_state(
            make_payload(inputs={"text": "hi"}, tts_params={"token_count": 0})
        )


def test_moss_tts_rejects_nonpositive_max_new_tokens() -> None:
    # An explicit max_new_tokens must be validated, not silently replaced by the
    # default: 0 and negatives are rejected (0 previously fell through ``or``).
    for bad in (0, -1, -100):
        with pytest.raises(ValueError):
            build_moss_tts_state(
                make_payload(inputs={"text": "hi"}, params={"max_new_tokens": bad})
            )


def test_moss_tts_explicit_max_new_tokens_is_preserved() -> None:
    state = build_moss_tts_state(
        make_payload(inputs={"text": "hi"}, params={"max_new_tokens": 128})
    )
    assert state.generation_kwargs["max_new_tokens"] == 128


def test_moss_preprocess_discards_handoff_after_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.moss_tts import request_builders as rb

    payload = make_payload(inputs="hello", request_id="abort-me")

    def fake_prepare(pl, *, processor, reference_encoder=None):
        del processor, reference_encoder
        # The abort fires while preprocessing is still running.
        rb.cleanup_prepared_moss_tts_request(pl.request_id)
        return rb.MossTTSPreparedRequest(
            state=MossTTSState(),
            input_ids_list=[],
            input_ids=torch.zeros(0, dtype=torch.long),
            prompt_rows=torch.zeros((0, 0), dtype=torch.long),
            gen_kwargs={},
        )

    monkeypatch.setattr(rb, "_prepare_moss_tts_request", fake_prepare)
    try:
        rb.set_moss_tts_preprocessing_context(processor=object())
        result = rb.preprocess_moss_tts_payload(payload)
        # note (Yue Yin): dropped handoff must not carry a marker the AR stage would
        # pop as missing state.
        assert rb._MOSS_TTS_PREPARED_MARKER not in result.data
        snap = rb._QUEUE.snapshot()
        assert "abort-me" not in snap.prepared
        assert not snap.prepared
    finally:
        rb.clear_moss_tts_preprocessing_context()


def test_moss_sample_tokens_seeded_is_reproducible() -> None:
    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner

    torch.manual_seed(0)
    logits = torch.randn(2, 64)
    temperature = torch.tensor([1.5, 1.7])
    top_p = torch.tensor([0.9, 0.8])
    top_k = torch.tensor([50, 25])

    def sample(seeds: list[int], positions: list[int]) -> torch.Tensor:
        return MossTTSModelRunner._sample_tokens(
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seeds=torch.tensor(seeds, dtype=torch.long),
            positions=torch.tensor(positions, dtype=torch.long),
        )

    # Reproducible: same per-row seeds + positions give identical output.
    assert torch.equal(sample([11, 22], [5, 5]), sample([11, 22], [5, 5]))

    # Neighbour-independence: a row's token depends only on its own seed and
    # position, never on its batch neighbours. Changing the *other* row's seed
    # must leave this row's sampled token unchanged.
    baseline = sample([11, 22], [5, 5])
    assert int(sample([11, 999], [5, 5])[0]) == int(baseline[0])
    assert int(sample([777, 22], [5, 5])[1]) == int(baseline[1])

    # Position is part of the sampling key: varying it changes some draws.
    outs = {tuple(sample([11, 22], [p, p]).tolist()) for p in range(8)}
    assert len(outs) > 1


def test_moss_tts_rejects_invalid_sampling_params() -> None:
    for bad in (
        {"audio_temperature": -1.0},
        {"audio_top_p": 1.5},
        {"text_top_p": 0.0},
        {"audio_repetition_penalty": 0.0},
        {"seed": -5},
    ):
        with pytest.raises(ValueError):
            build_moss_tts_state(make_payload(inputs={"text": "hi"}, tts_params=bad))


def test_moss_preprocess_pre_start_abort_does_not_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.moss_tts import request_builders as rb

    def fake_prepare(pl, *, processor, reference_encoder=None):
        del processor, reference_encoder
        return rb.MossTTSPreparedRequest(
            state=MossTTSState(),
            input_ids_list=[],
            input_ids=torch.zeros(0, dtype=torch.long),
            prompt_rows=torch.zeros((0, 0), dtype=torch.long),
            gen_kwargs={},
        )

    monkeypatch.setattr(rb, "_prepare_moss_tts_request", fake_prepare)
    try:
        rb.set_moss_tts_preprocessing_context(processor=object())
        # Abort for a request that never started preprocessing: no tombstone.
        rb.cleanup_prepared_moss_tts_request("ghost")
        assert not rb._QUEUE.snapshot().aborted
        # The same id can still run a normal preprocess and publish its handoff.
        rb.preprocess_moss_tts_payload(make_payload(inputs="hello", request_id="ghost"))
        snap = rb._QUEUE.snapshot()
        assert "ghost" in snap.prepared
        assert not snap.aborted
        assert not snap.inflight
    finally:
        rb.clear_moss_tts_preprocessing_context()
