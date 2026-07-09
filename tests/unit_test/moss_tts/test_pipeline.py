# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import sys
import types
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
from sglang_omni.config import build_stage_placement_plan
from sglang_omni.models.moss_tts.codec import split_moss_audio_segments
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


def install_fake_sglang(monkeypatch: pytest.MonkeyPatch) -> None:
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
            **kwargs,
        ) -> None:
            del kwargs
            self.rid = rid
            self.origin_input_text = origin_input_text
            self.origin_input_ids = origin_input_ids
            self.sampling_params = sampling_params
            self.eos_token_ids = eos_token_ids
            self.vocab_size = vocab_size
            self.output_ids = []
            self.prefix_indices = []
            self.extend_input_len = len(origin_input_ids)

    class FakeSamplingParams:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

        def normalize(self, tokenizer) -> None:
            del tokenizer

        def verify(self, vocab_size) -> None:
            self.vocab_size = vocab_size

    modules = {
        "sglang": types.ModuleType("sglang"),
        "sglang.srt": types.ModuleType("sglang.srt"),
        "sglang.srt.managers": types.ModuleType("sglang.srt.managers"),
        "sglang.srt.managers.schedule_batch": types.ModuleType(
            "sglang.srt.managers.schedule_batch"
        ),
        "sglang.srt.sampling": types.ModuleType("sglang.srt.sampling"),
        "sglang.srt.sampling.sampling_params": types.ModuleType(
            "sglang.srt.sampling.sampling_params"
        ),
    }
    for name in ("sglang", "sglang.srt", "sglang.srt.managers", "sglang.srt.sampling"):
        modules[name].__path__ = []
    modules["sglang"].srt = modules["sglang.srt"]
    modules["sglang.srt"].managers = modules["sglang.srt.managers"]
    modules["sglang.srt"].sampling = modules["sglang.srt.sampling"]
    modules["sglang.srt.managers"].schedule_batch = modules[
        "sglang.srt.managers.schedule_batch"
    ]
    modules["sglang.srt.sampling"].sampling_params = modules[
        "sglang.srt.sampling.sampling_params"
    ]
    modules["sglang.srt.managers.schedule_batch"].Req = FakeReq
    modules["sglang.srt.sampling.sampling_params"].SamplingParams = FakeSamplingParams
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


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
    assert config.gpu_placement == {"preprocessing": 0, "tts_engine": 0, "vocoder": 0}
    assert {stage.process for stage in config.stages} == {"pipeline"}
    stages = {stage.name: stage for stage in config.stages}
    assert stages["preprocessing"].factory_args == {
        "codec_model_path": "OpenMOSS-Team/MOSS-Audio-Tokenizer",
        "device": "cuda:0",
        "dtype": "float32",
    }
    assert stages["vocoder"].factory_args["codec_model_path"] == (
        "OpenMOSS-Team/MOSS-Audio-Tokenizer"
    )
    assert "gpu_id" not in stages["vocoder"].factory_args
    placement = build_stage_placement_plan(config)
    assert placement.stages["preprocessing"].gpu_ids == (0,)
    assert config.supports_uploaded_voice_references() is True
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


def test_moss_preprocessing_executor_loads_owned_codec_with_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.moss_tts import request_builders as rb
    from sglang_omni.models.moss_tts import stages

    captured: list[tuple[str, str, str]] = []

    class FakeCodec:
        sample_rate = 24000

    def fake_load_processor(model_path):
        assert model_path == "ckpt"
        return object()

    def fake_load_codec(codec_model_path, *, device, dtype):
        captured.append((codec_model_path, device, dtype))
        return FakeCodec()

    monkeypatch.setattr(stages, "_load_moss_processor", fake_load_processor)
    monkeypatch.setattr(stages, "load_moss_audio_tokenizer", fake_load_codec)
    try:
        monkeypatch.setattr(stages.torch.cuda, "is_available", lambda: True)
        stages.create_preprocessing_executor("ckpt")
        assert captured[-1] == (
            "OpenMOSS-Team/MOSS-Audio-Tokenizer",
            "cuda:0",
            "float32",
        )
        assert rb._PREPROCESSING_CONTEXT.audio_tokenizer.sample_rate == 24000

        monkeypatch.setattr(stages.torch.cuda, "is_available", lambda: False)
        stages.create_preprocessing_executor(
            "ckpt",
            codec_model_path="codec-local",
            device="cuda:1",
            dtype="bfloat16",
        )
        assert captured[-1] == ("codec-local", "cpu", "bfloat16")
    finally:
        clear_moss_tts_preprocessing_context()


def test_moss_audio_tokenizer_resolves_only_config_and_weight_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.moss_tts import audio_tokenizer

    captured: dict[str, object] = {}
    fake_hf = types.ModuleType("huggingface_hub")

    def fake_snapshot_download(repo_id, **kwargs):
        captured["repo_id"] = repo_id
        captured.update(kwargs)
        return "/tmp/moss-codec"

    fake_hf.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    assert audio_tokenizer._resolve_checkpoint(
        "OpenMOSS-Team/MOSS-Audio-Tokenizer"
    ) == ("/tmp/moss-codec")
    assert captured["repo_id"] == "OpenMOSS-Team/MOSS-Audio-Tokenizer"
    assert captured["allow_patterns"] == (
        "config.json",
        "*.safetensors",
        "*.safetensors.index.json",
    )


def test_moss_audio_tokenizer_loads_local_modeling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.moss_tts import audio_tokenizer
    from sglang_omni.models.moss_tts.configuration_moss_audio_tokenizer import (
        MossAudioTokenizerConfig,
    )
    from sglang_omni.models.moss_tts.modeling_moss_audio_tokenizer import (
        MossAudioTokenizerModel,
    )

    captured: dict[str, object] = {}

    class FakeConfig:
        sampling_rate = 24000
        quantizer_kwargs = {"num_quantizers": 32}

    class FakeModel:
        config = FakeConfig()

        def eval(self):
            captured["eval"] = True

        def to(self, **kwargs):
            captured["to"] = kwargs

    def fake_config_from_pretrained(cls, checkpoint_dir):
        del cls
        captured["config_checkpoint_dir"] = checkpoint_dir
        return FakeConfig()

    def fake_model_from_pretrained(cls, checkpoint_dir, **kwargs):
        del cls
        captured["model_checkpoint_dir"] = checkpoint_dir
        captured["model_kwargs"] = kwargs
        return FakeModel()

    monkeypatch.setattr(audio_tokenizer, "_resolve_checkpoint", lambda _: "/tmp/codec")
    monkeypatch.setattr(
        MossAudioTokenizerConfig,
        "from_pretrained",
        classmethod(fake_config_from_pretrained),
    )
    monkeypatch.setattr(
        MossAudioTokenizerModel,
        "from_pretrained",
        classmethod(fake_model_from_pretrained),
    )

    tokenizer = audio_tokenizer.load_moss_audio_tokenizer(
        "OpenMOSS-Team/MOSS-Audio-Tokenizer",
        device="cpu",
        dtype="float32",
    )

    assert tokenizer.checkpoint_dir == "/tmp/codec"
    assert captured["config_checkpoint_dir"] == "/tmp/codec"
    assert captured["model_checkpoint_dir"] == "/tmp/codec"
    assert captured["model_kwargs"]["local_files_only"] is True
    assert "trust_remote_code" not in captured["model_kwargs"]
    assert captured["eval"] is True
    assert captured["to"] == {"device": "cpu"}


def test_moss_tts_engine_uses_auto_mem_fraction_by_default(monkeypatch) -> None:
    from sglang_omni.models.moss_tts import request_builders, stages
    from sglang_omni.scheduling import bootstrap, omni_scheduler, sglang_backend

    captured: dict[str, object] = {"build_kwargs": []}

    def fake_build_sglang_server_args(model_path, context_length, **kwargs):
        captured["model_path"] = model_path
        captured["context_length"] = context_length
        captured["build_kwargs"].append(dict(kwargs))
        return SimpleNamespace(
            disable_cuda_graph=kwargs["disable_cuda_graph"],
            disable_overlap_schedule=False,
            max_running_requests=kwargs["max_running_requests"],
            cuda_graph_max_bs=kwargs["cuda_graph_max_bs"],
            cuda_graph_bs=kwargs["cuda_graph_bs"],
            enable_torch_compile=kwargs["enable_torch_compile"],
            torch_compile_max_bs=kwargs.get("torch_compile_max_bs"),
        )

    def fake_create_sglang_infrastructure(
        server_args,
        gpu_id,
        *,
        model_arch_override=None,
    ):
        captured["gpu_id"] = gpu_id
        captured["model_arch_override"] = model_arch_override
        model = object()
        model_runner = SimpleNamespace(
            model=model,
            init_device_graphs=lambda: captured.setdefault("graph_inits", 0) or None,
        )
        model_worker = SimpleNamespace(model_runner=model_runner)
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
        stages, "resolve_moss_checkpoint", lambda model_path: model_path
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


def test_moss_tts_state_round_trip_keeps_tensors_native() -> None:
    codes = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    state = MossTTSState(
        text="hello",
        ref_audio="ref.wav",
        ref_text="reference",
        language="en",
        instructions="warm",
        token_count=180,
        generation_kwargs={"max_new_tokens": 64},
        delayed_audio_codes=codes,
        assistant_start_length=2,
    )
    restored = MossTTSState.from_dict(state.to_dict())

    assert restored.text == "hello"
    assert restored.ref_audio == "ref.wav"
    assert restored.ref_text == "reference"
    assert restored.language == "en"
    assert restored.instructions == "warm"
    assert restored.token_count == 180
    assert torch.equal(restored.delayed_audio_codes, codes)
    assert restored.assistant_start_length == 2


def test_moss_tts_vocoder_uses_batch_base_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from sglang_omni.models.moss_tts import stages

    decoded_segments: list[list[list[int]]] = []

    class FakeCodec:
        sample_rate = 16000

        def decode_codes(self, segments):
            decoded_segments.extend(segment.tolist() for segment in segments)
            offset = float(len(decoded_segments) * 10)
            return [torch.tensor([offset, offset + 1], dtype=torch.float32)]

    monkeypatch.setattr(
        stages,
        "_load_moss_model_config",
        lambda model_path: SimpleNamespace(audio_pad_code=1024, sampling_rate=16000),
    )
    monkeypatch.setattr(
        stages,
        "load_moss_audio_tokenizer",
        lambda codec_model_path, *, device, dtype: FakeCodec(),
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


def test_moss_tts_maps_references_token_count_and_deterministic_defaults() -> None:
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
    assert state.generation_kwargs["audio_temperature"] == 1.7
    assert state.generation_kwargs["audio_top_p"] == 0.8
    assert state.generation_kwargs["audio_top_k"] == 25


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

    class FakeCodec:
        def __init__(self) -> None:
            self.paths = []

        def encode_paths(self, paths):
            self.paths.extend(paths)
            return [torch.tensor([[7, 8], [9, 10]], dtype=torch.long)]

    codec = FakeCodec()
    processor = FakeProcessor()
    payload = make_payload(
        inputs={"text": "hello", "references": [{"audio_path": "ref.wav"}]},
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
        set_moss_tts_preprocessing_context(processor=processor, audio_tokenizer=codec)
        prepared_payload = preprocess_moss_tts_payload(payload)
        data = build_sglang_moss_tts_request(prepared_payload, model=model)
    finally:
        clear_moss_tts_preprocessing_context()

    assert codec.paths == ["ref.wav"]
    assert torch.equal(
        processor.message_kwargs["reference"][0],
        torch.tensor([[7, 8], [9, 10]], dtype=torch.long),
    )
    assert processor.message_kwargs["tokens"] == 80
    assert processor.message_kwargs["language"] == "en"
    assert data.req._input_embeds_are_projected is True
    assert data.input_embeds_are_projected is True
    assert data.max_new_tokens == 12
    assert data.prompt_rows.shape == (3, 3)
    assert data.state.assistant_start_length == 0
    assert data.req.sampling_params.stop_token_ids == [151645]


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
            req=SimpleNamespace(extend_input_len=2, prefix_indices=[0]),
            prompt_rows=prompt_rows,
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


def test_moss_post_process_outputs_skips_im_end() -> None:
    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner

    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.model = SimpleNamespace(config=SimpleNamespace(im_end_token_id=14))
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


def test_moss_delay_codec_splits_non_pad_segments() -> None:
    delayed = torch.tensor(
        [
            [1, 1024],
            [2, 3],
            [1024, 4],
            [1024, 1024],
        ],
        dtype=torch.long,
    )

    segments = split_moss_audio_segments(delayed, audio_pad_code=1024)

    assert [segment.tolist() for segment in segments] == [[[1, 3], [2, 4]]]


def test_moss_vocoder_decodes_with_owned_codec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.moss_tts import stages

    class FakeCodec:
        sample_rate = 16000

        def __init__(self) -> None:
            self.decoded_segments = None

        def decode_codes(self, segments):
            self.decoded_segments = [segment.clone() for segment in segments]
            return [torch.tensor([0.25, -0.25, 0.5], dtype=torch.float32)]

    codec = FakeCodec()

    def fail_processor_load(*args, **kwargs):
        raise AssertionError("vocoder must decode through the owned audio tokenizer")

    monkeypatch.setattr(stages, "_load_moss_processor", fail_processor_load)
    monkeypatch.setattr(
        stages,
        "_load_moss_model_config",
        lambda model_path: SimpleNamespace(audio_pad_code=1024, sampling_rate=24000),
    )
    monkeypatch.setattr(
        stages,
        "load_moss_audio_tokenizer",
        lambda codec_model_path, *, device, dtype: codec,
    )

    scheduler = stages.create_vocoder_executor("ckpt", gpu_id=0)
    state = MossTTSState(
        delayed_audio_codes=torch.tensor(
            [[1, 1024], [2, 3], [1024, 4]],
            dtype=torch.long,
        ),
        assistant_start_length=0,
    )
    result = asyncio.run(
        scheduler._fn(
            StagePayload(
                request_id="decode",
                request=OmniRequest(inputs="hello"),
                data=state.to_dict(),
            )
        )
    )

    assert [segment.tolist() for segment in codec.decoded_segments] == [
        [[1, 3], [2, 4]]
    ]
    assert result.data["sample_rate"] == 16000
    assert result.data["modality"] == "audio"


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

    def fake_prepare(pl, *, processor, audio_tokenizer=None):
        del audio_tokenizer
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
        rb.preprocess_moss_tts_payload(payload)
        with rb._PREPARED_REQUESTS_LOCK:
            assert "abort-me" not in rb._PREPARED_REQUESTS
            assert not rb._PREPARED_REQUESTS
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

    def fake_prepare(pl, *, processor, audio_tokenizer=None):
        del audio_tokenizer
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
        with rb._PREPARED_REQUESTS_LOCK:
            assert not rb._ABORTED_REQUESTS
        # The same id can still run a normal preprocess and publish its handoff.
        rb.preprocess_moss_tts_payload(make_payload(inputs="hello", request_id="ghost"))
        with rb._PREPARED_REQUESTS_LOCK:
            assert "ghost" in rb._PREPARED_REQUESTS
            assert not rb._ABORTED_REQUESTS
            assert not rb._INFLIGHT_REQUESTS
    finally:
        rb.clear_moss_tts_preprocessing_context()
