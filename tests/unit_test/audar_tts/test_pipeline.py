# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import io
import sys
import threading
import time
import types
import wave
from typing import Any

import numpy as np
import pytest
import torch

from sglang_omni.client.client import Client
from sglang_omni.comm import stage_io
from sglang_omni.comm.data_ref import DataRef, TransportKind
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_factory_signature_args
from sglang_omni.config.schema import EndpointsConfig
from sglang_omni.models.audar_tts import stages
from sglang_omni.models.audar_tts.config import AudarTTSPipelineConfig
from sglang_omni.models.audar_tts.payload_types import AudarTTSState
from sglang_omni.models.audar_tts.protocol import build_prompt, parse_speech_codes
from sglang_omni.models.audar_tts.request_builders import build_audar_state
from sglang_omni.models.model_capabilities import get_model_capabilities
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.pipeline.control_plane import deserialize_message, serialize_message
from sglang_omni.pipeline.mp_runner import _build_stage_groups
from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime
from sglang_omni.proto import DataReadyMessage, OmniRequest, StagePayload
from sglang_omni.relay.shm import ShmRelay
from sglang_omni.serve.speech_errors import SpeechAPIError
from sglang_omni.serve.speech_service import SpeechRequestValidator
from sglang_omni.utils.imports import import_string
from tests.unit_test.fixtures.pipeline_fakes import FakeMpContext


class FakeCodec:
    def __init__(self) -> None:
        self.encode_calls = 0
        self.decode_calls = 0
        self.device = "cpu"

    def eval(self) -> "FakeCodec":
        return self

    def to(self, device: str) -> "FakeCodec":
        self.device = device
        return self

    def encode_code(self, waveform: torch.Tensor) -> torch.Tensor:
        self.encode_calls += 1
        assert waveform.shape == (1, 1, 80000)
        return torch.tensor([[[7, 8, 9]]])

    def decode_code(self, codes: torch.Tensor) -> torch.Tensor:
        self.decode_calls += 1
        assert codes.ndim == 3
        return torch.tensor([[[0.25, -0.5, 0.75]]])


def make_payload(
    *,
    inputs: Any = "",
    params: dict[str, Any] | None = None,
    tts_params: dict[str, Any] | None = None,
    state: AudarTTSState | None = None,
    request_id: str = "request",
) -> StagePayload:
    metadata = {"tts_params": tts_params} if tts_params is not None else {}
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=inputs, params=params or {}, metadata=metadata),
        data=state.to_dict() if state is not None else {},
    )


def five_second_wav(sample_value: int = 0) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.full(80000, sample_value, dtype="<i2").tobytes())
    return output.getvalue()


def test_prompt_matches_official_audar_protocol() -> None:
    prompt = build_prompt("مرحبا", "صوت مرجعي", [7, 8])

    assert prompt == (
        "user: Convert the text to speech:"
        "<|REF_TEXT_START|>صوت مرجعي<|REF_TEXT_END|>"
        "<|REF_SPEECH_START|><|speech_7|><|speech_8|><|REF_SPEECH_END|>"
        "<|TARGET_TEXT_START|>مرحبا<|TARGET_TEXT_END|>"
        "\nassistant:<|TARGET_CODES_START|>"
    )
    assert parse_speech_codes("x<|speech_5|><|speech_42|>y") == [5, 42]


def test_config_and_state_contracts() -> None:
    config = AudarTTSPipelineConfig(model_path="audarai/Audar-TTS-V1-Turbo")
    file_config = ConfigManager.from_file(
        "examples/configs/audar_tts_turbo.yaml"
    ).config
    assert isinstance(file_config, AudarTTSPipelineConfig)
    assert file_config.model_path == config.model_path
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "reference_encoder",
        "tts_engine",
        "vocoder",
    ]
    assert config.terminal_stages == ["vocoder"]
    assert config.supports_uploaded_voice_references() is True
    assert config.required_speech_reference_count == 1
    assert config.speech_reference_text_required is True
    assert config.additional_speech_languages == frozenset({"Arabic"})
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("AudarTTSForConditionalGeneration")
        is AudarTTSPipelineConfig
    )
    capabilities = get_model_capabilities("AudarTTSForConditionalGeneration")
    assert capabilities is not None
    assert capabilities.supports_reference_audio is True
    assert capabilities.supports_batch_vocoder is False
    assert capabilities.supports_streaming_vocoder is False
    assert AudarTTSState().to_dict() == {
        "sample_rate": 24000,
        "generation_kwargs": {},
    }

    state = AudarTTSState(
        target_text="target",
        reference_text="reference",
        reference_audio={"bytes": b"wav"},
        prompt="prompt",
        audio_codes=[1, 2],
        generation_kwargs={"temperature": 0.7},
        prompt_tokens=10,
        completion_tokens=20,
        engine_time_s=0.5,
    )
    assert AudarTTSState.from_dict(state.to_dict()) == state


@pytest.mark.parametrize(
    ("payload", "param"),
    [
        ({"input": "target"}, "ref_audio"),
        (
            {
                "input": "target",
                "ref_audio": "data:audio/wav;base64,UklGRg==",
            },
            "ref_text",
        ),
        (
            {
                "input": "target",
                "ref_audio": "data:audio/wav;base64,UklGRg==",
                "ref_text": "reference",
                "references": [
                    {
                        "data": "UklGRg==",
                        "media_type": "audio/wav",
                        "text": "reference",
                    }
                ],
            },
            "references",
        ),
    ],
)
def test_public_speech_validation_rejects_invalid_audar_references(
    payload: dict[str, Any], param: str
) -> None:
    config = AudarTTSPipelineConfig(model_path="audarai/Audar-TTS-V1-Turbo")
    validator = SpeechRequestValidator(
        default_model=config.model_path,
        required_speech_reference_count=config.required_speech_reference_count,
        speech_reference_text_required=config.speech_reference_text_required,
        additional_speech_languages=config.additional_speech_languages,
    )

    with pytest.raises(SpeechAPIError) as exc_info:
        validator.parse_generation_request(payload)

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == param


def test_arabic_language_is_scoped_to_audar() -> None:
    with pytest.raises(SpeechAPIError) as exc_info:
        SpeechRequestValidator(default_model="qwen3-tts").parse_request(
            {"input": "target", "language": "Arabic"}
        )
    assert exc_info.value.param == "language"

    validator = SpeechRequestValidator(
        default_model="audarai/Audar-TTS-V1-Turbo",
        additional_speech_languages=frozenset({"Arabic"}),
    )
    request = validator.parse_request({"input": "target", "language": "arabic"})
    assert request.language == "Arabic"


def test_config_dispatch_injects_model_path_and_gpu(tmp_path: Any) -> None:
    config = AudarTTSPipelineConfig(
        model_path="audarai/Audar-TTS-V1-Turbo",
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
    )
    prepared = prepare_pipeline_runtime(config)
    try:
        groups = _build_stage_groups(
            config,
            ctx=FakeMpContext(),
            stages_cfg=prepared.stages_cfg,
            name_map=prepared.name_map,
            endpoints=prepared.endpoints,
            placement_plan=prepared.placement_plan,
            process_plan=prepared.process_plan,
        )
    finally:
        assert prepared.runtime_dir is not None
        prepared.runtime_dir.close()

    resolved = {}
    for spec in (spec for group in groups for spec in group.specs):
        resolved[spec.stage_name] = resolve_factory_signature_args(
            import_string(spec.factory),
            spec.factory_args,
            defaults=spec.factory_arg_defaults,
        )

    assert resolved == {
        "preprocessing": {},
        "reference_encoder": {"gpu_id": 0},
        "tts_engine": {
            "model_path": "audarai/Audar-TTS-V1-Turbo",
            "gpu_id": 0,
        },
        "vocoder": {"gpu_id": 0},
    }


def test_request_lowering_keeps_audar_defaults_unless_explicit() -> None:
    reference = {"bytes": five_second_wav(), "text": "reference transcript"}
    implicit = make_payload(
        inputs={"text": "target", "references": [reference]},
        params={
            "temperature": 0.8,
            "top_p": 0.8,
            "top_k": 30,
            "repetition_penalty": 1.1,
        },
    )
    implicit_state = build_audar_state(implicit)
    assert implicit_state.generation_kwargs == {
        "max_new_tokens": 2048,
        "temperature": 1.0,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
    }

    explicit = make_payload(
        inputs={"text": "target", "references": [reference]},
        params={"temperature": 0.6, "top_k": 20, "max_new_tokens": 128},
        tts_params={
            "explicit_generation_params": [
                "temperature",
                "top_k",
                "max_new_tokens",
            ],
            "seed": 17,
        },
    )
    explicit_state = build_audar_state(explicit)
    assert explicit_state.generation_kwargs == {
        "max_new_tokens": 128,
        "temperature": 0.6,
        "top_k": 20,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "seed": 17,
    }


def test_openai_speech_request_lowers_to_audar_state() -> None:
    wav_bytes = five_second_wav()
    prepared = SpeechRequestValidator(
        default_model="audarai/Audar-TTS-V1-Turbo"
    ).parse_generation_request(
        {
            "input": "target text",
            "response_format": "pcm",
            "ref_audio": (
                "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode("ascii")
            ),
            "ref_text": "reference transcript",
            "max_new_tokens": 128,
            "temperature": 0.8,
            "top_k": 30,
            "seed": 17,
        }
    )
    validator = SpeechRequestValidator(default_model="audarai/Audar-TTS-V1-Turbo")
    generation_request = validator.build_generate_request(
        prepared.request,
        validate=False,
        reference_descriptors=prepared.reference_descriptors,
    )
    assert generation_request.metadata["tts_params"]["explicit_generation_params"] == [
        "max_new_tokens",
        "seed",
        "temperature",
        "top_k",
    ]
    payload = StagePayload(
        request_id="request",
        request=Client._build_omni_request(generation_request),
        data={},
    )

    state = build_audar_state(payload)

    assert state.target_text == "target text"
    assert state.reference_text == "reference transcript"
    assert state.reference_audio == {
        "data": base64.b64encode(wav_bytes).decode("ascii"),
        "media_type": "audio/wav",
    }
    assert state.generation_kwargs == {
        "max_new_tokens": 128,
        "temperature": 0.8,
        "top_k": 30,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "seed": 17,
    }


@pytest.mark.parametrize(
    "reference_audio",
    [
        {"bytes": five_second_wav()},
        {
            "data": base64.b64encode(five_second_wav()).decode("ascii"),
            "media_type": "audio/wav",
        },
    ],
)
def test_reference_audio_survives_control_plane_and_relay(
    reference_audio: dict[str, Any],
) -> None:
    async def round_trip() -> AudarTTSState:
        relay = ShmRelay(engine_id="audar-reference-round-trip", device="cpu")
        payload = make_payload(
            state=AudarTTSState(
                target_text="target",
                reference_text="reference",
                reference_audio=reference_audio,
            )
        )
        payload.data["relay_probe"] = torch.tensor([1], dtype=torch.int32)
        try:
            data_ref, operation = await stage_io.write_payload(
                relay,
                payload.request_id,
                payload,
                transport=TransportKind.SHM,
                from_stage="preprocessing",
                to_stage="reference_encoder",
            )
            message = DataReadyMessage(
                request_id=payload.request_id,
                from_stage="preprocessing",
                to_stage="reference_encoder",
                data_ref=data_ref.to_dict(),
            )
            restored_message = deserialize_message(serialize_message(message))
            restored_payload = await stage_io.read_payload(
                relay,
                payload.request_id,
                DataRef.from_dict(restored_message.data_ref),
            )
            operation.mark_receiver_done()
            await operation.wait_for_completion()
            assert torch.equal(restored_payload.data["relay_probe"], torch.tensor([1]))
            return AudarTTSState.from_dict(restored_payload.data)
        finally:
            relay.close()

    restored = asyncio.run(round_trip())

    assert restored.reference_audio == reference_audio


def test_request_lowering_requires_one_transcribed_reference() -> None:
    with pytest.raises(ValueError, match="reference audio"):
        build_audar_state(make_payload(inputs="target"))
    with pytest.raises(ValueError, match="reference transcript"):
        build_audar_state(
            make_payload(
                inputs={
                    "text": "target",
                    "references": [{"bytes": five_second_wav()}],
                }
            )
        )
    with pytest.raises(ValueError, match="exactly one"):
        build_audar_state(
            make_payload(
                inputs={
                    "text": "target",
                    "references": [
                        {"bytes": b"a", "text": "a"},
                        {"bytes": b"b", "text": "b"},
                    ],
                }
            )
        )


def test_reference_encoder_builds_prompt_and_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = FakeCodec()
    monkeypatch.setattr(stages, "_load_codec", lambda *args, **kwargs: codec)
    scheduler = stages.create_reference_encoder_executor(gpu_id=None)
    reference_audio = {"bytes": five_second_wav()}

    def encode(request_id: str) -> AudarTTSState:
        payload = make_payload(
            state=AudarTTSState(
                target_text="target",
                reference_text="reference",
                reference_audio=reference_audio,
            ),
            request_id=request_id,
        )
        return AudarTTSState.from_dict(scheduler._fn(payload).data)

    first = encode("first")
    second = encode("second")

    assert codec.encode_calls == 1
    assert first.prompt == build_prompt("target", "reference", [7, 8, 9])
    assert second.prompt == first.prompt
    assert first.reference_audio is None


def test_reference_encoder_singleflights_same_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = FakeCodec()
    encode_started = threading.Event()
    release_encode = threading.Event()
    second_normalized = threading.Event()
    normalize_lock = threading.Lock()
    normalize_calls = 0
    original_normalize = stages._normalize_reference

    def normalize(raw_input: Any):
        nonlocal normalize_calls
        item = original_normalize(raw_input)
        with normalize_lock:
            normalize_calls += 1
            if normalize_calls == 2:
                second_normalized.set()
        return item

    def encode_code(waveform: torch.Tensor) -> torch.Tensor:
        codec.encode_calls += 1
        encode_started.set()
        assert release_encode.wait(timeout=2)
        return torch.tensor([[[7, 8, 9]]])

    monkeypatch.setattr(stages, "_normalize_reference", normalize)
    monkeypatch.setattr(codec, "encode_code", encode_code)
    monkeypatch.setattr(stages, "_load_codec", lambda *args, **kwargs: codec)
    scheduler = stages.create_reference_encoder_executor(gpu_id=None, max_concurrency=2)
    reference_audio = {"bytes": five_second_wav()}

    def encode(request_id: str) -> AudarTTSState:
        payload = make_payload(
            state=AudarTTSState(
                target_text="target",
                reference_text="reference",
                reference_audio=reference_audio,
            ),
            request_id=request_id,
        )
        return AudarTTSState.from_dict(scheduler._fn(payload).data)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(encode, "first")
        assert encode_started.wait(timeout=2)
        second = executor.submit(encode, "second")
        assert second_normalized.wait(timeout=2)
        release_encode.set()
        results = [first.result(timeout=2), second.result(timeout=2)]

    assert scheduler._max_concurrency == 2
    assert codec.encode_calls == 1
    assert results[0].prompt == results[1].prompt


def test_reference_encoder_serializes_codec_for_different_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = FakeCodec()
    encode_lock = threading.Lock()
    active_calls = 0
    max_active_calls = 0

    def encode_code(waveform: torch.Tensor) -> torch.Tensor:
        nonlocal active_calls, max_active_calls
        with encode_lock:
            codec.encode_calls += 1
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
        time.sleep(0.02)
        with encode_lock:
            active_calls -= 1
        return torch.tensor([[[7, 8, 9]]])

    monkeypatch.setattr(codec, "encode_code", encode_code)
    monkeypatch.setattr(stages, "_load_codec", lambda *args, **kwargs: codec)
    scheduler = stages.create_reference_encoder_executor(gpu_id=None, max_concurrency=2)

    def encode(request_id: str, wav_bytes: bytes) -> AudarTTSState:
        payload = make_payload(
            state=AudarTTSState(
                target_text="target",
                reference_text="reference",
                reference_audio={"bytes": wav_bytes},
            ),
            request_id=request_id,
        )
        return AudarTTSState.from_dict(scheduler._fn(payload).data)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(encode, "first", five_second_wav(0)),
            executor.submit(encode, "second", five_second_wav(1)),
        ]
        results = [future.result(timeout=3) for future in futures]

    assert codec.encode_calls == 2
    assert max_active_calls == 1
    assert all(result.prompt for result in results)


def _reference_service(codec: FakeCodec) -> Any:
    hook = stages._AudarReferenceEncodeHook(
        codec=codec,
        device="cpu",
        codec_model="codec",
        codec_revision="revision",
    )
    return stages.ReferenceEncodeService(hook)


def test_reference_hook_preserves_key_and_tensor_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hook = stages._AudarReferenceEncodeHook(
        codec=FakeCodec(),
        device="cpu",
        codec_model="codec",
        codec_revision="revision",
    )
    item = hook.normalize_input({"bytes": five_second_wav()})
    key = hook.cache_key(item)

    assert key == stages.ReferenceEncodeKey(
        model_id="codec",
        model_revision="revision",
        encoder_id="neucodec",
        encoder_config_hash=stages.hash_bytes(b"sample_rate:16000"),
        artifact_kind="audar_reference_codes",
        input_key=stages._reference_key(item),
    )

    stored = hook.store_artifact(torch.tensor([7, 8, 9], dtype=torch.long))
    first = hook.load_artifact(stored)
    second = hook.load_artifact(stored)
    assert stored.device.type == "cpu" and stored.dtype == torch.int32
    assert first.device.type == "cpu" and first.dtype == torch.long
    assert torch.equal(first, second)
    assert first.data_ptr() != stored.data_ptr()
    assert first.data_ptr() != second.data_ptr()

    monkeypatch.setattr(stages, "_reference_key", lambda item: None)
    assert hook.cache_key(item) is None


def test_reference_encoder_propagates_singleflight_failure() -> None:
    codec = FakeCodec()
    encode_started = threading.Event()
    release_encode = threading.Event()

    def encode_code(waveform: torch.Tensor) -> torch.Tensor:
        codec.encode_calls += 1
        encode_started.set()
        assert release_encode.wait(timeout=2)
        raise RuntimeError("codec failed")

    codec.encode_code = encode_code
    service = _reference_service(codec)
    reference_audio = {"bytes": five_second_wav()}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        leader = executor.submit(service.get_or_encode, reference_audio)
        assert encode_started.wait(timeout=2)
        follower = executor.submit(service.get_or_encode, reference_audio)
        deadline = time.monotonic() + 2
        while service.stats()["merged"] < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.stats()["merged"] == 1
        release_encode.set()
        for future in (leader, follower):
            with pytest.raises(RuntimeError, match="codec failed"):
                future.result(timeout=2)

    assert codec.encode_calls == 1
    assert service.stats()["failed"] == 1


def test_reference_encoder_revalidates_changed_path(tmp_path) -> None:
    codec = FakeCodec()
    encode_started = threading.Event()
    release_encode = threading.Event()
    reference_path = tmp_path / "reference.wav"
    original_audio = five_second_wav(0)
    changed_audio = five_second_wav(1)
    reference_path.write_bytes(original_audio)

    def encode_code(waveform: torch.Tensor) -> torch.Tensor:
        codec.encode_calls += 1
        if codec.encode_calls == 1:
            encode_started.set()
            assert release_encode.wait(timeout=2)
        return torch.tensor([[[7, 8, 9]]])

    codec.encode_code = encode_code
    service = _reference_service(codec)
    reference_audio = {"audio_path": str(reference_path)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(service.get_or_encode, reference_audio)
        assert encode_started.wait(timeout=2)
        reference_path.write_bytes(changed_audio)
        release_encode.set()
        first.result(timeout=2)

    service.get_or_encode(reference_audio)
    reference_path.write_bytes(original_audio)
    service.get_or_encode(reference_audio)

    assert codec.encode_calls == 3


def test_reference_encoder_reports_cache_stats() -> None:
    codec = FakeCodec()
    service = _reference_service(codec)
    reference_audio = {"bytes": five_second_wav()}

    service.get_or_encode(reference_audio)
    service.get_or_encode(reference_audio)

    assert service.stats() == {
        "hits": 1,
        "misses": 1,
        "merged": 0,
        "entries": 1,
        "bytes": 12,
        "evictions": 0,
        "failed": 0,
        "uncacheable": 0,
    }


def test_codec_model_and_lock_are_shared_between_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = FakeCodec()
    loads = 0

    class FakeNeuCodec:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> FakeCodec:
            nonlocal loads
            loads += 1
            return codec

    monkeypatch.setitem(
        sys.modules, "neucodec", types.SimpleNamespace(NeuCodec=FakeNeuCodec)
    )
    stages._load_codec.cache_clear()
    stages._codec_lock.cache_clear()
    try:
        first = stages._load_codec("codec", "revision", "cpu")
        second = stages._load_codec("codec", "revision", "cpu")
        first_lock = stages._codec_lock("codec", "revision", "cpu")
        second_lock = stages._codec_lock("codec", "revision", "cpu")
    finally:
        stages._load_codec.cache_clear()
        stages._codec_lock.cache_clear()

    assert first is second is codec
    assert first_lock is second_lock
    assert loads == 1


def test_llama_cpp_stage_matches_official_generation_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLlama:
        instance: "FakeLlama"

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.generate_kwargs: dict[str, Any] | None = None
            self.seed: int | None = None
            self.reset_calls = 0
            FakeLlama.instance = self

        def tokenize(self, text: bytes, *, add_bos: bool, special: bool) -> list[int]:
            assert add_bos is False
            assert special is True
            return [99] if text == b"<|TARGET_CODES_END|>" else [1, 2, 3]

        def generate(self, tokens: list[int], **kwargs: Any):
            assert tokens == [1, 2, 3]
            self.generate_kwargs = kwargs
            yield 10
            yield 11
            yield 99

        def detokenize(self, tokens: list[int], *, special: bool) -> bytes:
            assert special is True
            return {10: b"<|speech_123|>", 11: b"<|speech_456|>"}[tokens[0]]

        def set_seed(self, seed: int) -> None:
            self.seed = seed

        def reset(self) -> None:
            self.reset_calls += 1

    monkeypatch.setitem(
        sys.modules,
        "llama_cpp",
        types.SimpleNamespace(LLAMA_SPLIT_MODE_NONE=0, Llama=FakeLlama),
    )
    monkeypatch.setattr(stages, "_resolve_gguf", lambda *args: "/model.gguf")
    payload = make_payload(
        state=AudarTTSState(
            prompt="prompt",
            generation_kwargs={
                "max_new_tokens": 16,
                "temperature": 1.0,
                "top_k": 40,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "seed": 23,
            },
        )
    )

    scheduler = stages.create_tts_engine_executor(
        "audarai/Audar-TTS-V1-Turbo", gpu_id=2
    )
    result = AudarTTSState.from_dict(scheduler._fn(payload).data)

    assert result.audio_codes == [123, 456]
    assert result.prompt is None
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 2
    assert FakeLlama.instance.seed == 23
    assert FakeLlama.instance.reset_calls == 1
    assert FakeLlama.instance.kwargs["main_gpu"] == 2
    assert FakeLlama.instance.generate_kwargs == {
        "temp": 1.0,
        "top_k": 40,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
    }
    assert scheduler._max_concurrency == 1


def test_vocoder_emits_24khz_audio_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = FakeCodec()
    monkeypatch.setattr(stages, "_load_codec", lambda *args, **kwargs: codec)
    scheduler = stages.create_vocoder_executor(gpu_id=None)
    payload = make_payload(
        state=AudarTTSState(
            audio_codes=[1, 2],
            prompt_tokens=3,
            completion_tokens=2,
            engine_time_s=0.25,
        )
    )

    result = asyncio.run(scheduler._fn(payload))

    assert codec.decode_calls == 1
    assert result.data["audio_waveform_shape"] == [3]
    assert result.data["audio_waveform_dtype"] == "float32"
    assert result.data["sample_rate"] == 24000
    assert result.data["modality"] == "audio"
    assert "audio_codes" not in result.data
    assert result.data["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
        "engine_time_s": 0.25,
    }


def test_vocoder_does_not_claim_batching_without_tensor_batch_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = FakeCodec()
    monkeypatch.setattr(stages, "_load_codec", lambda *args, **kwargs: codec)
    scheduler = stages.create_vocoder_executor(gpu_id=None)

    assert scheduler._batch_fn is None
    assert scheduler._max_batch_size == 1
