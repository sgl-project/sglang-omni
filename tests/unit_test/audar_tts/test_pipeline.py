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
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

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
from sglang_omni.models.audar_tts.vocoder_graph import (
    capture_decode_graphs,
    decode_hidden,
)
from sglang_omni.models.model_capabilities import get_model_capabilities
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.pipeline.control_plane import deserialize_message, serialize_message
from sglang_omni.pipeline.mp_runner import build_stage_groups
from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime
from sglang_omni.platforms import current_platform
from sglang_omni.proto import DataReadyMessage, OmniRequest, StagePayload
from sglang_omni.relay.shm import ShmRelay
from sglang_omni.serve.speech_errors import SpeechAPIError
from sglang_omni.serve.speech_service import SpeechRequestValidator
from sglang_omni.utils.imports import import_string
from tests.unit_test.fixtures.pipeline_fakes import FakeMpContext

requires_xpu = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="requires XPU",
)
NEUCODEC_HOP_LENGTH = 480
FSQ_CODEBOOK_SIZE = 4**8
# note (anupa): eager decode_code reruns on one Arc Pro B60 already differ by up
# to 1.1e-7, so graph and eager outputs are compared with a tolerance.
DECODE_TOLERANCE = 1e-5


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


class GraphCodec(FakeCodec):
    """A codec whose iSTFT head is reachable, as a decode-graph replay needs."""

    def __init__(self) -> None:
        super().__init__()
        self.head_inputs: list[torch.Tensor] = []
        self.generator = types.SimpleNamespace(head=self.head)

    def head(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.head_inputs.append(hidden)
        return torch.tensor([[[0.5, -0.25]]]), hidden


class FakeDecodeGraph:
    def __init__(self) -> None:
        self.replayed: list[list[int]] = []
        self.hidden = torch.ones((1, 4, 2))

    def replay(self, codes: torch.Tensor) -> torch.Tensor:
        self.replayed.append(codes.reshape(-1).tolist())
        return self.hidden


class RecordingLlama:
    """Replays one fixed generation and records how the stage drove it."""

    instance: RecordingLlama

    def __init__(
        self,
        *,
        model_path: str,
        n_ctx: int,
        n_gpu_layers: int,
        split_mode: int,
        main_gpu: int,
        verbose: bool,
    ) -> None:
        self.n_gpu_layers = n_gpu_layers
        self.main_gpu = main_gpu
        self.sampling: dict[str, float] | None = None
        self.seeds: list[int] = []
        self.evaluated: list[list[int]] = []
        self.reset_calls = 0
        RecordingLlama.instance = self

    def tokenize(self, text: bytes, *, add_bos: bool, special: bool) -> list[int]:
        assert add_bos is False
        assert special is True
        return [99] if text == b"<|TARGET_CODES_END|>" else [1, 2, 3]

    def eval(self, tokens: list[int]) -> None:
        self.evaluated.append(list(tokens))

    def generate(
        self,
        tokens: list[int],
        *,
        temp: float,
        top_k: int,
        top_p: float,
        repeat_penalty: float,
    ) -> Iterator[int]:
        assert tokens == [1, 2, 3]
        self.sampling = {
            "temp": temp,
            "top_k": top_k,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
        }
        yield 10
        yield 11
        yield 99

    def detokenize(self, tokens: list[int], *, special: bool) -> bytes:
        assert special is True
        return {10: b"<|speech_123|>", 11: b"<|speech_456|>"}[tokens[0]]

    def set_seed(self, seed: int) -> None:
        self.seeds.append(seed)

    def reset(self) -> None:
        self.reset_calls += 1


def fake_llama_cpp(*, gpu_offload: bool = True) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        LLAMA_DEFAULT_SEED=0xFFFFFFFF,
        LLAMA_SPLIT_MODE_NONE=0,
        Llama=RecordingLlama,
        llama_supports_gpu_offload=lambda: gpu_offload,
    )


class NeuCodecDecoder(nn.Module):
    """NeuCodec's decode path without the w2v-bert encoder its constructor downloads."""

    def __init__(self) -> None:
        from neucodec.codec_decoder_vocos import CodecDecoderVocos

        super().__init__()
        self.generator = CodecDecoderVocos(hop_length=NEUCODEC_HOP_LENGTH)
        self.fc_post_a = nn.Linear(2048, 1024)

    def decode_code(self, codes: torch.Tensor) -> torch.Tensor:
        from neucodec.model import NeuCodec

        return NeuCodec.decode_code(self, codes)


@pytest.fixture(scope="module")
def xpu_decoder() -> NeuCodecDecoder:
    pytest.importorskip("neucodec.model")
    torch.manual_seed(0)
    return NeuCodecDecoder().eval().to(torch.device("xpu", 0))


def random_xpu_codes(code_count: int, generator: torch.Generator) -> torch.Tensor:
    codes = torch.randint(0, FSQ_CODEBOOK_SIZE, (1, 1, code_count), generator=generator)
    return codes.to(torch.device("xpu", 0))


def engine_payload(*, seed: int | None) -> StagePayload:
    generation_kwargs: dict[str, int | float] = {
        "max_new_tokens": 16,
        "temperature": 1.0,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
    }
    if seed is not None:
        generation_kwargs["seed"] = seed
    else:
        pass
    return make_payload(
        state=AudarTTSState(prompt="prompt", generation_kwargs=generation_kwargs)
    )


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
        groups = build_stage_groups(
            config,
            ctx=FakeMpContext(),
            stages_cfg=prepared.stages_cfg,
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
            spec.factory_kwargs,
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
        request=Client.build_omni_request(generation_request),
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
    monkeypatch.setattr(stages, "load_codec", lambda *args, **kwargs: codec)
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
        return AudarTTSState.from_dict(scheduler.fn(payload).data)

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
    original_normalize = stages.normalize_reference

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

    monkeypatch.setattr(stages, "normalize_reference", normalize)
    monkeypatch.setattr(codec, "encode_code", encode_code)
    monkeypatch.setattr(stages, "load_codec", lambda *args, **kwargs: codec)
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
        return AudarTTSState.from_dict(scheduler.fn(payload).data)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(encode, "first")
        assert encode_started.wait(timeout=2)
        second = executor.submit(encode, "second")
        assert second_normalized.wait(timeout=2)
        release_encode.set()
        results = [first.result(timeout=2), second.result(timeout=2)]

    assert scheduler.max_concurrency == 2
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
    monkeypatch.setattr(stages, "load_codec", lambda *args, **kwargs: codec)
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
        return AudarTTSState.from_dict(scheduler.fn(payload).data)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(encode, "first", five_second_wav(0)),
            executor.submit(encode, "second", five_second_wav(1)),
        ]
        results = [future.result(timeout=3) for future in futures]

    assert codec.encode_calls == 2
    assert max_active_calls == 1
    assert all(result.prompt for result in results)


def reference_service(codec: FakeCodec) -> Any:
    hook = stages.AudarReferenceEncodeHook(
        codec=codec,
        device="cpu",
        codec_model="codec",
        codec_revision="revision",
    )
    return stages.ReferenceEncodeService(hook)


def test_reference_hook_preserves_key_and_tensor_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hook = stages.AudarReferenceEncodeHook(
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
        input_key=stages.reference_key(item),
    )

    stored = hook.store_artifact(torch.tensor([7, 8, 9], dtype=torch.long))
    first = hook.load_artifact(stored)
    second = hook.load_artifact(stored)
    assert stored.device.type == "cpu" and stored.dtype == torch.int32
    assert first.device.type == "cpu" and first.dtype == torch.long
    assert torch.equal(first, second)
    assert first.data_ptr() != stored.data_ptr()
    assert first.data_ptr() != second.data_ptr()

    monkeypatch.setattr(stages, "reference_key", lambda item: None)
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
    service = reference_service(codec)
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
    service = reference_service(codec)
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
    service = reference_service(codec)
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
        "batches": 0,
        "batched_items": 0,
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
    stages.load_codec.cache_clear()
    stages.codec_lock.cache_clear()
    try:
        first = stages.load_codec("codec", "revision", "cpu")
        second = stages.load_codec("codec", "revision", "cpu")
        first_lock = stages.codec_lock("codec", "revision", "cpu")
        second_lock = stages.codec_lock("codec", "revision", "cpu")
    finally:
        stages.load_codec.cache_clear()
        stages.codec_lock.cache_clear()

    assert first is second is codec
    assert first_lock is second_lock
    assert loads == 1


def test_llama_cpp_stage_keeps_a_cpu_resolution_off_the_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """n_gpu_layers defaults to -1 (offload everything); a cpu resolution must
    zero it or the stage silently runs on GPU 0 anyway."""
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama_cpp())
    monkeypatch.setattr(stages, "resolve_gguf", lambda *args: "/model.gguf")
    monkeypatch.setattr(current_platform, "device_type", "cpu", raising=False)

    stages.create_tts_engine_executor("unused", device="cpu")

    assert RecordingLlama.instance.n_gpu_layers == 0
    assert RecordingLlama.instance.main_gpu == 0
    assert RecordingLlama.instance.evaluated == []


def test_llama_cpp_stage_rejects_a_device_it_cannot_serve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """llama.cpp binds a bare main_gpu index; a device intent outside
    cuda/xpu/cpu would be silently mapped onto the wrong backend's card
    numbering."""
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama_cpp())
    monkeypatch.setattr(current_platform, "device_type", "mps", raising=False)
    with pytest.raises(ValueError, match="cuda, xpu or cpu"):
        stages.create_tts_engine_executor("unused", device="mps", gpu_id=1)


def test_llama_cpp_stage_refuses_xpu_on_a_build_that_cannot_offload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A llama-cpp-python build without SYCL would run every layer on the CPU
    while the stage reports xpu."""
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama_cpp(gpu_offload=False))
    monkeypatch.setattr(current_platform, "device_type", "xpu", raising=False)
    with pytest.raises(RuntimeError, match="GGML_SYCL"):
        stages.create_tts_engine_executor("unused", device="xpu", gpu_id=0)


def test_llama_cpp_stage_offloads_to_the_resolved_xpu_and_warms_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """llama.cpp's SYCL device i is torch's xpu:i, and the SYCL backend switches
    kernels on its first one-token decode, so the stage takes that decode
    before the first request."""
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama_cpp())
    monkeypatch.setattr(stages, "resolve_gguf", lambda *args: "/model.gguf")
    monkeypatch.setattr(current_platform, "device_type", "xpu", raising=False)

    scheduler = stages.create_tts_engine_executor("unused", device="xpu", gpu_id=1)
    llm = RecordingLlama.instance

    assert llm.main_gpu == 1
    assert llm.n_gpu_layers == -1
    assert llm.evaluated == [[99] * stages.XPU_WARMUP_PROMPT_TOKENS, [99]]
    assert llm.reset_calls == 1

    result = AudarTTSState.from_dict(scheduler.fn(engine_payload(seed=23)).data)

    assert result.audio_codes == [123, 456]
    assert llm.reset_calls == 2
    assert llm.seeds == [23]


def test_llama_cpp_stage_does_not_carry_a_seed_into_an_unseeded_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Llama keeps the last seed it was set, so an unseeded request after a
    seeded one would sample exactly like it unless the stage resets the seed."""
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama_cpp())
    monkeypatch.setattr(stages, "resolve_gguf", lambda *args: "/model.gguf")
    monkeypatch.setattr(current_platform, "device_type", "cuda", raising=False)
    scheduler = stages.create_tts_engine_executor("unused", gpu_id=0)

    scheduler.fn(engine_payload(seed=23))
    scheduler.fn(engine_payload(seed=None))
    scheduler.fn(engine_payload(seed=23))

    assert RecordingLlama.instance.seeds == [23, 0xFFFFFFFF, 23]
    assert RecordingLlama.instance.evaluated == []


def test_llama_cpp_stage_matches_official_generation_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama_cpp())
    monkeypatch.setattr(stages, "resolve_gguf", lambda *args: "/model.gguf")
    monkeypatch.setattr(current_platform, "device_type", "cuda", raising=False)
    payload = engine_payload(seed=23)

    scheduler = stages.create_tts_engine_executor(
        "audarai/Audar-TTS-V1-Turbo", gpu_id=2
    )
    result = AudarTTSState.from_dict(scheduler.fn(payload).data)

    assert result.audio_codes == [123, 456]
    assert result.prompt is None
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 2
    assert RecordingLlama.instance.seeds == [23]
    assert RecordingLlama.instance.reset_calls == 1
    assert RecordingLlama.instance.evaluated == []
    assert RecordingLlama.instance.main_gpu == 2
    assert RecordingLlama.instance.sampling == {
        "temp": 1.0,
        "top_k": 40,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
    }
    assert scheduler.max_concurrency == 1


def test_vocoder_emits_24khz_audio_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = FakeCodec()
    monkeypatch.setattr(stages, "load_codec", lambda *args, **kwargs: codec)
    scheduler = stages.create_vocoder_executor(gpu_id=None)
    payload = make_payload(
        state=AudarTTSState(
            audio_codes=[1, 2],
            prompt_tokens=3,
            completion_tokens=2,
            engine_time_s=0.25,
        )
    )

    result = asyncio.run(scheduler.fn(payload))

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
    monkeypatch.setattr(stages, "load_codec", lambda *args, **kwargs: codec)
    scheduler = stages.create_vocoder_executor(gpu_id=None)

    assert scheduler.batch_fn is None
    assert scheduler.max_batch_size == 1


def test_vocoder_captures_no_graph_unless_code_counts_are_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse_capture(
        codec: FakeCodec, device: torch.device, code_counts: list[int]
    ) -> dict[int, FakeDecodeGraph]:
        raise AssertionError("no graph was declared")

    monkeypatch.setattr(stages, "load_codec", lambda *args, **kwargs: FakeCodec())
    monkeypatch.setattr(stages, "capture_decode_graphs", refuse_capture)

    stages.create_vocoder_executor(device="cpu")


def test_vocoder_refuses_code_counts_given_as_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse_load(model: str, revision: str, device: str) -> FakeCodec:
        raise AssertionError("the codec should not load for a malformed knob")

    monkeypatch.setattr(stages, "load_codec", refuse_load)

    with pytest.raises(TypeError, match="YAML config"):
        stages.create_vocoder_executor(
            device="cpu", decode_graph_code_counts="[150,250]"
        )


def test_vocoder_replays_a_captured_code_count_and_decodes_the_rest_eagerly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A graph serves exactly the code count it was recorded at; any other
    count must take decode_code rather than a padded replay."""
    codec = GraphCodec()
    decode_graph = FakeDecodeGraph()
    declared: list[list[int]] = []

    def fake_capture(
        codec_arg: GraphCodec, device: torch.device, code_counts: list[int]
    ) -> dict[int, FakeDecodeGraph]:
        assert codec_arg is codec
        assert device == torch.device("cpu")
        declared.append(list(code_counts))
        return {2: decode_graph}

    monkeypatch.setattr(stages, "load_codec", lambda *args, **kwargs: codec)
    monkeypatch.setattr(stages, "capture_decode_graphs", fake_capture)
    scheduler = stages.create_vocoder_executor(
        device="cpu", decode_graph_code_counts=[2]
    )

    hit = asyncio.run(
        scheduler.fn(make_payload(state=AudarTTSState(audio_codes=[1, 2])))
    )
    miss = asyncio.run(
        scheduler.fn(make_payload(state=AudarTTSState(audio_codes=[1, 2, 3])))
    )

    assert declared == [[2]]
    assert decode_graph.replayed == [[1, 2]]
    assert codec.head_inputs == [decode_graph.hidden]
    assert codec.decode_calls == 1
    assert hit.data["audio_waveform_shape"] == [2]
    assert hit.data["sample_rate"] == 24000
    assert miss.data["audio_waveform_shape"] == [3]


def test_vocoder_graph_capture_needs_a_graph_capable_device() -> None:
    with pytest.raises(ValueError, match="graph-capable"):
        capture_decode_graphs(nn.Module(), torch.device("cpu"), [50])


def test_vocoder_graph_capture_rejects_a_code_count_below_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        current_platform, "get_device_graph_backend", lambda device: object()
    )
    with pytest.raises(ValueError, match="positive"):
        capture_decode_graphs(nn.Module(), torch.device("cpu"), [50, 0])


@requires_xpu
def test_neucodec_decoder_front_and_head_reproduce_decode_code(
    xpu_decoder: NeuCodecDecoder,
) -> None:
    codes = random_xpu_codes(50, torch.Generator().manual_seed(1))

    with torch.inference_mode():
        expected = xpu_decoder.decode_code(codes).cpu()
        split = xpu_decoder.generator.head(decode_hidden(xpu_decoder, codes))[0].cpu()

    assert split.shape == expected.shape == (1, 1, 50 * NEUCODEC_HOP_LENGTH)
    torch.testing.assert_close(split, expected, atol=DECODE_TOLERANCE, rtol=0)


@requires_xpu
def test_neucodec_decode_graphs_replay_new_codes_on_another_thread(
    xpu_decoder: NeuCodecDecoder,
) -> None:
    graphs = capture_decode_graphs(xpu_decoder, torch.device("xpu", 0), [50, 150])
    generator = torch.Generator().manual_seed(2)

    def replay(code_count: int, codes: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            hidden = graphs[code_count].replay(codes)
            return xpu_decoder.generator.head(hidden)[0].cpu()

    assert sorted(graphs) == [50, 150]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as worker:
        for code_count in (50, 150):
            recorded = graphs[code_count].graph
            previous = None
            for _ in range(3):
                codes = random_xpu_codes(code_count, generator)
                replayed = worker.submit(replay, code_count, codes).result()
                with torch.inference_mode():
                    expected = xpu_decoder.decode_code(codes).cpu()

                assert graphs[code_count].graph is recorded
                assert replayed.shape == (1, 1, code_count * NEUCODEC_HOP_LENGTH)
                assert torch.isfinite(replayed).all()
                torch.testing.assert_close(
                    replayed, expected, atol=DECODE_TOLERANCE, rtol=0
                )
                assert previous is None or not torch.equal(replayed, previous)
                previous = replayed
