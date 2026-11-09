# SPDX-License-Identifier: Apache-2.0
"""Minimal TTS request, public HTTP and stage-lifecycle contracts."""

import base64
import io
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import soundfile
import torch
import torchaudio
from fastapi.testclient import TestClient
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from transformers import EncoderDecoderCache, WhisperFeatureExtractor
from typer.testing import CliRunner

from sglang_omni.cli import app
from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateChunk, GenerateRequest
from sglang_omni.models.moss_speech.codec import AudioEncoderOutput, MossSpeechCodec
from sglang_omni.models.moss_speech.config import MossSpeechPipelineConfig
from sglang_omni.models.moss_speech.engine_builder import MossSpeechEngineBuilder
from sglang_omni.models.moss_speech.fsm import AUDIO_PAD, MODALITY_PAD, SOSP
from sglang_omni.models.moss_speech.hf_config import (
    MAX_CONTEXT_TOKENS,
    MossSpeechConfig,
)
from sglang_omni.models.moss_speech.request_builders import (
    MossSpeechState,
    RequestValidationError,
    build_result,
    build_sglang_moss_request,
    preprocess_request,
)
from sglang_omni.models.moss_speech.stages import create_audio_vocoder_executor
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.serve.launcher import model_capabilities_log_summary
from sglang_omni.serve.openai_api import create_app


class TextTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return list(text.encode("utf-8"))


class ReferenceEncoder(torch.nn.Module):
    def forward(
        self,
        *,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool,
        past_key_values: EncoderDecoderCache,
    ) -> AudioEncoderOutput:
        return SimpleNamespace(
            quantized_token_ids=torch.zeros(
                (1, input_features.shape[-1] // 8), dtype=torch.long
            )
        )


class OfflineDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feat_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000, n_fft=1024, hop_length=480, n_mels=80
        )
        self.campplus_session = Mock()
        self.campplus_session.get_inputs.return_value = [SimpleNamespace(name="speech")]
        self.campplus_session.run.return_value = [np.zeros((1, 192), dtype=np.float32)]

    def token2wav(
        self,
        token: torch.Tensor,
        uuid: str,
        *,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        finalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert finalize and token.shape == (1, 2)
        waveform = torch.rand(1, 1920) * 0.1 + prompt_feat.mean() / 1000
        return waveform, torch.zeros(1, 80, 4)


def make_codec() -> MossSpeechCodec:
    codec = object.__new__(MossSpeechCodec)
    codec.device = torch.device("cpu")
    codec.decoder = OfflineDecoder()
    codec.encoder = ReferenceEncoder()
    codec.feature_extractor = WhisperFeatureExtractor()
    codec.feature_frames_per_token = 8
    return codec


def audio_reference(frequency_hz: int = 220, sample_count: int = 24000) -> str:
    waveform = 0.1 * np.sin(2 * np.pi * frequency_hz * np.arange(sample_count) / 24000)
    encoded = io.BytesIO()
    soundfile.write(encoded, waveform, 24000, format="WAV")
    return "data:audio/wav;base64," + base64.b64encode(encoded.getvalue()).decode()


class SpeechClient(Client):
    def __init__(self, codec: MossSpeechCodec) -> None:
        self.codec = codec

    async def generate(
        self, request: GenerateRequest, request_id: str | None = None
    ) -> AsyncIterator[GenerateChunk]:
        request_id = request_id or "speech"
        prepared = preprocess_request(
            StagePayload(request_id, Client.build_omni_request(request), {}),
            TextTokenizer(),
        )
        native = build_sglang_moss_request(prepared)
        native.output_rows = [(SOSP, AUDIO_PAD), (MODALITY_PAD, 3), (MODALITY_PAD, 4)][
            : native.max_new_tokens
        ]
        native.finish_reason = "length"
        result = self.codec.decode(build_result(native))
        yield Client.default_result_builder(request_id, result.data)


def test_native_prompt_preserves_user_question_and_budget() -> None:
    """Questions must reach the model without an injected repetition instruction."""
    payload = StagePayload(
        "request",
        OmniRequest(
            inputs="What is the capital of France?",
            params={"seed": 19, "max_new_tokens": 32},
            metadata={"tts_params": {"ref_audio": "reference.wav"}},
        ),
        {},
    )
    prepared = preprocess_request(payload, TextTokenizer())
    state = MossSpeechState.from_dict(prepared.data)
    assert bytes(state.text_token_ids).decode() == (
        "<|im_start|>system\nYou are a helpful voice assistant. Answer the user's questions with spoken responses.<|im_end|>\n"
        "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n"
        "<|im_start|>assistant\n<|object_ref_start|>"
    )
    native = build_sglang_moss_request(prepared)
    assert state.reference_audio == "reference.wav"
    assert native.effective_seed == 19 and native.max_new_tokens == 32
    assert native.prompt_rows[:, 0].tolist() == state.text_token_ids
    assert native.prompt_rows[:, 1].tolist() == [AUDIO_PAD] * state.prompt_token_count
    assert native.req.sampling_params.max_new_tokens == 32


def test_remaining_context_and_explicit_budget_boundary() -> None:
    payload = StagePayload(
        "boundary",
        OmniRequest(
            inputs="x",
            params={},
            metadata={"tts_params": {"ref_audio": "reference.wav"}},
        ),
        {},
    )
    overhead = (
        MossSpeechState.from_dict(
            preprocess_request(payload, TextTokenizer()).data
        ).prompt_token_count
        - 1
    )
    payload.request.inputs = "x" * (MAX_CONTEXT_TOKENS - overhead - 1)
    prepared = preprocess_request(payload, TextTokenizer())
    assert build_sglang_moss_request(prepared).max_new_tokens == 1
    payload.request.params["max_new_tokens"] = 2
    with pytest.raises(RequestValidationError, match="budget"):
        preprocess_request(payload, TextTokenizer())
    payload.request.params.clear()
    payload.request.inputs += "x"
    with pytest.raises(RequestValidationError, match="budget"):
        preprocess_request(payload, TextTokenizer())


@pytest.mark.parametrize(
    "options",
    [
        {"stream": True},
        {"min_p": 0.2},
        {"stop": ["stop"]},
        {"top_k": -2},
        {"temperature": float("nan")},
        {"seed": 2**64},
        {"stage_params": {"ar_engine": {}}},
    ],
)
def test_unsupported_sampling_fails_before_generation(
    options: dict[str, object]
) -> None:
    with pytest.raises(RequestValidationError):
        preprocess_request(
            StagePayload(
                "invalid",
                OmniRequest(
                    inputs="Hello",
                    params=options,
                    metadata={"tts_params": {"ref_audio": "reference.wav"}},
                ),
                {},
            ),
            TextTokenizer(),
        )


@pytest.mark.parametrize(
    "body",
    [
        {"input": "Hello", "stream": True},
        {"input": "Hello", "voice": "other"},
        {"input": "Hello", "instructions": "whisper"},
        {"input": "Hello", "max_new_tokens": 1},
    ],
)
def test_http_rejects_unsupported_or_empty_audio_without_server_error(
    body: dict[str, object]
) -> None:
    with TestClient(
        create_app(SpeechClient(make_codec()), model_name="moss-speech")
    ) as http:
        response = http.post(
            "/v1/audio/speech", json={"ref_audio": audio_reference(), **body}
        )
        assert response.status_code == 400, response.text


def test_public_speech_encodes_waveform_and_chat_is_rejected() -> None:
    with TestClient(
        create_app(SpeechClient(make_codec()), model_name="moss-speech")
    ) as http:
        response = http.post(
            "/v1/audio/speech",
            json={
                "input": "Hello",
                "max_new_tokens": 3,
                "seed": 7,
                "ref_audio": audio_reference(),
            },
        )
        assert response.status_code == 200, response.text
        assert response.content[:4] == b"RIFF" and len(response.content) > 44
        response = http.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}], "stream": False},
        )
        assert response.status_code == 400, response.text


@pytest.mark.parametrize(
    "reference_audio",
    [None, "", "data:audio/wav;base64,YmFk", audio_reference(sample_count=1)],
)
def test_http_rejects_missing_or_invalid_reference_audio(
    reference_audio: str | None,
) -> None:
    with TestClient(
        create_app(SpeechClient(make_codec()), model_name="moss-speech")
    ) as http:
        response = http.post(
            "/v1/audio/speech",
            json={"input": "Hello", "max_new_tokens": 3, "ref_audio": reference_audio},
        )
        assert response.status_code == 400, response.text


def test_http_reference_audio_changes_per_request_without_restarting_codec() -> None:
    codec = make_codec()
    with TestClient(create_app(SpeechClient(codec), model_name="moss-speech")) as http:
        responses = [
            http.post(
                "/v1/audio/speech",
                json={
                    "input": "Hello",
                    "ref_audio": reference,
                    "max_new_tokens": 3,
                    "seed": 7,
                },
            )
            for reference in (
                audio_reference(220),
                audio_reference(880),
                audio_reference(220),
            )
        ]
        assert all(response.status_code == 200 for response in responses)
        assert responses[0].content == responses[2].content
        assert responses[0].content != responses[1].content


def test_native_reference_descriptor_survives_result_adapter() -> None:
    reference = audio_reference()
    prepared = preprocess_request(
        StagePayload(
            "reference",
            OmniRequest(
                inputs={
                    "text": "Hello",
                    "references": [
                        {"media_type": "audio/wav", "data": reference.split(",", 1)[1]}
                    ],
                }
            ),
            {},
        ),
        TextTokenizer(),
    )
    native = build_sglang_moss_request(prepared)
    native.output_rows = [(MODALITY_PAD, 3)]
    assert (
        MossSpeechState.from_dict(build_result(native).data).reference_audio
        == reference
    )


def test_codec_preserves_usage_rng_and_releases_models() -> None:
    codec = make_codec()
    state = MossSpeechState(
        reference_audio=audio_reference(),
        audio_codes=[3, 4],
        prompt_token_count=12,
        completion_token_count=3,
        effective_seed=7,
        finish_reason="length",
    )
    payload = StagePayload("codec", OmniRequest(inputs="Hello"), state.to_dict())
    generator_state = torch.get_rng_state().clone()
    first, second = codec.decode(payload), codec.decode(payload)
    assert first.data == second.data
    assert first.data["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "total_tokens": 15,
    }
    assert (
        first.data["sample_rate"] == 24000 and first.data["finish_reason"] == "length"
    )
    assert torch.equal(torch.get_rng_state(), generator_state)
    codec.close()
    assert codec.decoder is None and codec.encoder is None


def test_cli_binds_only_three_stages_and_vocoder_assets(tmp_path: Path) -> None:
    MossSpeechConfig(
        audio_vocab_size=16512, architectures=["MossSpeechForCausalLM"]
    ).to_json_file(tmp_path / "config.json")
    with patch("sglang_omni.cli.serve.launch_server") as launch:
        result = CliRunner().invoke(
            app,
            [
                "serve",
                "--model-path",
                str(tmp_path),
                "--audio_vocoder.factory.codec_path",
                "codec",
            ],
        )
        assert result.exit_code == 0, result.output + repr(result.exception)
        config = launch.call_args.args[0]
        assert isinstance(config, MossSpeechPipelineConfig)
        assert model_capabilities_log_summary(config)["reference_audio"] is True
        assert [stage.name for stage in config.stages] == [
            "preprocessing",
            "ar_engine",
            "audio_vocoder",
        ]
        assert config.stage_named("audio_vocoder").factory.codec_path == "codec"
    with pytest.raises(ValueError, match="codec_path"):
        create_audio_vocoder_executor(codec_path=None, device="cpu")


@pytest.mark.parametrize("prompt_tokens,budget", [(40959, 1), (39960, 1000)])
def test_full_context_survives_scheduler_admission(
    prompt_tokens: int, budget: int
) -> None:
    scheduler = object.__new__(OmniScheduler)
    scheduler.model_config = SimpleNamespace(context_len=40960)
    scheduler.max_total_num_tokens = 40962
    scheduler.model_runner = Mock()
    scheduler.kv_shard_widening = scheduler.page_size = 1
    scheduler.max_new_tokens_limit = scheduler.sliding_window_size = (
        scheduler.draft_worker
    ) = None
    scheduler.chunked_prefill_size = -1
    scheduler.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
        size=40962, dtype=torch.bfloat16, device="cpu", kvcache=None, need_sort=False
    )
    MossSpeechEngineBuilder().post_scheduler_setup(scheduler, scheduler.model_runner)
    state = MossSpeechState(text_token_ids=[12] * prompt_tokens, max_new_tokens=budget)
    request = build_sglang_moss_request(
        StagePayload("window", OmniRequest(inputs="Hello"), state.to_dict())
    )
    assert scheduler.prepare_request_limits(request) is None
    assert (
        request.req.sampling_params.max_new_tokens == request.max_new_tokens == budget
    )
    assert scheduler.request_kv_capacity_error(request.req) is None
