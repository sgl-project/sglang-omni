# SPDX-License-Identifier: Apache-2.0
"""Minimal TTS request, public HTTP and stage-lifecycle contracts."""

import base64
import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import soundfile
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from fastapi.testclient import TestClient
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from typer.testing import CliRunner

from sglang_omni.cli import app
from sglang_omni.models.moss_speech.config import MossSpeechPipelineConfig
from sglang_omni.models.moss_speech.engine_builder import MossSpeechEngineBuilder
from sglang_omni.models.moss_speech.fsm import MODALITY_PAD
from sglang_omni.models.moss_speech.hf_config import (
    MAX_CONTEXT_TOKENS,
    MossSpeechConfig,
)
from sglang_omni.models.moss_speech.media import decode_chat_audio
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
from sglang_omni.utils.audio import load_audio
from tests.unit_test.fixtures.moss_speech import (
    SpeechClient,
    TextTokenizer,
    audio_reference,
    make_codec,
)


def test_native_rejects_verbatim_synthesis_before_ar() -> None:
    with pytest.raises(RequestValidationError, match="chat/completions"):
        preprocess_request(
            StagePayload(
                "tts",
                OmniRequest(
                    inputs="What is the capital of France?", metadata={"task": "tts"}
                ),
                {},
            ),
            TextTokenizer(),
        )


def test_remaining_context_and_explicit_budget_boundary() -> None:
    payload = StagePayload(
        "boundary",
        OmniRequest(
            inputs=[{"role": "user", "content": "x"}],
            params={},
        ),
        {},
    )
    overhead = (
        MossSpeechState.from_dict(
            preprocess_request(payload, TextTokenizer()).data
        ).prompt_token_count
        - 1
    )
    payload.request.inputs[0]["content"] = "x" * (MAX_CONTEXT_TOKENS - overhead - 1)
    prepared = preprocess_request(payload, TextTokenizer())
    assert build_sglang_moss_request(prepared).max_new_tokens == 1
    payload.request.params["max_new_tokens"] = 2
    with pytest.raises(RequestValidationError, match="budget"):
        preprocess_request(payload, TextTokenizer())
    payload.request.params.clear()
    payload.request.inputs[0]["content"] += "x"
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
                    inputs=[{"role": "user", "content": "Hello"}],
                    params=options,
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
        {"input": "Hello", "unrecognized": True},
        {"input": "Hello", "max_new_tokens": 1},
    ],
)
def test_http_rejects_unsupported_or_empty_audio_without_server_error(
    body: dict[str, object]
) -> None:
    with TestClient(
        create_app(SpeechClient(make_codec()), architectures=["MossSpeechForCausalLM"])
    ) as http:
        response = http.post(
            "/v1/audio/speech", json={"ref_audio": audio_reference(), **body}
        )
        assert response.status_code == 400, response.text


@pytest.mark.parametrize(
    "reference_audio",
    [None, "", "data:audio/wav;base64,YmFk", audio_reference(sample_count=1)],
)
def test_http_rejects_missing_or_invalid_reference_audio(
    reference_audio: str | None,
) -> None:
    with (
        TestClient(
            create_app(
                SpeechClient(make_codec()), architectures=["MossSpeechForCausalLM"]
            )
        ) as http,
        patch("tests.unit_test.fixtures.moss_speech.build_sglang_moss_request") as ar,
    ):
        response = http.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "modalities": ["audio"],
                "audio": {"ref_audio": reference_audio},
            },
        )
        assert response.status_code == 400, response.text
        ar.assert_not_called()


def test_reference_isolated_and_repeated_voice_is_reproducible() -> None:
    with TestClient(
        create_app(SpeechClient(make_codec()), architectures=["MossSpeechForCausalLM"])
    ) as http:
        waves = []
        for frequency in [220, 880, 220]:
            response = http.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hello"}],
                    "modalities": ["audio"],
                    "audio": {"ref_audio": audio_reference(frequency)},
                    "seed": 7,
                },
            )
            assert response.status_code == 200, response.text
            waves.append(response.json()["choices"][0]["message"]["audio"]["data"])
        assert waves[0] == waves[2] and waves[0] != waves[1]


def test_inline_media_released_at_both_serialization_boundaries() -> None:
    source = audio_reference()
    reference = audio_reference(880)
    prepared = preprocess_request(
        StagePayload(
            "reference",
            OmniRequest(
                inputs=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio_url", "audio_url": {"url": source}}
                        ],
                    }
                ],
                metadata={
                    "output_modalities": ["audio"],
                    "audio_config": {"ref_audio": reference},
                },
            ),
            {},
        ),
        TextTokenizer(),
        make_codec(),
    )
    native = build_sglang_moss_request(prepared)
    native.output_rows = [(MODALITY_PAD, 3)]
    for payload in (prepared, build_result(native)):
        serialized = pickle.dumps(payload.to_dict())
        assert (
            source.encode() not in serialized and reference.encode() not in serialized
        )
        state = MossSpeechState.from_dict(payload.data)
        assert state.reference_sample_rate == 24000
        torch.testing.assert_close(
            state.reference_waveform, decode_chat_audio(reference, reference=True)[0]
        )


def test_codec_preserves_usage_rng_and_releases_models() -> None:
    codec = make_codec()
    state = MossSpeechState(
        reference_waveform=decode_chat_audio(audio_reference(), reference=True)[0],
        reference_sample_rate=24000,
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


@pytest.mark.parametrize("sample_rate", [16000, 24000, 44100, 48000])
def test_conditioning_uses_independent_token_and_speaker_resampling(
    tmp_path: Path, sample_rate: int
) -> None:
    """Token conditioning must resample the original audio directly to 16 kHz."""
    seconds = np.arange(sample_rate // 3) / sample_rate
    samples = 0.1 * np.sin(2 * np.pi * 7311 * seconds)
    path = tmp_path / "reference voice.wav"
    soundfile.write(path, samples, sample_rate, subtype="PCM_16")
    source = "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode()
    waveform, _ = torchaudio.load(path)
    expected_token_waveform = torchaudio.transforms.Resample(sample_rate, 16000)(
        waveform
    )
    expected_mel_waveform = torchaudio.transforms.Resample(sample_rate, 24000)(waveform)
    expected_speaker_waveform = torchaudio.transforms.Resample(24000, 16000)(
        expected_mel_waveform
    )
    codec = make_codec()
    codec.feature_extractor = Mock(wraps=codec.feature_extractor)
    codec.feature_extractor.hop_length = 160
    codec.feature_extractor.sampling_rate = 16000
    expected_mel = codec.decoder.feat_extractor(expected_mel_waveform)
    expected_speaker_mel = kaldi.fbank(
        expected_speaker_waveform,
        num_mel_bins=80,
        dither=0,
        sample_frequency=16000,
    )
    expected_speaker_mel -= expected_speaker_mel.mean(dim=0, keepdim=True)
    conditioning = codec.encode_reference(*decode_chat_audio(source, reference=True))
    np.testing.assert_array_equal(
        codec.feature_extractor.call_args.args[0][0], expected_token_waveform[0].numpy()
    )
    expected_mel = expected_mel.transpose(1, 2)
    assert torch.equal(
        conditioning["prompt_feat"],
        expected_mel[:, : conditioning["prompt_feat"].shape[1]],
    )
    np.testing.assert_array_equal(
        codec.decoder.campplus_session.run.call_args.args[1]["speech"],
        expected_speaker_mel.unsqueeze(0).numpy(),
    )


def test_stereo_reference_downmix_preserves_native_sample_rate(tmp_path: Path) -> None:
    samples = np.random.default_rng(42).uniform(-0.2, 0.2, (4410, 2)).astype(np.float32)
    path = tmp_path / "stereo.wav"
    soundfile.write(path, samples, 44100, subtype="FLOAT")
    waveform, sample_rate = load_audio(str(path), preserve_sample_rate=True)
    assert sample_rate == 44100
    np.testing.assert_array_equal(waveform, torch.from_numpy(samples.T).mean(dim=0))


@pytest.mark.parametrize("sample_count", [1, 31 * 24000])
def test_reference_duration_is_validated(tmp_path: Path, sample_count: int) -> None:
    path = tmp_path / "invalid-duration.wav"
    soundfile.write(path, np.zeros(sample_count), 24000)
    with pytest.raises(ValueError, match="short|30 seconds"):
        decode_chat_audio(
            "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode(),
            reference=True,
        )


def test_nonfinite_reference_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "nonfinite.wav"
    soundfile.write(path, np.full(24000, np.nan), 24000, subtype="FLOAT")
    with pytest.raises(ValueError, match="finite"):
        decode_chat_audio(
            "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode(),
            reference=True,
        )


def test_cli_binds_three_stages_and_both_codec_paths(tmp_path: Path) -> None:
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
                "--config",
                str(
                    Path(__file__).resolve().parents[3]
                    / "examples/configs/moss_speech.yaml"
                ),
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
        for stage in ("preprocessing", "audio_vocoder"):
            assert (
                config.stage_named(stage).factory.codec_path
                == "./weights/MOSS-Speech-Codec"
            )
    with pytest.raises(ValueError, match="codec_path"):
        create_audio_vocoder_executor(
            model_path="unused", codec_path=None, device="cpu"
        )


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


@pytest.mark.parametrize("dtype", ["bfloat16", "bf16"])
def test_engine_accepts_bf16_spellings(dtype: str) -> None:
    builder = MossSpeechEngineBuilder()
    builder.server_args_overrides = {"max_running_requests": 4}
    overrides = {"dtype": dtype}
    builder.adjust_overrides(overrides)
    assert overrides["dtype"] == "bfloat16"
    with pytest.raises(ValueError, match="rejects dtype"):
        builder.adjust_overrides({"dtype": "float16"})
