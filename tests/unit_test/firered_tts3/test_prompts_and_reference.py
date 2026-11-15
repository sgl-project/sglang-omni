# SPDX-License-Identifier: Apache-2.0
"""Prompt formatting, reference audio boundaries, and reference caching."""

from __future__ import annotations

import base64
import io
from collections.abc import Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateRequest
from sglang_omni.models.firered_tts3.hf_config import FireRedTTS3Config
from sglang_omni.models.firered_tts3.ref_encode import FireRedReferenceBundle
from sglang_omni.models.firered_tts3.request_builders import (
    build_sglang_firered_tts3_request,
)
from sglang_omni.models.firered_tts3.speaker_prompts import (
    compose_speaker_chatml,
    encode_speaker_text,
    resolve_speaker_token_id,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.utils.json import JsonValue
from tests.unit_test.fixtures.firered_tts3 import make_speaker_tokenizer
from tests.unit_test.fixtures.firered_tts3_preprocessing import (
    ReferenceLatentEncoder,
    make_preprocessing_payload,
    preprocess_payload,
)


@pytest.fixture
def audio_server_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    with ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    ) as server:
        server_thread = Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        server_thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            server_thread.join()


def test_speaker_chatml_rejects_bad_ratio() -> None:
    with pytest.raises(ValueError, match="interleave_ratio"):
        compose_speaker_chatml(
            "<|f531_speaker|>", dual_stream=True, interleave_ratio=(0, 1)
        )


@pytest.mark.parametrize(
    "text,expected_tokens",
    [("你 好世界", ["你", "好", "世", "界"]), ("生僻", ["生僻"])],
)
def test_speaker_text_expands_han_tokens_and_preserves_unknown_characters(
    text: str, expected_tokens: list[str]
) -> None:
    tokenizer = make_speaker_tokenizer()
    token_ids = encode_speaker_text(tokenizer, text)
    assert token_ids == [
        tokenizer.convert_tokens_to_ids(token) for token in expected_tokens
    ]


@pytest.mark.parametrize(
    "sample_rate,sample_count,cap_seconds,expected_patches",
    [
        (24000, 3840, None, 1),
        (24000, 3841, None, 2),
        (16000, 2561, None, 2),
        (44100, 7056, None, 1),
        (44100, 7057, None, 2),
        (16000, 320001, 20.0, 125),
        (16000, 2561, 0.16005, 2),
    ],
)
def test_audio_prompt_slots_follow_resampled_patch_boundaries(
    tmp_path: Path,
    audio_server_url: str,
    monkeypatch: pytest.MonkeyPatch,
    sample_rate: int,
    sample_count: int,
    cap_seconds: float | None,
    expected_patches: int,
) -> None:
    """Reference frames must reflect the normalized, capped, patch-padded audio."""
    encoded_audio = io.BytesIO()
    sf.write(
        encoded_audio,
        np.zeros(sample_count, dtype=np.float32),
        sample_rate,
        format="WAV",
        subtype="FLOAT",
    )
    audio_bytes = encoded_audio.getvalue()
    audio_path = tmp_path / "reference.wav"
    audio_path.write_bytes(audio_bytes)
    audio_uri = f"data:audio/wav;base64,{base64.b64encode(audio_bytes).decode('ascii')}"
    audio_url = f"{audio_server_url}/reference.wav"
    monkeypatch.setenv("REQUEST_TIMEOUT", "1")
    with patch(
        "sglang_omni.models.firered_tts3.ref_encode.load_redae_encoder",
        return_value=ReferenceLatentEncoder(FireRedTTS3Config()),
    ):
        bundle = FireRedReferenceBundle(
            "unused", FireRedTTS3Config(use_lm_head=True), "cpu"
        )
    for reference in (str(audio_path), audio_uri, audio_url):
        artifact = bundle.encode(reference, max_seconds=cap_seconds)
        assert artifact["prompt_latents"].shape == (
            1,
            expected_patches * bundle.config.patch_size,
            bundle.config.redae_dim,
        )


@pytest.mark.parametrize(
    "request_fields,tts_parameters,preprocessing_parameters,expected_instruction",
    [
        ({}, {}, {}, "adjust the speed to 0.8x"),
        ({}, {"speed": 1.2}, {}, "adjust the speed to 0.8x"),
        (
            {},
            {},
            {"acoustic_attribute": "Speed", "acoustic_value": 1.2},
            "adjust the speed to 1.2x",
        ),
        (
            {},
            {},
            {"acoustic_attribute": "Volume", "acoustic_value": 1.5},
            "adjust the volume to 1.5",
        ),
        (
            {},
            {},
            {"acoustic_attribute": "Pitch", "acoustic_value": 2},
            "shift the pitch by 2 steps",
        ),
        (
            {},
            {},
            {"acoustic_attribute": "Pitch", "acoustic_value": -1},
            "shift the pitch by -1 step",
        ),
        (
            {"instructions": "adjust the volume to 0.5"},
            {},
            {"acoustic_attribute": "Pitch", "acoustic_value": 2},
            "adjust the volume to 0.5",
        ),
    ],
)
def test_client_acoustic_parameters_reach_model_prompt(
    request_fields: dict[str, JsonValue],
    tts_parameters: dict[str, JsonValue],
    preprocessing_parameters: dict[str, JsonValue],
    expected_instruction: str,
) -> None:
    """SDK speed must reach AcousticEdit without overriding instructions or stage values."""
    request = GenerateRequest(
        prompt={
            "input": "",
            "task_type": "AcousticEdit",
            "references": [{"audio_path": "ref.wav"}],
            "speed": 0.8,
            **request_fields,
        },
        metadata={"tts_params": tts_parameters},
        stage_params={"preprocessing": preprocessing_parameters},
        stream=False,
    )
    payload = StagePayload(
        request_id="acoustic-edit",
        request=Client.build_omni_request(request),
        data={},
    )
    tokenizer = make_speaker_tokenizer()
    state = preprocess_payload(
        payload,
        config=FireRedTTS3Config(use_lm_head=True, dit_in_channels=1088),
        tokenizer=tokenizer,
    )
    assert f"{expected_instruction} /no_think" in tokenizer.decode(state.input_ids)


def test_speaker_special_token_must_exist_in_vocabulary() -> None:
    """An unknown token ID must not be accepted as a speaker special token."""
    backend = Tokenizer(models.WordLevel({"<unk>": 0, "hello": 1}, unk_token="<unk>"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    token = "<|text_holder|>"
    assert tokenizer.convert_tokens_to_ids(token) == tokenizer.unk_token_id
    with pytest.raises(ValueError, match="text_holder"):
        resolve_speaker_token_id(tokenizer, token)
    tokenizer.add_special_tokens({"additional_special_tokens": [token]})
    assert resolve_speaker_token_id(
        tokenizer, token
    ) == tokenizer.convert_tokens_to_ids(token)


SPEAKER_CONFIG = FireRedTTS3Config(
    text_latent_interleave_ratio=[2, 1],
    speaker_token="<|f531_speaker|>",
    dit_in_channels=1088,
    use_lm_head=False,
    redae_scale=0.4,
    generation_defaults={
        "n_timesteps": 10,
        "inference_cfg": 1.2,
        "stop_threshold": 0.5,
        "min_patches": 6,
        "max_patches": 600,
    },
)


def speaker_payload(
    *,
    inputs: dict[str, JsonValue] | None = None,
    params: dict[str, JsonValue] | None = None,
    tts_params: dict[str, JsonValue] | None = None,
) -> StagePayload:
    return make_preprocessing_payload(
        inputs=inputs or {"text": "晚上好，欢迎收听今天的节目。"},
        params=params,
        tts_params=tts_params,
    )


@pytest.mark.parametrize(
    "text,request_parameters,engine_token_budget",
    [
        ("你 好", {"dual_stream": False}, 620),
        ("你 好", {"dual_stream": True}, 1820),
        (
            "晚上好，欢迎收听今天的节目。",
            {"dual_stream": True, "max_new_tokens": 300},
            300,
        ),
    ],
)
def test_speaker_preprocessing_composes_prompt_and_budget(
    text: str, request_parameters: dict[str, JsonValue], engine_token_budget: int
) -> None:
    tokenizer = make_speaker_tokenizer()
    payload = speaker_payload(inputs={"text": text}, params=request_parameters)
    state = preprocess_payload(
        payload,
        tokenizer=tokenizer,
        config=SPEAKER_CONFIG,
        max_patches=600,
    )
    request = build_sglang_firered_tts3_request(
        payload,
        patch_size=4,
        vocab_size=151936,
        interleave_ratio=tuple(SPEAKER_CONFIG.text_latent_interleave_ratio),
    )
    assert request.max_new_tokens == engine_token_budget
    assert request.req.sampling_params.max_new_tokens == engine_token_budget
    dual_stream = request_parameters["dual_stream"]
    spoken_text = text.replace(" ", "")
    assert state.variant == "speaker"
    assert state.task_type == "tts"
    assert state.dual_stream is dual_stream
    assert state.inference_cfg == 1.2
    assert state.max_engine_tokens == engine_token_budget
    assert state.max_patches == 600
    if dual_stream:
        user_prompt = "Use a中性 tone. Align the interleaved text and speech chunks."
        assert state.text_stream_ids == [
            tokenizer.convert_tokens_to_ids(char) for char in spoken_text
        ]
        assert state.text_pad_id == tokenizer.convert_tokens_to_ids("<|text_pad|>")
    else:
        user_prompt = f"Convert text to speech.\n{spoken_text}"
        assert state.text_stream_ids == []
        assert state.text_pad_id == -1
    expected = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<|f531_speaker|><|audio_start|>"
    assert tokenizer.decode(state.input_ids) == expected
    assert tokenizer.convert_tokens_to_ids("<|text_holder|>") not in state.input_ids


def test_speaker_dual_stream_style_passthrough() -> None:
    tokenizer = make_speaker_tokenizer()
    styled = preprocess_payload(
        speaker_payload(params={"dual_stream": True, "style": "Use a 活泼 tone."}),
        tokenizer=tokenizer,
        config=SPEAKER_CONFIG,
        max_patches=600,
    )
    prompt = tokenizer.decode(styled.input_ids)
    assert "Use a活泼 tone. Align the interleaved text and speech chunks." in prompt
    assert "Use a中性 tone." not in prompt
    plain = preprocess_payload(
        speaker_payload(), tokenizer=tokenizer, config=SPEAKER_CONFIG, max_patches=600
    )
    styled_prefill = preprocess_payload(
        speaker_payload(params={"style": "Use a 活泼 tone."}),
        tokenizer=tokenizer,
        config=SPEAKER_CONFIG,
        max_patches=600,
    )
    assert styled_prefill.input_ids == plain.input_ids


def test_speaker_rejects_references_and_foreign_tasks() -> None:
    with pytest.raises(ValueError, match="text-only tts"):
        preprocess_payload(
            speaker_payload(
                inputs={
                    "text": "你好",
                    "references": [{"audio_path": "ref.wav", "text": "参考"}],
                }
            ),
            config=SPEAKER_CONFIG,
            max_patches=600,
        )
    with pytest.raises(ValueError, match="text-only tts"):
        preprocess_payload(
            speaker_payload(tts_params={"ref_text": "参考"}),
            config=SPEAKER_CONFIG,
            max_patches=600,
        )
    with pytest.raises(ValueError, match="task_type"):
        preprocess_payload(
            speaker_payload(tts_params={"task_type": "VoiceDesign"}),
            config=SPEAKER_CONFIG,
            max_patches=600,
        )
    with pytest.raises(ValueError, match="non-empty input text"):
        preprocess_payload(
            speaker_payload(inputs={"text": "  "}),
            config=SPEAKER_CONFIG,
            max_patches=600,
        )
