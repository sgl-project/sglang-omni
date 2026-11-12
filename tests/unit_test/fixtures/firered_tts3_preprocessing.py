# SPDX-License-Identifier: Apache-2.0
"""Shared FireRedTTS3 preprocessing request builders for unit tests."""

from __future__ import annotations

from transformers import PreTrainedTokenizerFast

from sglang_omni.models.firered_tts3.hf_config import FireRedTTS3Config
from sglang_omni.models.firered_tts3.payload_types import FireRedTTS3State
from sglang_omni.models.firered_tts3.preprocessing import (
    preprocess_firered_tts3_payload,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.utils.json import JsonValue
from tests.unit_test.fixtures.firered_tts3 import make_speaker_tokenizer


def make_preprocessing_payload(
    *,
    inputs: dict[str, JsonValue] | None = None,
    params: dict[str, JsonValue] | None = None,
    tts_params: dict[str, JsonValue] | None = None,
) -> StagePayload:
    return StagePayload(
        request_id="rid",
        request=OmniRequest(
            inputs=inputs
            or {
                "text": "欢迎使用点点语音合成系统。",
                "references": [{"audio_path": "ref.wav", "text": "今天天气很好。"}],
            },
            params=params or {},
            metadata={"tts_params": tts_params or {}},
        ),
        data={},
    )


def preprocess_payload(
    payload: StagePayload,
    *,
    config: FireRedTTS3Config | None = None,
    tokenizer: PreTrainedTokenizerFast | None = None,
    max_patches: int = 400,
) -> FireRedTTS3State:
    preprocess_firered_tts3_payload(
        payload,
        tokenizer=tokenizer or make_speaker_tokenizer(),
        config=config or FireRedTTS3Config(spk_in_dim=512, dit_in_channels=1600),
        n_timesteps=10,
        max_patches=max_patches,
    )
    return FireRedTTS3State.from_dict(payload.data)
