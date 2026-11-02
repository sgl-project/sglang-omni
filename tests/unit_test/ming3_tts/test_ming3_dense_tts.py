# SPDX-License-Identifier: Apache-2.0
"""Dense Ming-3.0-tts-0.5B routing and prompt layout."""

import json
from pathlib import Path

from sglang_omni.config.manager import resolve_config_cls_for_model_path
from sglang_omni.models.ming3_tts.config import (
    DENSE_ARCHITECTURE,
    Ming3TTSPipelineConfig,
)
from sglang_omni.models.ming3_tts.dense_prompt import (
    AUDIO_PATCH,
    AUDIO_START,
    DENSE_SPEAKER_PAD,
    DENSE_SPEAKER_START,
    DENSE_TOKEN_IDS,
    build_dense_speech_prompt,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.utils.hf import try_resolve_ming3_dense_tts

TINY_ARCHITECTURE = "LingV3TTSForConditionalGeneration"
MING_OMNI_ARCHITECTURE = "BailingMMNativeForConditionalGeneration"


def write_config(directory: Path, payload: dict[str, object]) -> str:
    (directory / "config.json").write_text(json.dumps(payload))
    return str(directory)


class TokenTokenizer:
    def convert_tokens_to_ids(self, token: str) -> int:
        return DENSE_TOKEN_IDS[token]

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        token_ids: list[int] = []
        cursor = 0
        names = sorted(DENSE_TOKEN_IDS, key=len, reverse=True)
        while cursor < len(text):
            matched = next(
                (name for name in names if text.startswith(name, cursor)), None
            )
            if matched is None:
                token_ids.append(ord(text[cursor]))
                cursor += 1
            else:
                token_ids.append(DENSE_TOKEN_IDS[matched])
                cursor += len(matched)
        return token_ids


def test_tiny_architecture_still_resolves_to_the_ming3_pipeline(tmp_path: Path) -> None:
    checkpoint = write_config(
        tmp_path,
        {"architectures": [TINY_ARCHITECTURE], "model_type": "lingv3_tts"},
    )
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config(TINY_ARCHITECTURE) is Ming3TTSPipelineConfig
    )
    assert resolve_config_cls_for_model_path(checkpoint) is Ming3TTSPipelineConfig


def test_dense_checkpoint_resolves_to_the_same_pipeline(tmp_path: Path) -> None:
    checkpoint = write_config(
        tmp_path,
        {
            "architectures": [MING_OMNI_ARCHITECTURE],
            "model_type": "dense",
            "ditar_config": {},
            "audio_tokenizer_config": {},
        },
    )
    assert try_resolve_ming3_dense_tts(checkpoint) == DENSE_ARCHITECTURE
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config(DENSE_ARCHITECTURE)
        is Ming3TTSPipelineConfig
    )
    assert resolve_config_cls_for_model_path(checkpoint) is Ming3TTSPipelineConfig


def test_ming_omni_layout_is_not_treated_as_dense_tts(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": [MING_OMNI_ARCHITECTURE],
                "model_type": "bailingmm",
            }
        )
    )
    assert try_resolve_ming3_dense_tts(str(tmp_path)) is None


def test_dense_prompt_without_a_speaker_ends_at_audio_start() -> None:
    tokenizer = TokenTokenizer()
    input_ids, speaker_position, latent_position = build_dense_speech_prompt(
        tokenizer,
        "hi",
        prompt_text=None,
        prompt_latent_steps=0,
        include_speaker=False,
    )
    assert speaker_position == -1
    assert DENSE_TOKEN_IDS[DENSE_SPEAKER_PAD] not in input_ids
    assert input_ids[-1] == DENSE_TOKEN_IDS[AUDIO_START]
    assert latent_position == -1
    assert input_ids.count(DENSE_TOKEN_IDS[AUDIO_PATCH]) == 0


def test_dense_prompt_with_a_speaker_keeps_one_pad_slot() -> None:
    tokenizer = TokenTokenizer()
    input_ids, speaker_position, _latent_position = build_dense_speech_prompt(
        tokenizer,
        "hi",
        prompt_text=None,
        prompt_latent_steps=0,
        include_speaker=True,
    )
    assert input_ids[speaker_position - 1 : speaker_position + 2] == [
        DENSE_TOKEN_IDS[DENSE_SPEAKER_START],
        DENSE_TOKEN_IDS[DENSE_SPEAKER_PAD],
        DENSE_TOKEN_IDS["<|vision_end|>"],
    ]


def test_dense_prompt_appends_one_patch_token_per_reference_latent() -> None:
    tokenizer = TokenTokenizer()
    input_ids, _speaker_position, latent_position = build_dense_speech_prompt(
        tokenizer,
        "hi",
        prompt_text="cars",
        prompt_latent_steps=3,
        include_speaker=True,
    )
    assert latent_position == len(input_ids) - 3
    assert input_ids[latent_position:] == [DENSE_TOKEN_IDS[AUDIO_PATCH]] * 3
