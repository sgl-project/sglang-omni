# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from sglang.srt.managers.schedule_batch import Modality

from sglang_omni.models.mimo_v25_asr.prompt import CHINESE_TAG, EMPTY
from sglang_omni.models.mimo_v25_asr.request_builders import (
    make_mimo_v25_asr_scheduler_adapters,
    validate_text_only_request,
)
from sglang_omni.proto import OmniRequest, StagePayload


class _Encoding:
    def __init__(self, ids: list[int]):
        self.ids = ids


class _FakeTokenizer:
    eos_token_id = 151645
    vocab_size = 151680
    special = {
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
        "<|sosp|>": 151665,
        "<|eosp|>": 151666,
        EMPTY: 151667,
        "<think>": 151668,
        "</think>": 151669,
        CHINESE_TAG: 151670,
        "<english>": 151671,
    }

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self.special.get(token)

    def encode(self, text: str, add_special_tokens: bool = False) -> _Encoding:
        del add_special_tokens
        ids: list[int] = []
        cursor = 0
        tokens = sorted(self.special, key=len, reverse=True)
        while cursor < len(text):
            special = next(
                (token for token in tokens if text.startswith(token, cursor)), None
            )
            if special is not None:
                ids.append(self.special[special])
                cursor += len(special)
            else:
                ids.append(1000 + ord(text[cursor]))
                cursor += 1
        return _Encoding(ids)

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        del skip_special_tokens, kwargs
        return f"{CHINESE_TAG}hello<|empty|>"


def _payload(*, language: str = "zh", modalities=None) -> StagePayload:
    request = OmniRequest(
        inputs={"audio_path": "dummy.wav"},
        params={"language": language, "temperature": 0.0, "max_new_tokens": 32},
        metadata={"output_modalities": modalities or ["text"]},
    )
    return StagePayload(
        request_id="req-1",
        request=request,
        data={
            "audio_codes": torch.zeros((4, 8), dtype=torch.int64),
            "audio_duration_s": 1.5,
            "audio_fingerprint": "abcd" * 8,
        },
    )


def test_validate_text_only_rejects_audio_modality() -> None:
    with pytest.raises(ValueError, match="text transcription only"):
        validate_text_only_request(_payload(modalities=["text", "audio"]))


def test_request_builder_injects_audio_placeholders() -> None:
    builder, adapter = make_mimo_v25_asr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=64,
    )
    data = builder(_payload(language="zh"))
    assert data.req.multimodal_inputs is not None
    item = data.req.multimodal_inputs.mm_items[0]
    assert item.modality == Modality.AUDIO
    assert data.language == "zh"
    assert data.audio_duration_s == 1.5
    assert data.req.sampling_params.stop_token_ids

    data.output_ids = [1, 2, 3]
    data.engine_start_s = 0.0
    result = adapter(data)
    assert result.data["text"] == "hello"
