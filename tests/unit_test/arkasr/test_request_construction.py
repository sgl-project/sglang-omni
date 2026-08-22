# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.arkasr import request_builders
from sglang_omni.proto import OmniRequest, StagePayload

_AUDIO_TOKEN = "<|audio|>"
_AUDIO_TOKEN_ID = 151663
_PROMPT_PREFIX_IDS = [101, 102]
_PROMPT_SUFFIX_IDS = [103, 104]


class _FakeTokenizer:
    eos_token_id = 2
    vocab_size = 200000
    all_special_ids = [2]

    def __init__(self, template_audio_tokens: int | None = None) -> None:
        self.call_texts: list[str] = []
        self.template_audio_tokens = template_audio_tokens

    def get_added_vocab(self) -> dict[str, int]:
        return {_AUDIO_TOKEN: _AUDIO_TOKEN_ID}

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        assert not add_special_tokens
        self.call_texts.append(text)
        num_audio_tokens = (
            text.count(_AUDIO_TOKEN)
            if self.template_audio_tokens is None
            else self.template_audio_tokens
        )
        return SimpleNamespace(
            input_ids=[
                *_PROMPT_PREFIX_IDS,
                *([_AUDIO_TOKEN_ID] * num_audio_tokens),
                *_PROMPT_SUFFIX_IDS,
            ]
        )


class _FeatureExtractor:
    def __init__(self, mel_frames: int) -> None:
        self.mel_frames = mel_frames

    def __call__(self, *args, **kwargs):
        return SimpleNamespace(
            input_features=torch.zeros((1, 128, self.mel_frames)),
            attention_mask=torch.ones((1, self.mel_frames), dtype=torch.long),
        )


def _payload(request_id: str) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs={"audio_bytes": b"wav"}),
        data={},
    )


@pytest.mark.parametrize(
    ("mel_frames", "expected_audio_tokens"),
    [
        (8, 1),
        (400, 50),
        (3000, 375),
    ],
)
def test_arkasr_request_builder_reuses_tokenized_prompt_template(
    monkeypatch: pytest.MonkeyPatch,
    mel_frames: int,
    expected_audio_tokens: int,
) -> None:
    tokenizer = _FakeTokenizer()
    monkeypatch.setattr(
        request_builders,
        "prepare_audio",
        lambda *args, **kwargs: SimpleNamespace(
            waveform=np.zeros(16000, dtype=np.float32),
            duration_s=1.0,
            fingerprint="audio-fingerprint",
            fingerprint_int=17,
        ),
    )
    request_builder, _ = request_builders.make_arkasr_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=32,
        feature_extractor=_FeatureExtractor(mel_frames),
    )

    for request_id in ("first", "second"):
        data = request_builder(_payload(request_id))
        audio_item = data.req.multimodal_inputs.mm_items[0]
        expected_ids = [
            *_PROMPT_PREFIX_IDS,
            *([audio_item.pad_value] * expected_audio_tokens),
            *_PROMPT_SUFFIX_IDS,
        ]

        assert data.prompt_token_ids == expected_ids
        assert audio_item.offsets == [
            (
                len(_PROMPT_PREFIX_IDS),
                len(_PROMPT_PREFIX_IDS) + expected_audio_tokens - 1,
            )
        ]

    assert len(tokenizer.call_texts) == 1
    assert tokenizer.call_texts[0].count(_AUDIO_TOKEN) == 1


@pytest.mark.parametrize("template_audio_tokens", [0, 2])
def test_arkasr_request_builder_requires_one_audio_token_in_template(
    template_audio_tokens: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="ARK-ASR prompt template must contain exactly one audio token",
    ):
        request_builders.make_arkasr_scheduler_adapters(
            tokenizer=_FakeTokenizer(template_audio_tokens),
            max_new_tokens=32,
            feature_extractor=_FeatureExtractor(100),
        )
