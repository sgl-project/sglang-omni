# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

import sglang_omni.preprocessing.transcription as transcription
from sglang_omni.models.arkasr.request_builders import make_arkasr_scheduler_adapters
from sglang_omni.proto import OmniRequest, StagePayload

EOS = 999
AUDIO_TOKEN_ID = 151663
MEL_FRAMES = 40


class FakeTokenizer:
    eos_token_id = EOS
    vocab_size = 1000
    all_special_ids = [EOS]

    def get_added_vocab(self) -> dict[str, int]:
        return {"<tool_call>": 3}

    def __call__(self, prompt: str, add_special_tokens: bool = False):
        del add_special_tokens
        ids: list[int] = []
        rest = prompt
        while rest:
            if rest.startswith("<|audio|>"):
                ids.append(AUDIO_TOKEN_ID)
                rest = rest[len("<|audio|>") :]
            else:
                ids.append(100 + ord(rest[0]) % 50)
                rest = rest[1:]
        return SimpleNamespace(input_ids=ids)

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        return " ".join(str(token_id) for token_id in ids)


def feature_extractor(audio, **kwargs):
    del audio, kwargs
    return SimpleNamespace(
        input_features=torch.zeros((1, 128, MEL_FRAMES)),
        attention_mask=torch.ones((1, MEL_FRAMES), dtype=torch.long),
    )


def make_payload() -> StagePayload:
    return StagePayload(
        request_id="req-arkasr-request-limits",
        request=OmniRequest(inputs={"audio_bytes": b"wav"}, params={}),
        data={},
    )


def test_arkasr_request_builder_enforces_scheduler_request_limits(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(1600, dtype=np.float32),
    )
    request_builder, _ = make_arkasr_scheduler_adapters(
        tokenizer=FakeTokenizer(),
        max_new_tokens=16,
        feature_extractor=feature_extractor,
    )

    data = request_builder(make_payload())

    assert data.enforce_request_limits is True
