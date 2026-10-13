# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch

from sglang_omni.models.firered_asr2 import request_builders
from sglang_omni.preprocessing.transcription import PreparedAudio
from sglang_omni.proto.request import OmniRequest, StagePayload


class Tokenizer:
    unk_token_id = None

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return [1, 2] if text.endswith("user\n") else [3, 4, 5]

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<speech>": 99, "<|im_end|>": 98}[token]

    def __len__(self) -> int:
        return 100


@pytest.fixture
def build_request(
    monkeypatch: pytest.MonkeyPatch,
) -> request_builders.FireRedRequestBuilder:
    audio = PreparedAudio(np.zeros(16000), 16000, 1.0, "0123456789abcdef")
    monkeypatch.setattr(request_builders, "prepare_audio", lambda *a, **kw: audio)
    builder, _ = request_builders.make_firered_asr2_scheduler_adapters(
        tokenizer=Tokenizer(),
        feature_extractor=lambda wave: torch.zeros(21, 80),
        max_new_tokens=256,
        downsample_rate=2,
    )
    return builder


def payload(**params: int) -> StagePayload:
    return StagePayload(
        request_id="asr-test", request=OmniRequest(inputs={}, params=params), data={}
    )


def test_native_multimodal_request_contract(
    build_request: request_builders.FireRedRequestBuilder,
) -> None:
    data = build_request(payload())
    req = data.req
    item = req.multimodal_inputs.mm_items[0]
    assert req.extra_key == "0123456789abcdef"
    assert item.offsets == [(2, 4)]
    assert item.num_audio_tokens == 3
    assert req.origin_input_ids == [1, 2] + [item.pad_value] * 3 + [3, 4, 5]
    assert data.input_ids.tolist() == req.origin_input_ids
    assert req.sampling_params.top_k == 1
    assert req.sampling_params.stop_token_ids == {98}
    assert data.max_new_tokens == 256


@pytest.mark.parametrize("limit", [0, -1])
def test_nonpositive_generation_limit_rejected(
    build_request: request_builders.FireRedRequestBuilder, limit: int
) -> None:
    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        build_request(payload(max_new_tokens=limit))
