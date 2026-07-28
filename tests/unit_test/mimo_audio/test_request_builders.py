# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import ClassVar

import torch

from sglang_omni.models.mimo_audio.request_builders import (
    audio_source_from_payload,
    make_mimo_scheduler_adapters,
    validate_text_only_request,
)
from sglang_omni.proto import OmniRequest, StagePayload


class _Encoding:
    def __init__(self, ids: list[int]):
        self.ids = ids


class _Tokenizer:
    eos_token_id = 151643
    vocab_size = 151680
    special: ClassVar[dict[str, int]] = {
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
        "<|sosp|>": 151665,
        "<|eosp|>": 151666,
        "<|empty|>": 151667,
    }

    def __len__(self):
        return self.vocab_size

    def convert_tokens_to_ids(self, token: str):
        return self.special.get(token, 0)

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        ids = []
        for token, token_id in self.special.items():
            text = text.replace(token, chr(token_id))
        for char in text:
            ids.append(ord(char))
        return _Encoding(ids)

    def decode(self, ids, **kwargs):
        del kwargs
        return "transcript<|im_end|>"


def _payload(modalities=None) -> StagePayload:
    return StagePayload(
        request_id="mimo-test",
        request=OmniRequest(
            inputs={"audio_bytes": b"RIFF"},
            params={"prompt": "Summarize the audio."},
            metadata={"output_modalities": modalities or ["text"]},
        ),
        data={
            "audio_codes": torch.zeros((8, 8), dtype=torch.long),
            "audio_duration_s": 1.0,
            "audio_fingerprint": "0" * 64,
        },
    )


def test_request_builder_routes_codes_as_audio_embeddings_and_suppresses_audio() -> (
    None
):
    tokenizer = _Tokenizer()
    build, _ = make_mimo_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=64,
    )

    data = build(_payload())

    assert data.input_ids.ndim == 1
    assert len(data.req.multimodal_inputs.mm_items) == 1
    item = data.req.multimodal_inputs.mm_items[0]
    assert item.feature.shape == (8, 8)
    assert item.offsets[0][1] - item.offsets[0][0] + 1 == 2
    assert data.req.sampling_params.logit_bias[str(151667)] < -1.0e8
    assert set(data.req.sampling_params.stop_token_ids) == {151643, 151645}


def test_audio_output_mode_is_rejected() -> None:
    payload = _payload(["text", "audio"])
    try:
        validate_text_only_request(payload)
    except ValueError as exc:
        assert "text output only" in str(exc)
    else:
        raise AssertionError("MiMo audio output request was accepted")


def test_missing_preprocessed_audio_is_rejected() -> None:
    build, _ = make_mimo_scheduler_adapters(
        tokenizer=_Tokenizer(),
        max_new_tokens=64,
    )
    payload = _payload()
    payload.data = None

    try:
        build(payload)
    except ValueError as exc:
        assert "preprocessed audio codes" in str(exc)
    else:
        raise AssertionError("missing MiMo codes were accepted")


def test_missing_raw_audio_input_is_rejected() -> None:
    payload = _payload()
    payload.request.inputs = None

    try:
        audio_source_from_payload(payload)
    except ValueError as exc:
        assert "missing audio input" in str(exc)
    else:
        raise AssertionError("missing MiMo raw audio input was accepted")


def test_result_adapter_strips_mimo_markers() -> None:
    build, adapt = make_mimo_scheduler_adapters(
        tokenizer=_Tokenizer(),
        max_new_tokens=64,
    )
    data = build(_payload())
    data.output_ids = [1, 2, 3]

    result = adapt(data)

    assert result.data["text"] == "transcript"
    assert result.data["modality"] == "text"
