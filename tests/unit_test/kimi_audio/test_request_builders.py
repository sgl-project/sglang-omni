# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.kimi_audio.processor import KimiPrompt
from sglang_omni.models.kimi_audio.request_builders import (
    make_kimi_audio_scheduler_adapters,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.serve.openai_errors import is_bad_request_error


class _Tokenizer:
    vocab_size = 152064

    def decode(self, ids, **kwargs):
        del kwargs
        return "decoded:" + ",".join(str(item) for item in ids)


class _Processor:
    tokenizer = _Tokenizer()
    model_config = SimpleNamespace(vocab_size=168448, kimia_text_output_vocab=152064)
    special = SimpleNamespace(text_eos=151667)

    @staticmethod
    def _encode_text(text):
        assert text == "done"
        return [10]

    def build_prompt(self, messages, audios):
        assert messages == [{"role": "user", "content": "transcribe"}]
        assert audios == ["clip.wav"]
        return KimiPrompt(
            audio_ids=[151670, 152070, 151671],
            text_ids=[151666, 151666, 151666],
            continuous_mask=[False, True, False],
            continuous_features=[torch.ones((1, 5120), dtype=torch.bfloat16)],
        )


def _payload(modalities=None, **params) -> StagePayload:
    return StagePayload(
        request_id="req-1",
        request=OmniRequest(
            inputs={
                "messages": [{"role": "user", "content": "transcribe"}],
                "audios": ["clip.wav"],
            },
            params={"max_new_tokens": 7, "temperature": 0.0, **params},
            metadata={"output_modalities": modalities or ["text"]},
        ),
        data=None,
    )


def test_request_builder_preserves_audio_ids_and_parallel_text_metadata() -> None:
    request_builder, _ = make_kimi_audio_scheduler_adapters(
        processor=_Processor(), max_new_tokens=64, context_length=32
    )

    data = request_builder(_payload())

    assert data.input_ids.tolist() == [151670, 152070, 151671]
    assert data.req.origin_input_ids == [151670, 152070, 151671]
    assert data.req.vocab_size == 152064
    assert data.req.sampling_params.max_new_tokens == 7
    assert data.req.sampling_params.stop_token_ids == {151667}
    item = data.req.multimodal_inputs.mm_items[0]
    assert item.text_input_ids.tolist() == [151666, 151666, 151666]
    assert item.continuous_mask.tolist() == [False, True, False]


def test_request_builder_rejects_audio_output() -> None:
    request_builder, _ = make_kimi_audio_scheduler_adapters(
        processor=_Processor(), max_new_tokens=64, context_length=32
    )

    with pytest.raises(ValueError, match="only supported modality"):
        request_builder(_payload(["text", "audio"]))


def test_result_adapter_decodes_generated_text_ids() -> None:
    request_builder, result_adapter = make_kimi_audio_scheduler_adapters(
        processor=_Processor(), max_new_tokens=64, context_length=32
    )
    data = request_builder(_payload())
    data.output_ids = [100, 101]
    data.finish_reason = "length"

    result = result_adapter(data)

    assert result.data["text"] == "decoded:100,101"
    assert result.data["modality"] == "text"
    assert result.data["finish_reason"] == "length"
    assert result.data["usage"]["prompt_tokens"] == 3
    assert result.data["usage"]["completion_tokens"] == 2
    assert result.data["usage"]["total_tokens"] == 5


def test_request_builder_normalizes_openai_sampling_fields() -> None:
    request_builder, _ = make_kimi_audio_scheduler_adapters(
        processor=_Processor(), max_new_tokens=64, context_length=32
    )

    data = request_builder(
        _payload(stop=["done"], min_p=0.2, seed=123, stop_token_ids=[42])
    )

    sampling = data.req.sampling_params
    assert sampling.stop_strs == ["done"]
    assert sampling.stop_token_ids == {42, 151667}
    assert sampling.min_p == 0.2
    assert sampling.sampling_seed == 123
    assert data.req.tokenizer is _Processor.tokenizer


def test_request_builder_does_not_mask_invalid_zero_sampling_values() -> None:
    request_builder, _ = make_kimi_audio_scheduler_adapters(
        processor=_Processor(), max_new_tokens=64, context_length=32
    )

    with pytest.raises(ValueError, match="Invalid Kimi-Audio request") as exc_info:
        request_builder(_payload(top_p=0))

    assert "top_p must be in" in str(exc_info.value)
    assert is_bad_request_error(exc_info.value)
