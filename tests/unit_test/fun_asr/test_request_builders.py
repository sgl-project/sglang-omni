# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import sglang_omni.models.fun_asr.request_builders as request_builders
from sglang_omni.models.fun_asr.tool_funcs.audio_lengths import (
    fun_asr_low_frame_rate_length,
)
from sglang_omni.proto import OmniRequest, StagePayload

_AUDIO_PAD = "<|object_ref_start|>"
_AUDIO_PAD_ID = 42  # arbitrary sentinel distinct from vocabulary ids below


class _FakeTokenizer:
    eos_token_id = 151645
    vocab_size = 151936

    def __init__(self) -> None:
        self.decode_calls: list[dict] = []

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == _AUDIO_PAD
        return _AUDIO_PAD_ID

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        assert not add_special_tokens
        # Mirror the real ChatML prompt shape: a fixed head/tail with N audio
        # placeholders in the middle. The request builder only inspects the
        # placeholder span, so the surrounding text need not be real tokens.
        audio_pad_count = text.count(_AUDIO_PAD)
        # system(3) + user-open(2) + [pad]*N + user-close/assistant(4)
        input_ids = (
            [10, 11, 12, 13, 14] + [_AUDIO_PAD_ID] * audio_pad_count + [15, 16, 17, 18]
        )
        return SimpleNamespace(input_ids=input_ids)

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        self.decode_calls.append(
            {
                "token_ids": list(token_ids),
                "skip_special_tokens": skip_special_tokens,
                "clean_up_tokenization_spaces": clean_up_tokenization_spaces,
            }
        )
        pieces = {20: "你好", 21: "世界", 30: "<|im_end|>"}
        text = "".join(pieces.get(t, "") for t in token_ids)
        if skip_special_tokens:
            text = text.replace("<|im_end|>", "")
        return text


def _feature_extractor(num_lfr_frames: int):
    """Stand-in for FunAsrNanoFeatureExtractor: returns [1, 560, T_lfr]."""

    def _call(
        audio,
        sampling_rate=None,
        return_tensors=None,
        return_attention_mask=True,
        padding="longest",
    ):
        return {
            "input_features": torch.zeros((1, 560, num_lfr_frames)),
            "attention_mask": torch.ones((1, num_lfr_frames), dtype=torch.long),
        }

    return _call


def test_fun_asr_request_builder_records_inclusive_audio_offsets(monkeypatch) -> None:
    # 17 LFR frames -> three ceil(x/2) reductions: 17->9->5->3 audio tokens
    num_lfr_frames = 17
    num_audio_tokens = fun_asr_low_frame_rate_length(num_lfr_frames)
    assert num_audio_tokens == 3

    monkeypatch.setattr(
        request_builders,
        "_load_audio",
        lambda source: np.zeros(1600 * 3, dtype=np.float32),
    )
    request_builder, _ = request_builders.make_fun_asr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=_feature_extractor(num_lfr_frames),
    )
    payload = StagePayload(
        request_id="req-fun-asr",
        request=OmniRequest(
            inputs={"audio_bytes": b"wav"}, params={"max_new_tokens": 32}
        ),
        data={},
    )

    data = request_builder(payload)

    audio_item = data.req.multimodal_inputs.mm_items[0]
    start, end = audio_item.offsets[0]
    assert audio_item.feature_attention_mask.shape == (1, num_lfr_frames)
    assert end - start + 1 == num_audio_tokens
    assert (
        data.prompt_token_ids[start : end + 1]
        == [audio_item.pad_value] * num_audio_tokens
    )
    # pad_value replaces the placeholder span (general_mm_embed_routine matches it
    # by pad_value, not by the original <|object_ref_start|> token id)
    assert audio_item.pad_value != _AUDIO_PAD_ID
    # greedy by default (Fun-ASR reference uses no sampling args): temperature=0.0
    # is normalized by sglang to top_k=1. The original intent is on FunASRRequestData.
    assert data.temperature == 0.0
    assert data.req.sampling_params.top_k == 1
    assert data.req.sampling_params.max_new_tokens == 32
    # mrope positions broadcast as [3, seq] degenerate (plain 1-D positions)
    seq_len = len(data.prompt_token_ids)
    assert data.req.multimodal_inputs.mrope_positions.shape == (3, seq_len)
    assert torch.equal(
        data.req.multimodal_inputs.mrope_positions[0],
        torch.arange(seq_len, dtype=torch.long),
    )


def test_fun_asr_request_builder_language_prompt(monkeypatch) -> None:

    monkeypatch.setattr(
        request_builders,
        "_load_audio",
        lambda source: np.zeros(1600, dtype=np.float32),
    )
    captured = {}

    class _CapturingTokenizer(_FakeTokenizer):
        def __call__(self, text: str, *, add_special_tokens: bool = False):
            captured["prompt_text"] = text
            return super().__call__(text, add_special_tokens=add_special_tokens)

    request_builder, _ = request_builders.make_fun_asr_scheduler_adapters(
        tokenizer=_CapturingTokenizer(),
        max_new_tokens=16,
        feature_extractor=_feature_extractor(11),
    )
    payload = StagePayload(
        request_id="req-fun-asr-en",
        request=OmniRequest(inputs={"audio_path": "x.wav"}, params={"language": "en"}),
        data={},
    )

    data = request_builder(payload)
    assert "语音转写成英文" in captured["prompt_text"]
    assert data.language == "en"


def test_fun_asr_result_adapter_decodes_transcript_directly() -> None:
    tokenizer = _FakeTokenizer()
    _, result_adapter = request_builders.make_fun_asr_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=32,
        feature_extractor=object(),
    )
    payload = StagePayload(
        request_id="req-fun-asr",
        request=OmniRequest(inputs={}),
        data={},
    )
    data = request_builders.FunASRRequestData(
        output_ids=[20, 21, 30],  # 你好 世界 <|im_end|>
        stage_payload=payload,
        language="zh",
        audio_duration_s=2.5,
    )

    result = result_adapter(data)

    # Fun-ASR emits the transcript directly after <|im_start|>assistant\n — no
    # forced prefix marker to strip. skip_special_tokens=True drops <|im_end|>.
    assert result.data["text"] == "你好世界"
    assert result.data["language"] == "zh"
    assert result.data["duration_s"] == 2.5
    assert result.data["modality"] == "text"
    assert tokenizer.decode_calls[-1] == {
        "token_ids": [20, 21, 30],
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }


def test_fun_asr_load_audio_uses_shared_audio_utility(monkeypatch) -> None:
    calls = []
    expected = np.zeros(1600, dtype=np.float32)

    def _shared(source, **kwargs):
        calls.append((source, kwargs))
        return expected

    monkeypatch.setattr(request_builders, "_shared_load_audio", _shared)

    actual = request_builders._load_audio(b"wav")

    assert actual is expected
    assert calls == [
        (
            b"wav",
            {"source_name": "Fun-ASR", "target_sample_rate": 16000},
        )
    ]


def test_fun_asr_request_builder_scales_default_token_budget(monkeypatch) -> None:
    monkeypatch.setattr(
        request_builders,
        "_load_audio",
        lambda source: np.zeros(16000 * 3, dtype=np.float32),
    )
    request_builder, _ = request_builders.make_fun_asr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=200,
        feature_extractor=_feature_extractor(17),
    )
    payload = StagePayload(
        request_id="req-fun-asr-budget",
        request=OmniRequest(inputs={"audio_bytes": b"wav"}),
        data={},
    )

    data = request_builder(payload)

    assert data.req.sampling_params.max_new_tokens == 20
    assert data.max_new_tokens == 20


def test_fun_asr_request_builder_rejects_audio_over_vad_limit(monkeypatch) -> None:
    monkeypatch.setattr(
        request_builders,
        "_load_audio",
        lambda source: np.zeros(16000 * 30 + 1, dtype=np.float32),
    )
    request_builder, _ = request_builders.make_fun_asr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=200,
        feature_extractor=_feature_extractor(17),
    )
    payload = StagePayload(
        request_id="req-fun-asr-too-long",
        request=OmniRequest(inputs={"audio_bytes": b"wav"}),
        data={},
    )

    with pytest.raises(ValueError, match=r"30(?:\.0)? seconds.*VAD"):
        request_builder(payload)


def test_fun_asr_request_builder_rejects_explicit_token_budget_over_cap(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        request_builders,
        "_load_audio",
        lambda source: np.zeros(16000, dtype=np.float32),
    )
    request_builder, _ = request_builders.make_fun_asr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=200,
        feature_extractor=_feature_extractor(17),
    )
    payload = StagePayload(
        request_id="req-fun-asr-token-cap",
        request=OmniRequest(
            inputs={"audio_bytes": b"wav"},
            params={"max_new_tokens": 201},
        ),
        data={},
    )

    with pytest.raises(ValueError, match=r"max_new_tokens.*200"):
        request_builder(payload)


def test_fun_asr_request_builder_wraps_string_hotword_as_single_entry(
    monkeypatch,
) -> None:
    # A bare string must stay one hotword, not be split into characters.
    monkeypatch.setattr(
        request_builders,
        "_load_audio",
        lambda source: np.zeros(1600, dtype=np.float32),
    )
    captured = {}

    class _CapturingTokenizer(_FakeTokenizer):
        def __call__(self, text: str, *, add_special_tokens: bool = False):
            captured["prompt_text"] = text
            return super().__call__(text, add_special_tokens=add_special_tokens)

    request_builder, _ = request_builders.make_fun_asr_scheduler_adapters(
        tokenizer=_CapturingTokenizer(),
        max_new_tokens=16,
        feature_extractor=_feature_extractor(11),
    )
    payload = StagePayload(
        request_id="req-fun-asr-hotwords-str",
        request=OmniRequest(
            inputs={"audio_path": "x.wav"},
            params={"hotwords": "人工智能"},
        ),
        data={},
    )

    request_builder(payload)
    assert "热词列表：[人工智能]" in captured["prompt_text"]


def test_fun_asr_request_builder_rejects_prompt_overrun_of_context_length(
    monkeypatch,
) -> None:
    # 17 LFR frames -> 3 audio tokens; FakeTokenizer yields 9 + 3 = 12 input ids.
    # With context_length=10 and max_new_tokens=5, 12 + 5 > 10 must raise a
    # clear bad-request error (caught by _BAD_REQUEST_MARKERS -> HTTP 400).
    monkeypatch.setattr(
        request_builders,
        "_load_audio",
        lambda source: np.zeros(16000, dtype=np.float32),
    )
    request_builder, _ = request_builders.make_fun_asr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=200,
        feature_extractor=_feature_extractor(17),
        context_length=10,
    )
    payload = StagePayload(
        request_id="req-fun-asr-overflow",
        request=OmniRequest(
            inputs={"audio_bytes": b"wav"},
            params={"max_new_tokens": 5},
        ),
        data={},
    )

    with pytest.raises(ValueError, match=r"longer than the model's context length"):
        request_builder(payload)
