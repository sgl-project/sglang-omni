# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.dots_tts.payload_types import (
    DotsTTSState,
    materialize_streaming_schedule,
)
from sglang_omni.models.dots_tts.stages import preprocess_dots_tts_payload
from sglang_omni.proto import OmniRequest, StagePayload


class _RecordingTokenizer:
    eos_token_id = 0

    def __init__(self) -> None:
        from dots_tts.utils.tokenizer import (
            AUDIO_COMP_SPAN_TOKEN,
            AUDIO_GEN_END_TOKEN,
            AUDIO_GEN_SPAN_TOKEN,
            AUDIO_GEN_START_TOKEN,
            TEXT_COND_END_TOKEN,
        )

        self.encoded_text: list[str] = []
        self._tokens = {
            AUDIO_GEN_START_TOKEN: 101,
            AUDIO_GEN_SPAN_TOKEN: 102,
            AUDIO_COMP_SPAN_TOKEN: 103,
            AUDIO_GEN_END_TOKEN: 104,
            TEXT_COND_END_TOKEN: 105,
        }

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        self.encoded_text.append(text)
        return [10] if text else []

    def decode(self, token_ids: list[int], **_kwargs) -> str:
        return " ".join(str(token_id) for token_id in token_ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._tokens[token]

    def __len__(self) -> int:
        return 256


def _payload(
    *,
    tts_params: dict | None = None,
    params: dict | None = None,
    references: list[dict] | None = None,
) -> StagePayload:
    return StagePayload(
        request_id="rid",
        request=OmniRequest(
            inputs={
                "input": "hello",
                "references": references
                or [
                    {
                        "audio_path": "data:audio/wav;base64,UklGRg==",
                        "text": "reference",
                    }
                ],
            },
            params=params or {},
            metadata={"tts_params": tts_params or {}},
        ),
        data={},
    )


def _preprocess(
    payload: StagePayload,
    tokenizer: _RecordingTokenizer,
    *,
    sampling=None,
    streaming=None,
) -> DotsTTSState:
    result = preprocess_dots_tts_payload(
        payload,
        tokenizer=tokenizer,
        model_config=SimpleNamespace(
            patch_size=4,
            vocoder=SimpleNamespace(sample_rate=48000),
            sampling=sampling,
            streaming=streaming,
        ),
        max_generate_length=20,
        max_sequence_length=128,
    )
    return DotsTTSState.from_dict(result.data)


def test_public_base_auto_and_generation_budget_reach_native_state(monkeypatch) -> None:
    monkeypatch.setattr("dots_tts.utils.text.detect", lambda _text: "en")
    tokenizer = _RecordingTokenizer()

    state = _preprocess(
        _payload(
            tts_params={"task_type": "Base", "language": "Auto"},
            params={"max_new_tokens": 3},
        ),
        tokenizer,
    )

    assert state.max_new_tokens == 3
    assert state.prompt_audio_path == "data:audio/wav;base64,UklGRg=="
    assert state.use_prompt_prefill is True
    assert any("[EN]reference" in text for text in tokenizer.encoded_text)


def test_preprocessing_rejects_unconsumed_extra_references() -> None:
    payload = _payload(
        references=[
            {"audio_path": "first.wav", "text": "first"},
            {"audio_path": "second.wav", "text": "second"},
        ]
    )

    with pytest.raises(ValueError, match="at most one reference"):
        _preprocess(payload, _RecordingTokenizer())


def test_preprocessing_uses_artifact_sampling_contract() -> None:
    from dots_tts.models.dots_tts.config import SamplingConfig

    sampling = SamplingConfig(solver="scm")
    state = _preprocess(_payload(), _RecordingTokenizer(), sampling=sampling)

    assert (state.ode_method, state.num_steps, state.guidance_scale) == (
        "euler",
        2,
        0.0,
    )

    with pytest.raises(ValueError, match="scm artifact requires"):
        _preprocess(
            _payload(params={"stage_params": {"latent_engine": {"num_steps": 1}}}),
            _RecordingTokenizer(),
            sampling=sampling,
        )


def test_stts_schedule_uses_artifact_cadence_after_reference_encode() -> None:
    from dots_tts.models.dots_tts.config import SamplingConfig, StreamingConfig

    tokenizer = _RecordingTokenizer()
    state = _preprocess(
        _payload(),
        tokenizer,
        sampling=SamplingConfig(solver="scm"),
        streaming=StreamingConfig(
            interleave_mode="buffered_ratio",
            initial_lookahead=3,
            ta_per_tta=1,
            warmup_ta=0,
        ),
    )

    assert state.interleaved
    assert state.generation_schedule.numel() == 0
    state.prompt_latents = torch.zeros(1, 8, 6)
    materialize_streaming_schedule(state)

    schedule = state.generation_schedule[0].tolist()
    audio_id = state.streaming_schedule["audio_span_id"]
    assert schedule.count(audio_id) == 20
    assert state.streaming_schedule["interleave_mode"] == "buffered_ratio"
