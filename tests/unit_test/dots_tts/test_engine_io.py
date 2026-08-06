# SPDX-License-Identifier: Apache-2.0
"""Schedule-splitting and adapter contracts for the dots.tts SGLang stage."""

from __future__ import annotations

import types

import pytest
import torch

from sglang_omni.models.dots_tts.engine_io import (
    MIN_DECODE_SPANS,
    DotsTtsSpecialTokens,
    make_dots_tts_scheduler_adapters,
    split_prefill_schedule,
)
from sglang_omni.models.dots_tts.payload_types import DotsTtsState, store_dots_tts_state
from sglang_omni.proto import OmniRequest, StagePayload

AUDIO_SPAN = 90002
AUDIO_END = 90003
LATENT_DIM = 8
LATENT_PATCH_SIZE = 4
NUM_STEPS = 4
MAX_AUDIO_PATCHES = 16


def _schedule(*, text_tokens: int, spans: int) -> list[int]:
    return list(range(1, text_tokens + 1)) + [AUDIO_SPAN] * spans


def test_split_prefill_schedule_stops_after_the_prompt_spans() -> None:
    prefill, span_start, max_new_tokens = split_prefill_schedule(
        _schedule(text_tokens=5, spans=10),
        audio_span_token_id=AUDIO_SPAN,
        prompt_patch_count=3,
    )

    assert span_start == 5
    assert prefill == [1, 2, 3, 4, 5, AUDIO_SPAN, AUDIO_SPAN, AUDIO_SPAN]
    assert max_new_tokens == 7


@pytest.mark.parametrize(
    ("schedule", "prompt_patch_count", "message"),
    [
        ([1, 2, 3], 1, "no audio spans"),
        ([AUDIO_SPAN, AUDIO_SPAN], 1, "must start with text"),
        (_schedule(text_tokens=2, spans=4), 0, "inside the audio span budget"),
        (_schedule(text_tokens=2, spans=4), 4, "inside the audio span budget"),
        (_schedule(text_tokens=2, spans=4), 3, "at least"),
    ],
)
def test_split_prefill_schedule_rejects_unusable_schedules(
    schedule: list[int], prompt_patch_count: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        split_prefill_schedule(
            schedule,
            audio_span_token_id=AUDIO_SPAN,
            prompt_patch_count=prompt_patch_count,
        )


def test_minimum_decode_spans_leaves_one_payload_patch() -> None:
    # The first decode span regenerates the dropped prompt tail, so the floor
    # has to leave at least one patch behind.
    assert MIN_DECODE_SPANS == 2


def _model() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        llm_config=types.SimpleNamespace(vocab_size=151672),
        latent_patch_size=LATENT_PATCH_SIZE,
        latent_dim=LATENT_DIM,
    )


def _adapters(reset_calls: list[str]):
    return make_dots_tts_scheduler_adapters(
        model=_model(),
        special_tokens=DotsTtsSpecialTokens(
            audio_gen_span=AUDIO_SPAN, audio_gen_end=AUDIO_END
        ),
        reset_request=reset_calls.append,
        num_steps=NUM_STEPS,
        max_audio_patches=MAX_AUDIO_PATCHES,
    )


def _payload(**overrides) -> StagePayload:
    state = DotsTtsState(
        text="hello",
        prompt_text="reference",
        num_steps=NUM_STEPS,
        max_audio_patches=MAX_AUDIO_PATCHES,
        generation_schedule_ids=_schedule(text_tokens=4, spans=MAX_AUDIO_PATCHES),
        prompt_patch_count=3,
        prompt_latent=torch.zeros((1, 3 * LATENT_PATCH_SIZE, LATENT_DIM)),
        spk_emb=torch.zeros((1, 512)),
    )
    for name, value in overrides.items():
        setattr(state, name, value)
    return store_dots_tts_state(
        StagePayload(request_id="req", request=OmniRequest(inputs={}), data={}), state
    )


def test_request_builder_lowers_the_schedule_onto_one_sglang_request() -> None:
    request_builder, _ = _adapters([])

    data = request_builder(_payload())

    assert data.prompt_span_start == 4
    assert data.prompt_patch_count == 3
    assert data.max_new_tokens == MAX_AUDIO_PATCHES - 3
    assert data.audio_span_token_id == AUDIO_SPAN
    assert data.audio_eos_token_id == AUDIO_END
    assert data.input_embeds_are_projected is True
    assert data.input_ids.tolist() == [1, 2, 3, 4] + [AUDIO_SPAN] * 3
    assert data.req.eos_token_ids == {AUDIO_END}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_steps": NUM_STEPS + 1}, "num_steps is fixed"),
        ({"max_audio_patches": MAX_AUDIO_PATCHES + 1}, "exceeds the tts_engine"),
        ({"generation_schedule_ids": None}, "requires a generation schedule"),
    ],
)
def test_request_builder_rejects_requests_the_engine_cannot_serve(
    overrides: dict, message: str
) -> None:
    request_builder, _ = _adapters([])

    with pytest.raises(ValueError, match=message):
        request_builder(_payload(**overrides))


def test_result_adapter_reshapes_latents_and_always_releases_the_request() -> None:
    reset_calls: list[str] = []
    request_builder, result_adapter = _adapters(reset_calls)
    data = request_builder(_payload())
    data.generated_latents = torch.zeros((5, LATENT_PATCH_SIZE, LATENT_DIM))
    data.finish_reason = "stop"

    payload = result_adapter(data)
    state = DotsTtsState.from_dict(payload.data)

    assert state.generated_latents.shape == (1, 5 * LATENT_PATCH_SIZE, LATENT_DIM)
    assert state.completion_tokens == 5
    assert state.finish_reason == "stop"
    # Conditioning is dropped so it never crosses into audio_decode.
    assert state.prompt_latent is None
    assert state.spk_emb is None
    assert state.generation_schedule_ids is None
    assert reset_calls == ["req"]


def test_result_adapter_releases_the_request_even_without_latents() -> None:
    reset_calls: list[str] = []
    request_builder, result_adapter = _adapters(reset_calls)
    data = request_builder(_payload())

    with pytest.raises(RuntimeError, match="generated no payload latents"):
        result_adapter(data)
    assert reset_calls == ["req"]
