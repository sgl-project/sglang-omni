# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sglang_omni.models.dots_tts.payload_types import DotsTTSState
from sglang_omni.models.dots_tts.request_builders import (
    apply_latent_result,
    build_sglang_dots_tts_request,
    build_stream_output,
)
from sglang_omni.proto import OmniRequest, StagePayload


def _payload(state: DotsTTSState) -> StagePayload:
    return StagePayload(
        request_id="rid",
        request=OmniRequest(inputs={"text": "hello"}, params={}),
        data=state.to_dict(),
    )


def test_request_budget_is_remaining_schedule_spans() -> None:
    state = DotsTTSState(
        generation_schedule=torch.tensor([[1, 2, 99, 99, 99, 99]]),
        audio_span_token_ids=[99],
        latent_patch_size=4,
        vocab_size=128,
        prompt_latents=torch.zeros(1, 8, 3),
    )
    data = build_sglang_dots_tts_request(_payload(state))

    assert data.prefill_end == 4
    assert data.prompt_span_positions.tolist() == [2, 3]
    assert data.req.origin_input_ids == [1, 2, 99, 99]
    assert data.req.sampling_params.max_new_tokens == 2
    assert data.control_token_id == 99


def test_public_budget_counts_only_emitted_post_prompt_patches() -> None:
    state = DotsTTSState(
        generation_schedule=torch.tensor([[1, 2, 99, 99, 99, 99, 99]]),
        audio_span_token_ids=[99],
        latent_patch_size=4,
        vocab_size=128,
        prompt_latents=torch.zeros(1, 8, 3),
        max_new_tokens=1,
    )

    data = build_sglang_dots_tts_request(_payload(state))

    # The first decoded patch regenerates the prompt tail and is not emitted.
    assert data.req.sampling_params.max_new_tokens == 2


def test_prompt_schedule_requires_a_patch_after_regeneration() -> None:
    state = DotsTTSState(
        generation_schedule=torch.tensor([[1, 2, 99, 99, 99]]),
        audio_span_token_ids=[99],
        latent_patch_size=4,
        vocab_size=128,
        prompt_latents=torch.zeros(1, 8, 3),
    )

    with pytest.raises(ValueError, match="at least one payload patch"):
        build_sglang_dots_tts_request(_payload(state))


def test_result_and_stream_use_pipeline_state(monkeypatch) -> None:
    state = DotsTTSState(
        stream=True,
        generation_schedule=torch.tensor([[1, 99]]),
        audio_span_token_ids=[99],
        vocab_size=128,
    )
    data = build_sglang_dots_tts_request(_payload(state))
    patch = torch.ones(1, 4, 3)
    data.latest_latent_patch = patch
    data.latent_patches = [patch]
    data.engine_start_s = 10.0
    monkeypatch.setattr(
        "sglang_omni.models.dots_tts.request_builders.time.perf_counter",
        lambda: 12.5,
    )

    [message] = list(build_stream_output("rid", data, None))
    payload = apply_latent_result(data)

    assert message.metadata == {
        "modality": "audio_latents",
        "stream": True,
        "chunk_id": 0,
    }
    restored = DotsTTSState.from_dict(payload.data)
    assert restored.generated_latents.shape == (1, 4, 3)
    assert restored.prompt_tokens == 1
    assert restored.completion_tokens == 1
    assert restored.engine_time_s == 2.5
