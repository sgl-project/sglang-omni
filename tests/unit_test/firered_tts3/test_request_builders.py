# SPDX-License-Identifier: Apache-2.0
"""Request adapters: payload -> SGLang Req, runner data -> result payload."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.firered_tts3.model_runner import TEXT_EOT_ID
from sglang_omni.models.firered_tts3.payload_types import (
    ENGINE_TOKEN_SLACK,
    TEXT_TOKEN_BUDGET,
    FireRedTTS3State,
    load_firered_tts3_state,
)
from sglang_omni.models.firered_tts3.request_builders import (
    CONTROL_TOKEN_ID,
    FireRedTTS3SGLangRequestData,
    apply_latent_result,
    build_sglang_firered_tts3_request,
    build_stream_output,
    flush_stream_output,
)
from sglang_omni.proto import OmniRequest, StagePayload

PATCH_SIZE = 4
VOCAB_SIZE = 151936


def make_payload(state: FireRedTTS3State) -> StagePayload:
    return StagePayload(
        request_id="rid",
        request=OmniRequest(inputs={"text": "hello"}, params={}),
        data=state.to_dict(),
    )


def build(state: FireRedTTS3State) -> FireRedTTS3SGLangRequestData:
    return build_sglang_firered_tts3_request(
        make_payload(state), patch_size=PATCH_SIZE, vocab_size=VOCAB_SIZE
    )


@pytest.mark.parametrize(
    "state,prefill_token_count,prompt_patch_count,engine_token_budget",
    [
        pytest.param(
            FireRedTTS3State(
                input_ids=[11, 22, 33],
                prompt_latents=torch.zeros(1, 8, 64),
                max_patches=400,
                max_engine_tokens=420,
            ),
            6,
            2,
            420,
            id="explicit-budget",
        ),
        pytest.param(
            FireRedTTS3State(input_ids=[11], max_patches=250),
            2,
            0,
            270,
            id="fallback-budget",
        ),
    ],
)
def test_request_layout_is_speaker_plus_text_plus_prompt_patches(
    state: FireRedTTS3State,
    prefill_token_count: int,
    prompt_patch_count: int,
    engine_token_budget: int,
) -> None:
    request_data = build(state)
    assert request_data.prefill_len == prefill_token_count
    assert request_data.prompt_patch_count == prompt_patch_count
    assert request_data.req.origin_input_ids == [CONTROL_TOKEN_ID] * prefill_token_count
    assert request_data.max_new_tokens == engine_token_budget
    assert request_data.req.sampling_params.max_new_tokens == engine_token_budget
    # temperature=0.0 normalizes to the greedy convention (top_k == 1).
    assert request_data.req.sampling_params.top_k == 1
    assert request_data.req.sampling_params.stop_token_ids in (None, [])
    assert request_data.req.eos_token_ids == set()
    assert request_data.input_embeds_are_projected is True


def test_request_rejects_missing_or_misaligned_inputs() -> None:
    with pytest.raises(RuntimeError, match="input_ids"):
        build(FireRedTTS3State())
    state = FireRedTTS3State(input_ids=[11], prompt_latents=torch.zeros(1, 6, 64))
    with pytest.raises(ValueError, match="patch size"):
        build(state)


def test_apply_result_restores_state_with_latents_and_usage() -> None:
    state = FireRedTTS3State(input_ids=[11, 22], max_patches=10)
    request_data = build(state)
    patches = [torch.full((1, 4, 64), float(index)) for index in range(3)]
    request_data.latent_patches = list(patches)
    request_data.finish_reason = "stop"

    payload = apply_latent_result(request_data)
    restored = load_firered_tts3_state(payload)
    assert restored.generated_latents is not None
    assert restored.generated_latents.shape == (1, 12, 64)
    assert torch.equal(restored.generated_latents, torch.cat(patches, dim=1))
    assert restored.prompt_tokens == request_data.prefill_len
    assert restored.completion_tokens == 3
    assert restored.finish_reason == "stop"


def streaming_data(
    num_patches: int, *, stream: bool = True
) -> FireRedTTS3SGLangRequestData:
    state = FireRedTTS3State(input_ids=[11, 22], max_patches=10, stream=stream)
    request_data = build(state)
    request_data.latent_patches = [
        torch.full((1, 4, 64), float(index)) for index in range(num_patches)
    ]
    return request_data


def test_stream_output_follows_the_patch_ramp() -> None:
    request_data = streaming_data(8)

    # chunk 0: one patch as soon as it exists; chunk 1: two; then four.
    sizes = []
    while request_data.streamed_patches < 8:
        messages = build_stream_output("rid", request_data, None)
        if not messages:
            # Hold a patch back the way the decode loop would.
            break
        else:
            pass
        assert len(messages) == 1
        chunk = messages[0]
        assert chunk.type == "stream"
        assert chunk.target == "vocoder"
        assert chunk.metadata["modality"] == "audio_latents"
        assert chunk.metadata["stream"] is True
        # CPU chunks carry no readiness event.
        assert "codes_ready_event" not in chunk.metadata
        sizes.append(int(chunk.data.shape[1]) // 4)
    assert sizes == [1, 2, 4]
    assert request_data.streamed_patches == 7
    assert request_data.stream_chunk_index == 3

    flushed = flush_stream_output("rid", request_data)
    assert len(flushed) == 1
    assert int(flushed[0].data.shape[1]) == 4
    assert request_data.streamed_patches == 8
    assert flush_stream_output("rid", request_data) == []


def test_stream_output_holds_back_below_the_ramp() -> None:
    request_data = streaming_data(2)
    assert [
        int(message.data.shape[1])
        for message in build_stream_output("rid", request_data, None)
    ] == [4]
    assert build_stream_output("rid", request_data, None) == []
    flushed = flush_stream_output("rid", request_data)
    assert [int(message.data.shape[1]) for message in flushed] == [4]


def test_stream_output_skips_non_streaming_requests() -> None:
    request_data = streaming_data(4, stream=False)
    assert build_stream_output("rid", request_data, None) == []
    assert flush_stream_output("rid", request_data) == []
    assert request_data.streamed_patches == 0


def test_instruct_request_uses_real_tokens_and_text_budget() -> None:
    state = FireRedTTS3State(
        variant="instruct",
        task_type="semantic_edit",
        input_ids=[11, 22, 33, 151655, 151655],
        audio_in_latents=torch.zeros(1, 8, 64),
        max_patches=400,
        max_engine_tokens=400 + ENGINE_TOKEN_SLACK + TEXT_TOKEN_BUDGET + 2,
        text_temperature=0.0,
    )
    request_data = build(state)
    # Real chatml tokens (the pad slots ride inline), no placeholder row.
    assert request_data.prefill_len == 5
    assert request_data.req.origin_input_ids == [11, 22, 33, 151655, 151655]
    assert request_data.prompt_patch_count == 0
    assert request_data.text_ar_mode == "text"
    assert request_data.req.sampling_params.max_new_tokens == 622
    # Greedy task: no engine sampling seed installed.
    assert request_data.req.sampling_params.sampling_seed is None
    # The text-phase terminator must never be an engine stop id.
    assert TEXT_EOT_ID not in (request_data.req.sampling_params.stop_token_ids or [])
    assert request_data.req.eos_token_ids == set()


def test_instruct_voice_design_sampling_and_seed() -> None:
    state = FireRedTTS3State(
        variant="instruct",
        task_type="voice_design",
        input_ids=[11, 22],
        max_patches=400,
        max_engine_tokens=622,
        text_temperature=0.7,
        text_top_p=0.8,
        text_top_k=20,
        seed=2,
    )
    request_data = build(state)
    sampling_params = request_data.req.sampling_params
    assert sampling_params.temperature == 0.7
    assert sampling_params.top_p == 0.8
    assert sampling_params.top_k == 20
    assert sampling_params.sampling_seed == 2
    assert request_data.text_ar_mode == "text"


def test_instruct_tts_task_has_no_text_phase() -> None:
    state = FireRedTTS3State(
        variant="instruct",
        task_type="tts",
        input_ids=[11, 22, 151656, 151656],
        prompt_latents=torch.zeros(1, 8, 64),
        max_patches=400,
        max_engine_tokens=420,
    )
    request_data = build(state)
    assert request_data.text_ar_mode == "audio"
    # ICL prompt patches are the trailing video_pad slots.
    assert request_data.prompt_patch_count == 2
    assert request_data.prefill_len == 4
    # temperature 0.0 normalizes to the greedy convention (top_k == 1).
    assert request_data.req.sampling_params.top_k == 1


def test_apply_result_decodes_gen_text() -> None:
    class StubTokenizer:
        def decode(self, ids: list[int]) -> str:
            return " ".join(f"tok{i}" for i in ids)

    state = FireRedTTS3State(
        variant="instruct", task_type="voice_design", input_ids=[11], max_patches=10
    )
    request_data = build(state)
    request_data.latent_patches = [torch.zeros(1, 4, 64)]
    request_data.text_token_ids = [100, 200]
    request_data.finish_reason = "stop"
    payload = apply_latent_result(request_data, text_tokenizer=StubTokenizer())
    restored = load_firered_tts3_state(payload)
    assert restored.gen_text == "tok100 tok200"

    # No text phase -> no gen_text.
    second_state = FireRedTTS3State(input_ids=[11], max_patches=10)
    second_request_data = build(second_state)
    second_request_data.latent_patches = [torch.zeros(1, 4, 64)]
    payload2 = apply_latent_result(second_request_data, text_tokenizer=StubTokenizer())
    assert load_firered_tts3_state(payload2).gen_text is None


def build_with_ratio(
    state: FireRedTTS3State, ratio: tuple[int, int] = (2, 1)
) -> FireRedTTS3SGLangRequestData:
    return build_sglang_firered_tts3_request(
        make_payload(state),
        patch_size=PATCH_SIZE,
        vocab_size=VOCAB_SIZE,
        interleave_ratio=ratio,
    )


def test_speaker_prefill_request_uses_real_chatml_tokens() -> None:
    # The speaker variant's prefill form: the full chatml row (text already
    # scattered into the holder slots by preprocessing), no placeholder row.
    state = FireRedTTS3State(
        variant="speaker",
        input_ids=[11, 22, 33, 151672],
        max_patches=600,
        max_engine_tokens=620,
    )
    request_data = build(state)
    assert request_data.prefill_len == 4
    assert request_data.req.origin_input_ids == [11, 22, 33, 151672]
    assert request_data.prompt_patch_count == 0
    assert request_data.text_ar_mode == "audio"
    assert request_data.dual_stream is False
    assert request_data.req.sampling_params.max_new_tokens == 620


def test_speaker_dual_stream_request_carries_the_text_stream() -> None:
    state = FireRedTTS3State(
        variant="speaker",
        input_ids=[11, 22, 151672],
        dual_stream=True,
        text_stream_ids=[101, 102, 103, 104],
        text_pad_id=151671,
        max_patches=600,
        max_engine_tokens=1820,
    )
    request_data = build_with_ratio(state)
    assert request_data.dual_stream is True
    assert request_data.interleave_ratio == (2, 1)
    assert request_data.req.sampling_params.max_new_tokens == 1820
    assert request_data.prefill_len == 3


def test_speaker_dual_stream_requires_ratio_and_text_stream() -> None:
    state = FireRedTTS3State(
        variant="speaker",
        input_ids=[11],
        dual_stream=True,
        text_stream_ids=[101],
        text_pad_id=151671,
    )
    with pytest.raises(ValueError, match="text_latent_interleave_ratio"):
        build(state)  # no interleave_ratio: the checkpoint lacks the capability
    with pytest.raises(ValueError, match="text_stream_ids"):
        build_with_ratio(
            FireRedTTS3State(
                variant="speaker", input_ids=[11], dual_stream=True, text_pad_id=151671
            )
        )
    with pytest.raises(ValueError, match="text_stream_ids"):
        build_with_ratio(
            FireRedTTS3State(
                variant="speaker",
                input_ids=[11],
                dual_stream=True,
                text_stream_ids=[101],
                text_pad_id=-1,
            )
        )


def test_dual_stream_stream_output_emits_every_patch() -> None:
    state = FireRedTTS3State(
        variant="speaker",
        input_ids=[11, 22],
        dual_stream=True,
        text_stream_ids=[101, 102],
        text_pad_id=151671,
        max_patches=10,
        stream=True,
    )
    request_data = build_with_ratio(state)
    request_data.latent_patches = [
        torch.full((1, 4, 64), float(index)) for index in range(3)
    ]
    # One patch per chunk, the moment it exists (reference per-patch decode).
    sizes = []
    for _ in range(3):
        messages = build_stream_output("rid", request_data, None)
        assert len(messages) == 1
        sizes.append(int(messages[0].data.shape[1]) // 4)
    assert sizes == [1, 1, 1]
    assert build_stream_output("rid", request_data, None) == []
    assert request_data.streamed_patches == 3
