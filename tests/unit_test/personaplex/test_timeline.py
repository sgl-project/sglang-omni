# SPDX-License-Identifier: Apache-2.0
"""The delayed timeline must place every stream where the reference reads it."""

import pytest
import torch

from sglang_omni.models.personaplex.architecture import (
    AUDIO_INITIAL_ID,
    DELAYS,
    PROMPT_SILENCE_FRAMES,
    SILENCE_CODES,
    SINE_CODES,
    TEXT_INITIAL_ID,
    TEXT_PAD_ID,
)
from sglang_omni.models.personaplex.timeline import (
    REFERENCE_CACHE_POSITIONS,
    UNKNOWN,
    build_prompt_frames,
    build_timeline,
    delay_stream,
    output_frame,
    voice_tail_codes_from_cache,
)

AGENT_DELAYS = DELAYS[1:9]


def make_voice_codes(frames: int) -> torch.Tensor:
    return torch.arange(frames * 8).view(frames, 8) + 100


def make_user_codes(frames: int) -> torch.Tensor:
    return torch.arange(frames * 8).view(frames, 8) + 500


def test_delay_stream_drops_frame_zero_and_seeds_initial_tokens():
    frames = torch.arange(1, 5).view(4, 1).repeat(1, 3)
    rows = delay_stream(frames, (0, 1, 1), initial=99)
    assert rows[:, 0].tolist() == [99, 2, 3, 4]
    assert rows[:, 1].tolist() == [99, 99, 2, 3]


def test_prompt_frames_follow_the_reference_order():
    prompt = build_prompt_frames(
        voice_frames=3, text_prompt_ids=[7, 8], voice_codes=make_voice_codes(3)
    )
    silence, text = PROMPT_SILENCE_FRAMES, 2
    assert prompt.num_frames == 3 + silence + text + silence
    assert (
        prompt.text.tolist()
        == [TEXT_PAD_ID] * (3 + silence) + [7, 8] + [TEXT_PAD_ID] * silence
    )
    assert prompt.agent[:3].tolist() == make_voice_codes(3).tolist()
    assert prompt.agent[3:].tolist() == [list(SILENCE_CODES)] * (
        silence + text + silence
    )
    assert prompt.user.tolist() == [list(SINE_CODES)] * prompt.num_frames


def test_timeline_rows_and_generation_boundary():
    prompt = build_prompt_frames(
        voice_frames=3, text_prompt_ids=[7, 8], voice_codes=make_voice_codes(3)
    )
    timeline = build_timeline(prompt, make_user_codes(5))
    num_prompt = prompt.num_frames
    rows = timeline.prefill_tokens
    assert rows.shape == (num_prompt, 17)
    assert rows[0].tolist() == [TEXT_INITIAL_ID] + [AUDIO_INITIAL_ID] * 16
    assert rows[1, 1].item() == 108 and rows[1, 2:9].tolist() == [AUDIO_INITIAL_ID] * 7
    assert (
        rows[1, 9].item() == SINE_CODES[0]
        and rows[1, 10:].tolist() == [AUDIO_INITIAL_ID] * 7
    )
    assert rows[2, 2:9].tolist() == list(range(109, 116))
    assert timeline.forced_agent_at_start.tolist() == [UNKNOWN, *SILENCE_CODES[1:]]
    assert timeline.agent_row_before_start.tolist() == list(SILENCE_CODES)
    assert timeline.user_rows[num_prompt].tolist() == [500, *SINE_CODES[1:]]
    assert timeline.user_rows[num_prompt + 1].tolist() == [508, *range(501, 508)]
    assert timeline.user_rows.shape[0] == num_prompt + 5
    assert timeline.num_frames == 5
    assert timeline.input_position(0) == num_prompt - 1


def test_packaged_voice_rows_come_from_embeddings():
    voice_frames = 4
    cache = torch.full((17, REFERENCE_CACHE_POSITIONS), -7, dtype=torch.long)
    codes = make_voice_codes(voice_frames)
    for frame in range(voice_frames):
        for k, delay in enumerate(AGENT_DELAYS):
            cache[1 + k, (frame + delay) % REFERENCE_CACHE_POSITIONS] = codes[frame, k]
    tail = voice_tail_codes_from_cache(cache, voice_frames)
    assert tail[1].tolist() == codes[3].tolist()
    assert (
        tail[0, 0].item() == UNKNOWN and tail[0, 1:].tolist() == codes[2, 1:].tolist()
    )

    prompt = build_prompt_frames(voice_frames=voice_frames, text_prompt_ids=[])
    embeddings = torch.randn(voice_frames - 1, 16)
    timeline = build_timeline(
        prompt, make_user_codes(2), voice_embeddings=embeddings, voice_tail_codes=tail
    )
    assert timeline.prefill_embedding_positions == [0, 1, 2]
    assert timeline.prefill_tokens[3, 1].item() == codes[3, 0].item()
    assert timeline.prefill_tokens[3, 2:9].tolist() == codes[2, 1:].tolist()
    assert timeline.prefill_tokens[4, 1].item() == SILENCE_CODES[0]
    assert timeline.prefill_tokens[4, 2:9].tolist() == codes[3, 1:].tolist()
    assert not (timeline.prefill_tokens[3:] == UNKNOWN).any()


def test_packaged_voice_row_count_is_checked():
    prompt = build_prompt_frames(voice_frames=4, text_prompt_ids=[])
    tail = torch.full((2, 8), 5, dtype=torch.long)
    with pytest.raises(ValueError, match="stored rows"):
        build_timeline(
            prompt,
            make_user_codes(1),
            voice_embeddings=torch.zeros(2, 4),
            voice_tail_codes=tail,
        )


def test_output_frame_takes_undelayed_codebook_from_previous_row():
    previous = torch.arange(8)
    current = torch.arange(8) + 10
    assert output_frame(previous, current).tolist() == [0, 11, 12, 13, 14, 15, 16, 17]
