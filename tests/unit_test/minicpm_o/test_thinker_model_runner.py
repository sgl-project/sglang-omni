# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o thinker runners: offline length penalty and the duplex unit sampler."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni.models.minicpm_o.duplex_sampler import filter_top_k_top_p
from sglang_omni.models.minicpm_o.native_config import MiniCPMODuplexSampling
from sglang_omni.models.minicpm_o.native_thinker_model_runner import (
    MiniCPMOThinkerModelRunner as DuplexThinkerModelRunner,
)
from sglang_omni.models.minicpm_o.session_adapters import ThinkerAdapter
from sglang_omni.models.minicpm_o.special_tokens import (
    REQUIRED_SPECIAL_TOKENS,
    MiniCPMOSpecialTokenIds,
    resolve_special_token_ids,
)
from sglang_omni.models.minicpm_o.thinker_model_runner import MiniCPMOThinkerModelRunner
from sglang_omni.models.minicpm_o.thinker_state import MiniCPMOThinkerSessionState
from sglang_omni.proto import OmniRequest, StagePayload

EOS_TOKEN_IDS = [1, 3]
VOCAB_SIZE = 128
SPECIAL_TOKEN_ID_START = 100
BAD_TOKEN_IDS = (7, 8, 94)
PLAIN_TOKEN_ID = 42
OTHER_TOKEN_ID = 55
THIRD_TOKEN_ID = 60


def bare_runner() -> MiniCPMOThinkerModelRunner:
    runner = object.__new__(MiniCPMOThinkerModelRunner)
    runner.eos_token_ids = EOS_TOKEN_IDS
    runner.eos_token_id_cache = None
    return runner


def scheduler_request(params: dict[str, Any]) -> SimpleNamespace:
    payload = StagePayload(
        request_id="request-0",
        request=OmniRequest(
            inputs=None,
            params=params,
            metadata={"output_modalities": ["text"]},
        ),
        data=None,
    )
    return SimpleNamespace(data=SimpleNamespace(stage_payload=payload))


def penalized_request(length_penalty: float) -> SimpleNamespace:
    return scheduler_request(
        {"stage_params": {"thinker": {"length_penalty": length_penalty}}}
    )


def test_length_penalty_scales_eos_logits_per_request() -> None:
    original = torch.tensor(
        [
            [0.5, 2.0, -1.0, -4.0],
            [0.5, 2.0, -1.0, -4.0],
            [0.5, 2.0, -1.0, -4.0],
        ]
    )
    logits_output = SimpleNamespace(next_token_logits=original.clone())

    bare_runner().process_sampling_logits(
        logits_output,
        [
            penalized_request(1.0),
            penalized_request(2.0),
            penalized_request(1.0),
        ],
    )

    torch.testing.assert_close(logits_output.next_token_logits[0], original[0])
    torch.testing.assert_close(logits_output.next_token_logits[2], original[2])
    expected = original[1].clone()
    expected[EOS_TOKEN_IDS] = torch.tensor([2.0 / 2.0, -4.0 * 2.0])
    torch.testing.assert_close(logits_output.next_token_logits[1], expected)


def test_lookahead_samples_from_penalized_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = torch.tensor([[0.5, 2.0, -1.0, -4.0]])
    logits_output = SimpleNamespace(next_token_logits=original.clone())
    logits_seen_by_parent = []

    def parent_sample_lookahead(self, logits_output, forward_batch, requests):
        logits_seen_by_parent.append(logits_output.next_token_logits.clone())
        return torch.tensor([0])

    monkeypatch.setattr(ThinkerModelRunner, "sample_lookahead", parent_sample_lookahead)

    bare_runner().sample_lookahead(
        logits_output, forward_batch=None, requests=[penalized_request(2.0)]
    )

    expected = original.clone()
    expected[0, EOS_TOKEN_IDS] = torch.tensor([2.0 / 2.0, -4.0 * 2.0])
    torch.testing.assert_close(logits_seen_by_parent[0], expected)


def test_missing_length_penalty_leaves_logits_unchanged() -> None:
    original = torch.tensor([[0.5, 2.0, -1.0, -4.0]])
    logits_output = SimpleNamespace(next_token_logits=original.clone())

    bare_runner().process_sampling_logits(logits_output, [scheduler_request({})])

    torch.testing.assert_close(logits_output.next_token_logits, original)


def duplex_tokenizer() -> Mock:
    tokenizer = Mock(unk_token_id=0, bad_token_ids=list(BAD_TOKEN_IDS))
    special_token_ids = range(
        SPECIAL_TOKEN_ID_START, SPECIAL_TOKEN_ID_START + len(REQUIRED_SPECIAL_TOKENS)
    )
    tokenizer.convert_tokens_to_ids.side_effect = dict(
        zip(REQUIRED_SPECIAL_TOKENS, special_token_ids)
    ).__getitem__
    return tokenizer


SPECIAL_TOKENS = resolve_special_token_ids(
    duplex_tokenizer(), bad_token_ids=BAD_TOKEN_IDS
)


def duplex_runner(
    special_tokens: MiniCPMOSpecialTokenIds | None,
) -> DuplexThinkerModelRunner:
    runner = DuplexThinkerModelRunner.__new__(DuplexThinkerModelRunner)
    runner.special_tokens = special_tokens
    runner.forbidden_token_index = None
    runner.eos_token_ids = []
    return runner


def session_state(
    **sampling_overrides: float | int | bool,
) -> MiniCPMOThinkerSessionState:
    sampling = {"greedy": True, "repetition_penalty": 1.0, **sampling_overrides}
    return MiniCPMOThinkerSessionState(sampling=MiniCPMODuplexSampling(**sampling))


def unit_request(
    state: MiniCPMOThinkerSessionState,
    *,
    generation_step: int = 1,
    is_listen_forced: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(
            thinker_state=state,
            generation_steps=generation_step,
            is_listen_forced=is_listen_forced,
        )
    )


def unit_logits(*rows: dict[int, float]) -> torch.Tensor:
    logits = torch.full((len(rows), VOCAB_SIZE), -torch.inf)
    for row_index, finite_logits in enumerate(rows):
        for token_id, value in finite_logits.items():
            logits[row_index, token_id] = value
    return logits


def sample(
    runner: DuplexThinkerModelRunner,
    logits: torch.Tensor,
    requests: list[SimpleNamespace],
) -> list[int]:
    return runner.sample_next_token_ids(
        SimpleNamespace(next_token_logits=logits), None, None, requests
    ).tolist()


@pytest.mark.parametrize(
    ("use_runner", "max_new_tokens", "generation_step", "closes"),
    [
        (True, 20, 0, False),
        (False, 20, 0, False),
        (False, 5, 4, True),
        (False, 6, 4, False),
    ],
)
def test_duplex_sample_masks_bad_tokens_and_closes_at_budget(
    use_runner: bool, max_new_tokens: int, generation_step: int, closes: bool
) -> None:
    """Both init paths keep the checkpoint mask; the token budget closes the chunk."""
    tokenizer = duplex_tokenizer()
    if use_runner:
        runner = duplex_runner(None)
        special_tokens = runner.resolve_special_tokens(
            SimpleNamespace(req=Mock(tokenizer=tokenizer))
        )
    else:
        special_tokens = ThinkerAdapter(tokenizer, VOCAB_SIZE).special
        runner = duplex_runner(special_tokens)
    logits = unit_logits(
        {BAD_TOKEN_IDS[0]: 100.0, special_tokens.tts_pad: 99.0, PLAIN_TOKEN_ID: 0.0}
    )
    state = session_state(max_new_tokens_per_unit=max_new_tokens)

    token_ids = sample(
        runner, logits, [unit_request(state, generation_step=generation_step)]
    )

    assert token_ids == [special_tokens.chunk_eos if closes else PLAIN_TOKEN_ID]


def test_forced_listen_applies_to_the_first_step_only() -> None:
    runner = duplex_runner(SPECIAL_TOKENS)
    first_step = session_state()
    later_step = session_state()
    logits = unit_logits({PLAIN_TOKEN_ID: 1.0}, {PLAIN_TOKEN_ID: 1.0})

    token_ids = sample(
        runner,
        logits,
        [
            unit_request(first_step, generation_step=0, is_listen_forced=True),
            unit_request(later_step, generation_step=1, is_listen_forced=True),
        ],
    )

    assert token_ids == [SPECIAL_TOKENS.listen, PLAIN_TOKEN_ID]
    assert (first_step.force_listen_counter, later_step.force_listen_counter) == (1, 0)
    assert (first_step.generated_history, later_step.generated_history) == (
        [],
        [PLAIN_TOKEN_ID],
    )


def test_chunk_eos_from_the_first_sample_skips_the_second_stage() -> None:
    runner = duplex_runner(SPECIAL_TOKENS)
    state = session_state()
    state.is_turn_ended = False
    logits = unit_logits({SPECIAL_TOKENS.chunk_eos: 2.0, PLAIN_TOKEN_ID: 1.0})

    assert sample(runner, logits, [unit_request(state)]) == [SPECIAL_TOKENS.chunk_eos]
    assert state.generated_history == []
    assert state.is_turn_ended is False


def test_second_stage_masks_forbidden_tokens_and_penalizes_each_row_by_its_history() -> (
    None
):
    runner = duplex_runner(SPECIAL_TOKENS)
    masked = session_state()
    penalized = session_state(repetition_penalty=2.0)
    penalized.generated_history = [PLAIN_TOKEN_ID]
    unpenalized = session_state()
    unpenalized.generated_history = [PLAIN_TOKEN_ID]
    logits = unit_logits(
        {BAD_TOKEN_IDS[0]: 9.0, SPECIAL_TOKENS.tts_pad: 8.0, PLAIN_TOKEN_ID: 1.0},
        {PLAIN_TOKEN_ID: 6.0, OTHER_TOKEN_ID: 4.0},
        {PLAIN_TOKEN_ID: 6.0, OTHER_TOKEN_ID: 4.0},
    )

    token_ids = sample(
        runner,
        logits,
        [unit_request(masked), unit_request(penalized), unit_request(unpenalized)],
    )

    assert token_ids == [PLAIN_TOKEN_ID, OTHER_TOKEN_ID, PLAIN_TOKEN_ID]
    assert penalized.generated_history == [PLAIN_TOKEN_ID, OTHER_TOKEN_ID]
    assert unpenalized.generated_history == [PLAIN_TOKEN_ID, PLAIN_TOKEN_ID]


def test_repeated_history_tokens_are_penalized_once() -> None:
    runner = duplex_runner(SPECIAL_TOKENS)
    state = session_state(repetition_penalty=2.0)
    state.generated_history = [PLAIN_TOKEN_ID, PLAIN_TOKEN_ID]
    logits = unit_logits({PLAIN_TOKEN_ID: 9.0, OTHER_TOKEN_ID: 4.0})

    assert sample(runner, logits, [unit_request(state)]) == [PLAIN_TOKEN_ID]


def test_history_keeps_only_the_repetition_window() -> None:
    runner = duplex_runner(SPECIAL_TOKENS)
    state = session_state(repetition_window_size=2)
    state.generated_history = [1, 2]

    sample(runner, unit_logits({PLAIN_TOKEN_ID: 1.0}), [unit_request(state)])

    assert state.generated_history == [2, PLAIN_TOKEN_ID]


@pytest.mark.parametrize(
    ("is_turn_ended", "expected_token_id"),
    [(True, SPECIAL_TOKENS.listen), (False, SPECIAL_TOKENS.tts_bos)],
)
def test_listen_becomes_tts_bos_mid_turn(
    is_turn_ended: bool, expected_token_id: int
) -> None:
    runner = duplex_runner(SPECIAL_TOKENS)
    state = session_state()
    state.is_turn_ended = is_turn_ended

    token_ids = sample(
        runner, unit_logits({SPECIAL_TOKENS.listen: 1.0}), [unit_request(state)]
    )

    assert token_ids == [expected_token_id]
    assert state.generated_history == [SPECIAL_TOKENS.listen]
    assert state.is_turn_ended is is_turn_ended


def test_turn_eos_ends_the_turn_and_a_plain_token_reopens_it() -> None:
    runner = duplex_runner(SPECIAL_TOKENS)
    state = session_state()
    state.is_turn_ended = False

    assert sample(
        runner, unit_logits({SPECIAL_TOKENS.turn_eos: 1.0}), [unit_request(state)]
    ) == [SPECIAL_TOKENS.turn_eos]
    assert state.is_turn_ended is True
    assert sample(
        runner, unit_logits({PLAIN_TOKEN_ID: 1.0}), [unit_request(state)]
    ) == [PLAIN_TOKEN_ID]
    assert state.is_turn_ended is False


@pytest.mark.parametrize(
    ("listen_prob_scale", "expected_token_id"),
    [(1.0, SPECIAL_TOKENS.listen), (0.5, PLAIN_TOKEN_ID)],
)
def test_listen_prob_scale_rescales_the_listen_logit(
    listen_prob_scale: float, expected_token_id: int
) -> None:
    runner = duplex_runner(SPECIAL_TOKENS)
    state = session_state(listen_prob_scale=listen_prob_scale)
    logits = unit_logits({SPECIAL_TOKENS.listen: 10.0, PLAIN_TOKEN_ID: 8.0})

    assert sample(runner, logits, [unit_request(state)]) == [expected_token_id]


def test_sampled_and_greedy_rows_share_one_step() -> None:
    runner = duplex_runner(SPECIAL_TOKENS)
    greedy = session_state()
    sampled = session_state(greedy=False, temperature=0.7, top_k=20, top_p=0.8)
    closing = session_state(greedy=False, temperature=0.7, top_k=20, top_p=0.8)
    logits = unit_logits(
        {PLAIN_TOKEN_ID: 1.0, OTHER_TOKEN_ID: 0.0},
        {OTHER_TOKEN_ID: 1.0},
        {SPECIAL_TOKENS.chunk_eos: 1.0},
    )

    token_ids = sample(
        runner,
        logits,
        [unit_request(greedy), unit_request(sampled), unit_request(closing)],
    )

    assert token_ids == [PLAIN_TOKEN_ID, OTHER_TOKEN_ID, SPECIAL_TOKENS.chunk_eos]
    assert (
        greedy.generated_history,
        sampled.generated_history,
        closing.generated_history,
    ) == ([PLAIN_TOKEN_ID], [OTHER_TOKEN_ID], [])


def test_sampled_rows_draw_from_their_filtered_support() -> None:
    runner = duplex_runner(SPECIAL_TOKENS)
    sampled = session_state(greedy=False, temperature=0.7, top_k=20, top_p=0.8)
    closing = session_state(greedy=False, temperature=0.7, top_k=20, top_p=0.8)
    zero_temperature = session_state(greedy=False, temperature=0.0)
    logits = unit_logits(
        {OTHER_TOKEN_ID: 1.0},
        {SPECIAL_TOKENS.chunk_eos: 1.0},
        {PLAIN_TOKEN_ID: 2.0, OTHER_TOKEN_ID: 1.0},
    )

    token_ids = sample(
        runner,
        logits,
        [unit_request(sampled), unit_request(closing), unit_request(zero_temperature)],
    )

    assert token_ids == [OTHER_TOKEN_ID, SPECIAL_TOKENS.chunk_eos, PLAIN_TOKEN_ID]
    assert (
        sampled.generated_history,
        closing.generated_history,
        zero_temperature.generated_history,
    ) == ([OTHER_TOKEN_ID], [], [PLAIN_TOKEN_ID])


DRAW_COUNT = 6000
FREQUENCY_TOLERANCE = 0.03


@pytest.mark.parametrize(
    ("top_p", "expected_frequencies"),
    [
        (1.0, {PLAIN_TOKEN_ID: 0.7, OTHER_TOKEN_ID: 0.2, THIRD_TOKEN_ID: 0.1}),
        (
            0.75,
            {PLAIN_TOKEN_ID: 0.7 / 0.9, OTHER_TOKEN_ID: 0.2 / 0.9, THIRD_TOKEN_ID: 0.0},
        ),
    ],
    ids=["softmax", "nucleus-drops-the-tail"],
)
def test_sampled_tokens_follow_the_filtered_softmax(
    top_p: float, expected_frequencies: dict[int, float]
) -> None:
    torch.manual_seed(0)
    runner = duplex_runner(SPECIAL_TOKENS)
    state = session_state(greedy=False, temperature=1.0, top_k=-1, top_p=top_p)
    logits = unit_logits(
        {
            PLAIN_TOKEN_ID: math.log(0.7),
            OTHER_TOKEN_ID: math.log(0.2),
            THIRD_TOKEN_ID: math.log(0.1),
        }
    )
    counts = {token_id: 0 for token_id in expected_frequencies}

    for _ in range(DRAW_COUNT):
        (token_id,) = sample(runner, logits, [unit_request(state)])
        counts[token_id] += 1

    for token_id, expected in expected_frequencies.items():
        assert abs(counts[token_id] / DRAW_COUNT - expected) <= FREQUENCY_TOLERANCE


def test_top_k_and_top_p_filter_each_row_of_a_batch() -> None:
    logits = torch.log(torch.tensor([[0.6, 0.3, 0.1], [0.4, 0.35, 0.25]]))

    top_k_filtered = filter_top_k_top_p(logits, top_k=2, top_p=1.0)
    top_p_filtered = filter_top_k_top_p(logits, top_k=-1, top_p=0.5)

    torch.testing.assert_close(
        top_k_filtered,
        torch.tensor(
            [
                [logits[0, 0], logits[0, 1], -torch.inf],
                [logits[1, 0], logits[1, 1], -torch.inf],
            ]
        ),
    )
    torch.testing.assert_close(
        top_p_filtered,
        torch.tensor(
            [
                [logits[0, 0], -torch.inf, -torch.inf],
                [logits[1, 0], logits[1, 1], -torch.inf],
            ]
        ),
    )


@pytest.mark.parametrize("greedy", [True, False])
def test_one_step_moves_token_ids_to_the_host_exactly_twice(
    monkeypatch: pytest.MonkeyPatch, greedy: bool
) -> None:
    """A step costs two Python-level transfers, not two per row, and never a multinomial sync."""
    transfers: list[str] = []
    multinomial_calls: list[int] = []
    original_item = torch.Tensor.item
    original_tolist = torch.Tensor.tolist
    original_multinomial = torch.multinomial

    def counted_item(tensor: torch.Tensor) -> int | float:
        transfers.append("item")
        return original_item(tensor)

    def counted_tolist(tensor: torch.Tensor) -> list[int]:
        transfers.append("tolist")
        return original_tolist(tensor)

    def counted_multinomial(*arguments: object, **keywords: object) -> torch.Tensor:
        multinomial_calls.append(1)
        return original_multinomial(*arguments, **keywords)

    monkeypatch.setattr(torch.Tensor, "item", counted_item)
    monkeypatch.setattr(torch.Tensor, "tolist", counted_tolist)
    monkeypatch.setattr(torch, "multinomial", counted_multinomial)
    runner = duplex_runner(SPECIAL_TOKENS)
    states = [session_state(greedy=greedy, temperature=0.7) for _ in range(6)]
    logits = unit_logits(*([{PLAIN_TOKEN_ID: 1.0}] * len(states)))

    runner.sample_next_token_ids(
        SimpleNamespace(next_token_logits=logits),
        None,
        None,
        [unit_request(state) for state in states],
    )

    assert transfers == ["tolist", "tolist"]
    assert multinomial_calls == []
