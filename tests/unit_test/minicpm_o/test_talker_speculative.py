# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o talker speculative sampling against the talker's own sampling procedure."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.kernels.ops.speculative.dspark.dspark_accept import finalize_accept_lens
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import build_out_tokens
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import Sampler
from sglang.srt.models.dspark import VanillaMarkov
from sglang.srt.runtime_context import get_context, get_parallel
from torch import nn

from sglang_omni.models.minicpm_o.talker_dspark import (
    DraftKvInjectionGraphs,
    ExactAcceptGraphs,
    FoldedTalkerDraw,
    MiniCPMOTalkerDSparkWorker,
    StaticSamplingRows,
    accept_block,
    commit_to_slots,
    draw_block,
    state_from_slots,
)
from sglang_omni.models.minicpm_o.talker_dspark_sampling import (
    TalkerSamplingState,
    exact_accept,
    nucleus,
    sample_draft_block,
    shape_logits,
    target_probabilities,
    window_counts,
)
from sglang_omni.models.minicpm_o.talker_model_runner import (
    REP_PENALTY_WINDOW,
    TalkerSlotState,
)
from sglang_omni.models.minicpm_o.talker_slot_kernels import commit_slot_rows
from sglang_omni.platforms import current_platform
from tests.unit_test.fixtures.accelerator import require_cuda

VOCAB_SIZE = 12
CODEC_EOS_ID = VOCAB_SIZE - 1
SAMPLER_VOCAB_SIZE = 32
ROUNDING_TIE_LOGIT = 1e-5


def recent_tokens(
    outputs: list[int], vocabulary_size: int = VOCAB_SIZE
) -> torch.Tensor:
    tail = outputs[-REP_PENALTY_WINDOW:]
    return torch.tensor(
        [vocabulary_size] * (REP_PENALTY_WINDOW - len(tail)) + tail, dtype=torch.long
    )


def make_state(
    histories: list[list[int]],
    *,
    device: torch.device | None = None,
    vocabulary_size: int = VOCAB_SIZE,
    **overrides: object,
) -> TalkerSamplingState:
    batch = len(histories)
    values: dict[str, object] = dict(
        recent_tokens=torch.stack(
            [recent_tokens(history, vocabulary_size) for history in histories]
        ),
        generated=torch.tensor([len(history) for history in histories]),
        min_new_tokens=torch.full((batch,), 6),
        penalties=torch.full((batch,), 1.3),
        suppress=torch.zeros(batch, vocabulary_size, dtype=torch.bool),
        temperatures=torch.full((batch,), 0.8),
        top_ks=torch.full((batch,), 6),
        top_ps=torch.full((batch,), 0.9),
        greedy=torch.zeros(batch, dtype=torch.bool),
        max_top_k=None,
        need_top_filtering=True,
        eos_id=vocabulary_size - 1,
    )
    values.update(overrides)
    values["is_all_greedy"] = values["greedy"].all()
    return TalkerSamplingState(
        **{
            name: value.to(device=device) if isinstance(value, torch.Tensor) else value
            for name, value in values.items()
        }
    )


def slot_state_shaping(
    logits: torch.Tensor, histories: list[list[int]], state: TalkerSamplingState
) -> torch.Tensor:
    """The runner's own shaping: TalkerSlotState after appending each history token."""
    slots = TalkerSlotState.allocate(
        len(histories), logits.shape[1], state.eos_id, logits.device
    )
    rows = torch.arange(len(histories), device=logits.device)
    slots.min_new_tokens[rows] = state.min_new_tokens
    slots.penalties[rows] = state.penalties
    slots.suppress[rows] = state.suppress
    for row, history in enumerate(histories):
        for token in history:
            slots.append(
                torch.tensor([row], device=logits.device),
                torch.tensor([token], device=logits.device),
            )
    shaped = logits.clone()
    slots.apply(shaped, rows)
    return shaped


def sample_native_distribution(
    logits: torch.Tensor,
    histories: list[list[int]],
    sampling_state: TalkerSamplingState,
    *,
    is_all_greedy: bool,
) -> torch.Tensor:
    count, vocabulary_size = logits.shape
    native_output = LogitsProcessorOutput(
        next_token_logits=slot_state_shaping(logits, histories, sampling_state),
        hidden_states=None,
    )
    slots = TalkerSlotState.allocate(
        count, vocabulary_size, sampling_state.eos_id, logits.device
    )
    slots.temperatures[:count] = sampling_state.temperatures[:, None]
    slots.top_ks[:count] = sampling_state.top_ks
    slots.top_ps[:count] = sampling_state.top_ps
    rows = torch.arange(count, device=logits.device)
    sampling_info = slots.sampling_info(rows)
    sampling_info.is_all_greedy = is_all_greedy
    sampling_info.is_any_greedy = is_all_greedy
    sampling_info.sampling_seed = torch.zeros(
        count, dtype=torch.long, device=logits.device
    )
    sampling_info.sampling_mask_batch_indices = rows
    sampling_info.sampling_support_logprobs_capture_indices = rows

    with (
        get_context().override_server_args(
            sampling_backend="pytorch", sampling_mask_max_tokens=vocabulary_size
        ),
        get_parallel().override(tp_group=SimpleNamespace(device_group=None)),
    ):
        sampler = Sampler()
        sampler(
            native_output,
            sampling_info,
            False,
            [0] * count,
            [[] for _ in histories],
            torch.zeros(count, dtype=torch.long, device=logits.device),
        )
    captured = native_output.sampling_mask_output
    assert captured is not None and captured.support_logprobs is not None
    return torch.zeros_like(logits, dtype=torch.float32).scatter(
        1, captured.token_ids.long(), captured.support_logprobs.exp()
    )


def sample_native_eager_distribution(
    logits: torch.Tensor,
    histories: list[list[int]],
    sampling_state: TalkerSamplingState,
    monkeypatch: pytest.MonkeyPatch,
) -> torch.Tensor:
    count, vocabulary_size = logits.shape
    native_output = LogitsProcessorOutput(
        next_token_logits=slot_state_shaping(logits, histories, sampling_state),
        hidden_states=None,
    )
    slots = TalkerSlotState.allocate(
        count, vocabulary_size, sampling_state.eos_id, logits.device
    )
    slots.temperatures[:count] = sampling_state.temperatures[:, None]
    slots.top_ks[:count] = sampling_state.top_ks.to(slots.top_ks.dtype)
    slots.top_ps[:count] = sampling_state.top_ps
    rows = torch.arange(count, device=logits.device)
    sampling_info = slots.sampling_info(rows)
    sampling_info.sampling_seed = None
    sampling_info.max_top_k = sampling_state.max_top_k
    sampling_info.need_top_k_sampling = sampling_state.need_top_filtering
    sampling_info.need_top_p_sampling = sampling_state.need_top_filtering
    sampling_info.is_any_greedy = sampling_state.need_top_filtering
    original_topk = torch.topk
    observed: list[tuple[torch.Tensor, torch.Tensor]] = []

    def record_topk(
        input: torch.Tensor,
        k: int,
        dim: int = -1,
        largest: bool = True,
        sorted: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = original_topk(input, k, dim=dim, largest=largest, sorted=sorted)
        observed.append(result)
        return result

    with (
        get_context().override_server_args(sampling_backend="pytorch"),
        get_parallel().override(tp_group=SimpleNamespace(device_group=None)),
        monkeypatch.context() as patcher,
    ):
        patcher.setattr(torch, "topk", record_topk)
        sampler = Sampler()
        sampler(
            native_output,
            sampling_info,
            False,
            [0] * count,
            [[] for _ in histories],
            torch.zeros(count, dtype=torch.long, device=logits.device),
        )
    assert native_output.sampling_mask_output is None
    if sampling_state.max_top_k is not None:
        assert len(observed) == 1
        weights, indices = observed[0]
        weights = weights.float()
        return torch.zeros_like(logits, dtype=torch.float32).scatter(
            1, indices.long(), weights / weights.sum(-1, keepdim=True)
        )
    else:
        assert not observed
        weights = native_output.next_token_logits.float()
        return weights / weights.sum(-1, keepdim=True)


def test_shaping_matches_the_talker_slot_state() -> None:
    torch.manual_seed(0)
    histories = [
        torch.randint(0, CODEC_EOS_ID, (length,)).tolist() for length in (0, 2, 16, 23)
    ]
    suppress = torch.zeros(len(histories), VOCAB_SIZE, dtype=torch.bool)
    suppress[1, 4] = True
    state = make_state(histories, suppress=suppress)
    logits = torch.randn(len(histories), VOCAB_SIZE) * 3

    shaped = shape_logits(
        logits,
        suppress=suppress,
        counts=window_counts(state.recent_tokens, VOCAB_SIZE),
        penalties=state.penalties,
        eos_masked=state.generated < state.min_new_tokens,
        eos_id=CODEC_EOS_ID,
    )

    assert torch.equal(shaped, slot_state_shaping(logits, histories, state))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_verify_rows_count_the_drafts_before_them(dtype: torch.dtype) -> None:
    torch.manual_seed(1)
    histories = [
        torch.randint(0, CODEC_EOS_ID, (length,)).tolist() for length in (1, 15, 30)
    ]
    drafts = torch.randint(0, VOCAB_SIZE, (len(histories), 4))
    state = make_state(histories)
    logits = (torch.randn(len(histories) * 5, VOCAB_SIZE) * 2).to(dtype)

    target = target_probabilities(logits, drafts, state)

    for row, history in enumerate(histories):
        for offset in range(5):
            outputs = history + drafts[row, :offset].tolist()
            one = make_state([outputs])
            shaped = slot_state_shaping(
                logits[row * 5 + offset : row * 5 + offset + 1], [outputs], one
            )
            expected = nucleus(
                shaped,
                temperatures=one.temperatures,
                top_ks=one.top_ks,
                top_ps=one.top_ps,
                greedy=one.greedy,
            )
            torch.testing.assert_close(target[row, offset], expected[0])


@pytest.mark.accelerator
@pytest.mark.parametrize("is_all_greedy", [False, True], ids=["mixed", "all-greedy"])
@pytest.mark.parametrize("is_compiled", [False, True], ids=["eager", "compiled"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_target_distribution_matches_the_native_sampler(
    dtype: torch.dtype, is_compiled: bool, is_all_greedy: bool
) -> None:
    """Verification preserves the head-dtype rounding of the plain sampler's distribution."""
    require_cuda()
    torch.manual_seed(21)
    device = torch.device("cuda")
    histories = [[], [3, 5, 3], [2, 4, 2, 7] * 5, [0] * 20]
    state = make_state(
        histories,
        device=device,
        vocabulary_size=SAMPLER_VOCAB_SIZE,
        temperatures=torch.tensor([0.6, 0.8, 1.3, 0.8]),
        top_ks=torch.tensor([2, 25, SAMPLER_VOCAB_SIZE, 1]),
        top_ps=torch.tensor([0.6, 0.85, 1.0, 1.0]),
        penalties=torch.tensor([1.0, 1.05, 1.3, 1.0]),
        greedy=torch.tensor([False, False, False, True]),
    )
    state.suppress[0, 2] = True
    state.suppress[1, 5] = True
    if is_all_greedy:
        state.greedy.fill_(True)
        state.top_ks.fill_(1)
        state.is_all_greedy.fill_(True)
    else:
        pass
    logits = (torch.randn(4, SAMPLER_VOCAB_SIZE, device=device) * 3).to(dtype)
    logits[3].zero_()
    logits[3, SAMPLER_VOCAB_SIZE - 2] = ROUNDING_TIE_LOGIT
    native_distribution = sample_native_distribution(
        logits, histories, state, is_all_greedy=is_all_greedy
    )
    probabilities = (
        torch.compile(
            target_probabilities,
            dynamic=True,
            options={"emulate_precision_casts": True},
        )
        if is_compiled
        else target_probabilities
    )
    actual = probabilities(
        logits, torch.empty(4, 0, dtype=torch.long, device=device), state
    )[:, 0]

    assert torch.equal(actual > 0, native_distribution > 0)
    torch.testing.assert_close(actual, native_distribution)


@pytest.mark.accelerator
@pytest.mark.parametrize("filtered", [True, False], ids=["eager-topk", "unfiltered"])
@pytest.mark.parametrize("is_compiled", [False, True], ids=["eager", "compiled"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_target_distribution_preserves_native_eager_sorting(
    dtype: torch.dtype,
    is_compiled: bool,
    filtered: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_cuda()
    device = torch.device("cuda")
    count, vocabulary_size = 3, SAMPLER_VOCAB_SIZE
    histories = [[] for _ in range(count)]
    state = make_state(
        histories,
        device=device,
        vocabulary_size=vocabulary_size,
        top_ks=torch.tensor([25, 1, 7] if filtered else [vocabulary_size] * count),
        top_ps=torch.ones(count),
        min_new_tokens=torch.zeros(count, dtype=torch.long),
        penalties=torch.ones(count),
        greedy=torch.tensor([False, filtered, False]),
        max_top_k=25 if filtered else None,
        need_top_filtering=filtered,
    )
    logits = torch.zeros(count, vocabulary_size, dtype=dtype, device=device)
    if not filtered:
        logits[:, 4] = 0.125
    else:
        pass
    native = sample_native_eager_distribution(logits, histories, state, monkeypatch)
    probabilities = (
        torch.compile(
            target_probabilities,
            dynamic=True,
            options={"emulate_precision_casts": True},
        )
        if is_compiled
        else target_probabilities
    )

    actual = probabilities(
        logits, torch.empty(count, 0, dtype=torch.long, device=device), state
    )[:, 0]

    assert torch.equal(actual > 0, native > 0)
    torch.testing.assert_close(actual, native)


@pytest.mark.accelerator
@pytest.mark.parametrize("mode", ["fullsort", "eager-topk", "unfiltered"])
@pytest.mark.parametrize("is_compiled", [False, True], ids=["eager", "compiled"])
def test_target_distribution_preserves_native_batches_at_each_verify_position(
    mode: str, is_compiled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    require_cuda()
    device = torch.device("cuda")
    count, gamma, vocabulary_size = 32, 7, 6562
    histories = [[index % 11] * (index % 4) for index in range(count)]
    filtered = mode != "unfiltered"
    settings = dict(
        vocabulary_size=vocabulary_size,
        top_ks=torch.tensor(
            [25, 1, 7, 25] * 8 if filtered else [vocabulary_size] * count
        ),
        top_ps=torch.full((count,), 0.85 if filtered else 1.0),
        min_new_tokens=torch.zeros(count, dtype=torch.long),
        penalties=torch.full((count,), 1.05),
        greedy=torch.tensor(
            [False, True, False, False] * 8 if filtered else [False] * count
        ),
        max_top_k=25 if mode == "eager-topk" else None,
        need_top_filtering=filtered,
    )
    state = make_state(histories, device=device, **settings)
    drafts = (torch.arange(count * gamma, device=device).view(count, gamma) % 17).long()
    logits = torch.zeros(
        count * (gamma + 1), vocabulary_size, dtype=torch.bfloat16, device=device
    )
    logits[:, 0] = 0.125
    expected = []
    for position in range(gamma + 1):
        position_histories = [
            history + drafts[row, :position].cpu().tolist()
            for row, history in enumerate(histories)
        ]
        position_state = make_state(position_histories, device=device, **settings)
        position_logits = logits.view(count, gamma + 1, vocabulary_size)[:, position]
        if mode == "fullsort":
            expected.append(
                sample_native_distribution(
                    position_logits,
                    position_histories,
                    position_state,
                    is_all_greedy=False,
                )
            )
        else:
            expected.append(
                sample_native_eager_distribution(
                    position_logits, position_histories, position_state, monkeypatch
                )
            )
    native = torch.stack(expected, dim=1)
    probabilities = (
        torch.compile(
            target_probabilities,
            dynamic=True,
            options={"emulate_precision_casts": True},
        )
        if is_compiled
        else target_probabilities
    )

    actual = probabilities(logits, drafts, state)

    assert torch.equal(actual > 0, native > 0)
    torch.testing.assert_close(actual, native)


def run_speculative_step(
    trials: int, history: list[int], greedy: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One draft block and verify per trial; returns committed tokens and the target logits."""
    drafts = 3
    state = make_state(
        [history] * trials, greedy=torch.full((trials,), greedy, dtype=torch.bool)
    )
    # Target logits depend on the row and on the token before it, as a model's would.
    position_logits = torch.randn(drafts + 1, VOCAB_SIZE) * 2
    token_logits = torch.randn(VOCAB_SIZE, VOCAB_SIZE)
    draft_base = torch.randn(drafts, VOCAB_SIZE) * 2
    markov_embedding = torch.randn(VOCAB_SIZE, 4) * 0.3
    markov_projection = torch.randn(VOCAB_SIZE, 4) * 0.3
    tokens, shaped = sample_draft_block(
        draft_base.unsqueeze(0).expand(trials, -1, -1),
        torch.full((trials,), history[-1]),
        markov_embedding,
        markov_projection,
        state,
        torch.empty(trials, drafts, VOCAB_SIZE).exponential_(),
    )
    previous = torch.cat([torch.full((trials, 1), history[-1]), tokens], dim=1)
    accepted, bonus = exact_accept(
        (position_logits.unsqueeze(0) + token_logits[previous]).reshape(-1, VOCAB_SIZE),
        shaped,
        tokens,
        state,
        torch.rand(trials, drafts),
        torch.empty(trials, VOCAB_SIZE).exponential_(),
    )
    committed = torch.cat([tokens, torch.zeros(trials, 1, dtype=torch.long)], dim=1)
    committed.scatter_(1, accepted.unsqueeze(1), bonus.unsqueeze(1))
    return committed, accepted, position_logits, token_logits


def talker_distribution(
    position_logits: torch.Tensor,
    token_logits: torch.Tensor,
    outputs: list[int],
    offset: int,
    greedy: bool = False,
) -> torch.Tensor:
    """The talker's next-token distribution after outputs, sampled one step at a time."""
    state = make_state([outputs], greedy=torch.tensor([greedy]))
    logits = (position_logits[offset] + token_logits[outputs[-1]]).unsqueeze(0)
    return target_probabilities(logits, torch.zeros(1, 0, dtype=torch.long), state)[
        0, 0
    ]


def test_proposal_filter_keeps_sampling_and_verify_probabilities_aligned() -> None:
    state = make_state(
        [[3], [5]],
        temperatures=torch.tensor([0.5, 1.0]),
        penalties=torch.ones(2),
        min_new_tokens=torch.zeros(2, dtype=torch.long),
    )
    logits = torch.full((2, 1, VOCAB_SIZE), -100.0)
    logits[:, 0, :4] = torch.tensor([0.0, -1.0, -2.0, -4.0])
    noise = torch.ones_like(logits)
    noise[:, :, 3] = 1e-20
    expected_probabilities = torch.zeros_like(logits)
    expected_probabilities[0, 0, :2] = torch.tensor([1.0, 0.135335283])
    expected_probabilities[1, 0, :3] = torch.tensor([1.0, 0.367879441, 0.135335283])
    expected_probabilities /= expected_probabilities.sum(dim=-1, keepdim=True)

    tokens, proposal_logits = sample_draft_block(
        logits,
        torch.tensor([3, 5]),
        torch.zeros(VOCAB_SIZE, 4),
        torch.zeros(VOCAB_SIZE, 4),
        state,
        noise,
    )
    probabilities = torch.softmax(
        proposal_logits / state.temperatures[:, None, None], dim=-1
    )

    assert torch.equal(tokens, torch.zeros(2, 1, dtype=torch.long))
    torch.testing.assert_close(probabilities, expected_probabilities)


def test_speculative_step_reproduces_talker_sampling() -> None:
    torch.manual_seed(2)
    trials, history = 400_000, [3, 5, 3, 7]
    committed, accepted, position_logits, token_logits = run_speculative_step(
        trials, history, greedy=False
    )

    first = committed[:, 0]
    first_target = talker_distribution(position_logits, token_logits, history, 0)
    first_observed = torch.bincount(first, minlength=VOCAB_SIZE).float() / trials
    first_sigma = (first_target * (1 - first_target) / trials).sqrt().clamp_min(1e-9)
    assert ((first_observed - first_target).abs() / first_sigma).max() < 4.5

    # note (0xtoward): target modes outside the proposal support cannot condition an accepted prefix.
    accepted_first_token = int(
        torch.bincount(first[accepted >= 1], minlength=VOCAB_SIZE).argmax()
    )
    picked = (accepted >= 1) & (first == accepted_first_token)
    assert picked.sum() >= 1000
    second_target = talker_distribution(
        position_logits, token_logits, history + [accepted_first_token], 1
    )
    second_observed = (
        torch.bincount(committed[picked, 1], minlength=VOCAB_SIZE).float()
        / picked.sum()
    )
    second_sigma = (
        (second_target * (1 - second_target) / picked.sum()).sqrt().clamp_min(1e-9)
    )
    assert ((second_observed - second_target).abs() / second_sigma).max() < 4.5


def test_greedy_rows_commit_the_talker_argmax() -> None:
    torch.manual_seed(3)
    history = [2, 9, 2]
    committed, accepted, position_logits, token_logits = run_speculative_step(
        64, history, greedy=True
    )

    outputs = list(history)
    for offset in range(int(accepted[0]) + 1):
        expected = talker_distribution(
            position_logits, token_logits, outputs, offset, greedy=True
        ).argmax()
        assert torch.all(committed[:, offset] == expected)
        outputs.append(int(expected))


class TinyDraft(nn.Module):
    """The draft model surface the folded draw reads: head, Markov head, block layout."""

    sample_from_anchor = False

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.lm_head = nn.Linear(hidden, VOCAB_SIZE, bias=False)
        self.markov_head = VanillaMarkov(vocab_size=VOCAB_SIZE, markov_rank=4)

    def compute_base_logits(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return (hidden @ self.lm_head.weight.T).float(), None


@pytest.mark.parametrize("capacity", [1, 2])
def test_folded_draw_warmup_stays_within_sampling_capacity(capacity: int) -> None:
    """A one-slot deployment failed at startup by warming two rows against one sampling row."""
    worker = object.__new__(MiniCPMOTalkerDSparkWorker)
    worker.gamma = 3
    worker.device = "cpu"
    worker.draft_model = TinyDraft(hidden=8)
    worker.sampling_rows = StaticSamplingRows(
        capacity=capacity,
        vocab=VOCAB_SIZE,
        eos_id=CODEC_EOS_ID,
        device=torch.device("cpu"),
    )
    warmed_batch_sizes = []

    def draw(
        base_logits: torch.Tensor,
        previous_tokens: torch.Tensor,
        markov_embedding: torch.Tensor,
        markov_projection: torch.Tensor,
        state: TalkerSamplingState,
        exponential_noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = sample_draft_block(
            base_logits,
            previous_tokens,
            markov_embedding,
            markov_projection,
            state,
            exponential_noise,
        )
        warmed_batch_sizes.append(result[0].shape[0])
        return result

    worker.warm_up_folded_draw(draw)

    assert warmed_batch_sizes == ([1] if capacity == 1 else [1, 2])


def test_folded_draw_hook_draws_the_talker_shaped_block() -> None:
    torch.manual_seed(5)
    gamma, hidden = 3, 8
    model = TinyDraft(hidden)
    rows = StaticSamplingRows(
        capacity=4, vocab=VOCAB_SIZE, eos_id=CODEC_EOS_ID, device=torch.device("cpu")
    )
    state = make_state([[3, 5, 3], [1], [2, 2, 2, 2]])
    rows.load(state)
    hook = FoldedTalkerDraw(
        model=model,
        gamma=gamma,
        max_bs=4,
        device=torch.device("cpu"),
        rows=rows,
        draw_kernel=sample_draft_block,
    )
    # The draft graph feeds the anchor and gamma mask positions of each request.
    hidden_states = torch.randn(3 * (gamma + 1), hidden)
    input_ids = torch.tensor([[3, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]).view(-1)

    torch.manual_seed(6)
    with torch.no_grad():
        hook(hidden_states, input_ids)
    torch.manual_seed(6)
    with torch.no_grad():
        base_logits = model.compute_base_logits(
            hidden_states.view(3, gamma + 1, hidden)[:, 1:].reshape(-1, hidden)
        )[0].view(3, gamma, -1)
        expected_tokens, expected_shaped = sample_draft_block(
            base_logits,
            torch.tensor([3, 1, 2]),
            model.markov_head.markov_w1.weight,
            model.markov_head.markov_w2.weight,
            state,
            torch.empty_like(base_logits).exponential_(),
        )
    assert torch.equal(hook.out[: 3 * gamma].view(3, gamma), expected_tokens)
    torch.testing.assert_close(
        hook.corrected_out[: 3 * gamma].view(3, gamma, -1), expected_shaped
    )


@pytest.mark.accelerator
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_accept_graphs_replay_the_greedy_verify_on_the_folded_block(
    dtype: torch.dtype,
) -> None:
    require_cuda()
    device = torch.device("cuda")
    torch.manual_seed(4)
    gamma = 3
    markov_head = VanillaMarkov(vocab_size=VOCAB_SIZE, markov_rank=4).to(device)
    rows = StaticSamplingRows(
        capacity=4, vocab=VOCAB_SIZE, eos_id=CODEC_EOS_ID, device=device
    )
    draft_tokens_buffer = torch.zeros(4 * gamma, dtype=torch.long, device=device)
    shaped_buffer = torch.zeros(4 * gamma, VOCAB_SIZE, device=device)
    graphs = ExactAcceptGraphs(
        batch_sizes=[1, 4],
        gamma=gamma,
        rows=rows,
        draft_tokens=draft_tokens_buffer,
        shaped_draft_logits=shaped_buffer,
        target_logits_dtype=dtype,
        fallback=accept_block,
        backend=current_platform.get_device_graph_backend(device),
        device=device,
    )
    cpu_state = make_state(
        [[3, 5, 3], [1], [2, 2, 2, 2]], greedy=torch.ones(3, dtype=torch.bool)
    )
    state = TalkerSamplingState(
        **{
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in vars(cpu_state).items()
        }
    )
    base_logits = torch.randn(3, gamma, VOCAB_SIZE, device=device)
    previous = torch.tensor([3, 1, 2], device=device)
    target_logits = torch.randn(3 * (gamma + 1), VOCAB_SIZE, device=device, dtype=dtype)
    target_logits[0].fill_(-9.0)
    target_logits[0, 1] = 0.58984375
    target_logits[0, 3] = 1.0
    weights = (markov_head.markov_w1.weight, markov_head.markov_w2.weight)
    tokens, shaped = draw_block(base_logits, previous, *weights, state)

    # A folded draw leaves its block in the buffers the graphs read.
    rows.load(state)
    draft_tokens_buffer[: 3 * gamma].copy_(tokens.reshape(-1))
    shaped_buffer[: 3 * gamma].copy_(shaped.reshape(3 * gamma, -1))
    accepted, bonus = graphs.accept(
        target_logits.float(),
        shaped_buffer[: 3 * gamma].view(3, gamma, -1),
        draft_tokens_buffer[: 3 * gamma].view(3, gamma),
        state,
    )
    expected_accepted, expected_bonus = accept_block(
        target_logits, shaped, tokens, state
    )
    assert torch.equal(accepted, expected_accepted)
    assert torch.equal(bonus, expected_bonus)

    # An eager draw's block is copied in first.
    accepted, bonus = graphs.accept(
        target_logits.float(), shaped.clone(), tokens.clone(), state
    )
    assert torch.equal(accepted, expected_accepted)
    assert torch.equal(bonus, expected_bonus)

    # One replay also gives the commit lengths and the output tokens.
    prefix_lens = torch.tensor([40, 7, 19], device=device)
    outs = graphs.accept_and_finalize(
        target_logits.float(), shaped, tokens, prefix_lens
    )
    expected = finalize_accept_lens(
        correct_len=expected_accepted,
        cap_trim_lens=torch.zeros(3, dtype=torch.int32, device=device),
        prefix_lens=prefix_lens,
    )
    assert torch.equal(outs.correct_len, expected_accepted)
    assert torch.equal(outs.bonus, expected_bonus)
    assert torch.equal(outs.commit_lens, expected.commit_lens)
    assert torch.equal(outs.new_seq_lens, expected.new_seq_lens)
    assert torch.equal(
        outs.out_tokens,
        build_out_tokens(
            draft_tokens=tokens,
            correct_len=expected_accepted,
            bonus=expected_bonus,
            verify_num_draft_tokens=gamma + 1,
            gamma=gamma,
        ),
    )

    larger_count = 5
    larger_state = TalkerSamplingState(
        **{
            name: (
                value[:1].expand(larger_count, *value.shape[1:])
                if isinstance(value, torch.Tensor) and value.ndim > 0
                else value
            )
            for name, value in vars(state).items()
        }
    )
    larger_logits = target_logits[: gamma + 1].repeat(larger_count, 1)
    larger_shaped = shaped[:1].expand(larger_count, -1, -1)
    larger_tokens = tokens[:1].expand(larger_count, -1)

    fallback_result = graphs.accept(
        larger_logits.float(), larger_shaped, larger_tokens, larger_state
    )
    expected_result = accept_block(
        larger_logits, larger_shaped, larger_tokens, larger_state
    )

    assert torch.equal(fallback_result[0], expected_result[0])
    assert torch.equal(fallback_result[1], expected_result[1])


@pytest.mark.accelerator
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_accept_graphs_keep_the_native_batch_branch_when_slots_change(
    dtype: torch.dtype,
) -> None:
    """A padded replay follows the real batch's greedy flag after both load and gather."""
    require_cuda()
    torch.manual_seed(7)
    device = torch.device("cuda")
    count, capacity, gamma = 3, 4, 1
    distinct_token = SAMPLER_VOCAB_SIZE - 2
    histories = [[] for _ in range(count)]
    sampling_rows = StaticSamplingRows(
        capacity=capacity,
        vocab=SAMPLER_VOCAB_SIZE,
        eos_id=SAMPLER_VOCAB_SIZE - 1,
        device=device,
    )
    draft_buffer = torch.full(
        (capacity * gamma,), distinct_token, dtype=torch.long, device=device
    )
    proposal_buffer = torch.full(
        (capacity * gamma, SAMPLER_VOCAB_SIZE), float("-inf"), device=device
    )
    proposal_buffer[:, distinct_token] = 0
    graphs = ExactAcceptGraphs(
        batch_sizes=[capacity],
        gamma=gamma,
        rows=sampling_rows,
        draft_tokens=draft_buffer,
        shaped_draft_logits=proposal_buffer,
        target_logits_dtype=dtype,
        fallback=accept_block,
        backend=current_platform.get_device_graph_backend(device),
        device=device,
    )
    logits = torch.zeros(count, SAMPLER_VOCAB_SIZE, dtype=dtype, device=device)
    logits[:, distinct_token] = ROUNDING_TIE_LOGIT
    target_logits = logits.repeat_interleave(gamma + 1, dim=0)
    slots = TalkerSlotState.allocate(
        capacity, SAMPLER_VOCAB_SIZE, SAMPLER_VOCAB_SIZE - 1, device
    )
    slot_order = torch.tensor([2, 0, 3], device=device)

    for is_all_greedy in (False, True, False):
        state = make_state(
            histories,
            device=device,
            vocabulary_size=SAMPLER_VOCAB_SIZE,
            min_new_tokens=torch.zeros(count, dtype=torch.long),
            penalties=torch.ones(count),
            temperatures=torch.full((count,), 0.8),
            top_ks=torch.tensor([1, 1, 1] if is_all_greedy else [25, 1, 25]),
            top_ps=torch.tensor([1.0, 1.0, 1.0]),
            greedy=torch.tensor(
                [True, True, True] if is_all_greedy else [False, True, False]
            ),
        )
        if is_all_greedy:
            slots.temperatures[slot_order] = state.temperatures[:, None]
            slots.top_ks[slot_order] = state.top_ks.to(slots.top_ks.dtype)
            slots.top_ps[slot_order] = state.top_ps
            sampling_rows.gather(
                slots,
                slot_order,
                is_all_greedy=is_all_greedy,
                max_top_k=None,
                need_top_filtering=True,
            )
        else:
            sampling_rows.load(state)
        native_distribution = sample_native_distribution(
            logits, histories, state, is_all_greedy=is_all_greedy
        )
        native_tokens = native_distribution.argmax(dim=-1)
        result = graphs.accept_and_finalize(
            target_logits,
            proposal_buffer[:count].view(count, gamma, -1),
            draft_buffer[:count].view(count, gamma),
            torch.zeros(count, dtype=torch.long, device=device),
        )

        assert result is not None
        assert torch.equal(result.bonus[1:2], native_tokens[1:2])
        assert torch.equal(
            result.correct_len[1:2], (native_tokens[1:2] == distinct_token).long()
        )


@pytest.mark.accelerator
def test_accept_graphs_follow_loaded_native_sorting_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_cuda()
    device = torch.device("cuda")
    count, capacity, gamma, vocab = 3, 4, 1, SAMPLER_VOCAB_SIZE
    histories = [[] for _ in range(count)]
    distinct_token = vocab - 2
    rows = StaticSamplingRows(
        capacity=capacity, vocab=vocab, eos_id=vocab - 1, device=device
    )
    tokens = torch.full((capacity,), distinct_token, dtype=torch.long, device=device)
    proposal = torch.full((capacity, vocab), float("-inf"), device=device)
    proposal[:, distinct_token] = 0.0
    graphs = ExactAcceptGraphs(
        batch_sizes=[capacity],
        gamma=gamma,
        rows=rows,
        draft_tokens=tokens,
        shaped_draft_logits=proposal,
        target_logits_dtype=torch.bfloat16,
        fallback=accept_block,
        backend=current_platform.get_device_graph_backend(device),
        device=device,
    )
    slots = TalkerSlotState.allocate(capacity, vocab, vocab - 1, device)
    slot_order = torch.tensor([2, 0, 3], device=device)

    for index, mode in enumerate(["fullsort", "eager-topk", "unfiltered", "fullsort"]):
        filtered = mode != "unfiltered"
        state = make_state(
            histories,
            device=device,
            vocabulary_size=vocab,
            top_ks=torch.tensor([25, 1, 7] if filtered else [vocab] * count),
            top_ps=torch.ones(count),
            penalties=torch.ones(count),
            min_new_tokens=torch.zeros(count, dtype=torch.long),
            greedy=torch.tensor([False, filtered, False]),
            max_top_k=25 if mode == "eager-topk" else None,
            need_top_filtering=filtered,
        )
        if index % 2 == 0:
            rows.load(state)
        else:
            slots.temperatures[slot_order] = state.temperatures[:, None]
            slots.top_ks[slot_order] = state.top_ks.to(slots.top_ks.dtype)
            slots.top_ps[slot_order] = state.top_ps
            rows.gather(
                slots,
                slot_order,
                is_all_greedy=False,
                max_top_k=state.max_top_k,
                need_top_filtering=state.need_top_filtering,
            )
        logits = torch.zeros(count, vocab, dtype=torch.bfloat16, device=device)
        logits[:, distinct_token] = ROUNDING_TIE_LOGIT
        if mode == "unfiltered":
            logits[1].fill_(float("-inf"))
            logits[1, distinct_token] = 0.0
        else:
            pass
        if mode == "fullsort":
            native = sample_native_distribution(
                logits, histories, state, is_all_greedy=False
            )
        else:
            native = sample_native_eager_distribution(
                logits, histories, state, monkeypatch
            )
        target_logits = logits.repeat_interleave(gamma + 1, dim=0)

        result = graphs.accept_and_finalize(
            target_logits,
            proposal[:count].view(count, gamma, -1),
            tokens[:count].view(count, gamma),
            torch.zeros(count, dtype=torch.long, device=device),
        )
        if mode == "fullsort":
            assert result is not None
            correct, bonus = result.correct_len, result.bonus
        else:
            assert (
                result is None
            ), "a captured fullsort graph ignored the current native sampling mode"
            correct, bonus = graphs.accept(
                target_logits,
                proposal[:count].view(count, gamma, -1),
                tokens[:count].view(count, gamma),
                state,
            )

        assert rows.rows(count).max_top_k == state.max_top_k
        assert rows.rows(count).need_top_filtering == state.need_top_filtering
        expected = native[1].argmax()
        assert torch.equal(bonus[1], expected)
        assert torch.equal(correct[1], (expected == distinct_token).long())


@pytest.mark.accelerator
def test_kv_injection_graphs_write_only_the_committed_rows() -> None:
    require_cuda()
    device = torch.device("cuda")
    rows_per_request, width, trash = 3, 4, 99
    cache = torch.zeros(100, device=device)

    def inject(*, target_hidden, cache_loc, positions, cache_loc_2d, commit_lens):
        # A stand-in for the draft's KV write: row value = hidden sum + position,
        # rows past the commit length land in a slot nobody reads.
        del cache_loc
        size = cache_loc_2d.shape[0]
        keep = torch.arange(rows_per_request, device=device) < commit_lens.view(-1, 1)
        locs = torch.where(keep, cache_loc_2d, torch.full_like(cache_loc_2d, trash))
        values = target_hidden.float().sum(-1) + positions.float()
        cache.scatter_(0, locs.reshape(-1), values.view(size * rows_per_request))

    graphs = DraftKvInjectionGraphs(
        batch_sizes=[1, 4],
        rows_per_request=rows_per_request,
        hidden_width=width,
        hidden_dtype=torch.bfloat16,
        inject=inject,
        backend=current_platform.get_device_graph_backend(device),
        device=device,
    )
    cache.zero_()
    hidden = torch.ones(
        2 * rows_per_request, width, dtype=torch.bfloat16, device=device
    )
    cache_loc_2d = torch.tensor([[10, 11, 12], [20, 21, 22]], device=device)
    positions = torch.tensor([5, 6, 7, 30, 31, 32], device=device)
    assert graphs.write(
        hidden=hidden,
        cache_loc_2d=cache_loc_2d,
        positions=positions,
        commit_lens=torch.tensor([2, 3], dtype=torch.int32, device=device),
    )
    written = {int(i): float(cache[i]) for i in torch.nonzero(cache).view(-1).tolist()}
    trash_value = written.pop(trash, None)
    assert written == {10: 9.0, 11: 10.0, 20: 34.0, 21: 35.0, 22: 36.0}
    assert trash_value is not None


def test_slot_state_gives_the_ordered_window_and_takes_committed_blocks() -> None:
    histories = [[3, 5, 3], list(range(10)) * 2, []]
    slots = TalkerSlotState.allocate(4, VOCAB_SIZE, CODEC_EOS_ID, torch.device("cpu"))
    rows = torch.tensor([2, 0, 3])
    for row, history in zip(rows.tolist(), histories):
        for token in history:
            slots.append(torch.tensor([row]), torch.tensor([token]))
    slots.min_new_tokens[rows] = torch.tensor([6, 6, 6])
    slots.penalties[rows] = torch.tensor([1.3, 1.3, 1.3])
    slots.top_ks[rows] = torch.tensor([6, 6, 1], dtype=torch.int32)

    state = state_from_slots(
        slots, rows, is_all_greedy=False, max_top_k=None, need_top_filtering=True
    )
    expected = make_state(histories)
    assert torch.equal(state.recent_tokens, expected.recent_tokens)
    assert torch.equal(state.generated, expected.generated)
    assert state.greedy.tolist() == [False, False, True]

    # A committed block appends its first counts tokens, as the talker would one by one.
    committed = torch.tensor([[7, 8, 9, 10], [1, 2, 3, 4], [5, 6, 7, 8]])
    counts = torch.tensor([2, 4, 1])
    commit_to_slots(slots, rows, committed, counts)
    after = [
        history + committed[i, : counts[i]].tolist()
        for i, history in enumerate(histories)
    ]
    state = state_from_slots(
        slots, rows, is_all_greedy=False, max_top_k=None, need_top_filtering=True
    )
    assert torch.equal(state.recent_tokens, make_state(after).recent_tokens)
    assert state.generated.tolist() == [len(history) for history in after]


def make_filled_slots(
    vocab: int, device: torch.device, generator: torch.Generator
) -> TalkerSlotState:
    """Slots whose windows have wrapped for some rows, with mixed sampling parameters."""
    slots = TalkerSlotState.allocate(8, vocab, vocab - 1, device)
    for row, length in enumerate([0, 3, 15, 16, 17, 31, 40, 9]):
        tokens = torch.randint(vocab, (length,), generator=generator).tolist()
        for token in tokens:
            slots.append(
                torch.tensor([row], device=device), torch.tensor([token], device=device)
            )
    rows = slots.generated.shape[0]
    slots.min_new_tokens.copy_(torch.randint(10, (rows,), generator=generator))
    slots.penalties.copy_(1 + torch.rand(rows, generator=generator))
    slots.temperatures.copy_(torch.rand(rows, 1, generator=generator) + 0.1)
    slots.top_ks.copy_(
        torch.randint(1, 4, (rows,), generator=generator).to(torch.int32)
    )
    slots.top_ps.copy_(torch.rand(rows, generator=generator))
    slots.suppress.copy_(torch.rand(rows, vocab, generator=generator) < 0.3)
    return slots


@pytest.mark.parametrize("vocab", [VOCAB_SIZE, 2500])
@pytest.mark.accelerator
def test_the_slot_gather_kernel_loads_what_state_from_slots_returns(vocab: int) -> None:
    require_cuda()
    device = torch.device("cuda")
    slots = make_filled_slots(vocab, device, torch.Generator().manual_seed(vocab))
    rows = torch.tensor([5, 1, 7, 3, 0], device=device)
    static = StaticSamplingRows(
        capacity=8, vocab=vocab, eos_id=vocab - 1, device=device
    )

    expected = state_from_slots(
        slots, rows, is_all_greedy=False, max_top_k=None, need_top_filtering=True
    )
    loaded = static.gather(
        slots, rows, is_all_greedy=False, max_top_k=None, need_top_filtering=True
    )

    assert static.loaded_count == len(rows), "the gather marks its rows as loaded"
    for name in (
        "recent_tokens",
        "generated",
        "min_new_tokens",
        "penalties",
        "suppress",
        "temperatures",
        "top_ks",
        "top_ps",
        "greedy",
        "is_all_greedy",
    ):
        assert torch.equal(
            getattr(loaded, name), getattr(expected, name)
        ), f"{name} bit-exact"


@pytest.mark.accelerator
def test_the_slot_commit_kernel_appends_what_commit_to_slots_appends() -> None:
    require_cuda()
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(7)
    expected = make_filled_slots(VOCAB_SIZE, device, generator)
    committed_slots = make_filled_slots(
        VOCAB_SIZE, device, torch.Generator().manual_seed(7)
    )
    rows = torch.tensor([6, 2, 4, 5], device=device)
    committed = torch.randint(VOCAB_SIZE, (4, 8), generator=generator).to(device)
    counts = torch.tensor([0, 8, 3, 1], dtype=torch.int32, device=device)

    commit_to_slots(expected, rows, committed, counts)
    commit_slot_rows(committed_slots, rows, committed, counts)

    assert torch.equal(committed_slots.windows, expected.windows), "windows bit-exact"
    assert torch.equal(
        committed_slots.generated, expected.generated
    ), "output counts bit-exact"
