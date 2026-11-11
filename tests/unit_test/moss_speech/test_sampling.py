# SPDX-License-Identifier: Apache-2.0
"""Independent sampling oracles, modality transitions, and request histories."""

import pytest
import torch
from transformers import (
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from sglang_omni.models.moss_speech.fsm import (
    AUDIO_PAD,
    AUDIO_VOCAB,
    EOSP,
    IM_END,
    MODALITY_PAD,
    MODE_AUDIO,
    MODE_TEXT,
    SOSP,
    TEXT_ENDOFTEXT,
    TEXT_VOCAB,
    Mode,
    MossSamplingParams,
    apply_warpers,
    channel_histories,
    finalize_row,
    initial_mode,
    next_mode,
    repetition_penalty_scores,
    row_selected_token,
    sample_row,
)
from sglang_omni.models.moss_speech.model_runner import MossSpeechModelRunner
from sglang_omni.models.moss_speech.request_builders import (
    MossSpeechSGLangRequestData,
    MossSpeechState,
    build_result,
    build_sglang_moss_request,
    extract_output_codes,
    preprocess_request,
)
from sglang_omni.models.moss_speech.sglang_model import MossSpeechModelOutput
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.types import SchedulerRequest
from tests.unit_test.fixtures.accelerator import require_cuda
from tests.unit_test.fixtures.moss_speech import TextTokenizer


@pytest.mark.parametrize(
    "scores,top_p,top_k,temperature",
    [
        ([1.0, 1.0, 0.0, -1.0], 0.3, -1, 1.0),
        ([0.0, 0.0, 0.0, 0.0], 0.3, -1, 1.0),
        ([0.0, 0.0, 0.0, 0.0], 0.5, -1, 1.0),
        ([0.0, 0.0, 0.0, 0.0], 0.500001, -1, 1.0),
        ([4.0, 2.0, 2.0, -1.0], 0.8, 2, 0.7),
        ([1.0, 1.0, 0.0, -1.0], 1.0, -1, 1.0),
        ([4.0, 2.0, 2.0, -1.0], 0.99, 1, 2.0),
    ],
)
def test_warpers_match_transformers_at_ties_and_boundaries(
    scores: list[float], top_p: float, top_k: int, temperature: float
) -> None:
    """Tied logits previously retained a different nucleus from Transformers."""
    logits = torch.tensor(scores, dtype=torch.float32)
    token_ids = torch.zeros((1, 1), dtype=torch.long)
    expected = logits[None].clone()
    if temperature != 1.0:
        expected = TemperatureLogitsWarper(temperature)(token_ids, expected)
    else:
        pass
    if top_k > 0:
        expected = TopKLogitsWarper(top_k)(token_ids, expected)
    else:
        pass
    if top_p < 1.0:
        expected = TopPLogitsWarper(top_p)(token_ids, expected)
    else:
        pass
    parameters = MossSamplingParams(
        do_sample=True, top_p=top_p, top_k=top_k, temperature=temperature
    )
    torch.testing.assert_close(
        apply_warpers(logits, parameters), expected[0], rtol=0, atol=0
    )


@pytest.mark.parametrize("penalty", [1.0, 1.7])
def test_repetition_penalty_matches_transformers_without_changing_input(
    penalty: float,
) -> None:
    logits = torch.tensor([2.0, -2.0, 1.0, -0.5, 0.0])
    history = torch.tensor([0, 1, 1, 4])
    original = logits.clone()
    expected = RepetitionPenaltyLogitsProcessor(penalty)(
        history[None], logits[None].clone()
    )[0]
    torch.testing.assert_close(
        repetition_penalty_scores(logits, history, penalty), expected, rtol=0, atol=0
    )
    assert torch.equal(logits, original)


@pytest.mark.parametrize(
    "prompt_row,current_mode,last_row,expected_mode",
    [
        ((MODALITY_PAD, AUDIO_PAD), MODE_TEXT, (SOSP, 7), MODE_AUDIO),
        ((MODALITY_PAD, 7), MODE_AUDIO, (10, EOSP), MODE_TEXT),
        ((SOSP, AUDIO_PAD), MODE_TEXT, (SOSP, EOSP), MODE_TEXT),
        ((12, AUDIO_PAD), MODE_TEXT, (10, 7), MODE_TEXT),
        ((MODALITY_PAD, 7), MODE_AUDIO, (SOSP, 7), MODE_AUDIO),
    ],
)
def test_modality_transitions_keep_audio_end_precedence(
    prompt_row: tuple[int, int],
    current_mode: Mode,
    last_row: tuple[int, int],
    expected_mode: Mode,
) -> None:
    """A row containing both SOSP and EOSP must finish in text mode."""
    assert initial_mode(torch.tensor([prompt_row])) == current_mode
    assert next_mode(last_row, current_mode) == expected_mode
    finalized = finalize_row(42, 7, expected_mode)
    assert finalized == ((MODALITY_PAD, 7) if expected_mode == MODE_AUDIO else (42, 7))
    assert row_selected_token(finalized) == (7 if expected_mode == MODE_AUDIO else 42)


def test_greedy_sampling_bypasses_warpers_and_penalizes_each_channel() -> None:
    text_logits = torch.tensor([5.0, 4.9, 4.9, -4.0])
    audio_logits = torch.full((AUDIO_VOCAB,), -10.0)
    audio_logits[3], audio_logits[4], audio_logits[16390] = 5.0, 4.9, 100.0
    original_text, original_audio = text_logits.clone(), audio_logits.clone()
    parameters = MossSamplingParams(
        do_sample=False, temperature=0.0, top_p=0.01, top_k=1, repetition_penalty=2.0
    )
    assert sample_row(
        text_logits, audio_logits, parameters, torch.tensor([0]), torch.tensor([3])
    ) == (1, 4)
    assert torch.equal(text_logits, original_text)
    assert torch.equal(audio_logits, original_audio)


def test_sample_row_uses_text_then_audio_rng_draw_order() -> None:
    text_logits = torch.tensor([1.0, -1.0, 0.5, 0.1])
    audio_logits = torch.tensor([0.0, 0.2, 0.3, -0.1])
    parameters = MossSamplingParams(do_sample=True, top_p=1.0)
    generator = torch.Generator().manual_seed(97)
    expected_generator = torch.Generator().manual_seed(97)
    expected = (
        int(
            torch.multinomial(text_logits.softmax(-1), 1, generator=expected_generator)
        ),
        int(
            torch.multinomial(audio_logits.softmax(-1), 1, generator=expected_generator)
        ),
    )
    row = sample_row(
        text_logits,
        audio_logits,
        parameters,
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        generator,
    )
    assert row == expected
    assert torch.equal(generator.get_state(), expected_generator.get_state())


def test_request_rng_history_and_retry_are_independent_of_other_requests() -> None:
    prompt = torch.tensor([[1, AUDIO_PAD], [SOSP, AUDIO_PAD]])
    first = MossSpeechSGLangRequestData(prompt_rows=prompt, effective_seed=123)
    unrelated = MossSpeechSGLangRequestData(
        prompt_rows=prompt.clone(), effective_seed=456
    )
    parameters = MossSamplingParams(do_sample=True)
    text_logits, audio_logits = torch.zeros(8), torch.zeros(8)
    caller_rng = torch.get_rng_state().clone()
    expected_generator = torch.Generator().manual_seed(123)
    expected_rows = [
        sample_row(
            text_logits,
            audio_logits,
            parameters,
            torch.tensor([1]),
            torch.tensor([AUDIO_PAD]),
            expected_generator,
        )
        for _ in range(3)
    ]
    for _ in range(3):
        for request in (unrelated, first):
            histories = channel_histories(request.prompt_rows, request.output_rows)
            request.output_rows.append(
                sample_row(
                    text_logits,
                    audio_logits,
                    parameters,
                    *histories,
                    MossSpeechModelRunner.generator_for(request, torch.device("cpu")),
                )
            )
    assert first.output_rows == expected_rows
    text_history, audio_history = channel_histories(
        first.prompt_rows, first.output_rows
    )
    assert text_history.tolist() == [1, SOSP] + [row[0] for row in expected_rows]
    assert audio_history.tolist() == [AUDIO_PAD, AUDIO_PAD] + [
        row[1] for row in expected_rows
    ]
    first = MossSpeechSGLangRequestData(prompt_rows=prompt, effective_seed=123)
    restarted = MossSpeechModelRunner.generator_for(first, torch.device("cpu"))
    assert (
        sample_row(
            text_logits,
            audio_logits,
            parameters,
            torch.tensor([1]),
            torch.tensor([AUDIO_PAD]),
            restarted,
        )
        == expected_rows[0]
    )
    assert torch.equal(caller_rng, torch.get_rng_state())


def assert_request_rng_preserves_torch_seed(device: torch.device, seed: int) -> None:
    request = MossSpeechSGLangRequestData(
        prompt_rows=torch.tensor([[1, AUDIO_PAD]]), effective_seed=seed
    )
    expected = torch.Generator(device=device).manual_seed(seed)
    actual = MossSpeechModelRunner.generator_for(request, device)
    assert actual.initial_seed() == expected.initial_seed()
    torch.testing.assert_close(
        torch.rand(16, generator=actual, device=device),
        torch.rand(16, generator=expected, device=device),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("seed", [-1, 2**63 + 7])
def test_request_rng_accepts_the_full_torch_seed_range(seed: int) -> None:
    """The accepted 64-bit seed was truncated before reaching the request generator."""
    assert_request_rng_preserves_torch_seed(torch.device("cpu"), seed)


@pytest.mark.accelerator
@pytest.mark.parametrize("seed", [-1, 2**63 + 7])
def test_cuda_request_rng_accepts_the_full_torch_seed_range(seed: int) -> None:
    require_cuda()
    assert_request_rng_preserves_torch_seed(torch.device("cuda", 0), seed)


@pytest.mark.parametrize(
    "natural_audio_end,expected_codes", [(True, 511), (False, 512)]
)
def test_terminal_audio_preserves_last_valid_code_at_length_boundary(
    natural_audio_end: bool, expected_codes: int
) -> None:
    terminal_code = EOSP if natural_audio_end else 123
    rows = [(MODALITY_PAD, index % 100) for index in range(511)] + [
        (MODALITY_PAD, terminal_code)
    ]
    codes = extract_output_codes(rows)
    assert len(codes) == expected_codes
    assert codes[-1] == (10 if natural_audio_end else 123)


@pytest.mark.parametrize(
    "terminal_token,termination",
    [(IM_END, "stop"), (TEXT_ENDOFTEXT, "stop"), (8, "length"), (EOSP, "length")],
)
def test_sampled_tokens_use_engine_termination_state(
    terminal_token: int, termination: str
) -> None:
    payload = StagePayload(
        "termination",
        OmniRequest(
            inputs=[{"role": "user", "content": "Hello"}],
            params={"max_new_tokens": 4 if termination == "stop" else 2},
        ),
        {},
    )
    native = build_sglang_moss_request(preprocess_request(payload, TextTokenizer()))
    request = SchedulerRequest(request_id=payload.request_id, data=native)
    runner = object.__new__(MossSpeechModelRunner)
    for index, token in enumerate((7, terminal_token)):
        text_logits = torch.full((1, TEXT_VOCAB), -10.0)
        text_logits[0, token] = 10.0
        output = MossSpeechModelOutput(
            next_token_logits=text_logits,
            text_logits=text_logits,
            audio_logits=torch.zeros(1, AUDIO_VOCAB),
        )
        selected = runner.sample_next_token_ids(output, None, None, [request])
        native.req.output_ids.append(int(selected[0]))
        native.req.update_finish_state()
        assert native.req.finished() is (index == 1)
    native.finish_reason = native.req.finished_reason.to_json()["type"]
    result = MossSpeechState.from_dict(build_result(native).data)
    assert result.finish_reason == termination
    assert result.completion_token_count == 2
