# SPDX-License-Identifier: Apache-2.0
"""Public talker contract: condition embeddings match the reference math."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from torch import nn

from sglang_omni.models.minicpm_o.components.sglang_talker import (
    MiniCPMOTalkerForCausalLM,
    MiniCPMTTSProjector,
)
from sglang_omni.models.minicpm_o.talker_model_runner import MiniCPMOTalkerModelRunner
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import SchedulerRequest

HIDDEN = 8
LLM_DIM = 16
NUM_TEXT = 20
TEXT_EOS = 5
AUDIO_BOS = 6
PENALTY_DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.accelerator,
            pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
        ],
    ),
]


def _bare_model() -> MiniCPMOTalkerForCausalLM:
    model = object.__new__(MiniCPMOTalkerForCausalLM)
    nn.Module.__init__(model)
    model.text_eos_token_id = TEXT_EOS
    model.audio_bos_token_id = AUDIO_BOS
    model.normalize_projected_hidden = True
    model.emb_text = nn.Embedding(NUM_TEXT, HIDDEN)
    model.projector_semantic = MiniCPMTTSProjector(LLM_DIM, HIDDEN)
    return model


def test_condition_matches_reference_math():
    model = _bare_model()
    tokens = torch.tensor([3, 7, 1], dtype=torch.long)
    hidden = torch.randn(3, LLM_DIM)

    condition = model.build_condition_embeddings(tokens, hidden)

    # reference: emb_text(t) + l2norm(projector(h)), then [text_eos, audio_bos]
    ref = model.emb_text(tokens) + F.normalize(
        model.projector_semantic(hidden), p=2, dim=-1
    )
    boundary = model.emb_text(torch.tensor([TEXT_EOS, AUDIO_BOS]))
    torch.testing.assert_close(condition, torch.cat([ref, boundary], dim=0))
    assert condition.shape == (5, HIDDEN)


def test_condition_empty_span_is_boundary_only():
    model = _bare_model()
    condition = model.build_condition_embeddings(
        torch.empty(0, dtype=torch.long), torch.empty(0, LLM_DIM)
    )
    boundary = model.emb_text(torch.tensor([TEXT_EOS, AUDIO_BOS]))
    torch.testing.assert_close(condition, boundary)


def test_condition_length_mismatch_raises():
    model = _bare_model()
    with pytest.raises(ValueError, match="length mismatch"):
        model.build_condition_embeddings(torch.tensor([1, 2]), torch.randn(3, LLM_DIM))


@dataclass(kw_only=True)
class CodecHistory:
    output_ids: list[int]


def make_penalty_request(
    history: list[int], penalty: float, request_id: str = "codec"
) -> SchedulerRequest:
    return SchedulerRequest(
        request_id=request_id,
        data=SGLangARRequestData(
            req=CodecHistory(output_ids=history.copy()),
            talker_model_inputs={"rep_penalty": penalty},
        ),
    )


def reference_window_penalty(
    logits: torch.Tensor, histories: list[list[int]], penalties: list[float]
) -> torch.Tensor:
    counts = torch.zeros_like(logits, dtype=torch.float32)
    for row_index, history in enumerate(histories):
        for token_id in history[-16:]:
            if 0 <= token_id < logits.shape[1]:
                counts[row_index, token_id] += 1
    scaling_factors = (
        torch.tensor(penalties, dtype=torch.float32, device=logits.device)
        .unsqueeze(1)
        .pow(counts)
    )
    scores = logits.float()
    penalized = torch.where(
        scores < 0, scores * scaling_factors, scores / scaling_factors
    )
    return torch.where(counts > 0, penalized, scores).to(logits.dtype)


@pytest.mark.parametrize("device", PENALTY_DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("penalty", [1.0, 0.8, 1.05, 1.1, 2.0])
def test_window_penalty_matches_reference(
    device: str, dtype: torch.dtype, penalty: float
) -> None:
    histories = [
        [],
        [2] * 16,
        [7] + [1, 3] * 8,
        [0, 1, 3, 4, 5, 6, 7],
        [-1, 8, 3, 3, 4],
        [2] + [-1] * 16,
        [1, 2, 2],
    ]
    penalties = [penalty] * 6 + [1.0]
    requests = [
        make_penalty_request(history, row_penalty, str(row_index))
        for row_index, (history, row_penalty) in enumerate(zip(histories, penalties))
    ]
    original = torch.tensor(
        [0.0, -0.0, -1.5, 2.5, -float("inf"), float("inf"), 0.125, -0.25],
        dtype=dtype,
        device=device,
    ).repeat(len(requests), 1)
    expected = reference_window_penalty(original, histories, penalties)
    output = LogitsProcessorOutput(next_token_logits=original.clone())
    runner = MiniCPMOTalkerModelRunner.__new__(MiniCPMOTalkerModelRunner)
    runner.process_sampling_logits(output, requests)
    torch.testing.assert_close(output.next_token_logits, expected, rtol=0, atol=0)
    assert torch.equal(torch.signbit(output.next_token_logits), torch.signbit(expected))
    assert [request.data.req.output_ids for request in requests] == histories


@pytest.mark.parametrize("device", PENALTY_DEVICES)
def test_window_penalty_uses_current_request_history(device: str) -> None:
    runner = MiniCPMOTalkerModelRunner.__new__(MiniCPMOTalkerModelRunner)
    first = make_penalty_request([1] * 16, 1.05, "first")
    second = make_penalty_request([2, 2, 3], 0.8, "second")
    for requests in ([first, second], [second], [second, first], [first]):
        histories = [request.data.req.output_ids.copy() for request in requests]
        penalties = [
            request.data.talker_model_inputs["rep_penalty"] for request in requests
        ]
        original = torch.linspace(-2, 2, 8, device=device).repeat(len(requests), 1)
        expected = reference_window_penalty(original, histories, penalties)
        output = LogitsProcessorOutput(next_token_logits=original.clone())
        runner.process_sampling_logits(output, requests)
        torch.testing.assert_close(output.next_token_logits, expected, rtol=0, atol=0)
        first.data.req.output_ids[:] = [4, 5, 4]
        second.data.req.output_ids.append(6)
        second.data.talker_model_inputs["rep_penalty"] = 1.1


@pytest.mark.parametrize("shape", [(0, 8), (1, 8), (8,), (1, 1, 8)])
def test_window_penalty_empty_or_unhandled_shape(shape: tuple[int, ...]) -> None:
    runner = MiniCPMOTalkerModelRunner.__new__(MiniCPMOTalkerModelRunner)
    original = torch.zeros(shape)
    output = LogitsProcessorOutput(next_token_logits=original)
    runner.process_sampling_logits(output, [])
    assert output.next_token_logits is original
    assert torch.count_nonzero(original) == 0


def test_window_penalty_missing_logits() -> None:
    runner = MiniCPMOTalkerModelRunner.__new__(MiniCPMOTalkerModelRunner)
    output = LogitsProcessorOutput(next_token_logits=None)
    runner.process_sampling_logits(output, [])
    assert output.next_token_logits is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("vocabulary_size", [8, 127, 1024, 6562])
@pytest.mark.parametrize("device", PENALTY_DEVICES)
def test_window_penalty_random_noncontiguous(
    dtype: torch.dtype, vocabulary_size: int, device: str
) -> None:
    generator = torch.Generator().manual_seed(2284)
    runner = MiniCPMOTalkerModelRunner.__new__(MiniCPMOTalkerModelRunner)
    for history_length in (1, 15, 16, 17, 65):
        histories = torch.randint(
            -2, vocabulary_size + 2, (7, history_length), generator=generator
        ).tolist()
        penalties = [1.0, 1.05, 1.1, 0.8, 2.0, 1.00000001, 1.9]
        requests = [
            make_penalty_request(history, penalty, str(row_index))
            for row_index, (history, penalty) in enumerate(zip(histories, penalties))
        ]
        backing = torch.randn(7, vocabulary_size * 2, generator=generator).to(
            device=device, dtype=dtype
        )
        untouched = backing[:, 1::2].clone()
        logits = backing[:, ::2]
        expected = reference_window_penalty(logits, histories, penalties)
        output = LogitsProcessorOutput(next_token_logits=logits)
        runner.process_sampling_logits(output, requests)
        assert output.next_token_logits is logits
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)
        torch.testing.assert_close(backing[:, 1::2], untouched, rtol=0, atol=0)


@pytest.mark.parametrize("device", PENALTY_DEVICES)
def test_window_penalty_preserves_unselected_nan_and_mask(device: str) -> None:
    original = torch.tensor([[float("nan"), -float("inf"), 0.5, -0.5]], device=device)
    requests = [make_penalty_request([1, 2, 2, 3], 1.05)]
    expected = reference_window_penalty(original, [[1, 2, 2, 3]], [1.05])
    output = LogitsProcessorOutput(next_token_logits=original.clone())
    runner = MiniCPMOTalkerModelRunner.__new__(MiniCPMOTalkerModelRunner)
    runner.process_sampling_logits(output, requests)
    torch.testing.assert_close(
        output.next_token_logits, expected, rtol=0, atol=0, equal_nan=True
    )
