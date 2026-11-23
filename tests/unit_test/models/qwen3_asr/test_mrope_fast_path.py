# SPDX-License-Identifier: Apache-2.0
"""Byte-identity tests for the ASR decode and speculative mrope fast paths."""

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

from sglang_omni.models.qwen3_asr import mrope_fast_path

RUNNER = SimpleNamespace(device="cpu")
FAST = mrope_fast_path.fast_compute_mrope_positions
ORIG = mrope_fast_path.ORIGINAL_COMPUTE_MROPE_POSITIONS
SPEC_FAST = mrope_fast_path.fast_compute_spec_mrope_positions
SPEC_ORIG = mrope_fast_path.ORIGINAL_COMPUTE_SPEC_MROPE_POSITIONS


@pytest.fixture(autouse=True)
def upstream(monkeypatch):
    monkeypatch.setattr(
        "sglang.srt.model_executor.forward_batch_info.get_exec",
        lambda: SimpleNamespace(
            deterministic=SimpleNamespace(rl_on_policy_target=None)
        ),
    )


def mm_input(prompt_len: int, *, delta: int = 0, degenerate: bool = True):
    """Mirror what request_builders constructs for a Qwen3-ASR request."""
    positions = torch.arange(prompt_len, dtype=torch.long)
    mm = SimpleNamespace(
        mrope_positions=positions.unsqueeze(0).expand(3, -1).clone(),
        mrope_position_delta=torch.tensor([[delta]], dtype=torch.long),
        mrope_position_delta_repeated_cache=None,
    )
    if degenerate:
        setattr(mm, mrope_fast_path.DEGENERATE_MROPE_FLAG, True)
    return mm


def run(fn, mode, seq_lens, mm_inputs, **batch_fields):
    fb = object.__new__(ForwardBatch)
    fb.forward_mode = mode
    fb.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
    fb.seq_lens = torch.tensor(seq_lens, dtype=torch.int64)
    fb.mrope_positions = None
    fn(fb, RUNNER, SimpleNamespace(multimodal_inputs=mm_inputs, **batch_fields))
    return fb.mrope_positions


@pytest.mark.parametrize(
    "seq_lens,prompt_lens",
    [
        ([73, 5, 129, 1], [65, 3, 100, 1]),
        ([2048, 1, 512], [2000, 1, 500]),
        ([10, 44], [None, 30]),  # a request with no multimodal input
    ],
)
def test_decode_matches_upstream(seq_lens, prompt_lens):
    mm = [None if p is None else mm_input(p) for p in prompt_lens]

    fast = run(FAST, ForwardMode.DECODE, seq_lens, mm)

    assert torch.equal(fast, run(ORIG, ForwardMode.DECODE, seq_lens, mm))
    expected = (
        (torch.tensor(seq_lens, dtype=torch.int64) - 1).unsqueeze(0).expand(3, -1)
    )
    assert torch.equal(fast, expected)


def test_non_degenerate_falls_back():
    mm = [mm_input(40, delta=7, degenerate=False)]

    fast = run(FAST, ForwardMode.DECODE, [50], mm)

    assert torch.equal(fast, run(ORIG, ForwardMode.DECODE, [50], mm))
    # delta is honoured, so the seq_len - 1 shortcut was not taken
    assert fast[0, 0].item() == 50 + 7 - 1


def test_extend_falls_back():
    mm = [mm_input(20)]
    fields = {
        "extend_lens": [4],
        "prefix_lens": [16],
        "reqs": [SimpleNamespace(session=None)],
    }

    fast = run(FAST, ForwardMode.EXTEND, [20], mm, **fields)

    assert torch.equal(fast, run(ORIG, ForwardMode.EXTEND, [20], mm, **fields))


def run_spec(compute_positions, multimodal_inputs, positions, *, from_spec_info: bool):
    forward_batch = object.__new__(ForwardBatch)
    forward_batch.seq_lens = torch.full(
        (len(multimodal_inputs),), 64, dtype=torch.int64
    )
    forward_batch.mrope_positions = None
    if from_spec_info:
        batch = SimpleNamespace(
            multimodal_inputs=multimodal_inputs,
            spec_info=SimpleNamespace(positions=positions),
        )
        compute_positions(forward_batch, RUNNER, batch)
    else:
        # draft extend passes its own positions and leaves spec_info unread
        batch = SimpleNamespace(multimodal_inputs=multimodal_inputs, spec_info=None)
        compute_positions(forward_batch, RUNNER, batch, positions)
    return forward_batch.mrope_positions


@pytest.mark.parametrize(
    "prompt_lengths,draft_tokens_per_request,dtype,from_spec_info",
    [
        ([65, None, 100], 8, torch.int64, True),  # verify, one text-only request
        ([40, 12], 4, torch.int32, False),  # draft extend with int32 positions
    ],
)
def test_spec_matches_upstream(
    monkeypatch, prompt_lengths, draft_tokens_per_request, dtype, from_spec_info
):
    """Every speculative forward used to copy the zero mrope deltas to the GPU."""

    def _sglang_path(forward_batch, model_runner, batch, seq_positions):
        raise AssertionError("a degenerate batch took the sglang delta copy")

    monkeypatch.setattr(
        mrope_fast_path, "ORIGINAL_COMPUTE_SPEC_MROPE_POSITIONS", _sglang_path
    )
    multimodal_inputs = [
        None if prompt_length is None else mm_input(prompt_length)
        for prompt_length in prompt_lengths
    ]
    positions = (
        torch.arange(len(prompt_lengths) * draft_tokens_per_request, dtype=dtype) + 50
    )

    fast = run_spec(
        SPEC_FAST, multimodal_inputs, positions, from_spec_info=from_spec_info
    )
    reference = run_spec(
        SPEC_ORIG, multimodal_inputs, positions, from_spec_info=from_spec_info
    )

    assert fast.dtype == reference.dtype
    assert torch.equal(fast, reference)


def test_spec_non_degenerate_falls_back():
    multimodal_inputs = [mm_input(40, delta=7, degenerate=False), mm_input(20)]
    positions = torch.arange(8, dtype=torch.int64) + 50

    fast = run_spec(SPEC_FAST, multimodal_inputs, positions, from_spec_info=True)
    reference = run_spec(SPEC_ORIG, multimodal_inputs, positions, from_spec_info=True)

    assert torch.equal(fast, reference)
    # delta is honoured, so the degenerate shortcut was not taken
    assert fast[0, 0].item() == 50 + 7


def test_apply_installs_both_shortcuts(monkeypatch):
    monkeypatch.setattr(ForwardBatch, "_compute_mrope_positions", ORIG)
    monkeypatch.setattr(ForwardBatch, "compute_spec_mrope_positions", SPEC_ORIG)
    monkeypatch.delenv("SGLANG_OMNI_ASR_FAST_MROPE", raising=False)

    mrope_fast_path.apply_asr_mrope_fast_path()

    assert (
        ForwardBatch._compute_mrope_positions is FAST
    )  # noqa: leading-underscore  # production name
    assert ForwardBatch.compute_spec_mrope_positions is SPEC_FAST
