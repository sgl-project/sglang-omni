# SPDX-License-Identifier: Apache-2.0
"""Byte-identity tests for the vectorized ASR decode mrope fast path."""

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

from sglang_omni.models.qwen3_asr import mrope_fast_path

_RUNNER = SimpleNamespace(device="cpu")
_FAST = mrope_fast_path._fast_compute_mrope_positions
_ORIG = ForwardBatch._compute_mrope_positions


@pytest.fixture(autouse=True)
def _upstream(monkeypatch):
    monkeypatch.setattr(
        "sglang.srt.model_executor.forward_batch_info.get_server_args",
        lambda: SimpleNamespace(rl_on_policy_target=None),
    )
    monkeypatch.setattr(mrope_fast_path, "_orig_compute_mrope_positions", _ORIG)


def _mm_input(prompt_len: int, *, delta: int = 0, degenerate: bool = True):
    """Mirror what request_builders constructs for a Qwen3-ASR request."""
    positions = torch.arange(prompt_len, dtype=torch.long)
    mm = SimpleNamespace(
        mrope_positions=positions.unsqueeze(0).expand(3, -1).clone(),
        mrope_position_delta=torch.tensor([delta], dtype=torch.long),
        mrope_position_delta_repeated_cache=None,
    )
    if degenerate:
        setattr(mm, mrope_fast_path.DEGENERATE_MROPE_FLAG, True)
    return mm


def _run(fn, mode, seq_lens, mm_inputs, **batch_fields):
    fb = object.__new__(ForwardBatch)
    fb.forward_mode = mode
    fb.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
    fb.mrope_positions = None
    fn(fb, _RUNNER, SimpleNamespace(multimodal_inputs=mm_inputs, **batch_fields))
    return fb.mrope_positions


@pytest.mark.parametrize(
    "seq_lens,prompt_lens",
    [
        ([73], [65]),
        ([73, 5, 129, 1], [65, 3, 100, 1]),
        ([2048, 1, 512], [2000, 1, 500]),
        ([10, 44], [None, 30]),  # a request with no multimodal input
    ],
)
def test_decode_matches_upstream(seq_lens, prompt_lens):
    mm = [None if p is None else _mm_input(p) for p in prompt_lens]

    fast = _run(_FAST, ForwardMode.DECODE, seq_lens, mm)

    assert torch.equal(fast, _run(_ORIG, ForwardMode.DECODE, seq_lens, mm))
    expected = (
        (torch.tensor(seq_lens, dtype=torch.int64) - 1).unsqueeze(0).expand(3, -1)
    )
    assert torch.equal(fast, expected)


def test_non_degenerate_falls_back():
    mm = [_mm_input(40, delta=7, degenerate=False)]

    fast = _run(_FAST, ForwardMode.DECODE, [50], mm)

    assert torch.equal(fast, _run(_ORIG, ForwardMode.DECODE, [50], mm))
    # delta is honoured, so the seq_len - 1 shortcut was not taken
    assert fast[0, 0].item() == 50 + 7 - 1


def test_extend_falls_back():
    mm = [_mm_input(20)]
    fields = {"extend_lens": [4], "prefix_lens": [16]}

    fast = _run(_FAST, ForwardMode.EXTEND, [20], mm, **fields)

    assert torch.equal(fast, _run(_ORIG, ForwardMode.EXTEND, [20], mm, **fields))
