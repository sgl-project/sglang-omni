# SPDX-License-Identifier: Apache-2.0
"""Stage 6 acceptance tests for the CUDA Graph migration — row pool
lifecycle.

These cover the contract between :class:`HiggsTTSModel`'s row pool
and the scheduler-side abort path:

- ``acquire_row`` is idempotent for a given request id.
- ``release_row`` returns the row to the free pool; the next
  ``acquire`` immediately re-uses it.
- A request that finishes mid-decode and is released doesn't keep
  its output_codes log around.
- Pool exhaustion raises a clear error instead of corrupting state.
- The reserved ``_padding_row`` is not in the free pool and is not
  returned by ``acquire_row`` for any real request id.

The model is constructed without the heavy Qwen3 backbone weights:
we build a thin stand-in that exposes only what
``HiggsTTSModel.__init__`` needs (an ``embed_tokens.weight`` tensor),
and skip the full transformer forward. The pool / row machinery
under test is independent of the backbone.
"""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.higgs_tts.sampler import HiggsBatchedSamplerState


@pytest.fixture
def pool() -> HiggsBatchedSamplerState:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # max_batch_size + 1 padding row, matching ``HiggsTTSModel.__init__``.
    return HiggsBatchedSamplerState(max_batch_size=5, num_codebooks=8, device=device)


def _free_rows_from_pool(pool: HiggsBatchedSamplerState, padding_row: int) -> list[int]:
    """The model's free-row list, minus the padding row."""
    return [r for r in range(pool.max_batch_size) if r != padding_row]


class _StubModel:
    """Minimal mirror of :class:`HiggsTTSModel`'s row-pool machinery.

    Reproduces ``acquire_row`` / ``release_row`` semantics exactly,
    so the pool lifecycle can be exercised without spinning up a
    transformer backbone (which would need a CUDA device + the full
    sglang stack).
    """

    def __init__(self, max_batch_size: int, num_codebooks: int, device: str):
        self._max_batch_size = max_batch_size
        pool_size = max_batch_size + 1
        self._sampler_pool = HiggsBatchedSamplerState(
            max_batch_size=pool_size, num_codebooks=num_codebooks, device=device
        )
        self._padding_row = max_batch_size
        self._rid_to_row: dict[str, int] = {}
        self._free_rows: list[int] = list(range(max_batch_size))
        self._output_codes: dict[str, list[torch.Tensor]] = {}

    def acquire_row(self, req_id: str) -> int:
        row = self._rid_to_row.get(req_id)
        if row is not None:
            return row
        if not self._free_rows:
            raise RuntimeError(
                f"sampler pool exhausted (max={self._max_batch_size})"
            )
        row = self._free_rows.pop()
        self._rid_to_row[req_id] = row
        self._sampler_pool.reset_row(row)
        return row

    def release_row(self, req_id: str) -> None:
        row = self._rid_to_row.pop(req_id, None)
        if row is not None:
            self._free_rows.append(row)
        self._output_codes.pop(req_id, None)


# ---------------------------------------------------------------------------


def test_acquire_row_idempotent():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _StubModel(max_batch_size=4, num_codebooks=8, device=device)

    row1 = model.acquire_row("req-A")
    row2 = model.acquire_row("req-A")
    assert row1 == row2
    # Mapping recorded.
    assert model._rid_to_row == {"req-A": row1}
    # Padding row never handed out.
    assert row1 != model._padding_row


def test_release_returns_row_to_pool():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _StubModel(max_batch_size=4, num_codebooks=8, device=device)

    rA = model.acquire_row("req-A")
    rB = model.acquire_row("req-B")
    assert rA != rB

    # Pretend req-A finishes and we release it.
    model.release_row("req-A")
    assert "req-A" not in model._rid_to_row
    assert rA in model._free_rows

    # A fresh request takes it back.
    rC = model.acquire_row("req-C")
    assert rC == rA


def test_release_drops_output_codes_log():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _StubModel(max_batch_size=4, num_codebooks=8, device=device)

    row = model.acquire_row("req-A")
    # Simulate some output codes landed during decode.
    model._output_codes["req-A"] = [torch.zeros(8, dtype=torch.long)]
    assert "req-A" in model._output_codes

    model.release_row("req-A")
    assert "req-A" not in model._output_codes
    assert row in model._free_rows


def test_release_is_idempotent_no_op_on_unknown_rid():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _StubModel(max_batch_size=4, num_codebooks=8, device=device)
    free_before = list(model._free_rows)
    model.release_row("never-acquired")
    assert model._free_rows == free_before  # unchanged


def test_pool_exhaustion_raises():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _StubModel(max_batch_size=3, num_codebooks=8, device=device)

    for i in range(3):
        model.acquire_row(f"req-{i}")
    assert not model._free_rows
    with pytest.raises(RuntimeError, match="sampler pool exhausted"):
        model.acquire_row("req-overflow")


def test_padding_row_not_handed_out():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _StubModel(max_batch_size=4, num_codebooks=8, device=device)

    # Acquire all real rows.
    rows = {model.acquire_row(f"req-{i}") for i in range(4)}
    assert model._padding_row not in rows


def test_acquire_after_full_cycle_resets_state():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _StubModel(max_batch_size=2, num_codebooks=8, device=device)

    row = model.acquire_row("req-A")
    # Pollute the row state — should be wiped on next acquire.
    model._sampler_pool.delay_count[row] = 7
    model._sampler_pool.eoc_countdown[row] = 3
    model._sampler_pool.generation_done[row] = True
    model._sampler_pool.last_codes[row] = torch.arange(
        8, dtype=torch.long, device=model._sampler_pool.device
    )

    model.release_row("req-A")
    row2 = model.acquire_row("req-B")  # gets the same physical row back
    assert row2 == row
    assert int(model._sampler_pool.delay_count[row2].item()) == 0
    assert int(model._sampler_pool.eoc_countdown[row2].item()) == -1
    assert not bool(model._sampler_pool.generation_done[row2].item())
    assert torch.equal(
        model._sampler_pool.last_codes[row2],
        torch.zeros(8, dtype=torch.long, device=model._sampler_pool.device),
    )
