# SPDX-License-Identifier: Apache-2.0
"""DSpark draft batch width on SGLang's FA3 backend."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend

from sglang_omni.model_runner import fa3_draft_width


@pytest.mark.parametrize(
    ("is_draft_worker", "sample_from_anchor", "expected_rows_per_request"),
    [(True, False, 8), (True, True, 7), (False, False, 7)],
)
def test_fa3_lays_out_a_dspark_draft_batch_at_the_draft_width(
    monkeypatch: pytest.MonkeyPatch,
    is_draft_worker: bool,
    sample_from_anchor: bool,
    expected_rows_per_request: int,
) -> None:
    def sglang_init(
        backend: FlashAttentionBackend,
        model_runner: SimpleNamespace,
        skip_prefill: bool = False,
        speculative_step_id: int = 0,
        topk: int = 0,
        speculative_num_steps: int = 0,
        fa_impl_ver: int = 3,
    ) -> None:
        # SGLang sizes every DSpark draft batch as gamma rows per request.
        backend.speculative_num_draft_tokens = 7

    monkeypatch.setattr(FlashAttentionBackend, "__init__", sglang_init)
    monkeypatch.setattr(fa3_draft_width, "fa3_draft_width_installed", False)
    monkeypatch.setattr(
        fa3_draft_width,
        "get_spec",
        lambda: SimpleNamespace(
            speculative_algorithm="DSPARK", speculative_num_draft_tokens=8
        ),
    )
    model_runner = SimpleNamespace(
        is_draft_worker=is_draft_worker,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(sample_from_anchor=sample_from_anchor)
        ),
    )

    fa3_draft_width.install_fa3_draft_width()
    backend = object.__new__(FlashAttentionBackend)
    FlashAttentionBackend.__init__(backend, model_runner)

    assert backend.speculative_num_draft_tokens == expected_rows_per_request
