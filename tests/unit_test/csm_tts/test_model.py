# SPDX-License-Identifier: Apache-2.0
"""Prefill request ids and last-token gather for the CSM backbone step."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.csm_tts.model import CsmTTSModel


class ExtractHost:
    """Stands in for CsmTTSModel so extract_batch_metadata can run unbound."""

    def infer_batch_size(self, forward_batch: SimpleNamespace) -> int:
        return 2


def empty_forward_batch() -> SimpleNamespace:
    return SimpleNamespace(
        rids=None,
        request_ids=None,
        sampling_info=None,
        csm_depth_params=None,
    )


def test_prefill_requires_real_request_ids() -> None:
    with pytest.raises(RuntimeError, match="omni_prefill_rids"):
        CsmTTSModel.extract_batch_metadata(
            ExtractHost(),
            empty_forward_batch(),
            None,
        )


def test_prefill_accepts_omni_prefill_rids() -> None:
    request_ids, params = CsmTTSModel.extract_batch_metadata(
        ExtractHost(),
        empty_forward_batch(),
        ["speech-aaa", "speech-bbb"],
    )
    assert request_ids == ["speech-aaa", "speech-bbb"]
    assert len(params) == 2


def test_last_token_hidden_gathers_extend_rows() -> None:
    hidden = torch.arange(12, dtype=torch.float32).view(6, 2)
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: True),
        extend_seq_lens=torch.tensor([2, 4]),
    )
    last = CsmTTSModel.last_token_hidden(hidden, forward_batch, is_decode=False)
    torch.testing.assert_close(last, hidden[[1, 5]])
