# SPDX-License-Identifier: Apache-2.0
"""Prefill rids and last-token gather for the CSM backbone step."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.unit_test.csm_tts.sglang_fakes import ensure_sglang_stubs

ensure_sglang_stubs()

from sglang_omni.models.csm_tts.model import CsmTTSModel


class _ExtractHost:
    gen_params_for_batch = staticmethod(CsmTTSModel.gen_params_for_batch)

    def infer_batch_size(self, forward_batch) -> int:
        del forward_batch
        return 2


def test_prefill_requires_real_request_ids() -> None:
    with pytest.raises(RuntimeError, match="omni_prefill_rids"):
        CsmTTSModel.extract_batch_metadata(
            _ExtractHost(),
            SimpleNamespace(rids=None, req_ids=None, sampling_info=None),
            None,
        )


def test_prefill_accepts_omni_prefill_rids() -> None:
    req_ids, params = CsmTTSModel.extract_batch_metadata(
        _ExtractHost(),
        SimpleNamespace(rids=None, req_ids=None, sampling_info=None),
        ["speech-aaa", "speech-bbb"],
    )
    assert req_ids == ["speech-aaa", "speech-bbb"]
    assert len(params) == 2


def test_last_token_hidden_gathers_extend_rows() -> None:
    hidden = torch.arange(12, dtype=torch.float32).view(6, 2)
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: True),
        extend_seq_lens=torch.tensor([2, 4]),
    )
    last = CsmTTSModel.last_token_hidden(hidden, forward_batch, is_decode=False)
    torch.testing.assert_close(last, hidden[[1, 5]])


def test_last_token_hidden_decode_keeps_rows() -> None:
    hidden = torch.arange(8, dtype=torch.float32).view(2, 4)
    last = CsmTTSModel.last_token_hidden(
        hidden, SimpleNamespace(extend_seq_lens=None), is_decode=True
    )
    torch.testing.assert_close(last, hidden)
    hidden_3d = hidden.unsqueeze(1)
    last_3d = CsmTTSModel.last_token_hidden(
        hidden_3d, SimpleNamespace(extend_seq_lens=None), is_decode=True
    )
    torch.testing.assert_close(last_3d, hidden)
