# SPDX-License-Identifier: Apache-2.0
"""Prefill sidecar contract: overlay embeds attach through OmniPrefillInputs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.unit_test.csm_tts.sglang_fakes import ensure_sglang_stubs

ensure_sglang_stubs()

pytest.importorskip("sglang.srt.platforms")

from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.models.csm_tts.model_runner import CsmTTSModelRunner


def test_csm_prefill_embeddings_attach_private_sidecar() -> None:
    runner = object.__new__(CsmTTSModelRunner)
    runner.extract_depth_sampling_params = lambda _requests: ([0.9], [50])
    raw_embeds = torch.arange(16 * 4, dtype=torch.float32).view(16, 4)
    runner.build_prefill_input_embeds = lambda _forward_batch, _requests: raw_embeds
    request = SimpleNamespace(request_id="request")
    forward_batch = SimpleNamespace(
        input_embeds=None,
        replace_embeds=None,
        mm_inputs=[None],
        input_ids=torch.zeros(16, dtype=torch.long),
        batch_size=1,
    )

    runner.before_prefill(forward_batch, None, [request])

    payload = get_omni_prefill_inputs(forward_batch)
    assert forward_batch.input_embeds is None
    assert forward_batch.mm_inputs == [None]
    assert payload is not None
    assert payload.input_embeds.shape == (16, 4)
    torch.testing.assert_close(payload.input_embeds, raw_embeds)
    assert forward_batch.req_ids == ["request"]
    assert forward_batch.csm_depth_params == [(0.9, 50)]


def test_csm_text_only_prefill_does_not_attach_sidecar() -> None:
    runner = object.__new__(CsmTTSModelRunner)
    runner.extract_depth_sampling_params = lambda _requests: ([0.9], [50])
    runner.build_prefill_input_embeds = lambda _forward_batch, _requests: None
    forward_batch = SimpleNamespace(
        input_embeds=None,
        replace_embeds=None,
        mm_inputs=[None],
        input_ids=torch.zeros(8, dtype=torch.long),
    )

    runner.before_prefill(forward_batch, None, [SimpleNamespace(request_id="plain")])

    assert get_omni_prefill_inputs(forward_batch) is None
    assert forward_batch.input_embeds is None
