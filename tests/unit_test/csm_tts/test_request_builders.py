# SPDX-License-Identifier: Apache-2.0
"""Request-builder contracts: extra_key namespace + adapter clamp.

``Req.extra_key`` namespaces context audio so prompts that share the same
``128002×F`` token prefix cannot alias if radix is re-enabled.
"""

from __future__ import annotations

import pytest
import torch

from tests.unit_test.csm_tts.sglang_fakes import (
    FakeReq,
    FakeSamplingParams,
    ensure_sglang_stubs,
)

ensure_sglang_stubs()
pytest.importorskip("sglang.srt.mem_cache")

from sglang_omni.models.csm_tts import request_builders
from sglang_omni.models.csm_tts.payload_types import CsmTtsState
from sglang_omni.models.csm_tts.utils import BACKBONE_CTX, NUM_CODEBOOKS
from sglang_omni.proto import OmniRequest, StagePayload


def _install_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(request_builders, "Req", FakeReq)
    monkeypatch.setattr(request_builders, "SamplingParams", FakeSamplingParams)


def test_context_audio_namespaces_radix_cache(monkeypatch) -> None:
    _install_fakes(monkeypatch)
    prompt = [128000, 128002, 128002, 128003, 128001]
    codes_a = torch.randint(0, 2048, (2, NUM_CODEBOOKS))
    codes_b = torch.randint(0, 2048, (2, NUM_CODEBOOKS))
    first = request_builders.build_sglang_csm_request(
        CsmTtsState(
            prompt_token_ids=prompt,
            context=[{"speaker_id": 0, "text": "hi"}],
            context_codes=[codes_a],
        ),
        request_id="first",
    ).req
    second = request_builders.build_sglang_csm_request(
        CsmTtsState(
            prompt_token_ids=prompt,
            context=[{"speaker_id": 0, "text": "hi"}],
            context_codes=[codes_b],
        ),
        request_id="second",
    ).req

    assert first.origin_input_ids == second.origin_input_ids
    assert first.extra_key != second.extra_key


def test_plain_tts_shares_radix_subtree(monkeypatch) -> None:
    _install_fakes(monkeypatch)
    first = request_builders.build_sglang_csm_request(
        CsmTtsState(prompt_token_ids=[128000, 1, 128001]),
        request_id="a",
    ).req
    second = request_builders.build_sglang_csm_request(
        CsmTtsState(prompt_token_ids=[128000, 1, 128001]),
        request_id="b",
    ).req
    assert first.extra_key is None
    assert second.extra_key is None


def test_scheduler_adapters_clamp_frame_budget(monkeypatch) -> None:
    _install_fakes(monkeypatch)
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr(request_builders, "perf_counter", lambda: next(ticks))
    request_builder, result_adapter = request_builders.make_csm_scheduler_adapters(
        max_new_tokens_cap=20,
    )
    long_prompt = list(range(BACKBONE_CTX - 5))
    payload = StagePayload(
        request_id="req-csm",
        request=OmniRequest(inputs={}, params={}),
        data=CsmTtsState(
            prompt_token_ids=long_prompt,
            max_new_tokens=125,
        ).to_dict(),
    )
    data = request_builder(payload)
    assert data.max_new_tokens == 4  # BACKBONE_CTX - 1 - len(prompt) = 4
    data.output_codes.append(torch.arange(NUM_CODEBOOKS, dtype=torch.long))
    result = result_adapter(data)
    state = CsmTtsState.from_dict(result.data)
    assert state.completion_tokens == 1
    assert state.engine_time_s == pytest.approx(2.5)
    assert state.output_codes is not None
    assert len(state.output_codes) == 1
