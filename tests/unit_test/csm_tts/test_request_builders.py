# SPDX-License-Identifier: Apache-2.0
"""Request-builder contracts: extra_key namespace + adapter clamp.

Req.extra_key namespaces context audio so prompts that share the same
128002×F token prefix cannot alias if radix is re-enabled.
"""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.csm_tts import request_builders
from sglang_omni.models.csm_tts.payload_types import CsmTtsState
from sglang_omni.models.csm_tts.utils import BACKBONE_CTX, NUM_CODEBOOKS
from sglang_omni.proto import OmniRequest, StagePayload


def install_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Moss-style local fakes so builders are not tied to real sglang Req."""

    class FakeReq:
        def __init__(
            self,
            *,
            rid,
            origin_input_text,
            origin_input_ids,
            sampling_params,
            vocab_size=None,
            extra_key=None,
            **kwargs,
        ) -> None:
            self.rid = rid
            self.origin_input_text = origin_input_text
            self.origin_input_ids = origin_input_ids
            self.sampling_params = sampling_params
            self.vocab_size = vocab_size
            self.extra_key = extra_key
            self._codec_suppress_tokens = None  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            self._input_embeds_are_projected = False  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            self.extend_len = len(origin_input_ids)

    class FakeSamplingParams:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)
            self.stop_strs = []
            self.stop_regex_strs = []

        def normalize(self, tokenizer) -> None:
            if self.stop_strs is None:
                self.stop_strs = []
            if self.stop_regex_strs is None:
                self.stop_regex_strs = []

    monkeypatch.setattr(request_builders, "Req", FakeReq)
    monkeypatch.setattr(request_builders, "SamplingParams", FakeSamplingParams)


def test_context_audio_namespaces_radix_cache(monkeypatch) -> None:
    install_fakes(monkeypatch)
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


def test_scheduler_adapters_clamp_frame_budget(monkeypatch) -> None:
    install_fakes(monkeypatch)
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
