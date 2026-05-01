# SPDX-License-Identifier: Apache-2.0
"""Regression tests for S2-Pro runtime request state transitions."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.engines.omni.types import RequestOutput, SchedulerRequest
from sglang_omni.models.fishaudio_s2_pro.runtime.s2pro_ar import S2ProStepOutput
from sglang_omni.models.fishaudio_s2_pro.runtime.s2pro_sglang_ar import (
    S2ProSGLangIterationController,
    S2ProSGLangRequestData,
)


class _CountingTreeCache:
    def __init__(self) -> None:
        self.cached_requests = 0

    def cache_unfinished_req(self, req) -> None:
        del req
        self.cached_requests += 1


def test_iteration_controller_excludes_terminal_im_end_from_audio_codes() -> None:
    im_end_token_id = 151645
    semantic_token_id = 151678
    tree_cache = _CountingTreeCache()
    controller = S2ProSGLangIterationController(tree_cache, im_end_token_id)
    sglang_req = SimpleNamespace(
        is_chunked=0,
        output_ids=[],
        decode_batch_idx=0,
        finished=lambda: False,
    )
    data = S2ProSGLangRequestData(
        input_ids=torch.tensor([], dtype=torch.long),
        req=sglang_req,
    )
    request = SchedulerRequest(request_id="req-terminal", data=data)

    semantic_codes = torch.tensor([[semantic_token_id], [11], [22]])
    semantic_output = RequestOutput(
        request_id=request.request_id,
        data=S2ProStepOutput(codes=semantic_codes),
    )
    controller.update_request(request, semantic_output)

    eos_codes = torch.tensor([[im_end_token_id], [33], [44]])
    eos_output = RequestOutput(
        request_id=request.request_id,
        data=S2ProStepOutput(codes=eos_codes),
    )
    controller.update_request(request, eos_output)

    assert sglang_req.output_ids == [semantic_token_id, im_end_token_id]
    assert len(data.output_codes) == 1
    assert torch.equal(data.output_codes[0], semantic_codes)
    assert data._previous_semantic_tokens == [semantic_token_id]
    assert torch.equal(data._last_codebook_values, semantic_codes[1:, 0])
    assert tree_cache.cached_requests == 1
    assert controller.is_finished(request, eos_output)
