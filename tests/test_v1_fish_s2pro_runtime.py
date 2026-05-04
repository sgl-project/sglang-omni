# SPDX-License-Identifier: Apache-2.0
"""Regression tests for V1 FishAudio S2-Pro request state transitions."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni_v1.models.fishaudio_s2_pro.fish_scheduler import (
    FishIterationController,
)
from sglang_omni_v1.models.fishaudio_s2_pro.model_runner import FishS2ProModelRunner
from sglang_omni_v1.models.fishaudio_s2_pro.request_builders import (
    S2ProSGLangRequestData,
)
from sglang_omni_v1.scheduling.types import SchedulerRequest


class _CountingTreeCache:
    def __init__(self) -> None:
        self.cached_requests = 0

    def cache_unfinished_req(self, req) -> None:
        del req
        self.cached_requests += 1


def _make_runner(im_end_token_id: int) -> FishS2ProModelRunner:
    runner = object.__new__(FishS2ProModelRunner)
    runner._im_end_token_id = im_end_token_id
    runner.model = SimpleNamespace()
    return runner


def _set_model_step(
    runner: FishS2ProModelRunner, semantic_token_id: int, residual_tokens: list[int]
) -> None:
    runner.model._output_semantic_ids = torch.tensor(
        [semantic_token_id], dtype=torch.long
    )
    runner.model._output_codes = torch.tensor(
        [[semantic_token_id, *residual_tokens]], dtype=torch.long
    )


def test_v1_s2pro_terminal_im_end_is_not_audio_codebook_frame() -> None:
    im_end_token_id = 151645
    semantic_token_id = 151678
    runner = _make_runner(im_end_token_id)
    tree_cache = _CountingTreeCache()
    controller = FishIterationController(tree_cache, im_end_token_id)
    req = SimpleNamespace(
        is_chunked=0,
        output_ids=[],
        decode_batch_idx=0,
        finished=lambda: False,
    )
    data = S2ProSGLangRequestData(
        input_ids=torch.tensor([], dtype=torch.long),
        req=req,
    )
    request = SchedulerRequest(request_id="req-terminal", data=data)
    batch_result = SimpleNamespace(next_token_ids=None)

    _set_model_step(runner, semantic_token_id, [11, 22])
    runner._collect_step_outputs(batch_result, [request])
    controller.update_request(request, int(batch_result.next_token_ids[0].item()))

    _set_model_step(runner, im_end_token_id, [33, 44])
    runner._collect_step_outputs(batch_result, [request])
    eos_token = int(batch_result.next_token_ids[0].item())
    controller.update_request(request, eos_token)

    assert controller.is_finished(request, eos_token)
    assert req.output_ids == [semantic_token_id, im_end_token_id]
    assert len(data.output_codes) == 1
    assert torch.equal(
        data.output_codes[0],
        torch.tensor([[semantic_token_id], [11], [22]], dtype=torch.long),
    )
    assert data.previous_semantic_tokens == [semantic_token_id]
    assert torch.equal(data.last_codebook_values, torch.tensor([11, 22]))
    assert tree_cache.cached_requests == 1
