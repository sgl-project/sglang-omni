# SPDX-License-Identifier: Apache-2.0
"""Contracts of the Qwen3-TTS finish payload.

A finished request's codes leave the device through a pinned non blocking copy,
the shape sglang's own overlap path uses for its step results. The builder
records the copy's event on the request data instead of waiting for it, and the
stage runtime waits that event before the payload is routed, so no reader sees
the tensor before the copy has landed. The first call of the builder in a
process pays one time costs that synchronize the device, the lazy load of its
kernels and the first fill of a pinned size class, so the CUDA timing test runs
the builder once on a warm up request first, as serving does.
"""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.qwen3_tts.request_builders import (
    Qwen3TTSSGLangRequestData,
    apply_sglang_qwen3_tts_result,
)
from sglang_omni.proto import OmniRequest, StagePayload

NUM_CODE_GROUPS = 3
STREAM_CYCLES = 1_000_000_000


@pytest.fixture(autouse=True)
def _require_cuda_for_accelerator_tests(request: pytest.FixtureRequest):
    if request.node.get_closest_marker("accelerator") and not torch.cuda.is_available():
        pytest.skip("pinned staging needs CUDA")


def _payload() -> StagePayload:
    return StagePayload(request_id="req", request=OmniRequest(inputs={}), data={})


def _data(device: torch.device, *, steps: int, ref_len: int):
    ref_code = torch.arange(ref_len * NUM_CODE_GROUPS, device=device).reshape(
        ref_len, NUM_CODE_GROUPS
    )
    codes = [
        torch.full((NUM_CODE_GROUPS,), 100 + step, dtype=torch.long, device=device)
        for step in range(steps)
    ]
    return Qwen3TTSSGLangRequestData(
        output_codes=codes,
        ref_code=ref_code if ref_len else None,
        ref_code_len=ref_len,
        max_new_tokens=steps + 1,
    )


def _expected(data: Qwen3TTSSGLangRequestData) -> torch.Tensor:
    parts = []
    if data.ref_code is not None:
        parts.append(data.ref_code.cpu())
    parts.append(torch.stack(data.output_codes).cpu())
    return torch.cat(parts, dim=0)


def test_cpu_codes_leave_as_a_plain_tensor_without_an_event():
    data = _data(torch.device("cpu"), steps=4, ref_len=2)

    result = apply_sglang_qwen3_tts_result(_payload(), data)

    codes = result.data["audio_codes"]
    assert torch.equal(codes, _expected(data))
    assert codes.dtype == torch.long
    assert data.result_ready_event is None
    assert result.data["completion_tokens"] == 4
    assert result.data["prompt_tokens"] == 2


def test_no_codes_gives_an_empty_tensor_without_an_event():
    data = _data(torch.device("cpu"), steps=0, ref_len=0)

    result = apply_sglang_qwen3_tts_result(_payload(), data)

    assert result.data["audio_codes"].shape == (0, 0)
    assert data.result_ready_event is None


@pytest.mark.accelerator
def test_cuda_codes_leave_as_a_pinned_copy_that_the_event_completes():
    device = torch.device("cuda")
    apply_sglang_qwen3_tts_result(_payload(), _data(device, steps=5, ref_len=2))
    torch.cuda.synchronize()
    data = _data(device, steps=5, ref_len=2)
    stream = torch.cuda.current_stream(device)

    torch.cuda._sleep(STREAM_CYCLES)
    result = apply_sglang_qwen3_tts_result(_payload(), data)
    returned_early = not stream.query()
    event = data.result_ready_event
    event.synchronize()
    codes = result.data["audio_codes"]

    assert returned_early
    assert codes.is_pinned()
    assert codes.device.type == "cpu"
    assert codes.dtype == torch.long
    assert torch.equal(codes, _expected(data))
    assert codes.shape == (7, NUM_CODE_GROUPS)


@pytest.mark.accelerator
def test_cuda_codes_keep_their_values_when_the_device_rows_change_afterwards():
    device = torch.device("cuda")
    data = _data(device, steps=3, ref_len=0)
    expected = _expected(data)

    result = apply_sglang_qwen3_tts_result(_payload(), data)
    for chunk in data.output_codes:
        chunk.fill_(-1)
    data.result_ready_event.synchronize()

    assert torch.equal(result.data["audio_codes"], expected)
