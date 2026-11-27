# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.voxtral_tts.audio_tokenizer import VoxtralCodecGraphs


def _landing_only_graphs(downsample_factor: int) -> VoxtralCodecGraphs:
    """Build a graphs holder without CUDA graph capture for landing tests."""
    graphs = VoxtralCodecGraphs.__new__(VoxtralCodecGraphs)
    graphs.tokenizer = SimpleNamespace(downsample_factor=downsample_factor)
    graphs.host_staging = None
    return graphs


def _windows(frames: list[int], codebooks: int) -> list[torch.Tensor]:
    return [torch.zeros((count, codebooks), dtype=torch.long) for count in frames]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA landing")
def test_codec_landing_matches_device_convert_copy() -> None:
    """The pinned raw-bfloat16 landing is bit-identical to the old path."""
    torch.manual_seed(42)
    graphs = _landing_only_graphs(downsample_factor=320)
    batch, width, codebooks = 3, 144000, 8
    audio = (
        torch.randn(batch, 1, width, device="cuda", dtype=torch.float32)
        .to(torch.bfloat16)
        .contiguous()
    )
    frames = [450, 300, 450]
    windows = _windows(frames, codebooks)
    landing = graphs.stage_landing(audio, windows)
    torch.cuda.current_stream().synchronize()
    landed = graphs.slice_landed(landing, windows)
    reference = audio[:batch, 0].float().cpu()
    for row, count in enumerate(frames):
        want = count * 320
        expected = reference[row, :want]
        assert landed[row].shape == expected.shape
        assert torch.equal(landed[row], expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA landing")
def test_codec_landing_reuses_staging_across_batch_sizes() -> None:
    """The staging regrows for a larger batch and is reused for a smaller one."""
    torch.manual_seed(7)
    graphs = _landing_only_graphs(downsample_factor=320)
    width, codebooks = 144000, 8
    audio = torch.zeros(4, 1, width, device="cuda", dtype=torch.bfloat16)
    audio[0, 0, 0] = 1.0
    expected_head = torch.zeros(10)
    expected_head[0] = 1.0

    small = _windows([100], codebooks)
    landing = graphs.stage_landing(audio, small)
    torch.cuda.current_stream().synchronize()
    assert graphs.host_staging is not None
    initial = graphs.host_staging
    landed = graphs.slice_landed(landing, small)
    assert landed[0].shape[0] == 100 * 320
    assert torch.equal(landed[0][:10], expected_head)

    big = _windows([450, 450, 450, 450], codebooks)
    landing = graphs.stage_landing(audio, big)
    torch.cuda.current_stream().synchronize()
    assert graphs.host_staging is not initial
    regrown = graphs.host_staging
    landed = graphs.slice_landed(landing, big)
    assert landed[3].shape[0] == 450 * 320
    assert torch.equal(landed[0][:10], expected_head)

    small = _windows([100], codebooks)
    landing = graphs.stage_landing(audio, small)
    torch.cuda.current_stream().synchronize()
    assert graphs.host_staging is regrown
    landed = graphs.slice_landed(landing, small)
    assert landed[0].shape[0] == 100 * 320
    assert torch.equal(landed[0][:10], expected_head)
