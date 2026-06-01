# SPDX-License-Identifier: Apache-2.0
"""Unit tests for _Code2WavStreamingExecutor (mock vocoder, CPU)."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest
import torch
from torch import nn

from sglang_omni.models.qwen3_omni.components.code2wav_executor import (
    _Code2WavStreamingExecutor,
    _DecodeRequest,
)


class _MockVocoder(nn.Module):
    def __init__(
        self,
        upsample: int = 4,
        codebook_size: int = 2048,
        hidden_size: int = 4,
    ) -> None:
        super().__init__()
        self.total_upsample = int(upsample)
        self._codebook_size = int(codebook_size)
        self._embed = nn.Embedding(codebook_size, hidden_size)
        with torch.no_grad():
            torch.manual_seed(0)
            self._embed.weight.copy_(
                torch.arange(
                    codebook_size * hidden_size,
                    dtype=torch.float32,
                ).reshape(codebook_size, hidden_size)
                * 0.001
            )

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        assert codes.dim() == 3, "expect [B, Q, T]"
        embedded = self._embed(codes)
        per_position = embedded.sum(dim=(1, 3))
        _, _, T = codes.shape
        upsampled = per_position.repeat_interleave(
            self.total_upsample,
            dim=1,
        )
        offset = (
            torch.arange(
                T * self.total_upsample,
                device=codes.device,
                dtype=torch.float32,
            )
            * 0.001
        )
        return (upsampled + offset).unsqueeze(1)


def _make_executor(
    *,
    upsample: int = 4,
    max_batch_size: int = 4,
    stream_chunk_size: int = 10,
    left_context_size: int = 0,
) -> _Code2WavStreamingExecutor:
    vocoder = _MockVocoder(upsample=upsample)
    return _Code2WavStreamingExecutor(
        vocoder,
        device="cpu",
        stream_chunk_size=stream_chunk_size,
        left_context_size=left_context_size,
        max_batch_size=max_batch_size,
    )


def test_vocoder_forward_produces_expected_shape() -> None:
    exe = _make_executor(upsample=4)
    codes = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    wav = exe._vocoder_forward(codes, trim_samples=0)

    assert wav.shape == (3 * 4,)
    assert np.all(np.isfinite(wav))


def test_vocoder_forward_honours_trim_samples() -> None:
    exe = _make_executor(upsample=4)
    codes = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    wav_full = exe._vocoder_forward(codes, trim_samples=0)
    wav_trimmed = exe._vocoder_forward(codes, trim_samples=4)

    assert wav_trimmed.shape == (wav_full.shape[0] - 4,)
    np.testing.assert_allclose(wav_trimmed, wav_full[4:], rtol=1e-5)


def test_forward_batch_rejects_out_of_range_pad_token_id() -> None:
    exe = _make_executor()
    exe._pad_token_id = exe._model._codebook_size + 10
    requests = [
        _DecodeRequest(
            request_id="short",
            codes_window=torch.tensor([[1, 2]], dtype=torch.long),
            trim_samples=0,
        ),
        _DecodeRequest(
            request_id="long",
            codes_window=torch.tensor([[3, 4, 5, 6]], dtype=torch.long),
            trim_samples=0,
        ),
    ]
    with pytest.raises(IndexError):
        exe._forward_batch(requests)

    exe._pad_token_id = 0
    outputs = exe._forward_batch(requests)
    assert len(outputs) == 2


def test_forward_batch_matches_independent_single_forwards() -> None:
    exe = _make_executor(upsample=4)
    codes_list = [
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        torch.tensor([[4, 5, 6, 7, 8]], dtype=torch.long),
        torch.tensor([[9, 10]], dtype=torch.long),
    ]
    trims = [0, 4, 0]

    requests = [
        _DecodeRequest(
            request_id=f"r{i}",
            codes_window=codes,
            trim_samples=trim,
        )
        for i, (codes, trim) in enumerate(zip(codes_list, trims))
    ]

    batch_outputs = exe._forward_batch(requests)

    assert len(batch_outputs) == 3
    for i, (codes, trim) in enumerate(zip(codes_list, trims)):
        single = exe._vocoder_forward(codes, trim_samples=trim)
        assert (
            batch_outputs[i].shape == single.shape
        ), f"req{i}: batch shape {batch_outputs[i].shape} != single {single.shape}"
        np.testing.assert_allclose(
            batch_outputs[i],
            single,
            rtol=1e-5,
            err_msg=f"req{i}: batched output diverges from single-path output",
        )


async def _shutdown_batch_loop(exe: _Code2WavStreamingExecutor) -> None:
    if exe._batch_loop_task is None:
        return
    exe._batch_loop_task.cancel()
    try:
        await exe._batch_loop_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_single_pending_request_takes_fast_path() -> None:
    exe = _make_executor()
    exe._ensure_batch_loop()
    try:
        loop = asyncio.get_running_loop()
        codes = torch.tensor([[1, 2, 3]], dtype=torch.long)
        req = _DecodeRequest(
            request_id="r0",
            codes_window=codes,
            trim_samples=0,
            result_future=loop.create_future(),
        )
        exe._pending.put_nowait(req)

        wav = await req.result_future

        expected = exe._vocoder_forward(codes, trim_samples=0)
        np.testing.assert_allclose(wav, expected, rtol=1e-5)
    finally:
        await _shutdown_batch_loop(exe)


@pytest.mark.asyncio
async def test_multiple_pending_requests_are_batched() -> None:
    exe = _make_executor()
    exe._ensure_batch_loop()
    try:
        loop = asyncio.get_running_loop()
        codes_list = [
            torch.tensor([[1, 2, 3]], dtype=torch.long),
            torch.tensor([[4, 5, 6, 7, 8]], dtype=torch.long),
            torch.tensor([[9, 10]], dtype=torch.long),
        ]
        reqs = [
            _DecodeRequest(
                request_id=f"r{i}",
                codes_window=c,
                trim_samples=0,
                result_future=loop.create_future(),
            )
            for i, c in enumerate(codes_list)
        ]
        for r in reqs:
            exe._pending.put_nowait(r)

        results = await asyncio.gather(*(r.result_future for r in reqs))

        assert len(results) == 3
        for i, (got, codes) in enumerate(zip(results, codes_list)):
            expected = exe._vocoder_forward(codes, trim_samples=0)
            np.testing.assert_allclose(
                got,
                expected,
                rtol=1e-5,
                err_msg=f"req{i}: batched dispatch diverges from single-path",
            )
    finally:
        await _shutdown_batch_loop(exe)


@pytest.mark.asyncio
async def test_aborted_request_is_cancelled() -> None:
    exe = _make_executor()
    exe._ensure_batch_loop()
    try:
        loop = asyncio.get_running_loop()
        codes = torch.tensor([[1, 2, 3]], dtype=torch.long)
        req = _DecodeRequest(
            request_id="r0",
            codes_window=codes,
            trim_samples=0,
            result_future=loop.create_future(),
        )
        exe._aborted.add("r0")
        exe._pending.put_nowait(req)

        with pytest.raises(asyncio.CancelledError):
            await req.result_future
    finally:
        await _shutdown_batch_loop(exe)


@pytest.mark.asyncio
async def test_mixed_batch_live_plus_aborted() -> None:
    exe = _make_executor()
    exe._ensure_batch_loop()
    try:
        loop = asyncio.get_running_loop()
        live_codes = torch.tensor([[1, 2, 3]], dtype=torch.long)
        live = _DecodeRequest(
            request_id="live",
            codes_window=live_codes,
            trim_samples=0,
            result_future=loop.create_future(),
        )
        aborted = _DecodeRequest(
            request_id="dead",
            codes_window=torch.tensor([[9, 9, 9]], dtype=torch.long),
            trim_samples=0,
            result_future=loop.create_future(),
        )
        exe._aborted.add("dead")
        exe._pending.put_nowait(live)
        exe._pending.put_nowait(aborted)

        wav = await live.result_future
        with pytest.raises(asyncio.CancelledError):
            await aborted.result_future

        expected = exe._vocoder_forward(live_codes, trim_samples=0)
        np.testing.assert_allclose(wav, expected, rtol=1e-5)
    finally:
        await _shutdown_batch_loop(exe)
