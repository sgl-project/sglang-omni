# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import sys
from typing import ClassVar

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State


class _StubEstimator(torch.nn.Module):
    def forward(self, x, mask, mu, t, spks, cond, *, streaming):
        del t, spks, cond, streaming
        return (0.1 * x + mu) * mask


class _SolvableDecoder:
    def __init__(self, channels: int) -> None:
        self.rand_noise = torch.zeros(1, channels, 1000)
        self.t_scheduler = "cosine"
        self.inference_cfg_rate = 0.7
        self.estimator = _StubEstimator()

    def forward_estimator(self, x, mask, mu, t, spks, cond, *, streaming):
        return self.estimator(x, mask, mu, t, spks, cond, streaming=streaming)


class _SolvableFlow(torch.nn.Module):
    def __init__(self, channels: int = 80) -> None:
        super().__init__()
        self.output_size = channels
        self.token_mel_ratio = 2
        self.input_embedding = torch.nn.Embedding(32, channels)
        self.spk_embed_affine_layer = torch.nn.Linear(192, channels)
        self.pre_lookahead_layer = torch.nn.Identity()
        self.decoder = _SolvableDecoder(channels)


class _FakeHiFT(torch.nn.Module):
    # cosyvoice3.yaml: upsample_rates [8, 5, 3], istft_params.hop_len 4.
    upsample_rates: ClassVar[list[int]] = [8, 5, 3]
    istft_params: ClassVar[dict[str, int]] = {"n_fft": 16, "hop_len": 4}

    def inference(self, *, speech_feat, finalize):
        del finalize
        batch, _, frames = speech_feat.shape
        return speech_feat.new_zeros(batch, frames * 480), None


def _decode_and_collect(flow: torch.nn.Module, caplog) -> list[str]:
    vocoder = stages._CosyVoice3Vocoder(flow, _FakeHiFT())
    state = FunCosyVoice3State(flow_embedding=torch.ones(1, 192))
    items = [(state, torch.tensor([1, 2])), (state, torch.tensor([3]))]
    with caplog.at_level(logging.DEBUG, logger=stages.logger.name):
        asyncio.run(vocoder.decode_batch(items))
    return [r.getMessage() for r in caplog.records if "flow solve:" in r.getMessage()]


def test_decode_batch_logs_one_line_per_solve(caplog) -> None:
    # Both requests share one frame bucket, so one solve and one line.
    [message] = _decode_and_collect(_SolvableFlow(), caplog)
    assert "batch_items=2" in message
    assert float(message.rsplit("solve_elapsed_ms=", 1)[1]) > 0.0


def test_pending_solve_events_are_skipped_not_awaited(caplog) -> None:
    class _PendingEvent:
        def query(self) -> bool:
            return False

        def synchronize(self) -> None:
            raise AssertionError("must not wait on the device")

    flow = stages.FunCosyVoice3Flow(_SolvableFlow())
    flow._last_solve = (1, _PendingEvent(), _PendingEvent())
    with caplog.at_level(logging.DEBUG, logger=stages.logger.name):
        flow.log_last_solve()
    assert not [r for r in caplog.records if "flow solve:" in r.getMessage()]


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_decode_batch_logs_stream_event_time_on_cuda(caplog) -> None:
    [message] = _decode_and_collect(_SolvableFlow().cuda(), caplog)
    assert "batch_items=2" in message
    assert float(message.rsplit("solve_elapsed_ms=", 1)[1]) > 0.0


def test_modelscope_guard_restores_changed_root_handler_levels(
    monkeypatch, tmp_path
) -> None:
    root = logging.getLogger()
    demoted = logging.StreamHandler()
    demoted.setLevel(logging.INFO)
    untouched = logging.FileHandler(tmp_path / "log.txt")
    untouched.setLevel(logging.WARNING)
    root.addHandler(demoted)
    root.addHandler(untouched)
    monkeypatch.delitem(sys.modules, "modelscope", raising=False)

    def _import_like_modelscope(name: str):
        assert name == "modelscope"
        demoted.setLevel(logging.ERROR)
        raise ImportError("side effects happen before the failure too")

    monkeypatch.setattr(stages.importlib, "import_module", _import_like_modelscope)
    try:
        stages._import_modelscope_preserving_root_handlers()
        assert demoted.level == logging.INFO
        assert untouched.level == logging.WARNING
    finally:
        root.removeHandler(demoted)
        root.removeHandler(untouched)
        untouched.close()
