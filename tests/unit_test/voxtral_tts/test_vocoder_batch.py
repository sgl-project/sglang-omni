# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Voxtral cross-request batched vocoder decode.

The batch path coalesces concurrent requests into a single
``decode_helper_batch_async`` call. With a deterministic fake tokenizer these
tests assert the batch path produces, per request, output identical to the
single path (warmup-trim / fade applied per request, results routed back to the
right payload) and that it issues exactly one decode call for the whole batch.

The fake decoder is deterministic, so equality here verifies routing/warmup/fade
only. The real on-device numerics — cross-request batching is quality-neutral
(bf16 batch-dependent conv-algo noise, not contamination, thanks to causality) —
are covered by the on-device WER validation, not here.
"""
from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import torch

from sglang_omni.models.voxtral_tts.io import VoxtralTTSState
from sglang_omni.models.voxtral_tts.pipeline import stages
from sglang_omni.proto import StagePayload


class _FakeTokenizer:
    """Returns a deterministic waveform per input so mis-routing is detectable."""

    downsample_factor = 100
    sampling_rate = 24_000

    def __init__(self) -> None:
        self.batch_lens: list[int] = []

    def decode_helper_batch_async(self, codes_list):
        self.batch_lens.append(len(codes_list))
        return [
            torch.arange(c.shape[0] * self.downsample_factor, dtype=torch.float32)
            for c in codes_list
        ]


def _payload(codes: torch.Tensor) -> StagePayload:
    state = VoxtralTTSState(audio_codes=codes)
    return StagePayload(request_id="r", request=Mock(), data=state.to_dict())


def _audio(payload: StagePayload) -> np.ndarray:
    return np.frombuffer(payload.data["audio_waveform"], dtype=np.float32)


def _make_scheduler(monkeypatch):
    fake = _FakeTokenizer()
    monkeypatch.setattr(stages, "_resolve_checkpoint", lambda p: p)
    monkeypatch.setattr(stages, "_load_audio_tokenizer", lambda *a, **k: fake)
    return stages.create_vocoder_executor("dummy", device="cpu"), fake


def test_create_vocoder_executor_enables_batching(monkeypatch):
    sched, _ = _make_scheduler(monkeypatch)
    assert sched._batch_fn is not None
    assert sched._max_batch_size == 4


def test_batch_matches_single_mixed_lengths(monkeypatch):
    sched, fake = _make_scheduler(monkeypatch)
    K = 37
    codes = [
        torch.randint(0, 1024, (t, K), dtype=torch.long) for t in (10, 25, 7)
    ]

    singles = [sched._fn(_payload(c.clone())) for c in codes]
    batched = sched._batch_fn([_payload(c.clone()) for c in codes])

    assert len(batched) == len(singles)
    for i in range(len(codes)):
        np.testing.assert_array_equal(_audio(batched[i]), _audio(singles[i]))

    # the batch path issued exactly one decode call covering all 3 requests
    assert fake.batch_lens[-1] == 3


def test_single_element_batch_matches_single(monkeypatch):
    sched, _ = _make_scheduler(monkeypatch)
    codes = torch.randint(0, 1024, (12, 37), dtype=torch.long)
    single = sched._fn(_payload(codes.clone()))
    batched = sched._batch_fn([_payload(codes.clone())])
    np.testing.assert_array_equal(_audio(batched[0]), _audio(single))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
