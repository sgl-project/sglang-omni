# SPDX-License-Identifier: Apache-2.0
"""ZONOS2 streaming-vocoder correctness tests (CPU-only, DAC mocked).

Regression coverage for the streaming emit-coalescing / tail-truncation class of
bug (PR #779): the streamed audio must reach the SAME aligned length as the
one-shot decode (after EOS trimming), at every emit chunk size, AND even when a
tail batch was never streamed -- `on_stream_done` must top the OLA decoder up
from the full code sequence (`audio_codes`) before the flush.

The DAC is mocked with a deterministic decode: each aligned frame becomes its
codebook-0 value repeated ``DAC_HOP_LENGTH`` times. So the streamed waveform is a
direct fingerprint of which frames were decoded -- a missing, duplicated, or
reordered tail is detectable by length and (with overlap=0) by content.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sglang_omni.models.zonos2.components import streaming_vocoder
from sglang_omni.models.zonos2.components.streaming_vocoder import (
    DAC_HOP_LENGTH,
    Zonos2StreamingVocoderScheduler,
)
from sglang_omni.models.zonos2.payload_types import (
    N_CODEBOOKS,
    ZONOS2_SAMPLE_RATE,
    Zonos2State,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload

HOP = DAC_HOP_LENGTH
WITHHOLD = N_CODEBOOKS - 1  # trailing delay/flush rows trimmed before decode


def _fake_decode(codes, eos_frame=None):
    """Deterministic stand-in for the DAC: codebook-0 of each aligned frame,
    repeated HOP times. Mirrors the real decode's length contract (drop the
    trailing WITHHOLD rows, cap at eos_frame). Column 0 is delay-0 so it is
    unchanged by shear_up -- a faithful fingerprint of the kept frames."""
    c = codes if codes.ndim == 2 else codes[0]
    valid = c.shape[0] - WITHHOLD
    if eos_frame is not None:
        valid = min(valid, max(0, int(eos_frame)))
    if valid <= 0:
        return torch.zeros(0, dtype=torch.float32)
    return c[:valid, 0].to(torch.float32).repeat_interleave(HOP)


class _FakeDAC:
    def decode(self, codes, eos_frame=None):
        return _fake_decode(codes, eos_frame)


@pytest.fixture(autouse=True)
def _mock_dac(monkeypatch):
    fake = _FakeDAC()
    monkeypatch.setattr(streaming_vocoder, "_get_vocoder", lambda device: fake)
    monkeypatch.setattr(
        streaming_vocoder,
        "decode_to_pcm",
        lambda codes, eos_frame=None, device="cpu": _fake_decode(
            torch.as_tensor(codes), eos_frame
        ),
    )


def _scheduler(steady=32, initial=5, overlap=2):
    return Zonos2StreamingVocoderScheduler(
        device="cpu",
        compute_fn=None,
        batch_compute_fn=None,
        steady_chunk_frames=steady,
        initial_chunk_frames=initial,
        overlap_frames=overlap,
    )


def _codes(n_frames: int) -> torch.Tensor:
    """Delayed [T, 9] rows; codebook-0 = frame index so the decoded waveform
    fingerprints which frames survived."""
    c = torch.zeros((n_frames, N_CODEBOOKS), dtype=torch.long)
    c[:, 0] = torch.arange(n_frames)
    return c


def _payload(codes: torch.Tensor, eos_frame: int, rid="req") -> StagePayload:
    state = Zonos2State(
        audio_codes=codes.clone(),
        eos_frame=eos_frame,
        sample_rate=ZONOS2_SAMPLE_RATE,
        prompt_tokens=1,
        completion_tokens=int(codes.shape[0]),
    )
    return StagePayload(
        request_id=rid,
        request=OmniRequest(inputs="", params={"stream": True}),
        data=state.to_dict(),
    )


def _meta():
    return {"stream": True, "modality": "audio_codes", "n_codebooks": N_CODEBOOKS}


def _pcm_from_messages(messages, rid) -> np.ndarray:
    chunks = []
    for m in messages:
        if getattr(m, "type", None) == "stream" and m.request_id == rid:
            d = m.data
            chunks.append(np.frombuffer(d["audio_waveform"], dtype=np.float32))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def _drive(sch, codes, eos_frame, *, emit, rid="req", stream_first_n=None):
    """Feed `codes` to the vocoder in coalesced batches of `emit` rows, then
    finish. `stream_first_n` streams only the first N rows (simulating a dropped
    tail) while the terminal payload still carries the full `audio_codes`."""
    sch._stream_payloads[rid] = _payload(codes, eos_frame, rid)
    meta = _meta()
    fed = codes if stream_first_n is None else codes[:stream_first_n]
    msgs = []
    cid = 0
    i = 0
    while i < fed.shape[0]:
        batch = fed[i : i + emit]
        msgs += sch.on_stream_chunk(
            rid,
            StreamItem(
                chunk_id=cid, data=batch.clone(), from_stage="tts_engine", metadata=meta
            ),
        )
        i += emit
        cid += 1
    msgs += sch.on_stream_done(rid)
    return _pcm_from_messages(msgs, rid)


def _aligned_len(codes, eos_frame):
    valid = min(codes.shape[0] - WITHHOLD, eos_frame)
    return max(0, valid) * HOP


# --- length alignment: streamed == one-shot aligned length, every emit size ---


@pytest.mark.parametrize("emit", [1, 8, 16, 32, 48])
def test_length_alignment_across_emit_sizes(emit):
    codes = _codes(220)
    eos = 200  # < 220 - WITHHOLD, so reference = 200 frames
    sch = _scheduler(steady=32, initial=5, overlap=2)
    pcm = _drive(sch, codes, eos, emit=emit)
    assert len(pcm) == _aligned_len(codes, eos)


def test_adaptive_emit_alignment():
    """Adaptive first-chunk batching (small first batch, large steady) must not
    change the aligned output length."""
    codes = _codes(220)
    eos = 200
    sch = _scheduler(steady=32, initial=5, overlap=2)
    # simulate adaptive at the message level: first batch 8, then steady 32
    rid = "req"
    sch._stream_payloads[rid] = _payload(codes, eos, rid)
    meta = _meta()
    sizes = [8] + [32] * 100
    msgs, i, cid = [], 0, 0
    for sz in sizes:
        if i >= codes.shape[0]:
            break
        batch = codes[i : i + sz]
        msgs += sch.on_stream_chunk(
            rid,
            StreamItem(
                chunk_id=cid, data=batch.clone(), from_stage="tts_engine", metadata=meta
            ),
        )
        i += sz
        cid += 1
    msgs += sch.on_stream_done(rid)
    assert len(_pcm_from_messages(msgs, rid)) == _aligned_len(codes, eos)


# --- the bug regression: on_stream_done tops up a dropped/held tail ---


@pytest.mark.parametrize("emit", [16, 32])
def test_on_stream_done_tops_up_dropped_tail(emit):
    """If a tail batch was never streamed (held/dropped under coalescing or
    retraction), on_stream_done must recover it from audio_codes so the output
    is NOT truncated. Without the top-up this returns ~half length."""
    codes = _codes(260)
    eos = 240
    sch = _scheduler(steady=32, initial=5, overlap=2)
    pcm = _drive(sch, codes, eos, emit=emit, stream_first_n=120)  # drop ~half the tail
    assert len(pcm) == _aligned_len(codes, eos)


def test_nothing_streamed_falls_back_to_full_decode():
    """No chunks streamed at all (slot-starved) -> on_stream_done still emits the
    full aligned audio."""
    codes = _codes(120)
    eos = 100
    sch = _scheduler(steady=32, initial=5, overlap=2)
    pcm = _drive(sch, codes, eos, emit=32, stream_first_n=0)
    assert len(pcm) == _aligned_len(codes, eos)


# --- content integrity (overlap=0 => exact concatenation, no cross-fade) ---


@pytest.mark.parametrize("emit", [1, 16, 32])
def test_content_no_missing_or_reordered_frames(emit):
    codes = _codes(160)
    eos = 140
    sch = _scheduler(steady=32, initial=5, overlap=0)
    pcm = _drive(sch, codes, eos, emit=emit)
    expected = np.repeat(np.arange(eos, dtype=np.float32), HOP)
    np.testing.assert_array_equal(pcm, expected)


# --- edge cases ---


def test_short_utterance_below_emit_size():
    """Utterance shorter than the emit chunk: nothing hits a steady emit, the
    whole thing rides the finish flush + top-up."""
    codes = _codes(20)  # < emit=32
    eos = 12
    sch = _scheduler(steady=32, initial=5, overlap=2)
    pcm = _drive(sch, codes, eos, emit=32)
    assert len(pcm) == _aligned_len(codes, eos)


@pytest.mark.parametrize("eos", [31, 32, 33, 64, 65])
def test_eos_near_chunk_boundary(eos):
    # eos_frame sits ~10 frames before the end (the real post-EOS countdown), so
    # it falls inside the steady decoder's held-back tail and the flush emits
    # exactly up to eos -- regardless of where eos lands vs the 32-frame boundary.
    codes = _codes(eos + 10)
    sch = _scheduler(steady=32, initial=5, overlap=2)
    pcm = _drive(sch, codes, eos, emit=32)
    assert len(pcm) == _aligned_len(codes, eos)
