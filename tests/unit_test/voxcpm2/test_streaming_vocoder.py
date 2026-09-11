# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 streaming vocoder: chunk cadence, overlap accounting, abort cleanup."""

from __future__ import annotations

import torch

from sglang_omni.models.voxcpm2.streaming_vocoder import VoxCPM2StreamingVocoder

_PATCH_SIZE = 4
_FEAT_DIM = 8
_DECODE_CHUNK = 3
_SAMPLES_PER_PATCH = _DECODE_CHUNK * _PATCH_SIZE


class _FakeAudioVAE:
    """Decodes latents to a ramp so every sample says which frame produced it."""

    decode_chunk_size = _DECODE_CHUNK
    out_sample_rate = 48000

    def __init__(self):
        self.decode_calls: list[int] = []

    def decode(self, latents, sample_rate=None):
        del sample_rate
        frames = int(latents.shape[-1])
        self.decode_calls.append(frames)
        samples = frames * _DECODE_CHUNK
        return latents.new_tensor(range(samples)).reshape(1, 1, samples)


def _vocoder(**kwargs):
    defaults = dict(
        device="cpu",
        patch_size=_PATCH_SIZE,
        stream_stride=4,
        stream_followup_stride=2,
        overlap_patches=1,
    )
    defaults.update(kwargs)
    return VoxCPM2StreamingVocoder(_FakeAudioVAE(), **defaults)


def _patch(value: float) -> torch.Tensor:
    return torch.full((_PATCH_SIZE, _FEAT_DIM), value)


def _feed(vocoder, count, *, request_id="req"):
    emitted = []
    for index in range(count):
        emitted.extend(vocoder.on_stream_chunk(request_id, {"patch": _patch(index)}))
    return emitted


def test_no_audio_before_the_first_stride_is_reached():
    vocoder = _vocoder(stream_stride=4)
    assert _feed(vocoder, 3) == []


def test_first_chunk_lands_on_the_stride():
    vocoder = _vocoder(stream_stride=4)
    emitted = _feed(vocoder, 4)
    assert len(emitted) == 1
    assert emitted[0].type == "stream"
    assert emitted[0].data["audio"].numel() == 4 * _SAMPLES_PER_PATCH


def test_later_chunks_follow_the_followup_stride():
    vocoder = _vocoder(stream_stride=4, stream_followup_stride=2)
    emitted = _feed(vocoder, 8)
    # One chunk at 4 patches, then one every 2: at 6 and at 8.
    assert len(emitted) == 3


def test_overlap_is_decoded_again_and_dropped_from_the_output():
    """Re-decoded patches give the causal decoder context; they must not ship."""
    vocoder = _vocoder(stream_stride=4, stream_followup_stride=2, overlap_patches=1)
    emitted = _feed(vocoder, 6)
    vae = vocoder._vae

    # Second window re-decodes the 1 overlap patch plus the 2 new ones.
    assert vae.decode_calls[-1] == 3 * _PATCH_SIZE
    # Only the new patches reach the caller.
    assert emitted[-1].data["audio"].numel() == 2 * _SAMPLES_PER_PATCH


def test_emitted_audio_is_contiguous_across_chunks():
    """The ramp makes a dropped or repeated sample visible as a jump."""
    vocoder = _vocoder(stream_stride=2, stream_followup_stride=2, overlap_patches=1)
    emitted = _feed(vocoder, 6)
    stitched = torch.cat([message.data["audio"] for message in emitted])
    # Each window restarts its ramp at 0, so the seam repeats the overlap span.
    assert stitched.numel() == 6 * _SAMPLES_PER_PATCH


def test_stream_done_flushes_the_tail():
    vocoder = _vocoder(stream_stride=4)
    _feed(vocoder, 5)
    flushed = vocoder.on_stream_done("req")
    assert len(flushed) == 1
    assert flushed[0].data["audio"].numel() == 1 * _SAMPLES_PER_PATCH


def test_stream_done_emits_nothing_when_everything_already_shipped():
    vocoder = _vocoder(stream_stride=4)
    _feed(vocoder, 4)
    assert vocoder.on_stream_done("req") == []


def test_requests_do_not_share_accumulated_patches():
    vocoder = _vocoder(stream_stride=4)
    _feed(vocoder, 3, request_id="a")
    assert _feed(vocoder, 3, request_id="b") == []
    assert len(_feed(vocoder, 1, request_id="a")) == 1


def test_clearing_one_request_leaves_the_other_alone():
    vocoder = _vocoder(stream_stride=4)
    _feed(vocoder, 3, request_id="a")
    _feed(vocoder, 3, request_id="b")

    vocoder.clear_stream_state("a")
    assert "a" not in vocoder._states
    assert "b" in vocoder._states


def test_clearing_is_idempotent():
    """The base scheduler calls this more than once for the same id, by design."""
    vocoder = _vocoder()
    _feed(vocoder, 2)
    vocoder.clear_stream_state("req")
    vocoder.clear_stream_state("req")
    assert vocoder.on_stream_done("req") == []
