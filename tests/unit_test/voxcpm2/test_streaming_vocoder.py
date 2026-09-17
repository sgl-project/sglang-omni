# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 streaming vocoder: chunk cadence, overlap accounting, abort cleanup."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sglang_omni.models.voxcpm2.streaming_vocoder import VoxCPM2StreamingVocoder
from sglang_omni.pipeline.stage.stream_queue import StreamItem

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


class _FakeRequest:
    def __init__(self, stream: bool):
        self.inputs = "hi"
        self.params = {"stream": stream}
        self.metadata = {}


class _FakePayload:
    def __init__(self, request_id: str, *, stream: bool = True):
        self.request_id = request_id
        self.request = _FakeRequest(stream)
        self.data = {}


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


def _item(patch: torch.Tensor, chunk_id: int) -> StreamItem:
    return StreamItem(
        chunk_id=chunk_id,
        data=patch,
        from_stage="tts_engine",
        metadata={"modality": "audio_latents", "stream": True},
    )


def _open(vocoder, request_id="req"):
    """Register the request the way the scheduler does before any chunk."""
    vocoder.handle_streaming_new_request(request_id, _FakePayload(request_id))


def _feed(vocoder, count, *, request_id="req"):
    emitted = []
    for index in range(count):
        emitted.extend(vocoder.on_stream_chunk(request_id, _item(_patch(index), index)))
    return emitted


def _audio(message) -> np.ndarray:
    payload = message.data
    return np.frombuffer(payload["audio_waveform"], dtype=np.float32)


def test_a_streaming_request_is_recognized_by_its_params():
    vocoder = _vocoder()
    assert vocoder.is_streaming_payload(_FakePayload("req", stream=True))
    assert not vocoder.is_streaming_payload(_FakePayload("req", stream=False))


def test_a_chunk_that_is_not_a_tensor_is_rejected():
    """The relay only carries tensors, so a wrapped patch is a wiring bug."""
    vocoder = _vocoder()
    _open(vocoder)
    item = _item(_patch(0), 0)
    item.data = {"patch": _patch(0)}
    with pytest.raises(TypeError):
        vocoder.on_stream_chunk("req", item)


def test_a_patch_of_the_wrong_shape_is_rejected():
    vocoder = _vocoder()
    _open(vocoder)
    with pytest.raises(ValueError):
        vocoder.on_stream_chunk("req", _item(torch.zeros(_PATCH_SIZE + 1, 2), 0))


def test_no_audio_before_the_first_stride_is_reached():
    vocoder = _vocoder(stream_stride=4)
    _open(vocoder)
    assert _feed(vocoder, 3) == []


def test_first_chunk_lands_on_the_stride():
    vocoder = _vocoder(stream_stride=4)
    _open(vocoder)
    emitted = _feed(vocoder, 4)
    assert len(emitted) == 1
    assert emitted[0].type == "stream"
    assert _audio(emitted[0]).size == 4 * _SAMPLES_PER_PATCH


def test_a_chunk_carries_the_sample_rate_the_client_needs():
    vocoder = _vocoder(stream_stride=4)
    _open(vocoder)
    emitted = _feed(vocoder, 4)
    assert emitted[0].data["sample_rate"] == _FakeAudioVAE.out_sample_rate
    assert emitted[0].data["modality"] == "audio"


def test_later_chunks_follow_the_followup_stride():
    vocoder = _vocoder(stream_stride=4, stream_followup_stride=2)
    _open(vocoder)
    emitted = _feed(vocoder, 8)
    # One chunk at 4 patches, then one every 2: at 6 and at 8.
    assert len(emitted) == 3


def test_overlap_is_decoded_again_and_dropped_from_the_output():
    """Re-decoded patches give the causal decoder context; they must not ship."""
    vocoder = _vocoder(stream_stride=4, stream_followup_stride=2, overlap_patches=1)
    _open(vocoder)
    emitted = _feed(vocoder, 6)
    vae = vocoder.vae

    # Second window re-decodes the 1 overlap patch plus the 2 new ones.
    assert vae.decode_calls[-1] == 3 * _PATCH_SIZE
    # Only the new patches reach the caller.
    assert _audio(emitted[-1]).size == 2 * _SAMPLES_PER_PATCH


def test_emitted_audio_is_contiguous_across_chunks():
    """The ramp makes a dropped or repeated sample visible as a jump."""
    vocoder = _vocoder(stream_stride=2, stream_followup_stride=2, overlap_patches=1)
    _open(vocoder)
    emitted = _feed(vocoder, 6)
    stitched = np.concatenate([_audio(message) for message in emitted])
    # Each window restarts its ramp at 0, so the seam repeats the overlap span.
    assert stitched.size == 6 * _SAMPLES_PER_PATCH


def test_stream_done_flushes_the_tail_then_closes_the_request():
    vocoder = _vocoder(stream_stride=4)
    _open(vocoder)
    _feed(vocoder, 5)
    flushed = vocoder.on_stream_done("req")
    assert [message.type for message in flushed] == ["stream", "result"]
    assert _audio(flushed[0]).size == 1 * _SAMPLES_PER_PATCH


def test_the_terminal_result_does_not_repeat_the_streamed_audio():
    """The client already has every sample; a waveform here would play twice."""
    vocoder = _vocoder(stream_stride=4)
    _open(vocoder)
    _feed(vocoder, 4)
    result = vocoder.on_stream_done("req")[-1]
    assert result.type == "result"
    assert "audio_waveform" not in result.data.data
    assert result.data.data["sample_rate"] == _FakeAudioVAE.out_sample_rate


def test_stream_done_emits_only_the_result_when_everything_already_shipped():
    vocoder = _vocoder(stream_stride=4)
    _open(vocoder)
    _feed(vocoder, 4)
    assert [message.type for message in vocoder.on_stream_done("req")] == ["result"]


def test_requests_do_not_share_accumulated_patches():
    vocoder = _vocoder(stream_stride=4)
    _open(vocoder, "a")
    _open(vocoder, "b")
    _feed(vocoder, 3, request_id="a")
    assert _feed(vocoder, 3, request_id="b") == []
    assert len(_feed(vocoder, 1, request_id="a")) == 1


def test_clearing_one_request_leaves_the_other_alone():
    vocoder = _vocoder(stream_stride=4)
    _open(vocoder, "a")
    _open(vocoder, "b")
    _feed(vocoder, 3, request_id="a")
    _feed(vocoder, 3, request_id="b")

    vocoder.clear_stream_state("a")
    assert "a" not in vocoder.stream_states
    assert "b" in vocoder.stream_states


def test_clearing_is_idempotent():
    """The base scheduler calls this more than once for the same id, by design."""
    vocoder = _vocoder()
    _open(vocoder)
    _feed(vocoder, 2)
    vocoder.clear_stream_state("req")
    vocoder.clear_stream_state("req")
    assert [message.type for message in vocoder.on_stream_done("req")] == ["result"]


class _IdentityAudioVAE(_FakeAudioVAE):
    def decode(self, latents, sample_rate=None):
        # Each latent frame has its own sample value, independent of the window.
        return latents[:, :1].repeat_interleave(_DECODE_CHUNK, dim=-1)


@pytest.mark.parametrize("context_len", [0, 1, 3])
def test_stream_samples_match_full_decode_without_prompt_or_duplicate_tail(context_len):
    vocoder = VoxCPM2StreamingVocoder(
        _IdentityAudioVAE(),
        device="cpu",
        patch_size=_PATCH_SIZE,
        stream_stride=4,
        stream_followup_stride=2,
        overlap_patches=1,
    )
    payload = _FakePayload("req")
    payload.data["context_len"] = context_len
    vocoder.handle_streaming_new_request("req", payload)
    patches = [_patch(index + 1) for index in range(context_len + 7)]
    messages = []
    for index, patch in enumerate(patches):
        item = _item(patch, index)
        item.metadata["context_len"] = context_len
        emitted = vocoder.on_stream_chunk("req", item)
        if index < context_len + 3:
            assert emitted == []
        messages.extend(emitted)
    vocoder.handle_stream_done("req")
    while not vocoder.outbox.empty():
        messages.append(vocoder.outbox.get_nowait())
    audio = np.concatenate([_audio(m) for m in messages if m.type == "stream"])
    expected = np.repeat(
        np.arange(context_len + 1, context_len + 8), _SAMPLES_PER_PATCH
    )
    np.testing.assert_array_equal(audio, expected)
    assert [m.type for m in messages].count("result") == 1
    assert "req" not in vocoder.stream_states


def test_stream_rejects_context_changes():
    vocoder = _vocoder()
    state = vocoder.create_stream_state("req")
    vocoder.latch_stream_contract(
        "req", state, {"context_len": 3}, origin="stream metadata"
    )
    with pytest.raises(ValueError, match="changed"):
        vocoder.latch_stream_contract(
            "req", state, {"context_len": 2}, origin="stream metadata"
        )


def test_full_and_fallback_decode_trim_the_same_prompt_context():
    from sglang_omni.models.voxcpm2.payload_types import VoxCPM2State
    from sglang_omni.scheduling.pipeline_state import store_state

    vocoder = VoxCPM2StreamingVocoder(
        _IdentityAudioVAE(), device="cpu", patch_size=_PATCH_SIZE
    )
    latents = torch.arange(5 * _PATCH_SIZE).float().expand(_FEAT_DIM, -1)
    payload = _FakePayload("req", stream=False)
    state = VoxCPM2State(generated_latents=latents, context_len=2)
    payload = store_state(payload, state)
    expected = latents[0, 2 * _PATCH_SIZE :].repeat_interleave(_DECODE_CHUNK)
    fallback = vocoder.fallback_full_decode(
        "req", payload, vocoder.create_stream_state("req")
    )
    torch.testing.assert_close(fallback.flatten(), expected)
    result = vocoder.decode_payload(payload)
    np.testing.assert_array_equal(
        np.frombuffer(result.data["audio_waveform"], dtype=np.float32), expected.numpy()
    )
