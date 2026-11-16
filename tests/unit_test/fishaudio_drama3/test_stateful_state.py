# SPDX-License-Identifier: Apache-2.0
"""Numerical isolation and ownership checks for the fixed codec arena."""

import pytest
import torch

from sglang_omni.models.fishaudio_drama3.stateful_decoder import StatefulDACDecoder
from sglang_omni.models.fishaudio_drama3.stateful_state import DecoderStateArena
from tests.unit_test.fixtures.fishaudio_codec import tiny_fishaudio_codec


@torch.no_grad()
def test_window_128_interleaved_reset_and_exact_shapes() -> None:
    codec = tiny_fishaudio_codec()
    codec.quantizer.post_module.window_size = 128
    codes = torch.randint(0, 16, (2, 10, 500))
    expected = codec.from_indices(codes.clone())
    decoder = StatefulDACDecoder(codec, capacity=2)
    states = [decoder.create_state(), decoder.create_state()]
    tensors = decoder.arena.persistent_tensors()
    addresses = [tensor.data_ptr() for tensor in tensors]
    outputs: list[list[torch.Tensor]] = [[], []]
    frame_counts = (10, 10, 12, 32, 64, 1, 2, 4, 63)
    position = 0
    chunk_index = 0
    while position < 500:
        end = min(position + frame_counts[chunk_index % len(frame_counts)], 500)
        for slot_index in (1, 0):
            outputs[slot_index].append(
                decoder.decode_delta(
                    codes[slot_index : slot_index + 1, :, position:end],
                    states[slot_index],
                )
            )
        assert [
            tensor.data_ptr() for tensor in decoder.arena.persistent_tensors()
        ] == addresses
        assert decoder.arena.positions.tolist() == [end, end, 0]
        position = end
        chunk_index += 1
    for slot_index in (0, 1):
        torch.testing.assert_close(
            torch.cat(outputs[slot_index], dim=-1),
            expected[slot_index : slot_index + 1],
            atol=1e-4,
            rtol=1e-4,
        )
    decoder.release_state(states[0])
    with pytest.raises(ValueError, match="stale"):
        decoder.decode_delta(codes[:1, :, :1], states[0])
    fresh = decoder.create_state(0)
    assert fresh.generation != states[0].generation
    assert all(torch.count_nonzero(tensor[0]) == 0 for tensor in tensors)
    torch.testing.assert_close(
        decoder.decode_delta(codes[:1, :, :12], fresh),
        expected[:1, :, : 12 * decoder.frame_length],
        atol=1e-4,
        rtol=1e-4,
    )
    assert [
        tensor.data_ptr() for tensor in decoder.arena.persistent_tensors()
    ] == addresses
    assert decoder.arena.positions[1] == 500
    assert decoder.arena.nbytes == sum(
        tensor.numel() * tensor.element_size() for tensor in tensors
    )
    assert (
        DecoderStateArena.required_nbytes(codec, 2, torch.float32)
        == decoder.arena.nbytes
    )
    assert (
        DecoderStateArena.required_nbytes(codec, 100_000, torch.float32)
        == decoder.arena.nbytes // 3 * 100_001
    )


@torch.no_grad()
def test_every_exact_tail_length_and_slot_input_changes() -> None:
    codec = tiny_fishaudio_codec()
    decoder = StatefulDACDecoder(codec, capacity=1)
    codes = torch.randint(0, 16, (1, 10, 64))
    device_slot = torch.tensor([decoder.scratch_slot_index], dtype=torch.int64)
    addresses = [tensor.data_ptr() for tensor in decoder.arena.persistent_tensors()]
    for frame_count in range(1, 65):
        decoder.reset_slot(decoder.scratch_slot_index)
        actual = decoder.decode_slot(codes[..., :frame_count], device_slot, frame_count)
        expected = codec.from_indices(codes[..., :frame_count].clone())
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        assert decoder.arena.positions.tolist() == [0, frame_count]
        assert [
            tensor.data_ptr() for tensor in decoder.arena.persistent_tensors()
        ] == addresses
    decoder.reset_slot(decoder.scratch_slot_index)
    assert all(
        torch.count_nonzero(tensor) == 0
        for tensor in decoder.arena.persistent_tensors()
    )
    with pytest.raises(ValueError, match="production"):
        decoder.create_state(decoder.scratch_slot_index)
    state = decoder.create_state()
    device_slot.fill_(state.slot_index)
    torch.testing.assert_close(
        decoder.decode_slot(codes[..., :10], device_slot, 10),
        codec.from_indices(codes[..., :10].clone()),
        atol=1e-4,
        rtol=1e-4,
    )


@torch.no_grad()
def test_slot_exhaustion_stale_generation_and_prelaunch_validation() -> None:
    codec = tiny_fishaudio_codec()
    decoder = StatefulDACDecoder(codec, capacity=1)
    state = decoder.create_state(0)
    codes = torch.randint(0, 16, (1, 10, 65))
    with pytest.raises(ValueError, match="active lease"):
        decoder.create_state(0)
    with pytest.raises(ValueError, match="capacity"):
        decoder.create_state()
    with pytest.raises(ValueError, match="64 frames"):
        decoder.decode_delta(codes, state)
    assert state.processed_frames == 0
    assert all(
        torch.count_nonzero(tensor) == 0
        for tensor in decoder.arena.persistent_tensors()
    )
    decoder.reset(0)
    with pytest.raises(ValueError, match="stale"):
        decoder.validate_delta(codes[..., :1], state)
    replacement = decoder.create_state(0)
    decoder.decode_delta(codes[..., :0], replacement, is_final=True)
    assert replacement.is_final and replacement.processed_frames == 0
    assert decoder.arena.positions.tolist() == [0, 0]
    decoder.release_state(replacement)
    with pytest.raises(ValueError, match="stale"):
        decoder.release_state(replacement)
    for invalid_slot in (-1, 2):
        with pytest.raises(ValueError, match="outside"):
            decoder.reset(invalid_slot)
