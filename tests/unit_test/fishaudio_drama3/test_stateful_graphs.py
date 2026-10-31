# SPDX-License-Identifier: Apache-2.0
"""CPU graph-boundary checks using the actual fixed-arena codec math."""

from dataclasses import dataclass, field

import numpy as np
import pytest
import torch

from sglang_omni.models.fishaudio_drama3.config import Drama3VocoderFactoryArgs
from sglang_omni.models.fishaudio_drama3.stateful_decoder import StatefulDACDecoder
from sglang_omni.models.fishaudio_drama3.stateful_graphs import (
    CAPTURE_FRAME_COUNTS,
    DeltaGraph,
    StatefulGraphDecoder,
)
from sglang_omni.models.fishaudio_drama3.vocoder import Drama3VocoderScheduler
from sglang_omni.scheduling.audio_stream import StreamVocoderState
from tests.unit_test.fixtures.fishaudio_codec import tiny_fishaudio_codec


@dataclass(kw_only=True)
class CpuDeltaGraph:
    backend: StatefulDACDecoder
    codes: torch.Tensor
    device_slot_index: torch.Tensor
    frame_count: int
    audio: torch.Tensor
    replay_count: int = 0
    should_fail: bool = False

    def replay(self) -> torch.Tensor:
        self.replay_count += 1
        self.audio.copy_(
            self.backend.decode_slot(
                self.codes, self.device_slot_index, self.frame_count
            )
        )
        if self.should_fail:
            raise RuntimeError("Injected replay failure after arena mutation")
        else:
            return self.audio


@dataclass(kw_only=True)
class CpuGraphBoundary:
    graphs: dict[int, CpuDeltaGraph] = field(default_factory=dict)
    capture_slots: list[int] = field(default_factory=list)
    operations: list[str] = field(default_factory=list)
    should_fail_capture: bool = False

    def capture(
        self,
        backend: StatefulDACDecoder,
        codes: torch.Tensor,
        device_slot_index: torch.Tensor,
        frame_count: int,
    ) -> DeltaGraph:
        self.capture_slots.append(int(device_slot_index[0]))
        audio = backend.decode_slot(codes, device_slot_index, frame_count)
        if self.should_fail_capture:
            raise RuntimeError("Injected capture failure after arena mutation")
        else:
            graph = CpuDeltaGraph(
                backend=backend,
                codes=codes,
                device_slot_index=device_slot_index,
                frame_count=frame_count,
                audio=audio,
            )
            self.graphs[frame_count] = graph
            return graph

    def wait_before_submit(self) -> None:
        self.operations.append("wait")

    def record_submission(self) -> None:
        self.operations.append("record")

    def synchronize(self) -> None:
        self.operations.append("synchronize")


@pytest.fixture
def decoder_pair() -> tuple[StatefulGraphDecoder, StatefulDACDecoder, CpuGraphBoundary]:
    codec = tiny_fishaudio_codec()
    backend = StatefulDACDecoder(codec, capacity=2)
    reference = StatefulDACDecoder(codec, capacity=2)
    boundary = CpuGraphBoundary()
    decoder = StatefulGraphDecoder(backend, graph_boundary=boundary)
    return decoder, reference, boundary


def test_capture_uses_only_scratch_and_preserves_live_arena() -> None:
    backend = StatefulDACDecoder(tiny_fishaudio_codec(), capacity=2)
    state = backend.create_state(0)
    codes = torch.randint(0, 16, (1, backend.num_codebooks, 5))
    backend.decode_delta(codes, state)
    before = [tensor.clone() for tensor in backend.arena.persistent_tensors()]
    boundary = CpuGraphBoundary()
    decoder = StatefulGraphDecoder(backend, graph_boundary=boundary)
    assert tuple(boundary.graphs) == CAPTURE_FRAME_COUNTS
    assert boundary.capture_slots == [backend.scratch_slot_index] * 64
    assert state.processed_frames == 5
    assert backend.arena.host_processed_frames[backend.scratch_slot_index] == 0
    assert backend.scratch_slot_index not in backend.arena.active_slots
    for previous, current in zip(
        before, backend.arena.persistent_tensors(), strict=True
    ):
        torch.testing.assert_close(
            current[: backend.capacity], previous[: backend.capacity]
        )
        assert torch.count_nonzero(current[backend.scratch_slot_index]) == 0
    assert len(decoder.graphs) == 64


@pytest.mark.parametrize("tail_frame_count", [1, 2, 4, 11, 31, 63])
def test_graph_slot_switch_tail_and_owned_output_match_eager(
    decoder_pair: tuple[StatefulGraphDecoder, StatefulDACDecoder, CpuGraphBoundary],
    tail_frame_count: int,
) -> None:
    decoder, reference, boundary = decoder_pair
    states = [decoder.create_state(slot_index) for slot_index in range(2)]
    reference_states = [reference.create_state(slot_index) for slot_index in range(2)]
    retained_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    input_addresses = {
        frame_count: codes.data_ptr()
        for frame_count, codes in decoder.static_codes.items()
    }
    slot_address = decoder.device_slot_index.data_ptr()
    arena_addresses = [
        tensor.data_ptr() for tensor in decoder.backend.arena.persistent_tensors()
    ]
    for slot_index, frame_count in (
        (0, 10),
        (1, 10),
        (0, 12),
        (1, tail_frame_count),
        (0, 32),
        (1, 64),
        (0, 64),
        (0, 64),
        (1, 10),
    ):
        codes = torch.randint(0, 16, (1, decoder.backend.num_codebooks, frame_count))
        expected = reference.decode_delta(codes, reference_states[slot_index])
        actual = decoder.decode_delta(codes, states[slot_index])
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        retained_outputs.append((actual, actual.clone()))
        assert (
            states[slot_index].processed_frames
            == reference_states[slot_index].processed_frames
        )
    for actual, snapshot in retained_outputs:
        torch.testing.assert_close(actual, snapshot)
    assert decoder.device_slot_index.data_ptr() == slot_address
    assert {
        count: codes.data_ptr() for count, codes in decoder.static_codes.items()
    } == input_addresses
    assert [
        tensor.data_ptr() for tensor in decoder.backend.arena.persistent_tensors()
    ] == arena_addresses
    assert sum(graph.replay_count for graph in boundary.graphs.values()) == 9
    assert len(boundary.graphs) == 64
    for actual, expected in zip(
        decoder.backend.arena.persistent_tensors(),
        reference.arena.persistent_tensors(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_empty_final_does_not_submit_or_advance_device_position(
    decoder_pair: tuple[StatefulGraphDecoder, StatefulDACDecoder, CpuGraphBoundary],
) -> None:
    decoder, _, boundary = decoder_pair
    state = decoder.create_state()
    codes = torch.randint(0, 16, (1, decoder.backend.num_codebooks, 10))
    decoder.decode_delta(codes, state)
    positions = decoder.backend.arena.positions.clone()
    operations = list(boundary.operations)
    output = decoder.decode_delta(codes[..., :0], state, is_final=True)
    assert output.shape == (1, 1, 0)
    assert state.is_final
    assert state.processed_frames == 10
    assert boundary.operations == operations
    torch.testing.assert_close(decoder.backend.arena.positions, positions)
    with pytest.raises(ValueError):
        decoder.decode_delta(codes, state)


@pytest.mark.parametrize(
    "invalid_kind",
    ["foreign", "stale", "range", "processed", "rope", "dtype", "batch", "oversize"],
)
def test_validation_precedes_input_copy_and_submission(
    decoder_pair: tuple[StatefulGraphDecoder, StatefulDACDecoder, CpuGraphBoundary],
    invalid_kind: str,
) -> None:
    decoder, reference, boundary = decoder_pair
    state = decoder.create_state(0)
    codes = torch.zeros((1, decoder.backend.num_codebooks, 10), dtype=torch.long)
    if invalid_kind == "foreign":
        state = reference.create_state(0)
    elif invalid_kind == "stale":
        decoder.reset(0)
    elif invalid_kind == "range":
        state.slot_index = decoder.backend.capacity + 1
    elif invalid_kind == "processed":
        state.processed_frames = 1
    elif invalid_kind == "rope":
        state.processed_frames = decoder.backend.rope_capacity
        decoder.backend.arena.host_processed_frames[state.slot_index] = (
            state.processed_frames
        )
    elif invalid_kind == "dtype":
        codes = codes.float()
    elif invalid_kind == "batch":
        codes = codes.expand(2, -1, -1)
    else:
        codes = torch.zeros((1, decoder.backend.num_codebooks, 65), dtype=torch.long)
    inputs = {count: tensor.clone() for count, tensor in decoder.static_codes.items()}
    slot_index = decoder.device_slot_index.clone()
    operations = list(boundary.operations)
    with pytest.raises((ValueError, RuntimeError)):
        decoder.decode_delta(codes, state)
    assert boundary.operations == operations
    torch.testing.assert_close(decoder.device_slot_index, slot_index)
    for count, tensor in decoder.static_codes.items():
        torch.testing.assert_close(tensor, inputs[count])
    assert not decoder.is_stopped


def test_release_waits_then_reuse_is_cold_and_stale_lease_rejected(
    decoder_pair: tuple[StatefulGraphDecoder, StatefulDACDecoder, CpuGraphBoundary],
) -> None:
    decoder, reference, boundary = decoder_pair
    state = decoder.create_state(0)
    codes = torch.randint(0, 16, (1, decoder.backend.num_codebooks, 10))
    first = decoder.decode_delta(codes, state)
    decoder.release_state(state)
    assert boundary.operations[-1] == "synchronize"
    replacement = decoder.create_state(0)
    expected = reference.decode_delta(codes, reference.create_state(0))
    torch.testing.assert_close(decoder.decode_delta(codes, replacement), expected)
    torch.testing.assert_close(first, expected)
    with pytest.raises(ValueError):
        decoder.decode_delta(codes, state)


def test_replay_failure_resets_affected_slot_and_stops_not_eager_fallback(
    decoder_pair: tuple[StatefulGraphDecoder, StatefulDACDecoder, CpuGraphBoundary],
) -> None:
    decoder, _, boundary = decoder_pair
    state = decoder.create_state(0)
    boundary.graphs[10].should_fail = True
    codes = torch.zeros((1, decoder.backend.num_codebooks, 10), dtype=torch.long)
    with pytest.raises(RuntimeError, match="Injected replay failure"):
        decoder.decode_delta(codes, state)
    assert boundary.operations[-1] == "synchronize"
    for tensor in decoder.backend.arena.persistent_tensors():
        assert torch.count_nonzero(tensor[0]) == 0
    with pytest.raises(RuntimeError, match="stopped"):
        decoder.decode_delta(codes, state)
    with pytest.raises(RuntimeError, match="stopped"):
        decoder.create_state(1)
    assert boundary.graphs[10].replay_count == 1
    assert state.is_failed
    assert state.slot_index in decoder.backend.arena.active_slots
    decoder.release_state(state)
    assert not decoder.backend.arena.active_slots
    with pytest.raises(ValueError, match="stale"):
        decoder.release_state(state)


def test_capture_failure_cleans_scratch_without_mutating_live_slot() -> None:
    backend = StatefulDACDecoder(tiny_fishaudio_codec(), capacity=1)
    state = backend.create_state(0)
    codes = torch.randint(0, 16, (1, backend.num_codebooks, 3))
    backend.decode_delta(codes, state)
    before = [tensor[0].clone() for tensor in backend.arena.persistent_tensors()]
    boundary = CpuGraphBoundary(should_fail_capture=True)
    with pytest.raises(RuntimeError, match="Injected capture failure"):
        StatefulGraphDecoder(backend, graph_boundary=boundary)
    for previous, current in zip(
        before, backend.arena.persistent_tensors(), strict=True
    ):
        torch.testing.assert_close(current[0], previous)
        assert torch.count_nonzero(current[backend.scratch_slot_index]) == 0
    assert state.processed_frames == 3


def test_hot_graphs_use_eager_attention_for_uncaptured_tails() -> None:
    codec = tiny_fishaudio_codec()
    backend = StatefulDACDecoder(codec, capacity=1)
    reference = StatefulDACDecoder(codec, capacity=1)
    boundary = CpuGraphBoundary()
    decoder = StatefulGraphDecoder(
        backend, graph_boundary=boundary, capture_frame_counts=(10, 12, 32, 64)
    )
    state = decoder.create_state()
    expected_state = reference.create_state()

    def forbid_compile(*args, **kwargs):
        raise AssertionError("Uncaptured tail must not invoke compiled attention")

    for count in (10, 1, 63, 64, 7):
        codes = torch.randint(0, 16, (1, backend.num_codebooks, count))
        original = backend.attention_decode
        if count not in decoder.graphs:
            backend.attention_decode = forbid_compile
        else:
            pass
        try:
            actual = decoder.decode_delta(codes, state)
        finally:
            backend.attention_decode = original
        torch.testing.assert_close(
            actual, reference.decode_delta(codes, expected_state), atol=1e-5, rtol=1e-5
        )
    assert sum(graph.replay_count for graph in boundary.graphs.values()) == 2
    assert len(decoder.graphs) == 4


@pytest.mark.parametrize("frame_count", [7, 10, 64])
def test_nonempty_final_returns_owned_exact_audio_and_release_preserves_peer(
    decoder_pair: tuple[StatefulGraphDecoder, StatefulDACDecoder, CpuGraphBoundary],
    frame_count: int,
) -> None:
    decoder, reference, _ = decoder_pair
    final_state = decoder.create_state(0)
    peer_state = decoder.create_state(1)
    peer_reference = reference.create_state(1)
    codes = torch.randint(0, 16, (1, decoder.backend.num_codebooks, frame_count))
    output = decoder.decode_delta(codes, final_state, is_final=True)
    expected = reference.decode_delta(codes, reference.create_state(0), is_final=True)
    torch.testing.assert_close(output, expected)
    assert output.shape[-1] == frame_count * decoder.backend.frame_length
    assert final_state.is_final
    assert final_state.processed_frames == frame_count
    decoder.decode_delta(codes, peer_state)
    reference.decode_delta(codes, peer_reference)
    decoder.release_state(final_state)
    torch.testing.assert_close(
        decoder.decode_delta(codes, peer_state),
        reference.decode_delta(codes, peer_reference),
    )
    torch.testing.assert_close(output, expected)
    assert len(decoder.graphs) == 64


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@torch.no_grad()
@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_cuda_exact_graphs_match_native_with_slot_reuse_and_owned_output(
    monkeypatch: pytest.MonkeyPatch,
    optimized: bool,
    dtype: torch.dtype,
) -> None:
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    codec = tiny_fishaudio_codec().cuda().to(dtype=dtype)
    backend = StatefulDACDecoder(
        codec,
        dtype=dtype,
        capacity=2,
        gemm_upsample=optimized,
        folded_codebooks=optimized,
        fuse_snake_history=optimized,
        compile_attention=optimized,
    )
    decoder = StatefulGraphDecoder(backend, compiled_frame_counts=(10, 12, 32, 64))
    tolerance = 0.02 if dtype == torch.float16 else 1e-4
    assert backend.waveform_decoder is not None
    states = [decoder.create_state(slot_index) for slot_index in range(2)]
    inputs = [
        torch.randint(0, 16, (1, backend.num_codebooks, 193), device="cuda")
        for _ in states
    ]
    expected = [codec.from_indices(codes.clone()).float() for codes in inputs]
    outputs: list[list[torch.Tensor]] = [[], []]
    snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
    position = 0
    for frame_count in (1, 63, 64, 64, 1):
        for slot_index in (1, 0):
            audio = decoder.decode_delta(
                inputs[slot_index][..., position : position + frame_count],
                states[slot_index],
                is_final=position + frame_count == 193,
            )
            outputs[slot_index].append(audio)
            snapshots.append((audio, audio.clone()))
        position += frame_count
    for slot_index in (0, 1):
        torch.testing.assert_close(
            torch.cat(outputs[slot_index], dim=-1).float(),
            expected[slot_index],
            atol=tolerance,
            rtol=tolerance,
        )
    decoder.release_state(states[0])
    replacement = decoder.create_state(0)
    cold_codes = inputs[0][..., :10]
    torch.testing.assert_close(
        decoder.decode_delta(cold_codes, replacement).float(),
        codec.from_indices(cold_codes.clone()).float(),
        atol=tolerance,
        rtol=tolerance,
    )
    decoder.release_state(replacement)
    decoder.release_state(states[1])
    factory = Drama3VocoderFactoryArgs()
    scheduler = Drama3VocoderScheduler(
        codec,
        device="cuda",
        stream_stride=factory.stream_stride,
        stream_followup_stride_ramp=factory.stream_followup_stride_ramp,
        stream_followup_stride=factory.stream_followup_stride,
        stream_overlap_tokens=factory.stream_overlap_tokens,
        stream_crossfade_samples=0,
        max_batch_size=1,
        max_batch_wait_ms=0,
        stateful_decoder=decoder,
        stateful_max_streams=2,
    )
    long_state = decoder.create_state()
    scheduler.request_slots["long-window"] = long_state
    scheduler.stream_states["long-window"] = StreamVocoderState(
        codes=[torch.cat((inputs[0][:, :1], inputs[0]), dim=1)[0]],
        total_tokens=193,
    )
    chunk = scheduler.decode_stream_delta("long-window", is_final=True)
    assert chunk is not None
    torch.testing.assert_close(
        torch.from_numpy(
            np.frombuffer(chunk["audio_waveform"], dtype=np.float32).copy()
        ),
        expected[0][0, 0].cpu(),
        atol=tolerance,
        rtol=tolerance,
    )
    assert long_state.processed_frames == 193
    assert long_state.is_final
    scheduler.clear_stream_state("long-window")
    for audio, snapshot in snapshots:
        torch.testing.assert_close(audio, snapshot)
    with pytest.raises(ValueError, match="stale"):
        decoder.release_state(states[0])


def test_cpu_without_boundary_rejects_capture_before_cuda_calls() -> None:
    backend = StatefulDACDecoder(tiny_fishaudio_codec(), capacity=1)
    with pytest.raises(ValueError, match="CUDA device"):
        StatefulGraphDecoder(backend)
