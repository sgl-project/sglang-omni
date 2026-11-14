# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.personaplex import mimi_encode_graph
from sglang_omni.models.personaplex.architecture import SAMPLES_PER_FRAME
from sglang_omni.models.personaplex.mimi_encode_graph import MimiEncodeSlots
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.models.personaplex.session_hooks import (
    BatchedMimiEncodeHooks,
    MimiEncodeHooks,
)
from sglang_omni.platforms import current_platform
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import SessionAppend, SessionContext

# Note (Junnan Li): Streams join and leave at different steps, so a batch mixes offsets
# and slots get reused.
PLAN = [(0, 1), (0, 1, 2), (1, 2), (0, 2), (2,), (0, 1, 2)]


def eager_slots(codec, num_slots: int) -> MimiEncodeSlots:
    platform = mimi_encode_graph.current_platform
    mimi_encode_graph.current_platform = SimpleNamespace(
        is_cuda=lambda: False, get_device_graph_backend=lambda device: None
    )
    try:
        return MimiEncodeSlots(codec, num_slots=num_slots)
    finally:
        mimi_encode_graph.current_platform = platform


def frames(device, streams: int, steps: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    waveform = torch.randn(streams, steps, SAMPLES_PER_FRAME, generator=generator)
    return (0.3 * waveform).to(device)


def encode_plan(slots: MimiEncodeSlots, waveform: torch.Tensor, plan) -> list:
    slot_of_stream = [slots.acquire() for _ in range(waveform.shape[0])]
    steps = [0] * waveform.shape[0]
    codes: list[list[torch.Tensor]] = [[] for _ in slot_of_stream]
    for frame, streams in enumerate(plan):
        batch = torch.stack([waveform[s, steps[s]] for s in streams])[:, None]
        step_codes = slots.encode_step(
            batch, [slot_of_stream[s] for s in streams], frame=frame
        )
        for row, stream in enumerate(streams):
            codes[stream].append(step_codes[row].clone())
            steps[stream] += 1
    for slot_index in slot_of_stream:
        slots.release(slot_index)
    return [torch.cat(stream_codes, dim=-1) for stream_codes in codes]


def test_streams_on_shared_slots_match_their_own_encoders(random_codec):
    slots = eager_slots(random_codec, num_slots=3)
    waveform = frames("cpu", 3, len(PLAN))
    batched = encode_plan(slots, waveform, PLAN)
    for stream in range(3):
        state = random_codec.init_encode_state(batch_size=1)
        alone = [
            random_codec.encode_step(waveform[stream, step].view(1, 1, -1), state)[0]
            for step in range(len(batched[stream][0]))
        ]
        assert torch.equal(batched[stream], torch.cat(alone, dim=-1))
    again = encode_plan(slots, waveform, PLAN)
    assert all(torch.equal(a, b) for a, b in zip(again, batched, strict=True))


@pytest.mark.accelerator
@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
def test_graph_replay_matches_eager_encode_across_slot_reuse(random_codec):
    codec = random_codec.to(torch.device("cuda"))
    graphed = MimiEncodeSlots(codec, num_slots=3)
    assert set(graphed.graphs) == {1, 2, 3}
    eager = eager_slots(codec, num_slots=3)
    waveform = frames("cuda", 3, 2 * len(PLAN))
    for _ in range(2):
        assert all(
            torch.equal(g, e)
            for g, e in zip(
                encode_plan(graphed, waveform, PLAN * 2),
                encode_plan(eager, waveform, PLAN * 2),
                strict=True,
            )
        )


def unit_append(session, index, waveform, emitted) -> SessionAppend:
    return SessionAppend(
        chunk=TimedChunk("audio", 80.0 * index, 80.0, index, b"", "pcm16"),
        payload=StagePayload(
            "unit",
            request=OmniRequest(inputs=None),
            data=PersonaPlexState(waveform=waveform).to_dict(),
        ),
        context=SessionContext(
            session_identity=session, cancelled=threading.Event(), emit=emitted.append
        ),
    )


def test_batched_hooks_encode_each_session_as_its_own_hooks_do(random_codec):
    sessions = [SessionIdentity("a"), SessionIdentity("b")]
    waveform = frames("cpu", 2, 4).reshape(2, -1)
    batched = BatchedMimiEncodeHooks(eager_slots(random_codec, num_slots=2))
    alone = MimiEncodeHooks(random_codec, torch.device("cpu"))
    for hooks in (batched, alone):
        for session in sessions:
            hooks.open(session, OmniRequest(inputs=None))
    # Note (Junnan Li): Session b sends two frames in one unit and sits the last step
    # out.
    steps = [(1, 1), (1, 2), (1, 0)]
    starts = [0, 0]
    for index, widths in enumerate(steps):
        appends, references = [], []
        for stream, (session, width) in enumerate(zip(sessions, widths, strict=True)):
            if width == 0:
                continue
            else:
                pass
            chunk = waveform[
                stream,
                starts[stream]
                * SAMPLES_PER_FRAME : (starts[stream] + width)
                * SAMPLES_PER_FRAME,
            ]
            starts[stream] += width
            appends.append(unit_append(session, index, chunk, []))
            references.append(unit_append(session, index, chunk, []))
        payloads = batched.append_batch(appends)
        for payload, reference in zip(payloads, references, strict=True):
            alone.append(reference.chunk, reference.payload, reference.context)
            encoded = PersonaPlexState.from_dict(payload.data)
            assert encoded.waveform is None
            assert torch.equal(
                encoded.user_codes,
                PersonaPlexState.from_dict(reference.payload.data).user_codes,
            )
    for session in sessions:
        batched.close(session)
    assert len(batched.encode_slots.free_slots) == 2


# Note (Junnan Li): Units of one, two and no frames, so frame steps hold different row
# counts.
FRAME_STEP_WIDTHS = [(1, 1), (1, 2), (2, 1), (1, 0)]


def encode_frame_steps(hooks, waveform) -> list[torch.Tensor]:
    sessions = [SessionIdentity("a"), SessionIdentity("b")]
    for session in sessions:
        hooks.open(session, OmniRequest(inputs=None))
    starts = [0, 0]
    codes = []
    for index, widths in enumerate(FRAME_STEP_WIDTHS):
        appends = []
        for stream, (session, width) in enumerate(zip(sessions, widths, strict=True)):
            if width == 0:
                continue
            else:
                pass
            chunk = waveform[
                stream,
                starts[stream]
                * SAMPLES_PER_FRAME : (starts[stream] + width)
                * SAMPLES_PER_FRAME,
            ]
            starts[stream] += width
            appends.append(unit_append(session, index, chunk, []))
        for payload in hooks.append_batch(appends):
            codes.append(PersonaPlexState.from_dict(payload.data).user_codes)
    for session in sessions:
        hooks.close(session)
    return codes


def test_frame_steps_encode_each_session_frame_by_frame(random_codec):
    waveform = frames("cpu", 2, 5).reshape(2, -1)
    codes = encode_frame_steps(
        BatchedMimiEncodeHooks(eager_slots(random_codec, 2)),
        waveform,
    )
    states = [random_codec.init_encode_state(batch_size=1) for _ in range(2)]
    starts = [0, 0]
    expected = []
    for widths in FRAME_STEP_WIDTHS:
        for stream, width in enumerate(widths):
            step_codes = []
            for frame in range(starts[stream], starts[stream] + width):
                step_codes.append(
                    random_codec.encode_step(
                        waveform[
                            stream,
                            frame * SAMPLES_PER_FRAME : (frame + 1) * SAMPLES_PER_FRAME,
                        ].view(1, 1, -1),
                        states[stream],
                    )[0]
                )
            starts[stream] += width
            if step_codes:
                expected.append(torch.cat(step_codes, dim=-1).T)
            else:
                pass
    assert [c.shape[0] for c in codes] == [1, 1, 1, 2, 2, 1, 1]
    assert all(torch.equal(c, e) for c, e in zip(codes, expected, strict=True))


@pytest.mark.accelerator
@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
def test_frame_steps_replay_graphs_as_eager_steps_encode(random_codec):
    codec = random_codec.to(torch.device("cuda"))
    waveform = frames("cuda", 2, 5).reshape(2, -1)
    graphed = MimiEncodeSlots(codec, num_slots=2)
    replayed = encode_frame_steps(BatchedMimiEncodeHooks(graphed), waveform)
    eager = encode_frame_steps(BatchedMimiEncodeHooks(eager_slots(codec, 2)), waveform)
    assert all(torch.equal(g, e) for g, e in zip(replayed, eager, strict=True))
