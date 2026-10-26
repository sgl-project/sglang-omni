# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.minicpm_o.components.code2wav import OUTPUT_SAMPLE_RATE
from sglang_omni.models.minicpm_o.components.token2wav.vocoder import (
    StreamCaches,
    StreamChunk,
)
from sglang_omni.models.minicpm_o.components.tts_runtime import (
    CHUNK_GRAPH_MAX_STREAMS,
    CODEC_CHUNK_SIZE,
    MiniCPMOVocoderRuntime,
    SynthesisRequest,
)
from sglang_omni.models.minicpm_o.native_config import (
    DEFAULT_MAX_SESSIONS,
    DEFAULT_SPEECH_SETTINGS,
    DEFAULT_SPEECH_STATE_BYTES_PER_SESSION,
)
from sglang_omni.models.minicpm_o.native_stages import SpeechHooks
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import (
    SessionAppend,
    SessionContext,
    SessionScheduler,
)
from tests.unit_test.scheduling.test_session_batch import message, opens

PRE_LOOKAHEAD = 3
UP_RATE = 2
# note (Junnan Li): Large enough that a session's pending-token list is a rounding error next to its caches.
OPENED_HISTORY_FRAMES = 8192
SPEAKING_TOKENS = list(range(CODEC_CHUNK_SIZE))


class FakeToken2Wav:
    def __init__(self) -> None:
        self.flow = SimpleNamespace(pre_lookahead_len=PRE_LOOKAHEAD, up_rate=UP_RATE)
        self.device = torch.device("cpu")
        self.forward_sizes: list[int] = []
        self.captured_graphs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self.vocoder_warm_ups: list[tuple[int, int]] = []

    def capture_chunk_graphs(
        self,
        prompt: None,
        caches: StreamCaches,
        *,
        stream_counts: tuple[int, ...],
        frame_counts: tuple[int, ...],
    ) -> None:
        self.captured_graphs.append((stream_counts, frame_counts))

    def warm_up_vocoder(self, max_rows: int, max_chunk_frames: int) -> None:
        self.vocoder_warm_ups.append((max_rows, max_chunk_frames))

    def open_stream(self, prompt: object) -> StreamCaches:
        return {"history": torch.zeros(1, OPENED_HISTORY_FRAMES)}, {
            "speech": torch.zeros(1, 0)
        }

    def stream_batch(
        self, chunks: list[StreamChunk]
    ) -> list[tuple[bytes, StreamCaches]]:
        self.forward_sizes.append(len(chunks))
        results = []
        for chunk in chunks:
            flow_cache, hift_cache = chunk.caches
            history = flow_cache["history"].shape[1]
            pcm = np.array(
                chunk.token_ids + [history, int(chunk.is_last_chunk)], dtype="<i2"
            ).tobytes()
            results.append(
                (pcm, ({"history": torch.zeros(1, history + 1)}, hift_cache))
            )
        return results


class FakeCode2Wav:
    def __init__(self) -> None:
        self.token2wav = FakeToken2Wav()
        self.decode_stream = torch.cpu.Stream()

    def resolve_reference_key(self, reference: bytes) -> tuple[str, bytes]:
        return reference.decode(), reference

    def prepare_references(self, references: list[bytes]) -> list[None]:
        return [None for _ in references]


def open_runtime(
    session_ids: list[str],
    max_state_bytes_per_session: int = DEFAULT_SPEECH_STATE_BYTES_PER_SESSION,
) -> MiniCPMOVocoderRuntime:
    runtime = MiniCPMOVocoderRuntime(
        FakeCode2Wav(),
        max_state_bytes_per_session=max_state_bytes_per_session,
        max_open_sessions=DEFAULT_MAX_SESSIONS,
    )
    for session_id in session_ids:
        runtime.open_session(session_id, reference_audio=b"voice")
    return runtime


def request(
    session_id: str,
    token_ids: list[int],
    *,
    is_turn_start: bool,
    end_of_turn: bool = False,
) -> SynthesisRequest:
    return SynthesisRequest(
        session_id=session_id,
        codec_token_ids=token_ids,
        is_turn_start=is_turn_start,
        end_of_turn=end_of_turn,
    )


UNITS = [
    [
        request("a", list(range(60)), is_turn_start=True),
        request("b", list(range(100, 125)), is_turn_start=True),
        request("c", list(range(200, 210)), is_turn_start=True, end_of_turn=True),
        request("d", [], is_turn_start=False),
    ],
    [
        request("a", list(range(60, 85)), is_turn_start=False),
        request("b", list(range(125, 150)), is_turn_start=False),
        request("c", list(range(210, 235)), is_turn_start=True),
        request("d", list(range(300, 303)), is_turn_start=True, end_of_turn=True),
    ],
]


def test_units_synthesized_together_match_units_synthesized_alone() -> None:
    alone, together = open_runtime(list("abcd")), open_runtime(list("abcd"))
    for requests in UNITS:
        expected: dict[str, np.ndarray | None] = {}
        for unit in requests:
            for index, waveform in alone.synthesize_batch([unit]):
                expected[unit.session_id] = waveform
        received = {
            requests[index].session_id: waveform
            for index, waveform in together.synthesize_batch(requests)
        }
        assert received.keys() == expected.keys()
        for session_id, waveform in expected.items():
            if waveform is None:
                assert received[session_id] is None
            else:
                np.testing.assert_array_equal(received[session_id], waveform)
        for session_id in "abcd":
            assert (
                together.sessions[session_id].pending_codec_token_ids
                == alone.sessions[session_id].pending_codec_token_ids
            )
    assert max(together.token2wav.forward_sizes) > 1


def test_forward_width_follows_the_unheld_state_budget() -> None:
    session_ids = [str(index) for index in range(48)]
    requests = [
        request(session_id, list(range(CODEC_CHUNK_SIZE)), is_turn_start=True)
        for session_id in session_ids
    ]
    runtime = open_runtime(session_ids)
    list(runtime.synthesize_batch(requests))
    assert runtime.token2wav.forward_sizes == [len(session_ids)]

    runtime = open_runtime(session_ids, max_state_bytes_per_session=1)
    list(runtime.synthesize_batch(requests))
    assert runtime.token2wav.forward_sizes == [1] * len(session_ids)


def test_units_finish_before_later_rounds_in_request_order() -> None:
    events: list[tuple[str, object]] = []
    runtime = open_runtime(list("abc"))
    decode = runtime.token2wav.stream_batch

    def recording_stream_batch(
        chunks: list[StreamChunk],
    ) -> list[tuple[bytes, StreamCaches]]:
        events.append(("forward", len(chunks)))
        return decode(chunks)

    runtime.token2wav.stream_batch = recording_stream_batch
    for index, waveform in runtime.synthesize_batch(
        [
            request("a", list(range(2 * CODEC_CHUNK_SIZE)), is_turn_start=True),
            request("b", list(range(CODEC_CHUNK_SIZE)), is_turn_start=True),
            request("c", [], is_turn_start=False),
        ]
    ):
        events.append(("done", "abc"[index]))
    assert events == [
        ("done", "c"),
        ("forward", 2),
        ("done", "b"),
        ("forward", 1),
        ("done", "a"),
    ]


@pytest.mark.parametrize("capture_chunk_graphs", [True, False])
@pytest.mark.parametrize(
    ("max_open_sessions", "stream_counts", "forward_widths"),
    [
        (1, (1,), (2,)),
        (2, (1, 2), (2,)),
        (6, (1, 2, 4, 6), (2,)),
        (CHUNK_GRAPH_MAX_STREAMS, (1, 2, 4, 8), (2,)),
        (CHUNK_GRAPH_MAX_STREAMS + 1, (1, 2, 4, 8), (2, CHUNK_GRAPH_MAX_STREAMS + 1)),
    ],
)
def test_warm_up_decodes_equal_and_ragged_forwards_of_every_reachable_width(
    max_open_sessions: int,
    stream_counts: tuple[int, ...],
    forward_widths: tuple[int, ...],
    capture_chunk_graphs: bool,
) -> None:
    runtime = MiniCPMOVocoderRuntime(
        FakeCode2Wav(),
        max_state_bytes_per_session=DEFAULT_SPEECH_STATE_BYTES_PER_SESSION,
        max_open_sessions=max_open_sessions,
    )
    forwards: list[list[tuple[int, bool, int]]] = []
    decode = runtime.token2wav.stream_batch

    def recording_stream_batch(
        chunks: list[StreamChunk],
    ) -> list[tuple[bytes, StreamCaches]]:
        forwards.append(
            [
                (
                    len(chunk.token_ids),
                    chunk.is_last_chunk,
                    chunk.caches[0]["history"].shape[1],
                )
                for chunk in chunks
            ]
        )
        return decode(chunks)

    runtime.token2wav.stream_batch = recording_stream_batch
    runtime.warm_up(b"voice", capture_chunk_graphs=capture_chunk_graphs)
    window = CODEC_CHUNK_SIZE + PRE_LOOKAHEAD
    if capture_chunk_graphs:
        assert runtime.token2wav.captured_graphs == [
            (stream_counts, (CODEC_CHUNK_SIZE * UP_RATE, (window - 1) * UP_RATE))
        ]
    else:
        assert runtime.token2wav.captured_graphs == []
    expected_forwards = []
    for width in forward_widths:
        expected_forwards += [
            [(window, False, OPENED_HISTORY_FRAMES)] * width,
            [(window, False, OPENED_HISTORY_FRAMES + 1)] * (width - 1)
            + [(PRE_LOOKAHEAD, True, OPENED_HISTORY_FRAMES + 1)],
        ]
    assert forwards == expected_forwards
    assert not runtime.sessions and not runtime.speakers
    assert runtime.token2wav.vocoder_warm_ups == []


def speech_hooks(runtime: MiniCPMOVocoderRuntime) -> SpeechHooks:
    return SpeechHooks(
        runtime, b"voice", gather_window_ms=DEFAULT_SPEECH_SETTINGS.gather_window_ms
    )


def test_speech_hooks_warm_the_vocoder_for_every_reachable_batch() -> None:
    runtime = open_runtime([])
    speech_hooks(runtime).warm_up_serving_thread()
    assert runtime.token2wav.vocoder_warm_ups == [
        (DEFAULT_MAX_SESSIONS, (CODEC_CHUNK_SIZE + PRE_LOOKAHEAD) * UP_RATE)
    ]


def speech_append(
    session_identity: SessionIdentity,
    emitted: list[TimedChunk],
    *,
    codec_tokens: list[int],
    is_listen: bool,
    end_of_turn: bool = False,
    eos: bool = False,
) -> SessionAppend:
    return SessionAppend(
        chunk=TimedChunk("text", 0, 1000, 0, None, eos=eos),
        payload=SimpleNamespace(
            data=dict(
                text="hi",
                codec_tokens=codec_tokens,
                talker_conditions=[] if is_listen else [object()],
                is_listen=is_listen,
                end_of_turn=end_of_turn,
                speech_turn_start=True,
                is_speech_cut=False,
            )
        ),
        context=SessionContext(
            session_identity=session_identity,
            cancelled=threading.Event(),
            emit=emitted.append,
        ),
    )


@pytest.mark.parametrize(
    ("is_listen", "codec_tokens", "end_of_turn", "eos", "gathers"),
    [
        (True, [], False, False, False),
        (False, list(range(CODEC_CHUNK_SIZE)), False, False, True),
        (True, [], True, False, True),
        (True, [], False, True, True),
    ],
    ids=["listening", "speaking", "turn-end", "input-end"],
)
def test_speech_hooks_gather_peers_only_for_units_the_vocoder_runs(
    is_listen: bool,
    codec_tokens: list[int],
    end_of_turn: bool,
    eos: bool,
    gathers: bool,
) -> None:
    hooks = speech_hooks(open_runtime([]))
    append = speech_append(
        SessionIdentity("s"), [], codec_tokens=codec_tokens, is_listen=is_listen
    )
    append.payload.data["end_of_turn"] = end_of_turn
    chunk = TimedChunk("text", 0, 1000, 0, None, eos=eos)

    assert hooks.gathers_peers(append.payload, chunk) is gathers


def test_speech_hooks_emit_one_voice_chunk_per_unit() -> None:
    runtime = open_runtime([])
    hooks = speech_hooks(runtime)
    listening, speaking = SessionIdentity("listening"), SessionIdentity("speaking")
    for session_identity in (listening, speaking):
        hooks.open(session_identity, SimpleNamespace(params={}))
    listening_chunks: list[TimedChunk] = []
    speaking_chunks: list[TimedChunk] = []
    appends = [
        speech_append(listening, listening_chunks, codec_tokens=[], is_listen=True),
        speech_append(
            speaking,
            speaking_chunks,
            codec_tokens=list(range(CODEC_CHUNK_SIZE)),
            is_listen=False,
        ),
    ]

    payloads = hooks.append_batch(appends)

    assert payloads == [append.payload for append in appends]
    assert [chunk.payload["pcm"] for chunk in listening_chunks] == [b""]
    (voice,) = speaking_chunks
    assert voice.duration_ms == 1000
    assert voice.payload["is_speech_cut"] is False
    assert len(voice.payload["pcm"]) == 2 * OUTPUT_SAMPLE_RATE
    assert runtime.token2wav.forward_sizes == [1]


def run_speech_stage(
    *, batch_sessions: bool
) -> tuple[dict[str, list[dict]], list[int]]:
    """Run two units of three sessions through the speech stage; returns each unit's voice chunks and the vocoder forward widths."""
    runtime = MiniCPMOVocoderRuntime(
        FakeCode2Wav(),
        max_state_bytes_per_session=DEFAULT_SPEECH_STATE_BYTES_PER_SESSION,
        max_open_sessions=3,
    )
    scheduler = SessionScheduler(
        speech_hooks(runtime), max_open_sessions=3, batch_sessions=batch_sessions
    )
    appends = []
    for seq, (talker_tokens, end_of_turn) in enumerate(
        [(CODEC_CHUNK_SIZE + PRE_LOOKAHEAD, False), (CODEC_CHUNK_SIZE, True)]
    ):
        for offset, session_id in enumerate("abc"):
            append = message(f"{session_id}{seq}", "append", session_id, seq)
            append.data.data = dict(
                text=session_id,
                codec_tokens=list(range(100 * offset, 100 * offset + talker_tokens)),
                talker_conditions=[object()],
                is_listen=False,
                end_of_turn=end_of_turn,
                speech_turn_start=seq == 0,
                is_speech_cut=False,
            )
            appends.append(append)
    chunks: dict[str, list[dict]] = {append.request_id: [] for append in appends}
    for queued in opens(*"abc") + appends:
        scheduler.inbox.put(queued)
    worker = threading.Thread(target=scheduler.start)
    worker.start()
    try:
        finals = 0
        while finals < len(appends) + 3:
            output = scheduler.outbox.get(timeout=5)
            if output.type == "stream":
                chunks[output.request_id].append(output.data)
            else:
                assert output.type == "result"
                finals += 1
    finally:
        scheduler.stop()
        worker.join(timeout=5)
    return chunks, runtime.token2wav.forward_sizes


def test_speech_stage_emits_the_same_voice_with_or_without_cross_session_batches() -> (
    None
):
    batched_chunks, batched_widths = run_speech_stage(batch_sessions=True)
    alone_chunks, alone_widths = run_speech_stage(batch_sessions=False)

    assert alone_chunks == batched_chunks
    assert all(len(unit_chunks) == 1 for unit_chunks in batched_chunks.values())
    assert max(batched_widths) == 3
    assert set(alone_widths) == {1}


@pytest.mark.parametrize(
    ("units", "expected_turns"),
    [
        (
            (
                ([], True, False, False),
                (SPEAKING_TOKENS, False, False, False),
                (SPEAKING_TOKENS, False, True, False),
                ([], True, False, False),
                (SPEAKING_TOKENS, False, False, False),
            ),
            [0, 0, 0, 1, 1],
        ),
        (
            (
                (SPEAKING_TOKENS, False, False, False),
                ([], True, False, True),
                (SPEAKING_TOKENS, False, False, False),
            ),
            [0, 0, 1],
        ),
    ],
    ids=["model_end_of_turn", "input_end"],
)
def test_speech_hooks_stamp_voice_chunks_with_the_reply_turn(
    units: tuple[tuple[list[int], bool, bool, bool], ...],
    expected_turns: list[int],
) -> None:
    """A reply's chunks share one turn up to its end of turn or the input end; the next reply gets the next turn."""
    hooks = speech_hooks(open_runtime([]))
    session_identity = SessionIdentity("s")
    hooks.open(session_identity, SimpleNamespace(params={}))
    chunks: list[TimedChunk] = []

    for codec_tokens, is_listen, end_of_turn, eos in units:
        hooks.append_batch(
            [
                speech_append(
                    session_identity,
                    chunks,
                    codec_tokens=codec_tokens,
                    is_listen=is_listen,
                    end_of_turn=end_of_turn,
                    eos=eos,
                )
            ]
        )

    assert [chunk.turn for chunk in chunks] == expected_turns
