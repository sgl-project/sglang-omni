# SPDX-License-Identifier: Apache-2.0
"""API tests for the native Sortformer server against the original Voxt's outputs.

Needs NATIVE_RUNTIME_BIN (the server binary directory) and CI_DATA_ROOT.
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import subprocess
import time
import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import sortformer_golden
from websockets.exceptions import InvalidHandshake
from websockets.sync.client import connect

RUNTIME_BIN = os.environ.get("NATIVE_RUNTIME_BIN")
DATA_ROOT = os.environ.get("CI_DATA_ROOT")
pytestmark = pytest.mark.skipif(
    not RUNTIME_BIN or not DATA_ROOT, reason="set NATIVE_RUNTIME_BIN and CI_DATA_ROOT"
)
GOLDEN = json.loads(
    (
        Path(__file__).resolve().parent / "golden" / "sortformer-4spk-v2.1.json"
    ).read_text()
)
FEED = GOLDEN["feed"]
TOLERANCE = GOLDEN["tolerance"]
SPEAKERS = 4
CLIP = "0064_en_mid"
# Note (Jiaxin Deng): yields the checkpoint's 188 frame update period; one sample
# more would yield 189.
MAX_FEED_SAMPLES = 188 * 1280 - 1
OTHER_CLIP = "0076_en_mid"


def samples(clip_id: str) -> np.ndarray:
    path = Path(DATA_ROOT) / "corpus" / "v1" / "clips" / f"{clip_id}.wav"
    with wave.open(str(path)) as reader:
        pcm = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768


def feeds(clip_id: str) -> list[np.ndarray]:
    """Voxt's meeting feeds: whole feeds, the last zero-padded to the minimum."""
    values = samples(clip_id)
    chunks = []
    for offset in range(0, values.size, FEED["samples_per_feed"]):
        chunk = values[offset : offset + FEED["samples_per_feed"]]
        if chunk.size < FEED["minimum_samples"]:
            chunk = np.pad(chunk, (0, FEED["minimum_samples"] - chunk.size))
        else:
            pass
        chunks.append(chunk)
    return chunks


def original(clip_id: str) -> tuple[np.ndarray, list, dict]:
    reference = GOLDEN["clips"][clip_id]
    probabilities = np.frombuffer(
        base64.b64decode(reference["probabilities"]), dtype="<f4"
    ).reshape(-1, SPEAKERS)
    return probabilities, reference["segments"], reference["state"]


def segments_of(
    probabilities: np.ndarray,
    offset_frames: int,
    threshold: float,
    min_duration: float,
    merge_gap: float,
) -> list:
    """Swift predsToSegments in float32, shifted by the frames already fed."""
    frame = np.float32(FEED["frame_seconds"])
    offset = np.float32(offset_frames) * frame
    found = []
    for speaker in range(probabilities.shape[1]):
        runs = []
        start = -1
        for index, active in enumerate(probabilities[:, speaker] > threshold):
            if active and start < 0:
                start = index
            elif not active and start >= 0:
                runs.append((start, index))
                start = -1
            else:
                pass
        if start >= 0:
            runs.append((start, probabilities.shape[0]))
        else:
            pass
        spans = [
            [np.float32(first) * frame, np.float32(last) * frame]
            for first, last in runs
            if np.float32(last) * frame - np.float32(first) * frame
            >= np.float32(min_duration)
        ]
        merged = []
        for span in spans:
            if (
                merge_gap > 0
                and merged
                and span[0] - merged[-1][1] <= np.float32(merge_gap)
            ):
                merged[-1][1] = span[1]
            else:
                merged.append(span)
        found += [[s + offset, e + offset, speaker] for s, e in merged]
    return sorted(found, key=lambda item: item[0])


class Server:
    def __init__(self) -> None:
        model = Path(DATA_ROOT) / "models" / GOLDEN["model"].replace("/", "_")
        self.process = subprocess.Popen(
            [
                str(Path(RUNTIME_BIN) / "qwen3_asr_server"),
                "--supervised",
                "--model-kind",
                "sortformer",
                "--model-directory",
                str(model),
            ],  # fmt: skip
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.ready = json.loads(self.process.stdout.readline())
        self.port = self.ready.get("port")

    def request(self, method: str, path: str) -> tuple[int, bytes]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=120)
        connection.request(method, path)
        response = connection.getresponse()
        return response.status, response.read()

    def stream(self, query: str = ""):
        suffix = f"?{query}" if query else ""
        return connect(
            f"ws://127.0.0.1:{self.port}/v1/diarization/stream{suffix}",
            max_size=None,
        )

    def health(self) -> dict:
        return json.loads(self.request("GET", "/health")[1])

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.stdin.write('{"command": "shutdown"}\n')
            self.process.stdin.flush()
            self.process.wait(timeout=10)
        else:
            pass


@pytest.fixture(scope="module")
def server() -> Iterator[Server]:
    running = Server()
    yield running
    running.stop()


def feed_all(socket, chunks: list[np.ndarray]) -> list[dict]:
    replies = []
    for chunk in chunks:
        socket.send(chunk.astype("<f4").tobytes())
        replies.append(json.loads(socket.recv(timeout=120)))
    return replies


def assert_matches_original(
    clip_id: str, replies: list[dict], segments_too: bool = True
) -> None:
    expected, segments, state = original(clip_id)
    actual = np.array(
        [row for reply in replies for row in reply["probabilities"]], dtype=np.float32
    )
    assert actual.shape == expected.shape
    difference = np.abs(actual.astype(np.float64) - expected)
    assert difference.max() <= TOLERANCE["max_abs_probability"]
    assert np.quantile(difference, 0.99) <= TOLERANCE["p99_abs_probability"]
    flipped = int(np.sum((actual > 0.5) != (expected > 0.5)))
    assert flipped <= TOLERANCE["max_flipped_decisions"] * expected.size
    assert replies[-1]["state"] == state
    if not segments_too:
        return
    else:
        pass
    found = [
        [item["start"], item["end"], item["speaker"]]
        for reply in replies
        for item in reply["segments"]
    ]
    assert (
        sortformer_golden.speech_mismatch(segments, found)
        <= TOLERANCE["max_speech_mismatch"]
    )


def test_ready_event_and_routes(server: Server) -> None:
    assert server.ready["event"] == "ready"
    assert server.ready["model_name"].startswith("voxt-sortformer-")
    assert server.health() == {
        "status": "healthy",
        "running": True,
        "request_states": {"running": 0, "streams": 0},
    }
    assert server.request("POST", "/v1/audio/transcriptions")[0] == 404
    assert server.request("POST", "/v1/vad/speech_timestamps")[0] == 404


def test_stream_matches_the_original(server: Server) -> None:
    chunks = feeds(CLIP)
    with server.stream() as socket:
        assert server.health()["request_states"]["streams"] == 1
        replies = feed_all(socket, chunks)
    assert_matches_original(CLIP, replies)
    processed = 0
    for chunk, reply in zip(chunks, replies):
        # Note (Jiaxin Deng): one frame more for the centred STFT's last frame.
        assert reply["frames"] == chunk.size // 1280 + 1
        assert reply["speakers"] == SPEAKERS
        assert len(reply["probabilities"]) == reply["frames"]
        expected = segments_of(
            np.array(reply["probabilities"], dtype=np.float32),
            processed,
            FEED["threshold"],
            FEED["min_duration"],
            FEED["merge_gap"],
        )
        actual = [
            [np.float32(s["start"]), np.float32(s["end"]), s["speaker"]]
            for s in reply["segments"]
        ]
        assert actual == expected
        processed += reply["frames"]
        assert reply["state"]["frames_processed"] == processed
        assert reply["state"]["fifo_length"] <= FEED["fifo_max"]
        assert reply["state"]["spkcache_length"] <= FEED["spkcache_max"]
    assert wait_for_no_streams(server)


def wait_for_no_streams(server: Server) -> bool:
    """The server counts a stream closed once its close handler has run."""
    deadline = time.monotonic() + 2
    while server.health()["request_states"]["streams"] != 0:
        if time.monotonic() > deadline:
            return False
        else:
            time.sleep(0.02)
    return True


def test_streams_keep_separate_state(server: Server) -> None:
    first, second = feeds(CLIP), feeds(OTHER_CLIP)
    with server.stream() as one, server.stream() as two:
        replies_one, replies_two = [], []
        for index in range(max(len(first), len(second))):
            if index < len(first):
                replies_one += feed_all(one, [first[index]])
            else:
                pass
            if index < len(second):
                replies_two += feed_all(two, [second[index]])
            else:
                pass
    assert_matches_original(CLIP, replies_one)
    assert_matches_original(OTHER_CLIP, replies_two)


@pytest.mark.parametrize(
    "query, options",
    [
        ("threshold=0.9&min_duration=1.2&merge_gap=0", (0.9, 1.2, 0.0)),
        ("threshold=0.9&merge_gap=0.5", (0.9, 0.0, 0.5)),
    ],
)
def test_options_shape_the_segments(server: Server, query: str, options) -> None:
    with server.stream(query) as socket:
        replies = feed_all(socket, feeds(CLIP))
    processed = 0
    changed = False
    for reply in replies:
        probabilities = np.array(reply["probabilities"], dtype=np.float32)
        expected = segments_of(probabilities, processed, *options)
        assert [
            [np.float32(s["start"]), np.float32(s["end"]), s["speaker"]]
            for s in reply["segments"]
        ] == expected
        changed |= expected != segments_of(
            probabilities,
            processed,
            FEED["threshold"],
            FEED["min_duration"],
            FEED["merge_gap"],
        )
        processed += reply["frames"]
    assert changed
    assert_matches_original(CLIP, replies, segments_too=False)


def test_small_state_limits_compress_the_cache(server: Server) -> None:
    chunks = feeds(CLIP)
    with server.stream("fifo_max=63&spkcache_max=100") as socket:
        replies = feed_all(socket, chunks)
    for reply in replies:
        assert reply["state"]["fifo_length"] <= 63
    # Note (Jiaxin Deng): compression shrinks the cache to the checkpoint's 188
    # frames, or leaves it while it is under spkcache_max.
    assert replies[-1]["state"]["spkcache_length"] in range(1, 189)


@pytest.mark.parametrize(
    "query",
    [
        "threshold=2",
        "threshold=high",
        "merge_gap=-1",
        "spkcache_max=0",
        "fifo_max=5000",
        "speakers=2",
        "threshold",
    ],
)
def test_invalid_options_are_refused(server: Server, query: str) -> None:
    with pytest.raises((InvalidHandshake, OSError, EOFError)):
        with server.stream(query) as socket:
            socket.send(np.zeros(1280, dtype="<f4").tobytes())
            socket.recv(timeout=30)


@pytest.mark.parametrize(
    "payload",
    [
        b"\x00\x00\x00",
        b"\x00\x00\x00\x00",
        "text audio",
        np.array([0.0, np.nan], dtype="<f4").tobytes(),
        np.zeros(480_001, dtype="<f4").tobytes(),
        np.zeros(MAX_FEED_SAMPLES + 1, dtype="<f4").tobytes(),
    ],
    ids=[
        "partial-sample",
        "one-sample",
        "text",
        "nan",
        "over-30-s",
        "over-update-period",
    ],
)
def test_a_malformed_feed_closes_the_stream(server: Server, payload) -> None:
    with server.stream() as socket:
        socket.send(payload)
        assert "error" in json.loads(socket.recv(timeout=30))
        with pytest.raises(Exception):
            socket.recv(timeout=30)


def test_the_longest_feeds_keep_the_fifo_bounded(server: Server) -> None:
    """Each feed adds at most what one update retires, however many come."""
    chunk = np.zeros(MAX_FEED_SAMPLES, dtype="<f4")
    chunk[::7] = 0.1
    with server.stream() as socket:
        for _ in range(8):
            socket.send(chunk.tobytes())
            state = json.loads(socket.recv(timeout=120))["state"]
            assert state["fifo_length"] <= FEED["fifo_max"]
            assert state["spkcache_length"] <= FEED["spkcache_max"]


@pytest.mark.parametrize(
    "query",
    [
        "spkcache_max=700&fifo_max=700",
        # Note (Jiaxin Deng): a compressed cache holds 188 frames, not spkcache_max.
        "spkcache_max=1&fifo_max=1310",
        "spkcache_max=100&fifo_max=1200",
    ],
)
def test_state_limits_beyond_the_encoder_are_refused(
    server: Server, query: str
) -> None:
    """Cache, FIFO, context and a feed must fit the 1500 encoder positions."""
    with pytest.raises((InvalidHandshake, OSError, EOFError)):
        with server.stream(query) as socket:
            socket.send(np.zeros(1280, dtype="<f4").tobytes())
            socket.recv(timeout=30)


@pytest.mark.parametrize(
    "query", ["spkcache_max=700&fifo_max=600", "spkcache_max=1&fifo_max=1123"]
)
def test_state_limits_within_the_encoder_are_accepted(
    server: Server, query: str
) -> None:
    with server.stream(query) as socket:
        socket.send(np.zeros(1280, dtype="<f4").tobytes())
        assert "state" in json.loads(socket.recv(timeout=30))


def test_shutdown_reports_stopped() -> None:
    running = Server()
    running.process.stdin.write('{"command": "shutdown"}\n')
    running.process.stdin.flush()
    assert json.loads(running.process.stdout.readline()) == {"event": "stopped"}
    assert running.process.wait(timeout=10) == 0
