# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the streaming TTS stress test: config, health metrics, and timing."""

import base64
import json
from pathlib import Path

import pytest

from benchmarks.tts_serving.stream_stress.config_parser import (
    StressConfig,
    arrival_offsets,
    parse_config,
)
from benchmarks.tts_serving.stream_stress.responder import (
    ChunkRecord,
    Responder,
    StreamRecorder,
    achieved_qpm,
    classify_timeout,
    compute_playback_underrun,
    percentile,
    summarize,
)

BYTES_PER_SECOND = 24000 * 2


def make_config(tmp_path: Path, *args: str) -> StressConfig:
    corpus = tmp_path / "corpus.jsonl"
    if not corpus.exists():
        corpus.write_text(
            "\n".join(
                json.dumps({"id": f"s{index}", "text": f"hello {index}"})
                for index in range(3)
            ),
            encoding="utf-8",
        )
    return parse_config(
        ["--text-jsonl", str(corpus), "--output", str(tmp_path / "out.jsonl"), *args]
    )


def audio_chunk(arrival_s: float, duration_s: float) -> ChunkRecord:
    audio_bytes = int(duration_s * BYTES_PER_SECOND)
    return ChunkRecord(
        index=0,
        arrival_s=arrival_s,
        raw_bytes=audio_bytes,
        audio_bytes=audio_bytes,
        audio_duration_s=duration_s,
    )


def pcm(duration_s: float) -> bytes:
    return b"\x00" * int(duration_s * BYTES_PER_SECOND)


def make_record(tmp_path: Path, *, sent_at: float, load_start: float = 0.0):
    config = make_config(tmp_path)
    recorder = StreamRecorder(
        config,
        sequence=0,
        request_id="r",
        sample_id="s",
        text="t",
        warmup=False,
    )
    recorder.start_response(200, {})
    recorder.mark_sent(sent_at)
    recorder.add_raw_bytes(pcm(1.0), sent_at + 0.1)
    recorder.mark_completed(sent_at + 0.5)
    return recorder.finalize(load_start=load_start)


class FakeTimeoutError(TimeoutError):
    pass


def named_timeout(name: str, message: str = "") -> TimeoutError:
    return type(name, (FakeTimeoutError,), {})(message)


class TestPlaybackUnderrun:
    def test_continuous_stream_has_no_underrun(self) -> None:
        chunks = [audio_chunk(0.0, 1.0), audio_chunk(0.5, 1.0), audio_chunk(1.5, 1.0)]
        assert compute_playback_underrun(chunks) == (0.0, 0)

    def test_gap_beyond_buffered_audio_is_underrun(self) -> None:
        chunks = [audio_chunk(0.0, 1.0), audio_chunk(1.5, 1.0), audio_chunk(2.0, 1.0)]
        maximum, count = compute_playback_underrun(chunks)
        assert maximum == pytest.approx(0.5)
        assert count == 1

    def test_large_gap_covered_by_buffer_is_not_underrun(self) -> None:
        chunks = [audio_chunk(0.0, 5.0), audio_chunk(3.0, 1.0)]
        assert compute_playback_underrun(chunks) == (0.0, 0)

    def test_deadline_accumulates_after_late_chunk(self) -> None:
        chunks = [
            audio_chunk(0.0, 1.0),
            audio_chunk(2.0, 1.0),
            audio_chunk(3.5, 1.0),
        ]
        maximum, count = compute_playback_underrun(chunks)
        assert maximum == pytest.approx(1.0)
        assert count == 2

    def test_needs_two_audio_chunks(self) -> None:
        control = ChunkRecord(
            index=1,
            arrival_s=9.0,
            raw_bytes=10,
            audio_bytes=0,
            audio_duration_s=0.0,
            kind="sse_done",
        )
        assert compute_playback_underrun([]) == (None, 0)
        assert compute_playback_underrun([audio_chunk(0.0, 1.0)]) == (None, 0)
        assert compute_playback_underrun([audio_chunk(0.0, 1.0), control]) == (
            None,
            0,
        )


class TestStatistics:
    def test_percentile_interpolates(self) -> None:
        assert percentile([], 50) is None
        assert percentile([3.0], 99) == 3.0
        assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
        assert percentile([4.0, 1.0], 100) == 4.0

    def test_summarize_empty_and_filled(self) -> None:
        assert summarize([])["count"] == 0
        assert summarize([])["mean"] is None
        summary = summarize([1.0, 2.0, 3.0])
        assert summary["count"] == 3
        assert summary["mean"] == 2.0
        assert summary["max"] == 3.0


class TestConfig:
    def test_qpm_is_the_only_rate_unit(self, tmp_path: Path) -> None:
        assert make_config(tmp_path).qpm == 60.0
        assert make_config(tmp_path, "--qpm", "120").qpm == 120.0
        with pytest.raises(SystemExit):
            make_config(tmp_path, "--request-rate", "2")

    def test_non_positive_qpm_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="--qpm"):
            make_config(tmp_path, "--qpm", "0")

    def test_stall_option_is_gone(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            make_config(tmp_path, "--stall-broken-s", "2")

    def test_uniform_offsets_follow_qpm(self) -> None:
        offsets, count = arrival_offsets(
            count=4,
            duration_s=None,
            qpm=120.0,
            mode="uniform",
            seed=0,
            ramp_up_s=0.0,
            ramp_start_fraction=0.25,
        )
        assert count == 4
        assert offsets == pytest.approx((0.0, 0.5, 1.0, 1.5))

    def test_burst_sends_everything_at_once(self) -> None:
        offsets, count = arrival_offsets(
            count=3,
            duration_s=None,
            qpm=60.0,
            mode="burst",
            seed=0,
            ramp_up_s=0.0,
            ramp_start_fraction=0.25,
        )
        assert (offsets, count) == ((0.0, 0.0, 0.0), 3)

    def test_duration_takes_precedence_over_count(self) -> None:
        offsets, count = arrival_offsets(
            count=1000,
            duration_s=5.0,
            qpm=60.0,
            mode="uniform",
            seed=0,
            ramp_up_s=0.0,
            ramp_start_fraction=0.25,
        )
        assert count == 5
        assert offsets[-1] < 5.0

    def test_poisson_is_seeded_and_starts_at_zero(self) -> None:
        kwargs = dict(
            count=20,
            duration_s=None,
            qpm=600.0,
            mode="poisson",
            seed=7,
            ramp_up_s=0.0,
            ramp_start_fraction=0.25,
        )
        first, _ = arrival_offsets(**kwargs)
        second, _ = arrival_offsets(**kwargs)
        assert first == second
        assert first[0] == 0.0
        assert list(first) == sorted(first)

    def test_ramp_up_keeps_offsets_monotonic_and_slower_at_start(self) -> None:
        offsets, _ = arrival_offsets(
            count=100,
            duration_s=None,
            qpm=600.0,
            mode="uniform",
            seed=0,
            ramp_up_s=10.0,
            ramp_start_fraction=0.25,
        )
        gaps = [later - earlier for earlier, later in zip(offsets, offsets[1:])]
        assert all(gap > 0 for gap in gaps)
        assert gaps[0] > gaps[-1]
        assert gaps[-1] == pytest.approx(0.1)


class TestStreamHealth:
    def test_long_gap_with_buffered_audio_is_not_broken(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        recorder = StreamRecorder(
            config, sequence=0, request_id="r", sample_id="s", text="t", warmup=False
        )
        recorder.start_response(200, {})
        recorder.mark_sent(100.0)
        recorder.add_raw_bytes(pcm(5.0), 100.2)
        recorder.add_raw_bytes(pcm(1.0), 103.2)
        recorder.mark_completed(103.3)
        record = recorder.finalize(load_start=100.0)

        assert record.max_inter_chunk_s == pytest.approx(3.0)
        assert record.max_playback_underrun_s == 0.0
        assert "chunk_stall" not in record.broken_reasons
        assert not record.is_broken_stream
        assert record.is_success

    def test_real_underrun_marks_stream_broken(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        recorder = StreamRecorder(
            config, sequence=0, request_id="r", sample_id="s", text="t", warmup=False
        )
        recorder.start_response(200, {})
        recorder.mark_sent(0.0)
        recorder.add_raw_bytes(pcm(1.0), 0.1)
        recorder.add_raw_bytes(pcm(1.0), 1.6)
        recorder.mark_completed(1.7)
        record = recorder.finalize(load_start=0.0)

        assert record.max_playback_underrun_s == pytest.approx(0.5)
        assert record.underrun_count == 1
        assert "playback_underrun" in record.broken_reasons
        assert record.continuity_pass == {"c50": False, "c100": False, "c200": False}

    def test_wav_header_split_across_chunks_is_stripped(self, tmp_path: Path) -> None:
        config = make_config(tmp_path, "--response-format", "wav")
        recorder = StreamRecorder(
            config, sequence=0, request_id="r", sample_id="s", text="t", warmup=False
        )
        header = b"RIFF" + b"\x00" * 40
        recorder.start_response(200, {})
        recorder.mark_sent(0.0)
        recorder.add_raw_bytes(header[:20], 0.1)
        recorder.add_raw_bytes(header[20:] + pcm(1.0), 0.2)
        recorder.mark_completed(0.3)
        record = recorder.finalize(load_start=0.0)

        assert record.audio_bytes == int(BYTES_PER_SECOND)
        assert record.audio_duration_s == pytest.approx(1.0)
        assert "invalid_wav_header" not in record.broken_reasons

    def test_invalid_wav_header_is_flagged(self, tmp_path: Path) -> None:
        config = make_config(tmp_path, "--response-format", "wav")
        recorder = StreamRecorder(
            config, sequence=0, request_id="r", sample_id="s", text="t", warmup=False
        )
        recorder.start_response(200, {})
        recorder.mark_sent(0.0)
        recorder.add_raw_bytes(b"NOPE" + b"\x00" * 40 + pcm(1.0), 0.1)
        recorder.mark_completed(0.2)
        record = recorder.finalize(load_start=0.0)

        assert "invalid_wav_header" in record.broken_reasons
        assert not record.is_success

    def test_no_audio_is_a_failure(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        recorder = StreamRecorder(
            config, sequence=0, request_id="r", sample_id="s", text="t", warmup=False
        )
        recorder.start_response(200, {})
        recorder.mark_sent(0.0)
        recorder.mark_completed(0.1)
        record = recorder.finalize(load_start=0.0)

        assert "no_audio" in record.broken_reasons
        assert not record.is_success

    def test_sse_stream_requires_done_event(self, tmp_path: Path) -> None:
        config = make_config(tmp_path, "--stream-format", "sse")
        encoded = base64.b64encode(pcm(1.0)).decode()

        missing = StreamRecorder(
            config, sequence=0, request_id="r", sample_id="s", text="t", warmup=False
        )
        missing.start_response(200, {})
        missing.mark_sent(0.0)
        missing.add_sse_audio(encoded, len(encoded), 0.1)
        missing.mark_completed(0.2)
        assert "missing_done_event" in missing.finalize(load_start=0.0).broken_reasons

        complete = StreamRecorder(
            config, sequence=1, request_id="r", sample_id="s", text="t", warmup=False
        )
        complete.start_response(200, {})
        complete.mark_sent(0.0)
        complete.add_sse_audio(encoded, len(encoded), 0.1)
        complete.add_sse_done({"finish_reason": "stop"}, 20, 0.2)
        complete.mark_completed(0.2)
        assert complete.finalize(load_start=0.0).is_success


class TestTimingOrigin:
    def test_latency_and_ttfa_start_at_planned_arrival(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        recorder = StreamRecorder(
            config,
            sequence=0,
            request_id="r",
            sample_id="s",
            text="t",
            warmup=False,
            planned_at_s=0.0,
        )
        recorder.mark_planned(10.0)
        recorder.start_response(200, {})
        recorder.mark_sent(12.0)
        recorder.add_raw_bytes(pcm(1.0), 12.5)
        recorder.mark_completed(14.0)
        record = recorder.finalize(load_start=10.0)

        assert record.dispatch_lag_s == pytest.approx(2.0)
        assert record.sent_at_s == pytest.approx(2.0)
        assert record.ttfa_s == pytest.approx(2.5)
        assert record.latency_s == pytest.approx(4.0)

    def test_unplanned_request_uses_send_time(self, tmp_path: Path) -> None:
        record = make_record(tmp_path, sent_at=5.0, load_start=5.0)
        assert record.dispatch_lag_s == 0.0
        assert record.ttfa_s == pytest.approx(0.1)
        assert record.latency_s == pytest.approx(0.5)

    def test_dispatch_lag_is_summarized(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        responder = Responder(config)
        for sequence, lag in enumerate((0.0, 1.0, 2.0)):
            recorder = StreamRecorder(
                config,
                sequence=sequence,
                request_id=f"r{sequence}",
                sample_id="s",
                text="t",
                warmup=False,
                planned_at_s=float(sequence),
            )
            recorder.mark_planned(float(sequence))
            recorder.start_response(200, {})
            recorder.mark_sent(sequence + lag)
            recorder.add_raw_bytes(pcm(1.0), sequence + lag + 0.1)
            recorder.mark_completed(sequence + lag + 0.2)
            responder.request_records.append(recorder.finalize(load_start=0.0))

        report = responder.build_report(wall_clock_s=5.0, load_stats={})
        assert report.dispatch_lag_s["count"] == 3
        assert report.dispatch_lag_s["max"] == pytest.approx(2.0)
        assert report.to_dict()["metrics"]["dispatch_lag_s"]["mean"] == 1.0


class TestAchievedQpm:
    def test_uses_inter_arrival_gaps(self, tmp_path: Path) -> None:
        records = [
            make_record(tmp_path, sent_at=float(second), load_start=0.0)
            for second in range(5)
        ]
        assert achieved_qpm(records) == pytest.approx(60.0)

    def test_undefined_for_single_or_simultaneous_sends(self, tmp_path: Path) -> None:
        assert achieved_qpm([]) is None
        one = make_record(tmp_path, sent_at=0.0)
        assert achieved_qpm([one]) is None
        assert achieved_qpm([one, make_record(tmp_path, sent_at=0.0)]) is None


class TestTimeoutClassification:
    def test_total_timeout(self) -> None:
        assert classify_timeout(TimeoutError(), received_chunk=True) == "total_timeout"

    def test_connect_timeout(self) -> None:
        exc = named_timeout("ConnectionTimeoutError")
        assert classify_timeout(exc, received_chunk=False) == "connect_timeout"

    def test_socket_timeout_depends_on_first_chunk(self) -> None:
        exc = named_timeout("SocketTimeoutError")
        assert classify_timeout(exc, received_chunk=False) == "first_chunk_timeout"
        assert classify_timeout(exc, received_chunk=True) == "chunk_timeout"

    def test_legacy_server_timeout_uses_message(self) -> None:
        connect = named_timeout("ServerTimeoutError", "Connection timeout to host x")
        read = named_timeout(
            "ServerTimeoutError", "Timeout on reading data from socket"
        )
        assert classify_timeout(connect, received_chunk=False) == "connect_timeout"
        assert classify_timeout(read, received_chunk=True) == "chunk_timeout"

    def test_fail_timeout_records_kind_and_is_fatal(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        recorder = StreamRecorder(
            config, sequence=0, request_id="r", sample_id="s", text="t", warmup=False
        )
        recorder.start_response(200, {})
        recorder.mark_sent(0.0)
        recorder.add_raw_bytes(pcm(1.0), 0.1)
        recorder.fail_timeout(named_timeout("SocketTimeoutError"), observed_at=30.1)
        record = recorder.finalize(load_start=0.0)

        assert "chunk_timeout" in record.broken_reasons
        assert "total_timeout" not in record.broken_reasons
        assert record.error is not None and "chunks" in record.error
        assert not record.is_success

    def test_fail_timeout_total(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        recorder = StreamRecorder(
            config, sequence=0, request_id="r", sample_id="s", text="t", warmup=False
        )
        recorder.mark_sent(0.0)
        recorder.fail_timeout(TimeoutError(), observed_at=300.0)
        record = recorder.finalize(load_start=0.0)

        assert "total_timeout" in record.broken_reasons
        assert record.latency_s == pytest.approx(300.0)
        assert not record.is_success


class TestSseParsing:
    @pytest.fixture(autouse=True)
    def sender(self):
        return pytest.importorskip("benchmarks.tts_serving.stream_stress.sender")

    def test_pop_sse_frame_handles_both_separators(self, sender) -> None:
        buffer = bytearray(b"data: a\n\ndata: b\r\n\r\ndata: c")
        assert sender.pop_sse_frame(buffer) == b"data: a"
        assert sender.pop_sse_frame(buffer) == b"data: b"
        assert sender.pop_sse_frame(buffer) is None
        assert bytes(buffer) == b"data: c"

    def test_handle_sse_frames(self, sender, tmp_path: Path) -> None:
        config = make_config(tmp_path, "--stream-format", "sse")
        recorder = StreamRecorder(
            config, sequence=0, request_id="r", sample_id="s", text="t", warmup=False
        )
        recorder.mark_sent(0.0)
        delta = json.dumps(
            {
                "type": "speech.audio.delta",
                "audio": base64.b64encode(pcm(1.0)).decode(),
            }
        )
        sender.handle_sse_frame(recorder, f"data: {delta}".encode(), 0.1)
        sender.handle_sse_frame(recorder, b": keepalive", 0.2)
        sender.handle_sse_frame(
            recorder,
            b'data: {"type": "speech.audio.done", "finish_reason": "stop"}',
            0.3,
        )

        assert recorder.audio_bytes_total == int(BYTES_PER_SECOND)
        assert recorder.sse_done == {
            "type": "speech.audio.done",
            "finish_reason": "stop",
        }
        assert recorder.error is None

    def test_handle_sse_error_and_malformed_frames(
        self, sender, tmp_path: Path
    ) -> None:
        config = make_config(tmp_path, "--stream-format", "sse")

        def new_recorder() -> StreamRecorder:
            recorder = StreamRecorder(
                config,
                sequence=0,
                request_id="r",
                sample_id="s",
                text="t",
                warmup=False,
            )
            recorder.mark_sent(0.0)
            return recorder

        errored = new_recorder()
        sender.handle_sse_frame(
            errored, b'data: {"type": "speech.audio.error", "error": "boom"}', 0.1
        )
        assert "sse_error" in errored.health_failures

        malformed = new_recorder()
        sender.handle_sse_frame(malformed, b"data: not-json", 0.1)
        assert "malformed_sse_frame" in malformed.health_failures

        unknown = new_recorder()
        sender.handle_sse_frame(unknown, b'data: {"type": "mystery"}', 0.1)
        assert "unknown_sse_event" in unknown.health_failures

        done = new_recorder()
        sender.handle_sse_frame(done, b"data: [DONE]", 0.1)
        assert done.sse_done is not None
