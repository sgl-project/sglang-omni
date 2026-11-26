# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the streaming TTS stress test: config, health metrics, and timing."""

import asyncio
import base64
import json
import threading
import wave
from pathlib import Path

import pytest

from benchmarks.benchmarker.data import RequestResult
from benchmarks.tts_serving.stream_stress.builder import write_generated_metadata
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


def write_reference_corpus(tmp_path: Path) -> Path:
    (tmp_path / "prompt.wav").write_bytes(b"RIFFprompt")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "cloned",
                        "text": "hello",
                        "language": "en",
                        "ref_audio": "prompt.wav",
                        "ref_text": "prompt words",
                    }
                ),
                json.dumps({"id": "plain", "text": "你好", "language": "zh"}),
                json.dumps({"id": "bare", "text": "third"}),
            ]
        ),
        encoding="utf-8",
    )
    return corpus


def run_against_local_server(
    tmp_path: Path, speak, *args: str
) -> tuple[StressConfig, Responder]:
    """Run the real sender against a local speech endpoint and write the saved audio's metadata."""
    sender = pytest.importorskip("benchmarks.tts_serving.stream_stress.sender")
    web = pytest.importorskip("aiohttp.web")

    async def run() -> tuple[StressConfig, Responder]:
        application = web.Application()
        application.router.add_post("/v1/audio/speech", speak)
        runner = web.AppRunner(application)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        config = make_config(
            tmp_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(runner.addresses[0][1]),
            "--arrival-mode",
            "burst",
            "--min-audio-duration-s",
            "0",
            *args,
        )
        responder = Responder(config)
        await sender.run_load(config, responder)
        await runner.cleanup()
        write_generated_metadata(config, responder.records)
        return config, responder

    return asyncio.run(run())


class TestCorpusLineOverrides:
    def test_a_line_sends_its_own_language_and_reference(self, tmp_path: Path) -> None:
        """Every row used to go out with the run's one language and reference, so a mixed corpus sent half its rows in the wrong language."""
        write_reference_corpus(tmp_path)
        config = make_config(
            tmp_path, "--language", "fr", "--ref-audio", "/server/run.wav"
        )
        cloned, plain, bare = (config.payload_for(entry) for entry in config.corpus)

        inline_audio = "data:audio/wav;base64," + base64.b64encode(
            b"RIFFprompt"
        ).decode("ascii")
        assert (cloned["language"], cloned["ref_audio"], cloned["ref_text"]) == (
            "en",
            inline_audio,
            "prompt words",
        )
        assert (plain["input"], plain["language"]) == ("你好", "zh")
        assert plain["ref_audio"] == "/server/run.wav"
        assert (bare["language"], bare["ref_audio"]) == ("fr", "/server/run.wav")
        assert "ref_text" not in bare

    def test_references_format_sends_a_line_reference_as_one_entry(
        self, tmp_path: Path
    ) -> None:
        write_reference_corpus(tmp_path)
        config = make_config(tmp_path, "--ref-format", "references")
        cloned, plain, _ = (config.payload_for(entry) for entry in config.corpus)

        assert cloned["references"] == [
            {
                "audio_path": "data:audio/wav;base64,"
                + base64.b64encode(b"RIFFprompt").decode("ascii"),
                "text": "prompt words",
            }
        ]
        assert "ref_audio" not in cloned and "ref_text" not in cloned
        assert "references" not in plain

    def test_a_reference_that_is_not_a_local_file_is_sent_unchanged(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "corpus.jsonl").write_text(
            json.dumps({"text": "hello", "ref_audio": "/server/voices/a.wav"}),
            encoding="utf-8",
        )
        config = make_config(tmp_path)
        assert config.payload_for(config.corpus[0])["ref_audio"] == (
            "/server/voices/a.wav"
        )

    def test_an_override_that_is_not_a_string_is_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "corpus.jsonl").write_text(
            json.dumps({"text": "hello", "language": 3}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="'language' must be a non-empty string"):
            make_config(tmp_path)


class TestSavedAudio:
    def test_audio_is_kept_only_for_measured_requests_of_a_saving_run(
        self, tmp_path: Path
    ) -> None:
        saving = make_config(tmp_path, "--save-audio-dir", str(tmp_path / "audio"))
        for config, warmup, kept in (
            (saving, False, [pcm(1.0)]),
            (saving, True, []),
            (make_config(tmp_path), False, []),
        ):
            recorder = StreamRecorder(
                config,
                sequence=0,
                request_id="r",
                sample_id="s",
                text="t",
                warmup=warmup,
            )
            recorder.mark_sent(0.0)
            recorder.add_raw_bytes(pcm(1.0), 0.1)
            assert recorder.audio_payloads == kept

    def test_a_save_directory_that_cannot_be_created_is_rejected(
        self, tmp_path: Path
    ) -> None:
        blocked = tmp_path / "blocked"
        blocked.write_text("a file where the directory should be", encoding="utf-8")
        with pytest.raises(ValueError, match="cannot create --save-audio-dir"):
            make_config(tmp_path, "--save-audio-dir", str(blocked))

    def test_a_request_whose_audio_cannot_be_written_is_still_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed write used to end the request's task before its record was submitted, so the request vanished from the report."""
        web = pytest.importorskip("aiohttp.web")

        def full_disk(path: str, mode: str) -> None:
            raise OSError("No space left on device")

        monkeypatch.setattr(wave, "open", full_disk)

        async def speak(request: web.Request) -> web.StreamResponse:
            response = web.StreamResponse()
            await response.prepare(request)
            await response.write(pcm(1.0))
            return response

        config, responder = run_against_local_server(
            tmp_path,
            speak,
            "--total-requests",
            "1",
            "--save-audio-dir",
            str(tmp_path / "audio"),
        )

        (record,) = responder.records
        assert record.is_success and record.wav_path is None
        (entry,) = json.loads(
            write_generated_metadata(config, responder.records).read_text("utf-8")
        )
        assert not entry["is_success"]
        assert entry["error"] == "audio was not saved"

    def test_a_run_saves_each_stream_and_writes_the_metadata_score_wer_reads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Drives the real sender against a local server: a WAV stream whose header is split, each row's own reference, and one failing request."""
        sender = pytest.importorskip("benchmarks.tts_serving.stream_stress.sender")
        web = pytest.importorskip("aiohttp.web")
        write_reference_corpus(tmp_path)
        audio_dir = tmp_path / "audio"
        speech = bytes(range(256)) * 188
        received: list[dict] = []
        saving_threads: list[int] = []
        save_audio = StreamRecorder.save_audio

        def save_audio_recording_thread(recorder: StreamRecorder) -> None:
            saving_threads.append(threading.get_ident())
            save_audio(recorder)

        monkeypatch.setattr(StreamRecorder, "save_audio", save_audio_recording_thread)

        async def speak(request: web.Request) -> web.StreamResponse:
            payload = await request.json()
            received.append(payload)
            if payload["input"] == "third":
                return web.Response(status=500, text="boom")
            else:
                pass
            response = web.StreamResponse(
                headers={"X-Sample-Rate": "16000", "X-Channels": "2"}
            )
            await response.prepare(request)
            header = b"RIFF" + b"\x00" * 40
            for network_bytes in (
                header[:20],
                header[20:] + speech[:1000],
                speech[1000:],
            ):
                await response.write(network_bytes)
            return response

        run_against_local_server(
            tmp_path,
            speak,
            "--response-format",
            "wav",
            "--total-requests",
            "3",
            "--warmup",
            "1",
            "--run-id",
            "team/run",
            "--save-audio-dir",
            str(audio_dir),
        )

        by_input = {payload["input"]: payload for payload in received[1:]}
        assert by_input["hello"]["ref_text"] == "prompt words"
        assert by_input["hello"]["ref_audio"].startswith("data:audio/wav;base64,")
        assert by_input["你好"]["language"] == "zh"
        assert "ref_audio" not in by_input["你好"]

        generated = json.loads((audio_dir / "generated.json").read_text("utf-8"))
        assert [entry["sample_id"] for entry in generated] == [
            "team/run-req-000000",
            "team/run-req-000001",
            "team/run-req-000002",
        ]
        assert len(saving_threads) == 2
        assert threading.get_ident() not in saving_threads
        cloned, plain, failed = generated
        assert (cloned["corpus_sample_id"], cloned["target_text"]) == (
            "cloned",
            "hello",
        )
        assert (cloned["language"], plain["language"]) == ("en", "zh")
        assert cloned["is_success"] and plain["is_success"]
        assert not failed["is_success"]
        assert failed["wav_path"] is None
        assert "HTTP 500" in failed["error"]
        assert sorted(path.name for path in audio_dir.glob("*.wav")) == [
            "team_run-req-000000.wav",
            "team_run-req-000001.wav",
        ]
        assert cloned["audio_duration_s"] == pytest.approx(len(speech) / 64000)
        assert cloned["latency_s"] > 0
        with wave.open(cloned["wav_path"], "rb") as wav_file:
            assert wav_file.getframerate() == 16000
            assert (wav_file.getnchannels(), wav_file.getsampwidth()) == (2, 2)
            assert wav_file.readframes(wav_file.getnframes()) == speech

    def test_the_command_writes_the_metadata_beside_the_audio(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_stress = pytest.importorskip(
            "benchmarks.tts_serving.stream_stress.run_stress"
        )
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(json.dumps({"text": "hello"}), encoding="utf-8")

        async def no_load(config, responder, stop) -> dict[str, float]:
            return {"wall_clock_s": 0.0}

        monkeypatch.setattr(run_stress, "execute", no_load)
        exit_code = run_stress.main(
            [
                "--text-jsonl",
                str(corpus),
                "--output",
                str(tmp_path / "out.jsonl"),
                "--save-audio-dir",
                str(tmp_path / "audio"),
                "--quiet",
            ]
        )

        assert exit_code == 0
        assert json.loads((tmp_path / "audio" / "generated.json").read_text()) == []

    def test_sse_audio_is_saved(self, tmp_path: Path) -> None:
        sender = pytest.importorskip("benchmarks.tts_serving.stream_stress.sender")
        config = make_config(
            tmp_path,
            "--stream-format",
            "sse",
            "--response-format",
            "wav",
            "--save-audio-dir",
            str(tmp_path / "audio"),
        )
        recorder = StreamRecorder(
            config, sequence=0, request_id="r", sample_id="s", text="t", warmup=False
        )
        recorder.mark_sent(0.0)
        for arrival_s, wav_header in ((0.1, b"RIFF" + b"\x00" * 40), (0.2, b"")):
            delta = json.dumps(
                {
                    "type": "speech.audio.delta",
                    "audio": base64.b64encode(wav_header + pcm(0.5)).decode(),
                }
            )
            sender.handle_sse_frame(recorder, f"data: {delta}".encode(), arrival_s)
        recorder.save_audio()
        assert recorder.audio_payloads == []

        with wave.open(recorder.finalize(load_start=0.0).wav_path, "rb") as wav_file:
            assert wav_file.getnframes() == 24000


class TestScoreWer:
    def test_saved_requests_are_scored_per_language(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Feeds a stress run's generated.json through the shared transcribe pipeline, with only the ASR request faked."""
        score_wer = pytest.importorskip(
            "benchmarks.tts_serving.stream_stress.score_wer"
        )
        tts = pytest.importorskip("benchmarks.tasks.tts")
        transcripts = {
            "run-req-000000": "hello world",
            "run-req-000001": "你好世界",
            "run-req-000003": "wrong words entirely here",
        }
        entries = [
            {
                "sample_id": sample_id,
                "corpus_sample_id": "s",
                "target_text": target_text,
                "language": language,
                "wav_path": str(tmp_path / f"{sample_id}.wav"),
                "is_success": is_success,
                "latency_s": 1.0,
                "audio_duration_s": 2.0,
                "error": None if is_success else "HTTP 500: boom",
            }
            for sample_id, target_text, language, is_success in (
                ("run-req-000000", "hello world", "en", True),
                ("run-req-000001", "你好世界", "zh", True),
                ("run-req-000002", "never spoken", None, False),
                ("run-req-000003", "good morning", None, True),
            )
        ]
        (tmp_path / "generated.json").write_text(
            json.dumps(entries, ensure_ascii=False), encoding="utf-8"
        )
        seen_languages: list[str] = []

        async def fake_transcription(samples, *, lang, **kwargs):
            seen_languages.append(lang)
            return (
                [
                    RequestResult(
                        request_id=sample.sample_id,
                        text=transcripts[sample.sample_id],
                        is_success=True,
                        latency_s=0.1,
                    )
                    for sample in samples
                ],
                1.0,
            )

        monkeypatch.setattr(tts, "run_asr_transcription", fake_transcription)

        assert (
            score_wer.main(["--audio-dir", str(tmp_path), "--use-existing-server"]) == 0
        )

        assert sorted(seen_languages) == ["en", "zh"]
        english = json.loads((tmp_path / "wer_en" / "wer_results.json").read_text())
        chinese = json.loads((tmp_path / "wer_zh" / "wer_results.json").read_text())
        assert chinese["summary"]["wer_corpus"] == 0.0
        assert [sample["id"] for sample in english["per_sample"]] == [
            "run-req-000000",
            "run-req-000002",
            "run-req-000003",
        ]
        exact, failed, runaway = english["per_sample"]
        assert exact["wer"] == 0.0
        assert not failed["is_success"]
        assert runaway["wer"] > 0.5
        assert english["summary"]["n_above_50_pct_wer"] == 1

    def test_a_language_the_scorer_does_not_cover_is_rejected(
        self, tmp_path: Path
    ) -> None:
        score_wer = pytest.importorskip(
            "benchmarks.tts_serving.stream_stress.score_wer"
        )
        (tmp_path / "generated.json").write_text(
            json.dumps([{"sample_id": "r", "language": "fr", "is_success": True}]),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="language 'fr'"):
            score_wer.split_by_language(tmp_path, "en")

    def test_a_run_without_saved_audio_is_not_scored(self, tmp_path: Path) -> None:
        score_wer = pytest.importorskip(
            "benchmarks.tts_serving.stream_stress.score_wer"
        )
        (tmp_path / "generated.json").write_text(
            json.dumps([{"sample_id": "r", "language": "en", "is_success": False}]),
            encoding="utf-8",
        )
        assert score_wer.main(["--audio-dir", str(tmp_path)]) == 2
        assert not (tmp_path / "wer_en").exists()
