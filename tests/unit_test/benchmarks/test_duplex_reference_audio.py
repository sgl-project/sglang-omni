# SPDX-License-Identifier: Apache-2.0
"""Verify immutable audio windows, lifecycle diagnostics and input integrity."""
import base64
import hashlib
import json
import shutil
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile
from numpy.typing import NDArray
from pydantic import JsonValue

from benchmarks.duplex import reference_audio
from benchmarks.duplex.reference_capture import TraceFormat
from benchmarks.duplex.reference_export import export_runs
from benchmarks.eval.benchmark_duplex_reference import main

SAMPLE_RATE = 16000
TRACE_ORIGIN_S = 1000.0
VariantName = Literal["overlap", "clean"]
TraceDirection = Literal["send", "receive", "error"]


def make_input_pcm(sample_count: int) -> bytes:
    generator = np.random.default_rng(sample_count)
    return generator.integers(-3000, 3000, sample_count, dtype="<i2").tobytes()


def encode_audio(audio: bytes) -> str:
    return base64.b64encode(audio).decode()


def make_constant_pcm(
    duration_s: float, sample_rate: int, amplitude: int = 1000
) -> bytes:
    return np.full(round(duration_s * sample_rate), amplitude, "<i2").tobytes()


def make_session_trace(
    input_pcm: bytes,
    *,
    audio_deltas: Sequence[tuple[float, bytes | str]] = (),
    output_sample_rate: int = SAMPLE_RATE,
    server_error_offsets_s: Sequence[float] = (),
    client_error_offsets_s: Sequence[float] = (),
    close_offset_s: float | Literal[False] | None = None,
    append_delays_s: dict[int, float] | None = None,
    omit_last_append: bool = False,
    corrupted_append_index: int | None = None,
    created_response_ids: Sequence[str] = ("r1",),
    completed_response_ids: Sequence[str] = (),
) -> str:
    """Records in time order; deltas are (elapsed_s, bytes or raw base64 string)."""
    input_samples = np.frombuffer(input_pcm, "<i2")
    window_s = len(input_samples) / SAMPLE_RATE
    rows: list[tuple[float, TraceDirection, dict[str, JsonValue]]] = []
    append_delays_s = append_delays_s or {}
    rows.append(
        (
            TRACE_ORIGIN_S - 0.01,
            "receive",
            {
                "type": "session.updated",
                "session": {
                    "audio": {
                        "output": {
                            "format": {"type": "audio/pcm", "rate": output_sample_rate}
                        }
                    }
                },
            },
        )
    )
    frame_indexes = range(-(-len(input_samples) // 1280))
    for index in frame_indexes:
        if omit_last_append and index == frame_indexes[-1]:
            continue
        else:
            pass
        input_chunk = input_samples[index * 1280 : (index + 1) * 1280]
        audio_payload = (
            input_chunk.tobytes()
            if corrupted_append_index != index
            else (input_chunk + 1).tobytes()
        )
        event = {
            "type": "input_audio_buffer.append",
            "event_id": f"a{index}",
            "audio": encode_audio(audio_payload),
            "sglang": {"seq": index, "t_start_ms": index * 1280 / SAMPLE_RATE * 1000},
        }
        rows.append(
            (
                TRACE_ORIGIN_S + index * 0.08 + append_delays_s.get(index, 0.0),
                "send",
                event,
            )
        )
    for response_id in created_response_ids:
        rows.append(
            (
                TRACE_ORIGIN_S + 0.01,
                "receive",
                {"type": "response.created", "response": {"id": response_id}},
            )
        )
    for elapsed_s, audio_payload in audio_deltas:
        event = {
            "type": "response.output_audio.delta",
            "response_id": "r1",
            "delta": (
                audio_payload
                if isinstance(audio_payload, str)
                else encode_audio(audio_payload)
            ),
        }
        rows.append((TRACE_ORIGIN_S + elapsed_s, "receive", event))
    for response_id in completed_response_ids:
        rows.append(
            (
                TRACE_ORIGIN_S + window_s + 0.3,
                "receive",
                {
                    "type": "response.done",
                    "response": {"id": response_id, "status": "completed"},
                },
            )
        )
    for elapsed_s in server_error_offsets_s:
        rows.append(
            (
                TRACE_ORIGIN_S + elapsed_s,
                "receive",
                {"type": "error", "error": {"message": "boom"}},
            )
        )
    for elapsed_s in client_error_offsets_s:
        rows.append(
            (TRACE_ORIGIN_S + elapsed_s, "error", {"message": "client failure"})
        )
    if close_offset_s is None:
        close_offset_s = window_s + 0.6
    else:
        pass
    if close_offset_s is not False:
        rows.append(
            (TRACE_ORIGIN_S + close_offset_s, "receive", {"type": "session.closed"})
        )
    else:
        pass
    rows.sort(key=lambda row: row[0])
    return "".join(
        json.dumps({"direction": direction, "time_s": time_s, "event": event}) + "\n"
        for time_s, direction, event in rows
    )


def make_send_receipts(
    trace_text: str, completion_delays_s: dict[int, float] | None = None
) -> list[dict[str, JsonValue]]:
    """One completed send receipt per append, 0.1 ms after its start unless overridden."""
    completion_delays_s = completion_delays_s or {}
    appends = [
        record
        for record in map(json.loads, trace_text.splitlines())
        if record["direction"] == "send"
    ]
    return [
        {
            "event_id": append["event"]["event_id"],
            "seq": append["event"]["sglang"]["seq"],
            "start_s": append["time_s"],
            "completed_s": append["time_s"] + completion_delays_s.get(index, 1e-4),
        }
        for index, append in enumerate(appends)
    ]


class AudioFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory: tempfile.TemporaryDirectory[str] = (
            tempfile.TemporaryDirectory()
        )
        self.root: Path = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def make_run(self, variants: dict[VariantName, tuple[bytes, str]]) -> Path:
        """Write a paired SGLang capture with the supplied PCM and trace text."""
        run = self.root / "run"
        sample_id = "user_interruption/1"
        states = {}
        for variant, (input_pcm, trace_text) in variants.items():
            directory = Path("samples") / sample_id / variant
            (run / directory).mkdir(parents=True)
            (run / directory / "input.pcm").write_bytes(input_pcm)
            (run / directory / "continuous.jsonl").write_text(trace_text)
            states[variant] = {
                "directory": str(directory),
                "status": "captured",
                "input": {"sha256": hashlib.sha256(input_pcm).hexdigest()},
                "source": {"file": f"{sample_id}/input.wav", "sha256": None},
            }
        (run / "manifest.json").write_text(json.dumps({"profile": "sglang"}))
        (run / "run.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "samples": [{"id": sample_id, "variants": states}],
                }
            )
        )
        return run

    def analyze(
        self,
        input_pcm: bytes,
        trace_text: str,
        expected_input_sha256: str | None = None,
        send_receipts: list[dict[str, JsonValue]] | None = None,
    ) -> tuple[dict[str, JsonValue], bytes | None, NDArray[np.int16] | None]:
        variant_directory = self.root / f"v{len(list(self.root.iterdir()))}"
        variant_directory.mkdir()
        (variant_directory / "input.pcm").write_bytes(input_pcm)
        (variant_directory / "continuous.jsonl").write_text(trace_text)
        if send_receipts is not None:
            (variant_directory / reference_audio.SEND_RECEIPTS).write_text(
                json.dumps({"appends": send_receipts})
            )
        else:
            pass
        expected_input_sha256 = (
            hashlib.sha256(input_pcm).hexdigest()
            if expected_input_sha256 is None
            else expected_input_sha256
        )
        return reference_audio.analyze_variant(
            variant_directory, TraceFormat.PCM16, expected_input_sha256
        )


class WindowEligibility(AudioFixture):
    pcm = make_input_pcm(8000)

    def test_capture_preserves_audio_and_send_evidence(self) -> None:
        pcm = make_input_pcm(7681)
        trace = make_session_trace(
            pcm,
            audio_deltas=[
                (0.1, make_constant_pcm(0.08, 24000)),
                (0.46, make_constant_pcm(0.1, 24000, 2000)),
                (0.50, make_constant_pcm(0.1, 24000, 3000)),
            ],
            output_sample_rate=24000,
            append_delays_s=dict.fromkeys(range(1, 7), 0.001),
        )
        record, input_pcm, audio = self.analyze(
            pcm, trace, send_receipts=make_send_receipts(trace)
        )
        self.assertTrue(record["window"]["valid"], record["window"]["reasons"])
        self.assertEqual(input_pcm, pcm)
        self.assertEqual(
            record["source"]["trace_sha256"], hashlib.sha256(trace.encode()).hexdigest()
        )
        self.assertEqual(
            record["source"]["input_pcm_sha256"], hashlib.sha256(pcm).hexdigest()
        )
        self.assertEqual(len(audio), 7681)
        self.assertEqual(record["input_check"]["append_completions_after_T"], 1)
        self.assertTrue(record["boundary"]["playout_active_at_T"])

    def test_missing_terminal_is_lifecycle_only(self) -> None:
        record, pcm, out = self.analyze(
            self.pcm,
            make_session_trace(
                self.pcm,
                audio_deltas=[(0.1, make_constant_pcm(0.08, 22050))],
                output_sample_rate=22050,
                created_response_ids=("r1", "r2"),
                completed_response_ids=("r1",),
            ),
        )
        self.assertTrue(record["window"]["valid"], record["window"]["reasons"])
        self.assertEqual(record["lifecycle"]["responses_missing_terminal"], ["r2"])
        self.assertIsNone(record["lifecycle"]["natural_completion"])
        self.assertEqual(len(out), 8000)
        self.assertEqual(pcm, self.pcm)

    def test_server_error_or_disconnect_before_window_end_invalidates(self) -> None:
        for kwargs, expect in (
            ({"server_error_offsets_s": [0.2]}, "native server error"),
            ({"client_error_offsets_s": [0.3]}, "client error"),
            ({"close_offset_s": 0.3}, "session.closed"),
        ):
            record, _, out = self.analyze(
                self.pcm, make_session_trace(self.pcm, **kwargs)
            )
            self.assertFalse(record["window"]["valid"])
            self.assertIn(expect, " ".join(record["window"]["reasons"]))
            self.assertIsNone(out)

    def test_liveness_after_window_must_be_proven(self) -> None:
        record, _, _ = self.analyze(
            self.pcm, make_session_trace(self.pcm, close_offset_s=False)
        )
        self.assertIn(
            "receiver liveness after window end unproven", record["window"]["reasons"]
        )

    def test_post_window_error_keeps_window_audio(self) -> None:
        record, _, out = self.analyze(
            self.pcm,
            make_session_trace(
                self.pcm,
                audio_deltas=[(0.1, make_constant_pcm(0.1, SAMPLE_RATE)), (0.9, "!!!")],
                server_error_offsets_s=[0.8],
                client_error_offsets_s=[0.85],
            ),
        )
        self.assertTrue(record["window"]["valid"], record["window"]["reasons"])
        self.assertEqual(len(record["lifecycle"]["post_window_errors"]), 3)
        self.assertEqual(int(np.count_nonzero(out)), 1600)

    def test_healthy_silence_is_valid(self) -> None:
        record, _, out = self.analyze(self.pcm, make_session_trace(self.pcm))
        self.assertTrue(record["window"]["valid"], record["window"]["reasons"])
        self.assertTrue(record["output"]["silent"])
        self.assertEqual(out.tolist(), [0] * 8000)

    def test_audio_after_window_is_excluded_not_backdated(self) -> None:
        record, _, out = self.analyze(
            self.pcm,
            make_session_trace(
                self.pcm, audio_deltas=[(0.5001, make_constant_pcm(0.2, SAMPLE_RATE))]
            ),
        )
        self.assertTrue(record["window"]["valid"])
        self.assertEqual(record["output"]["packets_after_window_excluded"], 1)
        self.assertFalse(np.any(out))

    def test_fifo_backlog_is_cropped_at_window_end(self) -> None:
        record, _, out = self.analyze(
            self.pcm,
            make_session_trace(
                self.pcm,
                audio_deltas=[
                    (0.05, make_constant_pcm(0.3, SAMPLE_RATE, 1)),
                    (0.06, make_constant_pcm(0.3, SAMPLE_RATE, 2)),
                ],
            ),
        )
        expect = np.zeros(8000, "<i2")
        expect[800:5600] = 1
        expect[5600:8000] = 2
        np.testing.assert_array_equal(out, expect)
        self.assertTrue(record["boundary"]["playout_active_at_T"])
        self.assertAlmostEqual(record["boundary"]["queued_audio_cropped_s"], 0.15)

    def test_native_rate_resampled_to_exact_input_count(self) -> None:
        pcm = make_input_pcm(7999)
        record, _, out = self.analyze(
            pcm,
            make_session_trace(
                pcm,
                audio_deltas=[(0.0, make_constant_pcm(0.6, 22050))],
                output_sample_rate=22050,
            ),
        )
        self.assertTrue(record["window"]["valid"], record["window"]["reasons"])
        self.assertEqual(len(out), 7999)
        self.assertEqual(
            record["output"]["resample"],
            {"method": "scipy.signal.resample_poly", "up": 320, "down": 441},
        )

    def test_malformed_output_in_window_invalidates(self) -> None:
        for bad in ("!!!", encode_audio(b"\x01"), ""):
            record, _, out = self.analyze(
                self.pcm, make_session_trace(self.pcm, audio_deltas=[(0.1, bad)])
            )
            self.assertFalse(record["window"]["valid"])
            self.assertIsNone(out)
        record, _, _ = self.analyze(
            self.pcm,
            make_session_trace(
                self.pcm,
                audio_deltas=[(0.1, make_constant_pcm(0.08, SAMPLE_RATE))],
                output_sample_rate=24000,
            ),
        )
        self.assertTrue(record["window"]["valid"])
        self.assertEqual(record["output"]["native_rate"], 24000)

    def test_malformed_input_hash_and_frames_invalidate(self) -> None:
        cases = [
            ({"corrupted_append_index": 6}, None, "serialized PCM16 differs"),
            ({"corrupted_append_index": 2}, None, "serialized PCM16 differs"),
            ({}, "0" * 64, "sha256 differs from run.json"),
            (
                {"append_delays_s": dict.fromkeys(range(3, 7), 0.09)},
                None,
                "pacing deviation",
            ),
            ({"omit_last_append": True}, None, "incomplete append population"),
        ]
        for kwargs, expected, message in cases:
            record, _, _ = self.analyze(
                self.pcm, make_session_trace(self.pcm, **kwargs), expected
            )
            self.assertFalse(record["window"]["valid"], kwargs)
            self.assertIn(message, " ".join(record["window"]["reasons"]))

    def test_missing_send_completion_invalidates(self) -> None:
        trace = make_session_trace(self.pcm)
        receipts = make_send_receipts(trace)
        receipts[3]["completed_s"] = None
        record, _, _ = self.analyze(self.pcm, trace, send_receipts=receipts)
        self.assertFalse(record["window"]["valid"])
        self.assertIn(
            "send completion not recorded", " ".join(record["window"]["reasons"])
        )


class TraceClock(AudioFixture):
    pcm = make_input_pcm(8000)

    def test_malformed_or_nonmonotonic_clock_invalidates(self) -> None:
        lines = make_session_trace(self.pcm).splitlines(keepends=True)
        for bad in ("NaN", "Infinity", '"1001.0"', "true"):
            text = lines[:]
            row = json.loads(text[5])
            text[5] = json.dumps(row).replace(
                f'"time_s": {row["time_s"]}', f'"time_s": {bad}'
            )
            text[5] += "\n"
            record, _, _ = self.analyze(self.pcm, "".join(text))
            self.assertFalse(record["window"]["valid"], bad)
            self.assertIn("clock", " ".join(record["window"]["reasons"]))
        text = lines[:]
        text[4], text[5] = text[5], text[4]
        record, _, _ = self.analyze(self.pcm, "".join(text))
        self.assertIn(
            "trace clock not monotonic", " ".join(record["window"]["reasons"])
        )

    def test_legacy_trace_inference_is_tagged(self) -> None:
        trace = make_session_trace(self.pcm)
        record, _, _ = self.analyze(self.pcm, trace)
        self.assertTrue(record["window"]["valid"], record["window"]["reasons"])
        self.assertTrue(record["input_check"]["legacy_inference"])
        self.assertIn("legacy", record["input_check"]["send_completion_evidence"])
        record, _, _ = self.analyze(
            self.pcm, trace, send_receipts=make_send_receipts(trace)
        )
        self.assertFalse(record["input_check"]["legacy_inference"])

    def test_completion_grace_accepts_tiny_tail_but_rejects_stall(self) -> None:
        pcm = make_input_pcm(7681)
        trace = make_session_trace(
            pcm, append_delays_s=dict.fromkeys(range(1, 7), 0.001)
        )
        record, _, audio = self.analyze(
            pcm, trace, send_receipts=make_send_receipts(trace)
        )
        self.assertTrue(record["window"]["valid"], record["window"]["reasons"])
        self.assertEqual(len(audio), 7681)
        self.assertEqual(record["input_check"]["append_completions_after_T"], 1)
        record, _, audio = self.analyze(
            pcm, trace, send_receipts=make_send_receipts(trace, {6: 0.3})
        )
        self.assertFalse(record["window"]["valid"])
        self.assertIn(
            "completed after v2 deadline", " ".join(record["window"]["reasons"])
        )
        self.assertIsNone(audio)


class ReferenceExport(AudioFixture):
    def test_engine_label_is_metadata_only(self) -> None:
        pcm = make_input_pcm(7681)
        run = self.make_run(
            {
                variant: (pcm, make_session_trace(pcm))
                for variant in ("overlap", "clean")
            }
        )
        original = export_runs([run], self.root / "original", "sglang")
        relabeled_output = self.root / "relabeled"
        self.assertEqual(
            main(
                [
                    "export",
                    "--engine",
                    "other-engine",
                    "--trace-format",
                    TraceFormat.PCM16.value,
                    "--run",
                    str(run),
                    "--out",
                    str(relabeled_output),
                ]
            ),
            0,
        )
        relabeled = json.loads(
            (relabeled_output / "reference-manifest.json").read_text()
        )
        self.assertEqual(relabeled["engine"], "other-engine")
        self.assertEqual(relabeled["trace_format"], TraceFormat.PCM16.value)
        self.assertEqual(original["samples"], relabeled["samples"])
        self.assertEqual(original["counts"], relabeled["counts"])

    def test_unknown_format_is_rejected_before_export(self) -> None:
        output = self.root / "export"
        with self.assertRaises(SystemExit) as error:
            main(
                [
                    "export",
                    "--engine",
                    "sglang",
                    "--trace-format",
                    "unknown",
                    "--run",
                    str(self.root / "missing"),
                    "--out",
                    str(output),
                ]
            )
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(output.exists())
        with self.assertRaises(ValueError):
            export_runs(
                [], self.root / "unknown", "other-engine", trace_format="unknown"
            )
        self.assertFalse((self.root / "unknown").exists())

    def test_declared_send_receipts_cannot_fall_back_to_legacy(self) -> None:
        pcm = make_input_pcm(8000)
        text = make_session_trace(pcm)
        run = self.make_run({v: (pcm, text) for v in ("overlap", "clean")})
        path = run / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["config"] = {
            "transport": {"input_send_receipts": reference_audio.SEND_RECEIPTS}
        }
        path.write_text(json.dumps(manifest))
        result = export_runs([run], self.root / "export", "sglang")
        self.assertEqual(result["counts"]["eligible_variants"], 0)
        for state in result["samples"][0]["variants"].values():
            self.assertIn("missing", " ".join(state["reasons"]))
            self.assertFalse(state["input_check"]["legacy_inference"])

    def test_export_uses_send_receipts_without_extending_the_audio_window(self) -> None:
        input_pcm = make_input_pcm(7681)
        capture_trace = make_session_trace(
            input_pcm,
            append_delays_s={6: 0.001},
            audio_deltas=[
                (0.46, make_constant_pcm(0.1, SAMPLE_RATE, 1000)),
                (0.49, make_constant_pcm(0.1, SAMPLE_RATE, 2000)),
            ],
        )
        run = self.make_run(
            {variant: (input_pcm, capture_trace) for variant in ("overlap", "clean")}
        )
        manifest_path = run / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["config"] = {
            "transport": {"input_send_receipts": reference_audio.SEND_RECEIPTS}
        }
        manifest_path.write_text(json.dumps(manifest))
        receipts = make_send_receipts(capture_trace)
        window_s = 7681 / SAMPLE_RATE
        receipts[-1]["completed_s"] = TRACE_ORIGIN_S + window_s + 0.04
        receipt_bytes = json.dumps({"appends": receipts}).encode()
        for variant in ("overlap", "clean"):
            (
                run
                / "samples/user_interruption/1"
                / variant
                / reference_audio.SEND_RECEIPTS
            ).write_bytes(receipt_bytes)

        output = self.root / "export"
        result = export_runs([run], output, "sglang")

        self.assertEqual(result["counts"]["eligible_variants"], 2)
        for variant in result["samples"][0]["variants"].values():
            self.assertFalse(variant["input_check"]["legacy_inference"])
            self.assertEqual(variant["input_check"]["append_starts_after_T"], 1)
            self.assertEqual(variant["input_check"]["append_completions_after_T"], 1)
            self.assertEqual(variant["input_check"]["completions_recorded"], 7)
            self.assertAlmostEqual(
                variant["input_check"]["max_append_completion_lateness_s"], 0.04
            )
            self.assertEqual(variant["output"]["packets_after_window_excluded"], 1)
            self.assertEqual(
                variant["source"]["input_send_receipts_sha256"],
                hashlib.sha256(receipt_bytes).hexdigest(),
            )
        expected_audio = np.zeros(7681, dtype=np.int16)
        expected_audio[7360:] = 1000
        for filename in ("output.wav", "clean_output.wav"):
            audio, sample_rate = soundfile.read(
                output / "user_interruption/1" / filename, dtype="int16"
            )
            self.assertEqual(sample_rate, SAMPLE_RATE)
            np.testing.assert_array_equal(audio, expected_audio)

        clean_receipts = (
            run / "samples/user_interruption/1/clean" / reference_audio.SEND_RECEIPTS
        )
        self.assertEqual(clean_receipts.read_bytes(), receipt_bytes)
        receipts[-1]["event_id"] = "unmatched-append"
        clean_receipts.write_text(json.dumps({"appends": receipts}))
        rejected_output = self.root / "mismatched-export"
        rejected = export_runs([run], rejected_output, "sglang")
        self.assertEqual(rejected["counts"]["eligible_variants"], 1)
        self.assertEqual(rejected["counts"]["selected_variants"], 2)
        variants = rejected["samples"][0]["variants"]
        self.assertTrue(variants["overlap"]["eligible"])
        self.assertIn(
            "append event_ids differ from trace", " ".join(variants["clean"]["reasons"])
        )
        self.assertFalse(
            (rejected_output / "user_interruption/1/clean_output.wav").exists()
        )

    def test_dataset_population_retains_missing_samples_and_source_hash_failures(
        self,
    ) -> None:
        input_pcm = make_input_pcm(8000)
        run = self.make_run(
            {
                variant: (input_pcm, make_session_trace(input_pcm))
                for variant in ("overlap", "clean")
            }
        )
        dataset = self.root / "dataset"
        sample_directory = dataset / "user_interruption/1"
        sample_directory.mkdir(parents=True)
        (dataset / "background_speech/2").mkdir(parents=True)
        run_path = run / "run.json"
        run_document = json.loads(run_path.read_text())
        variants = run_document["samples"][0]["variants"]
        for variant, filename in (
            ("overlap", "input.wav"),
            ("clean", "clean_input.wav"),
        ):
            source_path = sample_directory / filename
            soundfile.write(
                source_path,
                np.frombuffer(input_pcm, "<i2"),
                SAMPLE_RATE,
                subtype="PCM_16",
            )
            variants[variant]["source"] = {
                "file": f"user_interruption/1/{filename}",
                "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            }
        run_path.write_text(json.dumps(run_document))
        changed_audio = np.frombuffer(input_pcm, "<i2").copy()
        changed_audio[0] += 1
        soundfile.write(
            sample_directory / "clean_input.wav",
            changed_audio,
            SAMPLE_RATE,
            subtype="PCM_16",
        )

        output = self.root / "export"
        result = export_runs([run], output, "sglang", dataset_root=dataset)

        self.assertEqual(
            result["counts"],
            {
                "selected_pairs": 2,
                "selected_variants": 4,
                "eligible_pairs": 0,
                "eligible_variants": 1,
            },
        )
        samples = {sample["sample_id"]: sample for sample in result["samples"]}
        captured = samples["user_interruption/1"]["variants"]
        self.assertTrue(captured["overlap"]["eligible"])
        self.assertFalse(captured["clean"]["eligible"])
        self.assertEqual(
            captured["clean"]["reasons"],
            ["dataset source sha256 differs from run.json"],
        )
        for missing in samples["background_speech/2"]["variants"].values():
            self.assertFalse(missing["eligible"])
            self.assertEqual(
                missing["reasons"], ["pair not captured in any supplied run"]
            )
        self.assertEqual(
            set(samples["user_interruption/1"]["files"]), {"input.wav", "output.wav"}
        )
        self.assertEqual(samples["background_speech/2"]["files"], {})
        self.assertFalse((output / "user_interruption/1/clean_output.wav").exists())

    def test_later_capture_supersedes_an_incomplete_pair(self) -> None:
        input_pcm = make_input_pcm(8000)
        first_trace = make_session_trace(
            input_pcm,
            audio_deltas=[(0.1, make_constant_pcm(0.1, SAMPLE_RATE, 500))],
        )
        first_run = self.make_run(
            {variant: (input_pcm, first_trace) for variant in ("overlap", "clean")}
        ).rename(self.root / "first-run")
        first_run_path = first_run / "run.json"
        first_document = json.loads(first_run_path.read_text())
        first_document["samples"][0]["variants"]["clean"]["status"] = "error"
        first_run_path.write_text(json.dumps(first_document))
        retry_trace = make_session_trace(
            input_pcm,
            audio_deltas=[(0.1, make_constant_pcm(0.1, SAMPLE_RATE, 2000))],
        )
        retry_run = self.make_run(
            {variant: (input_pcm, retry_trace) for variant in ("overlap", "clean")}
        )

        output = self.root / "export"
        result = export_runs([first_run, retry_run], output, "sglang")

        self.assertEqual(result["counts"]["selected_pairs"], 1)
        self.assertEqual(result["counts"]["eligible_pairs"], 1)
        self.assertEqual(
            result["superseded"],
            [{"id": "user_interruption/1", "run": str(first_run.resolve())}],
        )
        self.assertEqual(
            [source["run"] for source in result["sources"]],
            [str(first_run.resolve()), str(retry_run.resolve())],
        )
        expected_audio = np.zeros(8000, dtype=np.int16)
        expected_audio[1600:3200] = 2000
        for filename in ("output.wav", "clean_output.wav"):
            audio, _ = soundfile.read(
                output / "user_interruption/1" / filename, dtype="int16"
            )
            np.testing.assert_array_equal(audio, expected_audio)

    def test_duplicate_complete_captures_are_rejected_before_export(self) -> None:
        input_pcm = make_input_pcm(8000)
        run = self.make_run(
            {
                variant: (input_pcm, make_session_trace(input_pcm))
                for variant in ("overlap", "clean")
            }
        )
        duplicate_run = self.root / "duplicate-run"
        shutil.copytree(run, duplicate_run)
        output = self.root / "export"

        with self.assertRaisesRegex(ValueError, "captured completely in two runs"):
            export_runs([run, duplicate_run], output, "sglang")

        self.assertFalse(output.exists())

    def test_public_export_preserves_population_and_source_bytes(self) -> None:
        pcm = make_input_pcm(8000)
        text = make_session_trace(
            pcm,
            output_sample_rate=24000,
            audio_deltas=[(0.1, make_constant_pcm(0.2, 24000))],
        )
        run = self.make_run({v: (pcm, text) for v in ("overlap", "clean")})
        before = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in run.rglob("*")
            if p.is_file()
        }
        out = self.root / "export"
        result = export_runs(
            [run], out, "sglang", ["user_interruption/1", "background_speech/2"]
        )
        self.assertEqual(
            result["counts"],
            {
                "selected_pairs": 2,
                "selected_variants": 4,
                "eligible_variants": 2,
                "eligible_pairs": 1,
            },
        )
        for variant in result["samples"][1]["variants"].values():
            self.assertFalse(variant["eligible"])
            self.assertTrue(variant["reasons"])
        audio, rate = soundfile.read(out / "user_interruption/1/output.wav")
        self.assertEqual((len(audio), rate), (8000, 16000))
        self.assertGreater(np.count_nonzero(audio), 0)
        after = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in run.rglob("*")
            if p.is_file()
        }
        self.assertEqual(before, after)
        with self.assertRaises(FileExistsError):
            export_runs([run], out, "sglang")
        with self.assertRaises(ValueError):
            export_runs([run], run / "export", "sglang")

    def test_export_cli_accepts_valid_silence(self) -> None:
        pcm = make_input_pcm(8000)
        run = self.make_run(
            {v: (pcm, make_session_trace(pcm)) for v in ("overlap", "clean")},
        )
        out = self.root / "export"
        self.assertEqual(
            main(
                ["export", "--engine", "sglang", "--run", str(run), "--out", str(out)]
            ),
            0,
        )
        audio, _ = soundfile.read(out / "user_interruption/1/output.wav")
        self.assertEqual(np.count_nonzero(audio), 0)
