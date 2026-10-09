# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import math
import sys
import threading
import types
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import soundfile
import websockets
from websockets.asyncio.server import ServerConnection

from benchmarks.duplex.run_artifacts import (
    TIMELINES,
    Timeline,
    create_output,
    load_output_transcripts,
    load_run,
)
from benchmarks.eval.benchmark_duplex_v15 import main
from tests.unit_test.benchmarks.test_duplex_client import DuplexPeer
from tests.unit_test.benchmarks.test_duplex_v15_runner import (
    SERVER_REVISION,
    write_dataset,
)

SELECTED = [
    "user_interruption/1",
    "user_backchannel/1",
    "talking_to_other/1",
    "background_speech/1",
]


@contextlib.contextmanager
def peer_server() -> Iterator[str]:
    """Serve fake native peers on a background loop so the CLI can own its own."""
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    state: dict = {}

    async def handler(websocket: ServerConnection) -> None:
        await DuplexPeer().handler(websocket)

    async def serve() -> None:
        state["stop"] = asyncio.Event()
        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            state["port"] = server.sockets[0].getsockname()[1]
            ready.set()
            await state["stop"].wait()

    thread = threading.Thread(target=loop.run_until_complete, args=(serve(),))
    thread.start()
    ready.wait(5)
    try:
        yield f"ws://127.0.0.1:{state['port']}/v1/realtime"
    finally:
        loop.call_soon_threadsafe(state["stop"].set)
        thread.join(5)
        loop.close()


def run_cli(argv: list[str]) -> tuple[int, dict]:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        code = main(argv)
    return code, json.loads(stdout.getvalue())


def tree_digest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def nonzero_segments(path: Path) -> dict:
    """Deterministic stand-in for Silero: runs of nonzero samples in the real WAV."""
    audio, rate = soundfile.read(str(path), dtype="int16")
    edges = np.flatnonzero(
        np.diff(np.concatenate([[0], (audio != 0).astype(int), [0]]))
    )
    return {
        "segments": (edges.reshape(-1, 2) / rate).tolist(),
        "duration_s": len(audio) / rate,
        "sample_rate": rate,
        "vad": {
            "package": "fixture-nonzero",
            "version": "0",
            "config": {},
            "config_hash": "0",
        },
    }


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("v15")
    dataset, run = root / "data", root / "run"
    write_dataset(dataset)
    (dataset / "talking_to_other" / "1" / "metadata.json").write_text("[]")
    (dataset / "background_speech" / "1" / "clean_input.wav").unlink()
    with peer_server() as url:
        code, summary = run_cli(
            ["record", "--dataset-root", str(dataset), "--url", url]
            + ["--output", str(run), "--server-revision", SERVER_REVISION]
            + ["--model", "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B"]
            + ["--dataset-revision", "fixture", "--timeout", "5"]
            + [arg for sample_id in SELECTED for arg in ("--sample-id", sample_id)]
        )
    return {"root": root, "run": run, "code": code, "summary": summary}


def test_record_reports_declared_available_selected_and_failures(
    recorded: dict,
) -> None:
    summary = recorded["summary"]
    assert recorded["code"] == 1
    assert summary["declared_pairs"] == 499 and summary["available_pairs"] == 4
    assert summary["per_subset"]["user_backchannel"] == {
        "declared": 99,
        "available": 1,
        "selected": 1,
    }
    assert summary["selected_pairs"] == 4 and summary["selected_variants"] == 8
    assert summary["attempted_variants"] == 4
    assert summary["variant_status"] == {"pass": 4, "invalid": 4}
    assert summary["qualified_pairs"] == 2
    assert {(f["sample_id"], f["variant"]) for f in summary["failures"]} == {
        ("talking_to_other/1", "overlap"),
        ("talking_to_other/1", "clean"),
        ("background_speech/1", "overlap"),
        ("background_speech/1", "clean"),
    }


class FakeWhisper:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict]] = []
        self.loads: list[tuple[str, str]] = []

    def transcribe(self, audio: np.ndarray, **options) -> dict:
        self.calls.append((len(audio), options))
        duration_s = len(audio) / 16000
        return {
            "text": " Yes, okay.",
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": duration_s,
                    "words": [
                        {
                            "word": " Yes,",
                            "start": 0.05,
                            "end": 0.2,
                            "probability": 0.9,
                        },
                        {
                            "word": " okay.",
                            "start": 0.35,
                            "end": duration_s,
                            "probability": 0.8,
                        },
                    ],
                }
            ],
        }


def transcribe_cli(
    run: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: FakeWhisper,
    timeline: Timeline = "simulated_playout",
) -> tuple[int, dict[str, int]]:
    model_path = tmp_path / "tiny.pt"
    model_path.write_bytes(b"weights")

    def load_model(name: str, device: str) -> FakeWhisper:
        model.loads.append((name, device))
        return model

    monkeypatch.setitem(
        sys.modules, "whisper", types.SimpleNamespace(load_model=load_model)
    )
    return run_cli(
        ["transcribe", "--run", str(run), "--output", str(tmp_path / "asr")]
        + ["--model-path", str(model_path), "--device", "cpu", "--timeline", timeline]
    )


def test_transcribe_preserves_audio_and_records_word_evidence(
    recorded: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = recorded["run"]
    before = tree_digest(run)
    fake = FakeWhisper()
    code, counts = transcribe_cli(run, tmp_path, monkeypatch, fake)

    assert code == 0 and counts == {"not_qualified:invalid": 4, "transcribed": 4}
    assert fake.loads == [(str(tmp_path / "tiny.pt"), "cpu")]
    assert fake.calls[0][1] == {
        "language": "en",
        "word_timestamps": True,
        "temperature": 0.0,
        "fp16": False,
    }
    transcripts = json.loads((tmp_path / "asr" / "transcripts.json").read_text())
    assert transcripts["asr"]["model_sha256"] == hashlib.sha256(b"weights").hexdigest()
    entry = next(e for e in transcripts["variants"] if e["status"] == "transcribed")
    assert entry["transcript"]["text"] == "Yes, okay."
    assert entry["transcript"]["chunks"][0] == {
        "text": "Yes,",
        "timestamp": [0.05, 0.2],
    }
    start_s, end_s = entry["transcript"]["chunks"][1]["timestamp"]
    assert start_s == 0.35 and end_s == pytest.approx(entry["duration_s"], abs=1e-3)
    raw = json.loads((tmp_path / "asr" / entry["raw_file"]).read_text())
    assert raw["segments"][0]["words"][1]["probability"] == 0.8

    assert tree_digest(run) == before


class FlakyWhisper(FakeWhisper):

    def transcribe(self, audio: np.ndarray, **options) -> dict:
        reply = super().transcribe(audio, **options)
        if len(self.calls) == 1:
            return {"text": " garbled", "language": "en"}
        elif len(self.calls) == 2:
            raise RuntimeError("CUDA error: device-side assert")
        return reply


def test_transcribe_keeps_going_after_per_variant_asr_failures(
    recorded: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = recorded["run"]
    code, counts = transcribe_cli(run, tmp_path, monkeypatch, FlakyWhisper())

    assert code == 1
    assert counts == {"error": 2, "not_qualified:invalid": 4, "transcribed": 2}
    entries = json.loads((tmp_path / "asr" / "transcripts.json").read_text())[
        "variants"
    ]
    assert len(entries) == 8
    malformed, crashed = entries[0], entries[1]
    assert (
        malformed["status"] == "error" and malformed["error"] == "KeyError: 'segments'"
    )
    raw = json.loads((tmp_path / "asr" / malformed["raw_file"]).read_text())
    assert raw == {"text": " garbled", "language": "en"}
    assert crashed["error"] == "RuntimeError: CUDA error: device-side assert"
    assert "raw_file" not in crashed and "transcript" not in crashed


class BadTimesWhisper(FakeWhisper):

    def transcribe(self, audio: np.ndarray, **options) -> dict:
        reply = super().transcribe(audio, **options)
        duration_s = len(audio) / 16000
        word = reply["segments"][0]["words"][1]
        bad_times = {
            1: (duration_s + 1.0, duration_s + 2.0),
            2: (0.3, 0.1),
            3: (float("nan"), 0.4),
        }
        if len(self.calls) in bad_times:
            word["start"], word["end"] = bad_times[len(self.calls)]
        return reply


def test_transcribe_rejects_invalid_word_times_instead_of_clipping(
    recorded: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = recorded["run"]
    code, counts = transcribe_cli(run, tmp_path, monkeypatch, BadTimesWhisper())

    assert code == 1
    assert counts == {"error": 3, "not_qualified:invalid": 4, "transcribed": 1}
    entries = [
        e
        for e in json.loads((tmp_path / "asr" / "transcripts.json").read_text())[
            "variants"
        ]
        if e["status"] in ("error", "transcribed")
    ]
    past_eof, reversed_word, nan_word, valid = entries
    assert "past audio end" in past_eof["error"]
    assert "reversed" in reversed_word["error"]
    assert "non-finite" in nan_word["error"]
    assert all("transcript" not in e for e in (past_eof, reversed_word, nan_word))
    raw = json.loads((tmp_path / "asr" / nan_word["raw_file"]).read_text())
    assert math.isnan(raw["segments"][0]["words"][1]["start"])
    assert valid["transcript"]["chunks"][1]["timestamp"][0] == 0.35


@pytest.mark.parametrize("timeline", ["media", "simulated_playout"])
def test_transcribe_selects_audio_timeline_and_rejects_mismatched_evidence(
    recorded: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timeline: Timeline,
) -> None:
    run = recorded["run"]
    code, _ = transcribe_cli(run, tmp_path, monkeypatch, FakeWhisper(), timeline)
    transcripts = json.loads((tmp_path / "asr" / "transcripts.json").read_text())
    assert code == 0 and transcripts["timeline"] == timeline
    assert all(
        entry["audio"].endswith(TIMELINES[timeline]["audio"])
        for entry in transcripts["variants"]
    )
    _, _, manifest_sha256 = load_run(run)
    evidence, _ = load_output_transcripts(
        tmp_path / "asr", run, manifest_sha256, timeline
    )
    assert len(evidence) == 4
    other_timeline = "media" if timeline == "simulated_playout" else "simulated_playout"
    with pytest.raises(ValueError, match="timeline"):
        load_output_transcripts(tmp_path / "asr", run, manifest_sha256, other_timeline)


def test_retired_score_command_is_rejected_without_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "score"
    with pytest.raises(SystemExit) as error:
        main(["score", "--run", str(tmp_path / "run"), "--output", str(output)])
    assert error.value.code == 2
    assert "invalid choice: 'score'" in capsys.readouterr().err
    assert not output.exists()


def test_outputs_never_overwrite_or_live_inside_the_run(
    recorded: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = recorded["run"]
    with pytest.raises(ValueError, match="outside the run"):
        create_output(run / "score", run)
    (tmp_path / "taken").mkdir()
    with pytest.raises(FileExistsError):
        create_output(tmp_path / "taken", run)

    loads = []
    monkeypatch.setitem(
        sys.modules,
        "whisper",
        types.SimpleNamespace(load_model=lambda *a, **k: loads.append(a)),
    )
    with pytest.raises(FileNotFoundError, match="local checkpoint"):
        main(
            ["transcribe", "--run", str(run), "--output", str(tmp_path / "asr")]
            + ["--model-path", "large-v3", "--device", "cpu"]
        )
    assert loads == [] and not (tmp_path / "asr").exists()
