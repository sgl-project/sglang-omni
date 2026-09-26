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

import httpx
import numpy as np
import pytest
import soundfile
import websockets
from websockets.asyncio.server import ServerConnection

from benchmarks.duplex import v15_scoring
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


@pytest.fixture
def fake_vad(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v15_scoring, "silero_speech_segments", nonzero_segments)


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


def test_score_without_asr_keeps_timing_and_leaves_behavior_unscored(
    recorded: dict, fake_vad: None, tmp_path: Path
) -> None:
    before = tree_digest(recorded["run"])
    code, printed = run_cli(
        ["score", "--run", str(recorded["run"]), "--output", str(tmp_path / "s")]
    )

    assert code == 0 and tree_digest(recorded["run"]) == before
    score = json.loads((tmp_path / "s" / "score.json").read_text())
    assert score["timeline"]["name"] == "simulated_playout"
    assert score["timeline"]["audio"] == "output-playout.wav"
    categories = score["timing"]["categories"]
    assert categories["backchannel"]["eligible"] == 1
    assert categories["backchannel"]["paired_eligible"] == 1
    assert categories["interruption"]["paired_eligible"] == 1
    assert categories["talking_to_other"]["selected"] == 1
    assert categories["talking_to_other"]["missing"] == 1
    assert categories["background_speech"]["missing"] == 1
    assert printed["timing"]["talking_to_other"]["eligible"] == 0

    rows = {row["sample_id"]: row for row in score["samples"]}
    clean = rows["user_backchannel/1"]["variants"]["clean"]["timing"]
    assert clean["status"] == "eligible"
    assert clean["segment_source"]["input"]["audio"].endswith("/clean/input.wav")
    assert clean["evaluation"] == "clean_reference"
    playout = recorded["run"] / "samples/user_backchannel/1/clean/output-playout.wav"
    info = soundfile.info(str(playout))
    assert clean["output_duration_s"] == info.frames / info.samplerate
    assert clean["observed_end_s"] >= clean["output_duration_s"]
    assert clean["observation_complete"] is True
    assert clean["segment_source"]["output"]["audio"].endswith(
        "/clean/output-playout.wav"
    )
    assert (
        rows["talking_to_other/1"]["variants"]["overlap"]["unscored_reason"]
        == "invalid_sample"
    )

    behavior = score["behavior"]["summary"]["categories"]
    assert behavior["backchannel"]["scored"] == 0
    assert behavior["backchannel"]["unscored_reasons"] == {
        "missing_transcript:clean_output": 1
    }
    assert behavior["talking_to_other"]["unscored_reasons"] == {"invalid_sample": 1}
    missing_clean = rows["background_speech/1"]
    assert missing_clean["errors"] == ["missing clean_input.wav"]
    assert missing_clean["behavior_input"]["reason"] == "invalid_sample"
    assert {v["unscored_reason"] for v in missing_clean["variants"].values()} == {
        "invalid_sample"
    }
    assert behavior["background_speech"]["unscored_reasons"] == {"invalid_sample": 1}
    inputs = (tmp_path / "s" / "judge-inputs.jsonl").read_text().splitlines()
    assert {json.loads(line)["status"] for line in inputs} == {"unscorable"}
    assert len(inputs) == 4


class FakeWhisper:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict]] = []

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


def test_transcribe_then_offline_judgements_score_behavior(
    recorded: dict, fake_vad: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = recorded["run"]
    model_path = tmp_path / "tiny.pt"
    model_path.write_bytes(b"weights")
    fake = FakeWhisper()
    loads = []

    def load_model(name: str, device: str) -> FakeWhisper:
        loads.append((name, device))
        return fake

    monkeypatch.setitem(
        sys.modules, "whisper", types.SimpleNamespace(load_model=load_model)
    )
    code, counts = run_cli(
        ["transcribe", "--run", str(run), "--output", str(tmp_path / "asr")]
        + ["--model-path", str(model_path), "--device", "cpu"]
    )

    assert code == 0 and counts == {"not_qualified:invalid": 4, "transcribed": 4}
    assert loads == [(str(model_path), "cpu")]
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

    first = tmp_path / "first"
    run_cli(
        ["score", "--run", str(run), "--output", str(first)]
        + ["--transcripts", str(tmp_path / "asr")]
    )
    ready = [
        json.loads(line)
        for line in (first / "judge-inputs.jsonl").read_text().splitlines()
        if json.loads(line)["status"] == "ready"
    ]
    assert [item["sample_id"] for item in ready] == SELECTED[:2]
    labels = dict(zip(SELECTED[:2], ("C_RESUME", "C_RESPOND")))
    noisy = ready[0]["payload"]["transcripts"]["noisy_output"]
    assert noisy["timestamp_source"] == "asr_aligned"
    assert (
        ready[0]["payload"]["transcripts"]["noisy_input"]["timestamp_source"]
        == "provided_aligned"
    )

    judgements = tmp_path / "judgements.jsonl"
    judgements.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": item["sample_id"],
                    "input_hash": item["input_hash"],
                    "rubric_version": item["rubric_version"],
                    "label": labels[item["sample_id"]],
                    "evidence": "okay",
                    "first_new_segment": {
                        "text": word["text"],
                        "start_s": word["start_s"],
                        "end_s": word["end_s"],
                    },
                    "annotator": {"id": "fixture", "kind": "human"},
                }
            )
            + "\n"
            for item in ready
            for word in item["payload"]["transcripts"]["noisy_output"]["words"][1:]
        )
    )
    code, printed = run_cli(
        ["score", "--run", str(run), "--output", str(tmp_path / "judged")]
        + ["--transcripts", str(tmp_path / "asr"), "--judgements", str(judgements)]
        + ["--segments", str(first / "segments.json")]
    )

    score = json.loads((tmp_path / "judged" / "score.json").read_text())
    by_label = {
        category: printed["behavior"][category]["label_counts"]
        for category in ("interruption", "backchannel")
    }
    assert by_label["backchannel"][labels["user_backchannel/1"]] == 1
    assert by_label["interruption"][labels["user_interruption/1"]] == 1
    assert printed["behavior"]["talking_to_other"]["scored"] == 0
    assert (
        score["behavior"]["asr"]["asr"]["model_sha256"]
        == transcripts["asr"]["model_sha256"]
    )
    source = score["samples"][0]["variants"]["overlap"]["timing"]["segment_source"][
        "output"
    ]
    assert source["kind"] == "supplied_segments"
    assert source["entry_source"]["kind"] == "silero_vad"
    assert (
        score["timing"]["categories"]
        == json.loads((first / "score.json").read_text())["timing"]["categories"]
    )


class FlakyWhisper(FakeWhisper):

    def transcribe(self, audio: np.ndarray, **options) -> dict:
        reply = super().transcribe(audio, **options)
        if len(self.calls) == 1:
            return {"text": " garbled", "language": "en"}
        elif len(self.calls) == 2:
            raise RuntimeError("CUDA error: device-side assert")
        return reply


def test_transcribe_keeps_going_after_per_variant_asr_failures(
    recorded: dict, fake_vad: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = recorded["run"]
    model_path = tmp_path / "tiny.pt"
    model_path.write_bytes(b"weights")
    monkeypatch.setitem(
        sys.modules,
        "whisper",
        types.SimpleNamespace(load_model=lambda name, device: FlakyWhisper()),
    )
    code, counts = run_cli(
        ["transcribe", "--run", str(run), "--output", str(tmp_path / "asr")]
        + ["--model-path", str(model_path), "--device", "cpu"]
    )

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

    run_cli(
        ["score", "--run", str(run), "--output", str(tmp_path / "s")]
        + ["--transcripts", str(tmp_path / "asr")]
    )
    behavior = json.loads((tmp_path / "s" / "score.json").read_text())["behavior"]
    reasons = {
        row["sample_id"]: row["unscored_reason"]
        for row in behavior["summary"]["samples"]
    }
    assert reasons["user_interruption/1"] == "missing_transcript:clean_output"
    assert reasons["user_backchannel/1"] == "missing_judgement"


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
    recorded: dict, fake_vad: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = recorded["run"]
    model_path = tmp_path / "tiny.pt"
    model_path.write_bytes(b"weights")
    monkeypatch.setitem(
        sys.modules,
        "whisper",
        types.SimpleNamespace(load_model=lambda name, device: BadTimesWhisper()),
    )
    code, counts = run_cli(
        ["transcribe", "--run", str(run), "--output", str(tmp_path / "asr")]
        + ["--model-path", str(model_path), "--device", "cpu"]
    )

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

    run_cli(
        ["score", "--run", str(run), "--output", str(tmp_path / "s")]
        + ["--transcripts", str(tmp_path / "asr")]
    )
    inputs = [
        json.loads(line)
        for line in (tmp_path / "s" / "judge-inputs.jsonl").read_text().splitlines()
    ]
    assert {item["status"] for item in inputs} == {"unscorable"}


def test_media_timeline_is_distinct_and_rejects_mismatched_transcripts(
    recorded: dict, fake_vad: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = recorded["run"]
    run_cli(
        [
            "score",
            "--run",
            str(run),
            "--output",
            str(tmp_path / "media"),
            "--timeline",
            "media",
        ]
    )
    score = json.loads((tmp_path / "media" / "score.json").read_text())
    assert score["timeline"]["name"] == "media"
    record = score["samples"][0]["variants"]["overlap"]["timing"]
    assert record["timeline"] == "media"
    assert record["segment_source"]["output"]["audio"].endswith(
        "/overlap/output-media.wav"
    )

    model_path = tmp_path / "tiny.pt"
    model_path.write_bytes(b"weights")
    monkeypatch.setitem(
        sys.modules,
        "whisper",
        types.SimpleNamespace(load_model=lambda name, device: FakeWhisper()),
    )
    run_cli(
        ["transcribe", "--run", str(run), "--output", str(tmp_path / "asr")]
        + ["--model-path", str(model_path), "--device", "cpu"]
    )
    with pytest.raises(ValueError, match="timeline"):
        main(
            [
                "score",
                "--run",
                str(run),
                "--output",
                str(tmp_path / "bad"),
                "--timeline",
                "media",
            ]
            + ["--transcripts", str(tmp_path / "asr")]
        )


def test_api_judge_runs_only_with_all_flags(
    recorded: dict, fake_vad: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = recorded["run"]
    with pytest.raises(SystemExit) as error:
        main(
            [
                "score",
                "--run",
                str(run),
                "--output",
                str(tmp_path / "x"),
                "--judge-model",
                "m",
            ]
        )
    assert error.value.code == 2 and not (tmp_path / "x").exists()

    model_path = tmp_path / "tiny.pt"
    model_path.write_bytes(b"weights")
    monkeypatch.setitem(
        sys.modules,
        "whisper",
        types.SimpleNamespace(load_model=lambda name, device: FakeWhisper()),
    )
    run_cli(
        ["transcribe", "--run", str(run), "--output", str(tmp_path / "asr")]
        + ["--model-path", str(model_path), "--device", "cpu"]
    )
    requests = []

    def reply(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        content = json.dumps(
            {"label": "C_UNKNOWN", "evidence": "okay", "first_new_segment": None}
        )
        return httpx.Response(
            200, json={"model": "m", "choices": [{"message": {"content": content}}]}
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda: real_client(transport=httpx.MockTransport(reply))
    )
    monkeypatch.setenv("JUDGE_KEY", "secret")
    code, printed = run_cli(
        ["score", "--run", str(run), "--output", str(tmp_path / "judged")]
        + [
            "--transcripts",
            str(tmp_path / "asr"),
            "--judge-base-url",
            "http://judge/v1",
        ]
        + ["--judge-model", "m", "--judge-api-key-env", "JUDGE_KEY"]
    )

    assert code == 0 and len(requests) == 2
    assert requests[0].headers["authorization"] == "Bearer secret"
    assert printed["behavior"]["backchannel"]["label_counts"]["C_UNKNOWN"] == 1
    lines = (tmp_path / "judged" / "judgements.jsonl").read_text().splitlines()
    assert {json.loads(line)["provenance"]["model"] for line in lines} == {"m"}
    assert "secret" not in (tmp_path / "judged" / "score.json").read_text()


def test_outputs_never_overwrite_or_live_inside_the_run(
    recorded: dict, fake_vad: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = recorded["run"]
    with pytest.raises(ValueError, match="outside the run"):
        main(["score", "--run", str(run), "--output", str(run / "score")])
    (tmp_path / "taken").mkdir()
    with pytest.raises(FileExistsError):
        main(["score", "--run", str(run), "--output", str(tmp_path / "taken")])

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
