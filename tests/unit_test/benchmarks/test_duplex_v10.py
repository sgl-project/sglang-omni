# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import io
import json
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile
from pydantic import JsonValue

from benchmarks.duplex.v10_dataset import (
    DECLARED_COUNTS,
    SUBSETS,
    discover_samples,
    inventory,
)
from benchmarks.eval.benchmark_duplex_v10 import main
from tests.unit_test.benchmarks.test_duplex_v15_cli import peer_server
from tests.unit_test.benchmarks.test_duplex_v15_runner import (
    FIXTURE_SAMPLES,
    SERVER_REVISION,
)

ANNOTATIONS = {
    "synthetic_pause_handling": ("pause.json", [{"timestamp": [0.1, 0.2]}]),
    "candor_pause_handling": (
        "pause.json",
        [{"timestamp": [0.1, 0.2]}, {"timestamp": [0.3, 0.4]}],
    ),
    "candor_turn_taking": ("turn_taking.json", [{"timestamp": [0.3, 0.3]}]),
    "synthetic_user_interruption": (
        "interrupt.json",
        [{"timestamp": [0.1, 0.3], "context": "tell me", "interrupt": "stop"}],
    ),
}


def write_sample(
    root: Path, subset: str, name: str = "1", annotation: str | None = None
) -> Path:
    sample_dir = root / subset / name
    sample_dir.mkdir(parents=True)
    soundfile.write(
        str(sample_dir / "input.wav"), FIXTURE_SAMPLES, 16000, subtype="PCM_16"
    )
    if subset in ANNOTATIONS:
        filename, entries = ANNOTATIONS[subset]
        (sample_dir / filename).write_text(
            annotation if annotation is not None else json.dumps(entries)
        )
    return sample_dir


def write_dataset(root: Path) -> None:
    for subset in SUBSETS:
        write_sample(root, subset)


def test_discovery_groups_subsets_by_task(tmp_path: Path) -> None:
    write_dataset(tmp_path)
    (tmp_path / "notes.md").write_text("")

    samples = discover_samples(tmp_path)

    assert [sample.id for sample in samples] == [f"{s}/1" for s in SUBSETS]
    assert [sample.task for sample in samples] == [
        "pause_handling",
        "pause_handling",
        "turn_taking",
        "user_interruption",
        "backchannel",
    ]
    assert all(not sample.errors for sample in samples)
    by_id = {sample.id: sample for sample in samples}
    assert by_id["candor_pause_handling/1"].events == [[0.1, 0.2], [0.3, 0.4]]
    assert by_id["synthetic_user_interruption/1"].texts == {
        "context": "tell me",
        "interrupt": "stop",
    }
    assert "annotation" not in by_id["icc_backchannel/1"].paths
    assert inventory(tmp_path) == {
        "declared": DECLARED_COUNTS,
        "observed": dict.fromkeys(SUBSETS, 1),
        "ignored": ["notes.md"],
    }


@pytest.mark.parametrize(
    ("subset", "annotation", "error"),
    [
        ("candor_turn_taking", "{", "turn_taking.json is not valid JSON"),
        ("candor_turn_taking", "[]", "turn_taking.json must be a nonempty list"),
        (
            "candor_turn_taking",
            json.dumps([{"timestamp": [0.1, 0.1]}, {"timestamp": [0.2, 0.2]}]),
            "turn_taking.json must hold exactly one event",
        ),
        (
            "synthetic_pause_handling",
            json.dumps([{"timestamp": [0.2, 0.1]}]),
            "pause.json[0] lacks a finite timestamp",
        ),
        (
            "synthetic_pause_handling",
            json.dumps([{"timestamp": [0.1, 9.0]}]),
            "pause.json[0] ends at 9.0 past input.wav",
        ),
        (
            "synthetic_user_interruption",
            json.dumps([{"timestamp": [0.1, 0.3], "context": "tell me"}]),
            "interrupt.json lacks interrupt text",
        ),
    ],
)
def test_malformed_annotations_stay_with_their_sample(
    tmp_path: Path, subset: str, annotation: str, error: str
) -> None:
    write_sample(tmp_path, subset, annotation=annotation)

    [sample] = discover_samples(tmp_path)

    assert any(message.startswith(error) for message in sample.errors), sample.errors


def test_missing_files_and_selection_errors(tmp_path: Path) -> None:
    write_dataset(tmp_path)
    (tmp_path / "candor_turn_taking" / "1" / "turn_taking.json").unlink()
    (tmp_path / "icc_backchannel" / "1" / "input.wav").unlink()

    samples = {sample.id: sample for sample in discover_samples(tmp_path)}

    assert samples["candor_turn_taking/1"].errors == ["missing turn_taking.json"]
    assert samples["icc_backchannel/1"].errors == ["missing input.wav"]
    assert [s.id for s in discover_samples(tmp_path, ["icc_backchannel/1"])] == [
        "icc_backchannel/1"
    ]
    with pytest.raises(ValueError, match="duplicate"):
        discover_samples(tmp_path, ["icc_backchannel/1", "icc_backchannel/1"])
    with pytest.raises(ValueError, match="unknown"):
        discover_samples(tmp_path, ["icc_backchannel/9"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        discover_samples(tmp_path, ["icc_backchannel/1"], 1)
    with pytest.raises(ValueError, match="positive"):
        discover_samples(tmp_path, max_per_subset=0)


def run_cli(argv: list[str]) -> tuple[int, dict[str, JsonValue]]:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        code = main(argv)
    return code, json.loads(stdout.getvalue())


class FakeNemoParakeet:
    """Stands in for nemo ASRModel: two fixed words, and the 16 kHz WAV it was fed."""

    def __init__(self) -> None:
        self.restored: list[tuple[str, str]] = []
        self.sample_rates: list[int] = []

    def eval(self) -> None:
        pass

    def transcribe(self, audio: list[str], timestamps: bool) -> list:
        assert timestamps is True and len(audio) == 1
        self.sample_rates.append(soundfile.info(audio[0]).samplerate)
        words = [
            {"word": "hello", "start": 0.1, "end": 0.3},
            {"word": "there", "start": 0.3, "end": 0.5},
        ]
        return [types.SimpleNamespace(timestamp={"word": words})]


def test_parakeet_recognizer_feeds_a_16k_wav_and_returns_whisper_shape() -> None:
    from benchmarks.duplex.v15_transcribe import ParakeetRecognizer, normalize_words

    parakeet = FakeNemoParakeet()
    raw = ParakeetRecognizer(parakeet).transcribe(
        np.zeros(16000, dtype=np.float32), timestamps=True
    )
    assert parakeet.sample_rates == [16000]
    assert normalize_words(raw, 1.0) == [
        {"text": "hello", "timestamp": [0.1, 0.3]},
        {"text": "there", "timestamp": [0.3, 0.5]},
    ]


def test_record_every_task(tmp_path: Path) -> None:
    dataset, run = tmp_path / "data", tmp_path / "run"
    write_dataset(dataset)
    (dataset / "candor_turn_taking" / "1" / "turn_taking.json").write_text("{")
    with peer_server() as url:
        code, summary = run_cli(
            ["record", "--dataset-root", str(dataset), "--url", url]
            + ["--output", str(run), "--server-revision", SERVER_REVISION]
            + ["--model", "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B"]
            + ["--dataset-revision", "fixture", "--timeout", "5", "--concurrency", "2"]
        )
    assert code == 1, summary
    assert summary["selected_variants"] == 5
    assert summary["load"]["concurrency"] == 2
    assert summary["variant_status"] == {"invalid": 1, "pass": 4}
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["kind"] == "full-duplex-bench-v1.0"
    assert manifest["server"]["model"] == "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B"
