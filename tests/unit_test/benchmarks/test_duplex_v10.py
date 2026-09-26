# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import io
import json
import sys
import types
from pathlib import Path

import pytest
import soundfile

from benchmarks.duplex import v10_scoring, v15_scoring
from benchmarks.duplex.v10_dataset import (
    DECLARED_COUNTS,
    SUBSETS,
    discover_samples,
    inventory,
)
from benchmarks.eval.benchmark_duplex_v10 import main
from tests.unit_test.benchmarks.test_duplex_v15_cli import (
    FakeWhisper,
    nonzero_segments,
    peer_server,
)
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


def write_sample(root: Path, subset: str, name: str = "1", annotation=None) -> Path:
    sample_dir = root / subset / name
    sample_dir.mkdir(parents=True)
    soundfile.write(
        str(sample_dir / "input.wav"), FIXTURE_SAMPLES, 16000, subtype="PCM_16"
    )
    if subset in ANNOTATIONS:
        filename, entries = ANNOTATIONS[subset]
        (sample_dir / filename).write_text(
            annotation if isinstance(annotation, str) else json.dumps(entries)
        )
    return sample_dir


def write_dataset(root: Path) -> None:
    for subset in SUBSETS:
        write_sample(root, subset)


def chunk(start_s: float, end_s: float, text: str = "word") -> dict:
    return {"text": text, "timestamp": [start_s, end_s]}


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


@pytest.mark.parametrize(
    ("chunks", "takeover"),
    [
        ([], False),
        ([chunk(0.0, 0.9)] * 3, False),
        ([chunk(0.0, 0.2)] * 4, True),
        ([chunk(0.0, 0.1), chunk(0.9, 1.0)], True),
    ],
)
def test_takeover_rule_matches_upstream_thresholds(
    chunks: list[dict], takeover: bool
) -> None:
    assert v10_scoring.takes_turn(chunks) is takeover


def test_pause_handling_only_counts_words_inside_the_input() -> None:
    record = v10_scoring.score_pause_handling(
        sample_id="p/1",
        chunks=[chunk(0.1, 0.2), chunk(2.0, 2.1), chunk(2.2, 3.5)],
        input_duration_s=1.0,
    )

    assert record["num_words"] == 1
    assert record["takeover"] is False
    assert record["window_s"] == [0.0, 1.0]


def test_response_latency_and_unexercised_interruption() -> None:
    chunks = [chunk(0.2, 0.4), chunk(1.5, 1.8), chunk(1.9, 2.8)]

    turn = v10_scoring.score_response(
        sample_id="t/1", task="turn_taking", chunks=chunks, event_end_s=1.0
    )
    silent = v10_scoring.score_response(
        sample_id="i/1",
        task="user_interruption",
        chunks=chunks,
        event_end_s=1.0,
        speaking_at_onset=False,
    )
    short = v10_scoring.score_response(
        sample_id="t/2", task="turn_taking", chunks=chunks[:1], event_end_s=0.1
    )

    assert turn["status"] == "scored"
    assert turn["num_words"] == 2
    assert turn["latency_s"] == pytest.approx(0.5)
    assert silent["status"] == "not_exercised"
    assert short["takeover"] is False and short["latency_s"] is None


def test_backchannel_takeover_rate_and_timing() -> None:
    reference = [0.25, 0.25, 0.25, 0.25]
    common = {"sample_id": "b/1", "input_duration_s": 2.0, "reference": reference}

    record = v10_scoring.score_backchannel(
        chunks=[chunk(0.1, 0.3, "yeah")],
        output_segments=[[0.1, 0.4], [2.5, 2.9]],
        **common,
    )
    long = v10_scoring.score_backchannel(
        chunks=[],
        output_segments=[[0.0, 1.9], [0.0, 3.5]],
        **{**common, "input_duration_s": 4.0},
    )
    wordy = v10_scoring.score_backchannel(
        chunks=[chunk(0.1, 0.2)] * 4, output_segments=[[0.1, 0.5]], **common
    )
    drawn_out = v10_scoring.score_backchannel(
        chunks=[chunk(0.1, 1.4, "hmm")], output_segments=[[0.1, 1.5]], **common
    )
    silent = v10_scoring.score_backchannel(chunks=[], output_segments=[], **common)
    unreferenced = v10_scoring.score_backchannel(
        chunks=[], output_segments=[[0.1, 0.4]], **{**common, "reference": None}
    )

    assert record["backchannels"] == [[0.1, 0.4]]
    assert record["takeover"] is False
    assert record["backchannel_rate_per_s"] == 0.5
    assert 0 < record["timing_jsd"] < 1
    assert long["takeover"] is True and long["backchannels"] == [[0.0, 1.9]]
    assert wordy["takeover"] is True
    assert drawn_out["takeover"] is True and drawn_out["backchannels"] == [[0.1, 1.5]]
    assert silent["timing_jsd"] == 1.0 and silent["backchannels"] == []
    assert unreferenced["timing_jsd"] is None


def test_summary_keeps_selected_denominator_and_rejects_strays() -> None:
    selected = {"t/1": "turn_taking", "t/2": "turn_taking", "p/1": "pause_handling"}
    turn = v10_scoring.score_response(
        sample_id="t/1",
        task="turn_taking",
        chunks=[chunk(1.2, 1.4), chunk(1.5, 2.5)],
        event_end_s=1.0,
    )

    summary = v10_scoring.summarize([turn], selected)

    turn_summary = summary["tasks"]["turn_taking"]
    assert (turn_summary["selected"], turn_summary["missing"]) == (2, 1)
    assert turn_summary["status_counts"] == {"scored": 1}
    assert turn_summary["takeover_rate"] == 1.0
    assert turn_summary["latency_s"]["n"] == 1
    assert turn_summary["latency_s"]["mean"] == pytest.approx(0.2)
    assert summary["tasks"]["pause_handling"]["missing"] == 1
    assert summary["tasks"]["pause_handling"]["takeover_rate"] is None
    with pytest.raises(ValueError, match="duplicate"):
        v10_scoring.summarize([turn, turn], selected)
    with pytest.raises(ValueError, match="not a selected sample"):
        v10_scoring.summarize([turn], {"p/1": "pause_handling"})


def run_cli(argv: list[str]) -> tuple[int, dict]:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        code = main(argv)
    return code, json.loads(stdout.getvalue())


def test_record_transcribe_and_score_every_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, run = tmp_path / "data", tmp_path / "run"
    write_dataset(dataset)
    (dataset / "candor_turn_taking" / "1" / "turn_taking.json").write_text("{")
    with peer_server() as url:
        code, summary = run_cli(
            ["record", "--dataset-root", str(dataset), "--url", url]
            + ["--output", str(run), "--server-revision", SERVER_REVISION]
            + ["--model", "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B"]
            + ["--dataset-revision", "fixture", "--timeout", "5"]
        )
    assert code == 1, summary
    assert summary["selected_variants"] == 5
    assert summary["variant_status"] == {"invalid": 1, "pass": 4}
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["kind"] == "full-duplex-bench-v1.0"
    assert manifest["server"]["model"] == "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B"

    model_path = tmp_path / "tiny.pt"
    model_path.write_bytes(b"weights")
    monkeypatch.setitem(
        sys.modules,
        "whisper",
        types.SimpleNamespace(load_model=lambda name, device: FakeWhisper()),
    )
    code, counts = run_cli(
        ["transcribe", "--run", str(run), "--output", str(tmp_path / "asr")]
        + ["--model-path", str(model_path), "--device", "cpu"]
    )
    assert code == 0 and counts == {"not_qualified:invalid": 1, "transcribed": 4}

    monkeypatch.setattr(v15_scoring, "silero_speech_segments", nonzero_segments)
    reference = tmp_path / "icc_gt_distribution.json"
    reference.write_text(json.dumps({"1": [0.5, 0.5]}))
    code, printed = run_cli(
        ["score", "--run", str(run), "--output", str(tmp_path / "score")]
        + ["--transcripts", str(tmp_path / "asr")]
        + ["--backchannel-reference", str(reference)]
    )

    assert code == 0
    tasks = printed["tasks"]
    assert {task: tasks[task]["selected"] for task in tasks} == {
        "pause_handling": 2,
        "turn_taking": 1,
        "user_interruption": 1,
        "backchannel": 1,
    }
    assert tasks["pause_handling"]["scored"] == 2
    assert tasks["turn_taking"]["scored"] == 0
    assert tasks["backchannel"]["timing_jsd"]["n"] == 1
    score = json.loads((tmp_path / "score" / "score.json").read_text())
    reasons = {row["sample_id"]: row["unscored_reason"] for row in score["samples"]}
    assert reasons["candor_turn_taking/1"] == "invalid_sample"
    assert score["backchannel_reference"]["path"] == str(reference.resolve())
