# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile
from pydantic import JsonValue

from benchmarks.duplex import v10_scoring
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


def word_chunk(
    start_s: float, end_s: float, text: str = "word"
) -> dict[str, JsonValue]:
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
        ([word_chunk(0.0, 0.9)] * 3, False),
        ([word_chunk(0.0, 0.2)] * 4, True),
        ([word_chunk(0.0, 0.1), word_chunk(0.9, 1.0)], True),
    ],
)
def test_takeover_rule_matches_upstream_thresholds(
    chunks: list[dict[str, JsonValue]], takeover: bool
) -> None:
    assert v10_scoring.takes_turn(chunks) is takeover


def test_scoring_config_hash_covers_every_threshold() -> None:
    assert v10_scoring.SCORING_CONFIG["thresholds"] == {
        "takeover_max_duration_s": v10_scoring.TAKEOVER_MAX_DURATION_S,
        "takeover_max_words": v10_scoring.TAKEOVER_MAX_WORDS,
        "backchannel_max_words": v10_scoring.BACKCHANNEL_MAX_WORDS,
        "backchannel_window_s": v10_scoring.BACKCHANNEL_WINDOW_S,
        "backchannel_epsilon": v10_scoring.BACKCHANNEL_EPSILON,
        "censor_tolerance_s": v10_scoring.CENSOR_TOLERANCE_S,
    }
    assert v10_scoring.SCORING_CONFIG["vad"] == v10_scoring.SILERO_VAD_CONFIG


def test_pause_handling_only_counts_words_inside_the_input() -> None:
    record = v10_scoring.score_pause_handling(
        sample_id="p/1",
        chunks=[word_chunk(0.1, 0.2), word_chunk(2.0, 2.1), word_chunk(2.2, 3.5)],
        input_duration_s=1.0,
    )

    assert record["num_words"] == 1
    assert record["takeover"] is False
    assert record["window_s"] == [0.0, 1.0]


def test_response_latency_gates_and_window() -> None:
    chunks = [
        word_chunk(0.2, 0.4),
        word_chunk(1.5, 1.8),
        word_chunk(1.9, 2.8),
        word_chunk(9.5, 9.9),
    ]
    common = {"chunks": chunks, "input_duration_s": 8.0}

    turn = v10_scoring.score_turn_taking(
        sample_id="t/1",
        turn_end_s=1.0,
        output_segments=[[0.2, 0.4], [1.5, 2.8]],
        **common,
    )
    talked_over = v10_scoring.score_turn_taking(
        sample_id="t/2", turn_end_s=1.0, output_segments=[[0.2, 2.8]], **common
    )
    silent = v10_scoring.score_interruption(
        sample_id="i/1",
        interruption_start_s=0.5,
        interruption_end_s=1.0,
        output_segments=[[1.5, 2.8]],
        **common,
    )
    interrupted = v10_scoring.score_interruption(
        sample_id="i/2",
        interruption_start_s=0.3,
        interruption_end_s=1.0,
        output_segments=[[0.2, 0.4], [1.5, 7.99]],
        **common,
    )
    talked_through = v10_scoring.score_interruption(
        sample_id="i/3",
        interruption_start_s=0.3,
        interruption_end_s=1.0,
        output_segments=[[0.2, 2.8]],
        **common,
    )
    paused = v10_scoring.score_interruption(
        sample_id="i/4",
        interruption_start_s=0.3,
        interruption_end_s=1.0,
        output_segments=[[0.2, 0.6], [1.5, 2.8]],
        **common,
    )
    short = v10_scoring.score_turn_taking(
        sample_id="t/3",
        chunks=chunks[:1],
        turn_end_s=0.1,
        input_duration_s=8.0,
        output_segments=[],
    )

    assert turn["status"] == "scored"
    assert turn["num_words"] == 2
    assert turn["latency_s"] == pytest.approx(0.5)
    assert turn["right_censored"] is False
    assert talked_over["status"] == "spoke_before_turn_end"
    assert talked_over["speaking_at_event"] is True
    assert silent["status"] == "not_exercised"
    assert interrupted["status"] == "scored" and interrupted["right_censored"] is True
    assert talked_through["status"] == "talked_through"
    assert talked_through["takeover"] is True and talked_through["latency_s"] == 0.5
    assert paused["status"] == "scored"
    assert short["takeover"] is False and short["latency_s"] is None


def test_backchannel_takeover_rate_and_timing() -> None:
    reference = [0.25, 0.25, 0.25, 0.25]
    common = {"sample_id": "b/1", "input_duration_s": 2.0, "reference": reference}

    record = v10_scoring.score_backchannel(
        chunks=[word_chunk(0.1, 0.3, "yeah")],
        output_segments=[[0.1, 0.4], [2.5, 2.9]],
        **common,
    )
    long = v10_scoring.score_backchannel(
        chunks=[],
        output_segments=[[0.0, 1.9], [0.0, 3.5]],
        **{**common, "input_duration_s": 4.0},
    )
    wordy = v10_scoring.score_backchannel(
        chunks=[word_chunk(0.1, 0.2)] * 4, output_segments=[[0.1, 0.5]], **common
    )
    drawn_out = v10_scoring.score_backchannel(
        chunks=[word_chunk(0.1, 1.4, "hmm")], output_segments=[[0.1, 1.5]], **common
    )
    silent = v10_scoring.score_backchannel(chunks=[], output_segments=[], **common)
    unreferenced = v10_scoring.score_backchannel(
        chunks=[], output_segments=[[0.1, 0.4]], **{**common, "reference": None}
    )

    assert record["backchannels"] == [[0.1, 0.4]]
    assert record["takeover"] is False
    assert record["backchannel_rate_per_s"] == 0.5
    assert 0 < record["timing_jsd"] < 1
    assert long["takeover"] is True and long["backchannels"] == []
    assert wordy["takeover"] is True and wordy["backchannels"] == []
    assert drawn_out["takeover"] is True and drawn_out["backchannels"] == []
    assert silent["timing_jsd"] == 1.0 and silent["backchannels"] == []
    assert unreferenced["timing_jsd"] is None


def test_summary_keeps_selected_denominator_and_rejects_strays() -> None:
    selected = {"t/1": "turn_taking", "t/2": "turn_taking", "p/1": "pause_handling"}
    turn = v10_scoring.score_turn_taking(
        sample_id="t/1",
        chunks=[word_chunk(1.2, 1.4), word_chunk(1.5, 2.5)],
        turn_end_s=1.0,
        input_duration_s=8.0,
        output_segments=[[1.2, 2.5]],
    )

    summary = v10_scoring.summarize([turn], selected)

    turn_summary = summary["tasks"]["turn_taking"]
    assert (turn_summary["selected"], turn_summary["missing"]) == (2, 1)
    assert turn_summary["status_counts"] == {"scored": 1}
    assert turn_summary["right_censored"] == 0
    assert turn_summary["takeover_rate"] == 1.0
    assert turn_summary["latency_s"]["n"] == 1
    assert turn_summary["latency_s"]["mean"] == pytest.approx(0.2)
    assert summary["tasks"]["pause_handling"]["missing"] == 1
    assert summary["tasks"]["pause_handling"]["takeover_rate"] is None
    with pytest.raises(ValueError, match="duplicate"):
        v10_scoring.summarize([turn, turn], selected)
    with pytest.raises(ValueError, match="not a selected sample"):
        v10_scoring.summarize([turn], {"p/1": "pause_handling"})


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


def install_fake_nemo(
    monkeypatch: pytest.MonkeyPatch, parakeet: FakeNemoParakeet
) -> None:
    def restore_from(restore_path: str, map_location: object) -> FakeNemoParakeet:
        parakeet.restored.append((restore_path, str(map_location)))
        return parakeet

    models = types.ModuleType("nemo.collections.asr.models")
    models.ASRModel = types.SimpleNamespace(restore_from=restore_from)
    for name in ("nemo", "nemo.collections", "nemo.collections.asr"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "nemo.collections.asr.models", models)


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
        + ["--asr", "whisper", "--model-path", str(model_path), "--device", "cpu"]
    )
    assert code == 0 and counts == {"not_qualified:invalid": 1, "transcribed": 4}
    transcripts = json.loads((tmp_path / "asr" / "transcripts.json").read_text())
    assert transcripts["kind"] == "fdb-output-asr"
    assert transcripts["run"]["kind"] == "full-duplex-bench-v1.0"
    assert transcripts["asr"]["backend"] == "whisper"

    nemo_path = tmp_path / "parakeet.nemo"
    nemo_path.write_bytes(b"weights")
    parakeet = FakeNemoParakeet()
    install_fake_nemo(monkeypatch, parakeet)
    code, counts = run_cli(
        ["transcribe", "--run", str(run), "--output", str(tmp_path / "asr-parakeet")]
        + ["--model-path", str(nemo_path), "--device", "cpu"]
    )
    assert code == 0 and counts == {"not_qualified:invalid": 1, "transcribed": 4}
    assert parakeet.restored == [(str(nemo_path), "cpu")]
    parakeet_transcripts = json.loads(
        (tmp_path / "asr-parakeet" / "transcripts.json").read_text()
    )
    assert parakeet_transcripts["asr"]["backend"] == "parakeet"
    assert parakeet_transcripts["asr"]["options"] == {"timestamps": True}
    transcribed = [
        variant
        for variant in parakeet_transcripts["variants"]
        if variant["status"] == "transcribed"
    ]
    assert transcribed[0]["transcript"] == {
        "text": "hello there",
        "chunks": [
            {"text": "hello", "timestamp": [0.1, 0.3]},
            {"text": "there", "timestamp": [0.3, 0.5]},
        ],
    }

    loads = []
    monkeypatch.setattr(v10_scoring, "load_silero_model", lambda: loads.append(1))
    monkeypatch.setattr(
        v10_scoring,
        "silero_speech_segments",
        lambda path, vad_model: nonzero_segments(path),
    )
    reference = tmp_path / "icc_gt_distribution.json"
    reference.write_text(json.dumps({"1": [0.5, 0.5]}))
    code, printed = run_cli(
        ["score", "--run", str(run), "--output", str(tmp_path / "score")]
        + ["--transcripts", str(tmp_path / "asr")]
        + ["--backchannel-reference", str(reference)]
    )

    assert code == 0
    assert loads == [1]
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
    scorings = {row["sample_id"]: row["scoring"] for row in score["samples"]}
    assert "output_vad" not in scorings["synthetic_pause_handling/1"]
    output_vad = scorings["icc_backchannel/1"]["output_vad"]
    assert output_vad["audio"] == "samples/icc_backchannel/1/input/output-playout.wav"
    assert (
        output_vad["audio_sha256"]
        == hashlib.sha256((run / output_vad["audio"]).read_bytes()).hexdigest()
    )
    assert output_vad["vad"]["package"] == "fixture-nonzero"
    assert scorings["synthetic_user_interruption/1"]["output_vad"]["audio"] == (
        "samples/synthetic_user_interruption/1/input/output-playout.wav"
    )


@pytest.mark.parametrize(
    "segments",
    [
        [[2.0, 1.0]],
        [[1.0, 3.0], [2.0, 4.0]],
        [[float("nan"), 1.0]],
        [[1.0, 11.0]],
        [[-0.1, 1.0]],
    ],
)
def test_vad_segments_reject_invalid_boundaries(segments: list[list[float]]) -> None:
    with pytest.raises(ValueError):
        v10_scoring.validate_segments(segments, 10.0, "output")


def test_silero_path_records_provenance(tmp_path: Path) -> None:
    pytest.importorskip("silero_vad")
    path = tmp_path / "silence.wav"
    soundfile.write(path, np.zeros(22050, dtype=np.float32), 22050, subtype="PCM_16")
    vad_model = v10_scoring.load_silero_model()
    result = v10_scoring.silero_speech_segments(path, vad_model)
    assert result["segments"] == []
    assert result["duration_s"] == pytest.approx(1.0)
    assert result["vad"]["config"] == v10_scoring.SILERO_VAD_CONFIG
    float_path = tmp_path / "float.wav"
    soundfile.write(
        float_path, np.zeros(16000, dtype=np.float32), 16000, subtype="FLOAT"
    )
    with pytest.raises(ValueError):
        v10_scoring.silero_speech_segments(float_path, vad_model)
