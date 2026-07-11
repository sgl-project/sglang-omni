# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from benchmarks.benchmarker.data import RequestResult
from benchmarks.metrics.transcribe_diarize_metrics import (
    DiarizationRow,
    _levenshtein_distance,
    clean_no_speaker,
    compute_diarization_metrics,
    split_clean_by_speaker,
)

BENCHMARK_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "benchmarks/eval/benchmark_asr_transcribe_diarize.py"
)


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_asr_transcribe_diarize_entry",
        BENCHMARK_SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_args_defaults_to_movies800times_preset() -> None:
    module = _load_benchmark_module()

    args = module.parse_args([])

    assert args.dataset == "movies800times"
    assert args.repo_id == module.MOVIES800_REPO_ID
    assert args.max_samples is None
    assert args.max_new_tokens == module.DEFAULT_MAX_NEW_TOKENS
    assert args.output_dir == module.MOVIES800TIMES_OUTPUT_DIR


def test_parse_args_uses_aishell4_long_preset() -> None:
    module = _load_benchmark_module()

    args = module.parse_args(["--dataset", "aishell4_long"])

    assert args.dataset == "aishell4_long"
    assert args.repo_id == module.AISHELL4_REPO_ID
    assert args.max_samples is None
    assert args.max_new_tokens == module.DEFAULT_MAX_NEW_TOKENS
    assert args.output_dir == module.AISHELL4_LONG_OUTPUT_DIR


@pytest.mark.parametrize(
    ("dataset", "expected_sample_count"),
    [
        ("movies800times", 800),
        ("aishell4_long", 20),
    ],
)
def test_load_samples_uses_dataset_expected_sample_count(
    monkeypatch: pytest.MonkeyPatch,
    dataset: str,
    expected_sample_count: int,
) -> None:
    module = _load_benchmark_module()
    captured_kwargs: dict[str, object] = {}

    def fake_load_movies800_samples(**kwargs: object) -> list[object]:
        captured_kwargs.update(kwargs)
        return []

    monkeypatch.setattr(module, "load_movies800_samples", fake_load_movies800_samples)

    module._load_samples(module.parse_args(["--dataset", dataset]))

    assert captured_kwargs["max_samples"] is None
    assert captured_kwargs["expected_sample_count"] == expected_sample_count


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[S01]我笑了", "我笑了"),
        ("[S01]她喜欢音乐", "她喜欢音乐"),
        ("[S01]I love music", "ilovemusic"),
        ("[S01][笑声]你好", "你好"),
        ("[S01]<silence>Hello [music]", "hello"),
    ],
)
def test_clean_no_speaker_only_strips_marked_events(text: str, expected: str) -> None:
    assert clean_no_speaker(text) == expected


def test_split_clean_by_speaker_preserves_spoken_event_words() -> None:
    assert split_clean_by_speaker(
        "[S01]我笑了[S02]I love music", implicit_single_speaker=False
    ) == {
        "[S1]": "我笑了",
        "[S2]": "ilovemusic",
    }


@pytest.mark.parametrize(
    ("reference", "prediction", "expected"),
    [
        ("", "", 0),
        ("abc", "", 3),
        ("", "abc", 3),
        ("kitten", "sitting", 3),
        ("你好世界", "你号世", 2),
        ("[S1]abc", "abc[S2]", 7),
    ],
)
def test_levenshtein_distance_matches_expected(
    reference: str,
    prediction: str,
    expected: int,
) -> None:
    assert _levenshtein_distance(reference, prediction) == expected


def test_compute_diarization_metrics_includes_timestamp_der_for_exact_match() -> None:
    result = compute_diarization_metrics(
        [
            DiarizationRow(
                sample_id="sample-1",
                audio_path="/tmp/sample-1.wav",
                reference_text="[0.00][S01]hello[1.00][1.00][S02]world[2.00]",
                prediction_text="[0.00][S01]hello[1.00][1.00][S02]world[2.00]",
            )
        ]
    )

    assert result.metrics["speaker_timestamp_der"] == pytest.approx(0.0)
    assert result.metrics["speaker_timestamp_der_valid_samples"] == 1
    assert result.metrics["speaker_timestamp_der_skipped"] == 0
    assert result.samples[0].speaker_timestamp_der_valid is True
    assert result.samples[0].speaker_timestamp_der == pytest.approx(0.0)


def test_compute_diarization_metrics_marks_missing_timestamp_prediction_invalid() -> (
    None
):
    result = compute_diarization_metrics(
        [
            DiarizationRow(
                sample_id="sample-1",
                audio_path="/tmp/sample-1.wav",
                reference_text="[0.00][S01]hello[1.00]",
                prediction_text="[S01]hello",
            )
        ]
    )

    assert result.metrics["speaker_timestamp_der"] is None
    assert result.metrics["speaker_timestamp_der_valid_samples"] == 0
    assert result.metrics["speaker_timestamp_der_skipped_no_pred_segments"] == 1
    assert result.samples[0].speaker_timestamp_der_valid is False
    assert (
        result.samples[0].speaker_timestamp_der_invalid_reason
        == "no_pred_timestamped_speaker_segments"
    )


def test_build_metrics_section_prints_timestamp_metrics() -> None:
    module = _load_benchmark_module()

    section = module._build_metrics_section(
        "diarization_metrics_percent",
        {
            "speaker_timestamp_der": 12.3456,
            "speaker_timestamp_der_valid_samples": 7,
            "speaker_timestamp_der_skipped": 1,
        },
        (
            "speaker_timestamp_der",
            "speaker_timestamp_der_valid_samples",
            "speaker_timestamp_der_skipped",
        ),
    )

    assert "speaker_timestamp_der:" in section
    assert "12.35" in section
    assert "speaker_timestamp_der_valid_samples:" in section


def test_compute_diarization_metrics_partitions_cer_above_50_percent() -> None:
    result = compute_diarization_metrics(
        [
            DiarizationRow(
                sample_id="ok",
                audio_path="/tmp/ok.wav",
                reference_text="[S01]hello",
                prediction_text="[S01]hello",
            ),
            DiarizationRow(
                sample_id="bad",
                audio_path="/tmp/bad.wav",
                reference_text="[S01]abc",
                prediction_text="[S01]" + "d" * 100,
            ),
        ]
    )

    assert result.metrics["cer_no_spk"] is not None
    assert result.metrics["cer_no_spk_below_50_corpus"] == pytest.approx(0.0)
    assert result.metrics["n_above_50_pct_cer"] == 1
    assert result.metrics_percent["cer_no_spk_below_50_corpus"] == pytest.approx(0.0)
    assert result.metrics_percent["n_above_50_pct_cer"] == 1


def test_build_key_metrics_section_prints_partitioned_cer_metrics() -> None:
    module = _load_benchmark_module()

    section = module._build_key_metrics_section(
        {
            "cer_no_spk": 21.68,
            "cer_no_spk_below_50_corpus": 5.50,
            "n_above_50_pct_cer": 1,
            "cp_cer": 14.42,
            "delta_cer": 7.85,
            "speaker_timestamp_der": 23.97,
        }
    )

    assert "cer_no_spk_below_50_corpus:" in section
    assert "5.5" in section
    assert "n_above_50_pct_cer:" in section


def test_build_key_metrics_section_prints_selected_metrics() -> None:
    module = _load_benchmark_module()

    section = module._build_key_metrics_section(
        {
            "cer_no_spk": 6.57,
            "cp_cer": 14.42,
            "delta_cer": 7.85,
            "speaker_timestamp_der": 23.97,
        }
    )

    assert "key_metrics" in section
    assert "cer_no_spk:" in section
    assert "6.57" in section
    assert "cp_cer:" in section
    assert "14.42" in section
    assert "delta_cer:" in section
    assert "7.85" in section
    assert "speaker_timestamp_der:" in section
    assert "23.97" in section


def test_extract_prediction_text_prefers_top_level_text_for_timestamps() -> None:
    from benchmarks.tasks.transcribe_diarize import extract_prediction_text

    payload = {
        "text": "[0.00][S01]hello[1.00][1.00][S02]world[2.00]",
        "segments": [
            {"text": "[S01]hello"},
            {"text": "[S02]world"},
        ],
    }

    assert extract_prediction_text(payload) == payload["text"]


def test_eval_saves_and_loads_aishell4_long_raw_asr_results(
    tmp_path: Path,
) -> None:
    module = _load_benchmark_module()

    args = Namespace(
        dataset="aishell4_long",
        repo_id="zhaochenyang20/AISHELL4",
        split="validation",
        audio_column="audio",
        expected_column="transcription",
        model_path="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        base_url=None,
        host="127.0.0.1",
        port=8001,
        concurrency=16,
        warmup=0,
        request_rate=float("inf"),
        request_timeout_s=1800,
        max_new_tokens=65536,
        max_samples=None,
        output_dir=str(tmp_path),
        asr_results_file="raw.json",
        speed_results_file="speed.json",
    )
    samples = [
        module.Movies800Sample(
            sample_id="sample-1",
            audio_path="/tmp/sample-1.wav",
            expected_text="[0.00][S01]hello[1.00]",
        )
    ]
    outputs = [
        RequestResult(
            request_id="sample-1",
            text="[0.00][S01]hello[1.00]",
            is_success=True,
            latency_s=1.2,
            audio_duration_s=4.0,
            rtf=0.3,
        )
    ]

    path = module._save_asr_results(args, samples, outputs, wall_clock_s=1.3)
    loaded_samples, loaded_outputs, loaded_config = module._load_asr_results(path)

    assert loaded_config["request_rate"] == "inf"
    assert loaded_config["max_new_tokens"] == 65536
    assert loaded_config["timing_scope"] == "asr_requests_only"
    assert loaded_samples == samples
    assert loaded_outputs[0].request_id == "sample-1"
    assert loaded_outputs[0].text == "[0.00][S01]hello[1.00]"
    assert loaded_outputs[0].is_success is True


def test_eval_saves_speed_results_before_accuracy_metrics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_benchmark_module()

    args = Namespace(
        dataset="aishell4_long",
        repo_id="zhaochenyang20/AISHELL4",
        split="validation",
        audio_column="audio",
        expected_column="transcription",
        model_path="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        base_url=None,
        host="127.0.0.1",
        port=8001,
        concurrency=16,
        warmup=0,
        request_rate=float("inf"),
        request_timeout_s=1800,
        max_new_tokens=65536,
        max_samples=None,
        output_dir=str(tmp_path),
        asr_results_file="raw.json",
        speed_results_file="speed.json",
    )
    outputs = [
        RequestResult(
            request_id="sample-1",
            text="[0.00][S01]hello[1.00]",
            is_success=True,
            latency_s=1.2,
            audio_duration_s=4.0,
            rtf=0.3,
        )
    ]

    path = module._save_and_print_speed_results(args, outputs, wall_clock_s=2.0)
    speed_payload = json.loads(Path(path).read_text())
    printed = capsys.readouterr().out

    assert speed_payload["config"]["timing_scope"] == "asr_requests_only"
    assert speed_payload["config"]["wall_clock_s"] == 2.0
    assert speed_payload["speed"]["completed_requests"] == 1
    assert speed_payload["speed"]["throughput_qps"] == 0.5
    assert speed_payload["speed"]["rtf_mean"] == 0.3
    assert "ASR Speed Result" in printed
    assert "ASR requests only" in printed
