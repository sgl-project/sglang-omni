# SPDX-License-Identifier: Apache-2.0
"""Offline audio signal diagnostics and CLI contracts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import soundfile
from numpy.typing import NDArray

from benchmarks.metrics.audio_signal import analyze_audio_file, compute_signal_metrics

REPO_ROOT = Path(__file__).resolve().parents[3]


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "benchmarks.eval.benchmark_audio_signal", *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_channel_metrics_preserve_amplitude_and_channels() -> None:
    positive_full_scale = 1.0 - 2.0**-15
    samples = np.array(
        [[-1.0, 0.2], [positive_full_scale, -0.4], [0.5, 0.6], [-0.25, -0.8]]
    )
    original = samples.copy()

    metrics = compute_signal_metrics(
        samples, 1000, positive_full_scale=positive_full_scale
    )

    assert metrics.sample_rate_hz == 1000
    assert metrics.frame_count == 4
    assert metrics.channel_count == 2
    assert metrics.duration_s == pytest.approx(0.004)
    assert metrics.positive_full_scale == positive_full_scale
    assert metrics.channels[0].sample_peak == 1.0
    assert metrics.channels[0].dc_offset == pytest.approx(
        (positive_full_scale - 0.75) / 4
    )
    assert metrics.channels[0].full_scale_sample_count == 2
    assert metrics.channels[1].sample_peak == 0.8
    assert metrics.channels[1].dc_offset == pytest.approx(-0.1)
    assert metrics.channels[1].full_scale_sample_count == 0
    np.testing.assert_array_equal(samples, original)


def test_silence_regions_include_partial_tail_and_merge_adjacent_windows() -> None:
    samples = np.zeros((11, 2))
    samples[2:4] = 0.01
    samples[8:10] = 0.01

    metrics = compute_signal_metrics(samples, 1000, silence_window_ms=2.0)

    assert metrics.silence_window_frames == 2
    assert metrics.window_count == 6
    assert metrics.silent_window_count == 4
    assert metrics.silent_window_ratio == pytest.approx(4 / 6)
    assert metrics.silent_duration_s == pytest.approx(0.007)
    assert [
        (region.start_frame, region.end_frame) for region in metrics.silence_regions
    ] == [(0, 2), (4, 8), (10, 11)]
    assert [region.start_s for region in metrics.silence_regions] == pytest.approx(
        [0.0, 0.004, 0.010]
    )
    assert [region.end_s for region in metrics.silence_regions] == pytest.approx(
        [0.002, 0.008, 0.011]
    )


@pytest.mark.parametrize("channel_values", [(0.25, -0.25), (0.0, 0.25)])
def test_silence_requires_every_channel_to_be_quiet(
    channel_values: tuple[float, float],
) -> None:
    samples = np.tile(channel_values, (20, 1))

    metrics = compute_signal_metrics(samples, 1000)

    assert metrics.silent_window_count == 0
    assert metrics.silent_window_ratio == 0.0
    assert metrics.silent_duration_s == 0.0
    assert metrics.silence_regions == []


def test_silence_uses_rms_and_actual_partial_window_length() -> None:
    samples = np.array([[0.0], [0.02], [0.02]])
    threshold_dbfs = float(20.0 * np.log10(0.015))

    metrics = compute_signal_metrics(
        samples,
        1000,
        silence_window_ms=2.0,
        silence_threshold_dbfs=threshold_dbfs,
    )

    assert metrics.window_count == 2
    assert metrics.silent_window_count == 1
    assert metrics.silent_duration_s == pytest.approx(0.002)
    assert metrics.silence_regions[0].end_frame == 2


def test_silence_threshold_is_inclusive() -> None:
    metrics = compute_signal_metrics(np.ones((2, 1)), 1000, silence_threshold_dbfs=0.0)

    assert metrics.silent_window_count == 1
    assert metrics.silent_duration_s == pytest.approx(0.002)
    assert metrics.channels[0].full_scale_sample_count == 2


@pytest.mark.parametrize(
    ("window_ms", "expected_frames"), [(0.1, 1), (2.4, 2), (2.6, 3)]
)
def test_silence_windows_round_to_at_least_one_frame(
    window_ms: float, expected_frames: int
) -> None:
    metrics = compute_signal_metrics(
        np.zeros((7, 1)), 1000, silence_window_ms=window_ms
    )

    assert metrics.silence_window_frames == expected_frames
    assert metrics.silent_duration_s == pytest.approx(0.007)
    assert len(metrics.silence_regions) == 1
    assert metrics.silence_regions[0].start_frame == 0
    assert metrics.silence_regions[0].end_frame == 7


@pytest.mark.parametrize(
    "samples",
    [
        np.zeros(3),
        np.zeros((1, 1, 1)),
        np.zeros((0, 1)),
        np.zeros((2, 0)),
        np.array([[np.nan]]),
        np.array([[np.inf]]),
        np.array([[-np.inf]]),
    ],
)
def test_invalid_samples_are_rejected(samples: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError):
        compute_signal_metrics(samples, 16000)


@pytest.mark.parametrize("sample_rate_hz", [0, -1])
def test_invalid_sample_rate_is_rejected(sample_rate_hz: int) -> None:
    with pytest.raises(ValueError):
        compute_signal_metrics(np.zeros((2, 1)), sample_rate_hz)


@pytest.mark.parametrize("window_ms", [0.0, -1.0, np.nan, np.inf])
def test_invalid_silence_window_is_rejected(window_ms: float) -> None:
    with pytest.raises(ValueError):
        compute_signal_metrics(np.zeros((2, 1)), 16000, silence_window_ms=window_ms)


@pytest.mark.parametrize("threshold_dbfs", [1.0, np.nan, np.inf, -np.inf])
def test_invalid_silence_threshold_is_rejected(threshold_dbfs: float) -> None:
    with pytest.raises(ValueError):
        compute_signal_metrics(
            np.zeros((2, 1)), 16000, silence_threshold_dbfs=threshold_dbfs
        )


@pytest.mark.parametrize("positive_full_scale", [0.0, -0.5, 1.1, np.nan, np.inf])
def test_invalid_positive_full_scale_is_rejected(positive_full_scale: float) -> None:
    with pytest.raises(ValueError):
        compute_signal_metrics(
            np.zeros((2, 1)), 16000, positive_full_scale=positive_full_scale
        )


@pytest.mark.parametrize(
    ("subtype", "bits"),
    [("PCM_U8", 8), ("PCM_16", 16), ("PCM_24", 24), ("PCM_32", 32)],
)
def test_pcm_wav_counts_both_integer_rails(
    tmp_path: Path, subtype: str, bits: int
) -> None:
    sample_step = 2.0 ** (1 - bits)
    positive_full_scale = 1.0 - sample_step
    samples = np.array(
        [
            -1.0,
            positive_full_scale,
            -1.0 + sample_step,
            positive_full_scale - sample_step,
            0.0,
        ]
    )
    audio_path = tmp_path / f"{subtype}.wav"
    soundfile.write(audio_path, samples, 22050, subtype=subtype)

    report = analyze_audio_file(audio_path)

    assert report.path == str(audio_path)
    assert report.format == "WAV"
    assert report.subtype == subtype
    assert report.metrics.sample_rate_hz == 22050
    assert report.metrics.frame_count == 5
    assert report.metrics.channel_count == 1
    assert report.metrics.duration_s == pytest.approx(5 / 22050)
    assert report.metrics.positive_full_scale == positive_full_scale
    assert report.metrics.channels[0].full_scale_sample_count == 2
    assert report.metrics.channels[0].sample_peak == 1.0
    assert report.metrics.channels[0].dc_offset == pytest.approx(
        -2.0 * sample_step / 5, abs=1e-12
    )


@pytest.mark.parametrize("subtype", ["FLOAT", "DOUBLE"])
def test_float_wav_preserves_samples_beyond_full_scale(
    tmp_path: Path, subtype: str
) -> None:
    audio_path = tmp_path / "overrange.wav"
    soundfile.write(audio_path, [-1.25, 1.125, 0.5], 8000, subtype=subtype)

    report = analyze_audio_file(audio_path)

    assert report.subtype == subtype
    assert report.metrics.positive_full_scale == 1.0
    assert report.metrics.channels[0].sample_peak == 1.25
    assert report.metrics.channels[0].dc_offset == pytest.approx(0.125)
    assert report.metrics.channels[0].full_scale_sample_count == 2


@pytest.mark.parametrize(
    ("audio_format", "subtype"), [("FLAC", "PCM_16"), ("WAV", "ULAW")]
)
def test_unsupported_containers_and_compressed_wav_are_rejected(
    tmp_path: Path, audio_format: str, subtype: str
) -> None:
    audio_path = tmp_path / "unsupported.wav"
    soundfile.write(audio_path, np.zeros(8), 8000, format=audio_format, subtype=subtype)

    with pytest.raises(ValueError):
        analyze_audio_file(audio_path)


@pytest.mark.parametrize("samples", [np.empty(0), np.array([np.nan, np.inf])])
def test_empty_or_nonfinite_float_wav_is_rejected(
    tmp_path: Path, samples: NDArray[np.float64]
) -> None:
    audio_path = tmp_path / "invalid.wav"
    soundfile.write(audio_path, samples, 8000, subtype="DOUBLE")

    with pytest.raises(ValueError):
        analyze_audio_file(audio_path)


def test_cli_recurses_deduplicates_and_keeps_diagnostics_informational(
    tmp_path: Path,
) -> None:
    nested_path = tmp_path / "nested"
    nested_path.mkdir()
    silent_path = tmp_path / "silent.wav"
    clipped_path = nested_path / "clipped.WAV"
    soundfile.write(silent_path, np.zeros(100), 1000)
    soundfile.write(clipped_path, [-1.0, 1.0], 1000, format="WAV", subtype="FLOAT")
    (tmp_path / "ignored.txt").write_text("not audio", encoding="utf-8")
    output_path = tmp_path / "report.json"

    completed = run_cli(
        str(tmp_path),
        str(clipped_path),
        "--output",
        str(output_path),
        "--silence-window-ms",
        "10",
        "--silence-threshold-dbfs",
        "-40",
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["config"] == {
        "silence_window_ms": 10.0,
        "silence_threshold_dbfs": -40.0,
    }
    assert report["errors"] == []
    assert [entry["path"] for entry in report["per_file"]] == sorted(
        [str(silent_path), str(clipped_path)]
    )
    metrics_by_path = {entry["path"]: entry["metrics"] for entry in report["per_file"]}
    assert metrics_by_path[str(silent_path)]["silent_window_ratio"] == 1.0
    assert (
        metrics_by_path[str(clipped_path)]["channels"][0]["full_scale_sample_count"]
        == 2
    )
    json.dumps(report, allow_nan=False)


def test_cli_retains_successes_and_reports_missing_and_invalid_files(
    tmp_path: Path,
) -> None:
    valid_path = tmp_path / "valid.wav"
    invalid_path = tmp_path / "invalid.wav"
    missing_path = tmp_path / "missing.wav"
    soundfile.write(valid_path, np.zeros(8), 8000)
    invalid_path.write_text("not a WAV file", encoding="utf-8")

    completed = run_cli(str(valid_path), str(invalid_path), str(missing_path))

    assert completed.returncode == 1, completed.stderr
    report = json.loads(completed.stdout)
    assert [entry["path"] for entry in report["per_file"]] == [str(valid_path)]
    assert {entry["path"] for entry in report["errors"]} == {
        str(invalid_path),
        str(missing_path),
    }
    assert all(entry["error"] for entry in report["errors"])
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--silence-window-ms", "0"),
        ("--silence-window-ms", "nan"),
        ("--silence-threshold-dbfs", "1"),
        ("--silence-threshold-dbfs", "inf"),
    ],
)
def test_cli_rejects_invalid_global_options(
    tmp_path: Path, option: str, value: str
) -> None:
    audio_path = tmp_path / "valid.wav"
    soundfile.write(audio_path, np.zeros(8), 8000)

    completed = run_cli(str(audio_path), option, value)

    assert completed.returncode == 2
    assert completed.stderr


def test_cli_rejects_directories_without_wav_files(tmp_path: Path) -> None:
    completed = run_cli(str(tmp_path))

    assert completed.returncode == 2
    assert completed.stderr


@pytest.mark.parametrize("alias_kind", ["same", "symlink", "hardlink"])
def test_cli_cannot_overwrite_input_audio(
    tmp_path: Path, alias_kind: Literal["same", "symlink", "hardlink"]
) -> None:
    audio_path = tmp_path / "original.wav"
    soundfile.write(audio_path, np.zeros(8), 8000)
    original_bytes = audio_path.read_bytes()
    if alias_kind == "same":
        output_path = audio_path
    elif alias_kind == "symlink":
        output_path = tmp_path / "symlink.json"
        output_path.symlink_to(audio_path)
    else:
        output_path = tmp_path / "hardlink.json"
        output_path.hardlink_to(audio_path)

    completed = run_cli(str(audio_path), "--output", str(output_path))

    assert completed.returncode == 2
    assert completed.stderr
    assert audio_path.read_bytes() == original_bytes


def test_offline_modules_do_not_import_model_or_server_dependencies() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import benchmarks.eval.benchmark_audio_signal; "
            "assert not any(name.split('.')[0] in "
            "{'torch', 'torchaudio', 'sglang', 'sglang_omni', 'transformers', 'fastapi'} "
            "for name in sys.modules)",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
