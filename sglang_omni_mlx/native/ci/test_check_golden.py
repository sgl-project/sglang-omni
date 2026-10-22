# SPDX-License-Identifier: Apache-2.0
"""Unit tests for check_golden.py's per-chip gate (no runtime needed)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import auk_golden
import check_golden
import check_model_golden
import pytest

CLIP = {
    "text": "hello",
    "language": "English",
    "generated_token_count": 2,
    "finish_reason": "stop",
}
METRICS = {"wer_en": 0.03, "cer_zh": 0.06, "mer_mixed": 0.16}


def golden(clip_count: int = 10) -> dict:
    return {
        "model": "m",
        "baseline": METRICS,
        "tolerance": {"metric_pp": 1.5, "min_identical": 0.8},
        "devices": {
            "Apple M5": {
                "metrics": METRICS,
                "clips": {f"c{i}": CLIP for i in range(clip_count)},
            }
        },
    }


def results(changed: int, clip_count: int = 10) -> dict:
    return {
        f"c{i}": {**CLIP, "text": "other"} if i < changed else CLIP
        for i in range(clip_count)
    }


def test_a_recorded_chip_requires_every_clip() -> None:
    assert check_golden.check(golden(), "Apple M5", results(0), METRICS)[1] == []
    failures = check_golden.check(golden(), "Apple M5", results(1), METRICS)[1]
    assert failures == ["`c0` differs from its golden output"]


def test_another_chip_is_held_to_the_error_rates() -> None:
    assert check_golden.check(golden(), "Apple M4 Pro", results(2), METRICS)[1] == []
    worse = {**METRICS, "cer_zh": METRICS["cer_zh"] + 0.02}
    failures = check_golden.check(golden(), "Apple M4 Pro", results(2), worse)[1]
    assert failures == ["cer_zh is +2.00 pp from Apple M5's"]


def test_another_chip_needs_most_clips_identical() -> None:
    failures = check_golden.check(golden(), "Apple M4 Pro", results(3), METRICS)[1]
    assert failures == ["only 7/10 clips match Apple M5's outputs"]


def test_an_empty_golden_file_fails() -> None:
    assert check_golden.check(golden(0), "Apple M5", {}, METRICS)[1] == [
        "the golden file has no clips"
    ]


def test_moss_tags_are_dropped_before_scoring() -> None:
    text = "[0.00][S01] Hello [sniff] world.[1.25][1.30][S02]你好[2.00]"
    assert (
        check_golden.spoken_text(text, "moss_transcribe_diarize") == "Hello world. 你好"
    )
    assert check_golden.spoken_text("[S01] kept", None) == "[S01] kept"


def test_a_pending_baseline_reports_no_delta() -> None:
    pending = {**golden(), "baseline": {"source": "pending", **METRICS}}
    lines, failures = check_golden.check(pending, "Apple M5", results(0), METRICS)
    assert failures == []
    assert "| wer_en | pending | 3.00% | n/a |" in lines


@pytest.fixture
def auk_runtime(tmp_path: Path) -> Path:
    """An external synthesis boundary that writes real PCM WAV files."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    executable = runtime / "auk_synthesize"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import argparse, json, math, pathlib, sys, wave\n"
        "parser = argparse.ArgumentParser()\n"
        "for flag in ('model-path', 'requests', 'output-directory', 'dtype'):\n"
        "    parser.add_argument('--' + flag, required=True)\n"
        "arguments = parser.parse_args()\n"
        "if pathlib.Path(__file__).with_name('fail').exists():\n"
        "    sys.exit('synthetic native execution failure')\n"
        "output = pathlib.Path(arguments.output_directory)\n"
        "output.mkdir()\n"
        "for line in pathlib.Path(arguments.requests).read_text().splitlines():\n"
        "    request = json.loads(line)\n"
        "    if 'reference_wav' in request:\n"
        "        assert pathlib.Path(request['reference_wav']).name.endswith('.wav')\n"
        "    count = math.ceil(request['seconds'] * 50) * 480\n"
        "    filename = output / (request['id'] + '.wav')\n"
        "    with wave.open(str(filename), 'wb') as audio:\n"
        "        audio.setparams((1, 2, 24000, count, 'NONE', 'not compressed'))\n"
        "        audio.writeframes(b'\\x01\\x00' * count)\n"
        "    print(json.dumps(dict(id=request['id'], file=str(filename), "
        "sample_count=count, sample_rate=24000, elapsed_s=0.01)))\n"
    )
    executable.chmod(0o755)
    return runtime


@pytest.fixture
def auk_manifest() -> auk_golden.Golden:
    manifest: auk_golden.Golden = json.loads(
        (Path(__file__).parent / "model_golden" / "auk.json").read_text()
    )
    manifest["devices"] = {}
    return manifest


def test_auk_unknown_chip_exports_without_accepting_outputs(
    tmp_path: Path, auk_runtime: Path, auk_manifest: auk_golden.Golden
) -> None:
    output_directory = tmp_path / "outputs"
    lines, failures = auk_golden.check(
        auk_manifest, auk_runtime, tmp_path, "Unknown chip", output_directory, None
    )
    assert any("no AuK golden for this chip" in failure for failure in failures)
    assert auk_manifest["devices"] == {}
    exported = json.loads((output_directory / "auk.json").read_text())
    assert len(exported["clips"]) == 4
    assert exported["device"] == "Unknown chip"
    assert auk_golden.validate_outputs(auk_manifest, exported["clips"]) == []
    auk_manifest["devices"]["Unknown chip"] = {
        "metrics": {},
        "clips": exported["clips"],
    }
    assert (
        auk_golden.check(
            auk_manifest, auk_runtime, tmp_path, "Unknown chip", None, None
        )[1]
        == []
    )
    exported["clips"]["en_text"]["sha256"] = "f" * 64
    assert any(
        "differs from this chip" in failure
        for failure in auk_golden.check(
            auk_manifest, auk_runtime, tmp_path, "Unknown chip", None, None
        )[1]
    )


@pytest.mark.parametrize("corruption", [None, "missing-case", "sample-count"])
def test_auk_import_rejects_incomplete_or_invalid_waveforms(
    tmp_path: Path,
    auk_runtime: Path,
    auk_manifest: auk_golden.Golden,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str | None,
) -> None:
    output_directory = tmp_path / "outputs"
    auk_golden.check(
        auk_manifest, auk_runtime, tmp_path, "Test chip", output_directory, None
    )
    output_path = output_directory / "auk.json"
    outputs = json.loads(output_path.read_text())
    if corruption == "missing-case":
        outputs["clips"].pop("en_text")
    elif corruption == "sample-count":
        outputs["clips"]["en_text"]["sample_count"] = 1
    else:
        pass
    output_path.write_text(json.dumps(outputs))
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(auk_manifest))
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_golden", "--golden", str(golden_path), "--import", str(output_path)],
    )
    if corruption is None:
        check_golden.run(check_golden.transcribe_qwen3_asr)
        assert (
            json.loads(golden_path.read_text())["devices"]["Test chip"]["clips"]
            == outputs["clips"]
        )
    else:
        with pytest.raises(SystemExit):
            check_golden.run(check_golden.transcribe_qwen3_asr)
        assert json.loads(golden_path.read_text())["devices"] == {}


def test_auk_native_failure_and_invalid_requests_cannot_pass(
    tmp_path: Path, auk_runtime: Path, auk_manifest: auk_golden.Golden
) -> None:
    (auk_runtime / "fail").touch()
    lines, failures = auk_golden.check(
        auk_manifest, auk_runtime, tmp_path, "Test chip", tmp_path / "outputs", None
    )
    assert any("synthetic native execution failure" in failure for failure in failures)
    assert json.loads((tmp_path / "outputs" / "auk.json").read_text())["clips"] == {}
    auk_manifest["cases"][0]["seconds"] = 0
    with pytest.raises(ValueError, match="Invalid AuK golden request"):
        auk_golden.check(auk_manifest, auk_runtime, tmp_path, "Test chip", None, None)


@pytest.mark.parametrize("golden_filename", ["auk.json", "auk-flash.json"])
def test_auk_cli_dispatch_records_and_checks_waveforms(
    tmp_path: Path,
    auk_runtime: Path,
    monkeypatch: pytest.MonkeyPatch,
    golden_filename: str,
) -> None:
    """AuK golden requests must not enter the ASR transcription or scoring path."""
    manifest = json.loads(
        (Path(__file__).parent / "model_golden" / golden_filename).read_text()
    )
    manifest["devices"] = {}
    golden_path = tmp_path / golden_filename
    golden_path.write_text(json.dumps(manifest))
    output_directory = tmp_path / "outputs"
    arguments = [
        "check_model_golden",
        "--runtime-bin",
        str(auk_runtime),
        "--data-root",
        str(tmp_path),
        "--golden",
        str(golden_path),
        "--outputs",
        str(output_directory),
    ]
    monkeypatch.setattr(check_golden, "chip", lambda: "Test chip")
    monkeypatch.setattr(sys, "argv", arguments + ["--write"])
    with pytest.raises(SystemExit) as recorded:
        check_golden.run(check_model_golden.transcribe_with_tool)
    assert recorded.value.code == 0
    exported = json.loads((output_directory / golden_filename).read_text())
    written = json.loads(golden_path.read_text())
    assert written["devices"]["Test chip"]["clips"] == exported["clips"]
    assert set(exported["clips"]) == {case["id"] for case in manifest["cases"]}
    assert exported["metrics"] == {}

    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit) as identical:
        check_golden.run(check_model_golden.transcribe_with_tool)
    assert identical.value.code == 0
    written["devices"]["Test chip"]["clips"]["en_text"]["sha256"] = "f" * 64
    golden_path.write_text(json.dumps(written))
    with pytest.raises(SystemExit) as changed:
        check_golden.run(check_model_golden.transcribe_with_tool)
    assert changed.value.code == 1
