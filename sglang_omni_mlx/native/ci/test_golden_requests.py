# SPDX-License-Identifier: Apache-2.0
"""Request selection and corpus coverage in the native accuracy CLI."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import check_golden
import pytest

TRANSCRIPT = {
    "text": "你好 world",
    "language": "Chinese",
    "generated_token_count": 4,
    "finish_reason": "stop",
}


@pytest.fixture
def fixed_golden_path(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "manifest.jsonl").write_text(
        json.dumps(
            {
                "id": "mixed",
                "duration": 1,
                "lang": "mixed",
                "stratum": "mixed",
                "reference": TRANSCRIPT["text"],
            }
        )
        + "\n"
    )
    path = tmp_path / "fixed.json"
    path.write_text(
        json.dumps(
            {
                "model": "model",
                "request": {
                    "layout": "voxt_swift",
                    "language": "English",
                    "max_new_tokens": 1024,
                    "stop_at_end_of_text": True,
                    "stop_on_token_loop": True,
                },
                "baseline": {"source": "pending"},
                "tolerance": {"metric_pp": 1.5, "min_identical": 0.8},
                "devices": {
                    "Apple M5": {
                        "metrics": {"wer_en": 0, "cer_zh": 0, "mer_mixed": 1},
                        "clips": {"mixed": {**TRANSCRIPT, "text": "Hello world"}},
                    }
                },
            }
        )
    )
    return path


@pytest.mark.parametrize("language", ["English", None])
def test_qwen3_asr_accepts_a_request_without_a_language_hint(
    tmp_path: Path, fixed_golden_path: Path, language: str | None
) -> None:
    """A golden request without language previously failed before transcription."""
    request = json.loads(fixed_golden_path.read_text())["request"]
    if language is None:
        request.pop("language")
    else:
        request["language"] = language
    clip = tmp_path / "mixed.wav"
    completed = subprocess.CompletedProcess(
        [], 0, stdout=json.dumps({"file": str(clip), **TRANSCRIPT}) + "\n"
    )
    with patch.object(
        check_golden.subprocess, "run", return_value=completed
    ) as runtime:
        outputs = check_golden.transcribe(tmp_path, tmp_path, [clip], request, None)
    command = runtime.call_args.args[0]
    if language is None:
        assert "--language" not in command
    else:
        assert command[command.index("--language") + 1] == language
    assert outputs == {"mixed": TRANSCRIPT}


@pytest.mark.parametrize(
    "automatic_language,has_transcript,exit_code",
    [(False, True, 1), (True, True, 0), (True, False, 1)],
)
def test_language_evaluation_preserves_the_fixed_golden(
    tmp_path: Path,
    fixed_golden_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    automatic_language: bool,
    has_transcript: bool,
    exit_code: int,
) -> None:
    """Automatic evaluation scores references; default checks keep exact goldens."""
    original = fixed_golden_path.read_bytes()
    arguments = [
        "check_golden.py",
        "--golden",
        str(fixed_golden_path),
        "--runtime-bin",
        str(tmp_path),
        "--data-root",
        str(tmp_path),
        "--outputs",
        str(tmp_path / "outputs"),
    ]
    if automatic_language:
        arguments.append("--automatic-language")
    else:
        pass
    monkeypatch.setattr(check_golden.sys, "argv", arguments)
    monkeypatch.setattr(check_golden, "CI_DIRECTORY", tmp_path)
    stdout = json.dumps({"file": "mixed.wav", **TRANSCRIPT}) if has_transcript else ""
    with (
        patch.object(check_golden, "chip", return_value="Apple M5"),
        patch.object(
            check_golden.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0, stdout=stdout),
        ) as runtime,
        pytest.raises(SystemExit) as stopped,
    ):
        check_golden.run(check_golden.transcribe_qwen3_asr)
    assert stopped.value.code == exit_code
    assert fixed_golden_path.read_bytes() == original
    output_name = "fixed-auto.json" if automatic_language else "fixed.json"
    exported = json.loads((tmp_path / "outputs" / output_name).read_text())
    if automatic_language:
        assert "--language" not in runtime.call_args.args[0]
        assert "language" not in exported["request"]
        assert not (tmp_path / "outputs" / "fixed.json").exists()
        report = capsys.readouterr().out
        assert "automatic language" in report
        if has_transcript:
            assert "| mer_mixed | 0.000% |" in report
        else:
            assert "transcribed clips do not match the corpus manifest" in report
    else:
        assert exported["request"]["language"] == "English"


@pytest.mark.parametrize("operation", ["--write", "--import"])
def test_automatic_evaluation_rejects_golden_updates(
    fixed_golden_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: Literal["--write", "--import"],
) -> None:
    """A request override must never overwrite the source golden."""
    original = fixed_golden_path.read_bytes()
    arguments = [
        "check_golden.py",
        "--golden",
        str(fixed_golden_path),
        "--automatic-language",
        operation,
    ]
    if operation == "--import":
        arguments.append(str(fixed_golden_path.parent / "outputs.json"))
    else:
        pass
    monkeypatch.setattr(check_golden.sys, "argv", arguments)
    with pytest.raises(SystemExit) as stopped:
        check_golden.run(check_golden.transcribe_qwen3_asr)
    assert stopped.value.code == 2
    assert fixed_golden_path.read_bytes() == original


@pytest.mark.parametrize("field", ["kind", "tool"])
def test_automatic_evaluation_rejects_other_model_formats(
    fixed_golden_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: Literal["kind", "tool"],
) -> None:
    golden = json.loads(fixed_golden_path.read_text())
    golden[field] = "other-model"
    fixed_golden_path.write_text(json.dumps(golden))
    monkeypatch.setattr(
        check_golden.sys,
        "argv",
        ["check_golden.py", "--golden", str(fixed_golden_path), "--automatic-language"],
    )
    with pytest.raises(SystemExit) as stopped:
        check_golden.run(check_golden.transcribe_qwen3_asr)
    assert stopped.value.code == 2


def test_golden_outputs_cannot_be_imported_into_another_language_condition(
    tmp_path: Path, fixed_golden_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Automatic outputs must not replace a fixed-English golden."""
    original = fixed_golden_path.read_bytes()
    request = json.loads(original)["request"]
    request.pop("language")
    output_path = tmp_path / "automatic-outputs.json"
    output_path.write_text(
        json.dumps(
            {
                "model": "model",
                "request": request,
                "device": "Apple M5",
                "metrics": {},
                "clips": {"mixed": TRANSCRIPT},
            }
        )
    )
    monkeypatch.setattr(
        check_golden.sys,
        "argv",
        [
            "check_golden.py",
            "--golden",
            str(fixed_golden_path),
            "--import",
            str(output_path),
        ],
    )
    with pytest.raises(SystemExit, match="different request"):
        check_golden.run(check_golden.transcribe_qwen3_asr)
    assert fixed_golden_path.read_bytes() == original
