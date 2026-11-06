# SPDX-License-Identifier: Apache-2.0
"""CLI parameter selection and golden-file write protection."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import check_golden
import pytest

TEST_DEVICE = "test-device"
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
    fixed_output = {**TRANSCRIPT, "text": "Hello world", "language": "English"}
    recorded = {"metrics": {}, "clips": {"mixed": fixed_output}}
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
                "devices": {TEST_DEVICE: recorded},
            }
        )
    )
    return path


@pytest.mark.parametrize(
    "automatic_language,has_transcript,exit_code",
    [(False, True, 1), (True, True, 0), (True, False, 1)],
    ids=["english-golden", "automatic-reference", "missing-transcript"],
)
def test_language_evaluation(
    tmp_path: Path,
    fixed_golden_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    automatic_language: bool,
    has_transcript: bool,
    exit_code: int,
) -> None:
    """Automatic mode scores references; English keeps exact checks and missing clips fail."""
    original = fixed_golden_path.read_bytes()
    arguments = [
        "check_golden.py", "--golden", str(fixed_golden_path),
        "--runtime-bin", str(tmp_path), "--data-root", str(tmp_path),
        "--outputs", str(tmp_path / "outputs"),
    ]  # fmt: skip
    if automatic_language:
        arguments.append("--automatic-language")
    else:
        pass
    monkeypatch.setattr(check_golden.sys, "argv", arguments)
    monkeypatch.setattr(check_golden, "CI_DIRECTORY", tmp_path)
    stdout = json.dumps({"file": "mixed.wav", **TRANSCRIPT}) if has_transcript else ""
    with (
        patch.object(check_golden, "chip", return_value=TEST_DEVICE),
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
    command = runtime.call_args.args[0]
    output_name = "fixed-auto.json" if automatic_language else "fixed.json"
    exported = json.loads((tmp_path / "outputs" / output_name).read_text())
    assert exported["clips"] == ({"mixed": TRANSCRIPT} if has_transcript else {})
    assert exported["request"].get("language") == (
        None if automatic_language else "English"
    )
    if automatic_language:
        assert "--language" not in command
        assert not (tmp_path / "outputs" / "fixed.json").exists()
    else:
        assert command[command.index("--language") + 1] == "English"
    if has_transcript:
        assert exported["metrics"]["mer_mixed"] == 0
    else:
        pass


@pytest.mark.parametrize(
    "automatic_language,operation",
    [(True, "--write"), (True, "--import"), (False, "--import")],
    ids=["automatic-write", "automatic-import", "mismatched-import"],
)
def test_golden_write_protection(
    fixed_golden_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    automatic_language: bool,
    operation: Literal["--write", "--import"],
) -> None:
    """Neither a language override nor mismatched imports may overwrite the golden."""
    original = fixed_golden_path.read_bytes()
    arguments = ["check_golden.py", "--golden", str(fixed_golden_path), operation]
    if operation == "--import":
        golden = json.loads(original)
        golden["request"].pop("language")
        output_path = fixed_golden_path.parent / "automatic-outputs.json"
        output_path.write_text(
            json.dumps(
                {
                    "model": golden["model"],
                    "request": golden["request"],
                    "device": TEST_DEVICE,
                    **golden["devices"][TEST_DEVICE],
                }
            )
        )
        arguments.append(str(output_path))
    else:
        pass
    if automatic_language:
        arguments.append("--automatic-language")
    else:
        pass
    monkeypatch.setattr(check_golden.sys, "argv", arguments)
    with pytest.raises(SystemExit) as stopped:
        check_golden.run(check_golden.transcribe_qwen3_asr)
    if automatic_language:
        assert stopped.value.code == 2
    else:
        assert "different request" in str(stopped.value)
    assert fixed_golden_path.read_bytes() == original
