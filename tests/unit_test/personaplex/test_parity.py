# SPDX-License-Identifier: Apache-2.0
"""Parity checks reject incomplete recordings, including partial final frames."""

import json
import subprocess
import sys
from dataclasses import asdict

import numpy as np
import pytest
import soundfile

from sglang_omni.models.personaplex.architecture import SAMPLE_RATE, SAMPLES_PER_FRAME
from tests.test_model.personaplex_repro import (
    ReferenceInputs,
    ensure_reference_run,
    file_digest,
    pinned_checkpoint,
    reference_checkout,
)
from tests.test_model.test_personaplex_parity import Reply, compare_frames
from tests.test_model.test_personaplex_parity import (
    test_greedy_matches_reference as check_greedy,
)


@pytest.mark.parametrize(
    "missing_samples", [1, SAMPLES_PER_FRAME, 400 * SAMPLES_PER_FRAME]
)
def test_parity_rejects_truncated_audio(missing_samples: int) -> None:
    reference = np.zeros(500 * SAMPLES_PER_FRAME, dtype=np.float32)
    with pytest.raises(ValueError, match="sample counts"):
        compare_frames(reference[:-missing_samples], reference, atol=1e-4)


def test_parity_checks_the_partial_final_frame() -> None:
    reference = np.zeros(SAMPLES_PER_FRAME + 17, dtype=np.float32)
    reply = reference.copy()
    reply[-1] = 1.0
    parity = compare_frames(reply, reference, atol=1e-4)
    assert parity.total_frames == 2
    assert parity.identical_frames == 1


def test_parity_rejects_empty_audio() -> None:
    with pytest.raises(ValueError, match="empty"):
        compare_frames(np.zeros(0), np.zeros(0), atol=1e-4)


@pytest.mark.parametrize(
    "changed", ["revision", "checkpoint", "files", "settings", "artifact"]
)
def test_reference_cache_rejects_changed_inputs(tmp_path, changed: str) -> None:
    artifact = tmp_path / "output.wav"
    artifact.write_bytes(b"original waveform")
    manifest = tmp_path / "manifest.json"
    inputs = ReferenceInputs(
        revision="a" * 40,
        checkpoint={"model": "weights-v1"},
        files={"caller": "audio-v1"},
        settings={"seed": 42},
    )
    manifest.write_text(
        json.dumps(
            {
                "inputs": asdict(inputs),
                "artifacts": {artifact.name: file_digest(artifact)},
            }
        )
    )
    ensure_reference_run(
        command=None,
        source=tmp_path,
        inputs=inputs,
        manifest=manifest,
        artifacts=(artifact,),
    )
    if changed == "artifact":
        artifact.write_bytes(b"truncated")
    else:
        values = asdict(inputs)
        values[changed] = "b" * 40 if changed == "revision" else {"changed": "value"}
        inputs = ReferenceInputs(**values)
    with pytest.raises(ValueError, match="stale reference cache"):
        ensure_reference_run(
            command=None,
            source=tmp_path,
            inputs=inputs,
            manifest=manifest,
            artifacts=(artifact,),
        )


def test_reference_run_records_provenance_and_does_not_reuse_bare_outputs(
    tmp_path,
) -> None:
    artifact = tmp_path / "output.wav"
    artifact.write_bytes(b"old output with no manifest")
    manifest = tmp_path / "manifest.json"
    inputs = ReferenceInputs(revision="a" * 40, checkpoint={}, files={}, settings={})
    with pytest.raises(ValueError, match="stale reference cache"):
        ensure_reference_run(
            command=None,
            source=tmp_path,
            inputs=inputs,
            manifest=manifest,
            artifacts=(artifact,),
        )
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('output.wav').write_bytes(b'new output')",
    ]
    ensure_reference_run(
        command=command,
        source=tmp_path,
        inputs=inputs,
        manifest=manifest,
        artifacts=(artifact,),
    )
    saved = json.loads(manifest.read_text())
    assert artifact.read_bytes() == b"new output"
    assert saved["artifacts"][artifact.name] == file_digest(artifact)
    assert saved["command"] == command
    assert saved["runtime"]["torch"]
    ensure_reference_run(
        command=None,
        source=tmp_path,
        inputs=inputs,
        manifest=manifest,
        artifacts=(artifact,),
    )


def test_parity_rejects_two_equally_truncated_outputs(tmp_path) -> None:
    recording = np.zeros(500 * SAMPLES_PER_FRAME, dtype=np.float32)
    soundfile.write(tmp_path / "input_assistant.wav", recording, SAMPLE_RATE)
    truncated = recording[: 100 * SAMPLES_PER_FRAME]
    with pytest.raises(AssertionError, match="expected .* samples"):
        check_greedy(
            "assistant",
            {"assistant": Reply(text="", audio=truncated)},
            {"assistant": (truncated, ["PAD"] * 100)},
            tmp_path,
        )


def test_parity_rejects_extra_text_when_all_audio_matches(tmp_path) -> None:
    recording = np.zeros(100 * SAMPLES_PER_FRAME + 17, dtype=np.float32)
    soundfile.write(tmp_path / "input_assistant.wav", recording, SAMPLE_RATE)
    with pytest.raises(AssertionError, match="text differs"):
        check_greedy(
            "assistant",
            {"assistant": Reply(text="hello extra", audio=recording)},
            {"assistant": (recording, ["hello"] + ["PAD"] * 100)},
            tmp_path,
        )


def test_remote_checkpoint_requires_an_immutable_revision(monkeypatch) -> None:
    monkeypatch.setenv("PERSONAPLEX_PARITY_CHECKPOINT", "nvidia/personaplex-7b-v1@main")
    with pytest.raises(ValueError, match="full checkpoint commit SHA"):
        pinned_checkpoint("PERSONAPLEX_PARITY_CHECKPOINT")


def test_reference_checkout_rejects_revision_mismatch_and_local_edits(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "reference"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "moshi").mkdir()
    module = source / "moshi" / "reference.py"
    module.write_text("version = 1\n")
    subprocess.run(["git", "-C", str(source), "add", "moshi"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "reference",
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    monkeypatch.setenv("PERSONAPLEX_REFERENCE_SOURCE", str(source))
    monkeypatch.setenv("PERSONAPLEX_REFERENCE_REVISION", revision)
    assert reference_checkout() == source
    monkeypatch.setenv("PERSONAPLEX_REFERENCE_REVISION", "0" * 40)
    with pytest.raises(ValueError, match="full commit SHA"):
        reference_checkout()
    monkeypatch.setenv("PERSONAPLEX_REFERENCE_REVISION", revision)
    module.write_text("version = 2\n")
    with pytest.raises(ValueError, match="must be clean"):
        reference_checkout()
