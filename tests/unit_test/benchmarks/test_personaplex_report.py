# SPDX-License-Identifier: Apache-2.0
"""Reconstruction and drift checks for retained quality evidence."""

import json
import tarfile
from pathlib import Path

import pytest

from benchmarks.eval.benchmark_personaplex_report import build_report, main


def test_report_rebuild_and_check_detect_changed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing a measured score must invalidate the checked-in report."""
    sources = {
        "v10-reference/summary.json": {"score": 1},
        "v10-reference/unified.json": {},
        "v15-scores/summary.json": {"engines": {"personaplex": {}}},
        "v15-judge/summary.json": {},
        "parity-warm/parity.json": {
            "input_sha256": "input",
            "full_reply": {"byte_identical": False},
            "caller_duration": {},
            "session_extra_samples": 3648,
        },
        "mimi-encoding-parity.json": {},
        "scoring-environment.json": {},
        "diagnostics.json": {"fixed_input_codes": {"equal": False}},
    }
    for relative, contents in sources.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(contents))
    conditions = tmp_path / "conditions.json"
    conditions.write_bytes(
        (
            Path(__file__).parents[3]
            / "benchmarks/duplex/personaplex-baseline-conditions.json"
        ).read_bytes()
    )
    output = tmp_path / "report.json"
    arguments = [
        "report",
        "--artifacts",
        str(tmp_path),
        "--conditions",
        str(conditions),
        "--output",
        str(output),
    ]
    bundle = tmp_path / "inputs.tar.gz"
    monkeypatch.setattr("sys.argv", [*arguments, "--bundle", str(bundle)])
    main()
    assert (
        json.loads(output.read_text())["measurements"]["greedy_comparison"][
            "full_reply"
        ]["byte_identical"]
        is False
    )
    monkeypatch.setattr("sys.argv", [*arguments, "--check"])
    main()
    extracted = tmp_path / "extracted"
    with tarfile.open(bundle) as archive:
        archive.extractall(extracted, filter="data")
    assert build_report(extracted, extracted / "conditions.json") == json.loads(
        output.read_text()
    )
    assert json.loads(output.read_text())["measurements"]["diagnostics"] == {
        "fixed_input_codes": {"equal": False}
    }
    (tmp_path / "v10-reference/summary.json").write_text('{"score": 2}')
    with pytest.raises(SystemExit) as failure:
        main()
    assert failure.value.code == 1
    conditions.write_text("{}")
    with pytest.raises(ValueError, match="Missing experiment conditions"):
        build_report(tmp_path, conditions)


def test_conditions_require_hardware_identity(tmp_path: Path) -> None:
    """A report must reject conditions that cannot identify the measured GPU."""
    conditions = tmp_path / "conditions.json"
    template = (
        Path(__file__).parents[3]
        / "benchmarks/duplex/personaplex-baseline-conditions.json"
    )
    contents = json.loads(template.read_text())
    contents["hardware"].pop("gpu")
    conditions.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="hardware.gpu"):
        build_report(tmp_path, conditions)
