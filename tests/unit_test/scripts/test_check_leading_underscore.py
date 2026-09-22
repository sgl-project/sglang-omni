# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the leading-underscore lint hook."""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKER_PATH = REPO_ROOT / "scripts" / "check_leading_underscore.py"
PROBE_PACKAGE = "_lint_underscore_probe"


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_leading_underscore", CHECKER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_checker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER_PATH), *args],
        check=False,
        capture_output=True,
        text=True,
    )


@contextmanager
def _probe_model_file(source: str) -> Iterator[tuple[object, Path]]:
    checker = _load_checker()
    probe_dir = checker.SOURCE_ROOT / "models" / PROBE_PACKAGE
    probe = probe_dir / "runner.py"
    probe_dir.mkdir(parents=True)
    probe.write_text(source, encoding="utf-8")
    try:
        yield checker, probe
    finally:
        probe.unlink(missing_ok=True)
        if probe_dir.exists():
            probe_dir.rmdir()


def _violations(source: str, tmp_path: Path) -> set[str]:
    checker = _load_checker()
    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")
    visitor = checker.LeadingUnderscoreVisitor(path, source.splitlines())
    visitor.visit(ast.parse(source))
    return {item.name for item in visitor.violations}


def test_current_sglang_omni_tree_is_clean() -> None:
    result = _run_checker()
    assert result.returncode == 0, result.stderr


def test_nested_function_dunder_and_noqa_are_allowed(tmp_path: Path) -> None:
    source = """
class Public:
    def __init__(self) -> None:
        def _inner() -> None:
            return None
        _inner()

def _kept() -> None:  # noqa: leading-underscore
    return None
"""
    assert _violations(source, tmp_path) == set()


def test_top_level_underscore_class_and_method_are_reported(tmp_path: Path) -> None:
    source = "class _Hidden:\n    def _method(self) -> None:\n        return None\n"
    assert _violations(source, tmp_path) == {"_Hidden", "_method"}


def test_new_model_package_file_is_checked_without_registration() -> None:
    """A newly added models/<name>/*.py file is in scope automatically."""
    with _probe_model_file("def _load_checkpoint() -> None:\n    return None\n") as (
        checker,
        probe,
    ):
        assert checker.is_in_scope(probe)
        result = _run_checker(str(probe))
        assert result.returncode == 1
        assert "_load_checkpoint" in result.stderr
        default_scan = _run_checker()
        assert default_scan.returncode == 1
        assert f"{PROBE_PACKAGE}/runner.py" in default_scan.stderr


def test_fix_renames_new_model_file_and_in_file_refs() -> None:
    source = (
        "class Runner:\n"
        "    def _setup(self) -> None:\n"
        "        return None\n"
        "    def run(self) -> None:\n"
        "        self._setup()\n"
    )
    with _probe_model_file(source) as (_checker, probe):
        result = _run_checker("--fix", str(probe))
        assert result.returncode == 0, result.stderr
        rewritten = probe.read_text(encoding="utf-8")
        assert "def setup(self)" in rewritten
        assert "self.setup()" in rewritten
        assert "def _setup" not in rewritten


def test_fix_skips_same_scope_public_name_collision() -> None:
    source = (
        "def load_checkpoint() -> None:\n"
        "    return None\n"
        "\n"
        "def _load_checkpoint() -> None:\n"
        "    return None\n"
    )
    with _probe_model_file(source) as (_checker, probe):
        result = _run_checker("--fix", str(probe))
        assert result.returncode == 1
        assert "_load_checkpoint" in result.stderr
        assert "def _load_checkpoint" in probe.read_text(encoding="utf-8")


def test_fix_preserves_third_party_attribute_with_same_name() -> None:
    source = """
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

def _get_feat_extract_output_lengths(lengths):
    return hf_modeling._get_feat_extract_output_lengths(lengths)

lengths = _get_feat_extract_output_lengths([100, 200])
"""
    with _probe_model_file(source) as (_checker, probe):
        result = _run_checker(str(probe))
        assert result.returncode == 1
        assert "1 leading-underscore" in result.stderr

        result = _run_checker("--fix", str(probe))
        assert result.returncode == 0, result.stderr
        expected = source.replace(
            "def _get_feat_extract_output_lengths(",
            "def get_feat_extract_output_lengths(",
        ).replace(
            "lengths = _get_feat_extract_output_lengths(",
            "lengths = get_feat_extract_output_lengths(",
        )
        assert probe.read_text(encoding="utf-8") == expected


def test_fix_preserves_noqa_definitions_and_references() -> None:
    kept = """
def _required_external_hook():  # noqa: leading-underscore
    return None

class _ExternalAdapter:  # noqa: leading-underscore
    def _required_external_hook(self):  # noqa: leading-underscore
        return _required_external_hook()

    def run(self):
        return self._required_external_hook()

adapter = _ExternalAdapter()
hook = _ExternalAdapter._required_external_hook
"""
    local = "\ndef _local_helper():\n    return None\n\n_local_helper()\n"
    with _probe_model_file(kept) as (_checker, probe):
        result = _run_checker(str(probe))
        assert result.returncode == 0, result.stderr

        probe.write_text(kept + local, encoding="utf-8")
        result = _run_checker("--fix", str(probe))
        assert result.returncode == 0, result.stderr
        assert probe.read_text(encoding="utf-8") == kept + local.replace(
            "_local_helper", "local_helper"
        )
