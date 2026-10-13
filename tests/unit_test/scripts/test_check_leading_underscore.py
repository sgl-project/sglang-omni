# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the leading-underscore lint hook."""

from __future__ import annotations

import ast
import importlib.util
import runpy
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKER_PATH = REPO_ROOT / "scripts" / "check_leading_underscore.py"
PROBE_PACKAGE = "_lint_underscore_probe"


def load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_leading_underscore", CHECKER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_checker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER_PATH), *args],
        check=False,
        capture_output=True,
        text=True,
    )


@contextmanager
def probe_model_file(source: str) -> Iterator[tuple[object, Path]]:
    checker = load_checker()
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


def violations(source: str, tmp_path: Path) -> set[str]:
    checker = load_checker()
    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")
    visitor = checker.LeadingUnderscoreVisitor(path, source.splitlines())
    visitor.visit(ast.parse(source))
    return {item.name for item in visitor.violations}


def test_current_tree_is_clean() -> None:
    result = run_checker()
    assert result.returncode == 0, result.stderr


def test_test_files_are_in_scope() -> None:
    checker = load_checker()
    assert checker.is_in_scope(Path(__file__))
    assert Path(__file__).resolve() in checker.iter_default_files()


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
    assert violations(source, tmp_path) == set()


def test_self_attribute_assignment_and_getattr_are_reported(tmp_path: Path) -> None:
    source = """
class Session:
    def attach(self, request):
        self._cache_key = request._cache_key
        request._cache_key = self._cache_key
        key = getattr(request, "_cache_key", None)
        return key
"""
    assert violations(source, tmp_path) == {"_cache_key"}


def test_noqa_on_a_wrapped_statement_covers_the_attribute(tmp_path: Path) -> None:
    source = """
value = (
    request._omni_prompt_cache_key
)  # noqa: leading-underscore
"""
    assert violations(source, tmp_path) == set()


def test_upstream_attribute_read_is_reported(tmp_path: Path) -> None:
    source = "value = hf_modeling._get_feat_extract_output_lengths(lengths)\n"
    assert violations(source, tmp_path) == {"_get_feat_extract_output_lengths"}


def test_top_level_underscore_class_and_method_are_reported(tmp_path: Path) -> None:
    source = "class _Hidden:\n    def _method(self) -> None:\n        return None\n"
    assert violations(source, tmp_path) == {"_Hidden", "_method"}


def test_new_model_package_file_is_checked_without_registration() -> None:
    """A newly added models/<name>/*.py file is in scope automatically."""
    with probe_model_file("def _load_checkpoint() -> None:\n    return None\n") as (
        checker,
        probe,
    ):
        assert checker.is_in_scope(probe)
        result = run_checker(str(probe))
        assert result.returncode == 1
        assert "_load_checkpoint" in result.stderr
        default_scan = run_checker()
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
    with probe_model_file(source) as (_checker, probe):
        result = run_checker("--fix", str(probe))
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
    with probe_model_file(source) as (_checker, probe):
        result = run_checker("--fix", str(probe))
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
    with probe_model_file(source) as (_checker, probe):
        result = run_checker(str(probe))
        assert result.returncode == 1
        assert "_get_feat_extract_output_lengths" in result.stderr

        result = run_checker("--fix", str(probe))
        assert result.returncode == 1
        rewritten = probe.read_text(encoding="utf-8")
        assert "def get_feat_extract_output_lengths(" in rewritten
        assert "hf_modeling._get_feat_extract_output_lengths" in rewritten
        assert "lengths = get_feat_extract_output_lengths(" in rewritten


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
    with probe_model_file(kept) as (_checker, probe):
        result = run_checker(str(probe))
        assert result.returncode == 0, result.stderr

        probe.write_text(kept + local, encoding="utf-8")
        result = run_checker("--fix", str(probe))
        assert result.returncode == 0, result.stderr
        assert probe.read_text(encoding="utf-8") == kept + local.replace(
            "_local_helper", "local_helper"
        )


def file_violations(source: str, tmp_path: Path) -> list[tuple[int, str]]:
    checker = load_checker()
    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")
    return [
        (violation.lineno, violation.name) for violation in checker.check_file(path)
    ]


def test_noqa_in_class_body_does_not_exempt_class_name(tmp_path: Path) -> None:
    source = """
class _Hidden:
    def method(self) -> int:
        return 1  # noqa: leading-underscore
"""
    assert file_violations(source, tmp_path) == [(2, "_Hidden")]


def test_noqa_in_function_body_does_not_exempt_def_name(tmp_path: Path) -> None:
    source = """
def _helper() -> int:
    value = 1  # noqa: leading-underscore
    return value
"""
    assert file_violations(source, tmp_path) == [(2, "_helper")]


def test_noqa_in_block_body_does_not_exempt_header_attributes(
    tmp_path: Path,
) -> None:
    source = """
def run(request) -> None:
    with request._lock:
        is_locked = True  # noqa: leading-underscore
    if request._ready:
        is_ready = True  # noqa: leading-underscore
    for entry in request._entries:
        is_seen = True  # noqa: leading-underscore
"""
    assert file_violations(source, tmp_path) == [
        (3, "_lock"),
        (5, "_ready"),
        (7, "_entries"),
    ]


def test_noqa_on_a_wrapped_block_header_covers_the_header(tmp_path: Path) -> None:
    source = """
def run(request) -> None:
    with (
        request._lock
    ):  # noqa: leading-underscore
        is_locked = True
"""
    assert file_violations(source, tmp_path) == []


def test_noqa_on_a_wrapped_except_header_covers_the_header(tmp_path: Path) -> None:
    source = """
def run() -> None:
    try:
        pass
    except (
        errors._Timeout,
    ):  # noqa: leading-underscore
        pass
"""
    assert file_violations(source, tmp_path) == []


def test_noqa_on_a_wrapped_header_with_an_inline_body(tmp_path: Path) -> None:
    source = """
def run(request) -> None:
    with (
        request._lock
    ): pass  # noqa: leading-underscore
"""
    assert file_violations(source, tmp_path) == []


def test_attribute_assignment_is_reported_once(tmp_path: Path) -> None:
    source = "def attach(request) -> None:\n    request._cache_key = 1\n"
    assert file_violations(source, tmp_path) == [(2, "_cache_key")]


def test_fix_renames_definitions_but_not_attributes() -> None:
    source = """
import threading


class Worker(threading.Thread):
    def _setup(self) -> None:
        self._cache = {}

    def run(self) -> None:
        self._setup()
        self._target(*self._args)
"""
    with probe_model_file(source) as (_checker, probe):
        result = run_checker("--fix", str(probe))
        rewritten = probe.read_text(encoding="utf-8")
    assert result.returncode == 1
    assert rewritten == source.replace("_setup", "setup")
    assert "--fix renames only class and function names" in result.stderr


def test_fix_renames_an_inherited_method_call_in_the_same_file() -> None:
    source = """
class Base:
    def _setup(self) -> int:
        return 42


class Derived(Base):
    def run(self) -> int:
        return self._setup()
"""
    with probe_model_file(source) as (_checker, probe):
        result = run_checker("--fix", str(probe))
        rewritten = probe.read_text(encoding="utf-8")
        namespace = runpy.run_path(str(probe))
    assert result.returncode == 0, result.stderr
    assert rewritten == source.replace("_setup", "setup")
    assert namespace["Derived"]().run() == 42


def test_fix_keeps_a_method_name_it_cannot_rename_everywhere() -> None:
    source = """
class Worker:
    def _setup(self) -> int:
        return 1


def use(worker: Worker) -> int:
    return worker._setup()
"""
    with probe_model_file(source) as (_checker, probe):
        result = run_checker("--fix", str(probe))
        rewritten = probe.read_text(encoding="utf-8")
        namespace = runpy.run_path(str(probe))
    assert result.returncode == 1
    assert rewritten == source
    assert namespace["use"](namespace["Worker"]()) == 1


def test_fix_keeps_a_function_name_that_a_local_already_uses() -> None:
    source = """
def _lock_for(name: str) -> str:
    return name


def build(name: str) -> str:
    lock_for = _lock_for(name)
    return lock_for
"""
    with probe_model_file(source) as (_checker, probe):
        result = run_checker("--fix", str(probe))
        rewritten = probe.read_text(encoding="utf-8")
        namespace = runpy.run_path(str(probe))
    assert result.returncode == 1
    assert rewritten == source
    assert namespace["build"]("value") == "value"
