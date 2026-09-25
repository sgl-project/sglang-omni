#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Lint and optionally rename leading-underscore names.

Every Python file under sglang_omni/ is in scope, including files added later
(new model packages under sglang_omni/models/, new runners, etc.). There is no
per-directory allowlist: a new path is checked as soon as it exists.

Checked names:
- class and function definitions at module or class scope
- every attribute read or write, including another object's attribute
- string literals passed to getattr

Nested functions, and classes defined inside functions, may keep a leading
underscore. Dunder names, a lone "_", and vendor copies are ignored. A line
may opt out with "# noqa: leading-underscore". ALLOWED_DEFS is only the
existing third-party method exceptions; those names are also ignored as
attributes in the same file.

--fix rewrites one file at a time, like isort. Same-scope public-name
collisions are left for the human to resolve. It does not rewrite other files.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "sglang_omni"
VENDOR_ROOT = SOURCE_ROOT / "vendor"
NOQA_CODE = "leading-underscore"

# (posix path relative to repo root, definition name)
ALLOWED_DEFS: frozenset[tuple[str, str]] = frozenset(
    {
        # queue.Queue internals
        ("sglang_omni/scheduling/threaded_simple_scheduler.py", "_init"),
        ("sglang_omni/scheduling/threaded_simple_scheduler.py", "_put"),
        ("sglang_omni/scheduling/threaded_simple_scheduler.py", "_get"),
        # HuggingFace PreTrainedModel
        (
            "sglang_omni/models/ming_omni/talker/audio_vae/modeling_audio_vae.py",
            "_init_weights",
        ),
        (
            "sglang_omni/models/fishaudio_s2_pro/fish_speech/models/dac/rvq.py",
            "_init_weights",
        ),
        # torch.nn.Conv1d
        (
            "sglang_omni/models/minicpm_o/components/token2wav/speech_tokenizer_model.py",
            "_conv_forward",
        ),
        # torch.nn.Module._buffers
        ("sglang_omni/models/qwen3_tts/codec_state_arena.py", "_buffers"),
        # SGLang ModelRunner / Scheduler / cache / MLX hooks
        ("sglang_omni/model_runner/sglang_model_runner.py", "_extend_forward_kwargs"),
        (
            "sglang_omni/model_runner/sglang_model_runner.py",
            "_resolve_draft_load_format",
        ),
        ("sglang_omni/model_runner/sglang_model_runner.py", "_profile_available_bytes"),
        ("sglang_omni/scheduling/omni_scheduler.py", "_add_request_to_queue"),
        ("sglang_omni/scheduling/pd_scheduler.py", "_add_request_to_queue"),
        (
            "sglang_omni/scheduling/sglang_backend/evict_heap_radix_cache.py",
            "_update_leaf_status",
        ),
        (
            "sglang_omni/models/fun_cosyvoice3/mlx/runner.py",
            "_select_tokens_with_logprobs",
        ),
        ("sglang_omni/models/fun_cosyvoice3/mlx/runner.py", "_load_model"),
        ("sglang_omni/models/qwen3_asr/mlx/runner.py", "_load_model"),
        # OmniPlatform hook kept to avoid infinite recursion
        ("sglang_omni/platforms/interface.py", "_get_device_graph_backend"),
        ("sglang_omni/platforms/cuda.py", "_get_device_graph_backend"),
        ("sglang_omni/platforms/rocm.py", "_get_device_graph_backend"),
        ("sglang_omni/platforms/npu.py", "_get_device_graph_backend"),
        ("sglang_omni/platforms/xpu.py", "_get_device_graph_backend"),
        # Same-scope public name already exists
        ("sglang_omni/scheduling/omni_scheduler.py", "_run_batch"),
        ("sglang_omni/models/minimax_music3/dit.py", "_transformer"),
        ("sglang_omni/models/qwen3_tts/reference_encoder_cuda_graph.py", "_encode"),
        ("sglang_omni/models/qwen3_omni/components/code2wav_cuda_graph.py", "_build"),
        # Remaining production exceptions carried from the rename
        ("sglang_omni/scheduling/dllm_scheduler.py", "_event_loop"),
        ("sglang_omni/mps/runtime.py", "_start"),
        ("sglang_omni/mps/runtime.py", "_verify"),
        ("sglang_omni/mps/runtime.py", "_retire_process_clients"),
        ("sglang_omni/mps/runtime.py", "_probe_failures"),
        ("sglang_omni/mps/runtime.py", "_close"),
        ("sglang_omni/preprocessing/resource_connector.py", "_assert_url_allowed"),
        ("sglang_omni/config/placement.py", "_resolve_stage_gpu_ids"),
        (
            "sglang_omni/models/moss_transcribe_diarize/encoder_service.py",
            "_lookup_cached_embedding",
        ),
        ("sglang_omni/models/ming_tts/sglang_model.py", "_is_layer_sparse"),
        ("sglang_omni/models/dots_tts/tail.py", "_log_graph_counters"),
        (
            "sglang_omni/models/fishaudio_s2_pro/streaming_vocoder.py",
            "_build_stream_vocoder_chunk",
        ),
        (
            "sglang_omni/models/fishaudio_s2_pro/streaming_vocoder.py",
            "_is_streaming_payload",
        ),
    }
)


@dataclass(frozen=True)
class Violation:
    path: Path
    lineno: int
    col: int
    name: str
    kind: str


@dataclass(frozen=True)
class Site:
    lineno: int
    col: int
    old: str
    new: str


@dataclass(frozen=True)
class RenamePlan:
    module_renames: dict[str, str]
    method_renames: dict[str, dict[str, str]]
    attribute_renames: dict[str, dict[str, str]]

    def is_empty(self) -> bool:
        return (
            not self.module_renames
            and not self.method_renames
            and not self.attribute_renames
        )


def is_leading_underscore_name(name: str) -> bool:
    if name == "_" or name.startswith("__"):
        return False
    return name.startswith("_")


def has_noqa(line: str) -> bool:
    marker = line.split("#", 1)
    if len(marker) != 2:
        return False
    comment = marker[1]
    if "noqa" not in comment:
        return False
    return NOQA_CODE in comment


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def is_in_scope(path: Path) -> bool:
    resolved = path.resolve()
    if not resolved.is_relative_to(SOURCE_ROOT):
        return False
    if resolved.is_relative_to(VENDOR_ROOT):
        return False
    return resolved.suffix == ".py"


def is_self_or_cls(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id in {"self", "cls"}


def iter_assign_targets(target: ast.AST) -> list[ast.AST]:
    if isinstance(target, (ast.Tuple, ast.List)):
        nested: list[ast.AST] = []
        for element in target.elts:
            nested.extend(iter_assign_targets(element))
        return nested
    if isinstance(target, ast.Starred):
        return iter_assign_targets(target.value)
    return [target]


class LeadingUnderscoreVisitor(ast.NodeVisitor):
    def __init__(self, path: Path, source_lines: list[str]) -> None:
        self.path = path
        self.source_lines = source_lines
        self.function_depth = 0
        self.noqa_defs: set[str] = set()
        self.parents_linked = False
        self.violations: list[Violation] = []

    def visit(self, node: ast.AST) -> None:
        if not self.parents_linked:
            for parent in ast.walk(node):
                for child in ast.iter_child_nodes(parent):
                    child.parent = parent  # type: ignore[attr-defined]
            self.parents_linked = True
        super().visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if self.function_depth == 0:
            self._record(node, "class")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node, "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node, "async function")

    def _visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, kind: str
    ) -> None:
        if self.function_depth == 0:
            self._record(node, kind)
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._record_assigned_attribute(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._record_assigned_attribute(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._record_assigned_attribute(node.target)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        column = (node.end_col_offset or 0) - len(node.attr)
        self._record_name(node.attr, node.lineno, column, "attribute", node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self._record_getattr(node)
        self.generic_visit(node)

    def _record_assigned_attribute(self, target: ast.AST) -> None:
        for item in iter_assign_targets(target):
            if not isinstance(item, ast.Attribute) or is_self_or_cls(item.value):
                continue
            column = (item.end_col_offset or 0) - len(item.attr)
            self._record_name(item.attr, item.lineno, column, "attribute", item)

    def _record_getattr(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name) or node.func.id != "getattr":
            return
        if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant):
            return
        attr_name = node.args[1].value
        if not isinstance(attr_name, str):
            return
        self._record_name(
            attr_name,
            node.args[1].lineno,
            node.args[1].col_offset,
            "attribute",
            node.args[1],
        )

    def _record(self, node: ast.AST, kind: str) -> None:
        self._record_name(
            node.name,  # type: ignore[attr-defined]
            node.lineno,
            node.col_offset,
            kind,
            node,
        )

    def statement_has_noqa(self, node: ast.AST) -> bool:
        current: ast.AST | None = node
        while current is not None and not isinstance(current, ast.stmt):
            current = getattr(current, "parent", None)
        if current is None:
            return False
        start = current.lineno
        end = current.end_lineno or start
        return any(
            has_noqa(self.source_lines[index]) for index in range(start - 1, end)
        )

    def _record_name(
        self, name: str, lineno: int, column: int, kind: str, node: ast.AST
    ) -> None:
        if not is_leading_underscore_name(name):
            return
        if lineno < 1 or lineno > len(self.source_lines):
            return
        if has_noqa(self.source_lines[lineno - 1]) or self.statement_has_noqa(node):
            return
        if (repo_relative(self.path), name) in ALLOWED_DEFS:
            return
        if kind == "attribute" and name in self.noqa_defs:
            return
        self.violations.append(Violation(self.path, lineno, column, name, kind))


def noqa_definition_names(
    tree: ast.AST, source_lines: list[str], path: Path
) -> set[str]:
    names: set[str] = set()
    relative = repo_relative(path)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if (relative, node.name) in ALLOWED_DEFS or has_noqa(
            source_lines[node.lineno - 1]
        ):
            names.add(node.name)
    return names


def check_file(path: Path) -> list[Violation]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    lines = source.splitlines()
    visitor = LeadingUnderscoreVisitor(path, lines)
    visitor.noqa_defs = noqa_definition_names(tree, lines, path)
    visitor.visit(tree)
    return visitor.violations


def iter_default_files() -> list[Path]:
    return sorted(path for path in SOURCE_ROOT.rglob("*.py") if is_in_scope(path))


def resolve_targets(raw_paths: list[str]) -> list[Path]:
    if not raw_paths:
        return iter_default_files()
    return [Path(item) for item in raw_paths if is_in_scope(Path(item))]


def format_violation(violation: Violation) -> str:
    rel = repo_relative(violation.path)
    return (
        f"{rel}:{violation.lineno}:{violation.col}: "
        f"leading-underscore {violation.kind} {violation.name!r}; "
        f"nested functions may keep '_'; run with --fix, or use a public "
        f"name / '# noqa: {NOQA_CODE}'"
    )


def public_name(name: str) -> str:
    return name[1:]


def definition_name_column(node: ast.AST, line: str) -> int:
    keyword = "class" if isinstance(node, ast.ClassDef) else "def"
    start = line.find(keyword, node.col_offset)
    if start < 0:
        start = line.find(keyword)
    start += len(keyword)
    while start < len(line) and line[start].isspace():
        start += 1
    return start


def apply_sites(source: str, sites: list[Site]) -> str:
    if not sites:
        return source
    lines = source.splitlines(keepends=True)
    ordered = sorted(sites, key=lambda site: (site.lineno, site.col), reverse=True)
    for site in ordered:
        line = lines[site.lineno - 1]
        stop = site.col + len(site.old)
        if line[site.col : stop] != site.old:
            raise RuntimeError(
                f"site mismatch line {site.lineno} col {site.col}: "
                f"expected {site.old!r} found {line[site.col:stop]!r}"
            )
        lines[site.lineno - 1] = line[: site.col] + site.new + line[stop:]
    return "".join(lines)


class ScopeIndex(ast.NodeVisitor):
    def __init__(self) -> None:
        self.function_depth = 0
        self.class_stack: list[str] = []
        self.module_names: set[str] = set()
        self.class_names: dict[str, set[str]] = defaultdict(set)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.add_name(node.name)
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_function(node)

    def visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if self.function_depth == 0:
            self.add_name(node.name)
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self.class_stack and is_self_or_cls(node.value):
            self.class_names[".".join(self.class_stack)].add(node.attr)
        self.generic_visit(node)

    def add_name(self, name: str) -> None:
        if self.function_depth > 0:
            return
        owner = ".".join(self.class_stack)
        if owner:
            self.class_names[owner].add(name)
        else:
            self.module_names.add(name)


class RenamePlanner(ast.NodeVisitor):
    def __init__(self, wanted: set[tuple[int, str]], index: ScopeIndex) -> None:
        self.wanted = wanted
        self.index = index
        self.function_depth = 0
        self.class_stack: list[str] = []
        self.module_renames: dict[str, str] = {}
        self.method_renames: dict[str, dict[str, str]] = defaultdict(dict)
        self.attribute_renames: dict[str, dict[str, str]] = defaultdict(dict)
        self.class_defs: dict[str, set[str]] = defaultdict(set)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.consider(node)
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_function(node)

    def visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if self.function_depth == 0:
            self.consider(node)
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

    def consider(self, node: ast.AST) -> None:
        if self.function_depth > 0:
            return
        name = node.name  # type: ignore[attr-defined]
        if (node.lineno, name) not in self.wanted:
            return
        new = public_name(name)
        owner = ".".join(self.class_stack)
        existing = (
            self.index.module_names
            if not owner
            else self.index.class_names.get(owner, set())
        )
        if new in existing:
            return
        if owner:
            self.method_renames[owner][name] = new
            self.class_defs[owner].add(name)
        else:
            self.module_renames[name] = new

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if is_self_or_cls(node.value) and (node.lineno, node.attr) in self.wanted:
            self.consider_attribute(node.attr)
        self.generic_visit(node)

    def consider_attribute(self, name: str) -> None:
        owner = ".".join(self.class_stack)
        if not owner or name in self.class_defs.get(owner, set()):
            return
        new = public_name(name)
        if new in self.index.class_names.get(owner, set()):
            return
        self.attribute_renames[owner][name] = new


def file_attribute_renames(plan: RenamePlan) -> dict[str, str]:
    agreed: dict[str, set[str]] = defaultdict(set)
    for mapping in plan.attribute_renames.values():
        for old, new in mapping.items():
            agreed[old].add(new)
    return {old: next(iter(news)) for old, news in agreed.items() if len(news) == 1}


def plan_renames(tree: ast.AST, violations: list[Violation]) -> RenamePlan:
    index = ScopeIndex()
    index.visit(tree)
    planner = RenamePlanner({(item.lineno, item.name) for item in violations}, index)
    planner.visit(tree)
    return RenamePlan(
        planner.module_renames,
        dict(planner.method_renames),
        dict(planner.attribute_renames),
    )


class FixVisitor(ast.NodeVisitor):
    def __init__(self, source_lines: list[str], plan: RenamePlan) -> None:
        self.source_lines = source_lines
        self.plan = plan
        self.file_attributes = file_attribute_renames(plan)
        self.function_depth = 0
        self.class_stack: list[str] = []
        self.sites: list[Site] = []

    def add_site(self, lineno: int, col: int, old: str, new: str) -> None:
        if old != new:
            self.sites.append(Site(lineno, col, old, new))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if self.function_depth == 0:
            self.maybe_rename_def(node)
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_function(node)

    def visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if self.function_depth == 0:
            self.maybe_rename_def(node)
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

    def maybe_rename_def(self, node: ast.AST) -> None:
        name = node.name  # type: ignore[attr-defined]
        new = self.rename_for_current_scope(name)
        if new is None:
            return
        line = self.source_lines[node.lineno - 1]
        self.add_site(node.lineno, definition_name_column(node, line), name, new)

    def rename_for_current_scope(self, name: str) -> str | None:
        owner = ".".join(self.class_stack)
        if owner:
            return self.plan.method_renames.get(owner, {}).get(name)
        return self.plan.module_renames.get(name)

    def visit_Name(self, node: ast.Name) -> None:
        new = self.plan.module_renames.get(node.id)
        if new:
            self.add_site(node.lineno, node.col_offset, node.id, new)

    def visit_Call(self, node: ast.Call) -> None:
        self.rename_getattr(node)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        new = self.attribute_rename(node)
        lineno = node.end_lineno or node.lineno
        if new and not has_noqa(self.source_lines[lineno - 1]):
            col = (node.end_col_offset or 0) - len(node.attr)
            self.add_site(lineno, col, node.attr, new)
        self.generic_visit(node)

    def rename_getattr(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name) or node.func.id != "getattr":
            return
        if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant):
            return
        old = node.args[1].value
        if not isinstance(old, str):
            return
        new = self.file_attributes.get(old)
        if new is None:
            return
        constant = node.args[1]
        if has_noqa(self.source_lines[constant.lineno - 1]):
            return
        literal = self.source_lines[constant.lineno - 1][
            constant.col_offset : constant.end_col_offset
        ]
        if old not in literal:
            return
        self.add_site(
            constant.lineno,
            constant.col_offset,
            literal,
            literal.replace(old, new, 1),
        )

    def attribute_rename(self, node: ast.Attribute) -> str | None:
        if is_self_or_cls(node.value):
            renamed = self.method_rename(node.attr)
            if renamed is not None:
                return renamed
            renamed = self.field_rename(node.attr)
            if renamed is not None:
                return renamed
            return self.file_attributes.get(node.attr)
        if isinstance(node.value, ast.Name):
            return self.class_attr_rename(node.value.id, node.attr)
        return None

    def field_rename(self, name: str) -> str | None:
        current = ".".join(self.class_stack)
        while current:
            mapping = self.plan.attribute_renames.get(current)
            if mapping and name in mapping:
                return mapping[name]
            if "." not in current:
                break
            current = current.rsplit(".", 1)[0]
        return None

    def method_rename(self, name: str) -> str | None:
        current = ".".join(self.class_stack)
        while current:
            mapping = self.plan.method_renames.get(current)
            if mapping and name in mapping:
                return mapping[name]
            if "." not in current:
                break
            current = current.rsplit(".", 1)[0]
        return None

    def class_attr_rename(self, class_name: str, attr: str) -> str | None:
        for qual, mapping in self.plan.method_renames.items():
            if qual.split(".")[-1] == class_name and attr in mapping:
                return mapping[attr]
        return None


def fix_file(path: Path) -> tuple[int, list[Violation]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    lines = source.splitlines()
    before = LeadingUnderscoreVisitor(path, lines)
    before.noqa_defs = noqa_definition_names(tree, lines, path)
    before.visit(tree)
    if not before.violations:
        return 0, []
    plan = plan_renames(tree, before.violations)
    if plan.is_empty():
        return 0, before.violations
    visitor = FixVisitor(source.splitlines(), plan)
    visitor.visit(tree)
    rewritten = apply_sites(source, visitor.sites)
    if rewritten != source:
        path.write_text(rewritten, encoding="utf-8")
    leftover = check_file(path)
    return max(len(before.violations) - len(leftover), 0), leftover


def report_violations(violations: list[Violation]) -> int:
    if not violations:
        return 0
    for violation in violations:
        print(format_violation(violation), file=sys.stderr)
    print(
        f"{len(violations)} leading-underscore class/function/attribute name(s) "
        f"in sglang_omni/",
        file=sys.stderr,
    )
    return 1


def run_check(paths: list[Path]) -> int:
    violations: list[Violation] = []
    for path in paths:
        try:
            violations.extend(check_file(path))
        except SyntaxError as exc:
            print(f"{path}: failed to parse: {exc}", file=sys.stderr)
            return 2
    return report_violations(violations)


def run_fix(paths: list[Path]) -> int:
    remaining: list[Violation] = []
    renamed = 0
    files = 0
    for path in paths:
        try:
            count, leftover = fix_file(path)
        except SyntaxError as exc:
            print(f"{path}: failed to parse: {exc}", file=sys.stderr)
            return 2
        if count:
            files += 1
            renamed += count
        remaining.extend(leftover)
    if renamed:
        print(f"renamed {renamed} leading-underscore name(s) in {files} file(s)")
    return report_violations(remaining)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", help="Python files (defaults to sglang_omni/)"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Rename violating names in-place (current file only)",
    )
    args = parser.parse_args(argv)
    targets = resolve_targets(args.paths)
    if args.fix:
        return run_fix(targets)
    return run_check(targets)


if __name__ == "__main__":
    raise SystemExit(main())
