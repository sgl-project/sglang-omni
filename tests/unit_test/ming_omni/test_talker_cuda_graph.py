# SPDX-License-Identifier: Apache-2.0
"""Ming talker CUDA graph source-level regression tests."""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TALKER_SOURCE = (
    _REPO_ROOT
    / "sglang_omni"
    / "models"
    / "ming_omni"
    / "talker"
    / "modeling_ming_omni_talker.py"
)


def _method_node(
    tree: ast.Module, class_name: str, method_name: str
) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"{class_name}.{method_name} not found")


def _torch_cuda_graph_calls(node: ast.AST) -> list[ast.Call]:
    return [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Attribute)
        and item.func.attr == "graph"
        and isinstance(item.func.value, ast.Attribute)
        and item.func.value.attr == "cuda"
        and isinstance(item.func.value.value, ast.Name)
        and item.func.value.value.id == "torch"
    ]


def _has_thread_local_capture_error_mode(call: ast.Call) -> bool:
    return any(
        keyword.arg == "capture_error_mode"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == "thread_local"
        for keyword in call.keywords
    )


def test_ming_lazy_cuda_graph_captures_use_thread_local_error_mode() -> None:
    tree = ast.parse(_TALKER_SOURCE.read_text())
    methods = [
        ("CFMGraphExecutor", "_initialize_graph"),
        ("MingOmniTalker", "generate"),
    ]

    for class_name, method_name in methods:
        method = _method_node(tree, class_name, method_name)
        graph_calls = _torch_cuda_graph_calls(method)
        assert graph_calls, f"{class_name}.{method_name} CUDA graph capture not found"
        assert all(
            _has_thread_local_capture_error_mode(call) for call in graph_calls
        ), (
            f"{class_name}.{method_name} CUDA graph capture must use "
            "thread-local error mode"
        )
