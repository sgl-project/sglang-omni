# SPDX-License-Identifier: Apache-2.0
"""Guard Omni's MLX adapters against drifting from SGLang's attribute names.

The adapters subclass SGLang's MLX runner and worker and share their instance
state. Renaming an SGLang-owned ``_x`` to ``x`` on the Omni side silently
creates a second attribute, which only fails at the first real forward. The
check parses source instead of importing it, so it runs without ``mlx``.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

SGLANG_SOURCES = (
    "srt/managers/tp_worker.py",
    "srt/hardware_backend/mlx/model_runner.py",
    "srt/hardware_backend/mlx/tp_worker.py",
    "srt/hardware_backend/mlx/scheduler_mixin.py",
)

OMNI_ADAPTERS = (
    "sglang_omni/model_runner/mlx_model_worker.py",
    "sglang_omni/model_runner/audio_mlx.py",
    "sglang_omni/models/fun_asr/mlx/runner.py",
    "sglang_omni/models/fun_cosyvoice3/mlx/runner.py",
    "sglang_omni/models/qwen3_asr/mlx/runner.py",
)


def member_names(source: str) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        else:
            pass
    return names


def shadowed_names(adapter_source: str, sglang_names: set[str]) -> list[str]:
    return sorted(
        name
        for name in member_names(adapter_source)
        if not name.startswith("_")
        and f"_{name}" in sglang_names
        and name not in sglang_names
    )


@pytest.fixture(scope="module")
def sglang_names() -> set[str]:
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("sglang is not installed")
    else:
        pass
    root = Path(next(iter(spec.submodule_search_locations)))
    paths = [root / relative for relative in SGLANG_SOURCES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        pytest.skip(f"sglang has no native MLX backend: {missing}")
    else:
        pass
    return set().union(*(member_names(path.read_text()) for path in paths))


@pytest.mark.parametrize("relative_path", OMNI_ADAPTERS)
def test_mlx_adapter_attribute_names_match_sglang(relative_path, sglang_names):
    source = (REPO_ROOT / relative_path).read_text()

    assert shadowed_names(source, sglang_names) == [], (
        f"{relative_path} uses public names that SGLang spells with a leading "
        "underscore; restore the upstream spelling with "
        "`# noqa: leading-underscore`."
    )


def test_guard_flags_a_de_underscored_sglang_attribute():
    sglang_names = member_names("self._req_caches = {}\nself.model = None\n")

    assert shadowed_names("cache = self.req_caches[rid]\n", sglang_names) == [
        "req_caches"
    ]
    assert shadowed_names("self.model(x)\n", sglang_names) == []
