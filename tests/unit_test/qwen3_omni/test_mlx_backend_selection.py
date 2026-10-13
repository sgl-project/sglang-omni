"""Backend imports work in fresh interpreters with optional MLX dependencies."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "script",
    [
        "import sglang_omni.models.qwen3_omni.request_builders",
        "import sglang; import sglang_omni.platforms",
        "import sglang_omni.scheduling.sglang_backend",
    ],
)
def test_apple_import_order(script: str) -> None:
    pytest.importorskip("mlx.core")
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_cuda_stage_imports_do_not_require_mlx():
    script = """
import builtins
import importlib.util

names = {"mlx", "mlx_lm", "mlx_vlm", "mlx_audio"}
find_spec = importlib.util.find_spec
original_import = builtins.__import__

def absent_spec(name, *args, **kwargs):
    if name.split(".")[0] in names:
        return None
    else:
        return find_spec(name, *args, **kwargs)

def guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in names:
        raise ImportError("MLX import forbidden: " + name)
    else:
        return original_import(name, *args, **kwargs)

importlib.util.find_spec = absent_spec
builtins.__import__ = guarded_import
import sglang_omni.models.qwen3_omni.stages
import sglang_omni.models.qwen3_omni.bootstrap
import sglang_omni.model_runner.mlx_model_worker
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "SGLANG_USE_MLX": "0"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
