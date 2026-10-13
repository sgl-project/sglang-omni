"""CUDA imports must work when optional MLX packages are absent."""

import os
import subprocess
import sys

import pytest


def test_native_factories_do_not_import_optional_model_libraries():
    pytest.importorskip("mlx.core")
    script = """
import importlib.abc
import sys
import mlx.core as mx
mx.set_default_device(mx.cpu)

class BlockOptionalModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"mlx_vlm", "mlx_audio"}:
            raise ImportError("Optional model library forbidden: " + fullname)
        return None

sys.meta_path.insert(0, BlockOptionalModels())
from sglang_omni.models.qwen3_omni.mlx.loading import load_component
from sglang_omni.models.qwen3_omni.mlx.encoders import (
    MlxAudioEncoder, MlxCode2Wav, MlxImageEncoder,
)
from sglang_omni.models.qwen3_omni.mlx.talker_prefill import MlxTalkerPrefillBuilder
from sglang_omni.models.qwen3_omni.mlx.runner import (
    make_qwen3_omni_thinker_mlx_runner_class,
    make_qwen3_omni_talker_mlx_runner_class,
)
from sglang_omni.models.qwen3_omni import stages, bootstrap
assert make_qwen3_omni_thinker_mlx_runner_class().component == "language"
assert make_qwen3_omni_talker_mlx_runner_class().component == "talker"
assert not any(name.split(".")[0] in {"mlx_vlm", "mlx_audio"} for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "SGLANG_USE_MLX": "1"},
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
