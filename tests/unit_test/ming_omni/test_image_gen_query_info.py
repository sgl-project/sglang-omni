# SPDX-License-Identifier: Apache-2.0
"""Lightweight image-generation query-token metadata tests."""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

_QUERY_INFO_PATH = (
    Path(__file__).resolve().parents[3]
    / "sglang_omni"
    / "models"
    / "ming_omni"
    / "diffusion"
    / "query_info.py"
)


def _load_query_info_module():
    module_name = "_ming_omni_query_info_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, _QUERY_INFO_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _write_f32_safetensors(path, name_or_tensors, tensor=None) -> None:
    tensors = (
        name_or_tensors
        if isinstance(name_or_tensors, dict)
        else {name_or_tensors: tensor}
    )
    header = {}
    raw_parts = []
    offset = 0
    for name, tensor_value in tensors.items():
        tensor_value = tensor_value.contiguous().float().cpu()
        values = [float(value) for value in tensor_value.flatten().tolist()]
        raw = struct.pack("<" + "f" * tensor_value.numel(), *values)
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor_value.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        raw_parts.append(raw)
        offset += len(raw)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * padding
    path.write_bytes(
        struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(raw_parts)
    )


def test_load_image_gen_query_info_reads_query_tokens(tmp_path) -> None:
    torch = pytest.importorskip("torch")

    model_dir = tmp_path / "model"
    mlp_dir = model_dir / "mlp"
    mlp_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps({"llm_config": {"image_patch_token": 22}}),
        encoding="utf-8",
    )
    _write_f32_safetensors(
        mlp_dir / "model.safetensors",
        "query_tokens",
        torch.arange(12, dtype=torch.float32).reshape(3, 4),
    )

    query_info = _load_query_info_module()

    info = query_info.load_image_gen_query_info(str(model_dir))

    assert info.image_patch_token_id == 22
    assert info.query_tokens.shape == (3, 4)
    assert info.query_tokens.dtype == torch.bfloat16
    assert info.query_tokens[2, 3].item() == pytest.approx(11.0)


def test_load_image_gen_query_info_reads_real_layout_in_config_order(
    tmp_path,
) -> None:
    torch = pytest.importorskip("torch")

    model_dir = tmp_path / "model"
    mlp_dir = model_dir / "mlp"
    mlp_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps({"llm_config": {"image_patch_token": 33}}),
        encoding="utf-8",
    )
    (mlp_dir / "config.json").write_text(
        json.dumps({"img_gen_scales": [1, 2]}),
        encoding="utf-8",
    )
    _write_f32_safetensors(
        mlp_dir / "model.safetensors",
        {
            "query_tokens_dict.2x2": torch.arange(20, 32, dtype=torch.float32).reshape(
                4, 3
            ),
            "query_tokens_dict.1x1": torch.arange(10, 13, dtype=torch.float32).reshape(
                1, 3
            ),
        },
    )

    query_info = _load_query_info_module()

    info = query_info.load_image_gen_query_info(str(model_dir))

    assert info.image_patch_token_id == 33
    assert info.query_tokens.shape == (5, 3)
    assert info.query_tokens.dtype == torch.bfloat16
    assert info.query_tokens.device.type == "cpu"
    torch.testing.assert_close(
        info.query_tokens.float(),
        torch.tensor(
            [
                [10, 11, 12],
                [20, 21, 22],
                [23, 24, 25],
                [26, 27, 28],
                [29, 30, 31],
            ],
            dtype=torch.float32,
        ),
    )


def test_load_image_gen_query_info_raises_for_missing_query_tokens(tmp_path) -> None:
    (tmp_path / "mlp").mkdir()
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    query_info = _load_query_info_module()

    with pytest.raises(FileNotFoundError, match="query token"):
        query_info.load_image_gen_query_info(str(tmp_path))


def test_load_image_gen_query_info_raises_for_missing_real_layout_config(
    tmp_path,
) -> None:
    torch = pytest.importorskip("torch")

    mlp_dir = tmp_path / "mlp"
    mlp_dir.mkdir()
    (tmp_path / "config.json").write_text(
        json.dumps({"llm_config": {"image_patch_token": 22}}),
        encoding="utf-8",
    )
    _write_f32_safetensors(
        mlp_dir / "model.safetensors",
        "other_tokens",
        torch.arange(4, dtype=torch.float32),
    )

    query_info = _load_query_info_module()

    with pytest.raises(FileNotFoundError, match="mlp.*/config.json"):
        query_info.load_image_gen_query_info(str(tmp_path))


def test_load_image_gen_query_info_raises_for_missing_query_token_tensor(
    tmp_path,
) -> None:
    torch = pytest.importorskip("torch")

    mlp_dir = tmp_path / "mlp"
    mlp_dir.mkdir()
    (mlp_dir / "config.json").write_text(
        json.dumps({"img_gen_scales": [2]}),
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"llm_config": {"image_patch_token": 22}}),
        encoding="utf-8",
    )
    _write_f32_safetensors(
        mlp_dir / "model.safetensors",
        "other_tokens",
        torch.arange(4, dtype=torch.float32),
    )

    query_info = _load_query_info_module()

    with pytest.raises(KeyError, match="query_tokens_dict.2x2.*other_tokens"):
        query_info.load_image_gen_query_info(str(tmp_path))
