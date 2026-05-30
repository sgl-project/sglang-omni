# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import torch


SCRIPT = Path(__file__).with_name("parity_compare.py")
HARNESS = Path(__file__).with_name("_encoder_parity_harness.py")


def _dump(
    path: Path,
    image_embeds: torch.Tensor,
    layers: dict[str, torch.Tensor] | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    with open(path, "wb") as f:
        payload = {"image_embeds": image_embeds, "deepstack": None}
        if extra:
            payload.update(extra)
        if layers is not None:
            payload["layers"] = layers
        pickle.dump(payload, f)


def test_tp_gate_passes_directional_match(tmp_path: Path) -> None:
    left = torch.eye(4, dtype=torch.float16)
    right = left.clone()
    right[0, 0] += 1e-4
    left_path = tmp_path / "left.pkl"
    right_path = tmp_path / "right.pkl"
    _dump(left_path, left)
    _dump(right_path, right)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(left_path), str(right_path), "--tp"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "TP parity: PASS" in result.stdout


def test_tp_gate_fails_low_cosine(tmp_path: Path) -> None:
    left_path = tmp_path / "left.pkl"
    right_path = tmp_path / "right.pkl"
    _dump(left_path, torch.eye(4, dtype=torch.float16))
    _dump(right_path, -torch.eye(4, dtype=torch.float16))

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(left_path), str(right_path), "--tp"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "TP parity: FAIL" in result.stdout


def test_layer_compare_skips_shard_local_shape_mismatch(tmp_path: Path) -> None:
    left_path = tmp_path / "left.pkl"
    right_path = tmp_path / "right.pkl"
    _dump(
        left_path,
        torch.eye(4, dtype=torch.float16),
        layers={
            "blk_00": torch.zeros((2, 4), dtype=torch.float16),
            "blk_00.attn.qkv_proj": torch.zeros((2, 8), dtype=torch.float16),
        },
    )
    _dump(
        right_path,
        torch.eye(4, dtype=torch.float16),
        layers={
            "blk_00": torch.zeros((2, 4), dtype=torch.float16),
            "blk_00.attn.qkv_proj": torch.zeros((2, 4), dtype=torch.float16),
        },
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(left_path),
            str(right_path),
            "--layers",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "skipped shape-mismatched shard-local tensors" in result.stdout
    assert "blk_00.attn.qkv_proj" in result.stdout


def test_compare_reports_video_audio_and_deepstack_labels(tmp_path: Path) -> None:
    left_path = tmp_path / "left.pkl"
    right_path = tmp_path / "right.pkl"
    base = torch.eye(4, dtype=torch.float16)
    _dump(
        left_path,
        base,
        extra={
            "video_embeds": base.clone(),
            "audio_embeds": base.clone(),
            "deepstack_visual_embeds_video": [base.clone()],
        },
    )
    _dump(
        right_path,
        base.clone(),
        extra={
            "video_embeds": base.clone(),
            "audio_embeds": base.clone(),
            "deepstack_visual_embeds_video": [base.clone()],
        },
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(left_path), str(right_path), "--tp"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "image_embeds:" in result.stdout
    assert "video_embeds:" in result.stdout
    assert "audio_embeds:" in result.stdout
    assert "deepstack_visual_embeds_video[0]:" in result.stdout


def test_tp_gate_fails_low_audio_cosine(tmp_path: Path) -> None:
    left_path = tmp_path / "left.pkl"
    right_path = tmp_path / "right.pkl"
    base = torch.eye(4, dtype=torch.float16)
    _dump(left_path, base, extra={"audio_embeds": base})
    _dump(right_path, base, extra={"audio_embeds": -base})

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(left_path),
            str(right_path),
            "--tp",
            "--labels",
            "audio_embeds",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "audio_embeds: mean cosine" in result.stdout


def test_harness_help_lists_modality() -> None:
    result = subprocess.run(
        [sys.executable, str(HARNESS), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "--modality" in result.stdout
    assert "image" in result.stdout
    assert "video" in result.stdout
    assert "audio" in result.stdout
