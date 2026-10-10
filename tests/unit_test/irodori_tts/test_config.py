# SPDX-License-Identifier: Apache-2.0
"""Architecture resolution for Irodori checkpoint layouts."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from sglang_omni.config.manager import resolve_config_cls_for_model_path
from sglang_omni.models.irodori_tts.config import IrodoriTTSPipelineConfig


def write_checkpoint_metadata(path: Path, *, text_model_type: str) -> None:
    metadata = {
        "config_json": json.dumps(
            {
                "text_encoder_type": "pretrained",
                "use_duration_predictor": True,
                "latent_dim": 32,
            }
        ),
        "text_encoder_config_json": json.dumps({"model_type": text_model_type}),
    }
    save_file({"weight": torch.zeros(1)}, str(path), metadata=metadata)


def test_local_checkpoint_metadata_resolves_irodori_pipeline(tmp_path: Path) -> None:
    write_checkpoint_metadata(
        tmp_path / "model.safetensors", text_model_type="modernbert"
    )

    assert (
        resolve_config_cls_for_model_path(str(tmp_path))
        is IrodoriTTSPipelineConfig
    )


def test_unrelated_checkpoint_metadata_does_not_resolve_as_irodori(
    tmp_path: Path,
) -> None:
    write_checkpoint_metadata(tmp_path / "model.safetensors", text_model_type="bert")

    with pytest.raises(ValueError, match="Could not resolve model architecture"):
        resolve_config_cls_for_model_path(str(tmp_path))
