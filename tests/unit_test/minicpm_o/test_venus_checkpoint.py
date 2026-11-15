# SPDX-License-Identifier: Apache-2.0
"""Realtime-Venus checkpoints keep their configuration in a root manifest and serve on the MiniCPM-o pipelines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import pytest
import yaml
from transformers import AutoConfig

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.minicpm_o.config import MiniCPMOSpeechPipelineConfig
from sglang_omni.models.minicpm_o.hf_config import (
    MiniCPMOConfig,
    register_minicpm_o_hf_config,
)

VenusVariant = Literal["omni", "audio"]

VENUS_ARCHITECTURES: dict[VenusVariant, str] = {
    "omni": "RealtimeVenusOmni",
    "audio": "MiniCPMO",
}
VENUS_MODEL_TYPES: dict[VenusVariant, str] = {
    "omni": "realtime_venus_omni",
    "audio": "minicpmo",
}


def write_venus_checkpoint(root: Path, variant: VenusVariant) -> Path:
    """Lay out a Venus download: a stub config.json whose fields live in the root config.yaml."""
    hf_config = {
        "model_type": VENUS_MODEL_TYPES[variant],
        "architectures": [VENUS_ARCHITECTURES[variant]],
        "attention_bias": False,
        "hidden_size": 64,
        "vocab_size": 151752,
    }
    manifest_bytes = yaml.safe_dump(
        {"models": {variant: {"hf_config": hf_config}}}
    ).encode()
    (root / "config.yaml").write_bytes(manifest_bytes)
    model_directory = root / f"Realtime-Venus-{variant.capitalize()}"
    model_directory.mkdir()
    stub_config = {
        "model_type": VENUS_MODEL_TYPES[variant],
        "architectures": [VENUS_ARCHITECTURES[variant]],
        "auto_map": {"AutoConfig": "configuration_minicpmo.MiniCPMOConfig"},
        "venus_config_ref": {
            "model": variant,
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
    }
    (model_directory / "config.json").write_text(json.dumps(stub_config))
    return model_directory


def test_venus_omni_directory_resolves_to_the_minicpm_o_pipeline(
    tmp_path: Path,
) -> None:
    """A Realtime-Venus-Omni directory names its architecture RealtimeVenusOmni and selects the MiniCPM-o pipeline."""
    model_directory = write_venus_checkpoint(tmp_path, "omni")
    config_manager = ConfigManager.from_model_path(str(model_directory))
    assert isinstance(config_manager.config, MiniCPMOSpeechPipelineConfig)


@pytest.mark.parametrize("variant", ["omni", "audio"])
@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_venus_stub_config_loads_its_fields_from_the_root_manifest(
    variant: VenusVariant, trust_remote_code: bool, tmp_path: Path
) -> None:
    """A Venus config.json holds only a venus_config_ref; the backbone fields come from the manifest beside the checkpoint, also when remote code is trusted, since the checkpoint's own config code is not run."""
    model_directory = write_venus_checkpoint(tmp_path, variant)
    register_minicpm_o_hf_config()
    config = AutoConfig.from_pretrained(
        model_directory, trust_remote_code=trust_remote_code
    )
    assert config.architectures == [VENUS_ARCHITECTURES[variant]]
    assert config.vocab_size == 151752
    assert config.get_text_config().hidden_size == 64


@pytest.mark.parametrize("variant", ["omni", "audio"])
def test_venus_stub_config_expands_on_a_direct_component_load(
    variant: VenusVariant, tmp_path: Path
) -> None:
    """The encoders load the shim class directly, without AutoConfig, and still need the manifest's fields."""
    model_directory = write_venus_checkpoint(tmp_path, variant)
    config = MiniCPMOConfig.from_pretrained(model_directory)
    assert config.vocab_size == 151752
    assert config.get_text_config().hidden_size == 64
