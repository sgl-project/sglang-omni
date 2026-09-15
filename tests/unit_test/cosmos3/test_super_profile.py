# SPDX-License-Identifier: Apache-2.0
"""Keep the checked-in full-Super qualification profile reproducible."""

from pathlib import Path

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.cosmos3.config import Cosmos3PipelineConfig

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROFILE = _REPO_ROOT / "examples/configs/cosmos3_super_t2i_1xa100_bf16.yaml"
_REVISION = "fe77b66696d645f663b8f27e942b3b43e4629e23"


def test_super_profile_pins_full_checkpoint_and_bf16_offload_topology():
    config = ConfigManager.from_file(str(_PROFILE)).config
    assert isinstance(config, Cosmos3PipelineConfig)
    assert config.model_path == f"nvidia/Cosmos3-Super@{_REVISION}"
    assert len(config.stages) == 1
    stage = config.stages[0]
    assert stage.gpu == 0
    assert stage.runtime_gpu_ids == [0]
    overrides = stage.factory.server_args_overrides
    assert overrides["performance_mode"] == "memory"
    assert overrides["layerwise_offload_components"] == ["dit"]
    assert overrides["dit_layerwise_resident_layers"] == 24
    assert overrides["enable_torch_compile"] is False
