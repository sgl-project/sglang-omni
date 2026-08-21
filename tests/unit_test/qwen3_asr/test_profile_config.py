# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_static_factory_args
from sglang_omni.models.qwen3_asr.config import Qwen3ASRPipelineConfig

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_qwen3_asr_rtx5090_profile_resolves_validated_runtime() -> None:
    config = ConfigManager.from_file(
        str(_REPO_ROOT / "examples/configs/qwen3_asr_rtx5090.yaml")
    ).config

    assert isinstance(config, Qwen3ASRPipelineConfig)
    args = resolve_stage_static_factory_args(config.stages[0], config)
    assert args["dtype"] == "bfloat16"
    assert args["max_running_requests"] == 16
    server_args = args["server_args_overrides"]
    assert server_args["cuda_graph_max_bs"] == 16
    assert server_args["mem_fraction_static"] == 0.65
