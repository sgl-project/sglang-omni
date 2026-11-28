# SPDX-License-Identifier: Apache-2.0
"""FishAudio S2-Pro RTX 4090 consumer profile contract."""

from __future__ import annotations

from pathlib import Path

from sglang_omni.config.manager import ConfigManager

CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "examples/configs/s2pro_tts_rtx4090.yaml"
)


def test_s2pro_rtx4090_profile_bounds_concurrency_and_graphs() -> None:
    config = ConfigManager.from_file(str(CONFIG_PATH)).config
    stages = {stage.name: stage for stage in config.stages}
    engine = stages["tts_engine"].engine
    preprocessing = stages["preprocessing"]

    assert config.model_path == "fishaudio/s2-pro"
    assert preprocessing.gpu is None
    assert preprocessing.total_reserve_bytes is None
    assert engine.max_running_requests == 4
    assert engine.mem_fraction_static == 0.65
    assert engine.cuda_graph_max_bs == 4
    assert list(engine.cuda_graph_bs) == [1, 2, 4]
    assert engine.enable_torch_compile is False
