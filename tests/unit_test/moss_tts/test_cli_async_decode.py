# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from sglang_omni.cli.serve import apply_async_decode_cli_overrides
from sglang_omni.config import resolve_stage_factory_args
from sglang_omni.models.moss_tts.config import MossTTSPipelineConfig


def _tts_engine_args(config):
    stage = next(s for s in config.stages if s.name == "tts_engine")
    return resolve_stage_factory_args(stage, config)


def test_moss_async_decode_cli_override_can_enable_and_disable() -> None:
    config = MossTTSPipelineConfig(model_path="dummy")

    apply_async_decode_cli_overrides(
        config, async_decode="on", async_decode_min_batch_size=4
    )
    args = _tts_engine_args(config)
    assert args["enable_async_decode"] is True
    assert args["async_decode_min_batch_size"] == 4

    apply_async_decode_cli_overrides(
        config, async_decode="off", async_decode_min_batch_size=None
    )
    assert _tts_engine_args(config)["enable_async_decode"] is False
