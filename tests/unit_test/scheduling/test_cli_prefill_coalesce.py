# SPDX-License-Identifier: Apache-2.0
"""Prefill coalescing is plain per-stage configuration.

The knobs live in the stage's ``factory`` group, so they are set with the
same dotted spelling as everything else (``--<stage>.factory.prefill_coalesce_requests``)
or under the stage's ``stages:`` entry in YAML. These tests pin that the
values reach the AR stage factories of every supported pipeline, and that
the schema rejects nonsense eagerly.
"""

from __future__ import annotations

import pytest

from sglang_omni.config import PipelineConfig
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import (
    apply_typed_stage_kwargs,
    resolve_factory_signature_args,
    resolve_stage_factory_arg_defaults,
    resolve_stage_factory_kwargs,
    resolve_stage_typed_kwargs,
)
from sglang_omni.models.fun_asr.config import FunASRPipelineConfig
from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig
from sglang_omni.models.moss_transcribe_diarize.config import (
    MossTranscribeDiarizePipelineConfig,
)
from sglang_omni.models.moss_tts_local.config import MossTTSLocalPipelineConfig
from sglang_omni.models.qwen3_asr.config import Qwen3ASRPipelineConfig
from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig
from sglang_omni.utils.imports import import_string


def _ar_stage_args(config: PipelineConfig, stage_name: str) -> dict[str, object]:
    # Compose the resolver sequence the launch path uses -- mp_runner resolves
    # the two kwarg channels in the parent, stage_workers overlays them against
    # the factory signature in the child.
    stage = next(s for s in config.stages if s.name == stage_name)
    factory = import_string(stage.factory_path)
    args = apply_typed_stage_kwargs(
        factory,
        resolve_stage_factory_kwargs(stage, config),
        resolve_stage_typed_kwargs(stage),
        stage_name=stage_name,
    )
    return resolve_factory_signature_args(
        factory,
        args,
        defaults=resolve_stage_factory_arg_defaults(stage, config),
    )


@pytest.mark.parametrize(
    ("config_cls", "stage_name"),
    [
        (HiggsTtsPipelineConfig, "tts_engine"),
        (MossTTSLocalPipelineConfig, "tts_engine"),
        (MossTranscribeDiarizePipelineConfig, "asr"),
        (FunASRPipelineConfig, "asr"),
        (Qwen3ASRPipelineConfig, "asr"),
        (Qwen3OmniPipelineConfig, "thinker"),
    ],
)
def test_dotted_flags_set_coalesce_args(config_cls, stage_name):
    config = config_cls(model_path="dummy")
    merged = ConfigManager(config).merge_config(
        [
            (f"{stage_name}.factory.prefill_coalesce_requests", "32"),
            (f"{stage_name}.factory.prefill_coalesce_wait_ms", "300.0"),
        ]
    )
    args = _ar_stage_args(merged, stage_name)
    assert args["prefill_coalesce_requests"] == 32
    assert args["prefill_coalesce_wait_ms"] == 300.0


def test_per_stage_yaml_settings_reach_the_factory(tmp_path):
    config_path = tmp_path / "moss_local.yaml"
    config_path.write_text(
        """
config_cls: MossTTSLocalPipelineConfig
model_path: dummy
stages:
  tts_engine:
    factory:
      prefill_coalesce_requests: 16
      prefill_coalesce_wait_ms: 200.0
"""
    )
    config = ConfigManager.from_file(str(config_path)).config
    assert config is not None

    args = _ar_stage_args(config, "tts_engine")
    assert args["prefill_coalesce_requests"] == 16
    assert args["prefill_coalesce_wait_ms"] == 200.0


def test_a_flag_for_one_stage_leaves_the_other_settings_alone():
    config = MossTTSLocalPipelineConfig(model_path="dummy")
    before = _ar_stage_args(config, "tts_engine")

    merged = ConfigManager(config).merge_config(
        [("tts_engine.factory.prefill_coalesce_wait_ms", "200.0")]
    )

    after = _ar_stage_args(merged, "tts_engine")
    assert after["prefill_coalesce_wait_ms"] == 200.0
    after.pop("prefill_coalesce_wait_ms")
    before.pop("prefill_coalesce_wait_ms", None)
    assert after == before


def test_a_stage_without_the_knob_refuses_the_flag():
    """Writing the knob on a stage whose factory cannot take it fails at
    launch with the path named, instead of silently dropping the value."""
    config = MossTTSLocalPipelineConfig(model_path="dummy")
    merged = ConfigManager(config).merge_config(
        [("vocoder.factory.prefill_coalesce_requests", "32")]
    )
    with pytest.raises(ValueError, match="prefill_coalesce_requests"):
        _ar_stage_args(merged, "vocoder")


def test_rejects_invalid_values_eagerly():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    manager = ConfigManager(config)
    with pytest.raises(ValueError, match="prefill_coalesce_requests"):
        manager.merge_config([("tts_engine.factory.prefill_coalesce_requests", "-1")])
    with pytest.raises(ValueError, match="prefill_coalesce_wait_ms"):
        manager.merge_config([("tts_engine.factory.prefill_coalesce_wait_ms", "0")])
