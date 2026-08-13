# SPDX-License-Identifier: Apache-2.0
"""CLI override tests for prefill admission coalescing.

``--prefill-coalesce-requests/--prefill-coalesce-wait-ms`` apply to the
supported OmniScheduler AR stages; omitting both is a no-op and unsupported
pipelines are rejected.
"""

from __future__ import annotations

import pytest
import typer

from sglang_omni.cli.serve import apply_prefill_coalesce_cli_overrides
from sglang_omni.config import PipelineConfig
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import (
    resolve_factory_signature_args,
    resolve_stage_factory_arg_defaults,
    resolve_stage_static_factory_args,
)
from sglang_omni.models.fun_asr.config import FunASRPipelineConfig
from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig
from sglang_omni.models.moss_transcribe_diarize.config import (
    MossTranscribeDiarizePipelineConfig,
)
from sglang_omni.models.moss_tts_local.config import MossTTSLocalPipelineConfig
from sglang_omni.models.qwen3_asr.config import Qwen3ASRPipelineConfig
from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig
from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig
from sglang_omni.models.whisper_asr.config import WhisperASRPipelineConfig
from sglang_omni.utils.imports import import_string


def _ar_stage_args(config: PipelineConfig, stage_name: str) -> dict[str, object]:
    # Compose the resolver sequence the launch path uses -- mp_runner resolves
    # static args plus defaults in the parent, stage_workers filters by factory
    # signature in the child -- rather than the resolve_stage_factory_args
    # convenience wrapper, which nothing on that path calls.
    stage = next(s for s in config.stages if s.name == stage_name)
    return resolve_factory_signature_args(
        import_string(stage.factory),
        resolve_stage_static_factory_args(stage, config),
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
        (WhisperASRPipelineConfig, "asr"),
        (Qwen3OmniPipelineConfig, "thinker"),
    ],
)
def test_cli_sets_coalesce_args(config_cls, stage_name):
    config = config_cls(model_path="dummy")
    apply_prefill_coalesce_cli_overrides(
        config, prefill_coalesce_requests=32, prefill_coalesce_wait_ms=300.0
    )
    args = _ar_stage_args(config, stage_name)
    assert args["prefill_coalesce_requests"] == 32
    assert args["prefill_coalesce_wait_ms"] == 300.0


def test_omitting_both_flags_is_a_noop():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    before = _ar_stage_args(config, "tts_engine")
    apply_prefill_coalesce_cli_overrides(
        config, prefill_coalesce_requests=None, prefill_coalesce_wait_ms=None
    )
    assert _ar_stage_args(config, "tts_engine") == before


def test_per_stage_yaml_runtime_overrides_reach_factory(tmp_path):
    config_path = tmp_path / "moss_local.yaml"
    config_path.write_text(
        """
config_cls: MossTTSLocalPipelineConfig
model_path: dummy
runtime_overrides:
  tts_engine:
    prefill_coalesce_requests: 16
    prefill_coalesce_wait_ms: 200.0
"""
    )
    config = ConfigManager.from_file(str(config_path)).config
    assert config is not None

    args = _ar_stage_args(config, "tts_engine")
    assert args["prefill_coalesce_requests"] == 16
    assert args["prefill_coalesce_wait_ms"] == 200.0


def test_wait_only_cli_respects_per_stage_runtime_override(caplog):
    config = MossTTSLocalPipelineConfig(model_path="dummy")
    config.runtime_overrides["tts_engine"] = {"prefill_coalesce_requests": 16}

    with caplog.at_level("WARNING", logger="sglang_omni.cli.serve"):
        apply_prefill_coalesce_cli_overrides(
            config,
            prefill_coalesce_requests=None,
            prefill_coalesce_wait_ms=200.0,
        )

    assert not any(
        "alone does not enable" in record.message for record in caplog.records
    )
    args = _ar_stage_args(config, "tts_engine")
    assert args["prefill_coalesce_requests"] == 16
    assert args["prefill_coalesce_wait_ms"] == 200.0


def test_wait_only_cli_validates_existing_per_stage_value():
    config = MossTTSLocalPipelineConfig(model_path="dummy")
    config.runtime_overrides["tts_engine"] = {"prefill_coalesce_requests": "bad"}

    with pytest.raises(typer.BadParameter, match="must be a number"):
        apply_prefill_coalesce_cli_overrides(
            config,
            prefill_coalesce_requests=None,
            prefill_coalesce_wait_ms=200.0,
        )


def test_rejects_unsupported_pipeline():
    config = Qwen3TTSPipelineConfig(model_path="dummy")
    with pytest.raises(typer.BadParameter):
        apply_prefill_coalesce_cli_overrides(
            config, prefill_coalesce_requests=32, prefill_coalesce_wait_ms=None
        )


def test_rejects_invalid_values():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    with pytest.raises(typer.BadParameter):
        apply_prefill_coalesce_cli_overrides(
            config, prefill_coalesce_requests=-1, prefill_coalesce_wait_ms=None
        )
    for bad_wait in (0.0, float("nan"), float("inf")):
        with pytest.raises(typer.BadParameter):
            apply_prefill_coalesce_cli_overrides(
                config,
                prefill_coalesce_requests=None,
                prefill_coalesce_wait_ms=bad_wait,
            )
