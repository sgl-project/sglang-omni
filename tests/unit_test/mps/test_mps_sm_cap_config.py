# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
from pydantic import ValidationError

from sglang_omni.config.patch import (
    ConfigPatch,
    ConfigPatchSet,
    ConfigSource,
    SourceKind,
)
from sglang_omni.config.resolver import ConfigResolver
from sglang_omni.config.schema import PipelineConfig, ProcessConfig, StageConfig
from sglang_omni.config.topology import compile_logical_processes


def config(*, mps="auto", gpu=0, tp_size=1, cap=80):
    return PipelineConfig(
        model_path="dummy",
        mps=mps,
        stages=[
            StageConfig(
                name="vocoder",
                process="audio",
                factory_path="unused.factory",
                gpu=gpu,
                tp_size=tp_size,
                terminal=True,
            )
        ],
        processes={"audio": ProcessConfig(sm_cap=cap)},
    )


def test_sm_cap_cli_patch_reaches_process_policy():
    source = ConfigSource(SourceKind.CLI_FLAG, "--processes.audio.sm_cap")
    resolved = (
        ConfigResolver(config(cap=None))
        .resolve(
            ConfigPatchSet([ConfigPatch.create("processes.audio.sm_cap", 80, source)])
        )
        .config
    )
    assert resolved.processes["audio"].sm_cap == 80
    plan, _ = compile_logical_processes(resolved)
    assert plan.get("audio").sm_cap == 80


@pytest.mark.parametrize("value", [0, -8, 2.5])
def test_sm_cap_requires_positive_integer(value):
    with pytest.raises(ValidationError, match="sm_cap"):
        ProcessConfig(sm_cap=value)


def test_sm_cap_requires_mps_in_pipeline_config():
    with pytest.raises(ValueError, match="sm_cap.*requires MPS"):
        config(mps="off")


@pytest.mark.parametrize("gpu,tp_size", [(None, 1), ([0, 1], 2)])
def test_sm_cap_rejects_ineligible_logical_process(gpu, tp_size):
    with pytest.raises(ValueError, match="sm_cap.*GPU.*non-TP"):
        compile_logical_processes(config(gpu=gpu, tp_size=tp_size))
