# SPDX-License-Identifier: Apache-2.0
"""Reject conflicting worker environment configuration before runtime planning."""

import pytest
from pydantic import ValidationError

from sglang_omni.config.patch import (
    ConfigPatch,
    ConfigPatchSet,
    ConfigSource,
    SourceKind,
)
from sglang_omni.config.resolver import ConfigResolver

_ENV = {
    "CUDA_VISIBLE_DEVICES": "1",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_MPS_PIPE_DIRECTORY": "/external/mps",
    "SGLANG_OMNI_WEIGHT_SHARE": "leader:/external/weights",
}


@pytest.mark.parametrize("scope", ["env_defaults", "stages.preprocessing.env"])
@pytest.mark.parametrize(
    "mps,weight_share,name",
    [(mode, "off", name) for mode in ("on", "auto") for name in _ENV]
    + [("off", "on", "SGLANG_OMNI_WEIGHT_SHARE")],
)
def test_config_rejects_managed_runtime_env_conflicts(
    pipeline_config, scope, mps, weight_share, name
):
    data = pipeline_config.model_dump()
    data.update(mps=mps, weight_share=weight_share)
    env = data["env_defaults"] if scope == "env_defaults" else data["stages"][0]["env"]
    env[name] = _ENV[name]

    with pytest.raises(ValidationError) as exc:
        type(pipeline_config).model_validate(data)

    assert f"{scope}.{name}" in str(exc.value)
    assert f"mps={mps}, weight_share={weight_share}" in str(exc.value)


@pytest.mark.parametrize("scope", ["env_defaults", "stages.preprocessing.env"])
@pytest.mark.parametrize("feature", ["mps", "weight_share"])
@pytest.mark.parametrize("patch_feature", [True, False])
def test_resolver_rejects_conflict_introduced_by_either_patch(
    pipeline_config, scope, feature, patch_feature
):
    env_path = f"{scope}.SGLANG_OMNI_WEIGHT_SHARE"
    source = ConfigSource(SourceKind.CLI_DOTTED, "command line")
    feature_patch = ConfigPatch.create(feature, "on", source)
    env_patch = ConfigPatch.create(env_path, "leader:/external/weights", source)
    baseline, conflict = (
        (env_patch, feature_patch) if patch_feature else (feature_patch, env_patch)
    )
    base = ConfigResolver(pipeline_config).resolve(ConfigPatchSet([baseline])).config

    with pytest.raises(ValidationError) as exc:
        ConfigResolver(base).resolve(ConfigPatchSet([conflict]))
    assert env_path in str(exc.value)


@pytest.mark.parametrize("mps", ["on", "auto"])
def test_managed_mps_and_weight_share_can_be_enabled_together(pipeline_config, mps):
    data = pipeline_config.model_dump()
    data.update(mps=mps, weight_share="on", env_defaults={"OMP_NUM_THREADS": "2"})
    data["stages"][0]["env"] = {"OMP_NUM_THREADS": "4"}

    config = type(pipeline_config).model_validate(data)
    assert config.mps == mps
    assert config.weight_share == "on"


def test_unmanaged_runtime_keeps_external_environment_configuration(pipeline_config):
    data = pipeline_config.model_dump()
    data["env_defaults"] = dict(_ENV)
    data["stages"][0]["env"] = dict(_ENV)

    config = type(pipeline_config).model_validate(data)
    assert (config.mps, config.weight_share) == ("off", "off")
    assert config.env_defaults == _ENV
    assert config.stages[0].env == _ENV
