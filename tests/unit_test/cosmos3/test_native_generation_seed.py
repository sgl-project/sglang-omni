# SPDX-License-Identifier: Apache-2.0
"""Preserve the SDK seed when lowering a native generation request."""

import pytest

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateRequest, SamplingParams
from sglang_omni.models.cosmos3.stages import build_sampling_params
from sglang_omni.proto import StagePayload


@pytest.mark.parametrize(
    "prompt,stage_seed,diffusion,stage_params,expected",
    [
        ("a lake", None, {}, {}, 7),
        ({"prompt": "a lake", "seed": 8}, None, {}, {}, 8),
        ("a lake", 0, {}, {}, 0),
        ("a lake", 0, {"seed": 9}, {}, 9),
        ("a lake", 0, {"seed": 9}, {"seed": 10}, 10),
    ],
)
def test_sdk_seed_precedence(prompt, stage_seed, diffusion, stage_params, expected):
    request = GenerateRequest(
        prompt=prompt,
        stream=False,
        sampling=SamplingParams(seed=7),
        stage_sampling={"generation": SamplingParams(seed=stage_seed)},
        stage_params={"generation": stage_params},
        extra_params={"diffusion": diffusion},
    )
    native = build_sampling_params(
        StagePayload("owned", Client.build_omni_request(request), None), "outputs"
    )
    assert native["seed"] == expected
    assert "temperature" not in native


def test_absent_sdk_seed_keeps_native_default():
    request = GenerateRequest(prompt="a lake", stream=False)
    native = build_sampling_params(
        StagePayload("owned", Client.build_omni_request(request), None), "outputs"
    )
    assert "seed" not in native
