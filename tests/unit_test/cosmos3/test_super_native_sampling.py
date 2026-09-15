# SPDX-License-Identifier: Apache-2.0
"""CPU compatibility checks against installed native Cosmos3 sampling code."""

import json
from types import SimpleNamespace

import pytest

from sglang_omni.models.cosmos3.stages import build_sampling_params
from sglang_omni.proto import OmniRequest, StagePayload
from tests.integration.cosmos3.test_super_gpu import _generation_case


@pytest.fixture(scope="module")
def native_sampling():
    cosmos = pytest.importorskip("sglang.multimodal_gen.configs.sample.cosmos3")
    from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import Cosmos3Config
    from sglang.multimodal_gen.configs.sample.sampling_params import (
        DataType,
        SamplingParams,
    )

    return cosmos, Cosmos3Config, DataType, SamplingParams


@pytest.mark.parametrize(
    "mode",
    [
        "t2i",
        "t2v",
        "i2v",
        "v2v",
        "sound",
        "policy",
        "inverse_dynamics",
        "forward_dynamics",
    ],
)
def test_gpu_cases_use_native_sampling_without_visual_default_overrides(
    monkeypatch, tmp_path, mode, native_sampling
):
    cosmos, Cosmos3Config, DataType, SamplingParams = native_sampling
    config = Cosmos3Config()
    args = SimpleNamespace(
        model_id=None,
        model_path=str(tmp_path),
        served_model_name="super",
        backend=None,
        pipeline_class_name=None,
        output_path=str(tmp_path),
        comfyui_mode=False,
        num_gpus=4,
        pipeline_config=config,
    )
    # Avoid downloading weights. The native sampling constructor/merge/adjust
    # methods below are real, including explicit-field tracking.
    monkeypatch.setattr(
        SamplingParams,
        "from_pretrained",
        lambda *a, **kw: cosmos.Cosmos3SamplingParams(),
    )
    inputs = _generation_case(mode, tmp_path)
    params = build_sampling_params(
        StagePayload("native", OmniRequest(inputs), None), str(tmp_path)
    )
    sampling = SamplingParams.from_user_sampling_params_args(
        str(tmp_path), args, **params
    )
    for key in (
        "seed",
        "num_frames",
        "guidance_scale",
        "num_inference_steps",
        "width",
        "height",
        "fps",
    ):
        assert getattr(sampling, key) == inputs[key]
    if mode in ("policy", "inverse_dynamics"):
        assert sampling.data_type == DataType.ACTION
        assert not sampling.save_output and not sampling.return_file_paths_only
        assert not sampling.use_system_prompt and not sampling.use_duration_template
        from sglang.multimodal_gen.runtime.entrypoints.action.protocol import (
            action_generation_response,
        )

        response = action_generation_response(
            {"actions": [[0.0] * 9] * 16, "action_mode": mode, "raw_action_dim": 9},
            args,
        )
        assert json.loads(json.dumps(response))["data"][0]["action"]["shape"] == [16, 9]
    elif mode == "sound":
        assert sampling.sound_duration == inputs["sound_duration"]
    elif mode == "v2v":
        assert sampling.video_path == inputs["video_path"]
        assert sampling.condition_frame_indexes == [0, 1]
