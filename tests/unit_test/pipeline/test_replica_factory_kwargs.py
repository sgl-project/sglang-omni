# SPDX-License-Identifier: Apache-2.0
"""Model factory-hook coverage for process-replica instance stages."""

from __future__ import annotations

import pytest

from sglang_omni.config import EndpointsConfig, PipelineConfig, ProcessConfig
from sglang_omni.config.runtime import resolve_stage_factory_kwargs
from sglang_omni.models.dots_tts.config import DotsTTSPipelineConfig
from sglang_omni.models.fun_asr.config import FunASRPipelineConfig
from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig
from sglang_omni.models.moss_transcribe_diarize.config import (
    MossTranscribeDiarizePipelineConfig,
)
from sglang_omni.models.moss_tts_local.config import MossTTSLocalPipelineConfig
from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig
from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig
from sglang_omni.models.zonos2.config import Zonos2PipelineConfig
from sglang_omni.pipeline.mp_runner import _build_stage_groups
from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime
from tests.unit_test.fixtures.pipeline_fakes import FakeMpContext


@pytest.mark.parametrize(
    ("config", "stage_name"),
    [
        pytest.param(
            Qwen3TTSPipelineConfig(
                model_path="model",
                enable_deterministic_inference=True,
            ),
            "preprocessing",
            id="qwen3-tts",
        ),
        pytest.param(
            Qwen3OmniSpeechPipelineConfig(model_path="model"),
            "code2wav",
            id="qwen3-omni",
        ),
        pytest.param(
            DotsTTSPipelineConfig(model_path="model"),
            "preprocessing",
            id="dots-tts",
        ),
        pytest.param(
            HiggsTtsPipelineConfig(model_path="model"),
            "vocoder",
            id="higgs-tts",
        ),
        pytest.param(
            Zonos2PipelineConfig(model_path="model"),
            "tts_engine",
            id="zonos2",
        ),
        pytest.param(
            MossTranscribeDiarizePipelineConfig(model_path="model"),
            "asr",
            id="moss-transcribe-diarize",
        ),
        pytest.param(
            FunASRPipelineConfig(model_path="model"),
            "asr",
            id="fun-asr",
        ),
        pytest.param(
            MossTTSLocalPipelineConfig(model_path="model"),
            "preprocessing",
            id="moss-tts-local",
        ),
    ],
)
def test_model_factory_kwargs_survive_replica_instance_names(
    config: PipelineConfig,
    stage_name: str,
) -> None:
    logical_stage = config.stage_named(stage_name)
    instance_stage = logical_stage.model_copy(update={"name": f"{stage_name}@r0"})

    expected = config.stage_factory_kwargs(stage_name)
    assert expected
    assert resolve_stage_factory_kwargs(instance_stage, config) == expected


def test_qwen3_tts_replica_launch_specs_keep_deterministic_factory_kwargs(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sglang_omni.pipeline.runtime_config._visible_device_count",
        lambda: 2,
    )
    config = Qwen3TTSPipelineConfig(
        model_path="model",
        enable_deterministic_inference=True,
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
        processes={"pipeline": ProcessConfig(num_replicas=2, replica_devices=[0, 1])},
    )
    prep = prepare_pipeline_runtime(config)
    try:
        groups = _build_stage_groups(
            config,
            ctx=FakeMpContext(),
            stages_cfg=prep.stages_cfg,
            endpoints=prep.endpoints,
            placement_plan=prep.placement_plan,
            process_plan=prep.process_plan,
            replica_topology=prep.replica_topology,
        )
    finally:
        prep.runtime_dir.close()

    specs = {spec.stage_name: spec for group in groups for spec in group.specs}
    for replica_id in range(2):
        suffix = f"@r{replica_id}"
        assert specs[f"preprocessing{suffix}"].factory_kwargs["max_concurrency"] == 1
        assert specs[f"tts_engine{suffix}"].factory_kwargs["server_args_overrides"][
            "enable_deterministic_inference"
        ]
        vocoder_kwargs = specs[f"vocoder{suffix}"].factory_kwargs
        assert vocoder_kwargs["enable_deterministic_inference"]
        assert vocoder_kwargs["initial_cuda_graph"] is False
        assert vocoder_kwargs["followup_cuda_graph"] is False
