# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.config.runtime import resolve_factory_signature_args
from sglang_omni.config.schema import EndpointsConfig, ProcessConfig
from sglang_omni.models.higgs_tts import stages
from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig
from sglang_omni.pipeline.mp_runner import _build_stage_groups
from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime
from sglang_omni.utils.imports import import_string
from tests.unit_test.fixtures.pipeline_fakes import FakeMpContext


def _same_gpu_frontend_replica_config(tmp_path) -> HiggsTtsPipelineConfig:
    base = HiggsTtsPipelineConfig(model_path="model")
    stages_cfg = [stage.model_copy(deep=True) for stage in base.stages]
    tts_engine = next(stage for stage in stages_cfg if stage.name == "tts_engine")
    tts_engine.runtime.resources.total_gpu_memory_fraction = 0.82
    return HiggsTtsPipelineConfig(
        model_path="model",
        stages=stages_cfg,
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
        processes={
            "tts_frontend": ProcessConfig(
                num_replicas=2,
                replica_devices=[0, 0],
            )
        },
    )


def test_higgs_frontend_replicas_inject_same_gpu_id(tmp_path) -> None:
    config = _same_gpu_frontend_replica_config(tmp_path)
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

    by_group = {group.group_name: group for group in groups}
    for replica_id in range(2):
        suffix = f"@r{replica_id}"
        specs = by_group[f"tts_frontend{suffix}"].specs
        assert [spec.stage_name for spec in specs] == [
            f"preprocessing{suffix}",
            f"audio_encoder{suffix}",
        ]

        preprocessing, audio_encoder = specs
        assert preprocessing.require_factory_gpu_id is False
        assert audio_encoder.require_factory_gpu_id is True
        factory_args = resolve_factory_signature_args(
            import_string(audio_encoder.factory),
            audio_encoder.factory_args,
            defaults=audio_encoder.factory_arg_defaults,
            require_gpu_id=audio_encoder.require_factory_gpu_id,
            stage_name=audio_encoder.stage_name,
        )
        assert factory_args["device"] == "cuda"
        assert factory_args["gpu_id"] == 0

    gpu_plan = prep.placement_plan.gpus[0]
    assert gpu_plan.total_gpu_memory_fraction == pytest.approx(0.98)


def test_higgs_audio_encoder_resolves_placement_gpu_id(monkeypatch) -> None:
    resolved: list[tuple[str, int | None]] = []
    codec_loads: list[tuple[str, str, str]] = []

    def resolve(device: str, gpu_id: int | None) -> str:
        resolved.append((device, gpu_id))
        return f"{device}:{gpu_id}"

    class FakeAdapter:
        def __init__(self, _tokenizer) -> None:
            pass

    class FakeCodec:
        SAMPLE_RATE = 8

        def __init__(self) -> None:
            self.model = SimpleNamespace(acoustic_encoder=object())

        def encode_reference(self, _waveform, *, sample_rate: int) -> torch.Tensor:
            assert sample_rate == self.SAMPLE_RATE
            return torch.zeros((1, 8), dtype=torch.long)

    fake_codec = FakeCodec()

    def load_codec(checkpoint: str, device: str, dtype: str) -> FakeCodec:
        codec_loads.append((checkpoint, device, dtype))
        return fake_codec

    monkeypatch.setattr(stages, "resolve_device_spec", resolve)
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda model_path: model_path)
    monkeypatch.setattr(stages.Tokenizer, "from_file", lambda _path: object())
    monkeypatch.setattr(
        stages,
        "PreTrainedTokenizerFast",
        lambda tokenizer_object: tokenizer_object,
    )
    monkeypatch.setattr(stages, "HiggsTokenizerAdapter", FakeAdapter)
    monkeypatch.setattr(stages, "get_or_load_codec", load_codec)
    monkeypatch.setattr(stages.torch, "compile", lambda module, **_kwargs: module)

    stages.create_audio_encoder_executor(
        "model",
        device="cuda",
        gpu_id=3,
    )

    assert resolved == [("cuda", 3)]
    assert codec_loads == [("model", "cuda:3", "bfloat16")]
