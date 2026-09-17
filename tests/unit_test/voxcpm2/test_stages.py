# SPDX-License-Identifier: Apache-2.0
"""Stage construction and device placement without loading model weights."""

from types import SimpleNamespace

import pytest

from sglang_omni.models.voxcpm2 import stages, streaming_vocoder


def _factory_dependencies(monkeypatch, *, fail_warmup=False):
    calls = []
    objects = []
    vae = object()
    monkeypatch.setattr(
        stages,
        "resolve_model_config",
        lambda _: ("checkpoint", SimpleNamespace(patch_size=4)),
    )

    def load_vae(checkpoint, config, *, device):
        calls.append(("load", device))
        return vae

    class Encoder:
        def __init__(self, model, **kwargs):
            assert model is vae
            objects.append(self)
            self.kwargs = kwargs

        def warmup(self):
            calls.append(("warmup", "encoder"))
            if fail_warmup:
                raise RuntimeError("encoder warmup failed")

        def encode_payload(self, payload):
            return payload

    class Vocoder:
        def __init__(self, model, **kwargs):
            assert model is vae
            objects.append(self)
            self.kwargs = kwargs

        def warmup_now(self):
            calls.append(("warmup", "vocoder"))
            if fail_warmup:
                raise RuntimeError("vocoder warmup failed")

    def scheduler(fn, **kwargs):
        calls.append(("scheduler", kwargs))
        return SimpleNamespace(fn=fn, **kwargs)

    monkeypatch.setattr(stages, "load_audio_vae", load_vae)
    monkeypatch.setattr(stages, "VoxCPM2ReferenceEncoder", Encoder)
    monkeypatch.setattr(streaming_vocoder, "VoxCPM2StreamingVocoder", Vocoder)
    monkeypatch.setattr(stages, "SimpleScheduler", scheduler)
    return calls, objects


@pytest.mark.parametrize("kind", ["reference_encode", "vocoder"])
def test_factory_warms_before_returning_a_scheduler(monkeypatch, kind):
    calls, objects = _factory_dependencies(monkeypatch)
    factory = getattr(stages, f"create_{kind}_executor")
    result = factory("model", device="cpu")
    assert calls[:2] == [
        ("load", "cpu"),
        ("warmup", "encoder" if kind == "reference_encode" else "vocoder"),
    ]
    if kind == "reference_encode":
        assert calls[2] == ("scheduler", {"max_concurrency": 8})
        assert result.fn.__self__ is objects[0]
    else:
        assert result is objects[0]
        assert len(calls) == 2


@pytest.mark.parametrize("kind", ["reference_encode", "vocoder"])
def test_factory_propagates_warmup_failure_without_returning_ready(monkeypatch, kind):
    calls, _ = _factory_dependencies(monkeypatch, fail_warmup=True)
    with pytest.raises(RuntimeError, match="warmup failed"):
        getattr(stages, f"create_{kind}_executor")("model", device="cpu")
    assert not any(event == "scheduler" for event, _ in calls)


@pytest.mark.parametrize("kind", ["reference_encode", "vocoder"])
@pytest.mark.parametrize("gpu_id", [0, 3])
def test_audio_factory_passes_placement_to_weight_loading(monkeypatch, kind, gpu_id):
    calls, objects = _factory_dependencies(monkeypatch)
    getattr(stages, f"create_{kind}_executor")("model", device="cuda", gpu_id=gpu_id)
    assert calls[0] == ("load", f"cuda:{gpu_id}")
    if kind == "vocoder":
        assert objects[0].kwargs["device"] == f"cuda:{gpu_id}"


def test_engine_factory_forwards_placement_and_overrides(monkeypatch):
    from sglang_omni.models.voxcpm2 import engine_builder

    built = []
    result = object()

    class Builder:
        def __init__(self, **kwargs):
            self.options = kwargs

        def build(self, model_path, **kwargs):
            built.append((model_path, self.options, kwargs))
            return result

    monkeypatch.setattr(engine_builder, "VoxCPM2EngineBuilder", Builder)
    overrides = {"disable_cuda_graph": True}
    assert (
        stages.create_tts_engine_executor(
            "model", gpu_id=3, max_running_requests=2, server_args_overrides=overrides
        )
        is result
    )
    assert built[0][2]["gpu_id"] == 3
    assert built[0][2]["device"] == "cuda"
    assert built[0][2]["server_args_overrides"] is overrides
    assert built[0][1]["max_running_requests"] == 2


@pytest.mark.parametrize("devices", [[0, 0], [2, 5]])
def test_replica_plan_injects_each_gpu_into_all_audio_factories(tmp_path, devices):
    from sglang_omni.config.runtime import (
        apply_typed_stage_kwargs,
        resolve_factory_signature_args,
    )
    from sglang_omni.config.schema import EndpointsConfig, ProcessConfig
    from sglang_omni.models.voxcpm2.config import VoxCPM2PipelineConfig
    from sglang_omni.pipeline.mp_runner import _build_stage_groups
    from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime
    from sglang_omni.utils.imports import import_string
    from tests.unit_test.fixtures.pipeline_fakes import FakeMpContext

    config = VoxCPM2PipelineConfig(model_path="model")
    stage_configs = [
        (
            stage.model_copy(update={"gpu_memory_fraction": 0.1})
            if stage.gpu is not None
            else stage.model_copy(deep=True)
        )
        for stage in config.stages
    ]
    config = VoxCPM2PipelineConfig(
        model_path="model",
        stages=stage_configs,
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
        processes={"pipeline": ProcessConfig(num_replicas=2, replica_devices=devices)},
    )
    prepared = prepare_pipeline_runtime(config)
    try:
        groups = _build_stage_groups(
            config,
            ctx=FakeMpContext(),
            stages_cfg=prepared.stages_cfg,
            endpoints=prepared.endpoints,
            placement_plan=prepared.placement_plan,
            process_plan=prepared.process_plan,
            replica_topology=prepared.replica_topology,
        )
        by_name = {group.group_name: group for group in groups}
        for replica, device in enumerate(devices):
            specs = by_name[f"pipeline@r{replica}"].specs
            assert len(specs) == 4
            for spec in specs:
                factory = import_string(spec.factory)
                kwargs = apply_typed_stage_kwargs(
                    factory,
                    spec.factory_kwargs,
                    spec.typed_kwargs,
                    stage_name=spec.stage_name,
                )
                kwargs = resolve_factory_signature_args(
                    factory,
                    kwargs,
                    defaults=spec.factory_arg_defaults,
                    require_gpu_id=spec.require_factory_gpu_id,
                    stage_name=spec.stage_name,
                )
                if spec.stage_name.startswith("preprocessing"):
                    assert not spec.require_factory_gpu_id
                    assert "gpu_id" not in kwargs
                else:
                    assert spec.require_factory_gpu_id
                    assert kwargs["gpu_id"] == device
    finally:
        prepared.runtime_dir.close()


@pytest.mark.parametrize(
    "factory_name, option, value",
    [
        ("create_reference_encode_executor", "dtype", "bfloat16"),
        ("create_reference_encode_executor", "max_batch_size", 8),
        ("create_reference_encode_executor", "max_batch_wait_ms", 10),
        ("create_tts_engine_executor", "min_len", 2),
        ("create_tts_engine_executor", "max_len", 2000),
    ],
)
def test_factories_reject_unsupported_options(factory_name, option, value):
    factory = getattr(stages, factory_name)
    with pytest.raises(TypeError, match=option):
        factory("model", **{option: value})
