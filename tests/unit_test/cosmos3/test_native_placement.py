# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from sglang_omni.config import StageConfig
from sglang_omni.config.placement import build_stage_placement_plan
from sglang_omni.models.cosmos3.config import Cosmos3PipelineConfig, Variants
from sglang_omni.models.cosmos3.stages import native_server_kwargs
from sglang_omni.pipeline import runtime_config
from sglang_omni.pipeline.mp_runner import build_stage_groups
from sglang_omni.pipeline.replicas import validate_device_assignment


def config(devices=(1, 3)):
    cfg = Cosmos3PipelineConfig(model_path="checkpoint")
    cfg.stages[0].gpu = devices[0]
    cfg.stages[0].runtime_gpu_ids = list(devices)
    return cfg


def test_native_workers_reserve_all_gpus_but_use_one_omni_process(monkeypatch):
    from sglang_omni.utils import cpu

    monkeypatch.setattr(runtime_config, "visible_device_count", lambda: 4)
    monkeypatch.setattr(cpu, "effective_cpu_count", lambda: 10)
    cfg = config()
    prep = runtime_config.prepare_pipeline_runtime(cfg)
    try:
        groups = build_stage_groups(
            cfg,
            stages_cfg=prep.stages_cfg,
            endpoints=prep.endpoints,
            placement_plan=prep.placement_plan,
            process_plan=prep.process_plan,
        )
        assert set(prep.placement_plan.gpus) == {1, 3}
        assert len(groups) == 1
        assert len(groups[0].specs) == 1
        spec = groups[0].specs[0]
        assert spec.tp_size == 1
        assert spec.factory_kwargs["runtime_gpu_ids"] == [1, 3]
        # Each of the two native workers gets its own share, not the plan's.
        assert spec.env_defaults["OMP_NUM_THREADS"] == "5"
    finally:
        prep.runtime_dir.close()


def test_native_gpu_budget_includes_child_devices():
    cfg = config()
    cfg.stages[0].gpu_memory_fraction = 0.6
    cfg.stages.append(
        StageConfig(
            name="other",
            factory_path=cfg.stages[0].factory_path,
            gpu=3,
            gpu_memory_fraction=0.6,
            terminal=True,
        )
    )
    with pytest.raises(ValueError, match="exceeds placement limit"):
        build_stage_placement_plan(cfg)


def test_native_gpu_range_is_validated_before_spawn():
    with pytest.raises(ValueError, match="GPU id 3 out of range"):
        validate_device_assignment(config().stages, device_count=2)


def test_native_parallel_options_use_the_declared_gpu_group():
    gen = native_server_kwargs("checkpoint", 1, {"tp_size": 2}, [1, 3])
    assert gen["gpu_ids"] == [1, 3]
    assert gen["num_gpus"] == gen["tp_size"] == 2


@pytest.mark.parametrize("devices", [[], [1, 1], [2, 3], [1, -1]])
def test_invalid_native_gpu_lists_are_rejected(devices):
    cfg = config()
    cfg.stages[0].runtime_gpu_ids = devices
    with pytest.raises(ValueError):
        build_stage_placement_plan(cfg)


def test_native_runtime_cannot_share_a_process_with_another_stage(monkeypatch):
    monkeypatch.setattr(runtime_config, "visible_device_count", lambda: 4)
    cfg = config()
    cfg.stages.append(
        StageConfig(
            name="other",
            process="generation",
            factory_path=cfg.stages[0].factory_path,
            terminal=True,
        )
    )
    with pytest.raises(ValueError, match="own its OS process"):
        runtime_config.prepare_pipeline_runtime(cfg)


@pytest.mark.parametrize("devices", [None, [3, 5]])
def test_generation_factory_passes_resolved_device_to_native(
    monkeypatch, tmp_path, devices
):
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    from sglang_omni.models.cosmos3.stages import create_generation_scheduler
    from sglang_omni.utils import device as device_utils

    monkeypatch.setattr(
        device_utils,
        "resolve_concrete_device",
        lambda device, index: SimpleNamespace(index=3),
    )

    def startup(**kwargs):
        if devices is None:
            assert kwargs["base_gpu_id"] == 3
        else:
            assert kwargs["gpu_ids"] == [3, 5]
        raise RuntimeError("native startup reached")

    monkeypatch.setattr(ServerArgs, "from_kwargs", startup)
    with pytest.raises(RuntimeError, match="native startup reached"):
        create_generation_scheduler(
            "checkpoint",
            device=None,
            gpu_id=None,
            runtime_gpu_ids=devices,
            output_dir=str(tmp_path),
        )


def test_generation_factory_rejects_cpu_before_native_startup(monkeypatch):
    from sglang_omni.models.cosmos3.stages import create_generation_scheduler
    from sglang_omni.utils import device as device_utils

    monkeypatch.setattr(
        device_utils,
        "resolve_concrete_device",
        lambda device, index: SimpleNamespace(index=None),
    )
    with pytest.raises(ValueError, match="indexed accelerator"):
        create_generation_scheduler("checkpoint", device="cpu")


@pytest.mark.parametrize("case", ["plugin", "plugin_missing", "two_gpus"])
def test_generation_factory_selects_cooperative_cancellation(
    monkeypatch, tmp_path, caplog, case
):
    import logging
    import os

    from sglang.multimodal_gen.runtime import scheduler_client, server_warmup
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )
    from sglang.multimodal_gen.runtime.platforms import plugins
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    from sglang_omni.models.cosmos3 import stages
    from sglang_omni.native_generation import cancellation
    from sglang_omni.utils import device as device_utils

    monkeypatch.setattr(
        device_utils,
        "resolve_concrete_device",
        lambda device, index: SimpleNamespace(index=0),
    )
    monkeypatch.delenv(cancellation.ACTIVATION_ENV, raising=False)
    activations = []
    applied = []
    monkeypatch.setattr(
        plugins,
        "apply_plugin_hooks",
        lambda: applied.append(os.environ.get(cancellation.ACTIVATION_ENV)),
    )

    def start_workers(args):
        # Native applies plugin hooks here too, then starts a worker that ran
        # the plugin's handshake.
        plugins.apply_plugin_hooks()
        directory = os.environ.get(cancellation.ACTIVATION_ENV)
        activations.append(directory)
        if case == "plugin":
            with open(cancellation.handshake_path(directory, 0), "x"):
                pass
        else:
            pass
        return SimpleNamespace(
            shutdown=lambda: None,
            local_scheduler_process=[],
            server_args=SimpleNamespace(resolved=case),
        )

    monkeypatch.setattr(ServerArgs, "from_kwargs", lambda **kwargs: kwargs)
    monkeypatch.setattr(DiffGenerator, "from_server_args", start_workers)
    monkeypatch.setattr(stages, "model_request_fields", lambda server_args: frozenset())
    monkeypatch.setattr(
        server_warmup,
        "should_run_synthetic_server_warmup",
        lambda server_args: server_args.resolved == "plugin",
    )
    monkeypatch.setattr(
        server_warmup, "should_run_explicit_client_warmup", lambda server_args: False
    )
    monkeypatch.setattr(server_warmup, "supports_synthetic_warmup", lambda args: True)
    monkeypatch.setattr(server_warmup, "is_realtime_serving", lambda args: False)
    warmups = []

    def warm_up(server_args, forward):
        # Native writes its warmup image under input_save_path, then fails.
        with open(os.path.join(server_args.input_save_path, "warmup_image"), "x"):
            pass
        warmups.append((server_args, forward))
        raise RuntimeError("warmup request failed")

    monkeypatch.setattr(server_warmup, "run_sync_client_warmup", warm_up)
    with caplog.at_level(logging.WARNING, logger="sglang_omni.models.cosmos3.stages"):
        scheduler = stages.create_generation_scheduler(
            "checkpoint",
            output_dir=str(tmp_path),
            runtime_gpu_ids=[0, 1] if case == "two_gpus" else None,
        )
    expected = "plugin" if case == "plugin" else None
    assert scheduler.cancellation_mode == expected
    assert ("keeps the generation stage busy" in caplog.text) is (expected is None)
    # Only a single GPU stage arms its workers, and the handshake directory
    # and the variable do not outlive startup.
    armed = case in ("plugin", "plugin_missing")
    assert (activations[0] is not None) is armed
    # Native applies hooks once per process, so the host's first application
    # must not see the activation it sets for its workers.
    assert applied[0] is None
    assert activations[0] is None or not os.path.exists(activations[0])
    assert cancellation.ACTIVATION_ENV not in os.environ
    # Arguments that select native's server warmup run it in the stage before
    # it is ready, through the client every request uses, failing open as
    # native does without explicit warmup resolutions, and its input goes. It
    # lives under the stage's output directory, with its request workspaces.
    if case == "plugin":
        [(warmup_args, forward)] = warmups
        assert forward == scheduler_client.sync_scheduler_client.forward
        assert warmup_args.resolved == case
        assert os.path.dirname(warmup_args.input_save_path) == str(tmp_path)
        assert not os.path.exists(warmup_args.input_save_path)
        assert "Synthetic server warmup failed" in caplog.text
    else:
        assert warmups == []
    # Without the server warmup, the stage warms in the background once it
    # serves.
    assert scheduler.warm_in_background is (case != "plugin")


def test_text_variant_serves_the_reasoner_as_an_engine_stage():
    from sglang_omni.cli.serve import engine_stage_names

    config = Variants["text"](model_path="m")
    assert engine_stage_names(config) == ["reasoner"]
    # The Reasoner has no generation stage, so it serves no native media routes.
    assert not any(stage.http_routes for stage in config.stages)
