# SPDX-License-Identifier: Apache-2.0
"""Launcher contract tests for encoder TP (Plan B).

These tests cover only the launcher / config layer — they do not
spawn child processes or load any model. They lock the contracts the
RFC calls "load-bearing":

- SGLang-backed encoder stages derive single-device process isolation from
  the resolved execution backend, including when the requested backend is
  driven by ``runtime_overrides`` (signature-default trap).
- ``get_stage_process_env`` honors that derived launch mode at ``tp_size=1``.
- ``serve/launcher.py`` ``needs_mp`` predicate routes single-stage,
  single-GPU, ``tp_size=1`` ``backend="sglang"`` configs through the
  multi-process runner.
- The TP preflight in ``mp_runner._build_stage_groups`` rejects:
   * Layer 1 — TP stage whose factory does not accept
     ``tp_rank/tp_size/nccl_port``;
   * Layer 2 — encoder TP stage whose resolved execution backend is not
     ``sglang``.
- Production factory signature defaults stay ``"local"`` forever
  (real-factory signature lock).
"""
from __future__ import annotations

import inspect
import multiprocessing
from types import SimpleNamespace
from typing import Any

import pytest

from examples.qwen3_omni_encoder_tp import (
    _apply_ar_mem_fraction,
    _apply_encoder_runtime,
    _resolve_effective_encoder_backend,
    _resolve_encoder_max_batch_size,
    _resolve_layout,
    _validate_optional_fraction,
)
from sglang_omni.config.schema import EndpointsConfig, PipelineConfig, StageConfig
from sglang_omni.pipeline.mp_runner import _build_stage_groups
from sglang_omni.pipeline.stage_process import (
    StageProcessSpec,
    get_stage_process_env,
    stage_requires_single_visible_device,
)

# ---------------------------------------------------------------------------
# Test factories — registered as importable callables so ``import_string``
# inside the runner can resolve them. Live module-level (not inside a
# fixture) so dotted-path import works.
# ---------------------------------------------------------------------------


def fake_factory_sglang(
    model_path: str,
    *,
    backend: str = "local",
    gpu_id: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
):
    """Encoder-shaped factory: accepts both backend and TP launch params."""
    return ("scheduler", model_path, backend, gpu_id, tp_rank, tp_size, nccl_port)


def fake_factory_thinker(
    model_path: str,
    *,
    gpu_id: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
):
    """Thinker-shaped factory: TP-capable but no `backend` param."""
    return ("thinker", model_path, gpu_id, tp_rank, tp_size, nccl_port)


def fake_factory_simple(model_path: str, *, gpu_id: int = 0):
    """SimpleScheduler-shaped factory: no TP params."""
    return ("simple", model_path, gpu_id)


def fake_factory_signature_auto_default(
    model_path: str,
    *,
    backend: str = "auto",  # signature default flipped to "auto"
    gpu_id: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
):
    """Factory whose signature default for `backend` is "auto".

    Used to lock the [Backend resolution contract]: launcher decisions
    must NOT introspect the signature default. A StageConfig that omits
    ``factory_args["backend"]`` should resolve to ``"local"`` here, not
    ``"auto"``, even though the factory body would default to ``"auto"``.
    """
    return ("auto-default", model_path, backend, gpu_id, tp_rank, tp_size, nccl_port)


# Module-level dotted paths for the helpers above.
_F_SGLANG = f"{__name__}.fake_factory_sglang"
_F_THINKER = f"{__name__}.fake_factory_thinker"
_F_SIMPLE = f"{__name__}.fake_factory_simple"
_F_AUTO_DEFAULT = f"{__name__}.fake_factory_signature_auto_default"
_F_AUTO_LOCAL = "tests.unit_test.fixtures.pipeline_fakes.auto_local_encoder_factory"


# Avoid socket probing in unit tests: this sandbox can block socket creation,
# while these tests only need stable parent-allocated port values.
@pytest.fixture(autouse=True)
def _deterministic_nccl_ports(monkeypatch):
    from sglang_omni.pipeline import mp_runner

    monkeypatch.setenv("FLASHINFER_WORKSPACE_BASE", "/tmp")

    next_port = {"value": 29500}

    def _allocate(self):
        port = next_port["value"]
        next_port["value"] += 1
        return port

    monkeypatch.setattr(mp_runner._NcclPortAllocator, "allocate", _allocate)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_example_resolves_colocated_two_gpu_layout():
    image_gpus, audio_gpus, thinker_gpu, talker_gpu = _resolve_layout(
        layout="colocated-2gpu",
        image_tp=2,
        audio_tp=2,
    )

    assert image_gpus == [0, 1]
    assert audio_gpus == [0, 1]
    assert thinker_gpu == 0
    assert talker_gpu == 1


def test_example_rejects_colocated_two_gpu_layout_above_tp2():
    with pytest.raises(SystemExit, match="colocated-2gpu"):
        _resolve_layout(layout="colocated-2gpu", image_tp=3, audio_tp=2)


def test_example_leaves_colocated_tp2_encoder_max_batch_size_unset():
    assert (
        _resolve_encoder_max_batch_size(
            explicit=None,
            layout="colocated-2gpu",
            image_tp=2,
            audio_tp=2,
        )
        is None
    )


def test_example_leaves_tp1_encoder_max_batch_size_unset():
    assert (
        _resolve_encoder_max_batch_size(
            explicit=None,
            layout="colocated-2gpu",
            image_tp=1,
            audio_tp=1,
        )
        is None
    )


def test_example_honors_explicit_encoder_max_batch_size():
    assert (
        _resolve_encoder_max_batch_size(
            explicit=4,
            layout="colocated-2gpu",
            image_tp=2,
            audio_tp=2,
        )
        == 4
    )


def test_example_honors_encoder_backend_for_tp1():
    assert _resolve_effective_encoder_backend("auto", tp_size=1) == "sglang"
    assert _resolve_effective_encoder_backend("sglang", tp_size=1) == "sglang"
    assert _resolve_effective_encoder_backend("local", tp_size=1) == "local"
    assert _resolve_effective_encoder_backend("auto", tp_size=2) == "sglang"
    assert _resolve_effective_encoder_backend("sglang", tp_size=2) == "sglang"


def _make_config(
    *,
    factory: str,
    factory_args: dict[str, Any] | None = None,
    runtime_overrides: dict[str, dict[str, Any]] | None = None,
    tp_size: int = 1,
    gpu: int | list[int] | None = 0,
) -> PipelineConfig:
    return PipelineConfig(
        model_path="dummy/model",
        stages=[
            StageConfig(
                name="image_encoder",
                process="image_encoder",
                factory=factory,
                factory_args=dict(factory_args or {}),
                tp_size=tp_size,
                gpu=gpu,
                terminal=True,
            ),
        ],
        runtime_overrides=runtime_overrides or {},
        endpoints=EndpointsConfig(base_path="/tmp/encoder_tp_test"),
    )


# ---------------------------------------------------------------------------
# 1. Derived SGLang launch isolation (mp_runner._build_stage_groups)
# ---------------------------------------------------------------------------


def test_sglang_backend_gets_single_rank_tp_launch_args(tmp_path):
    cfg = _make_config(
        factory=_F_SGLANG,
        factory_args={"backend": "sglang"},
    )
    ctx = multiprocessing.get_context("spawn")
    groups = _build_stage_groups(cfg, ctx=ctx)
    assert len(groups) == 1
    spec = groups[0].specs[0]
    assert stage_requires_single_visible_device(spec) is True
    assert spec.factory_args["tp_rank"] == 0
    assert spec.factory_args["tp_size"] == 1
    assert isinstance(spec.factory_args["nccl_port"], int)
    assert spec.nccl_port == spec.factory_args["nccl_port"]


def test_auto_backend_gets_single_rank_tp_launch_args():
    cfg = _make_config(factory=_F_SGLANG, factory_args={"backend": "auto"})
    ctx = multiprocessing.get_context("spawn")
    spec = _build_stage_groups(cfg, ctx=ctx)[0].specs[0]
    assert stage_requires_single_visible_device(spec) is True
    assert spec.factory_args["tp_rank"] == 0
    assert spec.factory_args["tp_size"] == 1
    assert isinstance(spec.factory_args["nccl_port"], int)


def test_launch_mode_map_records_auto_resolution():
    from sglang_omni.config.runtime import build_stage_launch_modes

    cfg = _make_config(factory=_F_AUTO_LOCAL, factory_args={"backend": "auto"})

    mode = build_stage_launch_modes(cfg)["image_encoder"]

    assert mode.requested_backend == "auto"
    assert mode.execution_backend == "local"
    assert mode.requires_sglang_launch is False
    assert mode.is_sglang_execution is False


def test_auto_backend_resolving_local_does_not_get_sglang_launch_args():
    cfg = _make_config(factory=_F_AUTO_LOCAL, factory_args={"backend": "auto"})
    ctx = multiprocessing.get_context("spawn")
    spec = _build_stage_groups(cfg, ctx=ctx)[0].specs[0]
    assert stage_requires_single_visible_device(spec) is False
    assert "tp_rank" not in spec.factory_args
    assert "tp_size" not in spec.factory_args
    assert "nccl_port" not in spec.factory_args
    assert spec.nccl_port is None


def test_local_backend_does_not_get_single_rank_tp_launch_args():
    cfg = _make_config(factory=_F_SGLANG, factory_args={"backend": "local"})
    ctx = multiprocessing.get_context("spawn")
    spec = _build_stage_groups(cfg, ctx=ctx)[0].specs[0]
    assert stage_requires_single_visible_device(spec) is False
    assert "tp_rank" not in spec.factory_args
    assert "tp_size" not in spec.factory_args
    assert "nccl_port" not in spec.factory_args


def test_sglang_launch_mode_honors_runtime_overrides():
    """Locks the [Backend resolution contract]: runtime_overrides drives the flip."""
    cfg = _make_config(
        factory=_F_SGLANG,
        factory_args={"backend": "local"},  # default
        runtime_overrides={"image_encoder": {"backend": "sglang"}},
    )
    ctx = multiprocessing.get_context("spawn")
    spec = _build_stage_groups(cfg, ctx=ctx)[0].specs[0]
    assert stage_requires_single_visible_device(spec) is True
    assert spec.factory_args["backend"] == "sglang"
    assert isinstance(spec.factory_args["nccl_port"], int)


def test_sglang_launch_mode_signature_default_trap():
    """Launcher must NOT read factory signature defaults.

    Factory body defaults to ``backend="auto"`` but ``factory_args``
    omits the key — the launcher should resolve to ``"local"``.
    """
    cfg = _make_config(factory=_F_AUTO_DEFAULT, factory_args={})
    ctx = multiprocessing.get_context("spawn")
    spec = _build_stage_groups(cfg, ctx=ctx)[0].specs[0]
    assert stage_requires_single_visible_device(spec) is False


# ---------------------------------------------------------------------------
# 2. get_stage_process_env early return
# ---------------------------------------------------------------------------


def _spec(
    *,
    tp_size: int = 1,
    backend: str = "local",
    gpu_id: int = 0,
):
    nccl_port = 29500 if tp_size > 1 or backend == "sglang" else None
    return StageProcessSpec(
        stage_name="image_encoder",
        tp_size=tp_size,
        gpu_id=gpu_id,
        nccl_port=nccl_port,
        factory_args={"backend": backend},
    )


def test_get_stage_process_env_returns_empty_for_plain_single_process_stage():
    assert get_stage_process_env(_spec()) == {}


def test_get_stage_process_env_remaps_for_sglang_backend_at_tp_size_1():
    env = {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}
    overrides = get_stage_process_env(
        _spec(backend="sglang", gpu_id=4),
        env=env,
    )
    assert overrides["CUDA_VISIBLE_DEVICES"] == "4"
    assert overrides["SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS"] == "true"


def test_get_stage_process_env_remaps_for_tp_size_2():
    """Existing tp_size>1 lane keeps working."""
    env = {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    overrides = get_stage_process_env(
        _spec(tp_size=2, gpu_id=2),
        env=env,
    )
    assert overrides["CUDA_VISIBLE_DEVICES"] == "2"


# ---------------------------------------------------------------------------
# 3. serve/launcher.py needs_mp predicate
# ---------------------------------------------------------------------------


def test_any_sglang_backend_stage_detects_sglang():
    from sglang_omni.pipeline.mp_runner import any_sglang_backend_stage

    cfg = _make_config(factory=_F_SGLANG, factory_args={"backend": "sglang"})
    assert any_sglang_backend_stage(cfg) is True


def test_any_sglang_backend_stage_detects_auto():
    from sglang_omni.pipeline.mp_runner import any_sglang_backend_stage

    cfg = _make_config(factory=_F_SGLANG, factory_args={"backend": "auto"})
    assert any_sglang_backend_stage(cfg) is True


def test_any_sglang_backend_stage_false_for_local():
    from sglang_omni.pipeline.mp_runner import any_sglang_backend_stage

    cfg = _make_config(factory=_F_SGLANG, factory_args={"backend": "local"})
    assert any_sglang_backend_stage(cfg) is False


def test_any_sglang_backend_stage_ignores_signature_default():
    """signature default = "auto" but factory_args is empty → resolves to local."""
    from sglang_omni.pipeline.mp_runner import any_sglang_backend_stage

    cfg = _make_config(factory=_F_AUTO_DEFAULT, factory_args={})
    assert any_sglang_backend_stage(cfg) is False


# ---------------------------------------------------------------------------
# 4. TP preflight Layers 1 & 2
# ---------------------------------------------------------------------------


def test_tp_preflight_layer1_rejects_factory_without_tp_params():
    cfg = _make_config(factory=_F_SIMPLE, tp_size=2, gpu=[0, 1])
    ctx = multiprocessing.get_context("spawn")
    with pytest.raises(ValueError, match="not TP-capable"):
        _build_stage_groups(cfg, ctx=ctx)


def test_tp_preflight_layer2_rejects_encoder_with_local_backend():
    cfg = _make_config(
        factory=_F_SGLANG,
        factory_args={"backend": "local"},
        tp_size=2,
        gpu=[0, 1],
    )
    ctx = multiprocessing.get_context("spawn")
    with pytest.raises(ValueError, match="requires backend='sglang'"):
        _build_stage_groups(cfg, ctx=ctx)


def test_tp_preflight_passes_encoder_with_auto_backend():
    cfg = _make_config(
        factory=_F_SGLANG,
        factory_args={"backend": "auto"},
        tp_size=2,
        gpu=[0, 1],
    )
    ctx = multiprocessing.get_context("spawn")
    groups = _build_stage_groups(cfg, ctx=ctx)
    assert len(groups[0].specs) == 2


def test_tp_preflight_rejects_auto_backend_when_resolver_selects_local():
    cfg = _make_config(
        factory=_F_AUTO_LOCAL,
        factory_args={"backend": "auto"},
        tp_size=2,
        gpu=[0, 1],
    )
    ctx = multiprocessing.get_context("spawn")
    with pytest.raises(ValueError, match="execution='local'"):
        _build_stage_groups(cfg, ctx=ctx)


def test_tp_preflight_layer2_rejects_encoder_without_explicit_backend():
    cfg = _make_config(
        factory=_F_SGLANG,
        factory_args={},
        tp_size=2,
        gpu=[0, 1],
    )
    ctx = multiprocessing.get_context("spawn")
    with pytest.raises(ValueError, match="requires backend='sglang'"):
        _build_stage_groups(cfg, ctx=ctx)


def test_tp_preflight_does_not_regress_thinker_tp():
    cfg = _make_config(factory=_F_THINKER, tp_size=2, gpu=[0, 1])
    ctx = multiprocessing.get_context("spawn")
    # Should succeed.
    groups = _build_stage_groups(cfg, ctx=ctx)
    assert len(groups[0].specs) == 2


def test_tp_preflight_passes_encoder_with_sglang_backend():
    cfg = _make_config(
        factory=_F_SGLANG,
        factory_args={"backend": "sglang"},
        tp_size=2,
        gpu=[0, 1],
    )
    ctx = multiprocessing.get_context("spawn")
    groups = _build_stage_groups(cfg, ctx=ctx)
    assert len(groups[0].specs) == 2
    for spec in groups[0].specs:
        assert stage_requires_single_visible_device(spec) is True
        assert spec.factory_args["nccl_port"] == spec.nccl_port


# ---------------------------------------------------------------------------
# 5. Real-factory signature lock
# ---------------------------------------------------------------------------


def test_real_factory_signature_lock_image_encoder():
    """The production image-encoder factory's signature default must stay "local".

    Bumping it to "auto" / "sglang" would silently desync launcher
    (sees "local") from factory body. See [Backend resolution contract].
    """
    from sglang_omni.models.qwen3_omni.stages import create_image_encoder_runner

    sig = inspect.signature(create_image_encoder_runner)
    assert sig.parameters["backend"].default == "local"


def test_real_factory_signature_lock_audio_encoder():
    from sglang_omni.models.qwen3_omni.stages import create_audio_encoder_runner

    sig = inspect.signature(create_audio_encoder_runner)
    assert sig.parameters["backend"].default == "local"


def test_sglang_encoder_factory_defaults_single_request_cap_to_activation_budget(
    monkeypatch,
):
    from sglang_omni.model_runner import sglang_encoder_runner
    from sglang_omni.models.qwen3_omni import stages
    from sglang_omni.models.qwen3_omni.components import common
    from sglang_omni.scheduling import encoder_scheduler

    captured: dict[str, Any] = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeScheduler:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    hf_config = SimpleNamespace(
        thinker_config=SimpleNamespace(
            vision_config=SimpleNamespace(
                spatial_merge_size=2,
                out_hidden_size=4,
                deepstack_visual_indexes=[],
            ),
            audio_config=SimpleNamespace(num_mel_bins=8, output_dim=4),
        )
    )
    monkeypatch.setattr(common, "load_thinker_config", lambda model_path: hf_config)
    monkeypatch.setattr(sglang_encoder_runner, "SGLangEncoderRunner", FakeRunner)
    monkeypatch.setattr(encoder_scheduler, "EncoderScheduler", FakeScheduler)

    stages.create_image_encoder_runner(
        "dummy/model",
        backend="sglang",
        tp_size=2,
        nccl_port=29500,
        encoder_activation_budget_bytes=1234,
        encoder_max_batch_size=1,
    )
    assert captured["max_single_request_cost"] == 1234
    assert captured["max_batch_size"] == 1

    stages.create_image_encoder_runner(
        "dummy/model",
        backend="sglang",
        tp_size=2,
        nccl_port=29500,
        encoder_activation_budget_bytes=1234,
        max_single_request_cost=999,
    )
    assert captured["max_single_request_cost"] == 999


# ---------------------------------------------------------------------------
# 6. TP launch params reject (footgun in user-supplied factory_args /
# runtime_overrides — Codex adversarial review finding)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,value",
    [
        ("tp_size", 2),
        ("tp_rank", 1),
        ("nccl_port", 29500),
    ],
)
def test_resolve_factory_args_rejects_tp_launch_params_in_factory_args(key, value):
    """Lock the TP topology source-of-truth contract.

    ``StageConfig.tp_size`` is the only public way to set TP size.
    Letting ``factory_args[tp_size]=N`` slip through while
    ``StageConfig.tp_size`` says 1 (or vice versa) silently desyncs
    the runner's spawn count from the runner's NCCL world size,
    hanging bootstrap. ``_resolve_factory_args`` must reject these
    keys with a clear error before any spec is built.
    """
    from sglang_omni.config.runtime import resolve_stage_factory_args

    cfg = _make_config(
        factory=_F_SGLANG,
        factory_args={"backend": "sglang", key: value},
    )
    with pytest.raises(ValueError, match=key):
        resolve_stage_factory_args(cfg.stages[0], cfg)


@pytest.mark.parametrize(
    "key,value",
    [
        ("tp_size", 2),
        ("tp_rank", 1),
        ("nccl_port", 29500),
    ],
)
def test_resolve_factory_args_rejects_tp_launch_params_in_runtime_overrides(key, value):
    """Same as above but via ``runtime_overrides`` (the CLI-friendly path)."""
    from sglang_omni.config.runtime import resolve_stage_factory_args

    cfg = _make_config(
        factory=_F_SGLANG,
        factory_args={"backend": "sglang"},
        runtime_overrides={"image_encoder": {key: value}},
    )
    with pytest.raises(ValueError, match=key):
        resolve_stage_factory_args(cfg.stages[0], cfg)


def test_user_supplied_tp_size_in_factory_args_blocks_at_build_stage_groups():
    """End-to-end regression: the silent NCCL-bootstrap-hang scenario.

    User sets ``StageConfig.tp_size=1`` (default) and
    ``factory_args={"backend":"sglang","tp_size":2}``. Pre-fix, the
    runner spawned one process but the factory got tp_size=2, hanging
    NCCL forever. Post-fix, ``_build_stage_groups`` raises before any
    spawn.
    """
    cfg = _make_config(
        factory=_F_SGLANG,
        factory_args={"backend": "sglang", "tp_size": 2},
        # NOTE: StageConfig.tp_size defaults to 1 — the bug surface.
    )
    ctx = multiprocessing.get_context("spawn")
    with pytest.raises(ValueError, match="tp_size"):
        _build_stage_groups(cfg, ctx=ctx)


# ---------------------------------------------------------------------------
# 7. Example launcher validation knobs
# ---------------------------------------------------------------------------


def test_example_launcher_ar_mem_fraction_updates_typed_runtime():
    stage = StageConfig(name="thinker", factory=_F_THINKER, gpu=2)

    updated = _apply_ar_mem_fraction(stage, 0.45)

    assert stage.runtime.resources.total_gpu_memory_fraction is None
    assert stage.runtime.sglang_server_args.mem_fraction_static is None
    assert updated.runtime.resources.total_gpu_memory_fraction == 0.45
    assert updated.runtime.sglang_server_args.mem_fraction_static == 0.45


def test_example_launcher_ar_mem_fraction_can_preserve_main_typed_runtime():
    stage = StageConfig(name="thinker", factory=_F_THINKER, gpu=2)
    stage.runtime.resources.total_gpu_memory_fraction = 0.7

    updated = _apply_ar_mem_fraction(stage, 0.45, typed_resource=False)

    assert updated.runtime.resources.total_gpu_memory_fraction is None
    assert updated.runtime.sglang_server_args.mem_fraction_static == 0.45


def test_example_launcher_encoder_runtime_updates_typed_resources():
    stage = StageConfig(name="image_encoder", factory=_F_SGLANG, gpu=[0, 1])

    updated = _apply_encoder_runtime(
        stage,
        activation_budget_bytes=1234,
        total_gpu_memory_fraction=0.05,
        encoder_max_batch_size=1,
    )

    assert stage.runtime.resources.encoder_activation_budget_bytes is None
    assert stage.runtime.resources.total_gpu_memory_fraction is None
    assert stage.runtime.resources.encoder_max_batch_size is None
    assert updated.runtime.resources.encoder_activation_budget_bytes == 1234
    assert updated.runtime.resources.total_gpu_memory_fraction == 0.05
    assert updated.runtime.resources.encoder_max_batch_size == 1


def test_example_launcher_encoder_runtime_can_preserve_main_typed_runtime():
    stage = StageConfig(name="image_encoder", factory=_F_SGLANG, gpu=0)
    stage.runtime.resources.total_gpu_memory_fraction = 0.1

    updated = _apply_encoder_runtime(
        stage,
        activation_budget_bytes=1234,
        total_gpu_memory_fraction=0.05,
        encoder_max_batch_size=None,
        typed_resource=False,
    )

    assert updated.runtime.resources.encoder_activation_budget_bytes == 1234
    assert updated.runtime.resources.total_gpu_memory_fraction is None
    assert updated.runtime.resources.encoder_max_batch_size is None


def test_example_launcher_rejects_invalid_ar_mem_fraction():
    with pytest.raises(SystemExit, match="must be in"):
        _validate_optional_fraction("--thinker-mem-fraction-static", 1.0)
