# SPDX-License-Identifier: Apache-2.0
"""Every GPU-placed stage factory must take device/gpu_id and honor gpu_id."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError, RepositoryNotFoundError

import sglang_omni.platforms as platforms
from sglang_omni.utils.imports import import_string

_MODELS_DIR = Path(importlib.import_module("sglang_omni.models").__file__).parent

# note (lennox): zonos2's preprocessing is CPU-only but declares gpu=0 to share
# the pipeline process with tts_engine; qwen3_omni's mm_aggregate is an identity
# stage placed on a GPU in the "text" topology for pure colocation.
_CPU_ONLY_GPU_PLACED = {
    ("qwen3_omni", "mm_aggregate"),
    ("zonos2", "preprocessing"),
}


def _iter_stages():
    for config_path in sorted(_MODELS_DIR.glob("*/config.py")):
        model = config_path.parent.name
        module = importlib.import_module(f"sglang_omni.models.{model}.config")
        topologies = {}
        if getattr(module, "EntryClass", None) is not None:
            topologies["default"] = module.EntryClass
        topologies.update(getattr(module, "Variants", None) or {})
        for label, config_cls in topologies.items():
            for stage in config_cls(model_path="unused").stages:
                yield model, label, stage


# note (lennox): only device/gpu_id are asserted on below, but a non-literal
# default elsewhere (a name, an arithmetic expr) would raise before reaching them.
def _literal_default(node: ast.expr) -> object:
    try:
        return ast.literal_eval(node)
    except ValueError:
        return ...


def _factory_parameters(dotted: str) -> dict[str, object]:
    try:
        return {
            name: (... if p.default is inspect.Parameter.empty else p.default)
            for name, p in inspect.signature(import_string(dotted)).parameters.items()
        }
    except ImportError:
        pass
    module_name, _, func_name = dotted.rpartition(".")
    source = (
        _MODELS_DIR.parent.parent / (module_name.replace(".", "/") + ".py")
    ).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            args = node.args
            positional = args.posonlyargs + args.args
            defaults = [...] * (len(positional) - len(args.defaults)) + [
                _literal_default(d) for d in args.defaults
            ]
            params = dict(zip((a.arg for a in positional), defaults))
            for a, d in zip(args.kwonlyargs, args.kw_defaults):
                params[a.arg] = ... if d is None else _literal_default(d)
            return params
    raise AssertionError(f"factory {dotted} not found in {module_name}")


# note (lennox): these factories raise on torch.cuda.is_available() before
# this test's mocks run, so they need a static accelerator mark (tests/README.md).
_REQUIRES_REAL_ACCELERATOR = {
    ("dots_tts", "reference_encode"),
    ("dots_tts", "latent_engine"),
    ("dots_tts", "vocoder"),
    ("minimax_music3", "minimax_music3_ar"),
    ("minimax_music3", "dit_dav"),
    ("zonos2", "tts_engine"),
}


def _gpu_stage_ids(*, mark_accelerator=False, include_exempt=False):
    ids = []
    for model, label, stage in _iter_stages():
        if stage.gpu is None:
            continue
        if not include_exempt and (model, stage.name) in _CPU_ONLY_GPU_PLACED:
            continue
        marks = (
            [pytest.mark.accelerator]
            if mark_accelerator and (model, stage.name) in _REQUIRES_REAL_ACCELERATOR
            else []
        )
        ids.append(
            pytest.param(
                model, label, stage, id=f"{model}-{label}-{stage.name}", marks=marks
            )
        )
    return ids


# note (lennox): exempt colocation-only stages still take the parameters --
# the launch-time gate injects gpu_id into every GPU-placed stage's factory.
@pytest.mark.parametrize("model,label,stage", _gpu_stage_ids(include_exempt=True))
def test_gpu_stage_factories_declare_device_and_gpu_id(model, label, stage):
    params = _factory_parameters(stage.factory_path)
    assert "gpu_id" in params, (
        f"{stage.factory_path} is placed on a GPU (stage.gpu={stage.gpu}) but has "
        "no gpu_id parameter"
    )
    assert (
        params["gpu_id"] is None
    ), f"{stage.factory_path}: gpu_id defaults to {params['gpu_id']!r}, should use None"
    assert "device" in params, f"{stage.factory_path} has no device parameter"
    assert (
        params["device"] is None
    ), f"{stage.factory_path}: device defaults to {params['device']!r}, should use None"


def _device_is_set(stage) -> bool:
    factory = stage.factory
    return "device" in factory.model_fields_set or "device" in (
        factory.model_extra or {}
    )


@pytest.mark.parametrize(
    "model,label,stage",
    [pytest.param(m, l, s, id=f"{m}-{l}-{s.name}") for m, l, s in _iter_stages()],
)
def test_config_device_never_carries_an_index(model, label, stage):
    if not _device_is_set(stage):
        return
    device = stage.factory.device
    assert ":" not in str(device), (
        f"stage {stage.name!r} of {model}/{label} sets device={device!r}; "
        "device must not contain index, set in stage.gpu"
    )


class _Settled(Exception):
    def __init__(self, device, index):
        self.device = device
        self.index = index


def _arm_device_spec_resolvers(monkeypatch, factory_path: str | None = None):
    import sglang_omni.utils.device as device_mod
    from sglang_omni.scheduling.engine_factory import SGLangGenerationEngineBuilder

    def _capture(device, index=None):
        raise _Settled(device, index)

    monkeypatch.setattr(device_mod, "resolve_concrete_device", _capture)
    # note (lennox): a factory module that imports resolve_concrete_device at module
    # scope (rather than inside the factory body) binds its own name to the
    # pre-patch function, so patching device_mod alone is invisible to it -- patch
    # the factory's own module too when it holds that name.
    if factory_path is not None:
        factory_module = importlib.import_module(factory_path.rsplit(".", 1)[0])
        if "resolve_concrete_device" in vars(factory_module):
            monkeypatch.setattr(factory_module, "resolve_concrete_device", _capture)
    # note (lennox): builders import lazily inside factory bodies, so patch each
    # known builder class directly; MRO bypasses a base-class-only patch.
    monkeypatch.setattr(
        SGLangGenerationEngineBuilder,
        "resolve_checkpoint",
        lambda self, model_path: model_path,
    )
    for _, builder_module, builder_class in _ENGINE_FACTORIES.values():
        try:
            builder = getattr(importlib.import_module(builder_module), builder_class)
        except ImportError:
            continue
        monkeypatch.setattr(
            builder, "resolve_checkpoint", lambda self, model_path: model_path
        )


@pytest.mark.parametrize("model,label,stage", _gpu_stage_ids(mark_accelerator=True))
def test_gpu_stage_factories_forward_gpu_id_into_device_spec_resolution(
    monkeypatch, model, label, stage
):
    try:
        factory = import_string(stage.factory_path)
    except ImportError as exc:
        pytest.skip(f"optional dependency missing: {exc}")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    _arm_device_spec_resolvers(monkeypatch, factory_path=stage.factory_path)
    kwargs: dict[str, object] = {"device": None, "gpu_id": 2}
    if "model_path" in _factory_parameters(stage.factory_path):
        kwargs["model_path"] = "unused"
    try:
        factory(**kwargs)
    except _Settled as settled:
        assert settled.index == 2, (
            f"{stage.factory_path} reached device-spec resolution but dropped gpu_id "
            f"(index={settled.index!r})"
        )
        assert settled.device is None
    except ModuleNotFoundError as exc:
        pytest.skip(f"optional dependency missing inside factory body: {exc}")
    except RuntimeError as exc:
        if isinstance(exc.__cause__, ImportError):
            pytest.skip(f"optional dependency missing inside factory body: {exc}")
        raise
    except TypeError as exc:
        pytest.fail(
            f"{stage.factory_path} rejected the standard (device, gpu_id) call: {exc}"
        )
    except (LocalEntryNotFoundError, RepositoryNotFoundError):
        # note (lennox): safety net for future factories that hit the network
        # before device resolution; no current factory reaches this.
        pytest.skip("factory resolves the checkpoint with a real HF Hub call")
    else:
        pytest.fail(
            f"{stage.factory_path} returned without ever consulting "
            "resolve_concrete_device; should not resolve device and "
            "gpu_id by hand (or ignore them)"
        )


# note (lennox): forwarding into resolve_device_spec isn't the same as binding
# its result -- this drives the real build() chain and checks what it fixed.
_ENGINE_FACTORIES = {
    "arkasr": (
        "sglang_omni.models.arkasr.stages.create_sglang_arkasr_executor",
        "sglang_omni.models.arkasr.engine_builder",
        "ArkasrEngineBuilder",
    ),
    "whisper_asr": (
        "sglang_omni.models.whisper_asr.stages.create_sglang_whisper_asr_executor",
        "sglang_omni.models.whisper_asr.engine_builder",
        "WhisperASREngineBuilder",
    ),
    "fun_asr": (
        "sglang_omni.models.fun_asr.stages.create_sglang_fun_asr_executor",
        "sglang_omni.models.fun_asr.engine_builder",
        "FunASREngineBuilder",
    ),
    "moss_transcribe_diarize": (
        "sglang_omni.models.moss_transcribe_diarize.stages."
        "create_sglang_moss_transcribe_diarize_executor",
        "sglang_omni.models.moss_transcribe_diarize.engine_builder",
        "MossTranscribeDiarizeEngineBuilder",
    ),
    "qwen3_asr": (
        "sglang_omni.models.qwen3_asr.stages.create_sglang_qwen3_asr_executor",
        "sglang_omni.models.qwen3_asr.engine_builder",
        "Qwen3ASREngineBuilder",
    ),
    "dots_tts": (
        "sglang_omni.models.dots_tts.stages.create_sglang_latent_engine_executor",
        "sglang_omni.models.dots_tts.engine_builder",
        "DotsTTSEngineBuilder",
    ),
    "moss_tts": (
        "sglang_omni.models.moss_tts.stages.create_sglang_tts_engine_executor",
        "sglang_omni.models.moss_tts.engine_builder",
        "MossTtsEngineBuilder",
    ),
    "moss_tts_local": (
        "sglang_omni.models.moss_tts_local.stages.create_sglang_tts_engine_executor",
        "sglang_omni.models.moss_tts_local.engine_builder",
        "MossTtsLocalEngineBuilder",
    ),
    "ming_tts": (
        "sglang_omni.models.ming_tts.stages.create_sglang_tts_engine_executor",
        "sglang_omni.models.ming_tts.engine_builder",
        "MingTtsEngineBuilder",
    ),
    "voxtral_tts": (
        "sglang_omni.models.voxtral_tts.pipeline.stages.create_generation_executor",
        "sglang_omni.models.voxtral_tts.pipeline.engine_builder",
        "VoxtralTtsEngineBuilder",
    ),
    "fishaudio_s2_pro": (
        "sglang_omni.models.fishaudio_s2_pro.stages.create_sglang_tts_engine_executor",
        "sglang_omni.models.fishaudio_s2_pro.engine_builder",
        "FishS2ProEngineBuilder",
    ),
    "higgs_tts": (
        "sglang_omni.models.higgs_tts.stages.create_sglang_tts_engine_executor",
        "sglang_omni.models.higgs_tts.engine_builder",
        "HiggsTtsEngineBuilder",
    ),
    "minimax_music3": (
        "sglang_omni.models.minimax_music3.stages.create_ar_executor",
        "sglang_omni.models.minimax_music3.engine_builder",
        "MiniMaxMusic3EngineBuilder",
    ),
    "fun_cosyvoice3": (
        "sglang_omni.models.fun_cosyvoice3.stages.create_sglang_tts_engine_executor",
        "sglang_omni.models.fun_cosyvoice3.engine_builder",
        "FunCosyVoice3EngineBuilder",
    ),
    "qwen3_tts": (
        "sglang_omni.models.qwen3_tts.stages.create_sglang_tts_engine_executor",
        "sglang_omni.models.qwen3_tts.engine_builder",
        "Qwen3TtsEngineBuilder",
    ),
    "zonos2": (
        "sglang_omni.models.zonos2.stages.create_sglang_omni_tts_engine_executor",
        "sglang_omni.models.zonos2.engine_builder",
        "Zonos2EngineBuilder",
    ),
}


# note (lennox): same three CUDA-only models as _REQUIRES_REAL_ACCELERATOR,
# at this test's per-model (not per-stage) granularity.
_ACCELERATOR_ONLY_ENGINE_MODELS = {"dots_tts", "minimax_music3", "zonos2"}


def _engine_factory_ids():
    return [
        pytest.param(
            model,
            marks=(
                [pytest.mark.accelerator]
                if model in _ACCELERATOR_ONLY_ENGINE_MODELS
                else []
            ),
        )
        for model in sorted(_ENGINE_FACTORIES)
    ]


@pytest.mark.parametrize("model", _engine_factory_ids())
def test_engine_factories_bind_the_placed_gpu(monkeypatch, model):
    factory_path, builder_module, builder_class = _ENGINE_FACTORIES[model]
    try:
        factory = import_string(factory_path)
        builder = getattr(importlib.import_module(builder_module), builder_class)
    except ImportError as exc:
        pytest.skip(f"optional dependency missing: {exc}")

    final: dict[str, object] = {}

    class _Stop(Exception):
        pass

    def capture(self, checkpoint_dir):
        del checkpoint_dir
        final["device"] = self.device
        final["gpu_id"] = self.gpu_id
        raise _Stop

    monkeypatch.setattr(
        builder, "resolve_checkpoint", lambda self, model_path: model_path
    )
    monkeypatch.setattr(builder, "pre_infra_setup", capture)
    # note (lennox): pinned to "cuda" so the assertion below is
    # host-independent.
    monkeypatch.setattr(
        platforms.current_platform, "device_type", "cuda", raising=False
    )

    with pytest.raises(_Stop):
        factory(model_path="unused", device=None, gpu_id=2)
    assert final == {"device": "cuda:2", "gpu_id": 2}

    final.clear()
    with pytest.raises(_Stop):
        factory(model_path="unused", device="cuda", gpu_id=2)
    assert final == {"device": "cuda:2", "gpu_id": 2}


# note (lennox): these AR factories build ServerArgs themselves instead of going
# through SGLangGenerationEngineBuilder.build(), so the builder-level test above
# cannot see whether they pin the resolved device type into ServerArgs.
_SELF_BUILT_SERVER_ARGS_FACTORIES = [
    "sglang_omni.models.llada2_uni.stages.create_sglang_dllm_thinker_executor_from_config",
    "sglang_omni.models.ming_omni.stages.create_sglang_thinker_executor_from_config",
    "sglang_omni.models.qwen3_omni.stages.create_sglang_thinker_executor_from_config",
    "sglang_omni.models.qwen3_omni.stages.create_talker_ar_executor_from_config",
]


@pytest.mark.parametrize(
    "factory_path",
    [
        pytest.param(p, id=p.split(".")[2] + "-" + p.rsplit(".", 1)[-1])
        for p in _SELF_BUILT_SERVER_ARGS_FACTORIES
    ],
)
def test_self_built_server_args_carry_the_resolved_device_type(
    monkeypatch, factory_path
):
    try:
        factory = import_string(factory_path)
    except ImportError as exc:
        pytest.skip(f"optional dependency missing: {exc}")

    from sglang_omni.scheduling import sglang_backend

    captured: dict[str, object] = {}

    class _Stop(Exception):
        pass

    def fake_build(model_path, **kwargs):
        del model_path
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(sglang_backend, "build_sglang_server_args", fake_build)
    factory_module = importlib.import_module(factory_path.rsplit(".", 1)[0])
    if "build_sglang_server_args" in vars(factory_module):
        monkeypatch.setattr(factory_module, "build_sglang_server_args", fake_build)
    # note (lennox): pinned to "cuda" so the assertion below is host-independent.
    monkeypatch.setattr(
        platforms.current_platform, "device_type", "cuda", raising=False
    )

    with pytest.raises(_Stop):
        factory("unused", device=None, gpu_id=2)
    assert captured["device"] == "cuda"

    with pytest.raises(ValueError, match="stage placement"):
        factory(
            "unused", device=None, gpu_id=2, server_args_overrides={"device": "xpu"}
        )


_MODELS = sorted(p.parent.name for p in _MODELS_DIR.glob("*/config.py"))


@pytest.mark.parametrize("factory_name", ["image_encoder", "audio_encoder"])
def test_qwen3_omni_encoder_stages_resolve_none_to_the_platform(
    monkeypatch: pytest.MonkeyPatch, factory_name: str
) -> None:
    from sglang_omni.models.qwen3_omni import stages
    from sglang_omni.scheduling import simple_scheduler

    built: dict[str, object] = {}

    class _Encoder:
        def __init__(
            self,
            *,
            model_path,
            device,
            dtype,
            enable_layer_cuda_graph: bool | None = None,
        ):
            del model_path, dtype
            built["device"] = device
            built["enable_layer_cuda_graph"] = enable_layer_cuda_graph

        def __getattr__(self, name):
            del name
            return 2

    encoder_attr = {
        "image_encoder": "Qwen3OmniImageEncoder",
        "audio_encoder": "Qwen3OmniAudioEncoder",
    }[factory_name]
    monkeypatch.setattr(stages, encoder_attr, _Encoder)
    monkeypatch.setattr(
        simple_scheduler, "SimpleScheduler", lambda *a, **k: SimpleNamespace()
    )

    getattr(stages, f"create_{factory_name}_executor")("unused", device=None)

    import torch

    built_device = torch.device(built["device"])
    assert built_device.type == platforms.current_platform.device_type
    if built_device.type != "cpu":
        # Placement was not requested, so the backend's current card is bound.
        assert built_device.index is not None
    assert built["enable_layer_cuda_graph"] is (
        False if factory_name == "audio_encoder" else None
    )


def test_qwen3_omni_code2wav_resolves_none_to_a_concrete_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.qwen3_omni.components import code2wav_scheduler

    model = SimpleNamespace(
        total_upsample=1, config=SimpleNamespace(num_quantizers=4), eval=lambda: None
    )
    model.eval = lambda: model
    monkeypatch.setattr(
        code2wav_scheduler, "load_code2wav_model", lambda *a, **k: model
    )

    scheduler = code2wav_scheduler.create_code2wav_scheduler("unused", device=None)

    assert scheduler._device.type == platforms.current_platform.device_type
    if platforms.current_platform.device_type != "cpu":
        # Placement was not requested, so the backend's current card is bound.
        # A cpu device correctly carries no index.
        assert scheduler._device.index is not None


def test_qwen3_asr_stage_forwards_none_to_the_shared_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The factory must hand None down rather than substitute a literal.

    Patching the base builder's build() also proves it is the builder in play: a
    factory using an unrelated builder would leave this spy untouched. What build()
    then does with None is covered in test_server_args_builder_device.py.
    """
    from sglang_omni.models.qwen3_asr import stages
    from sglang_omni.scheduling import engine_factory

    seen: dict[str, object] = {}

    def spy_build(self, model_path, **kwargs):
        del self, model_path
        seen.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        engine_factory.SGLangGenerationEngineBuilder, "build", spy_build
    )

    stages.create_sglang_qwen3_asr_executor("unused", device=None, gpu_id=1)

    assert "device" in seen, "the factory did not route through the shared builder"
    assert seen["device"] is None
    # Placement injects gpu_id only when the signature declares it. Without it the
    # builder resolved a bare accelerator and told SGLang card 0.
    assert seen["gpu_id"] == 1


# note (lennox): this topology's own placement policy rejects process replicas
# (models/qwen3_omni/placement.py).
_REPLICA_REJECTED_TOPOLOGIES = {
    ("qwen3_omni", "speech-colocated"),
}


def _config_with_one_replicated_process(config_cls, model):
    from sglang_omni.config.schema import ProcessConfig
    from sglang_omni.config.topology import stage_process_name

    config = config_cls(model_path="unused")
    by_process: dict[str, list] = {}
    for stage in config.stages:
        by_process.setdefault(stage_process_name(stage), []).append(stage)
    candidates = []
    for process_name, members in by_process.items():
        gpu_members = [
            stage
            for stage in members
            if stage.gpu is not None
            and (model, stage.name) not in _CPU_ONLY_GPU_PLACED
            and stage.tp_size == 1
        ]
        if gpu_members and all(stage.tp_size == 1 for stage in members):
            candidates.append((process_name, gpu_members))
    if not candidates:
        return None, None
    # A process with a single GPU stage needs no colocation memory budget; fall
    # back to the fullest process and budget every member when none exists.
    singles = [c for c in candidates if len(c[1]) == 1]
    process_name, gpu_members = (
        singles[0] if singles else max(candidates, key=lambda c: len(c[1]))
    )
    stages_cfg = list(config.stages)
    if len(gpu_members) > 1:
        member_names = {stage.name for stage in gpu_members}
        stages_cfg = [
            (
                stage.model_copy(update={"gpu_memory_fraction": 0.3})
                if stage.name in member_names
                else stage
            )
            for stage in stages_cfg
        ]
    processes = dict(config.processes)
    existing = processes.get(process_name)
    replica_fields = {"num_replicas": 2, "replica_devices": [2, 3]}
    processes[process_name] = (
        existing.model_copy(update=replica_fields)
        if existing is not None
        else ProcessConfig(**replica_fields)
    )
    return (
        config.model_copy(update={"stages": stages_cfg, "processes": processes}),
        [stage.name for stage in gpu_members],
    )


@pytest.mark.parametrize(
    "model,label",
    sorted({(m, l) for m, l, _ in _iter_stages()}),
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_every_model_routes_each_process_replica_to_its_own_gpu(
    monkeypatch, model, label
):
    """Replicating any model's GPU process must hand each replica its own card;
    a factory that cannot take that gpu_id is exactly what disables replicas."""
    from sglang_omni.config.runtime import (
        apply_typed_stage_kwargs,
        resolve_factory_signature_args,
    )
    from sglang_omni.pipeline import runtime_config
    from sglang_omni.pipeline.mp_runner import _build_stage_groups
    from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime
    from tests.unit_test.fixtures.pipeline_fakes import FakeMpContext

    module = importlib.import_module(f"sglang_omni.models.{model}.config")
    topologies = {}
    if getattr(module, "EntryClass", None) is not None:
        topologies["default"] = module.EntryClass
    topologies.update(getattr(module, "Variants", None) or {})
    config, replicated_stages = _config_with_one_replicated_process(
        topologies[label], model
    )
    if config is None:
        pytest.skip("no tp_size=1 GPU process to replicate")

    # note (lennox): the replica cards (2, 3) need not exist on this host; the
    # test stops at the launch specs, before any process or device is touched.
    monkeypatch.setattr(runtime_config, "_visible_device_count", lambda: None)
    # note (lennox): budget here is unrelated to gpu routing, set it only for local test run on Windows.
    monkeypatch.setattr(runtime_config, "_IPC_SUN_PATH_BUDGET", 10_000)

    if (model, label) in _REPLICA_REJECTED_TOPOLOGIES:
        with pytest.raises(ValueError, match="does not support process replicas"):
            prepare_pipeline_runtime(config)
        return

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
    for replica_id, expected_gpu in ((0, 2), (1, 3)):
        for stage_name in replicated_stages:
            spec = specs[f"{stage_name}@r{replica_id}"]
            assert spec.factory_arg_defaults["gpu_id"] == expected_gpu
            try:
                factory = import_string(spec.factory)
            except ImportError as exc:
                pytest.skip(f"optional dependency missing: {exc}")
            factory_args = apply_typed_stage_kwargs(
                factory,
                spec.factory_kwargs,
                spec.typed_kwargs,
                stage_name=spec.stage_name,
            )
            factory_args = resolve_factory_signature_args(
                factory,
                factory_args,
                defaults=spec.factory_arg_defaults,
                require_gpu_id=spec.require_factory_gpu_id,
                stage_name=spec.stage_name,
            )
            assert factory_args["gpu_id"] == expected_gpu
