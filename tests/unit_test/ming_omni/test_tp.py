# SPDX-License-Identifier: Apache-2.0
"""TP=2 support tests."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterable
from types import ModuleType, SimpleNamespace

import pytest

_PROJECT_PREFIX = "sglang_omni."


def _ming_config_with_thinker_tp2(config_cls):
    config = config_cls(model_path="dummy")
    stages = [stage.model_copy(deep=True) for stage in config.stages]
    for stage in stages:
        if stage.name == "thinker":
            stage.tp_size = 2
            stage.parallelism = stage.parallelism.model_copy(update={"tp": 2})
            stage.gpu = [0, 1]
    return stages


def _build_groups(config, build_stage_groups, prepare_pipeline_runtime):
    prep = prepare_pipeline_runtime(config)
    try:
        return build_stage_groups(
            config,
            stages_cfg=prep.stages_cfg,
            name_map=prep.name_map,
            endpoints=prep.endpoints,
            placement_plan=prep.placement_plan,
            process_plan=prep.process_plan,
        )
    finally:
        if prep.runtime_dir is not None:
            prep.runtime_dir.close()


def _thinker_specs(config, build_stage_groups, prepare_pipeline_runtime):
    groups = _build_groups(config, build_stage_groups, prepare_pipeline_runtime)
    return [s for g in groups for s in g.specs if s.stage_name == "thinker"]


def _module_refs_any(module: ModuleType, refs: Iterable[object]) -> bool:
    ref_ids = {id(ref) for ref in refs}
    return any(id(value) in ref_ids for value in vars(module).values())


def _purge_project_modules_with_fake_refs(
    snapshot: set[str],
    refs: Iterable[object],
) -> set[str]:
    purged: set[str] = set()
    for name, module in list(sys.modules.items()):
        if not name.startswith(_PROJECT_PREFIX) or not isinstance(module, ModuleType):
            continue
        if name not in snapshot or _module_refs_any(module, refs):
            _remove_module(name, module)
            purged.add(name)
    _remove_cached_attrs_from_purged_modules(purged)
    return purged


def _remove_module(name: str, module: ModuleType) -> None:
    sys.modules.pop(name, None)
    parent_name, _, child_name = name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if isinstance(parent, ModuleType) and getattr(parent, child_name, None) is module:
        delattr(parent, child_name)


def _remove_cached_attrs_from_purged_modules(purged: set[str]) -> None:
    if not purged:
        return
    for module in list(sys.modules.values()):
        if not isinstance(module, ModuleType) or not module.__name__.startswith(
            _PROJECT_PREFIX
        ):
            continue
        for attr_name, value in list(vars(module).items()):
            value_module = getattr(value, "__module__", "")
            if _module_name_matches_any(value_module, purged):
                delattr(module, attr_name)


def _module_name_matches_any(module_name: str, candidates: set[str]) -> bool:
    return any(
        module_name == candidate or module_name.startswith(f"{candidate}.")
        for candidate in candidates
    )


def _project_modules_with_ref(ref: object) -> list[str]:
    return [
        name
        for name, module in sys.modules.items()
        if name.startswith(_PROJECT_PREFIX)
        and isinstance(module, ModuleType)
        and _module_refs_any(module, (ref,))
    ]


def _cached_attrs_from_missing_project_modules() -> list[str]:
    stale_attrs: list[str] = []
    for module_name, module in sys.modules.items():
        if not module_name.startswith(_PROJECT_PREFIX) or not isinstance(
            module, ModuleType
        ):
            continue
        for attr_name, value in vars(module).items():
            value_module = getattr(value, "__module__", "")
            if not value_module.startswith(_PROJECT_PREFIX):
                continue
            if value_module not in sys.modules:
                stale_attrs.append(f"{module_name}.{attr_name}->{value_module}")
    return stale_attrs


def test_ming_thinker_tp2_builds_rank_specific_stage_specs(monkeypatch) -> None:
    importlib.import_module("sglang_omni.pipeline")
    before_import = set(sys.modules)
    fake_torch = ModuleType("torch")
    fake_torch.Tensor = object
    fake_torch_nn = ModuleType("torch.nn")
    fake_torch_nn.Module = object
    fake_torch_distributed = ModuleType("torch.distributed")
    fake_torch_distributed.ProcessGroup = object
    fake_torch.distributed = fake_torch_distributed
    fake_profiler = ModuleType("torch.profiler")
    fake_profiler.ProfilerActivity = SimpleNamespace(CPU="cpu", CUDA="cuda")

    class FakeProfile:
        def __init__(self, *args, **kwargs):
            pass

    fake_profiler.profile = FakeProfile

    class FakePretrainedConfig:
        def __init__(self, **kwargs):
            for name, value in kwargs.items():
                setattr(self, name, value)

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoConfig = SimpleNamespace(
        from_pretrained=lambda *a, **k: None,
        register=lambda *a, **k: None,
    )
    fake_transformers.PretrainedConfig = FakePretrainedConfig
    fake_transformers_init = ModuleType("transformers.initialization")
    fake_transformers_init.no_init_weights = lambda: SimpleNamespace(
        __enter__=lambda self: None,
        __exit__=lambda self, exc_type, exc, tb: None,
    )
    fake_transformers_utils = ModuleType("transformers.utils")
    fake_transformers_hub = ModuleType("transformers.utils.hub")
    fake_transformers_hub.cached_file = lambda *a, **k: ""
    fake_msgpack = ModuleType("msgpack")
    fake_zmq = ModuleType("zmq")
    fake_zmq_asyncio = ModuleType("zmq.asyncio")
    fake_zmq_asyncio.Context = object
    fake_zmq_asyncio.Socket = object
    fake_zmq.asyncio = fake_zmq_asyncio
    fake_zmq.PUSH = object()
    fake_zmq.PULL = object()
    fake_zmq.PUB = object()
    fake_zmq.SUB = object()
    fake_zmq.SUBSCRIBE = object()
    fake_zmq.POLLIN = 1
    fake_nixl = ModuleType("sglang_omni.relay.nixl")

    class FakeConnection:
        pass

    class FakeNixlOperation:
        pass

    class FakeNixlRelay:
        pass

    fake_nixl.NIXL_AVAILABLE = False
    fake_nixl.Connection = FakeConnection
    fake_nixl.NixlOperation = FakeNixlOperation
    fake_nixl.NixlRelay = FakeNixlRelay
    fake_refs = (
        fake_torch,
        fake_torch_nn,
        fake_torch_distributed,
        fake_profiler,
        fake_transformers,
        fake_transformers_init,
        fake_transformers_utils,
        fake_transformers_hub,
        fake_msgpack,
        fake_zmq,
        fake_zmq_asyncio,
        fake_nixl,
        FakeConnection,
        FakeNixlOperation,
        FakeNixlRelay,
    )

    try:
        with monkeypatch.context() as mp:
            mp.setitem(sys.modules, "torch", fake_torch)
            mp.setitem(sys.modules, "torch.nn", fake_torch_nn)
            mp.setitem(sys.modules, "torch.distributed", fake_torch_distributed)
            mp.setitem(sys.modules, "torch.profiler", fake_profiler)
            mp.setitem(sys.modules, "transformers", fake_transformers)
            mp.setitem(
                sys.modules, "transformers.initialization", fake_transformers_init
            )
            mp.setitem(sys.modules, "transformers.utils", fake_transformers_utils)
            mp.setitem(sys.modules, "transformers.utils.hub", fake_transformers_hub)
            mp.setitem(sys.modules, "msgpack", fake_msgpack)
            mp.setitem(sys.modules, "zmq", fake_zmq)
            mp.setitem(sys.modules, "zmq.asyncio", fake_zmq_asyncio)
            mp.setitem(sys.modules, "sglang_omni.relay.nixl", fake_nixl)

            from sglang_omni.models.ming_omni.config import MingOmniPipelineConfig
            from sglang_omni.pipeline.mp_runner import _build_stage_groups
            from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime

            config = MingOmniPipelineConfig(
                model_path="dummy",
                stages=_ming_config_with_thinker_tp2(MingOmniPipelineConfig),
            )

            groups = _build_groups(
                config, _build_stage_groups, prepare_pipeline_runtime
            )
            all_specs = [s for g in groups for s in g.specs]
            specs = [s for s in all_specs if s.stage_name == "thinker"]
            cpu_stage_specs = {
                s.stage_name: s
                for s in all_specs
                if s.stage_name in {"preprocessing", "mm_aggregate", "decode"}
            }
            assert {name: spec.gpu_id for name, spec in cpu_stage_specs.items()} == {
                "preprocessing": None,
                "mm_aggregate": None,
                "decode": None,
            }
            assert [spec.role for spec in specs] == ["leader", "follower"]
            assert [spec.gpu_id for spec in specs] == [0, 1]
            assert [spec.tp_rank for spec in specs] == [0, 1]
            assert {spec.tp_size for spec in specs} == {2}
            assert specs[0].nccl_port is not None
            assert specs[0].nccl_port == specs[1].nccl_port
            assert all("gpu_id" not in spec.factory_args for spec in specs)
            assert [spec.factory_arg_defaults["gpu_id"] for spec in specs] == [0, 1]
            assert [spec.factory_args["tp_rank"] for spec in specs] == [0, 1]
            assert {spec.factory_args["tp_size"] for spec in specs} == {2}
            assert specs[0].factory_args["nccl_port"] == specs[0].nccl_port
            assert specs[1].factory_args["nccl_port"] == specs[0].nccl_port

            explicit_stages = _ming_config_with_thinker_tp2(MingOmniPipelineConfig)
            for stage in explicit_stages:
                if stage.name == "thinker":
                    stage.gpu = [2, 4]
            explicit_config = MingOmniPipelineConfig(
                model_path="dummy",
                stages=explicit_stages,
            )
            explicit_specs = _thinker_specs(
                explicit_config, _build_stage_groups, prepare_pipeline_runtime
            )
            assert [spec.gpu_id for spec in explicit_specs] == [2, 4]
    finally:
        _purge_project_modules_with_fake_refs(before_import, fake_refs)

    assert _project_modules_with_ref(fake_torch) == []
    assert _cached_attrs_from_missing_project_modules() == []


def test_ming_speech_rejects_talker_inside_explicit_thinker_tp_gpus() -> None:
    from sglang_omni.models.ming_omni.config import MingOmniSpeechPipelineConfig

    stages = _ming_config_with_thinker_tp2(MingOmniSpeechPipelineConfig)
    for stage in stages:
        if stage.name == "talker":
            stage.gpu = 1

    with pytest.raises(ValueError, match="collides"):
        MingOmniSpeechPipelineConfig(model_path="dummy", stages=stages)


@pytest.mark.parametrize(
    ("config_cls_name", "stage_name", "gpu"),
    [
        ("MingOmniPipelineConfig", "audio_encoder", [0, 1]),
        ("MingOmniSpeechPipelineConfig", "audio_encoder", [0, 1]),
        ("MingOmniSpeechPipelineConfig", "talker", [2, 3]),
        ("MingOmniStreamingSpeechPipelineConfig", "audio_encoder", [2, 3]),
        ("MingOmniStreamingSpeechPipelineConfig", "segmenter", None),
        ("MingOmniStreamingSpeechPipelineConfig", "talker_stream", [2, 3]),
    ],
)
def test_ming_rejects_non_ar_stage_tp_size_gt_one(
    config_cls_name: str,
    stage_name: str,
    gpu: list[int] | None,
) -> None:
    import sglang_omni.models.ming_omni.config as ming_config

    config_cls = getattr(ming_config, config_cls_name)
    config = config_cls(model_path="dummy")
    stages = [stage.model_copy(deep=True) for stage in config.stages]
    for stage in stages:
        if stage.name == stage_name:
            stage.tp_size = 2
            stage.parallelism = stage.parallelism.model_copy(update={"tp": 2})
            stage.gpu = gpu

    with pytest.raises(ValueError, match=f"{stage_name}.*does not support TP"):
        config_cls(model_path="dummy", stages=stages)


def test_ming_streaming_speech_allows_thinker_tp_size_gt_one() -> None:
    from sglang_omni.models.ming_omni.config import (
        MingOmniStreamingSpeechPipelineConfig,
    )

    stages = _ming_config_with_thinker_tp2(MingOmniStreamingSpeechPipelineConfig)
    for stage in stages:
        if stage.name == "talker_stream":
            stage.gpu = 2

    config = MingOmniStreamingSpeechPipelineConfig(model_path="dummy", stages=stages)

    assert config.gpu_placement["thinker"] == [0, 1]
    assert config.gpu_placement["talker_stream"] == 2


def test_ming_text_allows_image_encoder_tp_size_gt_one() -> None:
    from sglang_omni.models.ming_omni.config import MingOmniPipelineConfig

    config = MingOmniPipelineConfig(model_path="dummy")
    stages = [stage.model_copy(deep=True) for stage in config.stages]
    for stage in stages:
        if stage.name == "image_encoder":
            stage.tp_size = 2
            stage.parallelism = stage.parallelism.model_copy(update={"tp": 2})
            stage.gpu = [2, 3]

    rebuilt = MingOmniPipelineConfig(model_path="dummy", stages=stages)

    assert rebuilt.gpu_placement["image_encoder"] == [2, 3]


def test_ming_speech_rejects_talker_on_non_contiguous_thinker_tp_gpu() -> None:
    from sglang_omni.models.ming_omni.config import MingOmniSpeechPipelineConfig

    stages = _ming_config_with_thinker_tp2(MingOmniSpeechPipelineConfig)
    for stage in stages:
        if stage.name == "thinker":
            stage.gpu = [0, 2]
        if stage.name == "talker":
            stage.gpu = 2

    with pytest.raises(ValueError, match="collides"):
        MingOmniSpeechPipelineConfig(model_path="dummy", stages=stages)


def test_ming_speech_rejects_any_talker_gpu_list_overlap_with_thinker_tp() -> None:
    from sglang_omni.models.ming_omni.config import MingOmniSpeechPipelineConfig

    stages = _ming_config_with_thinker_tp2(MingOmniSpeechPipelineConfig)
    for stage in stages:
        if stage.name == "thinker":
            stage.gpu = [0, 2]
        if stage.name == "talker":
            stage.gpu = [2]

    with pytest.raises(ValueError, match="collides"):
        MingOmniSpeechPipelineConfig(model_path="dummy", stages=stages)


def test_ming_speech_allows_talker_outside_explicit_thinker_tp_gpus() -> None:
    from sglang_omni.models.ming_omni.config import MingOmniSpeechPipelineConfig

    stages = _ming_config_with_thinker_tp2(MingOmniSpeechPipelineConfig)
    for stage in stages:
        if stage.name == "talker":
            stage.gpu = 2

    config = MingOmniSpeechPipelineConfig(model_path="dummy", stages=stages)

    assert config.gpu_placement["thinker"] == [0, 1]
    assert config.gpu_placement["talker"] == 2


def test_ming_bootstrap_aligns_server_args_tp_size_before_infra(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    common_module = ModuleType("sglang_omni.models.ming_omni.components.common")
    common_module.load_ming_tokenizer = lambda _model_path: SimpleNamespace(
        vocab_size=32000,
        eos_token_id=2,
        unk_token_id=0,
        convert_tokens_to_ids=lambda token: 0,
    )
    common_module.load_ming_config = lambda _model_path: SimpleNamespace(
        llm_config=SimpleNamespace(vocab_size=32000)
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.ming_omni.components.common",
        common_module,
    )

    runner_module = ModuleType("sglang_omni.model_runner.ming_thinker_model_runner")

    class FakeMingThinkerModelRunner:
        def __init__(self, model_worker, output_proc):
            self.model_worker = model_worker
            self.output_proc = output_proc

    runner_module.MingThinkerModelRunner = FakeMingThinkerModelRunner
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.model_runner.ming_thinker_model_runner",
        runner_module,
    )

    scheduling_bootstrap_module = ModuleType("sglang_omni.scheduling.bootstrap")

    def fake_create_sglang_infrastructure(
        server_args,
        gpu_id,
        *,
        tp_rank,
        nccl_port,
        model_arch_override,
    ):
        captured["server_args_tp_size"] = server_args.tp_size
        captured["gpu_id"] = gpu_id
        captured["tp_rank"] = tp_rank
        captured["nccl_port"] = nccl_port
        captured["model_arch_override"] = model_arch_override
        model = object()
        model_worker = SimpleNamespace(
            model_runner=SimpleNamespace(model=model),
        )
        return (
            model_worker,
            "tree_cache",
            "req_to_token_pool",
            "token_to_kv_pool_allocator",
            "prefill_mgr",
            "decode_mgr",
            "model_config",
        )

    scheduling_bootstrap_module.create_sglang_infrastructure = (
        fake_create_sglang_infrastructure
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.scheduling.bootstrap",
        scheduling_bootstrap_module,
    )

    omni_scheduler_module = ModuleType("sglang_omni.scheduling.omni_scheduler")

    class FakeOmniScheduler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    omni_scheduler_module.OmniScheduler = FakeOmniScheduler
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.scheduling.omni_scheduler",
        omni_scheduler_module,
    )

    sglang_backend_module = ModuleType("sglang_omni.scheduling.sglang_backend")

    class FakeSGLangOutputProcessor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    sglang_backend_module.SGLangOutputProcessor = FakeSGLangOutputProcessor
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.scheduling.sglang_backend",
        sglang_backend_module,
    )

    bootstrap = importlib.import_module("sglang_omni.models.ming_omni.bootstrap")
    server_args = SimpleNamespace(tp_size=1)

    scheduler = bootstrap.create_thinker_scheduler(
        server_args,
        model_path="dummy",
        gpu_id=1,
        tp_rank=1,
        tp_size=2,
        nccl_port=29500,
    )

    assert captured["server_args_tp_size"] == 2
    assert server_args.tp_size == 2
    assert captured["gpu_id"] == 1
    assert captured["tp_rank"] == 1
    assert captured["nccl_port"] == 29500
    assert captured["model_arch_override"] == "BailingMoeV2ForCausalLM"
    assert scheduler.kwargs["server_args"] is server_args


@pytest.mark.asyncio
async def test_tp_leader_skips_fanout_work_for_omni_scheduler() -> None:
    """OmniScheduler-backed leaders must not call fanout_work()."""
    import queue as _queue_mod
    from unittest.mock import MagicMock

    importlib.import_module("sglang_omni.pipeline")
    before_import = set(sys.modules)
    fake_torch = ModuleType("torch")
    fake_torch.Tensor = object
    fake_torch.uint8 = object()
    fake_profiler = ModuleType("torch.profiler")
    fake_profiler.ProfilerActivity = SimpleNamespace(CPU="cpu", CUDA="cuda")

    class FakeProfile:
        def __init__(self, *args, **kwargs):
            pass

    fake_profiler.profile = FakeProfile
    fake_torch.profiler = fake_profiler
    fake_nixl = ModuleType("sglang_omni.relay.nixl")

    class FakeConnection:
        pass

    class FakeNixlOperation:
        pass

    class FakeNixlRelay:
        pass

    fake_nixl.NIXL_AVAILABLE = False
    fake_nixl.Connection = FakeConnection
    fake_nixl.NixlOperation = FakeNixlOperation
    fake_nixl.NixlRelay = FakeNixlRelay
    fake_refs = (
        fake_torch,
        fake_profiler,
        fake_nixl,
        FakeConnection,
        FakeNixlOperation,
        FakeNixlRelay,
    )

    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "torch", fake_torch)
            mp.setitem(sys.modules, "torch.profiler", fake_profiler)
            mp.setitem(sys.modules, "sglang_omni.relay.nixl", fake_nixl)

            from sglang_omni.pipeline.stage.runtime import Stage
            from sglang_omni.pipeline.tp_control import TPLeaderFanout

            fanout = MagicMock(spec=TPLeaderFanout)

            omni_like = SimpleNamespace(
                inbox=_queue_mod.Queue(),
                requires_tp_work_fanout=False,
            )
            simple_like = SimpleNamespace(
                inbox=_queue_mod.Queue(),
                requires_tp_work_fanout=True,
            )

            def _make_stage(scheduler):
                return Stage(
                    name="test_stage",
                    role="leader",
                    get_next=lambda _name: [],
                    gpu_id=0,
                    endpoints={},
                    control_plane=MagicMock(),
                    relay=MagicMock(),
                    scheduler=scheduler,
                    tp_fanout=fanout,
                )

            payload = SimpleNamespace(request_id="req-1")

            stage_omni = _make_stage(omni_like)
            await stage_omni._execute(payload)
            fanout.fanout_work.assert_not_called()
            assert omni_like.inbox.get_nowait().request_id == "req-1"

            fanout.reset_mock()

            stage_simple = _make_stage(simple_like)
            await stage_simple._execute(payload)
            fanout.fanout_work.assert_called_once_with(payload)
            assert simple_like.inbox.get_nowait().request_id == "req-1"
    finally:
        _purge_project_modules_with_fake_refs(before_import, fake_refs)

    assert _project_modules_with_ref(fake_torch) == []
    assert _cached_attrs_from_missing_project_modules() == []
