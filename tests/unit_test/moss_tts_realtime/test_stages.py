# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sglang_omni.models.moss_tts_realtime import request_builders, stages
from sglang_omni.models.moss_tts_realtime.engine_builder import (
    MossTTSRealtimeEngineBuilder,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


def _builder(**overrides: Any) -> MossTTSRealtimeEngineBuilder:
    values: dict[str, Any] = {
        "max_seq_len": 40960,
        "total_gpu_memory_fraction": 0.90,
        "max_sessions": 7,
        "max_held_sessions": 5,
        "max_active_turns": 3,
        "max_pending_text_tokens": 64,
        "max_pending_text_bytes": 2048,
        "max_input_updates": 32,
        "terminal_tombstone_limit": 77,
        "input_idle_timeout_s": 1.5,
        "turn_timeout_s": 2.5,
        "session_idle_ttl_s": 3.5,
    }
    values.update(overrides)
    return MossTTSRealtimeEngineBuilder(**values)


def test_transformers_compat_hashes_hf_snapshot_symlink_sources(tmp_path) -> None:
    from transformers import dynamic_module_utils

    blobs = tmp_path / "blobs"
    snapshot = tmp_path / "snapshots" / "revision"
    blobs.mkdir()
    snapshot.mkdir(parents=True)
    config_blob = blobs / "config-hash"
    model_blob = blobs / "model-hash"
    config_blob.write_text("class CodecConfig: pass\n", encoding="utf-8")
    model_blob.write_text(
        "from .configuration_codec import CodecConfig\n",
        encoding="utf-8",
    )
    (snapshot / "configuration_codec.py").symlink_to(config_blob)
    model_source = snapshot / "modeling_codec.py"
    model_source.symlink_to(model_blob)

    original = dynamic_module_utils._compute_local_source_files_hash
    with stages.moss_transformers_processor_compat():
        source_hash = dynamic_module_utils._compute_local_source_files_hash(
            snapshot,
            model_source,
        )

    assert len(source_hash) == 16
    assert dynamic_module_utils._compute_local_source_files_hash is original


def test_load_processor_uses_checkpoint_auto_map(monkeypatch) -> None:
    import transformers

    calls: dict[str, Any] = {}
    processor = object()
    loaded_config = object()

    monkeypatch.setattr(stages, "resolve_checkpoint", lambda _: "/resolved")
    monkeypatch.setattr(
        stages,
        "moss_transformers_processor_compat",
        contextlib.nullcontext,
    )
    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda checkpoint_dir, **kwargs: (
            calls.__setitem__("processor_checkpoint", checkpoint_dir),
            calls.__setitem__("processor_kwargs", kwargs),
            processor,
        )[-1],
    )
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda checkpoint_dir, **kwargs: (
            calls.__setitem__("config_checkpoint", checkpoint_dir),
            calls.__setitem__("config_kwargs", kwargs),
            loaded_config,
        )[-1],
    )
    monkeypatch.setattr(
        stages,
        "bind_moss_tts_realtime_processor_config",
        lambda config, processor: calls.__setitem__(
            "config_binding",
            (config, processor),
        ),
    )

    actual = stages.load_moss_tts_realtime_processor("model")

    assert actual is processor
    assert calls == {
        "processor_checkpoint": "/resolved",
        "processor_kwargs": {"trust_remote_code": True},
        "config_checkpoint": "/resolved",
        "config_kwargs": {"trust_remote_code": True},
        "config_binding": (loaded_config, processor),
    }


def test_engine_pre_infra_reuses_processor_model_config(monkeypatch) -> None:
    model_config = SimpleNamespace(
        language_config=SimpleNamespace(max_position_embeddings=2048),
        delay_tokens_len=12,
    )
    processor = SimpleNamespace(model_config=model_config)
    monkeypatch.setattr(
        stages,
        "load_moss_tts_realtime_processor",
        lambda checkpoint_dir: processor,
    )
    builder = _builder(max_seq_len=None, total_gpu_memory_fraction=None)

    builder.pre_infra_setup("checkpoint")

    assert builder.processor is processor
    assert builder.context_length == 2048


@pytest.mark.parametrize(
    ("component", "dtype", "expected_keys"),
    [
        ("encoder", None, {"encoder.weight", "quantizer.weight"}),
        ("decoder", None, {"decoder.weight", "quantizer.weight"}),
        (
            "decoder",
            torch.bfloat16,
            {"decoder.weight", "quantizer.weight"},
        ),
    ],
)
def test_codec_loader_reads_only_requested_component_weights(
    monkeypatch,
    tmp_path,
    component: str,
    dtype: torch.dtype | None,
    expected_keys: set[str],
) -> None:
    import safetensors
    import transformers
    from safetensors.torch import save_file
    from torch import nn

    class TinyCodec(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Linear(2, 2, bias=False)
            self.decoder = nn.Linear(2, 2, bias=False)
            self.quantizer = nn.Linear(2, 2, bias=False)
            self.config = object()

    expected_weights = {
        "encoder.weight": torch.full((2, 2), 1.0),
        "decoder.weight": torch.full((2, 2), 2.0),
        "quantizer.weight": torch.full((2, 2), 3.0),
    }
    save_file(
        {
            "encoder.weight": expected_weights["encoder.weight"],
            "quantizer.weight": expected_weights["quantizer.weight"],
        },
        tmp_path / "model-00001-of-00002.safetensors",
    )
    save_file(
        {"decoder.weight": expected_weights["decoder.weight"]},
        tmp_path / "model-00002-of-00002.safetensors",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "encoder.weight": "model-00001-of-00002.safetensors",
                    "quantizer.weight": "model-00001-of-00002.safetensors",
                    "decoder.weight": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(stages, "resolve_checkpoint", lambda _: tmp_path)
    monkeypatch.setattr(
        stages,
        "moss_transformers_processor_compat",
        contextlib.nullcontext,
    )
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_config",
        lambda *args, **kwargs: TinyCodec(),
    )

    loaded_keys: set[str] = set()
    real_safe_open = safetensors.safe_open

    class TrackingSafeOpen:
        def __init__(self, filename, *args, **kwargs) -> None:
            self._context = real_safe_open(filename, *args, **kwargs)
            self._reader = None

        def __enter__(self):
            self._reader = self._context.__enter__()
            return self

        def __exit__(self, *args):
            return self._context.__exit__(*args)

        def get_tensor(self, name: str) -> torch.Tensor:
            loaded_keys.add(name)
            return self._reader.get_tensor(name)

    monkeypatch.setattr(safetensors, "safe_open", TrackingSafeOpen)

    codec = stages.load_moss_tts_realtime_codec(
        "codec",
        component=component,
        device="cpu",
        dtype=dtype,
    )

    assert loaded_keys == expected_keys
    assert set(codec.state_dict()) == expected_keys
    for name, value in codec.state_dict().items():
        assert value.device.type == "cpu"
        expected_dtype = (
            dtype
            if dtype is not None and name.startswith(f"{component}.")
            else torch.float32
        )
        assert value.dtype is expected_dtype
        torch.testing.assert_close(value.float(), expected_weights[name])


def test_codec_memory_estimate_scales_streaming_state_with_stream_slots(
    monkeypatch,
) -> None:
    import transformers
    from torch import nn

    class StateModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._streaming_state: Any | None = None

    class TinyCodec(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Linear(2, 2, bias=False)
            self.decoder = nn.Sequential(
                nn.Linear(2, 3, bias=False),
                StateModule(),
            )
            self.quantizer = nn.Linear(2, 2, bias=False)

        @contextlib.contextmanager
        def streaming(self, batch_size: int):
            cache = torch.empty(batch_size, 5, dtype=torch.float32)
            state = SimpleNamespace(
                cache=cache,
                cache_alias=cache,
                exec_mask=torch.empty(batch_size, dtype=torch.bool),
            )
            self.decoder[1]._streaming_state = state
            try:
                yield
            finally:
                self.decoder[1]._streaming_state = None

    monkeypatch.setattr(stages, "resolve_checkpoint", lambda _: "/codec")
    monkeypatch.setattr(
        stages,
        "moss_transformers_processor_compat",
        contextlib.nullcontext,
    )
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_config",
        lambda *args, **kwargs: TinyCodec(),
    )

    decoder_bytes, state_bytes = stages.estimate_moss_tts_realtime_codec_memory(
        "codec",
        stream_slots=4,
    )

    assert decoder_bytes == 40
    assert state_bytes == 84


def test_create_preprocessing_executor_wires_codec_and_cleanup(monkeypatch) -> None:
    calls: dict[str, Any] = {}
    processor = SimpleNamespace(model_config=SimpleNamespace(rvq=16))
    codec = object()
    codec_encoder = object()
    audio_encoder = object()
    reference_cache = object()

    def fake_load_processor(model_path: str) -> object:
        calls["processor_path"] = model_path
        return processor

    monkeypatch.setattr(
        stages,
        "load_moss_tts_realtime_processor",
        fake_load_processor,
    )

    def fake_load_codec(
        model_path: str,
        *,
        component: str,
        device: str,
        dtype: torch.dtype | None = None,
    ) -> object:
        calls["codec"] = (model_path, component, device, dtype)
        return codec

    class FakeAudioEncoder:
        def __new__(
            cls,
            loaded_codec: Any,
            *,
            device: str,
            num_quantizers: int,
        ) -> object:
            calls["encoder"] = (loaded_codec, device, num_quantizers)
            return codec_encoder

    class FakeBatchedEncoder:
        def __new__(
            cls,
            loaded_encoder: Any,
            *,
            max_batch_size: int,
            max_batch_wait_ms: int,
        ) -> object:
            calls["batched"] = (
                loaded_encoder,
                max_batch_size,
                max_batch_wait_ms,
            )
            return audio_encoder

    class FakeReferenceEncoder:
        def __new__(
            cls,
            loaded_encoder: Any,
            **kwargs: Any,
        ) -> object:
            calls["reference_cache"] = (loaded_encoder, kwargs)
            return reference_cache

    def fake_set_context(
        *,
        processor: Any,
        audio_encoder: Any,
        reference_encoder: Any,
    ) -> None:
        calls["context"] = (processor, audio_encoder, reference_encoder)

    monkeypatch.setattr(stages, "load_moss_tts_realtime_codec", fake_load_codec)
    monkeypatch.setattr(stages, "MossTTSRealtimeAudioEncoder", FakeAudioEncoder)
    monkeypatch.setattr(
        stages,
        "BatchedMossTTSRealtimeAudioEncoder",
        FakeBatchedEncoder,
    )
    monkeypatch.setattr(
        stages,
        "MossTTSRealtimeReferenceEncoder",
        FakeReferenceEncoder,
    )

    def fake_resolve_codec(model_path: str) -> str:
        calls["codec_source"] = model_path
        return "/resolved-codec"

    monkeypatch.setattr(stages, "resolve_checkpoint", fake_resolve_codec)
    monkeypatch.setattr(
        stages,
        "set_moss_tts_realtime_preprocessing_context",
        fake_set_context,
    )

    scheduler = stages.create_preprocessing_executor(
        "model",
        device=None,
        gpu_id=3,
        codec_model_path="codec",
        max_concurrency=6,
        encode_batch_size=4,
        encode_batch_wait_ms=7,
        ref_audio_cache_max_items=17,
        ref_audio_cache_max_bytes=4096,
    )

    assert isinstance(scheduler, SimpleScheduler)
    assert scheduler._fn is request_builders.preprocess_moss_tts_realtime_payload
    assert (
        scheduler._abort_callback
        is request_builders.cleanup_prepared_moss_tts_realtime_request
    )
    assert scheduler._max_concurrency == 6
    assert calls == {
        "processor_path": "model",
        "codec_source": "codec",
        "codec": ("/resolved-codec", "encoder", "cuda:3", None),
        "encoder": (codec, "cuda:3", 16),
        "batched": (codec_encoder, 4, 7),
        "reference_cache": (
            audio_encoder,
            {
                "model_revision": "/resolved-codec",
                "num_quantizers": 16,
                "max_items": 17,
                "max_bytes": 4096,
            },
        ),
        "context": (processor, audio_encoder, reference_cache),
    }


def test_create_preprocessing_executor_reference_cache_kill_switch(
    monkeypatch,
) -> None:
    processor = SimpleNamespace(model_config=SimpleNamespace(rvq=16))
    audio_encoder = object()
    contexts: list[tuple[Any, Any]] = []
    cache_calls = 0

    monkeypatch.setattr(
        stages,
        "load_moss_tts_realtime_processor",
        lambda _: processor,
    )
    monkeypatch.setattr(
        stages,
        "load_moss_tts_realtime_codec",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda path: path)
    monkeypatch.setattr(
        stages,
        "MossTTSRealtimeAudioEncoder",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        stages,
        "BatchedMossTTSRealtimeAudioEncoder",
        lambda *args, **kwargs: audio_encoder,
    )

    def create_cache(*args: Any, **kwargs: Any) -> object:
        nonlocal cache_calls
        cache_calls += 1
        return object()

    monkeypatch.setattr(stages, "MossTTSRealtimeReferenceEncoder", create_cache)
    monkeypatch.setattr(
        stages,
        "set_moss_tts_realtime_preprocessing_context",
        lambda *, processor, audio_encoder, reference_encoder: contexts.append(
            (audio_encoder, reference_encoder)
        ),
    )

    stages.create_preprocessing_executor(
        "model",
        device="cpu",
        ref_audio_cache=False,
    )
    assert contexts[-1] == (audio_encoder, audio_encoder)

    monkeypatch.setenv("MOSS_REF_AUDIO_CACHE", "0")
    stages.create_preprocessing_executor("model", device="cpu")
    assert contexts[-1] == (audio_encoder, audio_encoder)
    assert cache_calls == 0


def test_engine_factory_builds_realtime_scheduler_and_wires_outbox(
    monkeypatch,
) -> None:
    from sglang_omni.models.moss_tts_realtime import model_runner, scheduler
    from sglang_omni.scheduling import bootstrap, engine_factory, sglang_backend
    from sglang_omni.utils import gpu_memory

    calls: dict[str, Any] = {}
    runners: list[Any] = []

    def fake_build_server_args(
        model_path: str, *, context_length: int, **kwargs: Any
    ) -> Any:
        assert calls["processor_loaded"] is True
        calls["server_args"] = {
            "model_path": model_path,
            "context_length": context_length,
            **kwargs,
        }
        resolved_kwargs = dict(kwargs)
        cuda_graph_max_bs = resolved_kwargs.pop("cuda_graph_max_bs")
        cuda_graph_bs = resolved_kwargs.pop("cuda_graph_bs")
        return SimpleNamespace(
            model_path=model_path,
            context_length=context_length,
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(
                    max_bs=cuda_graph_max_bs,
                    bs=cuda_graph_bs,
                ),
                prefill=SimpleNamespace(backend="disabled", bs=None, max_bs=None),
            ),
            _cuda_graph_config_locked=set(),
            **resolved_kwargs,
        )

    language_model = object()

    def init_frame_decode_graphs(batch_sizes: list[int]) -> None:
        calls["frame_decode_graphs"] = batch_sizes

    underlying_runner = SimpleNamespace(
        model=SimpleNamespace(
            language_model=language_model,
            config=SimpleNamespace(
                language_config=SimpleNamespace(max_position_embeddings=40960)
            ),
            _decode_input_embedding=torch.nn.Embedding(7, 4),
            init_frame_decode_graphs=init_frame_decode_graphs,
        ),
        init_cuda_graphs=lambda: calls.__setitem__("cuda_graphs", True),
    )
    worker = SimpleNamespace(
        gpu_id=2,
        model_runner=underlying_runner,
        model_config=SimpleNamespace(),
        enable_prefill_input_embeds=False,
    )

    def fake_create_infrastructure(
        server_args: Any, gpu_id: int, **kwargs: Any
    ) -> tuple[Any, ...]:
        calls["infrastructure"] = (server_args, gpu_id, kwargs)
        return (
            worker,
            "tree-cache",
            "req-pool",
            "kv-pool",
            worker.model_config,
        )

    class FakeRealtimeRunner:
        def __init__(self, model_worker: Any, output_processor: Any) -> None:
            self.model_worker = model_worker
            self.output_processor = output_processor
            self.stream_outbox = None
            runners.append(self)

        def set_stream_outbox(self, outbox: Any) -> None:
            self.stream_outbox = outbox

    class FakeScheduler:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.outbox = object()
            calls["scheduler"] = self

    request_builder = object()
    result_adapter = object()
    monkeypatch.setattr(
        MossTTSRealtimeEngineBuilder,
        "pre_infra_setup",
        lambda self, checkpoint_dir: (
            calls.__setitem__("processor_loaded", True),
            setattr(self, "context_length", 40960),
            setattr(self, "processor", "processor"),
            setattr(self, "minimum_codec_mem_reserve", 0.10),
        ),
    )
    monkeypatch.setattr(
        stages,
        "bind_moss_tts_realtime_processor_config",
        lambda config, processor: calls.__setitem__(
            "processor_binding",
            (config, processor),
        ),
    )
    monkeypatch.setattr(engine_factory, "_resolve_checkpoint", lambda path: path)
    monkeypatch.setattr(
        sglang_backend,
        "build_sglang_server_args",
        fake_build_server_args,
    )
    monkeypatch.setattr(
        bootstrap,
        "create_sglang_infrastructure_defer_cuda_graph",
        lambda server_args, gpu_id, **kwargs: (
            not bool(server_args.disable_cuda_graph),
            fake_create_infrastructure(server_args, gpu_id, **kwargs),
        ),
    )
    monkeypatch.setattr(
        sglang_backend,
        "SGLangOutputProcessor",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        request_builders,
        "make_moss_tts_realtime_scheduler_adapters",
        lambda **_: (request_builder, result_adapter),
    )
    monkeypatch.setattr(
        model_runner,
        "MossTTSRealtimeModelRunner",
        FakeRealtimeRunner,
    )
    monkeypatch.setattr(scheduler, "MossTTSRealtimeScheduler", FakeScheduler)
    monkeypatch.setattr(gpu_memory, "get_process_gpu_memory_bytes", lambda _: 1024)

    built = _builder().build(
        "model",
        device="cuda:2",
        server_args_overrides={"enable_streaming_session": False},
    )

    assert built is calls["scheduler"]
    assert calls["server_args"]["context_length"] == 40960
    assert calls["server_args"]["max_running_requests"] == 7
    assert calls["server_args"]["mem_fraction_static"] == pytest.approx(0.80)
    assert "attention_backend" not in calls["server_args"]
    assert calls["server_args"]["enable_streaming_session"] is True
    assert calls["server_args"]["disable_cuda_graph"] is False
    assert calls["server_args"]["disable_overlap_schedule"] is True

    server_args, gpu_id, infra_kwargs = calls["infrastructure"]
    assert gpu_id == 2
    assert server_args.enable_streaming_session is True
    assert not hasattr(server_args, "cuda_graph_bs")
    assert infra_kwargs == {
        "total_gpu_memory_fraction": pytest.approx(0.80),
        "model_arch_override": "MossTTSRealtimeSGLangModel",
    }
    assert worker.moss_tts_realtime_max_history_frames == 40960
    assert worker.moss_tts_realtime_max_active_turns == 3
    assert calls["processor_binding"] == (
        underlying_runner.model.config,
        "processor",
    )
    assert calls["cuda_graphs"] is True
    assert calls["frame_decode_graphs"] == [1, 2, 3]

    scheduler_kwargs = built.kwargs
    assert scheduler_kwargs["request_builder"] is request_builder
    assert scheduler_kwargs["result_adapter"] is result_adapter
    assert (
        scheduler_kwargs["abort_callback"]
        is request_builders.cleanup_prepared_moss_tts_realtime_request
    )
    assert scheduler_kwargs["enable_async_decode"] is False
    for key, value in _builder().limits.model_dump().items():
        assert scheduler_kwargs[key] == value
    assert len(runners) == 1
    assert runners[0].stream_outbox is built.outbox


def test_engine_factory_honors_disabled_cuda_graph(monkeypatch) -> None:
    from sglang_omni.models.moss_tts_realtime import model_runner, scheduler
    from sglang_omni.scheduling import bootstrap, engine_factory, sglang_backend

    calls: dict[str, Any] = {"cuda_graphs": 0, "frame_decode_graphs": 0}

    def fake_build_server_args(
        model_path: str, *, context_length: int, **kwargs: Any
    ) -> Any:
        return SimpleNamespace(
            model_path=model_path,
            context_length=context_length,
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="disabled", bs=None, max_bs=None)
            ),
            _cuda_graph_config_locked=set(),
            **kwargs,
        )

    def init_cuda_graphs() -> None:
        calls["cuda_graphs"] += 1

    def init_frame_decode_graphs(_batch_sizes: list[int]) -> None:
        calls["frame_decode_graphs"] += 1

    underlying_runner = SimpleNamespace(
        model=SimpleNamespace(
            language_model=object(),
            _decode_input_embedding=torch.nn.Embedding(7, 4),
            config=SimpleNamespace(
                language_config=SimpleNamespace(max_position_embeddings=40960)
            ),
            init_frame_decode_graphs=init_frame_decode_graphs,
        ),
        init_cuda_graphs=init_cuda_graphs,
    )
    worker = SimpleNamespace(
        gpu_id=0,
        model_runner=underlying_runner,
        model_config=SimpleNamespace(),
        enable_prefill_input_embeds=False,
    )

    def fake_deferred_infrastructure(
        server_args: Any, gpu_id: int, **kwargs: Any
    ) -> tuple[bool, tuple[Any, ...]]:
        del gpu_id, kwargs
        return (
            not bool(server_args.disable_cuda_graph),
            (
                worker,
                "tree-cache",
                "req-pool",
                "kv-pool",
                worker.model_config,
            ),
        )

    class FakeRealtimeRunner:
        def __init__(self, model_worker: Any, output_processor: Any) -> None:
            del model_worker, output_processor

        def set_stream_outbox(self, outbox: Any) -> None:
            del outbox

    class FakeScheduler:
        def __init__(self, **kwargs: Any) -> None:
            self.outbox = object()
            self.kwargs = kwargs

    monkeypatch.setattr(
        MossTTSRealtimeEngineBuilder,
        "pre_infra_setup",
        lambda self, checkpoint_dir: (
            setattr(self, "context_length", 40960),
            setattr(self, "processor", "processor"),
        ),
    )
    monkeypatch.setattr(
        stages,
        "bind_moss_tts_realtime_processor_config",
        lambda config, processor: config,
    )

    monkeypatch.setattr(engine_factory, "_resolve_checkpoint", lambda path: path)
    monkeypatch.setattr(
        sglang_backend,
        "build_sglang_server_args",
        fake_build_server_args,
    )
    monkeypatch.setattr(
        bootstrap,
        "create_sglang_infrastructure_defer_cuda_graph",
        fake_deferred_infrastructure,
    )
    monkeypatch.setattr(
        sglang_backend,
        "SGLangOutputProcessor",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        request_builders,
        "make_moss_tts_realtime_scheduler_adapters",
        lambda **_: (object(), object()),
    )
    monkeypatch.setattr(
        model_runner,
        "MossTTSRealtimeModelRunner",
        FakeRealtimeRunner,
    )
    monkeypatch.setattr(scheduler, "MossTTSRealtimeScheduler", FakeScheduler)

    _builder(total_gpu_memory_fraction=None).build(
        "model",
        server_args_overrides={"disable_cuda_graph": True},
    )

    assert calls == {"cuda_graphs": 0, "frame_decode_graphs": 0}


def test_create_vocoder_executor_threads_slot_limit(monkeypatch) -> None:
    calls: dict[str, Any] = {}
    codec = object()
    processor = SimpleNamespace(model_config=SimpleNamespace(rvq=16))

    class FakeVocoderScheduler:
        def warmup_now(self) -> None:
            calls["warmup"] = calls.get("warmup", 0) + 1

    scheduler = FakeVocoderScheduler()

    def fake_load_codec(
        model_path: str,
        *,
        component: str,
        device: str,
        dtype: torch.dtype,
    ) -> object:
        calls["codec"] = (model_path, component, device, dtype)
        return codec

    def fake_scheduler(loaded_codec: Any, **kwargs: Any) -> object:
        calls["scheduler"] = (loaded_codec, kwargs)
        return scheduler

    monkeypatch.setattr(stages, "load_moss_tts_realtime_codec", fake_load_codec)
    monkeypatch.setattr(
        stages,
        "load_moss_tts_realtime_processor",
        lambda model_path: (
            calls.__setitem__("processor", model_path),
            processor,
        )[-1],
    )
    monkeypatch.setattr(
        stages,
        "MossTTSRealtimeStreamingVocoderScheduler",
        fake_scheduler,
    )

    result = stages.create_vocoder_executor(
        "model",
        device=None,
        gpu_id=2,
        codec_model_path="codec",
        stream_slots=4,
        max_batch_size=3,
        max_batch_wait_ms=7,
        cuda_graph=False,
        cuda_graph_frames=[1, 3],
        cuda_graph_min_free_gb=5.0,
        dtype="float32",
        session_idle_ttl_s=12.0,
    )

    assert result is scheduler
    assert calls == {
        "processor": "model",
        "codec": ("codec", "decoder", "cuda:2", torch.float32),
        "scheduler": (
            codec,
            {
                "n_vq": 16,
                "stream_slots": 4,
                "max_batch_size": 3,
                "max_batch_wait_ms": 7,
                "cuda_graph": False,
                "cuda_graph_frames": [1, 3],
                "cuda_graph_min_free_gb": 5.0,
                "session_idle_ttl_s": 12.0,
            },
        ),
        "warmup": 1,
    }


def test_create_vocoder_executor_defaults_to_bfloat16_before_warmup(
    monkeypatch,
) -> None:
    events: list[tuple[str, Any]] = []
    codec = object()
    processor = SimpleNamespace(model_config=SimpleNamespace(rvq=16))

    class FakeVocoderScheduler:
        def warmup_now(self) -> None:
            events.append(("warmup", None))

    monkeypatch.setattr(
        stages,
        "load_moss_tts_realtime_codec",
        lambda *args, **kwargs: (
            events.append(("load_dtype", kwargs.get("dtype"))),
            codec,
        )[-1],
    )
    monkeypatch.setattr(
        stages,
        "load_moss_tts_realtime_processor",
        lambda _: processor,
    )
    monkeypatch.setattr(
        stages,
        "configure_moss_tts_realtime_vocoder_decoder",
        lambda loaded_codec, *, dtype: events.append(
            ("configure", (loaded_codec, dtype))
        ),
    )
    monkeypatch.setattr(
        stages,
        "MossTTSRealtimeStreamingVocoderScheduler",
        lambda loaded_codec, **kwargs: (
            events.append(("scheduler", loaded_codec)),
            FakeVocoderScheduler(),
        )[-1],
    )

    stages.create_vocoder_executor(
        "model",
        codec_model_path="codec",
    )

    assert events == [
        ("load_dtype", torch.bfloat16),
        ("configure", (codec, torch.bfloat16)),
        ("scheduler", codec),
        ("warmup", None),
    ]
