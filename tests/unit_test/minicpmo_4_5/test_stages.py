from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from sglang_omni.models.minicpmo_4_5 import stages


def test_stage_factory_has_no_nested_runtime_or_rpc_dependency() -> None:
    source = inspect.getsource(stages)
    assert "process_runtime" not in source
    assert "runtime_worker" not in source
    assert "subprocess" not in source


class _FakeTTSRuntime:
    created: list[dict] = []

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        runtime = cls()
        runtime.closed = False
        runtime.default_prompt_wav_path = "/checkpoint/assets/HT_ref_audio.wav"
        cls.created.append({"model_path": model_path, **kwargs, "runtime": runtime})
        return runtime

    def close(self):
        self.closed = True


class _FakePerception:
    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        obj = cls()
        obj.model_path = model_path
        obj.kwargs = kwargs
        obj.tokenizer = object()
        return obj


class _FakeOutputProcessor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeModelRunner:
    def __init__(self, worker, output_processor):
        self.worker = worker
        self.output_processor = output_processor
        self.tokenizer = None

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer


class _FakeScheduler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _fake_dependencies(call_order: list[str], *, fail_infrastructure=False):
    model = object()
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))

    def build_server_args(model_path, context_length, **kwargs):
        call_order.append("server_args")
        return SimpleNamespace(
            model_path=model_path,
            context_length=context_length,
            **kwargs,
        )

    def create_infrastructure(server_args, gpu_id, **kwargs):
        call_order.append("infrastructure")
        if fail_infrastructure:
            raise RuntimeError("load failed")
        return (worker, "cache", "req_pool", "kv_pool", "prefill", "decode", "cfg")

    class RecordingTTS(_FakeTTSRuntime):
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            call_order.append("tts")
            return super().from_pretrained(model_path, **kwargs)

    class RecordingPerception(_FakePerception):
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            call_order.append("perception")
            return super().from_pretrained(model_path, **kwargs)

    return stages._FactoryDependencies(
        build_server_args=build_server_args,
        create_infrastructure=create_infrastructure,
        perception_cls=RecordingPerception,
        tts_runtime_cls=RecordingTTS,
        model_runner_cls=_FakeModelRunner,
        output_processor_cls=_FakeOutputProcessor,
        scheduler_cls=_FakeScheduler,
    )


def test_factory_builds_native_components_in_one_call_path(monkeypatch) -> None:
    call_order: list[str] = []
    deps = _fake_dependencies(call_order)
    monkeypatch.setattr(stages, "_load_factory_dependencies", lambda: deps)
    monkeypatch.setattr(stages, "_resolve_torch_dtype", lambda _: "torch.bfloat16")

    scheduler = stages.create_minicpmo_duplex_scheduler(
        "openbmb/MiniCPM-o-4_5",
        revision="checkpoint-sha",
        gpu_id=2,
        ref_audio_path="ref.wav",
        prompt_wav_path="prompt.wav",
        max_pending_units=3,
        max_pending_commands=7,
        session_ttl_s=42,
        duplex_sampling={
            "generate_audio": True,
            "tts_temperature": 0.6,
            "tts_repetition_penalty": 1.03,
        },
        server_args_overrides={"mem_fraction_static": 0.7},
    )

    # TTS/token2wav is resident before ModelWorker profiles memory; perception
    # needs the loaded native MiniCPM-o model and therefore follows it.
    assert call_order == ["tts", "server_args", "infrastructure", "perception"]
    assert isinstance(scheduler, _FakeScheduler)
    kwargs = scheduler.kwargs
    assert (
        kwargs["tp_worker"].model_runner.model is kwargs["perception"].kwargs["model"]
    )
    assert kwargs["tokenizer"] is kwargs["perception"].tokenizer
    assert kwargs["model_runner"].tokenizer is kwargs["tokenizer"]
    assert kwargs["tts_runtime"] is _FakeTTSRuntime.created[-1]["runtime"]
    assert _FakeTTSRuntime.created[-1]["revision"] == "checkpoint-sha"
    assert _FakeTTSRuntime.created[-1]["temperature"] == pytest.approx(0.6)
    assert _FakeTTSRuntime.created[-1]["repetition_penalty"] == pytest.approx(1.03)
    assert kwargs["perception"].kwargs["revision"] == "checkpoint-sha"
    assert kwargs["server_args"].revision == "checkpoint-sha"
    assert kwargs["ref_audio_path"] == "ref.wav"
    assert kwargs["prompt_wav_path"] == "prompt.wav"
    assert kwargs["max_sessions"] == 1
    assert kwargs["enable_overlap"] is False
    assert kwargs["enable_async_decode"] is False


def test_factory_forces_streaming_session_and_safe_first_release(monkeypatch) -> None:
    call_order: list[str] = []
    deps = _fake_dependencies(call_order)
    captured: dict = {}
    original = deps.build_server_args

    def capture(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        stages,
        "_load_factory_dependencies",
        lambda: SimpleNamespace(**{**deps.__dict__, "build_server_args": capture}),
    )
    monkeypatch.setattr(stages, "_resolve_torch_dtype", lambda _: "torch.bfloat16")

    stages.create_minicpmo_duplex_scheduler("model")

    assert captured["enable_streaming_session"] is True
    assert captured["max_running_requests"] == 1
    assert captured["tp_size"] == 1
    assert captured["pp_size"] == 1
    assert captured["disable_overlap_schedule"] is True
    assert captured["disable_cuda_graph"] is True
    assert captured["enable_return_hidden_states"] is True
    assert captured["sampling_backend"] == "pytorch"
    assert captured["trust_remote_code"] is True


def test_factory_uses_checkpoint_prompt_as_default_system_reference(
    monkeypatch,
) -> None:
    call_order: list[str] = []
    deps = _fake_dependencies(call_order)
    monkeypatch.setattr(stages, "_load_factory_dependencies", lambda: deps)
    monkeypatch.setattr(stages, "_resolve_torch_dtype", lambda _: "torch.bfloat16")

    scheduler = stages.create_minicpmo_duplex_scheduler("model")

    assert scheduler.kwargs["ref_audio_path"] == "/checkpoint/assets/HT_ref_audio.wav"
    assert scheduler.kwargs["prompt_wav_path"] is None


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"max_running_requests": 2}, "max_running_requests=1"),
        ({"enable_streaming_session": False}, "enable_streaming_session=True"),
        ({"disable_overlap_schedule": False}, "disable_overlap_schedule=True"),
        ({"model_path": "other"}, "cannot own model_path"),
        ({"dtype": "float16"}, "cannot own dtype"),
        ({"revision": "other"}, "cannot own revision"),
    ],
)
def test_factory_rejects_overrides_that_break_native_invariants(
    override, message
) -> None:
    with pytest.raises(ValueError, match=message):
        stages.create_minicpmo_duplex_scheduler("model", server_args_overrides=override)


def test_factory_validates_dtype_before_loading_components(monkeypatch) -> None:
    monkeypatch.setattr(
        stages,
        "_load_factory_dependencies",
        lambda: (_ for _ in ()).throw(AssertionError("must not load dependencies")),
    )

    with pytest.raises(ValueError, match="dtype must be one of"):
        stages.create_minicpmo_duplex_scheduler(
            "model",
            dtype="float32",
            duplex_sampling={"generate_audio": False},
        )


def test_factory_skips_tts_when_audio_output_is_disabled(monkeypatch) -> None:
    call_order: list[str] = []
    deps = _fake_dependencies(call_order)
    monkeypatch.setattr(stages, "_load_factory_dependencies", lambda: deps)

    scheduler = stages.create_minicpmo_duplex_scheduler(
        "model", duplex_sampling={"generate_audio": False}
    )

    assert "tts" not in call_order
    assert scheduler.kwargs["tts_runtime"] is None
    assert scheduler.kwargs["ref_audio_path"] is None


def test_factory_closes_tts_if_sglang_startup_fails(monkeypatch) -> None:
    call_order: list[str] = []
    deps = _fake_dependencies(call_order, fail_infrastructure=True)
    monkeypatch.setattr(stages, "_load_factory_dependencies", lambda: deps)
    monkeypatch.setattr(stages, "_resolve_torch_dtype", lambda _: "torch.bfloat16")

    with pytest.raises(RuntimeError, match="load failed"):
        stages.create_minicpmo_duplex_scheduler("model")

    assert _FakeTTSRuntime.created[-1]["runtime"].closed is True
