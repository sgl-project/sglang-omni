# SPDX-License-Identifier: Apache-2.0
"""MLX offload lifecycle tests runnable without Apple hardware."""

from __future__ import annotations

import importlib
import sys
import threading
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.minimax_music3.serial_offload import SerialOffloadCoordinator
from sglang_omni.scheduling.message import IncomingMessage


class RecordingCoordinator:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def enable(self) -> None:
        self.events.append("enable")

    def acquire_ar(self, request_id: str, *, should_abort) -> None:
        assert not should_abort()
        self.events.append(f"acquire:{request_id}")

    def begin_dit_handoff(self, request_id: str) -> None:
        self.events.append(f"handoff:{request_id}")

    def cancel_ar(self, request_id: str) -> None:
        self.events.append(f"cancel:{request_id}")

    def fail_closed(self, exc: BaseException) -> None:
        self.events.append(f"failed:{type(exc).__name__}")


def fake_mlx(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> ModuleType:
    mlx = ModuleType("mlx")
    core = ModuleType("mlx.core")
    core.gpu = object()
    core.float16 = np.float16
    core.new_thread_local_stream = lambda device: "stream"
    core.stream = lambda stream: nullcontext()
    core.eval = lambda *values: None
    core.synchronize = lambda stream: events.append("synchronize")
    core.get_active_memory = lambda: 0
    core.get_cache_memory = lambda: 0
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    return core


def import_ar_scheduler(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    events: list[str],
):
    fake_mlx(monkeypatch, events)
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *args: None)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    package_name = "sglang_omni.models.minimax_music3.mlx"
    package = ModuleType(package_name)
    package.__path__ = [
        str(Path(__file__).parents[3] / "sglang_omni/models/minimax_music3/mlx")
    ]
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.delitem(sys.modules, f"{package_name}.ar_scheduler", raising=False)

    model_config = SimpleNamespace(model_path=str(tmp_path), audio_cfg_token_id=7)

    class Model:
        def __init__(self) -> None:
            self.config = model_config
            self.language_model = object()
            self.rvq_depth_decoder = object()

    loader = ModuleType(f"{package_name}.loader")
    loader.MiniMaxMusic3MlxARModel = Model
    loader.resolve_mlx_artifact = lambda path, revision: (
        tmp_path,
        {},
        model_config,
    )

    def load_model(path, revision=None, *, artifact=None):
        del path, revision, artifact
        events.append("load")
        return Model()

    loader.load_mlx_ar_model = load_model
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    ar = ModuleType(f"{package_name}.ar")
    ar.generate_frame_hiddens = lambda *args, **kwargs: np.zeros(
        (1, 2, 4), dtype=np.float32
    )
    monkeypatch.setitem(sys.modules, ar.__name__, ar)

    module = importlib.import_module(f"{package_name}.ar_scheduler")
    monkeypatch.setattr(
        module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(module, "validate_tokenizer_ids", lambda tokenizer: None)
    coordinator = RecordingCoordinator(events)
    monkeypatch.setattr(module, "get_coordinator", lambda: coordinator)
    (tmp_path / "tokenizer").mkdir()
    return module, coordinator


def import_acoustic_decoder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    events: list[str],
):
    importlib.import_module("sglang_omni.models.minimax_music3.acoustic")
    fake_mlx(monkeypatch, events)
    package_name = "sglang_omni.models.minimax_music3.mlx"
    package = ModuleType(package_name)
    package.__path__ = [
        str(Path(__file__).parents[3] / "sglang_omni/models/minimax_music3/mlx")
    ]
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.delitem(sys.modules, f"{package_name}.acoustic", raising=False)

    config = SimpleNamespace(dit_in_channels=4)

    class Model:
        def __init__(self) -> None:
            weight = SimpleNamespace(dtype="float16")
            self.condition_encoder = SimpleNamespace(
                proj=SimpleNamespace(weight=weight)
            )

    loader = ModuleType(f"{package_name}.loader")
    loader.MiniMaxMusic3MlxAcousticModel = Model
    loader.resolve_mlx_artifact = lambda path, revision: (
        tmp_path,
        {"dtype": "float16"},
        config,
    )

    def load_model(path, revision=None, *, artifact=None):
        del path, revision, artifact
        events.append("load")
        return Model()

    loader.load_mlx_acoustic_model = load_model
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    euler = ModuleType(f"{package_name}.euler")
    euler.denoise_chunk = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, euler.__name__, euler)
    module = importlib.import_module(f"{package_name}.acoustic")
    coordinator = RecordingCoordinator(events)
    monkeypatch.setattr(module, "get_coordinator", lambda: coordinator)
    return module


def test_ar_offload_loads_after_ownership_and_hands_off_before_publish(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    module, _ = import_ar_scheduler(monkeypatch, tmp_path, events)
    monkeypatch.setattr(
        module.MiniMaxMusic3State,
        "from_dict",
        lambda data: SimpleNamespace(
            prompt="song",
            max_audio_frames=301,
            seed=7,
            generated_frames=0,
            finish_reason=None,
            caption="caption",
            lyrics="lyrics",
        ),
    )
    monkeypatch.setattr(module, "build_text_pair", lambda *args: object())
    monkeypatch.setattr(module, "store_state", lambda payload, state: state)
    source_hidden = np.zeros((1, 301, 4), dtype=np.float16)

    class SharedArray:
        def __init__(self, array):
            self.array = array

        @property
        def shape(self):
            return self.array.shape

        def __getitem__(self, key):
            return SharedArray(self.array[key])

        def astype(self, dtype):
            assert dtype == np.float16
            return self

        def __array__(self, dtype=None, copy=None):
            del copy
            return np.asarray(self.array, dtype=dtype)

    monkeypatch.setattr(
        module,
        "generate_frame_hiddens",
        lambda *args, **kwargs: SharedArray(source_hidden),
    )
    scheduler = module.MiniMaxMusic3MlxARScheduler(str(tmp_path), serial_offload=True)
    assert "load" not in events

    scheduler.generate(SimpleNamespace(request_id="req-1", data={}))

    assert scheduler.model is None
    assert scheduler.outbox.qsize() == 3
    assert events == [
        "enable",
        "acquire:req-1",
        "load",
        "synchronize",
        "handoff:req-1",
    ]
    message = scheduler.outbox.get_nowait()
    source_hidden.fill(1)
    assert message.data.device.type == "cpu"
    assert not message.data.bool().any()


def test_ar_load_failure_releases_owner_after_synchronizing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    module, _ = import_ar_scheduler(monkeypatch, tmp_path, events)
    monkeypatch.setattr(
        module.MiniMaxMusic3State,
        "from_dict",
        lambda data: SimpleNamespace(prompt="song"),
    )

    def fail_load(*args, **kwargs):
        raise OSError("load failed")

    monkeypatch.setattr(module, "load_mlx_ar_model", fail_load)
    scheduler = module.MiniMaxMusic3MlxARScheduler(str(tmp_path), serial_offload=True)

    with pytest.raises(OSError, match="load failed"):
        scheduler.generate(SimpleNamespace(request_id="req-1", data={}))

    assert events[-2:] == ["synchronize", "cancel:req-1"]


def test_ar_resident_mode_loads_during_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    module, _ = import_ar_scheduler(monkeypatch, tmp_path, events)

    scheduler = module.MiniMaxMusic3MlxARScheduler(str(tmp_path), serial_offload=False)

    assert scheduler.model is not None
    assert events == ["load"]


@pytest.mark.parametrize("failure_point", ["load", "generation", "eval"])
def test_ar_errors_release_models_and_arrays_before_readmission(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_point: str,
) -> None:
    events: list[str] = []
    module, _ = import_ar_scheduler(monkeypatch, tmp_path, events)
    coordinator = SerialOffloadCoordinator()
    monkeypatch.setattr(module, "get_coordinator", lambda: coordinator)
    monkeypatch.setattr(
        module.MiniMaxMusic3State,
        "from_dict",
        lambda data: SimpleNamespace(prompt="song", max_audio_frames=1, seed=7),
    )
    monkeypatch.setattr(module, "build_text_pair", lambda *args: None)
    references: list[weakref.ReferenceType] = []
    model_type = module.MiniMaxMusic3MlxARModel

    class WeightHolder:
        pass

    def fail() -> None:
        intermediate = np.zeros((1, 2, 4), dtype=np.float32)
        references.append(weakref.ref(intermediate))
        try:
            raise ValueError("inner failure")
        except ValueError as error:
            raise RuntimeError("injected failure") from error

    def load_model(*args, **kwargs):
        assert all(reference() is None for reference in references)
        model = model_type()
        model.language_model = WeightHolder()
        model.rvq_depth_decoder = WeightHolder()
        references.extend(
            weakref.ref(component)
            for component in (model, model.language_model, model.rvq_depth_decoder)
        )
        if failure_point == "load":
            fail()
        return model

    def generate_hidden(language_model, depth, config, text_ids, **kwargs):
        if failure_point == "generation":
            fail()
        hidden = np.zeros((1, 2, 4), dtype=np.float32)
        references.append(weakref.ref(hidden))
        return hidden

    def fail_eval(*arrays) -> None:
        fail()

    monkeypatch.setattr(module, "load_mlx_ar_model", load_model)
    monkeypatch.setattr(module, "generate_frame_hiddens", generate_hidden)
    if failure_point == "eval":
        monkeypatch.setattr(module.mx, "eval", fail_eval)
    scheduler = module.MiniMaxMusic3MlxARScheduler(str(tmp_path), serial_offload=True)

    for request_id in ("first", "second"):
        scheduler.inbox.put(
            IncomingMessage(
                request_id=request_id,
                type="new_request",
                data=SimpleNamespace(request_id=request_id, data={}),
            )
        )
    scheduler_thread = threading.Thread(target=scheduler.start)
    scheduler_thread.start()
    errors = []
    try:
        for request_id in ("first", "second"):
            message = scheduler.outbox.get(timeout=5)
            assert message.request_id == request_id
            assert message.type == "error"
            assert isinstance(message.data, RuntimeError)
            assert message.data.__cause__.args == ("inner failure",)
            errors.append(message.data)
    finally:
        scheduler.stop()
        scheduler_thread.join(timeout=5)
    assert not scheduler_thread.is_alive()
    assert len(errors) == 2
    assert all(reference() is None for reference in references)
    assert coordinator.try_acquire_ar("next")


def test_acoustic_decode_failure_drops_unwound_arrays_and_chained_frames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    module = import_acoustic_decoder(monkeypatch, tmp_path, events)
    decoder = module.MiniMaxMusic3MlxAcousticDecoder(str(tmp_path), serial_offload=True)
    references: list[weakref.ReferenceType] = []

    def fail_decode(hidden, **kwargs):
        array = np.zeros((1, 4), dtype=np.float32)
        references.append(weakref.ref(array))
        try:
            raise ValueError("array failure")
        except ValueError as error:
            raise RuntimeError("decode failure") from error

    monkeypatch.setattr(decoder, "decode_chunk", fail_decode)
    with pytest.raises(RuntimeError, match="decode failure") as caught:
        decoder.decode_with_state(torch.zeros(1, 4), seed=7, chunk_idx=0)
    assert caught.value.__cause__.args == ("array failure",)
    assert all(reference() is None for reference in references)


def test_acoustic_offload_loads_once_for_chunks_then_releases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    module = import_acoustic_decoder(monkeypatch, tmp_path, events)
    decoder = module.MiniMaxMusic3MlxAcousticDecoder(str(tmp_path), serial_offload=True)
    assert decoder.model is None
    assert events == ["enable"]

    decoder.ensure_resident()
    first_model = decoder.model
    decoder.ensure_resident()

    assert decoder.model is first_model
    assert events == ["enable", "load"]
    decoder.release_residency()
    assert decoder.model is None
    assert events == ["enable", "load", "synchronize"]


def test_acoustic_partial_load_failure_releases_model_before_ar_readmission(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from sglang_omni.models.minimax_music3 import acoustic

    events: list[str] = []
    module = import_acoustic_decoder(monkeypatch, tmp_path, events)
    coordinator = SerialOffloadCoordinator()
    monkeypatch.setattr(module, "get_coordinator", lambda: coordinator)
    monkeypatch.setattr(acoustic, "get_coordinator", lambda: coordinator)
    decoder = module.MiniMaxMusic3MlxAcousticDecoder(str(tmp_path), serial_offload=True)
    scheduler = acoustic.MiniMaxMusic3AcousticScheduler(decoder)
    assert coordinator.try_acquire_ar("failed")
    coordinator.begin_dit_handoff("failed")
    references: list[weakref.ReferenceType] = []

    def fail_load(*args, **kwargs):
        partial_model = module.MiniMaxMusic3MlxAcousticModel()
        references.append(weakref.ref(partial_model))
        raise OSError("partial acoustic load failed")

    monkeypatch.setattr(module, "load_mlx_acoustic_model", fail_load)
    with pytest.raises(OSError, match="partial acoustic load failed") as caught:
        decoder.decode_with_state(torch.zeros(1, 4), seed=7, chunk_idx=0)
    scheduler.emit_error("failed", caught.value)
    scheduler.abort("failed")
    assert all(reference() is None for reference in references)
    assert coordinator.try_acquire_ar("next")
    assert scheduler.outbox.qsize() == 1


def test_acoustic_offload_is_stable_across_20_reload_cycles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    module = import_acoustic_decoder(monkeypatch, tmp_path, events)
    decoder = module.MiniMaxMusic3MlxAcousticDecoder(str(tmp_path), serial_offload=True)

    for _ in range(20):
        decoder.ensure_resident()
        decoder.ensure_resident()
        decoder.release_residency()

    assert decoder.model is None
    assert events.count("load") == 20
    assert events.count("synchronize") == 20
