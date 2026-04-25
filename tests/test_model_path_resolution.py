# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Qwen3 preprocessor recovery and encoder cache wiring."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang_omni.engines.omni.runtime.encoder import EncoderRequestData
from sglang_omni.engines.omni.types import SchedulerRequest
from sglang_omni.models import weight_loader
from sglang_omni.models.qwen3_omni.components import preprocessor
from sglang_omni.models.qwen3_omni.io import PipelineState
from sglang_omni.proto import OmniRequest, StagePayload


class _DummyProcessor:
    tokenizer = SimpleNamespace(chat_template="dummy-template")

    @staticmethod
    def apply_chat_template(*args, **kwargs):
        return "prompt"

    def __call__(self, *args, **kwargs):
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "pixel_values_videos": torch.zeros((2, 3), dtype=torch.float32),
            "video_grid_thw": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "video_second_per_grid": torch.tensor([0.5], dtype=torch.float32),
            "input_features": torch.zeros((1, 4), dtype=torch.float32),
            "feature_attention_mask": torch.ones((1, 4), dtype=torch.long),
            "audio_feature_lengths": torch.tensor([4], dtype=torch.long),
        }


class _DummyProcessorFactory:
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        return SimpleNamespace(
            tokenizer=SimpleNamespace(chat_template="dummy-template"),
            apply_chat_template=lambda *args, **kwargs: "prompt",
            __call__=None,
        )


def _stub_module(monkeypatch, name: str, **attrs) -> ModuleType:
    module = ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _patch_module(monkeypatch, module: ModuleType, **attrs) -> None:
    for name, value in attrs.items():
        monkeypatch.setitem(module.__dict__, name, value)


def _install_qwen3_stages_stubs(monkeypatch) -> None:
    _stub_module(
        monkeypatch,
        "sglang_omni.engines.ar.sglang_backend.server_args_builder",
        build_sglang_server_args=lambda *args, **kwargs: None,
        apply_encoder_mem_reserve=lambda *args, **kwargs: None,
    )
    _stub_module(
        monkeypatch,
        "sglang_omni.engines.omni",
        create_ar_engine=lambda *args, **kwargs: None,
        create_sglang_ar_engine=lambda *args, **kwargs: None,
        create_single_pass_engine=lambda *args, **kwargs: None,
    )

    class DummyEngineExecutor:
        def __init__(
            self, engine, request_builder, result_builder, stream_builder=None
        ):
            self._engine = engine
            self.request_builder = request_builder
            self.result_builder = result_builder
            self.stream_builder = stream_builder

    class DummyPreprocessingExecutor:
        def __init__(self, fn):
            self.fn = fn

    _stub_module(
        monkeypatch,
        "sglang_omni.executors",
        EngineExecutor=DummyEngineExecutor,
        PreprocessingExecutor=DummyPreprocessingExecutor,
    )
    _stub_module(
        monkeypatch,
        "sglang_omni.models.qwen3_omni.components.audio_encoder",
        Qwen3OmniAudioEncoder=object,
    )
    _stub_module(
        monkeypatch,
        "sglang_omni.models.qwen3_omni.components.image_encoder",
        Qwen3OmniImageEncoder=object,
    )
    _stub_module(
        monkeypatch,
        "sglang_omni.models.qwen3_omni.components.talker_executor",
        TalkerStreamingExecutor=object,
    )
    _stub_module(
        monkeypatch,
        "sglang_omni.models.qwen3_omni.components.thinker",
        Qwen3OmniSplitThinker=object,
    )
    _stub_module(
        monkeypatch,
        "sglang_omni.models.qwen3_omni.pipeline.engine_io",
        apply_encoder_result=lambda *args, **kwargs: None,
        apply_thinker_result=lambda *args, **kwargs: None,
        build_encoder_request=lambda *args, **kwargs: None,
        build_sglang_thinker_request=lambda *args, **kwargs: None,
        build_thinker_request=lambda *args, **kwargs: None,
    )
    _stub_module(
        monkeypatch,
        "sglang_omni.models.qwen3_omni.pipeline.merge",
        decode_events=lambda *args, **kwargs: [],
    )
    _stub_module(
        monkeypatch,
        "sglang_omni.models.qwen3_omni.pipeline.next_stage",
        AUDIO_STAGE="audio_encoder",
        CODE_PREDICTOR_STAGE="code_predictor",
        IMAGE_STAGE="image_encoder",
        TALKER_AR_STAGE="talker_ar",
        THINKER_STAGE="thinker",
    )
    _stub_module(
        monkeypatch,
        "sglang_omni.models.qwen3_omni.pipeline.state_io",
        load_state=lambda payload: None,
        store_state=lambda payload, state: payload,
    )


def _import_qwen3_stages_for_test(monkeypatch):
    module_name = "sglang_omni.models.qwen3_omni.pipeline.stages"
    original = sys.modules.get(module_name)
    if original is not None:
        monkeypatch.setitem(sys.modules, module_name, original)
    sys.modules.pop(module_name, None)
    _install_qwen3_stages_stubs(monkeypatch)
    return importlib.import_module(module_name)


@pytest.fixture
def qwen3_preprocessor_testbed(monkeypatch, tmp_path):
    model_dir = tmp_path / "snapshot"
    model_dir.mkdir()

    _patch_module(
        monkeypatch,
        preprocessor,
        resolve_model_path=lambda model_path, *, local_files_only=False: model_dir,
        Qwen3OmniMoeProcessor=_DummyProcessorFactory,
        compute_image_cache_key=lambda *_: None,
        compute_video_cache_key=lambda *_: "video-key",
        compute_audio_cache_key=lambda *_: None,
    )

    proc = preprocessor.Qwen3OmniPreprocessor("Qwen/Qwen3-Omni-30B-A3B-Instruct")
    proc.processor = _DummyProcessor()
    return proc


def test_qwen3_preprocessor_falls_back_to_remote_processor_download(
    monkeypatch, tmp_path
) -> None:
    local_snapshot = tmp_path / "snapshot"
    refreshed_snapshot = tmp_path / "resolved-after-remote"
    local_snapshot.mkdir()
    refreshed_snapshot.mkdir()

    calls: list[tuple[str, bool]] = []
    resolve_calls: list[bool] = []

    class DummyProcessorFactory:
        @classmethod
        def from_pretrained(cls, model_path: str, **kwargs):
            calls.append((str(model_path), bool(kwargs.get("local_files_only"))))
            if kwargs.get("local_files_only"):
                raise OSError("missing processor assets")
            return SimpleNamespace(
                tokenizer=SimpleNamespace(chat_template="dummy-template")
            )

    def fake_resolve_model_path(model_path: str, *, local_files_only: bool = False):
        resolve_calls.append(local_files_only)
        return local_snapshot if local_files_only else refreshed_snapshot

    _patch_module(
        monkeypatch,
        preprocessor,
        resolve_model_path=fake_resolve_model_path,
        Qwen3OmniMoeProcessor=DummyProcessorFactory,
    )

    proc = preprocessor.Qwen3OmniPreprocessor("Qwen/Qwen3-Omni-30B-A3B-Instruct")

    assert proc.processor.tokenizer.chat_template == "dummy-template"
    assert proc.model_dir == str(refreshed_snapshot)
    assert len(calls) == 2
    assert calls[0] == (str(local_snapshot), True)
    assert calls[1] == ("Qwen/Qwen3-Omni-30B-A3B-Instruct", False)
    assert resolve_calls == [True, False]


def test_resolve_local_model_dir_propagates_unexpected_errors(monkeypatch) -> None:
    def fake_resolve_model_path(model_path: str, *, local_files_only: bool = False):
        raise RuntimeError("boom")

    _patch_module(monkeypatch, preprocessor, resolve_model_path=fake_resolve_model_path)

    with pytest.raises(RuntimeError, match="boom"):
        preprocessor._resolve_local_model_dir("Qwen/Qwen3-Omni-30B-A3B-Instruct")


def test_qwen3_encoder_executor_forwards_cache_settings(monkeypatch) -> None:
    stages = _import_qwen3_stages_for_test(monkeypatch)
    captured: dict[str, object] = {}
    sentinel_engine = object()

    def fake_create_single_pass_engine(
        model,
        *,
        device: str,
        use_cache: bool,
        cache_size: int | None,
        cache_max_bytes: int | None,
        cache_device: str | torch.device | None,
        max_batch_size: int,
        request_cost_fn,
        max_batch_cost: int | None,
    ):
        captured["model"] = model
        captured["device"] = device
        captured["use_cache"] = use_cache
        captured["cache_size"] = cache_size
        captured["cache_max_bytes"] = cache_max_bytes
        captured["cache_device"] = cache_device
        captured["max_batch_size"] = max_batch_size
        captured["request_cost_fn"] = request_cost_fn
        captured["max_batch_cost"] = max_batch_cost
        return sentinel_engine

    _patch_module(
        monkeypatch,
        stages,
        create_single_pass_engine=fake_create_single_pass_engine,
    )

    model = torch.nn.Linear(2, 2)
    executor = stages._create_encoder_executor(
        stage_name="image_encoder",
        model=model,
        device="cuda:1",
        use_cache=False,
        cache_size=17,
    )

    assert captured == {
        "model": model,
        "device": "cuda:1",
        "use_cache": False,
        "cache_size": 17,
        "cache_max_bytes": stages.QWEN3_ENCODER_CACHE_MAX_BYTES,
        "cache_device": "cpu",
        "max_batch_size": 32,
        "request_cost_fn": None,
        "max_batch_cost": None,
    }
    assert executor._engine is sentinel_engine


def test_qwen3_image_executor_uses_visual_batch_budget(monkeypatch) -> None:
    stages = _import_qwen3_stages_for_test(monkeypatch)
    visual_budget = importlib.import_module(
        "sglang_omni.models.qwen3_omni.pipeline.visual_budget"
    )
    captured: dict[str, object] = {}
    sentinel_engine = object()

    class FakeImageEncoder:
        spatial_merge_size = 2
        out_hidden_size = 8
        deepstack_layers = 3
        visual_dtype_bytes = 2

        def __init__(self, *, model_path: str, device: str, dtype: str | None):
            captured["model_path"] = model_path
            captured["model_device"] = device
            captured["model_dtype"] = dtype

    def fake_create_single_pass_engine(
        model,
        *,
        device: str,
        use_cache: bool,
        cache_size: int | None,
        cache_max_bytes: int | None,
        cache_device: str | torch.device | None,
        max_batch_size: int,
        request_cost_fn,
        max_batch_cost: int | None,
    ):
        captured["model"] = model
        captured["device"] = device
        captured["use_cache"] = use_cache
        captured["cache_size"] = cache_size
        captured["cache_max_bytes"] = cache_max_bytes
        captured["cache_device"] = cache_device
        captured["max_batch_size"] = max_batch_size
        captured["request_cost_fn"] = request_cost_fn
        captured["max_batch_cost"] = max_batch_cost
        return sentinel_engine

    _patch_module(
        monkeypatch,
        stages,
        Qwen3OmniImageEncoder=FakeImageEncoder,
        create_single_pass_engine=fake_create_single_pass_engine,
    )

    executor = stages.create_image_encoder_executor(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        device="cuda:1",
        dtype="bfloat16",
        max_batch_size=7,
    )

    request = SchedulerRequest(
        request_id="video",
        data=EncoderRequestData(
            input_dict={
                "pixel_values_videos": torch.zeros((16, 3), dtype=torch.float32),
                "video_grid_thw": torch.tensor([[2, 4, 4]], dtype=torch.long),
            },
        ),
    )

    assert executor._engine is sentinel_engine
    assert captured["model_path"] == "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    assert captured["model_device"] == "cuda:1"
    assert captured["model_dtype"] == "bfloat16"
    assert captured["device"] == "cuda:1"
    assert captured["max_batch_size"] == 7
    assert (
        captured["max_batch_cost"]
        == visual_budget.QWEN3_IMAGE_ENCODER_BATCH_BUDGET_BYTES
    )
    request_cost_fn = captured["request_cost_fn"]
    assert callable(request_cost_fn)
    assert request_cost_fn(request) > 0


def test_qwen3_visual_request_cost_counts_raw_and_deepstack_bytes(
    monkeypatch,
) -> None:
    _import_qwen3_stages_for_test(monkeypatch)
    visual_budget = importlib.import_module(
        "sglang_omni.models.qwen3_omni.pipeline.visual_budget"
    )
    model = SimpleNamespace(
        spatial_merge_size=2,
        out_hidden_size=8,
        deepstack_layers=3,
        visual_dtype_bytes=2,
    )
    request = SchedulerRequest(
        request_id="mixed",
        data=EncoderRequestData(
            input_dict={
                "pixel_values_videos": torch.zeros((16, 3), dtype=torch.float32),
                "video_grid_thw": torch.tensor([[2, 4, 4]], dtype=torch.long),
                "pixel_values": torch.zeros((4, 3), dtype=torch.float16),
                "image_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.long),
            },
        ),
    )

    cost = visual_budget.create_qwen3_visual_request_cost_fn(model)(request)

    raw_video_bytes = 16 * 3 * 4
    raw_image_bytes = 4 * 3 * 2
    video_output_bytes = 8 * 8 * 2 * 4
    image_output_bytes = 4 * 8 * 2 * 4
    expected = (
        raw_video_bytes + raw_image_bytes + video_output_bytes + image_output_bytes
    ) * visual_budget.QWEN3_IMAGE_ENCODER_ACTIVATION_MULTIPLIER
    assert cost == expected


def test_qwen3_visual_request_cost_ignores_cached_skip(monkeypatch) -> None:
    _import_qwen3_stages_for_test(monkeypatch)
    visual_budget = importlib.import_module(
        "sglang_omni.models.qwen3_omni.pipeline.visual_budget"
    )
    model = SimpleNamespace(
        spatial_merge_size=2,
        out_hidden_size=8,
        deepstack_layers=3,
        visual_dtype_bytes=2,
    )
    request = SchedulerRequest(
        request_id="cached",
        data=EncoderRequestData(input_dict={"_skip": True}),
    )

    assert visual_budget.create_qwen3_visual_request_cost_fn(model)(request) == 0


def test_weight_loader_force_refreshes_partial_remote_snapshot(
    monkeypatch, tmp_path
) -> None:
    partial_snapshot = tmp_path / "partial"
    refreshed_snapshot = tmp_path / "refreshed"
    partial_snapshot.mkdir()
    refreshed_snapshot.mkdir()

    refresh_calls: list[tuple[str, bool, bool]] = []
    load_attempts: list[Path] = []

    weight_loader.resolve_model_path.cache_clear()

    def fake_resolve_model_path(model_path: str, *, local_files_only: bool = False):
        assert model_path == "Qwen/Qwen3-Omni-30B-A3B-Instruct"
        assert local_files_only is False
        return partial_snapshot

    fake_resolve_model_path.cache_clear = lambda: None

    def fake_snapshot_download(
        model_path: str, *, local_files_only: bool = False, force_download: bool = False
    ) -> str:
        refresh_calls.append((model_path, local_files_only, force_download))
        return str(refreshed_snapshot)

    def fake_load_safetensors_sharded(model_dir: Path, prefix: str):
        load_attempts.append(model_dir)
        if model_dir == refreshed_snapshot and prefix == "thinker.visual.":
            return {"proj.weight": "loaded"}
        return {}

    _patch_module(
        monkeypatch,
        weight_loader,
        resolve_model_path=fake_resolve_model_path,
        snapshot_download=fake_snapshot_download,
        _load_safetensors_sharded=fake_load_safetensors_sharded,
        _load_safetensors_single=lambda *_: {},
        _load_bin_sharded=lambda *_: {},
        _load_bin_single=lambda *_: {},
    )

    state_dict = weight_loader.load_weights_by_prefix(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        prefix=("thinker.visual.", "visual."),
        local_files_only=False,
    )

    assert state_dict == {"proj.weight": "loaded"}
    assert refresh_calls == [
        ("Qwen/Qwen3-Omni-30B-A3B-Instruct", False, True),
    ]
    assert load_attempts[0] == partial_snapshot
    assert refreshed_snapshot in load_attempts


def test_weight_loader_force_refreshes_missing_remote_shard(
    monkeypatch, tmp_path
) -> None:
    partial_snapshot = tmp_path / "partial"
    refreshed_snapshot = tmp_path / "refreshed"
    partial_snapshot.mkdir()
    refreshed_snapshot.mkdir()

    refresh_calls: list[tuple[str, bool, bool]] = []
    load_attempts: list[Path] = []

    weight_loader.resolve_model_path.cache_clear()

    def fake_resolve_model_path(model_path: str, *, local_files_only: bool = False):
        assert model_path == "Qwen/Qwen3-Omni-30B-A3B-Instruct"
        assert local_files_only is False
        return partial_snapshot

    fake_resolve_model_path.cache_clear = lambda: None

    def fake_snapshot_download(
        model_path: str, *, local_files_only: bool = False, force_download: bool = False
    ) -> str:
        refresh_calls.append((model_path, local_files_only, force_download))
        return str(refreshed_snapshot)

    def fake_load_safetensors_sharded(model_dir: Path, prefix: str):
        load_attempts.append(model_dir)
        if model_dir == partial_snapshot and prefix == "thinker.visual.":
            raise FileNotFoundError("missing shard")
        if model_dir == refreshed_snapshot and prefix == "thinker.visual.":
            return {"proj.weight": "loaded"}
        return {}

    _patch_module(
        monkeypatch,
        weight_loader,
        resolve_model_path=fake_resolve_model_path,
        snapshot_download=fake_snapshot_download,
        _load_safetensors_sharded=fake_load_safetensors_sharded,
        _load_safetensors_single=lambda *_: {},
        _load_bin_sharded=lambda *_: {},
        _load_bin_single=lambda *_: {},
    )

    state_dict = weight_loader.load_weights_by_prefix(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        prefix=("thinker.visual.", "visual."),
        local_files_only=False,
    )

    assert state_dict == {"proj.weight": "loaded"}
    assert refresh_calls == [
        ("Qwen/Qwen3-Omni-30B-A3B-Instruct", False, True),
    ]
    assert load_attempts[0] == partial_snapshot
    assert refreshed_snapshot in load_attempts


@pytest.mark.asyncio
async def test_preprocessor_cache_keys_include_preprocessing_context(
    monkeypatch, qwen3_preprocessor_testbed
) -> None:
    video_kwargs: dict[str, object] = {}

    async def fake_ensure_video_list_async(*args, **kwargs):
        video_kwargs.update(kwargs)
        return [torch.zeros((2, 3), dtype=torch.float32)], [7.5], []

    async def fake_ensure_image_list_async(*args, **kwargs):
        return []

    async def fake_ensure_audio_list_async(*args, **kwargs):
        return [torch.zeros(4)]

    _patch_module(
        monkeypatch,
        preprocessor,
        ensure_video_list_async=fake_ensure_video_list_async,
        ensure_image_list_async=fake_ensure_image_list_async,
        ensure_audio_list_async=fake_ensure_audio_list_async,
        compute_audio_cache_key=lambda *_: "audio-key",
    )

    proc = qwen3_preprocessor_testbed

    payload = StagePayload(
        request_id="req-1",
        request=OmniRequest(
            inputs={
                "messages": [{"role": "user", "content": "describe the video"}],
                "videos": ["video.mp4"],
                "audios": ["audio.wav"],
                "audio_target_sr": 22050,
                "video_fps": 12.0,
                "video_max_frames": 128,
                "video_max_pixels": 401408,
                "video_seconds_per_chunk": 2.5,
            }
        ),
        data=None,
    )

    result = await proc(payload)
    state = PipelineState.from_dict(result.data)

    assert video_kwargs["fps"] == 12.0
    assert video_kwargs["max_frames"] == 128
    assert video_kwargs["max_pixels"] == 401408
    assert (
        state.encoder_inputs["image_encoder"]["cache_key"]
        == "video-key|fps=(7.5,)|max_frames=128|max_pixels=401408"
    )
    assert (
        state.encoder_inputs["audio_encoder"]["cache_key"]
        == "audio-key|target_sr=22050"
    )


@pytest.mark.asyncio
async def test_preprocessor_rejects_default_generation_over_context(
    monkeypatch, qwen3_preprocessor_testbed
) -> None:
    async def fake_ensure_video_list_async(*args, **kwargs):
        return [], None, []

    async def fake_ensure_image_list_async(*args, **kwargs):
        return []

    async def fake_ensure_audio_list_async(*args, **kwargs):
        return []

    _patch_module(
        monkeypatch,
        preprocessor,
        ensure_video_list_async=fake_ensure_video_list_async,
        ensure_image_list_async=fake_ensure_image_list_async,
        ensure_audio_list_async=fake_ensure_audio_list_async,
    )
    proc = qwen3_preprocessor_testbed
    proc.max_seq_len = 4

    payload = StagePayload(
        request_id="req-context",
        request=OmniRequest(
            inputs={"messages": [{"role": "user", "content": "describe"}]},
            params={},
        ),
        data=None,
    )

    with pytest.raises(ValueError, match="Requested token count exceeds"):
        await proc(payload)


@pytest.mark.asyncio
async def test_preprocessor_cache_keys_canonicalize_explicit_video_fps(
    monkeypatch, qwen3_preprocessor_testbed
) -> None:
    async def fake_ensure_video_list_async(*args, **kwargs):
        return [torch.zeros((2, 3), dtype=torch.float32)], None, []

    async def fake_ensure_image_list_async(*args, **kwargs):
        return []

    async def fake_ensure_audio_list_async(*args, **kwargs):
        return []

    _patch_module(
        monkeypatch,
        preprocessor,
        ensure_video_list_async=fake_ensure_video_list_async,
        ensure_image_list_async=fake_ensure_image_list_async,
        ensure_audio_list_async=fake_ensure_audio_list_async,
    )
    proc = qwen3_preprocessor_testbed

    payload = StagePayload(
        request_id="req-fps",
        request=OmniRequest(
            inputs={
                "messages": [{"role": "user", "content": "describe the video"}],
                "videos": ["video.mp4"],
                "video_fps": 7.5,
            }
        ),
        data=None,
    )

    result = await proc(payload)
    state = PipelineState.from_dict(result.data)

    assert state.encoder_inputs["image_encoder"]["cache_key"] == "video-key|fps=(7.5,)"


@pytest.mark.asyncio
async def test_preprocessor_cache_keys_include_video_audio_extraction_context(
    monkeypatch, qwen3_preprocessor_testbed
) -> None:
    async def fake_ensure_video_list_async(*args, **kwargs):
        return [torch.zeros((2, 3), dtype=torch.float32)], [6.0], [torch.zeros(4)]

    async def fake_ensure_image_list_async(*args, **kwargs):
        return []

    async def fake_ensure_audio_list_async(*args, **kwargs):
        return []

    _patch_module(
        monkeypatch,
        preprocessor,
        ensure_video_list_async=fake_ensure_video_list_async,
        ensure_image_list_async=fake_ensure_image_list_async,
        ensure_audio_list_async=fake_ensure_audio_list_async,
    )
    proc = qwen3_preprocessor_testbed

    payload = StagePayload(
        request_id="req-2",
        request=OmniRequest(
            inputs={
                "messages": [{"role": "user", "content": "describe the video"}],
                "videos": ["video.mp4"],
                "use_audio_in_video": True,
                "audio_target_sr": 24000,
                "video_seconds_per_chunk": 1.5,
            }
        ),
        data=None,
    )

    result = await proc(payload)
    state = PipelineState.from_dict(result.data)

    assert state.encoder_inputs["image_encoder"]["cache_key"] == "video-key|fps=(6.0,)"
    assert (
        state.encoder_inputs["audio_encoder"]["cache_key"]
        == "video-key|extracted_audio=True|target_sr=24000"
    )
