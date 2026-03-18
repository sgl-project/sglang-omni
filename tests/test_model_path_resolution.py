# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Qwen3 preprocessor recovery and encoder cache wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models import weight_loader
from sglang_omni.models.qwen3_omni.components import preprocessor
from sglang_omni.models.qwen3_omni.io import PipelineState
from sglang_omni.models.qwen3_omni.pipeline import stages
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


@pytest.fixture
def qwen3_preprocessor_testbed(monkeypatch, tmp_path):
    model_dir = tmp_path / "snapshot"
    model_dir.mkdir()

    monkeypatch.setattr(
        preprocessor,
        "resolve_model_path",
        lambda model_path, *, local_files_only=False: model_dir,
    )
    monkeypatch.setattr(preprocessor, "Qwen3OmniMoeProcessor", _DummyProcessorFactory)
    monkeypatch.setattr(preprocessor, "compute_image_cache_key", lambda *_: None)
    monkeypatch.setattr(preprocessor, "compute_video_cache_key", lambda *_: "video-key")
    monkeypatch.setattr(preprocessor, "compute_audio_cache_key", lambda *_: None)

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

    monkeypatch.setattr(preprocessor, "resolve_model_path", fake_resolve_model_path)
    monkeypatch.setattr(preprocessor, "Qwen3OmniMoeProcessor", DummyProcessorFactory)

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

    monkeypatch.setattr(preprocessor, "resolve_model_path", fake_resolve_model_path)

    with pytest.raises(RuntimeError, match="boom"):
        preprocessor._resolve_local_model_dir("Qwen/Qwen3-Omni-30B-A3B-Instruct")


def test_qwen3_encoder_executor_forwards_cache_settings(monkeypatch) -> None:
    captured: dict[str, object] = {}
    sentinel_engine = object()

    def fake_create_single_pass_engine(
        model, *, device: str, use_cache: bool, cache_size: int | None
    ):
        captured["model"] = model
        captured["device"] = device
        captured["use_cache"] = use_cache
        captured["cache_size"] = cache_size
        return sentinel_engine

    monkeypatch.setattr(
        stages, "create_single_pass_engine", fake_create_single_pass_engine
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
    }
    assert executor._engine is sentinel_engine


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

    monkeypatch.setattr(weight_loader, "resolve_model_path", fake_resolve_model_path)
    monkeypatch.setattr(weight_loader, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(
        weight_loader, "_load_safetensors_sharded", fake_load_safetensors_sharded
    )
    monkeypatch.setattr(weight_loader, "_load_safetensors_single", lambda *_: {})
    monkeypatch.setattr(weight_loader, "_load_bin_sharded", lambda *_: {})
    monkeypatch.setattr(weight_loader, "_load_bin_single", lambda *_: {})

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

    monkeypatch.setattr(weight_loader, "resolve_model_path", fake_resolve_model_path)
    monkeypatch.setattr(weight_loader, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(
        weight_loader, "_load_safetensors_sharded", fake_load_safetensors_sharded
    )
    monkeypatch.setattr(weight_loader, "_load_safetensors_single", lambda *_: {})
    monkeypatch.setattr(weight_loader, "_load_bin_sharded", lambda *_: {})
    monkeypatch.setattr(weight_loader, "_load_bin_single", lambda *_: {})

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
    async def fake_ensure_video_list_async(*args, **kwargs):
        return [torch.zeros((2, 3), dtype=torch.float32)], [7.5], []

    async def fake_ensure_image_list_async(*args, **kwargs):
        return []

    async def fake_ensure_audio_list_async(*args, **kwargs):
        return [torch.zeros(4)]

    monkeypatch.setattr(
        preprocessor, "ensure_video_list_async", fake_ensure_video_list_async
    )
    monkeypatch.setattr(
        preprocessor, "ensure_image_list_async", fake_ensure_image_list_async
    )
    monkeypatch.setattr(
        preprocessor, "ensure_audio_list_async", fake_ensure_audio_list_async
    )
    monkeypatch.setattr(preprocessor, "compute_audio_cache_key", lambda *_: "audio-key")

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
                "video_seconds_per_chunk": 2.5,
            }
        ),
        data=None,
    )

    result = await proc(payload)
    state = PipelineState.from_dict(result.data)

    assert state.encoder_inputs["image_encoder"]["cache_key"] == "video-key|fps=(7.5,)"
    assert (
        state.encoder_inputs["audio_encoder"]["cache_key"]
        == "audio-key|target_sr=22050"
    )


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

    monkeypatch.setattr(
        preprocessor, "ensure_video_list_async", fake_ensure_video_list_async
    )
    monkeypatch.setattr(
        preprocessor, "ensure_image_list_async", fake_ensure_image_list_async
    )
    monkeypatch.setattr(
        preprocessor, "ensure_audio_list_async", fake_ensure_audio_list_async
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

    monkeypatch.setattr(
        preprocessor, "ensure_video_list_async", fake_ensure_video_list_async
    )
    monkeypatch.setattr(
        preprocessor, "ensure_image_list_async", fake_ensure_image_list_async
    )
    monkeypatch.setattr(
        preprocessor, "ensure_audio_list_async", fake_ensure_audio_list_async
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
