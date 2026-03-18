# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Qwen3 preprocessor recovery and encoder cache wiring."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models import weight_loader
from sglang_omni.models.qwen3_omni.components import preprocessor
from sglang_omni.models.qwen3_omni.io import PipelineState
from sglang_omni.proto import OmniRequest, StagePayload


def test_qwen3_preprocessor_falls_back_to_remote_processor_download(
    monkeypatch, tmp_path
) -> None:
    model_dir = tmp_path / "snapshot"
    model_dir.mkdir()

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
        return model_dir

    monkeypatch.setattr(preprocessor, "resolve_model_path", fake_resolve_model_path)
    monkeypatch.setattr(preprocessor, "Qwen3OmniMoeProcessor", DummyProcessorFactory)

    proc = preprocessor.Qwen3OmniPreprocessor("Qwen/Qwen3-Omni-30B-A3B-Instruct")

    assert proc.processor.tokenizer.chat_template == "dummy-template"
    assert len(calls) == 2
    assert calls[0] == (str(model_dir), True)
    assert calls[1] == ("Qwen/Qwen3-Omni-30B-A3B-Instruct", False)
    assert resolve_calls == [True, True]


def test_qwen3_encoder_executor_forwards_cache_settings() -> None:
    stages_path = (
        Path(__file__).resolve().parent.parent
        / "sglang_omni"
        / "models"
        / "qwen3_omni"
        / "pipeline"
        / "stages.py"
    )
    tree = ast.parse(stages_path.read_text(encoding="utf-8"))

    target_fn = None
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_create_encoder_executor"
        ):
            target_fn = node
            break

    assert target_fn is not None, "_create_encoder_executor() not found"

    engine_call = None
    for node in ast.walk(target_fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "create_single_pass_engine":
                engine_call = node
                break

    assert engine_call is not None, "create_single_pass_engine() call not found"

    keywords = {kw.arg: kw.value for kw in engine_call.keywords if kw.arg is not None}
    assert isinstance(keywords.get("use_cache"), ast.Name)
    assert keywords["use_cache"].id == "use_cache"
    assert isinstance(keywords.get("cache_size"), ast.Name)
    assert keywords["cache_size"].id == "cache_size"


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


@pytest.mark.asyncio
async def test_preprocessor_cache_keys_include_preprocessing_context(
    monkeypatch, tmp_path
) -> None:
    model_dir = tmp_path / "snapshot"
    model_dir.mkdir()

    class DummyProcessorFactory:
        @classmethod
        def from_pretrained(cls, model_path: str, **kwargs):
            return SimpleNamespace(
                tokenizer=SimpleNamespace(chat_template="dummy-template"),
                apply_chat_template=lambda *args, **kwargs: "prompt",
                __call__=None,
            )

    class DummyProcessor:
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

    monkeypatch.setattr(
        preprocessor,
        "resolve_model_path",
        lambda model_path, *, local_files_only=False: model_dir,
    )
    monkeypatch.setattr(preprocessor, "Qwen3OmniMoeProcessor", DummyProcessorFactory)

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
    monkeypatch.setattr(preprocessor, "compute_image_cache_key", lambda *_: None)
    monkeypatch.setattr(preprocessor, "compute_video_cache_key", lambda *_: "video-key")
    monkeypatch.setattr(preprocessor, "compute_audio_cache_key", lambda *_: "audio-key")

    proc = preprocessor.Qwen3OmniPreprocessor("Qwen/Qwen3-Omni-30B-A3B-Instruct")
    proc.processor = DummyProcessor()

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

    assert (
        state.encoder_inputs["image_encoder"]["cache_key"]
        == "video-key|fps=(7.5,)|seconds_per_chunk=2.5"
    )
    assert (
        state.encoder_inputs["audio_encoder"]["cache_key"]
        == "audio-key|target_sr=22050"
    )
