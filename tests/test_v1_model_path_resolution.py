# SPDX-License-Identifier: Apache-2.0
"""V1 model-path and encoder-cache regression tests."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sglang_omni_v1.models import weight_loader
from sglang_omni_v1.models.qwen3_omni import request_builders
from sglang_omni_v1.models.qwen3_omni.components import preprocessor
from sglang_omni_v1.models.qwen3_omni.payload_types import PipelineState
from sglang_omni_v1.proto import OmniRequest, StagePayload

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def _patch_module(monkeypatch, module, **attrs) -> None:
    for name, value in attrs.items():
        monkeypatch.setattr(module, name, value)


class RecordingProcessor:
    tokenizer = SimpleNamespace(chat_template="dummy-template")

    def __init__(self) -> None:
        self.template_messages = None
        self.call_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_messages = messages
        return "prompt"

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        output = {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }
        if kwargs.get("images") is not None:
            output.update(
                {
                    "pixel_values": torch.zeros((1, 3), dtype=torch.float32),
                    "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long),
                }
            )
        if kwargs.get("videos") is not None:
            output.update(
                {
                    "pixel_values_videos": torch.zeros((2, 3), dtype=torch.float32),
                    "video_grid_thw": torch.tensor([[1, 2, 3]], dtype=torch.long),
                    "video_second_per_grid": torch.tensor([0.5], dtype=torch.float32),
                }
            )
        if kwargs.get("audio") is not None:
            output.update(
                {
                    "input_features": torch.zeros((1, 4), dtype=torch.float32),
                    "feature_attention_mask": torch.ones((1, 4), dtype=torch.long),
                    "audio_feature_lengths": torch.tensor([4], dtype=torch.long),
                }
            )
        return output


@pytest.fixture
def qwen3_preprocessor_testbed(monkeypatch, tmp_path):
    model_dir = tmp_path / "snapshot"
    model_dir.mkdir()
    processor_instance = RecordingProcessor()

    class DummyProcessorFactory:
        @classmethod
        def from_pretrained(cls, model_path: str, **kwargs):
            assert model_path == str(model_dir)
            assert kwargs["local_files_only"] is True
            return processor_instance

    _patch_module(
        monkeypatch,
        preprocessor,
        resolve_model_path=lambda model_path, *, local_files_only=False: model_dir,
        Qwen3OmniMoeProcessor=DummyProcessorFactory,
        ensure_chat_template=lambda tokenizer, *, model_path: None,
        compute_image_cache_key=lambda value: "image-key" if value else None,
        compute_video_cache_key=lambda value: "video-key" if value else None,
        compute_audio_cache_key=lambda value: "audio-key" if value else None,
    )

    proc = preprocessor.Qwen3OmniPreprocessor(MODEL_ID)
    return SimpleNamespace(proc=proc, processor=processor_instance)


def test_v1_qwen3_preprocessor_falls_back_to_remote_processor_download(
    monkeypatch, tmp_path
) -> None:
    local_snapshot = tmp_path / "snapshot"
    refreshed_snapshot = tmp_path / "resolved-after-remote"
    local_snapshot.mkdir()
    refreshed_snapshot.mkdir()
    processor_calls: list[tuple[str, bool]] = []
    resolve_calls: list[bool] = []

    class DummyProcessorFactory:
        @classmethod
        def from_pretrained(cls, model_path: str, **kwargs):
            processor_calls.append(
                (str(model_path), bool(kwargs.get("local_files_only")))
            )
            if kwargs.get("local_files_only"):
                raise OSError("missing processor assets")
            return SimpleNamespace(tokenizer=SimpleNamespace(chat_template="dummy"))

    def fake_resolve_model_path(model_path: str, *, local_files_only: bool = False):
        assert model_path == MODEL_ID
        resolve_calls.append(local_files_only)
        return local_snapshot if local_files_only else refreshed_snapshot

    _patch_module(
        monkeypatch,
        preprocessor,
        resolve_model_path=fake_resolve_model_path,
        Qwen3OmniMoeProcessor=DummyProcessorFactory,
        ensure_chat_template=lambda tokenizer, *, model_path: None,
    )

    proc = preprocessor.Qwen3OmniPreprocessor(MODEL_ID)

    assert proc.model_dir == str(refreshed_snapshot)
    assert processor_calls == [(str(local_snapshot), True), (MODEL_ID, False)]
    assert resolve_calls == [True, False]


def test_v1_resolve_local_model_dir_propagates_unexpected_errors(monkeypatch) -> None:
    def fake_resolve_model_path(model_path: str, *, local_files_only: bool = False):
        raise RuntimeError("boom")

    _patch_module(monkeypatch, preprocessor, resolve_model_path=fake_resolve_model_path)

    with pytest.raises(RuntimeError, match="boom"):
        preprocessor._resolve_local_model_dir(MODEL_ID)


def test_v1_preprocessor_builds_contextual_encoder_cache_keys_and_video_kwargs(
    monkeypatch, qwen3_preprocessor_testbed
) -> None:
    video_loader_kwargs: dict[str, Any] = {}
    audio_loader_kwargs: dict[str, Any] = {}

    async def fake_ensure_video_list_async(videos, **kwargs):
        assert videos == ["video.mp4"]
        video_loader_kwargs.update(kwargs)
        return [torch.zeros((2, 3), dtype=torch.float32)], [7.5], []

    async def fake_ensure_image_list_async(images, **kwargs):
        assert images is None
        return []

    async def fake_ensure_audio_list_async(audios, **kwargs):
        assert audios == ["audio.wav"]
        audio_loader_kwargs.update(kwargs)
        return [torch.zeros(4)]

    _patch_module(
        monkeypatch,
        preprocessor,
        ensure_video_list_async=fake_ensure_video_list_async,
        ensure_image_list_async=fake_ensure_image_list_async,
        ensure_audio_list_async=fake_ensure_audio_list_async,
    )

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

    result = asyncio.run(qwen3_preprocessor_testbed.proc(payload))
    state = PipelineState.from_dict(result.data)
    processor_kwargs = qwen3_preprocessor_testbed.processor.call_kwargs

    assert video_loader_kwargs["fps"] == 12.0
    assert video_loader_kwargs["max_frames"] == 128
    assert video_loader_kwargs["max_pixels"] == 401408
    assert video_loader_kwargs["extract_audio"] is False
    assert video_loader_kwargs["audio_target_sr"] == 22050
    assert audio_loader_kwargs["target_sr"] == 22050
    assert processor_kwargs["videos_kwargs"]["fps"] == 7.5
    assert processor_kwargs["videos_kwargs"]["max_frames"] == 128
    assert processor_kwargs["videos_kwargs"]["max_pixels"] == 401408
    assert processor_kwargs["videos_kwargs"]["seconds_per_chunk"] == 2.5
    assert (
        state.encoder_inputs["image_encoder"]["cache_key"]
        == "video-key|fps=(7.5,)|max_frames=128|max_pixels=401408|seconds_per_chunk=2.5"
    )
    assert (
        state.encoder_inputs["audio_encoder"]["cache_key"]
        == "audio-key|target_sr=22050"
    )


def test_v1_preprocessor_rejects_default_generation_over_context(
    qwen3_preprocessor_testbed,
) -> None:
    qwen3_preprocessor_testbed.proc.max_seq_len = 4
    payload = StagePayload(
        request_id="req-over-context",
        request=OmniRequest(
            inputs={"messages": [{"role": "user", "content": "too long"}]},
            params={},
        ),
        data=None,
    )

    with pytest.raises(ValueError, match="Requested token count exceeds"):
        asyncio.run(qwen3_preprocessor_testbed.proc(payload))


def test_v1_preprocessor_cache_key_uses_explicit_fps_when_loader_returns_none(
    monkeypatch, qwen3_preprocessor_testbed
) -> None:
    async def fake_ensure_video_list_async(videos, **kwargs):
        assert videos == ["video.mp4"]
        assert kwargs["fps"] == 7.5
        return [torch.zeros((2, 3), dtype=torch.float32)], None, []

    async def fake_ensure_image_list_async(images, **kwargs):
        assert images is None
        return []

    async def fake_ensure_audio_list_async(audios, **kwargs):
        assert audios is None
        return []

    _patch_module(
        monkeypatch,
        preprocessor,
        ensure_video_list_async=fake_ensure_video_list_async,
        ensure_image_list_async=fake_ensure_image_list_async,
        ensure_audio_list_async=fake_ensure_audio_list_async,
    )

    payload = StagePayload(
        request_id="req-explicit-fps",
        request=OmniRequest(
            inputs={
                "messages": [{"role": "user", "content": "describe"}],
                "videos": ["video.mp4"],
                "video_fps": 7.5,
            }
        ),
        data=None,
    )

    result = asyncio.run(qwen3_preprocessor_testbed.proc(payload))
    state = PipelineState.from_dict(result.data)

    assert (
        qwen3_preprocessor_testbed.processor.call_kwargs["videos_kwargs"]["fps"] == 7.5
    )
    assert state.encoder_inputs["image_encoder"]["cache_key"] == "video-key|fps=(7.5,)"


def test_v1_preprocessor_video_audio_has_no_extra_audio_placeholder(
    monkeypatch, qwen3_preprocessor_testbed
) -> None:
    video_loader_kwargs: dict[str, Any] = {}

    async def fake_ensure_video_list_async(videos, **kwargs):
        assert videos == ["video.mp4"]
        video_loader_kwargs.update(kwargs)
        return (
            [
                torch.zeros((2, 3), dtype=torch.float32),
            ],
            [6.0],
            [torch.zeros(4, dtype=torch.float32)],
        )

    async def fake_ensure_image_list_async(images, **kwargs):
        assert images is None
        return []

    async def fake_ensure_audio_list_async(audios, **kwargs):
        assert audios is None
        return []

    _patch_module(
        monkeypatch,
        preprocessor,
        ensure_video_list_async=fake_ensure_video_list_async,
        ensure_image_list_async=fake_ensure_image_list_async,
        ensure_audio_list_async=fake_ensure_audio_list_async,
    )

    payload = StagePayload(
        request_id="req-video-audio",
        request=OmniRequest(
            inputs={
                "messages": [{"role": "user", "content": "transcribe"}],
                "videos": ["video.mp4"],
                "use_audio_in_video": True,
                "audio_target_sr": 24000,
            }
        ),
        data=None,
    )

    result = asyncio.run(qwen3_preprocessor_testbed.proc(payload))
    state = PipelineState.from_dict(result.data)
    content_parts = qwen3_preprocessor_testbed.processor.template_messages[-1][
        "content"
    ]

    assert [part["type"] for part in content_parts] == ["video", "text"]
    assert video_loader_kwargs["extract_audio"] is True
    assert video_loader_kwargs["audio_target_sr"] == 24000
    assert qwen3_preprocessor_testbed.processor.call_kwargs["audio"] is not None
    assert (
        qwen3_preprocessor_testbed.processor.call_kwargs["videos_kwargs"][
            "use_audio_in_video"
        ]
        is True
    )
    assert state.encoder_inputs["image_encoder"]["cache_key"] == "video-key|fps=(6.0,)"
    assert (
        state.encoder_inputs["audio_encoder"]["cache_key"]
        == "video-key|extracted_audio=True|target_sr=24000"
    )


def test_v1_project_preprocessing_to_encoder_routes_one_encoder_input() -> None:
    payload = _payload_with_state(
        PipelineState(
            encoder_inputs={
                "image_encoder": {
                    "pixel_values": torch.ones((1, 3)),
                    "cache_key": "image-key",
                },
                "audio_encoder": {
                    "input_features": torch.ones((1, 4)),
                    "cache_key": "audio-key",
                },
            }
        )
    )

    image_payload = request_builders.project_preprocessing_to_image_encoder(payload)
    audio_payload = request_builders.project_preprocessing_to_audio_encoder(payload)

    image_state = PipelineState.from_dict(image_payload.data)
    audio_state = PipelineState.from_dict(audio_payload.data)
    assert set(image_state.encoder_inputs) == {"image_encoder"}
    assert image_state.encoder_inputs["image_encoder"]["cache_key"] == "image-key"
    assert set(audio_state.encoder_inputs) == {"audio_encoder"}
    assert audio_state.encoder_inputs["audio_encoder"]["cache_key"] == "audio-key"


def test_v1_mm_aggregate_projection_keeps_metadata_not_raw_tensors() -> None:
    payload = _payload_with_state(
        PipelineState(
            prompt={
                "prompt_text": "prompt",
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
            },
            mm_inputs={
                "image": {
                    "pixel_values": torch.ones((1, 3)),
                    "image_grid_thw": torch.tensor([[1, 1, 1]]),
                },
                "video": {
                    "pixel_values_videos": torch.ones((2, 3)),
                    "video_grid_thw": torch.tensor([[1, 2, 3]]),
                    "video_second_per_grid": torch.tensor([0.5]),
                    "use_audio_in_video": True,
                },
                "audio": {
                    "input_features": torch.ones((1, 4)),
                    "feature_attention_mask": torch.ones((1, 4)),
                    "audio_feature_lengths": torch.tensor([4]),
                },
            },
            encoder_inputs={
                "image_encoder": {
                    "pixel_values": torch.ones((1, 3)),
                    "pixel_values_videos": torch.ones((2, 3)),
                    "cache_key": "image-key",
                },
                "audio_encoder": {
                    "input_features": torch.ones((1, 4)),
                    "cache_key": "audio-key",
                },
            },
            stream_state={"token_ids": [1], "text": "h"},
        )
    )

    projected = request_builders.project_preprocessing_to_mm_aggregate(payload)
    state = PipelineState.from_dict(projected.data)

    assert state.prompt["prompt_text"] == "prompt"
    assert state.stream_state == {"token_ids": [1], "text": "h"}
    assert state.encoder_inputs == {
        "image_encoder": {"cache_key": "image-key"},
        "audio_encoder": {"cache_key": "audio-key"},
    }
    assert "pixel_values" not in state.mm_inputs["image"]
    assert "pixel_values_videos" not in state.mm_inputs["video"]
    assert "input_features" not in state.mm_inputs["audio"]
    assert "image_grid_thw" in state.mm_inputs["image"]
    assert "video_grid_thw" in state.mm_inputs["video"]
    assert "feature_attention_mask" in state.mm_inputs["audio"]


def test_v1_build_encoder_request_strips_cache_key_before_model_call() -> None:
    state = PipelineState(
        encoder_inputs={
            "image_encoder": {
                "pixel_values": torch.ones((1, 3)),
                "image_grid_thw": torch.tensor([[1, 1, 1]]),
                "cache_key": "media-key",
            }
        }
    )

    request = request_builders.build_encoder_request(
        state,
        stage_name="image_encoder",
    )

    assert request.cache_key == "media-key"
    assert "cache_key" not in request.model_inputs
    assert set(request.model_inputs) == {"pixel_values", "image_grid_thw"}


def test_v1_image_encoder_request_cost_counts_raw_and_deepstack_bytes() -> None:
    from sglang_omni_v1.models.qwen3_omni import stages

    model = SimpleNamespace(
        spatial_merge_size=2,
        out_hidden_size=8,
        deepstack_layers=3,
        visual_dtype_bytes=2,
    )
    payload = StagePayload(
        request_id="mixed",
        request=OmniRequest(inputs={}),
        data=PipelineState(
            encoder_inputs={
                "image_encoder": {
                    "pixel_values_videos": torch.zeros(
                        (16, 3),
                        dtype=torch.float32,
                    ),
                    "video_grid_thw": torch.tensor([[2, 4, 4]], dtype=torch.long),
                    "pixel_values": torch.zeros((4, 3), dtype=torch.float16),
                    "image_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.long),
                }
            }
        ).to_dict(),
    )

    cost = stages._create_image_encoder_request_cost_fn(model)(payload)

    raw_video_bytes = 16 * 3 * 4
    raw_image_bytes = 4 * 3 * 2
    video_output_bytes = 8 * 8 * 2 * 4
    image_output_bytes = 4 * 8 * 2 * 4
    expected = (
        raw_video_bytes + raw_image_bytes + video_output_bytes + image_output_bytes
    ) * stages.QWEN3_IMAGE_ENCODER_ACTIVATION_MULTIPLIER
    assert cost == expected


def test_v1_image_encoder_request_cost_ignores_cached_skip() -> None:
    from sglang_omni_v1.models.qwen3_omni import stages

    model = SimpleNamespace(
        spatial_merge_size=2,
        out_hidden_size=8,
        deepstack_layers=3,
        visual_dtype_bytes=2,
    )
    payload = StagePayload(
        request_id="cached",
        request=OmniRequest(inputs={}),
        data=PipelineState(
            encoder_inputs={"image_encoder": {"_skip": True, "_result": {}}}
        ).to_dict(),
    )

    assert stages._create_image_encoder_request_cost_fn(model)(payload) == 0


def test_v1_encoder_stage_cache_key_bypasses_model_on_hit() -> None:
    from sglang_omni_v1.models.qwen3_omni import stages

    cache = FakeCache()
    model = CountingModel()

    first = stages._run_single_encoder_payload(
        _encoder_payload(request_id="req-1", cache_key="media-key"),
        stage_name="image_encoder",
        model=model,
        cache_manager=cache,
    )
    second = stages._run_single_encoder_payload(
        _encoder_payload(request_id="req-2", cache_key="media-key"),
        stage_name="image_encoder",
        model=model,
        cache_manager=cache,
    )

    first_state = PipelineState.from_dict(first.data)
    second_state = PipelineState.from_dict(second.data)
    assert len(model.calls) == 1
    assert "cache_key" not in model.calls[0]
    assert cache.get_keys == ["media-key", "media-key"]
    assert cache.put_keys == ["media-key"]
    assert torch.equal(
        first_state.encoder_outs["image_encoder"]["encoded"],
        second_state.encoder_outs["image_encoder"]["encoded"],
    )


def test_v1_batch_image_encoder_payloads_deduplicates_same_batch_cache_key() -> None:
    from sglang_omni_v1.models.qwen3_omni import stages

    cache = FakeCache()
    model = BatchImageModel()
    payloads = [
        _batch_encoder_payload(request_id="req-1", cache_key="same-key"),
        _batch_encoder_payload(request_id="req-2", cache_key="same-key"),
    ]

    results = stages._batch_image_encoder_payloads(
        payloads,
        model=model,
        cache_manager=cache,
    )

    states = [PipelineState.from_dict(result.data) for result in results]
    assert len(model.calls) == 1
    assert cache.get_keys == ["same-key", "same-key"]
    assert cache.put_keys == ["same-key"]
    assert torch.equal(
        states[0].encoder_outs["image_encoder"]["image_embeds"],
        states[1].encoder_outs["image_encoder"]["image_embeds"],
    )


def test_v1_weight_loader_force_refreshes_partial_remote_snapshot(
    monkeypatch, tmp_path
) -> None:
    partial_snapshot = tmp_path / "partial"
    refreshed_snapshot = tmp_path / "refreshed"
    partial_snapshot.mkdir()
    refreshed_snapshot.mkdir()
    refresh_calls: list[tuple[str, bool, bool]] = []
    load_attempts: list[Path] = []

    def fake_resolve_model_path(model_path: str, *, local_files_only: bool = False):
        assert model_path == MODEL_ID
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
        MODEL_ID,
        prefix=("thinker.visual.", "visual."),
        local_files_only=False,
    )

    assert state_dict == {"proj.weight": "loaded"}
    assert refresh_calls == [(MODEL_ID, False, True)]
    assert load_attempts[0] == partial_snapshot
    assert refreshed_snapshot in load_attempts


def test_v1_weight_loader_force_refreshes_missing_remote_shard(
    monkeypatch, tmp_path
) -> None:
    partial_snapshot = tmp_path / "partial"
    refreshed_snapshot = tmp_path / "refreshed"
    partial_snapshot.mkdir()
    refreshed_snapshot.mkdir()
    refresh_calls: list[tuple[str, bool, bool]] = []
    load_attempts: list[Path] = []

    def fake_resolve_model_path(model_path: str, *, local_files_only: bool = False):
        assert model_path == MODEL_ID
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
        MODEL_ID,
        prefix=("thinker.visual.", "visual."),
        local_files_only=False,
    )

    assert state_dict == {"proj.weight": "loaded"}
    assert refresh_calls == [(MODEL_ID, False, True)]
    assert load_attempts[0] == partial_snapshot
    assert refreshed_snapshot in load_attempts


def _payload_with_state(state: PipelineState) -> StagePayload:
    return StagePayload(
        request_id="req",
        request=OmniRequest(inputs={}),
        data=state.to_dict(),
    )


def _encoder_payload(*, request_id: str, cache_key: str) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs={}),
        data=PipelineState(
            encoder_inputs={
                "image_encoder": {
                    "pixel_values": torch.ones((1, 3)),
                    "image_grid_thw": torch.tensor([[1, 1, 1]]),
                    "cache_key": cache_key,
                }
            }
        ).to_dict(),
    )


def _batch_encoder_payload(*, request_id: str, cache_key: str) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs={}),
        data=PipelineState(
            encoder_inputs={
                "image_encoder": {
                    "pixel_values": torch.ones((1, 3)),
                    "image_grid_thw": torch.tensor([[1, 1, 1]]),
                    "cache_key": cache_key,
                }
            }
        ).to_dict(),
    )


class FakeCache:
    def __init__(self) -> None:
        self.store = {}
        self.get_keys: list[str] = []
        self.put_keys: list[str] = []

    def get(self, request):
        key = request.data.cache_key
        self.get_keys.append(key)
        value = self.store.get(key)
        return SimpleNamespace(data=value) if value is not None else None

    def put(self, request, output):
        key = request.data.cache_key
        self.put_keys.append(key)
        self.store[key] = output.data


class CountingModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"encoded": torch.tensor([len(self.calls)])}


class BatchImageModel:
    spatial_merge_size = 1

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        grid = kwargs["image_grid_thw"].to(dtype=torch.long)
        token_counts = grid.prod(dim=-1)
        token_total = int(token_counts.sum().item())
        return {
            "image_embeds": torch.arange(token_total, dtype=torch.float32).unsqueeze(1),
            "image_grid_thw": grid,
            "image_token_counts": token_counts,
            "deepstack_visual_embeds_image": [
                torch.arange(token_total, dtype=torch.float32).unsqueeze(1)
            ],
        }
