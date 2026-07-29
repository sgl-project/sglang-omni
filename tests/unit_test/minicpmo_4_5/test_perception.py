# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io
import sys
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

if sys.platform == "darwin":
    # Note:(Chenchen Hong) SGLang decorates unrelated CUDA kernels while this
    # dependency-light test imports shared request data; macOS has no Triton.
    _torch_compile = torch.compile

    def _identity_compile(*args: Any, **kwargs: Any):
        if args and callable(args[0]):
            return args[0]
        return lambda function: function

    torch.compile = _identity_compile

from sglang_omni.models.minicpmo_4_5.perception import (
    MiniCPMO45Perception,
    PerceptionError,
)

if sys.platform == "darwin":
    torch.compile = _torch_compile


class _Batch(dict):
    def to(self, device: Any):
        self["moved_to"] = str(device)
        return self


class _Tokenizer:
    unk_token_id = 0

    _ids = {
        "<unit>": 1,
        "</unit>": 6,
        "<image>": 2,
        "</image>": 3,
        "<slice>": 4,
        "</slice>": 5,
        "<|listen|>": 7,
        "<|speak|>": 8,
        "<|tts_bos|>": 9,
        "<|tts_eos|>": 10,
        "<|tts_pad|>": 11,
        "<|chunk_eos|>": 12,
        "<|chunk_tts_eos|>": 13,
        "<|turn_eos|>": 14,
        "<unk>": 0,
    }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._ids.get(token, self.unk_token_id)


class _ImageProcessor:
    def get_sliced_grid(
        self,
        image_size: tuple[int, int],
        max_slice_nums: int,
        nerver_split: bool = False,
    ):
        del image_size, nerver_split
        return (1, 1) if max_slice_nums > 1 else None


class _Processor:
    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()
        self.image_processor = _ImageProcessor()
        self.mode_kwargs: dict[str, Any] | None = None
        self.chunk_index = 0
        self.processed_chunks: list[np.ndarray] = []
        self.restored: list[dict[str, int]] = []

    def set_streaming_mode(self, **kwargs: Any) -> None:
        self.mode_kwargs = kwargs

    def reset_streaming(self) -> None:
        self.chunk_index = 0

    def get_streaming_chunk_size(self) -> int:
        return 16_480 if self.chunk_index == 0 else 16_000

    def get_streaming_snapshot(self) -> dict[str, int]:
        return {"chunk_index": self.chunk_index}

    def restore_streaming_snapshot(self, snapshot: dict[str, int]) -> None:
        self.restored.append(snapshot)
        self.chunk_index = snapshot["chunk_index"]

    def process_audio_streaming(self, chunk: np.ndarray, **kwargs: Any) -> _Batch:
        self.processed_chunks.append(chunk.copy())
        frames = 102 if self.chunk_index == 0 else 104
        self.chunk_index += 1
        return _Batch(
            audio_features=torch.zeros(1, 80, frames),
            audio_feature_lens=[torch.tensor([frames])],
            streaming_info={"emitted_frames": frames},
        )

    def process_audio(self, audios: list[np.ndarray], **kwargs: Any) -> _Batch:
        assert len(audios) == 1
        return _Batch(
            audio_features=torch.zeros(1, 80, 100),
            audio_feature_lens=[torch.tensor([100])],
        )

    def process_image(self, frames: list[Image.Image], **kwargs: Any) -> _Batch:
        assert all(frame.mode == "RGB" for frame in frames)
        num_slices = 2 if kwargs["max_slice_nums"] > 1 else 1
        total_slices = len(frames) * num_slices
        return _Batch(
            pixel_values=[[torch.zeros(3, 2, 2) for _ in range(total_slices)]],
            tgt_sizes=[torch.tensor([[1, 1]] * total_slices, dtype=torch.int32)],
        )


class _Model:
    def __init__(self) -> None:
        self.audio_calls: list[dict[str, Any]] = []

    def encode_audio_streaming(self, batch: _Batch, **kwargs: Any):
        self.audio_calls.append({"batch": batch, **kwargs})
        call = len(self.audio_calls)
        return [[torch.full((3, 4), float(call))]], f"cache-{call}"

    def get_image_feature(self, items: list[Any]) -> torch.Tensor:
        count = sum(len(item.feature) for item in items)
        return torch.arange(count * 2 * 4, dtype=torch.float32).reshape(count, 2, 4)

    def get_audio_feature(self, items: list[Any]) -> torch.Tensor:
        assert items[0].audio_feature_lens[0].item() == 100
        return torch.ones(2, 4)


class _FailingModel(_Model):
    def encode_audio_streaming(self, batch: _Batch, **kwargs: Any):
        raise RuntimeError("encoder exploded")


def _pcm(value: int = 1_000) -> str:
    samples = np.full(16_000, value, dtype="<i2")
    return base64.b64encode(samples.tobytes()).decode("ascii")


def _frame() -> str:
    image = Image.new("RGB", (4, 3), color=(10, 20, 30))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def _perception(
    model: Any | None = None,
) -> tuple[MiniCPMO45Perception, list[_Processor]]:
    processors: list[_Processor] = []

    def factory() -> _Processor:
        processor = _Processor()
        processors.append(processor)
        return processor

    return (
        MiniCPMO45Perception(
            processor_factory=factory,
            model=model or _Model(),
        ),
        processors,
    )


def test_tokenizer_is_reused_by_first_session_and_exact_mode_is_configured() -> None:
    perception, processors = _perception()

    assert perception.tokenizer.convert_tokens_to_ids("<unit>") == 1
    assert len(processors) == 1
    perception.open_session("one")

    assert len(processors) == 1
    assert processors[0].mode_kwargs == {
        "mode": "exact",
        "chunk_ms": 1_000,
        "first_chunk_ms": 1_035,
        "cnn_redundancy_ms": 20,
        "enable_sliding_window": True,
        "slide_trigger_seconds": 30.0,
        "slide_stride_seconds": 10.0,
    }


def test_audio_units_keep_per_session_cache_and_exact_first_chunk_alignment() -> None:
    model = _Model()
    perception, processors = _perception(model)
    perception.open_session("one")

    first = perception.prepare_unit("one", {"audio_pcm16_b64": _pcm()})
    second = perception.prepare_unit("one", {"audio_pcm16_b64": _pcm(2_000)})

    processor = processors[0]
    assert [chunk.size for chunk in processor.processed_chunks] == [16_480, 16_000]
    assert np.count_nonzero(processor.processed_chunks[0][:560]) == 0
    assert np.allclose(processor.processed_chunks[0][560:], 1_000 / 32768.0)
    assert np.allclose(processor.processed_chunks[1][:80], 1_000 / 32768.0)
    assert first.audio is not None
    assert first.audio.consumed_samples == 16_480
    assert first.audio.buffered_samples == 80
    assert first.audio.emitted_frames == 102
    assert first.audio.prefix_extra_frames == 0
    assert second.audio is not None
    assert second.audio.consumed_samples == 16_000
    assert second.audio.buffered_samples == 80
    assert second.audio.emitted_frames == 104
    assert second.audio.prefix_extra_frames == 2
    assert [call["past_key_values"] for call in model.audio_calls] == [
        None,
        "cache-1",
    ]
    assert first.input_ids == (1, 0, 0, 0)
    assert first.embedding_spans[0].modality == "audio"
    assert (first.embedding_spans[0].start, first.embedding_spans[0].end) == (
        1,
        4,
    )


def test_video_slices_become_boundary_tokens_and_embedding_spans() -> None:
    perception, _ = _perception()
    perception.open_session("vision")

    unit = perception.prepare_unit(
        "vision",
        {"video_frames": [_frame()], "max_slice_nums": 2},
    )

    assert unit.mode == "VISION"
    assert unit.input_ids == (1, 2, 0, 0, 3, 4, 0, 0, 5)
    assert [(span.start, span.end, span.modality) for span in unit.embedding_spans] == [
        (2, 4, "image"),
        (6, 8, "image"),
    ]
    merged = unit.materialize_embeddings(torch.zeros(len(unit.input_ids), 4))
    assert torch.equal(merged[2:4], unit.embedding_spans[0].embedding)
    assert torch.equal(merged[6:8], unit.embedding_spans[1].embedding)


def test_omni_unit_orders_vision_before_audio_like_official_duplex_prefill() -> None:
    perception, _ = _perception()
    perception.open_session("omni")

    unit = perception.prepare_unit(
        "omni",
        {
            "audio_pcm16_b64": _pcm(),
            "video_frames": [_frame()],
            "max_slice_nums": 1,
        },
    )

    assert unit.mode == "OMNI"
    assert [span.modality for span in unit.embedding_spans] == ["image", "audio"]
    assert unit.input_ids[:5] == (1, 2, 0, 0, 3)
    assert unit.embedding_spans[1].start == 5


def test_encoder_failure_restores_mel_snapshot_and_poisons_cache_state() -> None:
    perception, processors = _perception(_FailingModel())
    perception.open_session("broken")

    with pytest.raises(PerceptionError, match="encoder exploded"):
        perception.prepare_unit("broken", {"audio_pcm16_b64": _pcm()})

    assert processors[0].chunk_index == 0
    assert processors[0].restored == [{"chunk_index": 0}]
    with pytest.raises(PerceptionError, match="close and reopen"):
        perception.prepare_unit("broken", {"audio_pcm16_b64": _pcm()})


def test_session_processors_and_whisper_caches_are_isolated() -> None:
    model = _Model()
    perception, processors = _perception(model)
    perception.open_session("a")
    perception.open_session("b")

    perception.prepare_unit("a", {"audio_pcm16_b64": _pcm()})
    perception.prepare_unit("b", {"audio_pcm16_b64": _pcm()})

    assert processors[0] is not processors[1]
    assert [call["past_key_values"] for call in model.audio_calls] == [None, None]
    perception.close_session("a")
    with pytest.raises(PerceptionError, match="unknown perception session"):
        perception.prepare_unit("a", {"audio_pcm16_b64": _pcm()})


def test_reference_audio_uses_offline_encoder_without_touching_stream_cache() -> None:
    perception, processors = _perception()
    perception.open_session("voice")

    embedding = perception.prepare_reference_audio(
        "voice", np.zeros(16_000, dtype=np.float32)
    )

    assert embedding.shape == (2, 4)
    assert processors[0].chunk_index == 0


def test_missing_session_safe_audio_interface_fails_fast() -> None:
    perception, _ = _perception(model=object())
    perception.open_session("missing")

    with pytest.raises(PerceptionError, match="session-safe encode_audio_streaming"):
        perception.prepare_unit("missing", {"audio_pcm16_b64": _pcm()})


def test_from_pretrained_pins_remote_processor_revision(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_from_pretrained(model_path: str, **kwargs: Any) -> _Processor:
        calls.append((model_path, kwargs))
        return _Processor()

    monkeypatch.setattr(
        "transformers.AutoProcessor.from_pretrained",
        fake_from_pretrained,
    )
    perception = MiniCPMO45Perception.from_pretrained(
        "openbmb/MiniCPM-o-4_5",
        model=_Model(),
        revision="checkpoint-sha",
    )

    assert isinstance(perception.tokenizer, _Tokenizer)
    assert calls == [
        (
            "openbmb/MiniCPM-o-4_5",
            {"trust_remote_code": True, "revision": "checkpoint-sha"},
        )
    ]
