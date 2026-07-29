# SPDX-License-Identifier: Apache-2.0
"""Session-scoped perception for native MiniCPM-o 4.5 duplex inference.

The official processor owns mutable exact-streaming Mel state.  This module
keeps that state, together with the Whisper KV cache, inside one duplex
session.  It emits ordinary token ids plus an explicit embedding-span ledger;
the SGLang request builder can therefore keep the language-model context in
the normal paged KV cache without importing or launching the demo runtime.
"""

from __future__ import annotations

import base64
import io
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch

from .protocol import (
    INPUT_SAMPLE_RATE,
    MAX_REFERENCE_AUDIO_SAMPLES,
    normalize_append_data,
)
from .state import EmbeddingSpan, MiniCPMOSpecialTokens

CHUNK_MS = 1_000
FIRST_CHUNK_MS = 1_035
CNN_REDUNDANCY_MS = 20
PREFIX_EXTRA_FRAMES = 2
SUFFIX_EXTRA_FRAMES = 2


class PerceptionError(RuntimeError):
    """Raised when a duplex perception session cannot safely continue."""


@dataclass(frozen=True)
class StreamingAudioMetadata:
    """Exact-streaming accounting for one accepted PCM unit."""

    chunk_index: int
    input_samples: int
    consumed_samples: int
    buffered_samples: int
    emitted_frames: int
    prefix_extra_frames: int
    suffix_extra_frames: int
    processor_info: Mapping[str, Any]


@dataclass(frozen=True)
class PreparedPerceptionUnit:
    """One finite SGLang append: token ids and their media replacements."""

    input_ids: tuple[int, ...]
    embedding_spans: tuple[EmbeddingSpan, ...]
    mode: Literal["AUDIO", "VISION", "OMNI"]
    unit_index: int
    video_frame_count: int
    audio: StreamingAudioMetadata | None = None

    def materialize_embeddings(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Apply the span ledger to a request-local token embedding tensor."""

        if token_embeddings.ndim != 2:
            raise ValueError("token_embeddings must have shape [tokens, hidden]")
        if token_embeddings.shape[0] != len(self.input_ids):
            raise ValueError("token embedding rows must match input_ids")
        result = token_embeddings.clone()
        for span in self.embedding_spans:
            if span.embedding.shape[1] != result.shape[1]:
                raise ValueError(
                    "media and token embeddings have different hidden sizes"
                )
            result[span.start : span.end] = span.embedding.to(
                device=result.device,
                dtype=result.dtype,
            )
        return result


@dataclass
class _SessionState:
    processor: Any
    token_ids: MiniCPMOSpecialTokens
    audio_buffer: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    audio_past_key_values: Any = None
    audio_chunk_index: int = 0
    unit_index: int = 0
    poisoned_reason: str | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)


@dataclass
class _MediaItem:
    feature: Any
    model_specific_data: dict[str, Any]
    format: Any = None

    def __getattr__(self, name: str) -> Any:
        try:
            return self.model_specific_data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class MiniCPMO45Perception:
    """Native, same-process audio/video preparation for duplex requests."""

    def __init__(
        self,
        *,
        processor_factory: Callable[[], Any],
        model: Any,
        device: str | torch.device | None = None,
        chunk_ms: int = CHUNK_MS,
        first_chunk_ms: int = FIRST_CHUNK_MS,
        cnn_redundancy_ms: int = CNN_REDUNDANCY_MS,
    ) -> None:
        if not callable(processor_factory):
            raise TypeError("processor_factory must be callable")
        if chunk_ms != CHUNK_MS or first_chunk_ms != FIRST_CHUNK_MS:
            raise ValueError(
                "MiniCPM-o duplex requires 1000 ms units and a 1035 ms first chunk"
            )
        if cnn_redundancy_ms != CNN_REDUNDANCY_MS:
            raise ValueError("MiniCPM-o duplex requires 20 ms CNN redundancy")

        self._processor_factory = processor_factory
        self._model = model
        self._device = device
        self._chunk_ms = chunk_ms
        self._first_chunk_ms = first_chunk_ms
        self._cnn_redundancy_ms = cnn_redundancy_ms
        self._sessions: dict[str, _SessionState] = {}
        self._idle_processor: Any = None
        self._active_processor_ids: set[int] = set()
        self._lock = threading.RLock()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        model: Any,
        revision: str | None = None,
        device: str | torch.device | None = None,
        trust_remote_code: bool = True,
        processor_kwargs: Mapping[str, Any] | None = None,
    ) -> "MiniCPMO45Perception":
        """Create the adapter from the checkpoint-owned HF processor."""

        if not isinstance(model_path, str) or not model_path.strip():
            raise ValueError("model_path must be a non-empty string")
        try:
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required to load the MiniCPM-o processor"
            ) from exc

        load_kwargs = dict(processor_kwargs or {})
        load_kwargs.setdefault("trust_remote_code", trust_remote_code)
        if revision is not None:
            load_kwargs.setdefault("revision", revision)

        def processor_factory() -> Any:
            return AutoProcessor.from_pretrained(model_path, **load_kwargs)

        return cls(
            processor_factory=processor_factory,
            model=model,
            device=device,
        )

    @property
    def tokenizer(self) -> Any:
        """Return the checkpoint tokenizer without loading a second processor."""

        with self._lock:
            if self._sessions:
                processor = next(iter(self._sessions.values())).processor
            else:
                if self._idle_processor is None:
                    self._idle_processor = self._new_processor()
                processor = self._idle_processor
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise PerceptionError("MiniCPM-o processor does not expose tokenizer")
        return tokenizer

    def open_session(self, session_id: str) -> None:
        """Allocate independent processor and audio-cache state for a session."""

        session_id = _validate_session_id(session_id)
        with self._lock:
            if session_id in self._sessions:
                raise PerceptionError(
                    f"perception session {session_id!r} already exists"
                )
            processor = self._idle_processor
            self._idle_processor = None
            if processor is None:
                processor = self._new_processor()
            if id(processor) in self._active_processor_ids:
                raise PerceptionError(
                    "processor_factory reused mutable state across active sessions"
                )
            self._configure_streaming(processor)
            state = _SessionState(
                processor=processor,
                token_ids=MiniCPMOSpecialTokens.from_tokenizer(processor.tokenizer),
            )
            self._sessions[session_id] = state
            self._active_processor_ids.add(id(processor))

    def close_session(self, session_id: str) -> None:
        """Release all processor, buffered PCM, and Whisper-cache state."""

        session_id = _validate_session_id(session_id)
        with self._lock:
            state = self._sessions.pop(session_id, None)
            if state is None:
                return
            self._active_processor_ids.discard(id(state.processor))
        with state.lock:
            state.audio_past_key_values = None
            state.audio_buffer = np.empty(0, dtype=np.float32)
            reset = getattr(state.processor, "reset_streaming", None)
            if callable(reset):
                reset()

    def prepare_unit(
        self,
        session_id: str,
        data: Mapping[str, Any],
    ) -> PreparedPerceptionUnit:
        """Preprocess one normalized one-second duplex input unit."""

        state = self._get_session(session_id)
        normalized = normalize_append_data(data)
        with state.lock:
            if state.poisoned_reason is not None:
                raise PerceptionError(
                    "perception session is unusable after an encoder failure; "
                    f"close and reopen it ({state.poisoned_reason})"
                )

            frames = [_decode_image(value) for value in normalized["video_frames"]]
            has_audio = "audio_pcm16_b64" in normalized
            mode: Literal["AUDIO", "VISION", "OMNI"]
            if has_audio and frames:
                mode = "OMNI"
            elif has_audio:
                mode = "AUDIO"
            else:
                mode = "VISION"

            input_ids = [state.token_ids.unit_start]
            spans: list[EmbeddingSpan] = []
            if frames:
                self._append_video(
                    state,
                    frames,
                    normalized["max_slice_nums"],
                    input_ids,
                    spans,
                )

            audio_metadata = None
            audio_commit: tuple[np.ndarray, Any, int] | None = None
            if has_audio:
                waveform = _decode_pcm16(normalized["audio_pcm16_b64"])
                audio_embeddings, audio_metadata, audio_commit = self._encode_audio(
                    state,
                    waveform,
                )
                _append_embedding(
                    input_ids,
                    spans,
                    audio_embeddings,
                    state.token_ids.media_placeholder,
                    "audio",
                )

            unit = PreparedPerceptionUnit(
                input_ids=tuple(input_ids),
                embedding_spans=tuple(spans),
                mode=mode,
                unit_index=state.unit_index,
                video_frame_count=len(frames),
                audio=audio_metadata,
            )
            if audio_commit is not None:
                state.audio_buffer, state.audio_past_key_values, next_chunk = (
                    audio_commit
                )
                state.audio_chunk_index = next_chunk
            state.unit_index += 1
            return unit

    def prepare_reference_audio(
        self,
        session_id: str,
        waveform: np.ndarray,
    ) -> torch.Tensor:
        """Encode the optional system-prompt reference audio without stream KV."""

        state = self._get_session(session_id)
        audio = np.asarray(waveform, dtype=np.float32)
        if audio.ndim != 1 or audio.size == 0:
            raise ValueError("reference audio must be a non-empty mono waveform")
        if audio.size > MAX_REFERENCE_AUDIO_SAMPLES:
            raise ValueError("reference audio exceeds 30 seconds")
        if not np.isfinite(audio).all():
            raise ValueError("reference audio contains non-finite samples")

        with state.lock:
            process = getattr(state.processor, "process_audio", None)
            if not callable(process):
                raise PerceptionError("MiniCPM-o processor lacks process_audio")
            batch = process([audio], sampling_rate=INPUT_SAMPLE_RATE)
            batch = _move_to_device(batch, self._device)

            encode = getattr(self._model, "encode_audio_offline", None)
            if callable(encode):
                embeddings = encode(batch)
            else:
                get_audio_feature = getattr(self._model, "get_audio_feature", None)
                if not callable(get_audio_feature):
                    raise PerceptionError(
                        "MiniCPM-o model must expose encode_audio_offline or "
                        "get_audio_feature"
                    )
                features = _batch_get(batch, "audio_features")
                lengths = _batch_get(batch, "audio_feature_lens")
                embeddings = get_audio_feature(
                    [
                        _MediaItem(
                            feature=[features],
                            model_specific_data={"audio_feature_lens": lengths},
                        )
                    ]
                )
            return _flatten_embeddings(embeddings, "reference audio")

    def _new_processor(self) -> Any:
        processor = self._processor_factory()
        if processor is None:
            raise PerceptionError("processor_factory returned None")
        return processor

    def _get_session(self, session_id: str) -> _SessionState:
        session_id = _validate_session_id(session_id)
        with self._lock:
            state = self._sessions.get(session_id)
        if state is None:
            raise PerceptionError(f"unknown perception session {session_id!r}")
        return state

    def _configure_streaming(self, processor: Any) -> None:
        set_mode = getattr(processor, "set_streaming_mode", None)
        if not callable(set_mode):
            raise PerceptionError(
                "MiniCPM-o checkpoint processor lacks exact streaming support"
            )
        for method_name in (
            "process_audio_streaming",
            "get_streaming_chunk_size",
            "process_image",
            "reset_streaming",
        ):
            if not callable(getattr(processor, method_name, None)):
                raise PerceptionError(
                    f"MiniCPM-o checkpoint processor lacks {method_name}"
                )
        set_mode(
            mode="exact",
            chunk_ms=self._chunk_ms,
            first_chunk_ms=self._first_chunk_ms,
            cnn_redundancy_ms=self._cnn_redundancy_ms,
            enable_sliding_window=True,
            slide_trigger_seconds=30.0,
            slide_stride_seconds=10.0,
        )
        processor.reset_streaming()

    def _append_video(
        self,
        state: _SessionState,
        frames: Sequence[Any],
        max_slice_nums: int,
        input_ids: list[int],
        spans: list[EmbeddingSpan],
    ) -> None:
        processed = state.processor.process_image(
            list(frames),
            max_slice_nums=max_slice_nums,
        )
        processed = _move_to_device(processed, self._device)
        pixel_values = _batch_get(processed, "pixel_values")
        tgt_sizes = _batch_get(processed, "tgt_sizes")
        slices_per_frame = _vision_slice_counts(
            state.processor,
            frames,
            max_slice_nums,
        )
        items = _make_vision_items(
            pixel_values,
            tgt_sizes,
            slices_per_frame,
        )

        get_image_feature = getattr(self._model, "get_image_feature", None)
        if not callable(get_image_feature):
            raise PerceptionError("MiniCPM-o model lacks get_image_feature")
        encoded = get_image_feature(items)
        slice_embeddings = _split_vision_embeddings(
            encoded,
            sum(slices_per_frame),
        )

        offset = 0
        for slice_count in slices_per_frame:
            for slice_index in range(slice_count):
                is_source = slice_index == 0
                input_ids.append(
                    state.token_ids.image_start
                    if is_source
                    else state.token_ids.slice_start
                )
                _append_embedding(
                    input_ids,
                    spans,
                    slice_embeddings[offset],
                    state.token_ids.media_placeholder,
                    "image",
                )
                input_ids.append(
                    state.token_ids.image_end
                    if is_source
                    else state.token_ids.slice_end
                )
                offset += 1

    def _encode_audio(
        self,
        state: _SessionState,
        waveform: np.ndarray,
    ) -> tuple[
        torch.Tensor,
        StreamingAudioMetadata,
        tuple[np.ndarray, Any, int],
    ]:
        candidate = np.concatenate((state.audio_buffer, waveform))
        if state.audio_chunk_index == 0:
            declared_first = int(self._first_chunk_ms * INPUT_SAMPLE_RATE / 1_000)
            if candidate.size < declared_first:
                candidate = np.pad(candidate, (declared_first - candidate.size, 0))

        need_samples = state.processor.get_streaming_chunk_size()
        if type(need_samples) is not int or need_samples <= 0:
            raise PerceptionError("processor returned an invalid streaming chunk size")
        if candidate.size < need_samples:
            raise PerceptionError(
                "streaming processor needs "
                f"{need_samples} samples, got {candidate.size}"
            )
        chunk = candidate[:need_samples]
        remaining = candidate[need_samples:].copy()

        snapshot = _take_streaming_snapshot(state.processor)
        model_called = False
        try:
            batch = state.processor.process_audio_streaming(
                chunk,
                reset=False,
                return_batch_feature=True,
            )
            features = _batch_get(batch, "audio_features")
            if not isinstance(features, torch.Tensor) or features.ndim != 3:
                raise PerceptionError(
                    "streaming processor must return audio_features "
                    "[batch, mel, frames]"
                )
            if features.shape[0] != 1 or features.shape[-1] == 0:
                raise PerceptionError("streaming processor returned empty audio frames")

            prefix_frames = 0 if state.audio_chunk_index == 0 else PREFIX_EXTRA_FRAMES
            _batch_set(batch, "chunk_idx", state.audio_chunk_index)
            _batch_set(batch, "use_extra_context", True)
            _batch_set(batch, "prefix_extra_frames", prefix_frames)
            _batch_set(batch, "suffix_extra_frames", SUFFIX_EXTRA_FRAMES)
            batch = _move_to_device(batch, self._device)

            encode = getattr(self._model, "encode_audio_streaming", None)
            if not callable(encode):
                raise PerceptionError(
                    "MiniCPM-o model lacks session-safe encode_audio_streaming; "
                    "the method must accept and return explicit past_key_values"
                )
            model_called = True
            encoded = encode(
                batch,
                past_key_values=state.audio_past_key_values,
                use_extra_context=True,
                prefix_extra_frames=prefix_frames,
                suffix_extra_frames=SUFFIX_EXTRA_FRAMES,
            )
            embeddings, new_cache = _unpack_streaming_result(encoded)
            flat_embeddings = _flatten_embeddings(embeddings, "streaming audio")
            if new_cache is None:
                raise PerceptionError(
                    "encode_audio_streaming did not return a Whisper KV cache"
                )
        except Exception as exc:
            _restore_streaming_snapshot(state.processor, snapshot)
            if model_called:
                state.poisoned_reason = str(exc) or type(exc).__name__
            if isinstance(exc, PerceptionError):
                raise
            raise PerceptionError(f"streaming audio encoding failed: {exc}") from exc

        info = _batch_get(batch, "streaming_info", default={})
        if not isinstance(info, Mapping):
            info = {}
        emitted_frames = int(info.get("emitted_frames", features.shape[-1]))
        metadata = StreamingAudioMetadata(
            chunk_index=state.audio_chunk_index,
            input_samples=waveform.size,
            consumed_samples=need_samples,
            buffered_samples=remaining.size,
            emitted_frames=emitted_frames,
            prefix_extra_frames=prefix_frames,
            suffix_extra_frames=SUFFIX_EXTRA_FRAMES,
            processor_info=dict(info),
        )
        return (
            flat_embeddings,
            metadata,
            (remaining, new_cache, state.audio_chunk_index + 1),
        )


def _validate_session_id(session_id: str) -> str:
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")
    return session_id.strip()


def _decode_pcm16(encoded: str) -> np.ndarray:
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("audio_pcm16_b64 is not valid base64") from exc
    pcm = np.frombuffer(raw, dtype="<i2")
    if pcm.size != INPUT_SAMPLE_RATE:
        raise ValueError(
            f"audio unit must contain exactly {INPUT_SAMPLE_RATE} PCM16 samples"
        )
    return pcm.astype(np.float32) / 32768.0


def _decode_image(encoded: str) -> Any:
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("video frame is not valid base64") from exc
    try:
        from PIL import Image

        with Image.open(io.BytesIO(raw)) as image:
            return image.convert("RGB").copy()
    except ImportError as exc:
        raise RuntimeError("Pillow is required for MiniCPM-o video input") from exc
    except Exception as exc:
        raise ValueError("video frame is not a decodable image") from exc


def _batch_get(batch: Any, key: str, default: Any = None) -> Any:
    if isinstance(batch, Mapping):
        return batch.get(key, default)
    return getattr(batch, key, default)


def _batch_set(batch: Any, key: str, value: Any) -> None:
    try:
        batch[key] = value
    except (TypeError, AttributeError, KeyError):
        setattr(batch, key, value)


def _move_to_device(value: Any, device: str | torch.device | None) -> Any:
    if device is None:
        return value
    move = getattr(value, "to", None)
    if callable(move):
        moved = move(device)
        return value if moved is None else moved
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _make_vision_items(
    pixel_values: Any,
    tgt_sizes: Any,
    slices_per_frame: Sequence[int],
) -> list[_MediaItem]:
    if not isinstance(pixel_values, Sequence) or isinstance(pixel_values, (str, bytes)):
        raise PerceptionError("process_image returned invalid pixel_values")
    if not isinstance(tgt_sizes, Sequence) and not isinstance(tgt_sizes, torch.Tensor):
        raise PerceptionError("process_image returned invalid tgt_sizes")
    frame_count = len(slices_per_frame)
    if len(pixel_values) == frame_count and len(tgt_sizes) == frame_count:
        pixel_groups = [list(group) for group in pixel_values]
        size_groups = [list(group) for group in tgt_sizes]
    elif len(pixel_values) == 1 and len(tgt_sizes) == 1:
        flat_pixels = list(pixel_values[0])
        flat_sizes = list(tgt_sizes[0])
        expected = sum(slices_per_frame)
        if len(flat_pixels) != expected or len(flat_sizes) != expected:
            raise PerceptionError(
                "flattened image processor output does not match slice metadata"
            )
        pixel_groups = []
        size_groups = []
        offset = 0
        for count in slices_per_frame:
            pixel_groups.append(flat_pixels[offset : offset + count])
            size_groups.append(flat_sizes[offset : offset + count])
            offset += count
    else:
        raise PerceptionError("process_image returned an unsupported batch layout")

    items: list[_MediaItem] = []
    for expected, pixel_list, size_list in zip(
        slices_per_frame,
        pixel_groups,
        size_groups,
    ):
        if expected <= 0 or len(pixel_list) != expected or len(size_list) != expected:
            raise PerceptionError(
                "each video frame must have matching non-empty slices and tgt_sizes"
            )
        items.append(
            _MediaItem(
                feature=pixel_list,
                model_specific_data={"tgt_size": size_list},
            )
        )
    return items


def _vision_slice_counts(
    processor: Any,
    frames: Sequence[Any],
    max_slice_nums: int,
) -> list[int]:
    image_processor = getattr(processor, "image_processor", None)
    get_grid = getattr(image_processor, "get_sliced_grid", None)
    if not callable(get_grid):
        if max_slice_nums == 1:
            return [1] * len(frames)
        raise PerceptionError(
            "MiniCPM-o image processor cannot report HD slice grouping"
        )

    counts: list[int] = []
    for frame in frames:
        size = getattr(frame, "size", None)
        if size is None:
            raise PerceptionError("video frame does not expose an image size")
        grid = get_grid(size, max_slice_nums, nerver_split=False)
        if grid is None:
            counts.append(1)
            continue
        if len(grid) != 2 or int(grid[0]) <= 0 or int(grid[1]) <= 0:
            raise PerceptionError("image processor returned an invalid slice grid")
        counts.append(1 + int(grid[0]) * int(grid[1]))
    return counts


def _split_vision_embeddings(value: Any, expected_slices: int) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []

    def collect(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            if item.ndim == 3:
                tensors.extend(item[index] for index in range(item.shape[0]))
            elif item.ndim == 2:
                tensors.append(item)
            else:
                raise PerceptionError(
                    "vision embeddings must have shape [slices, tokens, hidden]"
                )
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect(child)
        else:
            raise PerceptionError("model returned invalid vision embeddings")

    collect(value)
    if len(tensors) == 1 and expected_slices > 1:
        combined = tensors[0]
        if combined.shape[0] % expected_slices:
            raise PerceptionError(
                "flattened vision embeddings cannot be split by slice"
            )
        tensors = list(combined.chunk(expected_slices, dim=0))
    if len(tensors) != expected_slices:
        raise PerceptionError(
            f"model returned {len(tensors)} vision slices, expected {expected_slices}"
        )
    return [_as_embedding_matrix(tensor, "vision") for tensor in tensors]


def _flatten_embeddings(value: Any, label: str) -> torch.Tensor:
    tensors: list[torch.Tensor] = []

    def collect(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            if item.ndim == 3 and item.shape[0] == 1:
                item = item.squeeze(0)
            tensors.append(_as_embedding_matrix(item, label))
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect(child)
        else:
            raise PerceptionError(f"model returned invalid {label} embeddings")

    collect(value)
    if not tensors:
        raise PerceptionError(f"model returned empty {label} embeddings")
    hidden_sizes = {tensor.shape[1] for tensor in tensors}
    if len(hidden_sizes) != 1:
        raise PerceptionError(f"model returned inconsistent {label} hidden sizes")
    return torch.cat(tensors, dim=0)


def _as_embedding_matrix(value: torch.Tensor, label: str) -> torch.Tensor:
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise PerceptionError(
            f"{label} embeddings must have non-empty shape [tokens, hidden]"
        )
    return value


def _append_embedding(
    input_ids: list[int],
    spans: list[EmbeddingSpan],
    embeddings: torch.Tensor,
    placeholder_id: int,
    modality: Literal["audio", "image"],
) -> None:
    embeddings = _as_embedding_matrix(embeddings, modality)
    start = len(input_ids)
    input_ids.extend([placeholder_id] * embeddings.shape[0])
    spans.append(
        EmbeddingSpan(
            start=start,
            end=len(input_ids),
            embedding=embeddings,
            modality=modality,
        )
    )


def _unpack_streaming_result(value: Any) -> tuple[Any, Any]:
    if isinstance(value, tuple) and len(value) == 2:
        return value
    if isinstance(value, Mapping):
        if "embeddings" in value and "past_key_values" in value:
            return value["embeddings"], value["past_key_values"]
    if hasattr(value, "embeddings") and hasattr(value, "past_key_values"):
        return value.embeddings, value.past_key_values
    raise PerceptionError(
        "encode_audio_streaming must return (embeddings, past_key_values)"
    )


def _take_streaming_snapshot(processor: Any) -> Any:
    snapshot = getattr(processor, "get_streaming_snapshot", None)
    if callable(snapshot):
        return snapshot()
    mel_processor = getattr(processor, "_streaming_mel_processor", None)
    snapshot = getattr(mel_processor, "get_snapshot", None)
    if callable(snapshot):
        return snapshot()
    return None


def _restore_streaming_snapshot(processor: Any, snapshot: Any) -> None:
    if snapshot is None:
        return
    restore = getattr(processor, "restore_streaming_snapshot", None)
    if callable(restore):
        restore(snapshot)
        return
    mel_processor = getattr(processor, "_streaming_mel_processor", None)
    restore = getattr(mel_processor, "restore_snapshot", None)
    if callable(restore):
        restore(snapshot)


__all__ = [
    "CHUNK_MS",
    "CNN_REDUNDANCY_MS",
    "EmbeddingSpan",
    "FIRST_CHUNK_MS",
    "MiniCPMO45Perception",
    "PerceptionError",
    "PreparedPerceptionUnit",
    "StreamingAudioMetadata",
]
