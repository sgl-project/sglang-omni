# SPDX-License-Identifier: Apache-2.0
"""dots.tts AudioVAE adapters for Omni's shared vocoder schedulers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.dots_tts.codec import DotsAudioCodec
from sglang_omni.models.dots_tts.payload_types import DotsTTSState, load_dots_tts_state
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload

_LENGTH_BUCKET_FRAMES = 32
logger = logging.getLogger(__name__)


class DotsTTSBatchVocoder(BatchVocoderBase):
    def __init__(self, codec: DotsAudioCodec) -> None:
        self.codec = codec
        self._logged_batch = False

    def prepare_item(self, payload: StagePayload) -> tuple[DotsTTSState, torch.Tensor]:
        state = load_dots_tts_state(payload)
        if state.generated_latents is None:
            raise RuntimeError("dots.tts vocoder received no generated latents")
        return state, state.generated_latents

    async def decode_batch(
        self, items: list[tuple[DotsTTSState, torch.Tensor]]
    ) -> list[tuple[torch.Tensor, int]]:
        outputs: list[tuple[torch.Tensor, int] | None] = [None] * len(items)
        buckets: dict[int, list[tuple[int, torch.Tensor]]] = {}
        for index, (_state, latents) in enumerate(items):
            self._validate_latents(latents)
            frames = int(latents.shape[1])
            bucket = (frames + _LENGTH_BUCKET_FRAMES - 1) // _LENGTH_BUCKET_FRAMES
            buckets.setdefault(bucket, []).append((index, latents))

        bucket_sizes = [len(bucket_items) for bucket_items in buckets.values()]
        if not self._logged_batch and max(bucket_sizes, default=0) > 1:
            logger.info(
                "dots.tts AudioVAE batched decode is active: requests=%d "
                "bucket_sizes=%s",
                len(items),
                bucket_sizes,
            )
            self._logged_batch = True

        with self.codec.lock:
            for bucket_items in buckets.values():
                max_frames = max(int(latents.shape[1]) for _, latents in bucket_items)
                padded = bucket_items[0][1].new_zeros(
                    len(bucket_items),
                    max_frames,
                    self.codec.latent_dim,
                    device=self.codec.device,
                )
                for row, (_index, latents) in enumerate(bucket_items):
                    frames = int(latents.shape[1])
                    padded[row, :frames].copy_(latents[0].to(self.codec.device))

                waveform_batch = self.codec.inference.decode_latents(padded)
                if waveform_batch.shape[0] != len(bucket_items):
                    raise RuntimeError(
                        "dots.tts AudioVAE returned an unexpected batch size: "
                        f"expected {len(bucket_items)}, got {waveform_batch.shape[0]}"
                    )
                for row, (index, latents) in enumerate(bucket_items):
                    valid_samples = int(latents.shape[1]) * self.codec.hop_size
                    outputs[index] = (
                        waveform_batch[row : row + 1, ..., :valid_samples],
                        self.codec.sample_rate,
                    )

        if any(output is None for output in outputs):
            raise RuntimeError("dots.tts AudioVAE did not produce every batch result")
        return [output for output in outputs if output is not None]

    def _validate_latents(self, latents: torch.Tensor) -> None:
        if latents.ndim != 3 or latents.shape[0] != 1:
            raise ValueError(
                "dots.tts latents must have shape [1, frames, latent_dim], "
                f"got {tuple(latents.shape)}"
            )
        if latents.shape[1] == 0:
            raise ValueError("dots.tts latents must contain at least one frame")
        if latents.shape[2] != self.codec.latent_dim:
            raise ValueError(
                f"dots.tts latent_dim must be {self.codec.latent_dim}, "
                f"got {latents.shape[2]}"
            )

    def store_result(
        self,
        payload: StagePayload,
        state: DotsTTSState,
        wav: torch.Tensor,
        sample_rate: int,
    ) -> StagePayload:
        payload.data = audio_waveform_payload(
            wav,
            sample_rate=sample_rate,
            modality="audio",
            source_hint="dots.tts",
        )
        usage = build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload

    async def decode_payload(self, payload: StagePayload) -> StagePayload:
        state, latents = self.prepare_item(payload)
        [(wav, sample_rate)] = await self.decode_batch([(state, latents)])
        return self.store_result(payload, state, wav, sample_rate)


@dataclass
class _DotsStreamState:
    codec_state: Any
    pending: list[torch.Tensor] = field(default_factory=list)
    received_patches: int = 0


@dataclass
class _CapturedVocoderGraph:
    graph: torch.cuda.CUDAGraph
    latents: torch.Tensor
    hidden_h: torch.Tensor
    hidden_c: torch.Tensor
    window: torch.Tensor
    valid_frames: torch.Tensor
    output: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class _DotsVocoderCudaGraphs:
    """Native exact-shape graphs that coexist with the backbone CUDA Graph."""

    def __init__(self, codec: DotsAudioCodec, *, chunk_size: int) -> None:
        self._codec = codec
        self._chunk_size = int(chunk_size)
        self._graphs: dict[int, _CapturedVocoderGraph] = {}
        self._replays = 0
        self._fallbacks = 0
        try:
            self._capture(
                tuple(range(codec.patch_size, chunk_size + 1, codec.patch_size))
            )
        except Exception:
            logger.warning(
                "dots.tts vocoder CUDA graph setup failed; using eager fallback",
                exc_info=True,
            )

    @torch.no_grad()
    def _capture(self, frame_counts: tuple[int, ...]) -> None:
        device = torch.device(self._codec.device)
        if device.type != "cuda":
            return
        inference = self._codec.inference
        with torch.cuda.device(device):
            current_stream = torch.cuda.current_stream(device)
            capture_stream = torch.cuda.Stream(device=device)
            capture_stream.wait_stream(current_stream)
            graph_pool = torch.cuda.graph_pool_handle()
            for frames in sorted(frame_counts, reverse=True):
                state = inference.init_stream_state(
                    batch_size=1, chunk_size=self._chunk_size
                )
                hidden_h, hidden_c = state.lstm_hidden
                latents = hidden_h.new_zeros(1, self._codec.latent_dim, frames)
                window = state.decoder.window
                valid_frames = window.new_zeros((), dtype=torch.int64)
                run = lambda: inference._compiled_stream_step(
                    latents, hidden_h, hidden_c, window, valid_frames
                )
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.stream(capture_stream):
                        for _ in range(2):
                            run()
                    capture_stream.synchronize()
                    with torch.cuda.graph(
                        graph,
                        pool=graph_pool,
                        stream=capture_stream,
                        capture_error_mode="thread_local",
                    ):
                        output = run()
                    self._graphs[frames] = _CapturedVocoderGraph(
                        graph,
                        latents,
                        hidden_h,
                        hidden_c,
                        window,
                        valid_frames,
                        output,
                    )
                except Exception as exc:
                    graph.reset()
                    logger.warning(
                        "dots.tts vocoder CUDA graph capture failed for frames=%d: "
                        "%s; using eager fallback",
                        frames,
                        exc,
                    )
            current_stream.wait_stream(capture_stream)
            torch.cuda.synchronize(device)
        logger.info(
            "dots.tts streaming vocoder CUDA graphs: frames=%s",
            sorted(self._graphs),
        )

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, state: Any) -> torch.Tensor | None:
        captured = self._graphs.get(int(latents.shape[1]))
        if captured is None or captured.window.shape != state.decoder.window.shape:
            self._fallbacks += 1
            if self._fallbacks == 1 or (self._replays + self._fallbacks) % 1024 == 0:
                self._log_hit_rate()
            return None
        self._replays += 1
        if self._replays == 1 or (self._replays + self._fallbacks) % 1024 == 0:
            self._log_hit_rate()
        captured.latents.copy_(latents.transpose(1, 2))
        captured.hidden_h.copy_(state.lstm_hidden[0])
        captured.hidden_c.copy_(state.lstm_hidden[1])
        captured.window.copy_(state.decoder.window)
        captured.valid_frames.fill_(
            min(state.decoder.total_frames, state.decoder.window.size(-1))
        )
        captured.graph.replay()
        audio_window, hidden_h, hidden_c, window = captured.output
        state.lstm_hidden = (hidden_h.clone(), hidden_c.clone())
        state.decoder.window = window.clone()
        state.decoder.total_frames += int(latents.shape[1])
        return self._codec.inference._slice_stream_audio_window(
            audio_window, state, final=False
        ).clone()

    def __len__(self) -> int:
        return len(self._graphs)

    def _log_hit_rate(self) -> None:
        calls = self._replays + self._fallbacks
        logger.info(
            "dots.tts vocoder CUDA graph hit rate: %.1f%% (%d/%d)",
            100.0 * self._replays / calls,
            self._replays,
            calls,
        )


class DotsTTSStreamingVocoder(StreamingVocoderBase[_DotsStreamState, None]):
    def __init__(
        self,
        codec: DotsAudioCodec,
        *,
        optimize: bool,
        enable_streaming_audio_vae_cuda_graph: bool = False,
        merge_steps: int = 4,
        max_batch_size: int = 4,
        max_batch_wait_ms: int = 2,
    ) -> None:
        if merge_steps < 1:
            raise ValueError("dots.tts vocoder merge_steps must be positive")
        if max_batch_size < 1:
            raise ValueError("dots.tts vocoder max_batch_size must be positive")
        if max_batch_wait_ms < 0:
            raise ValueError("dots.tts vocoder max_batch_wait_ms must be non-negative")
        self.codec = codec
        self.optimize = bool(optimize)
        self.merge_steps = int(merge_steps) if optimize else 1
        self._batch_vocoder = DotsTTSBatchVocoder(codec)
        self._cuda_graphs = (
            _DotsVocoderCudaGraphs(
                codec, chunk_size=codec.patch_size * self.merge_steps
            )
            if self.optimize and enable_streaming_audio_vae_cuda_graph
            else None
        )
        super().__init__(
            self._batch_vocoder.decode_payload,
            batch_compute_fn=self._batch_vocoder.decode_payloads,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            sample_rate=codec.sample_rate,
            stream_source_hint="dots.tts",
            stream_input_modality="audio_latents",
        )

    def validate_non_streaming_payload(self, payload: StagePayload) -> None:
        _state, latents = self._batch_vocoder.prepare_item(payload)
        self._batch_vocoder._validate_latents(latents)

    def create_stream_state(self, request_id: str) -> _DotsStreamState:
        del request_id
        with self.codec.lock:
            codec_state = self.codec.inference.init_stream_state(
                batch_size=1,
                chunk_size=self.codec.patch_size * self.merge_steps,
            )
        return _DotsStreamState(codec_state=codec_state)

    def validate_chunk(
        self,
        request_id: str,
        state: _DotsStreamState,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        del request_id, state
        if codes.ndim != 3 or codes.shape[0] != 1:
            raise ValueError(
                "dots.tts latent chunks must have shape [1, frames, latent_dim]"
            )
        if codes.shape[-1] != self.codec.latent_dim:
            raise ValueError(
                f"dots.tts latent_dim must be {self.codec.latent_dim}, "
                f"got {codes.shape[-1]}"
            )
        return codes.to(self.codec.device)

    def ingest(
        self, request_id: str, state: _DotsStreamState, codes: torch.Tensor
    ) -> None:
        del request_id
        state.pending.append(codes)
        state.received_patches += 1

    def should_decode(self, state: _DotsStreamState, *, is_final: bool) -> bool:
        if is_final:
            return bool(state.pending)
        if state.received_patches <= 2:
            return bool(state.pending)
        return len(state.pending) >= self.merge_steps

    def decode_delta(
        self, request_id: str, state: _DotsStreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        del request_id
        chunks: list[torch.Tensor] = []
        if state.pending:
            take = (
                len(state.pending)
                if is_final
                else (1 if state.received_patches <= 2 else self.merge_steps)
            )
            patches, state.pending = state.pending[:take], state.pending[take:]
            # note (db-ol): compiled stream step cudagraph trees corrupt the
            # backbone decode graph replay in this process, see issue 1392.
            # Native graphs above do not use Inductor allocator checkpointing.
            with self.codec.lock:
                latents = torch.cat(patches, dim=1)
                chunk = (
                    self._cuda_graphs.decode(latents, state.codec_state)
                    if self._cuda_graphs is not None
                    else None
                )
                if chunk is None:
                    chunk = self.codec.inference.stream_step(
                        latents,
                        stream_state=state.codec_state,
                        optimize=self.optimize,
                        use_compiled=False,
                    )
            if chunk.numel():
                chunks.append(chunk)
        if is_final:
            with self.codec.lock:
                tail = self.codec.inference.flush(state.codec_state)
            if tail.numel():
                chunks.append(tail)
        if not chunks:
            return None
        return torch.cat(chunks, dim=-1)

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: _DotsStreamState
    ) -> dict[str, Any]:
        del request_id, state
        tts_state = load_dots_tts_state(payload)
        result: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": self.codec.sample_rate,
        }
        usage = build_usage(tts_state)
        if usage is not None:
            result["usage"] = usage
        return result

    def fallback_full_decode(
        self, request_id: str, payload: StagePayload, state: _DotsStreamState
    ) -> torch.Tensor | None:
        del request_id, state
        tts_state = load_dots_tts_state(payload)
        if tts_state.generated_latents is None:
            return None
        with self.codec.lock:
            return self.codec.inference.decode_latents(
                tts_state.generated_latents.to(self.codec.device)
            )

    @property
    def cuda_graph_count(self) -> int:
        return len(self._cuda_graphs) if self._cuda_graphs is not None else 0


__all__ = ["DotsTTSBatchVocoder", "DotsTTSStreamingVocoder"]
