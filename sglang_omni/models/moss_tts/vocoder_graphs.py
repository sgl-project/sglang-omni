# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph replay for batch-of-one MOSS-TTS Delay vocoder decodes."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import torch

from sglang_omni.models.moss_tts.audio_tokenizer import (
    MossAudioTokenizerProjectedTransformer,
    MossAudioTokenizerVocoderDecoder,
)

logger = logging.getLogger(__name__)

MIN_CAPTURE_FREE_BYTES = 2 * 1024**3


@dataclass(kw_only=True)
class CapturedDecodeGraph:
    graph: torch.cuda.CUDAGraph
    static_codes: torch.Tensor
    static_lengths: torch.Tensor
    audio: torch.Tensor
    audio_samples: int


class RowDecode(Protocol):
    def __call__(
        self, codes: torch.Tensor, lengths: torch.Tensor, frames: int
    ) -> torch.Tensor: ...


class MossVocoderSegmentGraphs:
    """Replay exact frame counts for a single unpadded segment."""

    def __init__(
        self, decoder: MossAudioTokenizerVocoderDecoder, *, row_decode: RowDecode
    ) -> None:
        self.decoder = decoder
        self.row_decode = row_decode
        self.graphs: dict[int, CapturedDecodeGraph] = {}
        self.pool: tuple[int, int] | None = None
        # note (Zhuofeng Li): one stream avoids stranding allocator blocks per graph.
        self.capture_stream: torch.cuda.Stream | None = None
        self.retained_cache_tensors: list[torch.Tensor] = []

    def covers(self, codes: torch.Tensor) -> bool:
        captured = self.graphs.get(int(codes.shape[0]))
        return captured is not None and int(captured.static_codes.shape[0]) == int(
            codes.shape[1]
        )

    @torch.inference_mode()
    def replay(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode one [frames, n_vq] row to CPU float32 audio [channels, samples]."""

        captured = self.graphs[int(codes.shape[0])]
        captured.static_codes.copy_(codes.transpose(0, 1).unsqueeze(1))
        captured.graph.replay()
        return captured.audio[0, :, : captured.audio_samples].to(
            device="cpu", dtype=torch.float32
        )

    def capture(
        self, frame_counts: Iterable[int], *, n_vq: int, device: torch.device
    ) -> None:
        """Capture graphs shortest first while the device keeps free headroom."""
        if device.type != "cuda":
            return
        else:
            pass
        for frames in sorted({int(count) for count in frame_counts}):
            free_bytes, _ = torch.cuda.mem_get_info(device)
            if free_bytes < MIN_CAPTURE_FREE_BYTES:
                logger.warning(
                    f"MOSS-TTS segment graph capture stopped at {frames} frames: "
                    f"{free_bytes / 1024**3:.2f} GiB free"
                )
                break
            else:
                pass
            try:
                self.graphs[frames] = self.capture_frames(
                    frames, n_vq=n_vq, device=device
                )
            except (RuntimeError, torch.OutOfMemoryError):
                logger.exception(
                    f"MOSS-TTS segment graph capture failed at {frames} frames; "
                    "using eager decode"
                )
                self.graphs.clear()
                self.pool = None
                self.retained_cache_tensors.clear()
                self.capture_stream = None
                torch.cuda.empty_cache()
                return
            self.retain_cache_tensors()
        logger.info(
            f"Captured {len(self.graphs)} MOSS-TTS segment graphs for frame counts "
            f"{min(self.graphs, default=None)}..{max(self.graphs, default=None)}"
        )

    def retain_cache_tensors(self) -> None:
        """Retain captured position/RoPE storage when longer decodes replace caches."""
        for stage in self.decoder:
            if isinstance(stage, MossAudioTokenizerProjectedTransformer):
                self.retained_cache_tensors.extend(
                    stage.position_ids_cache.items.values()
                )
                rope_cache = stage.transformer.packed_rope_cache
                for tensor in (rope_cache.cos, rope_cache.sin, rope_cache.cos_sin):
                    if tensor is not None:
                        self.retained_cache_tensors.append(tensor)
                    else:
                        pass
            else:
                pass

    @torch.inference_mode()
    def capture_frames(
        self, frames: int, *, n_vq: int, device: torch.device
    ) -> CapturedDecodeGraph:
        static_codes = torch.zeros(n_vq, 1, frames, device=device, dtype=torch.long)
        static_lengths = torch.full((1,), frames, device=device, dtype=torch.int32)
        if self.capture_stream is None:
            self.capture_stream = torch.cuda.Stream(device=device)
        else:
            pass
        stream = self.capture_stream
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            self.row_decode(static_codes, static_lengths, frames)
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        else:
            pass
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph,
            pool=self.pool,
            stream=stream,
            capture_error_mode="thread_local",
        ):
            audio = self.row_decode(static_codes, static_lengths, frames)
        torch.cuda.current_stream(device).wait_stream(stream)
        return CapturedDecodeGraph(
            graph=graph,
            static_codes=static_codes,
            static_lengths=static_lengths,
            audio=audio,
            audio_samples=int(self.decoder.output_lengths([frames])[0]),
        )
