# SPDX-License-Identifier: Apache-2.0
"""Drama 3 reference audio to codec codes, with only device work under the lock."""

from __future__ import annotations

import bisect
import logging
import math
import time

import torch
import torchaudio
from dac.nn.quantize import VectorQuantize

from sglang_omni.models.fishaudio_drama3.codec_encoder import CodecEncoder
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac.modded_dac import DAC
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac.rvq import (
    DownsampleResidualVectorQuantize,
)
from sglang_omni.models.fishaudio_s2_pro.stages import FishReferenceEncodeHook
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import ReplayableGraph

logger = logging.getLogger(__name__)

# A new shape's first runs compile and specialize its kernels, so each shape runs
# twice before its capture records the final ones.
GRAPH_WARMUP_RUNS = 2


def cast_autocast_weights(module: torch.nn.Module, dtype: torch.dtype) -> None:
    """The conv and linear weights, which autocast with cache_enabled=False casts on
    every call, cast once to the same values."""
    for layer in module.modules():
        if isinstance(layer, (torch.nn.Conv1d, torch.nn.Linear)):
            for parameter in layer.parameters(recurse=False):
                parameter.data = parameter.data.to(dtype)
        else:
            pass


def vector_quantized(
    vector_quantizer: VectorQuantize, residual: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The codes and output of VectorQuantize.forward, without its training losses."""
    projected = vector_quantizer.in_proj(residual)
    nearest, codes = vector_quantizer.decode_latents(projected)
    # The forward's straight-through sum rounds differently from nearest, and the next
    # codebook quantizes what is left after it.
    return codes, vector_quantizer.out_proj(projected + (nearest - projected))


def reference_codes_of(
    quantizer: DownsampleResidualVectorQuantize, latent: torch.Tensor
) -> torch.Tensor:
    """quantizer.encode_codes(latent) by the same operations, minus the losses and the
    projection after the last codebook, which no code reads."""
    hidden_states = quantizer.pre_module(quantizer.downsample(latent))
    (semantic_quantizer,) = quantizer.semantic_quantizer.quantizers
    semantic_codes, semantic_latent = vector_quantized(
        semantic_quantizer, hidden_states
    )
    residual = hidden_states - semantic_latent
    codes = [semantic_codes]
    *leading_quantizers, last_quantizer = quantizer.quantizer.quantizers
    for vector_quantizer in leading_quantizers:
        quantizer_codes, quantized_latent = vector_quantized(vector_quantizer, residual)
        codes.append(quantizer_codes)
        residual = residual - quantized_latent
    codes.append(last_quantizer.decode_latents(last_quantizer.in_proj(residual))[1])
    return torch.stack(codes, dim=1)


class Drama3ReferenceEncodeHook(FishReferenceEncodeHook):
    """The Fish reference encode whose lock holds no host sync on an accelerator, and
    whose references up to graph_max_seconds replay a device graph."""

    def __init__(
        self,
        *,
        codec: DAC,
        checkpoint_id: str,
        device: str,
        autocast_dtype: torch.dtype | None,
        max_reference_audio_seconds: float,
        graph_max_seconds: float,
    ) -> None:
        super().__init__(
            codec=codec,
            checkpoint_id=checkpoint_id,
            device=device,
            autocast_dtype=autocast_dtype,
            max_reference_audio_seconds=max_reference_audio_seconds,
        )
        self.frame_length = int(codec.frame_length)
        # Built on the first encode: CPU hooks and test stand-in codecs never get one.
        self.codec_encoder: CodecEncoder | None = None
        self.graphs: dict[int, ReplayableGraph] = {}
        self.graph_audio: dict[int, torch.Tensor] = {}
        self.graph_codes: dict[int, torch.Tensor] = {}
        if self.device.type == "cpu" or graph_max_seconds == 0:
            self.graph_frame_counts: tuple[int, ...] = ()
        else:
            # The encoder is causal, so zeros padded after a reference's last frame
            # feed none of its frames; one graph per second of audio serves all lengths.
            frames_per_second = math.ceil(codec.sample_rate / self.frame_length)
            max_frame_count = math.ceil(
                min(graph_max_seconds, max_reference_audio_seconds)
                * codec.sample_rate
                / self.frame_length
            )
            self.graph_frame_counts = tuple(
                range(
                    frames_per_second,
                    max_frame_count + frames_per_second,
                    frames_per_second,
                )
            )
            self.capture_graphs()

    def capture_graphs(self) -> None:
        graph_backend = current_platform.get_device_graph_backend(self.device)
        if graph_backend is None:
            raise RuntimeError(
                f"no device graph backend for {self.device}; set "
                f"stages.preprocessing.factory.reference_encode_graph_max_seconds to 0 "
                f"to encode references eagerly"
            )
        else:
            pass
        started_seconds = time.perf_counter()
        device_module = torch.get_device_module(self.device)
        current_stream = device_module.current_stream(self.device)
        capture_stream = device_module.Stream(device=self.device)
        capture_stream.wait_stream(current_stream)
        with device_module.device(self.device):
            with device_module.stream(capture_stream):
                for frame_count in reversed(self.graph_frame_counts):
                    self.graph_audio[frame_count] = torch.zeros(
                        (1, 1, frame_count * self.frame_length), device=self.device
                    )
                    for _ in range(GRAPH_WARMUP_RUNS):
                        self.encode_padded(self.graph_audio[frame_count])
            capture_stream.synchronize()
            graph_pool = graph_backend.graph_pool_handle()
            for frame_count in reversed(self.graph_frame_counts):
                with graph_backend.capture(
                    pool=graph_pool, stream=capture_stream, thread_local_errors=True
                ) as graph:
                    self.graph_codes[frame_count] = self.encode_padded(
                        self.graph_audio[frame_count]
                    )
                self.graphs[frame_count] = graph
        current_stream.wait_stream(capture_stream)
        device_module.synchronize(self.device)
        # The warm runs' blocks belong to the capture stream, where nothing reuses them.
        device_module.empty_cache()
        logger.info(
            f"Drama 3 reference encode: {len(self.graphs)} graphs up to "
            f"{self.graph_frame_counts[-1]} frames captured in "
            f"{time.perf_counter() - started_seconds:.1f} s"
        )

    def prepare_codec(self) -> None:
        """Before the first encode: the weights autocast would cast on every call are
        cast once, and the encoder gets its fused walker."""
        quantizer = self.codec.quantizer
        if self.autocast_dtype is not None:
            for module in (
                self.codec.encoder,
                quantizer.downsample,
                quantizer.pre_module,
                quantizer.semantic_quantizer,
                quantizer.quantizer,
            ):
                cast_autocast_weights(module, self.autocast_dtype)
        else:
            pass
        self.codec_encoder = CodecEncoder(self.codec.encoder, self.autocast_dtype)

    def encode_padded(self, audio: torch.Tensor) -> torch.Tensor:
        """[1, codebooks, frames] codes of [1, 1, samples] device audio that is a whole
        number of frames long."""
        if self.codec_encoder is None:
            self.prepare_codec()
        else:
            pass
        with (
            torch.no_grad(),
            torch.autocast(
                self.device.type,
                dtype=self.autocast_dtype,
                enabled=self.autocast_dtype is not None,
                cache_enabled=False,
            ),
        ):
            return reference_codes_of(self.codec.quantizer, self.codec_encoder(audio))

    def encode_reference_waveform(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        if self.device.type == "cpu":
            return super().encode_reference_waveform(audio, sr)
        else:
            pass
        return self.encode_waveform(self.reference_waveform(audio, sr))

    def reference_waveform(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """The [1, samples] CPU waveform the codec encodes: mono, at the codec's rate,
        within the length limit, and clamped to [-1, 1] under autocast. Mirrors
        FishReferenceEncodeHook.encode_reference_waveform up to the codec call."""
        if (
            self.max_reference_audio_seconds is not None
            and audio.shape[-1] > self.max_reference_audio_seconds * sr
        ):
            raise ValueError(
                f"reference audio lasts {audio.shape[-1] / sr:.1f} s, longer than "
                f"the {self.max_reference_audio_seconds:g} s limit; shorten it or "
                f"raise max_reference_audio_seconds"
            )
        else:
            pass
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        else:
            pass
        audio = torchaudio.functional.resample(audio, sr, self.codec.sample_rate)
        audios = audio.squeeze(0).unsqueeze(0)
        if self.autocast_dtype is not None:
            audios = audios.clamp(-1.0, 1.0)
        else:
            pass
        return audios

    def encode_waveform(self, audios: torch.Tensor) -> torch.Tensor:
        """[codebooks, frames] CPU codes of a reference_waveform result."""
        sample_count = int(audios.shape[-1])
        if sample_count == 0:
            raise ValueError("reference audio is empty")
        else:
            pass
        frame_count = math.ceil(sample_count / self.frame_length)
        graph_index = bisect.bisect_left(self.graph_frame_counts, frame_count)
        is_graphed = graph_index < len(self.graph_frame_counts)
        padded_frame_count = (
            self.graph_frame_counts[graph_index] if is_graphed else frame_count
        )
        host_audio = torch.zeros(
            (1, 1, padded_frame_count * self.frame_length),
            dtype=torch.float32,
            pin_memory=True,
        )
        host_audio[0, :, :sample_count] = audios
        device_module = torch.get_device_module(self.device)
        encoded = device_module.Event()
        # A host sync under the lock would queue every other request's encode behind
        # this one's device work, so the event is waited on after the lock.
        with self.encode_lock:
            if is_graphed:
                self.graph_audio[padded_frame_count].copy_(
                    host_audio, non_blocking=True
                )
                self.graphs[padded_frame_count].replay()
                device_codes = self.graph_codes[padded_frame_count]
            else:
                device_codes = self.encode_padded(
                    host_audio.to(self.device, non_blocking=True)
                )
            host_codes = torch.empty(
                (device_codes.shape[1], frame_count),
                dtype=device_codes.dtype,
                pin_memory=True,
            )
            host_codes.copy_(device_codes[0, :, :frame_count], non_blocking=True)
            encoded.record(device_module.current_stream(self.device))
        encoded.synchronize()
        return host_codes
