# SPDX-License-Identifier: Apache-2.0
"""Ming 3 TTS vocoder stage: AudioVAE decode with the decoder transformer as CUDA graphs."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import Qwen2Model

from sglang_omni.models.ming3_tts.checkpoint import (
    SpeechConfig,
    is_dense_checkpoint,
    load_audio_vae,
    load_checkpoint_config,
)
from sglang_omni.models.ming3_tts.streaming_vocoder import Ming3StreamingVocoder
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.device import resolve_concrete_device

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class DecoderGraph:
    graph: torch.cuda.CUDAGraph
    frames: torch.Tensor
    hidden_states: torch.Tensor


class DecoderGraphs:
    """One captured decoder-transformer replay per upsampled frame count.

    A replay runs the eager kernels on tensors laid out like eager decode's,
    the transposed upsampler output in and the transformer's own output
    strides out, so the vocoder output matches eager decode bit for bit.
    Replays never overlap, so their intermediates share one memory pool.
    """

    def __init__(
        self,
        transformer: Qwen2Model,
        frame_counts: list[int],
        hidden_size: int,
        device: torch.device,
    ) -> None:
        self.transformer = transformer
        self.graphs: dict[int, DecoderGraph] = {}
        if not frame_counts:
            return
        else:
            pass
        started = time.perf_counter()
        memory_pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream(device=device)
        frame_storage = torch.zeros(
            hidden_size * max(frame_counts), device=device, dtype=torch.bfloat16
        )
        hidden_storage = torch.empty_like(frame_storage)
        with torch.inference_mode():
            for frame_count in sorted(set(frame_counts), reverse=True):
                frames = frame_storage.as_strided(
                    (1, frame_count, hidden_size),
                    (hidden_size * frame_count, 1, frame_count),
                )
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    eager_states = transformer(
                        inputs_embeds=frames, use_cache=False
                    ).last_hidden_state
                torch.cuda.current_stream(device).wait_stream(stream)
                hidden_states = hidden_storage.as_strided(
                    eager_states.shape, eager_states.stride()
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    graph,
                    pool=memory_pool,
                    stream=stream,
                    capture_error_mode="thread_local",
                ):
                    hidden_states.copy_(
                        transformer(
                            inputs_embeds=frames, use_cache=False
                        ).last_hidden_state
                    )
                self.graphs[frame_count] = DecoderGraph(
                    graph=graph, frames=frames, hidden_states=hidden_states
                )
        buffer_bytes = (
            frame_storage.numel() + hidden_storage.numel()
        ) * frame_storage.element_size()
        logger.info(
            f"Ming 3 TTS vocoder graph setup: {len(self.graphs)} graphs, {time.perf_counter() - started:.3f}s, {buffer_bytes} shared buffer bytes"
        )

    def run(self, frames: torch.Tensor) -> torch.Tensor:
        """Return the transformer output; an uncaptured frame count runs eagerly."""
        captured = self.graphs.get(frames.shape[1])
        if captured is None:
            return self.transformer(
                inputs_embeds=frames, use_cache=False
            ).last_hidden_state
        else:
            captured.frames.copy_(frames)
            captured.graph.replay()
            return captured.hidden_states


def create_vocoder_executor(
    model_path: str,
    *,
    graph_latent_steps: int = 0,
    stream_chunk_steps: int = 8,
    device: str | None = None,
    gpu_id: int | None = None,
) -> Ming3StreamingVocoder | SimpleScheduler:
    resolved_device = resolve_concrete_device(device, gpu_id)
    checkpoint = Path(resolve_checkpoint(model_path))
    configuration = load_checkpoint_config(checkpoint)
    if not is_dense_checkpoint(configuration):
        graph_latent_steps = min(
            graph_latent_steps,
            SpeechConfig.model_validate(
                configuration["tts_config"]
            ).maximum_decode_steps,
        )
    else:
        pass
    vae, sample_rate = load_audio_vae(checkpoint, resolved_device, component="decoder")
    decoder = vae.decoder
    frames_per_latent = decoder.upsampling.scale_factor
    # note (rsi): every length up to graph_latent_steps is captured before
    # serving, so no request pays a capture or a new cuDNN attention plan.
    decoder_graphs = DecoderGraphs(
        decoder.decoder,
        (
            [
                latent_steps * frames_per_latent
                for latent_steps in range(1, graph_latent_steps + 1)
            ]
            if resolved_device.type == "cuda"
            else []
        ),
        decoder.decoder.config.hidden_size,
        resolved_device,
    )

    @torch.inference_mode()
    def decode(payload: StagePayload) -> StagePayload:
        latents = payload.data["latents"].to(resolved_device, dtype=torch.bfloat16)
        # note (rsi): the non-streaming AudioVAE.decode, with its transformer
        # replayed; upsampling and the ISTFT head stay exact-length eager ops.
        frames = decoder.upsampling.upsampler(
            decoder.fc1(latents.reshape(1, -1, latents.shape[-1])).transpose(1, 2)
        ).transpose(1, 2)
        waveform, _, _, _ = decoder.head(
            decoder_graphs.run(frames),
            streaming=False,
            audio_buffer=None,
            window_buffer=None,
            last_chunk=False,
        )
        usage = payload.data["usage"]
        finish_reason = payload.data["finish_reason"]
        payload.data = audio_waveform_payload(
            waveform.float().cpu(),
            sample_rate=sample_rate,
            modality="audio",
            source_hint="Ming-3.0-TTS",
        )
        payload.data["usage"] = usage
        payload.data["finish_reason"] = finish_reason
        return payload

    if resolved_device.type == "cpu":
        return SimpleScheduler(decode)
    else:
        return Ming3StreamingVocoder(
            decode,
            vae=vae,
            sample_rate=sample_rate,
            stream_chunk_steps=stream_chunk_steps,
        )
