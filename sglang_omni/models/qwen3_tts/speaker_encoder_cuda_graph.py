# SPDX-License-Identifier: Apache-2.0
"""Device graphs for the Qwen3-TTS speaker encoder at bucketed mel lengths."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn

from sglang_omni.models.qwen3_tts.reference_encoder_cuda_graph import (
    ReferenceEncoderGraphStats,
    smallest_bucket,
)
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import ReplayableGraph
from sglang_omni.utils.device import device_guard, supports_device_streams

logger = logging.getLogger(__name__)

SPEAKER_MEL_N_FFT = 1024
SPEAKER_MEL_HOP = 256
SPEAKER_MEL_FMIN = 0
SPEAKER_MEL_FMAX = 12000
SPEAKER_MEL_CLIP = 1e-5
SPEAKER_GRAPH_BATCH_SIZES = (1, 2, 4, 8)


def reflect_index(length: torch.Tensor, width: int, pad: int) -> torch.Tensor:
    """Gather index of a reflect pad of pad frames around the first length frames
    of a width frame buffer."""
    position = torch.arange(-pad, width + pad, device=length.device)
    index = position.abs()
    over = index - (length - 1)
    index = torch.where(over > 0, (length - 1) - over, index)
    return index.clamp(0, width - 1)


def time_delay_block(
    block: torch.nn.Module, x: torch.Tensor, indices: dict[int, torch.Tensor]
) -> torch.Tensor:
    conv = block.conv
    pad = conv.dilation[0] * (conv.kernel_size[0] - 1) // 2
    if pad:
        x = x.index_select(2, indices[pad])
    else:
        pass
    return F.relu(F.conv1d(x, conv.weight, conv.bias, dilation=conv.dilation))


def masked_mean(
    x: torch.Tensor, mask: torch.Tensor, length: torch.Tensor
) -> torch.Tensor:
    """Mean of x (1, channels, width) over its first length frames, accumulated and
    divided in the precision torch.mean uses and rounded to the dtype of x once."""
    accumulate = torch.promote_types(x.dtype, torch.float32)
    return ((x * mask).sum(2, keepdim=True, dtype=accumulate) / length).to(x.dtype)


def encode_bucketed(
    encoder: torch.nn.Module,
    mels: torch.Tensor,
    length: torch.Tensor,
    pads: Iterable[int],
) -> torch.Tensor:
    """Encode equal-length rows exactly for the first length mel frames."""
    width = mels.shape[2]
    indices = {pad: reflect_index(length, width, pad) for pad in pads}
    mask = (torch.arange(width, device=mels.device) < length).to(mels.dtype)[None, None]
    x = time_delay_block(encoder.blocks[0], mels, indices)
    outputs = []
    for layer in encoder.blocks[1:]:
        residual = x
        x = time_delay_block(layer.tdnn1, x, indices)
        first, *rest = torch.chunk(x, layer.res2net_block.scale, dim=1)
        parts = [first]
        for block, part in zip(layer.res2net_block.blocks, rest):
            if len(parts) > 1:
                part = part + parts[-1]
            else:
                pass
            parts.append(time_delay_block(block, part, indices))
        x = time_delay_block(layer.tdnn2, torch.cat(parts, dim=1), indices)
        se = layer.se_block
        mean = masked_mean(x, mask, length)
        x = x * torch.sigmoid(se.conv2(F.relu(se.conv1(mean)))) + residual
        outputs.append(x)
    x = time_delay_block(encoder.mfa, torch.cat(outputs, dim=1), indices)
    asp = encoder.asp
    statistics = (
        asp._compute_statistics
    )  # noqa: leading-underscore  # upstream spelling
    mean, std = statistics(x, mask / length.to(x.dtype))
    attention = torch.cat(
        [
            x,
            mean.unsqueeze(2).expand(-1, -1, width),
            std.unsqueeze(2).expand(-1, -1, width),
        ],
        dim=1,
    )
    attention = asp.conv(torch.tanh(time_delay_block(asp.tdnn, attention, indices)))
    attention = F.softmax(attention.masked_fill(mask == 0, float("-inf")), dim=2)
    mean, std = statistics(x, attention)
    return encoder.fc(torch.cat((mean, std), dim=1).unsqueeze(2)).squeeze(-1)


@dataclass
class CapturedSpeakerEncoderGraph:
    graph: ReplayableGraph
    static_mels: torch.Tensor
    static_frames: torch.Tensor
    static_embeddings: torch.Tensor


class Qwen3TTSSpeakerEncoderCudaGraphRunner:
    """The speaker embedding of a waveform: the checkpoint's mel front end on the CPU,
    then one captured encode per bucket length, eager past the largest bucket."""

    def __init__(self, encoder: torch.nn.Module, *, sample_rate: int) -> None:
        self.encoder = encoder
        self.mel_basis = torch.from_numpy(
            librosa_mel_fn(
                sr=sample_rate,
                n_fft=SPEAKER_MEL_N_FFT,
                n_mels=encoder.blocks[0].conv.in_channels,
                fmin=SPEAKER_MEL_FMIN,
                fmax=SPEAKER_MEL_FMAX,
            )
        ).float()
        self.mel_window = torch.hann_window(
            SPEAKER_MEL_N_FFT, dtype=torch.float32, device="cpu"
        )
        self.pads = frozenset(
            conv.dilation[0] * (conv.kernel_size[0] - 1) // 2
            for conv in encoder.modules()
            if isinstance(conv, torch.nn.Conv1d)
        ) - {0}
        self.lock = threading.Lock()
        self.graphs_by_batch_size: dict[int, dict[int, CapturedSpeakerEncoderGraph]] = {
            batch_size: {} for batch_size in SPEAKER_GRAPH_BATCH_SIZES
        }
        self.bucket_frames: list[int] = []
        self.disable_reason: str | None = None
        self.replays = 0
        self.misses = 0

    def mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Log mel (1, mels, frames) of a CPU waveform (1, samples)."""
        padding = (SPEAKER_MEL_N_FFT - SPEAKER_MEL_HOP) // 2
        spectrum = torch.stft(
            F.pad(waveform, (padding, padding), mode="reflect"),
            SPEAKER_MEL_N_FFT,
            hop_length=SPEAKER_MEL_HOP,
            win_length=SPEAKER_MEL_N_FFT,
            window=self.mel_window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        magnitude = torch.sqrt(torch.view_as_real(spectrum).pow(2).sum(-1) + 1e-9)
        return torch.log(
            torch.clamp(torch.matmul(self.mel_basis, magnitude), min=SPEAKER_MEL_CLIP)
        )

    def capture(self, codec_frame_buckets: Iterable[int], codec_hop: int) -> None:
        """Capture common batch sizes for every mel frame bucket."""
        param = next(self.encoder.parameters())
        buckets = sorted(
            frames * codec_hop // SPEAKER_MEL_HOP for frames in codec_frame_buckets
        )
        self.bucket_frames = buckets
        graph_backend = current_platform.get_device_graph_backend(param.device)
        if not buckets:
            return
        elif not supports_device_streams(param.device) or graph_backend is None:
            self.disable_reason = f"no graph backend on {param.device.type}"
            return
        else:
            pass
        device_module = torch.get_device_module(param.device)
        graphs_by_batch_size: dict[int, dict[int, CapturedSpeakerEncoderGraph]] = {
            batch_size: {} for batch_size in SPEAKER_GRAPH_BATCH_SIZES
        }
        try:
            with device_guard(param.device):
                pool = graph_backend.graph_pool_handle()
                stream = device_module.Stream(device=param.device)
                # note(ratish): largest first so the shared pool is sized once.
                for batch_size in reversed(SPEAKER_GRAPH_BATCH_SIZES):
                    captured_graphs = graphs_by_batch_size[batch_size]
                    for frames in reversed(buckets):
                        static_mels = torch.zeros(
                            (batch_size, self.mel_basis.shape[0], frames),
                            device=param.device,
                            dtype=param.dtype,
                        )
                        static_frames = torch.full(
                            (1,), frames, device=param.device, dtype=torch.long
                        )
                        stream.wait_stream(device_module.current_stream(param.device))
                        with torch.inference_mode(), device_module.stream(stream):
                            for _ in range(2):
                                encode_bucketed(
                                    self.encoder,
                                    static_mels,
                                    static_frames,
                                    self.pads,
                                )
                        with (
                            torch.inference_mode(),
                            graph_backend.capture(
                                pool=pool,
                                stream=stream,
                                thread_local_errors=True,
                            ) as graph,
                        ):
                            static_embeddings = encode_bucketed(
                                self.encoder,
                                static_mels,
                                static_frames,
                                self.pads,
                            )
                        stream.synchronize()
                        captured_graphs[frames] = CapturedSpeakerEncoderGraph(
                            graph=graph,
                            static_mels=static_mels,
                            static_frames=static_frames,
                            static_embeddings=static_embeddings,
                        )
        except Exception as exc:
            self.disable_reason = f"capture_failed: {type(exc).__name__}: {exc}"
            logger.warning(
                "Qwen3-TTS speaker encoder graph capture disabled the runner: "
                f"{self.disable_reason}",
                exc_info=True,
            )
            return
        self.graphs_by_batch_size = graphs_by_batch_size
        logger.info(
            f"Qwen3-TTS speaker encoder graphs captured for batch sizes "
            f"{SPEAKER_GRAPH_BATCH_SIZES} and {buckets} frames"
        )

    def replay(
        self,
        captured: CapturedSpeakerEncoderGraph,
        mels: torch.Tensor,
        frames: int,
    ) -> torch.Tensor:
        """Replay one captured speaker graph and return owned embeddings."""
        batch_size = mels.shape[0]
        assert batch_size <= captured.static_mels.shape[0]
        with self.lock:
            captured.static_mels[:batch_size, :, :frames].copy_(mels)
            captured.static_mels[batch_size:, :, :frames].zero_()
            captured.static_frames.fill_(frames)
            captured.graph.replay()
            self.replays += 1
            return captured.static_embeddings[:batch_size].clone()

    def stats(self) -> ReferenceEncoderGraphStats:
        one_clip_graphs = self.graphs_by_batch_size[1]
        return {
            "enabled": bool(one_clip_graphs),
            "disable_reason": self.disable_reason,
            "bucket_frames": list(self.bucket_frames),
            "captured": sorted(one_clip_graphs),
            "replays": self.replays,
            "misses": self.misses,
        }

    def embed(self, waveform: np.ndarray) -> torch.Tensor:
        """Speaker embedding (enc_dim,) of a waveform at the encoder's sample rate."""
        mels = self.mel(torch.from_numpy(waveform).unsqueeze(0))
        frames = mels.shape[2]
        assert frames > max(self.pads)
        param = next(self.encoder.parameters())
        mels = mels.to(param.device).to(param.dtype)
        one_clip_graphs = self.graphs_by_batch_size[1]
        bucket = smallest_bucket(frames, one_clip_graphs)
        if bucket is None:
            # note (Haoling Pu): preprocessing threads embed concurrently, and
            # model_info reports this count.
            with self.lock:
                self.misses += 1
            return self.encoder(mels.transpose(1, 2))[0]
        else:
            pass
        captured = one_clip_graphs[bucket]
        return self.replay(captured, mels, frames)[0]

    def embed_batch(self, waveforms: list[np.ndarray]) -> list[torch.Tensor]:
        """Speaker embeddings in input order, batching only equal-length waveforms."""
        groups: dict[int, list[tuple[int, np.ndarray]]] = {}
        for index, waveform in enumerate(waveforms):
            groups.setdefault(len(waveform), []).append((index, waveform))

        embeddings: dict[int, torch.Tensor] = {}
        param = next(self.encoder.parameters())
        for group in groups.values():
            if len(group) == 1:
                index, waveform = group[0]
                embeddings[index] = self.embed(waveform)
                continue
            else:
                pass
            waveform_batch = torch.stack(
                [torch.from_numpy(waveform) for _, waveform in group]
            )
            mels = self.mel(waveform_batch)
            assert mels.shape[2] > max(self.pads)
            mels = mels.to(device=param.device, dtype=param.dtype)
            graph_batch_size = smallest_bucket(len(group), self.graphs_by_batch_size)
            batch_graphs = (
                self.graphs_by_batch_size[graph_batch_size]
                if graph_batch_size is not None
                else {}
            )
            bucket = smallest_bucket(mels.shape[2], batch_graphs)
            if bucket is not None:
                group_embeddings = self.replay(
                    batch_graphs[bucket], mels, mels.shape[2]
                )
            else:
                group_embeddings = self.encoder(mels.transpose(1, 2))
            for (index, _), embedding in zip(group, group_embeddings, strict=True):
                embeddings[index] = embedding

        return [embeddings[index] for index in range(len(waveforms))]
