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


def reflect_index(length: torch.Tensor, width: int, pad: int) -> torch.Tensor:
    """Gather index of a reflect pad of pad frames around the first length frames
    of a width frame buffer."""
    position = torch.arange(-pad, width + pad, device=length.device)
    index = position.abs()
    over = index - (length - 1)
    index = torch.where(over > 0, (length - 1) - over, index)
    return index.clamp(0, width - 1)


def log_mel(
    waveform: torch.Tensor,
    window: torch.Tensor,
    mel_basis: torch.Tensor,
    padded_frames: int | None = None,
) -> torch.Tensor:
    """Log mel (1, mels, frames) of a waveform (1, samples) through the checkpoint's front
    end, whose reflect pad of (n_fft - hop) / 2 a side gives samples // hop frames; with
    padded_frames, zeros after the reflect pad extend it to that many frames."""
    padding_samples = (SPEAKER_MEL_N_FFT - SPEAKER_MEL_HOP) // 2
    padded_waveform = F.pad(
        waveform, (padding_samples, padding_samples), mode="reflect"
    )
    if padded_frames is not None:
        # note (ratish): up to the longest input with padded_frames frames, so the count
        # is never negative; no frame inside the clip reads the zeros.
        tail_samples = SPEAKER_MEL_HOP * (padded_frames + 1) - 1 - waveform.shape[1]
        padded_waveform = F.pad(padded_waveform, (0, tail_samples))
    else:
        pass
    spectrum = torch.stft(
        padded_waveform,
        SPEAKER_MEL_N_FFT,
        hop_length=SPEAKER_MEL_HOP,
        win_length=SPEAKER_MEL_N_FFT,
        window=window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    magnitude = torch.sqrt(torch.view_as_real(spectrum).pow(2).sum(-1) + 1e-9)
    return torch.log(
        torch.clamp(torch.matmul(mel_basis, magnitude), min=SPEAKER_MEL_CLIP)
    )


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
    """The encoder's forward over its own weights, exact for the first length frames
    of mels (1, mels, width) whatever fills the rest of the width."""
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
    return encoder.fc(torch.cat((mean, std), dim=1).unsqueeze(2)).squeeze(-1)[0]


@dataclass
class CapturedSpeakerEncoderGraph:
    graph: ReplayableGraph
    static_mels: torch.Tensor
    static_frames: torch.Tensor
    static_embedding: torch.Tensor


class Qwen3TTSSpeakerEncoderCudaGraphRunner:
    """The speaker embedding of a waveform: the checkpoint's mel front end on the CPU,
    then one captured encode per bucket length; past the largest bucket, on CUDA the
    same mel on the device and the encode at a rounded width, elsewhere both eager."""

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
        self.device_mel_window: torch.Tensor | None = None
        self.device_mel_basis: torch.Tensor | None = None
        self.lock = threading.Lock()
        self.graphs: dict[int, CapturedSpeakerEncoderGraph] = {}
        self.bucket_frames: list[int] = []
        self.disable_reason: str | None = None
        self.replays = 0
        self.misses = 0

    def mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Log mel (1, mels, frames) of a CPU waveform (1, samples)."""
        return log_mel(waveform, self.mel_window, self.mel_basis)

    def capture(self, codec_frame_buckets: Iterable[int], codec_hop: int) -> None:
        """One graph per bucket of the reference encoder's ladder, in mel frames;
        none when a capture fails, so every clip runs eager."""
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
        graphs: dict[int, CapturedSpeakerEncoderGraph] = {}
        try:
            with device_guard(param.device):
                pool = graph_backend.graph_pool_handle()
                stream = device_module.Stream(device=param.device)
                # note(ratish): largest first so the shared pool is sized once.
                for frames in reversed(buckets):
                    static_mels = torch.zeros(
                        (1, self.mel_basis.shape[0], frames),
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
                                self.encoder, static_mels, static_frames, self.pads
                            )
                    with (
                        torch.inference_mode(),
                        graph_backend.capture(
                            pool=pool,
                            stream=stream,
                            thread_local_errors=True,
                        ) as graph,
                    ):
                        static_embedding = encode_bucketed(
                            self.encoder, static_mels, static_frames, self.pads
                        )
                    stream.synchronize()
                    graphs[frames] = CapturedSpeakerEncoderGraph(
                        graph=graph,
                        static_mels=static_mels,
                        static_frames=static_frames,
                        static_embedding=static_embedding,
                    )
        except Exception as exc:
            self.disable_reason = f"capture_failed: {type(exc).__name__}: {exc}"
            logger.warning(
                "Qwen3-TTS speaker encoder graph capture disabled the runner: "
                f"{self.disable_reason}",
                exc_info=True,
            )
            return
        self.graphs = graphs
        if current_platform.is_cuda():
            self.device_mel_window = self.mel_window.to(param.device)
            self.device_mel_basis = self.mel_basis.to(param.device)
            # note (ratish): the first device FFT initializes cuFFT, which waits for the
            # whole device, so it runs here, at the first width past the graphs.
            with torch.inference_mode(), device_guard(param.device):
                log_mel(
                    torch.zeros(
                        (1, SPEAKER_MEL_HOP * 2 * buckets[-1]),
                        device=param.device,
                        dtype=torch.float32,
                    ),
                    self.device_mel_window,
                    self.device_mel_basis,
                )
        else:
            pass
        logger.info(f"Qwen3-TTS speaker encoder graphs captured for {buckets} frames")

    def stats(self) -> ReferenceEncoderGraphStats:
        return {
            "enabled": bool(self.graphs),
            "disable_reason": self.disable_reason,
            "bucket_frames": list(self.bucket_frames),
            "captured": sorted(self.graphs),
            "replays": self.replays,
            "misses": self.misses,
        }

    def embed(self, waveform: np.ndarray) -> torch.Tensor:
        """Speaker embedding (enc_dim,) of a waveform at the encoder's sample rate."""
        samples = torch.from_numpy(waveform).unsqueeze(0)
        frames = samples.shape[1] // SPEAKER_MEL_HOP
        assert frames > max(self.pads)
        param = next(self.encoder.parameters())
        bucket = smallest_bucket(frames, self.graphs)
        if bucket is None and self.device_mel_basis is not None:
            # note (ratish): past the graphs the CPU mel sets a long clip's preprocessing, so
            # the same mel runs on the device, zero padded after its reflect pad to a width
            # rounded up to the largest bucket so cuDNN and cuFFT plan each width once.
            largest_frames = max(self.graphs)
            padded_frames = -(-frames // largest_frames) * largest_frames
            mels = log_mel(
                samples.to(param.device),
                self.device_mel_window,
                self.device_mel_basis,
                padded_frames,
            )
            with self.lock:
                self.misses += 1
            return encode_bucketed(
                self.encoder,
                mels.to(param.dtype),
                torch.full((1,), frames, device=param.device),
                self.pads,
            )
        elif bucket is None:
            mels = self.mel(samples).to(param.device).to(param.dtype)
            # note (Haoling Pu): preprocessing threads embed concurrently, and
            # model_info reports this count.
            with self.lock:
                self.misses += 1
            return self.encoder(mels.transpose(1, 2))[0]
        else:
            mels = self.mel(samples).to(param.device).to(param.dtype)
        captured = self.graphs[bucket]
        # note(ratish): every worker enqueues on one shared stream, so the lock only
        # orders the enqueue and the clone lands before the next replay.
        with self.lock:
            captured.static_mels[:, :, :frames].copy_(mels)
            captured.static_frames.fill_(frames)
            captured.graph.replay()
            self.replays += 1
            return captured.static_embedding.clone()
