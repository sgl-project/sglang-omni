# SPDX-License-Identifier: Apache-2.0
"""CUDA graphs for the Qwen3-TTS speaker encoder at bucketed mel lengths."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn

from sglang_omni.models.qwen3_tts.reference_encoder_cuda_graph import smallest_bucket

logger = logging.getLogger(__name__)

SPEAKER_MEL_N_FFT = 1024
SPEAKER_MEL_HOP = 256
SPEAKER_MEL_FMIN = 0
SPEAKER_MEL_FMAX = 12000
SPEAKER_MEL_CLIP = 1e-5
# note(ratish): a replay costs a launch floor, not the padded frames (the same
# 1.05 ms at steps 32, 64 and 128, measured on H100), so the step only trades graph
# count against padding; 1920 frames is the reference encoder's 256 codec frame
# ceiling, so both captured encoders run eager past the same reference length.
DEFAULT_QWEN3_TTS_SPEAKER_ENCODER_BUCKET_FRAMES = tuple(range(128, 1921, 128))


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
        mean = (x * mask).sum(2, keepdim=True) / length.to(x.dtype)
        x = x * torch.sigmoid(se.conv2(F.relu(se.conv1(mean)))) + residual
        outputs.append(x)
    x = time_delay_block(encoder.mfa, torch.cat(outputs, dim=1), indices)
    asp = encoder.asp
    weights = mask / length.to(x.dtype)
    mean = (weights * x).sum(2)
    std = torch.sqrt((weights * (x - mean.unsqueeze(2)).pow(2)).sum(2).clamp(asp.eps))
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
    mean = (attention * x).sum(2)
    std = torch.sqrt((attention * (x - mean.unsqueeze(2)).pow(2)).sum(2).clamp(asp.eps))
    return encoder.fc(torch.cat((mean, std), dim=1).unsqueeze(2)).squeeze(-1)[0]


@dataclass
class CapturedSpeakerEncoderGraph:
    graph: torch.cuda.CUDAGraph
    static_mels: torch.Tensor
    static_frames: torch.Tensor
    static_embedding: torch.Tensor


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
        # note(ratish): the engine constructs models under a CUDA default device.
        self.mel_window = torch.hann_window(SPEAKER_MEL_N_FFT, device="cpu")
        self.pads = frozenset(
            conv.dilation[0] * (conv.kernel_size[0] - 1) // 2
            for conv in encoder.modules()
            if isinstance(conv, torch.nn.Conv1d)
        ) - {0}
        self.lock = threading.Lock()
        self.graphs: dict[int, CapturedSpeakerEncoderGraph] = {}
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

    def capture(self, bucket_frames: Iterable[int]) -> None:
        param = next(self.encoder.parameters())
        if param.device.type not in {"cuda", "musa"}:
            return
        else:
            pass
        buckets = sorted(bucket_frames)
        with torch.cuda.device(param.device):
            pool = torch.cuda.graph_pool_handle()
            stream = torch.cuda.Stream(device=param.device)
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
                stream.wait_stream(torch.cuda.current_stream(param.device))
                with torch.inference_mode(), torch.cuda.stream(stream):
                    for _ in range(2):
                        encode_bucketed(
                            self.encoder, static_mels, static_frames, self.pads
                        )
                graph = torch.cuda.CUDAGraph()
                with (
                    torch.inference_mode(),
                    torch.cuda.graph(
                        graph,
                        pool=pool,
                        stream=stream,
                        capture_error_mode="thread_local",
                    ),
                ):
                    static_embedding = encode_bucketed(
                        self.encoder, static_mels, static_frames, self.pads
                    )
                stream.synchronize()
                self.graphs[frames] = CapturedSpeakerEncoderGraph(
                    graph=graph,
                    static_mels=static_mels,
                    static_frames=static_frames,
                    static_embedding=static_embedding,
                )
        logger.info(f"Qwen3-TTS speaker encoder graphs captured for {buckets} frames")

    def embed(self, waveform: np.ndarray) -> torch.Tensor:
        """Speaker embedding (enc_dim,) of a waveform at the encoder's sample rate."""
        mels = self.mel(torch.from_numpy(waveform).unsqueeze(0))
        frames = mels.shape[2]
        assert frames > max(self.pads)
        param = next(self.encoder.parameters())
        mels = mels.to(param.device).to(param.dtype)
        bucket = smallest_bucket(frames, self.graphs)
        if bucket is None:
            self.misses += 1
            return self.encoder(mels.transpose(1, 2))[0]
        else:
            pass
        captured = self.graphs[bucket]
        # note(ratish): every worker enqueues on one shared stream, so the lock only
        # orders the enqueue and the clone lands before the next replay.
        with self.lock:
            captured.static_mels[:, :, :frames].copy_(mels)
            captured.static_frames.fill_(frames)
            captured.graph.replay()
            self.replays += 1
            return captured.static_embedding.clone()
