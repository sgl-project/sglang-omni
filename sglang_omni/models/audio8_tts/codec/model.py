# SPDX-License-Identifier: Apache-2.0
"""Load the Audio8 reference codec without executing checkpoint Python code."""

from __future__ import annotations

from _thread import LockType
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from threading import Lock

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sglang_omni.models.audio8_tts.codec.attention import CodecTransformerConfig
from sglang_omni.models.audio8_tts.codec.convolution import (
    CODEC_CHANNEL_COUNT,
    CodecDecoder,
    CodecEncoder,
)
from sglang_omni.models.audio8_tts.codec.quantization import (
    CODEBOOK_COUNT,
    DownsampleQuantizer,
)
from sglang_omni.models.audio8_tts.hf_config import FRAME_SAMPLES, ArkttsConfig


class Audio8Codec(nn.Module):
    def __init__(self, config: ArkttsConfig) -> None:
        super().__init__()
        self.encoder: CodecEncoder = CodecEncoder()
        self.quantizer: DownsampleQuantizer = DownsampleQuantizer(
            CodecTransformerConfig(
                layer_count=config.codec_post_n_layer,
                attention_head_count=config.codec_post_n_head,
                key_value_head_count=config.codec_post_n_local_heads,
                hidden_size=CODEC_CHANNEL_COUNT,
                intermediate_size=config.codec_post_intermediate_size,
            )
        )
        self.decoder: CodecDecoder = CodecDecoder()

    def encode(self, waveform: Tensor) -> Tensor:
        assert waveform.ndim == 1 and waveform.numel()
        waveform = F.pad(waveform, (0, -waveform.numel() % FRAME_SAMPLES))
        return self.quantizer(self.encoder(waveform[None, None]))[0]

    def decode(self, audio_codes: Tensor) -> Tensor:
        assert audio_codes.ndim == 2 and audio_codes.shape[0] == CODEBOOK_COUNT
        return self.decoder(self.quantizer.decode(audio_codes[None]))[0, 0]


@dataclass(kw_only=True)
class LoadedCodec:
    model: Audio8Codec
    device: torch.device
    dtype: torch.dtype
    lock: LockType = field(default_factory=Lock)

    @torch.inference_mode()
    def encode(self, waveform: Tensor) -> Tensor:
        with self.lock:
            return self.model.encode(waveform.to(device=self.device, dtype=self.dtype))

    @torch.inference_mode()
    def decode(self, audio_codes: Tensor) -> Tensor:
        with self.lock:
            return self.model.decode(
                audio_codes.to(device=self.device, dtype=torch.long)
            )


@lru_cache(maxsize=2)
def load_audio8_codec(checkpoint: Path, device: torch.device) -> LoadedCodec:
    config = ArkttsConfig.from_pretrained(checkpoint)
    codec_filename = Path(config.codec_filename)
    if codec_filename.name != config.codec_filename:
        raise ValueError("Audio8 codec_filename must name a file in the checkpoint")
    else:
        pass
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    with torch.device("meta"):
        codec = Audio8Codec(config)
    state = torch.load(
        checkpoint / codec_filename, map_location="cpu", weights_only=True, mmap=True
    )
    if "state_dict" in state:
        state = state["state_dict"]
    else:
        pass
    if any("generator." in name for name in state):
        state = {
            name.replace("generator.", ""): tensor
            for name, tensor in state.items()
            if "generator." in name
        }
    else:
        pass
    codec_weights: dict[str, Tensor] = {
        name: tensor
        for name, tensor in state.items()
        if not name.endswith(("freqs_cis", "causal_mask"))
    }
    codec.load_state_dict(codec_weights, strict=True, assign=True)
    return LoadedCodec(
        model=codec.to(device=device, dtype=dtype).eval(), device=device, dtype=dtype
    )
