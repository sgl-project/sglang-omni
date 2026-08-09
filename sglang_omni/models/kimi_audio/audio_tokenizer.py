# SPDX-License-Identifier: Apache-2.0
"""GLM-4-Voice tokenizer encoder used by Kimi-Audio.

The checkpoint is a Whisper encoder truncated at its VQ layer.  Keeping the
small inference-only implementation here avoids a runtime dependency on a
source checkout of GLM-4-Voice.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig, WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer

_ENCODER_WEIGHT_PREFIXES = (
    "conv1.",
    "conv2.",
    "embed_positions.",
    "embed_positions2.",
    "layers.",
    "codebook.",
)


def _encoder_weight_name(checkpoint_name: str) -> str | None:
    """Normalize the two GLM-4-Voice checkpoint layouts in circulation."""
    name = checkpoint_name.removeprefix("model.encoder.")
    if name.startswith(_ENCODER_WEIGHT_PREFIXES):
        return name
    return None


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kernel_size = kwargs.get("kernel_size", args[2] if len(args) > 2 else 1)
        dilation = kwargs.get("dilation", 1)
        kwargs["padding"] = 0
        super().__init__(*args, **kwargs)
        kernel = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        dil = dilation[0] if isinstance(dilation, tuple) else dilation
        self.left_padding = dil * (kernel - 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return super().forward(nn.functional.pad(inputs, (self.left_padding, 0)))


class WhisperVQEncoder(nn.Module):
    """Inference-only equivalent of GLM-4-Voice's ``WhisperVQEncoder``."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        conv_cls = CausalConv1d if config.encoder_causal_convolution else nn.Conv1d
        self.conv1 = conv_cls(
            config.num_mel_bins, config.d_model, kernel_size=3, padding=1
        )
        self.conv2 = conv_cls(
            config.d_model,
            config.d_model,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.embed_positions = nn.Embedding(config.max_source_positions, config.d_model)
        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(config) for _ in range(config.quantize_position)]
        )
        self.pooling_layer = nn.AvgPool1d(config.pooling_kernel_size)
        pooled_positions = math.ceil(
            config.max_source_positions / config.pooling_kernel_size
        )
        self.codebook = nn.Embedding(config.quantize_vocab_size, config.d_model)
        self.embed_positions2 = nn.Embedding(pooled_positions, config.d_model)

    @staticmethod
    def _block_causal_mask(
        attention_mask: torch.Tensor, dtype: torch.dtype, block_size: int
    ) -> torch.Tensor:
        seq_len = attention_mask.shape[1]
        causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=attention_mask.device)
        )
        for start in range(0, seq_len, block_size):
            causal[start : start + block_size, start : start + block_size] = True
        allowed = causal.unsqueeze(0) & attention_mask[:, None, :].bool()
        mask = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
        mask.masked_fill_(~allowed, torch.finfo(dtype).min)
        return mask.unsqueeze(1)

    def forward(
        self, input_features: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden = nn.functional.gelu(self.conv1(input_features))
        hidden = nn.functional.gelu(self.conv2(hidden)).transpose(1, 2)
        seq_len = hidden.shape[1]
        hidden = hidden + self.embed_positions.weight[:seq_len]
        attention_mask = attention_mask[
            :, :: self.conv1.stride[0] * self.conv2.stride[0]
        ]
        block_size = int(self.config.quantize_causal_block_size)
        attn_mask = self._block_causal_mask(attention_mask, hidden.dtype, block_size)

        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, attn_mask)
            if index + 1 == self.config.pooling_position:
                hidden = hidden.transpose(1, 2)
                remainder = hidden.shape[-1] % self.config.pooling_kernel_size
                if remainder:
                    hidden = nn.functional.pad(
                        hidden, (0, self.config.pooling_kernel_size - remainder)
                    )
                hidden = self.pooling_layer(hidden).transpose(1, 2)
                attention_mask = attention_mask[:, :: self.config.pooling_kernel_size]
                attn_mask = self._block_causal_mask(
                    attention_mask,
                    hidden.dtype,
                    block_size // self.config.pooling_kernel_size,
                )

        flat = hidden.reshape(-1, hidden.shape[-1])
        codebook = self.codebook.weight
        distances = (
            flat.square().sum(dim=1, keepdim=True)
            + codebook.square().sum(dim=1)
            - 2 * flat @ codebook.t()
        )
        return distances.argmin(dim=1).reshape(hidden.shape[:2])

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str) -> WhisperVQEncoder:
        config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=False)
        for key, default in {
            "encoder_causal_convolution": True,
            "quantize_position": 16,
            "pooling_position": 16,
            "pooling_kernel_size": 4,
            "quantize_vocab_size": 16384,
            "quantize_causal_block_size": 200,
        }.items():
            if not hasattr(config, key):
                setattr(config, key, default)
        model = cls(config)
        state: dict[str, torch.Tensor] = {}
        for path in Path(checkpoint_dir).glob("model*.safetensors"):
            for name, tensor in load_file(str(path), device="cpu").items():
                local_name = _encoder_weight_name(name)
                if local_name is None:
                    continue
                if local_name.startswith("layers."):
                    layer_index = int(local_name.split(".")[1])
                    if layer_index >= config.quantize_position:
                        continue
                state[local_name] = tensor
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "GLM-4-Voice tokenizer checkpoint does not match the supported "
                f"encoder (missing={missing}, unexpected={unexpected})"
            )
        return model


class Glm4SpeechTokenizer(nn.Module):
    def __init__(self, checkpoint_dir: str) -> None:
        super().__init__()
        self.encoder = WhisperVQEncoder.from_pretrained(checkpoint_dir).eval()
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(checkpoint_dir)

    @torch.inference_mode()
    def tokenize(
        self, waveform: torch.Tensor, sample_rate: int = 16000
    ) -> torch.Tensor:
        if waveform.ndim != 1:
            waveform = waveform.reshape(-1)
        stride = (
            self.encoder.conv1.stride[0]
            * self.encoder.conv2.stride[0]
            * self.encoder.config.pooling_kernel_size
            * self.feature_extractor.hop_length
        )
        all_tokens: list[torch.Tensor] = []
        for start in range(0, waveform.numel(), 30 * sample_rate):
            segment = waveform[start : start + 30 * sample_rate].cpu().numpy()
            features = self.feature_extractor(
                segment,
                sampling_rate=sample_rate,
                return_attention_mask=True,
                return_tensors="pt",
                padding="longest",
                pad_to_multiple_of=stride,
            )
            device = self.encoder.conv1.weight.device
            input_features = features.input_features.to(
                device=device, dtype=self.encoder.conv1.weight.dtype
            )
            attention_mask = features.attention_mask.to(device=device)
            tokens = self.encoder(input_features, attention_mask)
            valid = attention_mask[
                :, :: self.encoder.conv1.stride[0] * self.encoder.conv2.stride[0]
            ]
            valid = valid[:, :: self.encoder.config.pooling_kernel_size].bool()
            all_tokens.append(tokens[valid])
        return torch.cat(all_tokens) if all_tokens else torch.empty(0, dtype=torch.long)


__all__ = ["Glm4SpeechTokenizer", "WhisperVQEncoder"]
