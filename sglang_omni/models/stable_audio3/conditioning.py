# SPDX-License-Identifier: Apache-2.0
# Adapted from Stability-AI/stable-audio-3 at commit 790793b89ddceea612be9ce42ff6830d18f6fa57.
# Copyright (c) 2026 Stability AI. The MIT license notice is retained in licenses/.
"""Prompt and duration conditioning: T5Gemma text tokens and a Fourier duration embedding."""

from pathlib import Path
from typing import Protocol

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    T5GemmaEncoderModel,
)

from sglang_omni.models.stable_audio3.layers import expo_fourier_features
from sglang_omni.models.stable_audio3.model_config import ConditioningSection

SECONDS_FEATURES = 256


class EncodeText(Protocol):
    def __call__(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor: ...


class PromptConditioner(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.padding_embedding = nn.Parameter(torch.zeros(dim))


class NumberEmbedder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        # note (ruilin): the identity stands for the parameter-free Fourier features, keeping the embedding.1 name.
        self.embedding = nn.Sequential(nn.Identity(), nn.Linear(SECONDS_FEATURES, dim))

    @property
    def projection(self) -> nn.Linear:
        return self.embedding[1]


class SecondsConditioner(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.embedder = NumberEmbedder(dim)


class Conditioner(nn.Module):
    """Checkpoint tensors under conditioner; the text encoder is loaded from its own subfolder."""

    def __init__(self, conditioning: ConditioningSection) -> None:
        super().__init__()
        prompt_section, seconds_section = conditioning.configs
        self.max_length: int = prompt_section.config.max_length
        self.seconds_range: tuple[float, float] = (
            seconds_section.config.min_val,
            seconds_section.config.max_val,
        )
        self.conditioners = nn.ModuleDict(
            {
                "prompt": PromptConditioner(conditioning.cond_dim),
                "seconds_total": SecondsConditioner(conditioning.cond_dim),
            }
        )

    def embed_prompts(
        self,
        prompts: list[str],
        tokenizer: PreTrainedTokenizerBase,
        encode: EncodeText,
    ) -> torch.Tensor:
        """Text tokens padded to max_length, with the learned embedding at padding positions; encode maps token ids and mask to hidden states."""
        padding_embedding = self.conditioners["prompt"].padding_embedding
        encoded = tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        ).to(padding_embedding.device)
        attention_mask = encoded["attention_mask"].bool()
        hidden_states = encode(encoded["input_ids"], attention_mask)
        return torch.where(
            attention_mask[..., None],
            hidden_states.to(padding_embedding.dtype),
            padding_embedding,
        )

    def embed_seconds(self, seconds: list[float]) -> torch.Tensor:
        projection = self.conditioners["seconds_total"].embedder.projection
        min_seconds, max_seconds = self.seconds_range
        normalized = (
            torch.tensor(seconds, device=projection.weight.device).clamp(
                min_seconds, max_seconds
            )
            - min_seconds
        ) / (max_seconds - min_seconds)
        return projection(
            expo_fourier_features(
                normalized.to(projection.weight.dtype), SECONDS_FEATURES
            )
        )


def load_text_encoder(
    text_encoder_dir: Path, device: torch.device
) -> tuple[PreTrainedTokenizerBase, T5GemmaEncoderModel]:
    """The T5Gemma encoder in its checkpoint dtype, as the model was trained to read it."""
    config = AutoConfig.from_pretrained(text_encoder_dir)
    config.is_encoder_decoder = False
    text_encoder = T5GemmaEncoderModel.from_pretrained(text_encoder_dir, config=config)
    return (
        AutoTokenizer.from_pretrained(text_encoder_dir),
        text_encoder.to(device).eval(),
    )
