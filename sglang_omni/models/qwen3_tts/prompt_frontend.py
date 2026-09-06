# SPDX-License-Identifier: Apache-2.0
"""Standalone Qwen3-TTS prompt frontend.

Holds only what request preprocessing needs from the talker (embedding tables, the
text projection, the predictor codec embeddings and the speaker encoder) so the
preprocessing stage can run in its own process without the 1.7B transformer.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from typing import Any

import torch
from torch import nn

from sglang_omni.models.qwen3_tts.compat import (
    apply_qwen_tts_transformers_compatibility_patches,
)
from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSPromptBuilderMixin

_TALKER_PREFIX = "talker."
_SPEAKER_ENCODER_PREFIX = "speaker_encoder."


class _PromptProjection(nn.Module):
    """Linear-SiLU-Linear with the talker checkpoint field names."""

    def __init__(self, in_size: int, intermediate_size: int, out_size: int) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(in_size, intermediate_size, bias=True)
        self.act = nn.SiLU()
        self.linear_fc2 = nn.Linear(intermediate_size, out_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act(self.linear_fc1(x)))


class _PromptEmbeddings(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.text_embedding = nn.Embedding(
            config.text_vocab_size, config.text_hidden_size
        )
        # Note (Jiaxin Deng): the prompt builders only read this buffer's device
        # and dtype; the talker's real feedback buffer lives in the engine.
        self.register_buffer(
            "_feedback_buffer", torch.zeros(1, config.hidden_size), persistent=False
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.codec_embedding

    def get_text_embeddings(self) -> nn.Embedding:
        return self.text_embedding


class _PromptPredictorEmbeddings(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        cp_config = config.code_predictor_config
        self.model = nn.Module()
        self.model.codec_embedding = nn.ModuleList(
            [
                nn.Embedding(cp_config.vocab_size, config.hidden_size)
                for _ in range(config.num_code_groups - 1)
            ]
        )


class Qwen3TTSPromptFrontend(Qwen3TTSPromptBuilderMixin, nn.Module):
    """Talker-compatible prompt builder without the transformer stacks."""

    def __init__(self, root_config: Any, *, device: Any, dtype: torch.dtype) -> None:
        nn.Module.__init__(self)
        config = root_config.talker_config
        self.root_config = root_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.tts_model_type = getattr(root_config, "tts_model_type", "base")
        self.speaker_encoder_sample_rate = getattr(
            getattr(root_config, "speaker_encoder_config", None),
            "sample_rate",
            24000,
        )
        self.text_projection = _PromptProjection(
            config.text_hidden_size, config.text_hidden_size, config.hidden_size
        )
        self.model = _PromptEmbeddings(config)
        self.code_predictor = _PromptPredictorEmbeddings(config)
        if self.tts_model_type == "base":
            apply_qwen_tts_transformers_compatibility_patches()
            from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder

            self.speaker_encoder = Qwen3TTSSpeakerEncoder(
                root_config.speaker_encoder_config
            )
        else:
            self.speaker_encoder = None
        self.speech_tokenizer = None
        self.to(device=device, dtype=dtype)
        self.requires_grad_(False)

    def checkpoint_weight_names(self) -> set[str]:
        names = set()
        for name, _ in self.named_parameters():
            if name.startswith(_SPEAKER_ENCODER_PREFIX):
                names.add(name)
            else:
                names.add(_TALKER_PREFIX + name)
        return names

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, tensor in weights:
            if name.startswith(_TALKER_PREFIX):
                target = name[len(_TALKER_PREFIX) :]
            elif name.startswith(_SPEAKER_ENCODER_PREFIX):
                target = name
            else:
                continue
            param = params.get(target)
            if param is None:
                continue
            param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
            loaded.add(target)
        missing = sorted(set(params) - loaded)
        if missing:
            raise RuntimeError(
                f"Qwen3-TTS prompt frontend is missing {len(missing)} weights "
                f"(e.g. {missing[:3]})"
            )


def iter_checkpoint_tensors(
    checkpoint_dir: str, names: set[str]
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield only the named tensors from a safetensors checkpoint directory."""
    from safetensors import safe_open

    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted({weight_map[name] for name in names if name in weight_map})
    else:
        shards = sorted(
            entry
            for entry in os.listdir(checkpoint_dir)
            if entry.endswith(".safetensors")
        )
    for shard in shards:
        with safe_open(
            os.path.join(checkpoint_dir, shard), framework="pt", device="cpu"
        ) as handle:
            for name in handle.keys():
                if name in names:
                    yield name, handle.get_tensor(name)


def load_qwen3_tts_prompt_frontend(
    checkpoint_dir: str, *, device: Any, dtype: torch.dtype
) -> Qwen3TTSPromptFrontend:
    from transformers import AutoConfig

    root_config = AutoConfig.from_pretrained(checkpoint_dir)
    frontend = Qwen3TTSPromptFrontend(root_config, device=device, dtype=dtype)
    frontend.load_weights(
        iter_checkpoint_tensors(checkpoint_dir, frontend.checkpoint_weight_names())
    )
    return frontend
