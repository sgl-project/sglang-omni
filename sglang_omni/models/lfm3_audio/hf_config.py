# SPDX-License-Identifier: Apache-2.0
"""LFM3-Audio backbone config for the SGLang runtime.

The checkpoint's ``config.json`` is in Liquid's export format (``lfm``,
``encoder`` and ``depthformer`` sections, custom code). The AR stage serves
the ``lfm`` section as an LFM2 config so the runtime allocates the short-conv
state pool, plus the LFM3 additions (sliding attention layers, sandwich norm,
per-layer RoPE) and the speech-output constants.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from sglang.srt.configs.lfm2 import Lfm2Config

MODEL_TYPE = "lfm3_audio_sglang"
SGLANG_ARCHITECTURE = "Lfm3AudioSGLangModel"
CHECKPOINT_FILE_NAME = "model.safetensors"
ATTENTION_LAYER_TYPES = ("full_attention", "sliding_attention")


@dataclass(frozen=True)
class Lfm3AudioTokens:
    """Special token ids and speech constants of the released checkpoint."""

    audio_token_id: int = 125031
    audio_out_token_id: int = 125032
    audio_start_token_id: int = 125012
    text_end_token_id: int = 125014
    mixed_start_token_id: int = 125015
    eos_token_ids: tuple[int, ...] = (124900, 125014)
    num_codebooks: int = 12
    codebook_size: int = 2017
    audio_eos_code: int = 2016
    maskgit_steps: int = 3
    interleaved_n_text: int = 6
    interleaved_n_audio: int = 12

    @property
    def maskgit_schedule(self) -> list[int]:
        schedule = [self.num_codebooks // self.maskgit_steps] * self.maskgit_steps
        schedule[-1] += self.num_codebooks - sum(schedule)
        return schedule

    @property
    def placeholder_token_ids(self) -> tuple[int, int]:
        return (self.audio_token_id, self.audio_out_token_id)


class Lfm3SGLangConfig(Lfm2Config):
    """LFM3 text backbone; an LFM2 config so the runtime allocates the conv-state pool."""

    model_type = MODEL_TYPE

    def __init__(
        self,
        *,
        sliding_window: int | None = None,
        use_sandwich_norm: bool = False,
        rope_parameters: dict | None = None,
        lfm3_rope_parameters: dict | None = None,
        head_dim: int | None = None,
        depth_config: dict | None = None,
        num_codebooks: int = 12,
        codebook_size: int = 2017,
        maskgit_steps: int = 3,
        interleaved_n_text: int = 6,
        interleaved_n_audio: int = 12,
        **kwargs: object,
    ) -> None:
        self.sliding_window = sliding_window
        self.use_sandwich_norm = use_sandwich_norm
        # Per-layer-type RoPE ({"full_attention": ..., "sliding_attention": ...}). A
        # saved config carries it as `lfm3_rope_parameters`; `rope_parameters` is
        # LFM2's single table and keeps its default.
        if lfm3_rope_parameters is None and rope_parameters is not None:
            if set(rope_parameters) <= set(ATTENTION_LAYER_TYPES):
                lfm3_rope_parameters = rope_parameters
            else:
                pass
        else:
            pass
        self.lfm3_rope_parameters = lfm3_rope_parameters or {}
        self.depth_config = depth_config or {}
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.maskgit_steps = maskgit_steps
        self.interleaved_n_text = interleaved_n_text
        self.interleaved_n_audio = interleaved_n_audio
        super().__init__(**kwargs)
        self.head_dim = head_dim or self.hidden_size // self.num_attention_heads

    @property
    def full_attention_layer_ids(self) -> list[int]:
        """Every attention layer keeps KV, sliding ones included."""
        return [
            index
            for index, layer_type in enumerate(self.layer_types)
            if layer_type in ATTENTION_LAYER_TYPES
        ]

    def rope_for(self, layer_type: str) -> dict:
        return dict(self.lfm3_rope_parameters[layer_type])


def load_checkpoint_config(checkpoint_dir: str) -> dict:
    with open(os.path.join(checkpoint_dir, "config.json")) as handle:
        return json.load(handle)


def lfm3_config_from_checkpoint(checkpoint_dir: str) -> Lfm3SGLangConfig:
    """Build the backbone config from the checkpoint's ``lfm`` section."""
    raw = load_checkpoint_config(checkpoint_dir)
    if "lfm" not in raw:
        raise ValueError(
            f"{checkpoint_dir}/config.json has no `lfm` section; expected an LFM3-Audio checkpoint"
        )
    else:
        pass
    text = raw["lfm"]
    return Lfm3SGLangConfig(
        architectures=[SGLANG_ARCHITECTURE],
        vocab_size=text["vocab_size"],
        hidden_size=text["hidden_size"],
        intermediate_size=text["intermediate_size"],
        num_hidden_layers=text["num_hidden_layers"],
        num_attention_heads=text["num_attention_heads"],
        num_key_value_heads=text["num_key_value_heads"],
        head_dim=text.get("head_dim"),
        max_position_embeddings=text["max_position_embeddings"],
        norm_eps=text["norm_eps"],
        conv_bias=text["conv_bias"],
        conv_L_cache=text["conv_L_cache"],
        layer_types=list(text["layer_types"]),
        sliding_window=text.get("sliding_window"),
        use_sandwich_norm=bool(text.get("use_sandwich_norm", False)),
        rope_parameters=text["rope_parameters"],
        tie_word_embeddings=bool(text.get("tie_word_embeddings", True)),
        pad_token_id=text.get("pad_token_id"),
        bos_token_id=text.get("bos_token_id"),
        eos_token_id=text.get("eos_token_id"),
        # The released export keeps the FFN width as stored; LFM2's auto-adjust would shrink it.
        block_auto_adjust_ff_dim=False,
        depth_config=raw.get("depthformer", {}),
        num_codebooks=raw.get("codebooks", 12),
        codebook_size=raw.get("audio_vocab_size", 2017),
        maskgit_steps=raw.get("maskgit_steps", 3),
        interleaved_n_text=raw.get("interleaved_n_text", 6),
        interleaved_n_audio=raw.get("interleaved_n_audio", 12),
    )


def lfm3_tokens_from_checkpoint(checkpoint_dir: str) -> Lfm3AudioTokens:
    raw = load_checkpoint_config(checkpoint_dir)
    defaults = Lfm3AudioTokens()
    generation_path = os.path.join(checkpoint_dir, "generation_config.json")
    eos = defaults.eos_token_ids
    if os.path.exists(generation_path):
        with open(generation_path) as handle:
            generation = json.load(handle)
        value = generation.get("eos_token_id")
        if value is not None:
            eos = tuple(value) if isinstance(value, list) else (int(value),)
        else:
            pass
    else:
        pass
    return Lfm3AudioTokens(
        audio_token_id=raw.get("audio_token_id", defaults.audio_token_id),
        audio_out_token_id=raw.get("audio_out_token_id", defaults.audio_out_token_id),
        audio_start_token_id=raw.get(
            "audio_start_token_id", defaults.audio_start_token_id
        ),
        text_end_token_id=raw.get("text_end_token_id", defaults.text_end_token_id),
        mixed_start_token_id=raw.get(
            "mixed_start_token_id", defaults.mixed_start_token_id
        ),
        eos_token_ids=eos,
        num_codebooks=raw.get("codebooks", defaults.num_codebooks),
        codebook_size=raw.get("audio_vocab_size", defaults.codebook_size),
        audio_eos_code=raw.get("audio_vocab_size", defaults.codebook_size) - 1,
        maskgit_steps=raw.get("maskgit_steps", defaults.maskgit_steps),
        interleaved_n_text=raw.get("interleaved_n_text", defaults.interleaved_n_text),
        interleaved_n_audio=raw.get(
            "interleaved_n_audio", defaults.interleaved_n_audio
        ),
    )
