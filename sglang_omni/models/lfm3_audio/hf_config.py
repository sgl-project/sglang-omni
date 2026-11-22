# SPDX-License-Identifier: Apache-2.0
"""LFM3 backbone config for the SGLang runtime, for LFM3-Audio and LFM3-Omni.

LFM3-Audio's ``config.json`` is in Liquid's export format (``lfm``, ``encoder``
and ``depthformer`` sections); LFM3-Omni's is a Transformers config
(``text_config``, ``audio_encoder_config``, ``depth_config``,
``vision_config``). Both share the text backbone, depth decoder and speech
constants. The AR stage serves the text section as an LFM2 config so the
runtime allocates the short-conv state pool, plus the LFM3 additions (sliding
attention layers, sandwich norm, per-layer RoPE).
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
    """Special token ids and speech constants of a released checkpoint."""

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
    # LFM3-Omni only.
    image_token_id: int | None = None
    video_token_id: int | None = None

    @property
    def maskgit_schedule(self) -> list[int]:
        schedule = [self.num_codebooks // self.maskgit_steps] * self.maskgit_steps
        schedule[-1] += self.num_codebooks - sum(schedule)
        return schedule

    @property
    def placeholder_token_ids(self) -> tuple[int, ...]:
        """Media placeholders, which are never sampled."""
        ids = (
            self.audio_token_id,
            self.audio_out_token_id,
            self.image_token_id,
            self.video_token_id,
        )
        return tuple(token_id for token_id in ids if token_id is not None)


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


def is_export_format(raw: dict) -> bool:
    """LFM3-Audio's export layout; LFM3-Omni ships a Transformers config."""
    return "lfm" in raw


def text_section(raw: dict, checkpoint_dir: str) -> dict:
    if is_export_format(raw):
        return raw["lfm"]
    elif "text_config" in raw:
        return raw["text_config"]
    else:
        raise ValueError(
            f"{checkpoint_dir}/config.json has neither an `lfm` nor a `text_config` "
            "section; expected an LFM3-Audio or LFM3-Omni checkpoint"
        )


def speech_constants(raw: dict) -> dict[str, int]:
    if is_export_format(raw):
        codebook_size = raw.get("audio_vocab_size", 2017)
        return {
            "num_codebooks": raw.get("codebooks", 12),
            "codebook_size": codebook_size,
            "audio_eos_code": codebook_size - 1,
            "maskgit_steps": raw.get("maskgit_steps", 3),
        }
    else:
        codebook_size = raw.get("codebook_size", 2017)
        return {
            "num_codebooks": raw.get("num_codebooks", 12),
            "codebook_size": codebook_size,
            "audio_eos_code": raw.get("audio_eos_code", codebook_size - 1),
            "maskgit_steps": len(raw.get("maskgit_schedule", [4, 4, 4])),
        }


def lfm3_config_from_checkpoint(checkpoint_dir: str) -> Lfm3SGLangConfig:
    """Build the backbone config from the checkpoint's text section."""
    raw = load_checkpoint_config(checkpoint_dir)
    text = text_section(raw, checkpoint_dir)
    speech = speech_constants(raw)
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
        # The released checkpoints keep the FFN width as stored; LFM2's auto-adjust would shrink it.
        block_auto_adjust_ff_dim=False,
        depth_config=raw.get("depthformer") or raw.get("depth_config") or {},
        num_codebooks=speech["num_codebooks"],
        codebook_size=speech["codebook_size"],
        maskgit_steps=speech["maskgit_steps"],
        interleaved_n_text=raw.get("interleaved_n_text", 6),
        interleaved_n_audio=raw.get("interleaved_n_audio", 12),
    )


def load_generation_config(checkpoint_dir: str) -> dict:
    path = os.path.join(checkpoint_dir, "generation_config.json")
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)
    else:
        return {}


def lfm3_tokens_from_checkpoint(checkpoint_dir: str) -> Lfm3AudioTokens:
    raw = load_checkpoint_config(checkpoint_dir)
    defaults = Lfm3AudioTokens()
    eos = load_generation_config(checkpoint_dir).get("eos_token_id")
    if eos is None:
        eos_ids = defaults.eos_token_ids
    elif isinstance(eos, list):
        eos_ids = tuple(eos)
    else:
        eos_ids = (int(eos),)
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
        eos_token_ids=eos_ids,
        interleaved_n_text=raw.get("interleaved_n_text", defaults.interleaved_n_text),
        interleaved_n_audio=raw.get(
            "interleaved_n_audio", defaults.interleaved_n_audio
        ),
        image_token_id=raw.get("image_token_id"),
        video_token_id=raw.get("video_token_id"),
        **speech_constants(raw),
    )


@dataclass(frozen=True)
class SpeechSamplingDefaults:
    """The checkpoint's speech-frame sampling defaults (generation_config.json)."""

    temperature: float = 1.0
    top_k: int | None = 128
    repetition_penalty: float = 1.1
    repetition_window: int = 64


def speech_defaults_from_checkpoint(checkpoint_dir: str) -> SpeechSamplingDefaults:
    generation = load_generation_config(checkpoint_dir)
    defaults = SpeechSamplingDefaults()
    return SpeechSamplingDefaults(
        temperature=generation.get("audio_temperature", defaults.temperature),
        top_k=generation.get("audio_top_k", defaults.top_k),
        repetition_penalty=generation.get(
            "audio_repetition_penalty", defaults.repetition_penalty
        ),
        repetition_window=generation.get(
            "audio_repetition_window", defaults.repetition_window
        ),
    )
