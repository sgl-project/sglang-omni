# SPDX-License-Identifier: Apache-2.0
"""LFM3 backbone config for the SGLang runtime, for LFM3-Audio and LFM3-Omni.

LFM3-Audio's config.json is in Liquid's export layout (lfm, encoder and depthformer sections); LFM3-Omni's is a Transformers config (text_config, audio_encoder_config, depth_config, vision_config). Both share the text backbone, depth decoder and speech constants. The AR stage serves the text section as an LFM2 config, so the runtime allocates the short-conv state pool, plus the LFM3 additions: sliding attention layers, sandwich norm and per-layer RoPE.
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
# One layer type's RoPE settings, as the checkpoint writes them.
RopeParameters = dict[str, float | str]


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
        token_ids = (
            self.audio_token_id,
            self.audio_out_token_id,
            self.image_token_id,
            self.video_token_id,
        )
        return tuple(token_id for token_id in token_ids if token_id is not None)


class Lfm3SGLangConfig(Lfm2Config):
    """LFM3 text backbone; an LFM2 config so the runtime allocates the conv-state pool."""

    model_type = MODEL_TYPE

    def __init__(
        self,
        *,
        sliding_window: int | None = None,
        use_sandwich_norm: bool = False,
        rope_parameters: dict[str, RopeParameters] | RopeParameters | None = None,
        lfm3_rope_parameters: dict[str, RopeParameters] | None = None,
        head_dim: int | None = None,
        depth_config: dict[str, object] | None = None,
        num_codebooks: int = 12,
        codebook_size: int = 2017,
        maskgit_steps: int = 3,
        interleaved_n_text: int = 6,
        interleaved_n_audio: int = 12,
        checkpoint_dir: str = "",
        **kwargs: object,
    ) -> None:
        # The source checkpoint, whose own code builds the media encoders.
        self.checkpoint_dir = checkpoint_dir
        self.sliding_window = sliding_window
        self.use_sandwich_norm = use_sandwich_norm
        # note (ruilin): a saved config carries the per-layer-type RoPE as lfm3_rope_parameters; rope_parameters is LFM2's single table.
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

    def rope_for(self, layer_type: str) -> RopeParameters:
        return dict(self.lfm3_rope_parameters[layer_type])


def load_checkpoint_config(checkpoint_dir: str) -> dict[str, object]:
    with open(os.path.join(checkpoint_dir, "config.json")) as handle:
        return json.load(handle)


def is_export_format(checkpoint_config: dict[str, object]) -> bool:
    """LFM3-Audio's export layout; LFM3-Omni ships a Transformers config."""
    return "lfm" in checkpoint_config


@dataclass(frozen=True, kw_only=True)
class SpeechConstants:
    """The codebook layout, read from either checkpoint layout."""

    num_codebooks: int
    codebook_size: int
    audio_eos_code: int
    maskgit_steps: int


def speech_constants(checkpoint_config: dict[str, object]) -> SpeechConstants:
    if is_export_format(checkpoint_config):
        codebook_size = int(checkpoint_config["audio_vocab_size"])
        return SpeechConstants(
            num_codebooks=int(checkpoint_config["codebooks"]),
            codebook_size=codebook_size,
            audio_eos_code=codebook_size - 1,
            maskgit_steps=int(checkpoint_config["maskgit_steps"]),
        )
    else:
        return SpeechConstants(
            num_codebooks=int(checkpoint_config["num_codebooks"]),
            codebook_size=int(checkpoint_config["codebook_size"]),
            audio_eos_code=int(checkpoint_config["audio_eos_code"]),
            maskgit_steps=len(checkpoint_config["maskgit_schedule"]),
        )


def lfm3_config_from_checkpoint(checkpoint_dir: str) -> Lfm3SGLangConfig:
    """Build the backbone config from the checkpoint's text section."""
    checkpoint_config = load_checkpoint_config(checkpoint_dir)
    if is_export_format(checkpoint_config):
        text_config, depth_config = (
            checkpoint_config["lfm"],
            checkpoint_config["depthformer"],
        )
    elif "text_config" in checkpoint_config:
        text_config, depth_config = (
            checkpoint_config["text_config"],
            checkpoint_config["depth_config"],
        )
    else:
        raise ValueError(
            f"{checkpoint_dir}/config.json has neither an lfm nor a text_config section; "
            "expected an LFM3-Audio or LFM3-Omni checkpoint"
        )
    speech = speech_constants(checkpoint_config)
    return Lfm3SGLangConfig(
        architectures=[SGLANG_ARCHITECTURE],
        vocab_size=text_config["vocab_size"],
        hidden_size=text_config["hidden_size"],
        intermediate_size=text_config["intermediate_size"],
        num_hidden_layers=text_config["num_hidden_layers"],
        num_attention_heads=text_config["num_attention_heads"],
        num_key_value_heads=text_config["num_key_value_heads"],
        head_dim=text_config["head_dim"],
        max_position_embeddings=text_config["max_position_embeddings"],
        norm_eps=text_config["norm_eps"],
        conv_bias=text_config["conv_bias"],
        conv_L_cache=text_config["conv_L_cache"],
        layer_types=list(text_config["layer_types"]),
        sliding_window=text_config["sliding_window"],
        use_sandwich_norm=bool(text_config["use_sandwich_norm"]),
        rope_parameters=text_config["rope_parameters"],
        tie_word_embeddings=bool(text_config["tie_word_embeddings"]),
        pad_token_id=text_config["pad_token_id"],
        bos_token_id=text_config["bos_token_id"],
        eos_token_id=text_config["eos_token_id"],
        # note (ruilin): the checkpoints store the FFN width as trained; LFM2's auto-adjust would shrink it.
        block_auto_adjust_ff_dim=False,
        depth_config=depth_config,
        num_codebooks=speech.num_codebooks,
        codebook_size=speech.codebook_size,
        maskgit_steps=speech.maskgit_steps,
        interleaved_n_text=checkpoint_config["interleaved_n_text"],
        interleaved_n_audio=checkpoint_config["interleaved_n_audio"],
    )


def load_generation_config(checkpoint_dir: str) -> dict[str, object]:
    with open(os.path.join(checkpoint_dir, "generation_config.json")) as handle:
        return json.load(handle)


def lfm3_tokens_from_checkpoint(checkpoint_dir: str) -> Lfm3AudioTokens:
    """Token ids and speech constants; LFM3-Audio's export config omits the token ids, which are the class defaults."""
    checkpoint_config = load_checkpoint_config(checkpoint_dir)
    defaults = Lfm3AudioTokens()
    speech = speech_constants(checkpoint_config)
    return Lfm3AudioTokens(
        audio_token_id=checkpoint_config.get("audio_token_id", defaults.audio_token_id),
        audio_out_token_id=checkpoint_config.get(
            "audio_out_token_id", defaults.audio_out_token_id
        ),
        audio_start_token_id=checkpoint_config.get(
            "audio_start_token_id", defaults.audio_start_token_id
        ),
        text_end_token_id=checkpoint_config.get(
            "text_end_token_id", defaults.text_end_token_id
        ),
        mixed_start_token_id=checkpoint_config.get(
            "mixed_start_token_id", defaults.mixed_start_token_id
        ),
        eos_token_ids=tuple(load_generation_config(checkpoint_dir)["eos_token_id"]),
        num_codebooks=speech.num_codebooks,
        codebook_size=speech.codebook_size,
        audio_eos_code=speech.audio_eos_code,
        maskgit_steps=speech.maskgit_steps,
        interleaved_n_text=checkpoint_config["interleaved_n_text"],
        interleaved_n_audio=checkpoint_config["interleaved_n_audio"],
        image_token_id=checkpoint_config.get("image_token_id"),
        video_token_id=checkpoint_config.get("video_token_id"),
    )


@dataclass(frozen=True, kw_only=True)
class SpeechSamplingDefaults:
    """The checkpoint's speech-frame sampling defaults, from its generation_config.json."""

    temperature: float
    top_k: int | None
    repetition_penalty: float
    repetition_window: int


def speech_defaults_from_checkpoint(checkpoint_dir: str) -> SpeechSamplingDefaults:
    generation_config = load_generation_config(checkpoint_dir)
    return SpeechSamplingDefaults(
        temperature=float(generation_config["audio_temperature"]),
        top_k=generation_config["audio_top_k"],
        repetition_penalty=float(generation_config["audio_repetition_penalty"]),
        repetition_window=int(generation_config["audio_repetition_window"]),
    )
