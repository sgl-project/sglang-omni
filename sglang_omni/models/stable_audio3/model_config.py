# SPDX-License-Identifier: Apache-2.0
"""The subset of a Stable Audio 3 model_config.json that serving reads; unsupported layouts fail validation."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

MODEL_CONFIG_FILENAME = "model_config.json"
# note (ruilin): the Medium checkpoint's sample_size, 16777216 samples at 44.1 kHz.
MAX_DURATION_SECONDS = 380.0
# note (ruilin): the Medium model_config.json omits the encoder chunk_size, so the encoder runs at this default.
DEFAULT_ENCODER_CHUNK_SIZE = 32


class AttentionOptions(BaseModel):
    qk_norm: Literal["rms"]
    differential: Literal[True]


class NormOptions(BaseModel):
    force_fp32: Literal[True]


class FeedForwardOptions(BaseModel):
    mult: float


class DiTOptions(BaseModel):
    io_channels: int
    embed_dim: int
    depth: int
    num_heads: int
    cond_token_dim: int
    global_cond_dim: int
    local_add_cond_dim: int
    num_memory_tokens: int
    global_cond_type: Literal["adaLN"]
    timestep_features_type: Literal["expo"]
    norm_type: Literal["rms_norm"]
    attn_kwargs: AttentionOptions
    norm_kwargs: NormOptions
    ff_kwargs: FeedForwardOptions


class DiffusionSection(BaseModel):
    type: Literal["dit"]
    diffusion_objective: Literal["rf_denoiser"]
    sampling_distribution_shift_options: None = None
    config: DiTOptions


class EncoderOptions(BaseModel):
    strides: list[int] = Field(min_length=1, max_length=1)
    chunk_size: int = DEFAULT_ENCODER_CHUNK_SIZE


class DecoderOptions(BaseModel):
    out_channels: int
    channels: int
    c_mults: list[int] = Field(min_length=1, max_length=1)
    strides: list[int] = Field(min_length=1, max_length=1)
    latent_dim: int
    transformer_depths: list[int] = Field(min_length=1, max_length=1)
    sinusoidal_blocks: list[int] = Field(min_length=1, max_length=1)
    sliding_window: tuple[Literal[1], Literal[1]]
    dim_heads: int
    differential: Literal[True]
    variable_stride: Literal[True]
    dyt: Literal[True] = True
    mask_noise: float


class EncoderSection(BaseModel):
    config: EncoderOptions


class DecoderSection(BaseModel):
    type: Literal["taae_v2"]
    config: DecoderOptions


class BottleneckOptions(BaseModel):
    noise_augment_dim: Literal[0]
    noise_regularize: Literal[True]
    auto_scale: Literal[True]


class BottleneckSection(BaseModel):
    type: Literal["softnorm"]
    config: BottleneckOptions


class PatchOptions(BaseModel):
    patch_size: int
    channels: int


class PatchSection(BaseModel):
    type: Literal["patched"]
    config: PatchOptions


class AutoencoderOptions(BaseModel):
    pretransform: PatchSection
    encoder: EncoderSection
    decoder: DecoderSection
    bottleneck: BottleneckSection
    latent_dim: int
    downsampling_ratio: int


class PretransformSection(BaseModel):
    type: Literal["autoencoder"]
    chunked: Literal[True]
    config: AutoencoderOptions


class PromptConditionerOptions(BaseModel):
    max_length: int
    padding_mode: Literal["learned"]
    subfolder: str


class PromptConditionerSection(BaseModel):
    id: Literal["prompt"]
    type: Literal["t5gemma"]
    config: PromptConditionerOptions


class SecondsConditionerOptions(BaseModel):
    min_val: float
    max_val: float
    fourier_features_type: Literal["expo"]


class SecondsConditionerSection(BaseModel):
    id: Literal["seconds_total"]
    type: Literal["number"]
    config: SecondsConditionerOptions


class ConditioningSection(BaseModel):
    configs: tuple[PromptConditionerSection, SecondsConditionerSection]
    cond_dim: int


class ModelSection(BaseModel):
    pretransform: PretransformSection
    conditioning: ConditioningSection
    diffusion: DiffusionSection


class StableAudio3Config(BaseModel):
    model_type: Literal["diffusion_cond_inpaint"]
    sample_size: int
    sample_rate: int
    audio_channels: int
    model: ModelSection

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: Path) -> "StableAudio3Config":
        try:
            return cls.model_validate_json(
                (checkpoint_dir / MODEL_CONFIG_FILENAME).read_text()
            )
        except ValidationError as error:
            raise ValueError(
                f"{checkpoint_dir} is not the Stable Audio 3 Medium layout this server runs: {error}"
            ) from error
