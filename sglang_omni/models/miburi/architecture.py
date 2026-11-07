# SPDX-License-Identifier: Apache-2.0
"""Shapes, windows and special codes of the released gesture checkpoint.

The checkpoint's config.yaml files are training configs: they carry the layer and head counts but not the widths, windows or code layout, which are collected here once. The defaults are the release values; tests build tiny models from the same specs.
"""

from __future__ import annotations

from dataclasses import dataclass

# History of a session before its first frame; embeds to exactly zero.
INITIAL_CODE = -1
# Text and audio ids of the unconditional guidance rows; embed to exactly zero.
NULL_CONDITION_ID = -1
# Mimi codebooks per agent frame that condition the gesture LM, beside the text token.
SPEECH_AUDIO_CODEBOOKS = 8


@dataclass(frozen=True, kw_only=True)
class GestureLMSpec:
    """The gesture LM: a temporal stack that predicts codebook 0 per frame and a depth stack that predicts the rest, one codebook per step."""

    # Codebooks per body part, in emission order: upper body with hands, lower body with the root, face.
    codebooks_per_part: tuple[int, int, int] = (8, 8, 4)
    # Codes per codebook; the pad code is one past the last and has its own embedding.
    codebook_size: int = 2048
    dim: int = 256
    codebook_dim: int = 128
    condition_dim: int = 4096
    text_vocab: int = 32001
    audio_vocab: int = 2049
    # Hidden width of the gated feed-forwards, 21/8 of dim for their nominal 4x expansion.
    ffn_dim: int = 672
    temporal_layers: int = 4
    temporal_heads: int = 2
    depth_layers: int = 2
    depth_heads: int = 1
    # Frames of the temporal self-attention and of its attention over the speech memories (2 s and 4 s).
    self_context: int = 25
    memory_context: int = 50
    num_speakers: int = 101
    rope_max_period: float = 10_000.0

    @property
    def num_codebooks(self) -> int:
        return sum(self.codebooks_per_part)

    @property
    def depth_steps(self) -> int:
        return self.num_codebooks - 1

    @property
    def pad_code(self) -> int:
        return self.codebook_size

    @property
    def lower_body_codebooks(self) -> range:
        """Codebooks whose depth steps attend to no speech memory and skip guidance: the lower body relates little to speech."""
        start = self.codebooks_per_part[0]
        return range(start, start + self.codebooks_per_part[1])


@dataclass(frozen=True, kw_only=True)
class MotionCodecSpec:
    """One body part's streaming decoder: residual codebooks, a causal transformer and a SEANet decoder that doubles the frame rate."""

    features: int
    codebooks: int
    num_layers: int
    num_heads: int
    dim: int = 256
    codebook_dim: int = 128
    codebook_size: int = 2048
    ffn_dim: int = 2048
    # Latent steps the transformer attends to, the current one included.
    context: int = 25
    layer_scale: float = 0.01
    rope_max_period: float = 10_000.0
    n_filters: int = 256
    # Motion frames per code.
    upsample_ratio: int = 2
    residual_blocks: int = 2
    kernel_size: int = 7
    last_kernel_size: int = 7
    residual_kernel_size: int = 3
    dilation_base: int = 2
    compress: int = 2


__all__ = [
    "INITIAL_CODE",
    "NULL_CONDITION_ID",
    "SPEECH_AUDIO_CODEBOOKS",
    "GestureLMSpec",
    "MotionCodecSpec",
]
