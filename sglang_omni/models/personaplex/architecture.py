# SPDX-License-Identifier: Apache-2.0
"""Shapes and special tokens of the PersonaPlex checkpoint family.

PersonaPlex is a fine-tune of Moshi 7B, so the checkpoint ships no HF config:
the reference loader hard-codes every dimension. They are collected here, once,
so the Mimi codec, the temporal transformer shim and the depformer all read
from the same place.
"""

from __future__ import annotations

from dataclasses import dataclass

SAMPLE_RATE = 24_000
FRAME_RATE = 12.5
SAMPLES_PER_FRAME = int(SAMPLE_RATE / FRAME_RATE)

AUDIO_CODEBOOKS_PER_STREAM = 8
NUM_AUDIO_STREAMS = 2 * AUDIO_CODEBOOKS_PER_STREAM
NUM_STREAMS = 1 + NUM_AUDIO_STREAMS
AGENT_STREAM_OFFSET = 1
USER_STREAM_OFFSET = 1 + AUDIO_CODEBOOKS_PER_STREAM

AUDIO_CARD = 2048
TEXT_CARD = 32_000
# Note (wilsonzheng0327): One past the vocabulary: the token every stream starts from.
AUDIO_INITIAL_ID = AUDIO_CARD
TEXT_INITIAL_ID = TEXT_CARD

TEXT_EPAD_ID = 0
TEXT_BOS_ID = 1
TEXT_EOS_ID = 2
TEXT_PAD_ID = 3
TEXT_MARKER_IDS = frozenset({TEXT_EPAD_ID, TEXT_BOS_ID, TEXT_EOS_ID, TEXT_PAD_ID})

# Note (wilsonzheng0327): Text, then agent and user codebooks; each side's acoustic
# codebooks lag its first codebook by one frame.
DELAYS = (0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1)
MAX_DELAY = max(DELAYS)

# Note (wilsonzheng0327): Mimi codes of a silent frame and of a 440 Hz sine, forced
# during the prompt.
SILENCE_CODES = (948, 243, 1178, 546, 1736, 1030, 1978, 2008)
SINE_CODES = (430, 1268, 381, 1611, 1095, 1495, 56, 472)

PROMPT_SILENCE_FRAMES = int(0.5 * FRAME_RATE)

SYSTEM_TAG = "<system>"

DEFAULT_TEXT_TEMPERATURE = 0.7
DEFAULT_TEXT_TOP_K = 25
DEFAULT_AUDIO_TEMPERATURE = 0.8
DEFAULT_AUDIO_TOP_K = 250

MOSHI_WEIGHTS_NAME = "model.safetensors"
MIMI_WEIGHTS_GLOB = "tokenizer-*.safetensors"


@dataclass(frozen=True)
class TemporalTransformerSpec:
    """The Helium-style backbone: Llama-shaped, with interleaved RoPE."""

    dim: int = 4096
    num_heads: int = 32
    num_layers: int = 32
    ffn_hidden: int = (2 * int(4.125 * 4096)) // 3
    rms_norm_eps: float = 1e-8
    rope_max_period: float = 10_000.0
    context: int = 3000

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads


@dataclass(frozen=True)
class DepformerSpec:
    """The per-frame depth transformer that spells out the 8 agent codebooks."""

    dim: int = 1024
    num_heads: int = 16
    num_layers: int = 6
    ffn_hidden: int = (2 * int(4.125 * 1024)) // 3
    rms_norm_eps: float = 1e-8
    # Note (wilsonzheng0327): The checkpoint has 16 steps (agent, then user). The
    # user codes are always given and agent steps never attend to them: load 8.
    steps: int = AUDIO_CODEBOOKS_PER_STREAM
    input_dim: int = TemporalTransformerSpec.dim

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads


@dataclass(frozen=True)
class MimiSpec:
    dim: int = 512
    n_filters: int = 64
    # Note (wilsonzheng0327): Decoder order; the encoder strides in reverse.
    ratios: tuple[int, ...] = (8, 6, 5, 4)
    kernel_size: int = 7
    last_kernel_size: int = 3
    residual_kernel_size: int = 3
    compress: int = 2
    num_heads: int = 8
    num_layers: int = 8
    ffn_dim: int = 2048
    layer_norm_eps: float = 1e-5
    layer_scale: float = 0.01
    rope_max_period: float = 10_000.0
    context: int = 250
    codebook_dim: int = 256
    codebook_size: int = AUDIO_CARD
    num_semantic_codebooks: int = 1
    num_codebooks: int = AUDIO_CODEBOOKS_PER_STREAM
    frame_ratio: int = 2

    @property
    def hop_length(self) -> int:
        length = 1
        for ratio in self.ratios:
            length *= ratio
        return length


TEMPORAL_TRANSFORMER = TemporalTransformerSpec()
DEPFORMER = DepformerSpec()
MIMI = MimiSpec()

assert MIMI.hop_length * MIMI.frame_ratio == SAMPLES_PER_FRAME
