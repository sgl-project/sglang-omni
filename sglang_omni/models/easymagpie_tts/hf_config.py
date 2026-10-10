# SPDX-License-Identifier: Apache-2.0
"""EasyMagpie checkpoint config and weight-name contract.

Kept free of SGLang imports so preprocessing, the talker, and the codec
stage read the stacked-codebook layout from one place.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch

NUM_AUDIO_SPECIAL_TOKENS = 8
AUDIO_BOS_OFFSET = 0
AUDIO_EOS_OFFSET = 1

CHECKPOINT_BACKBONE_PREFIX = "decoder."
SGLANG_BACKBONE_PREFIX = "backbone."
CODEC_WEIGHT_PREFIX = "audio_decoder."

# Converted-config spelling -> SGLang NemotronHConfig keyword. SGLang already
# normalises chunk_size and hybrid_override_pattern itself.
LEGACY_BACKBONE_KEYS = {
    "n_groups": "mamba_n_groups",
    "conv_kernel": "mamba_d_conv",
    "use_conv_bias": "mamba_conv_bias",
}

REQUIRED_TTS_KEYS = (
    "text_vocab_size",
    "text_eos_id",
    "embedding_dim",
    "audio_embedding_dim",
    "num_audio_codebooks",
    "codebook_size",
    "frame_stacking_factor",
    "phoneme_stacking_factor",
    "phoneme_vocab_size",
    "phoneme_bos_id",
    "phoneme_eos_id",
    "phoneme_unk_id",
    "streaming_phonemes_delay",
    "streaming_speech_delay",
    "local_transformer_n_layers",
    "local_transformer_n_heads",
    "local_transformer_hidden_dim",
)


@dataclass(frozen=True)
class EasyMagpieTTSConfig:
    """TTS fields carried alongside the Nemotron-H backbone config."""

    text_vocab_size: int
    text_eos_id: int
    embedding_dim: int
    audio_embedding_dim: int
    num_audio_codebooks: int
    codebook_size: int
    frame_stacking_factor: int
    phoneme_stacking_factor: int
    phoneme_vocab_size: int
    phoneme_bos_id: int
    phoneme_eos_id: int
    phoneme_unk_id: int
    streaming_phonemes_delay: int
    streaming_speech_delay: int
    local_transformer_n_layers: int
    local_transformer_n_heads: int
    local_transformer_hidden_dim: int
    phoneme_confidence_unk_threshold: float = 0.0
    forced_audio_bos_id: int | None = None
    forced_audio_eos_id: int | None = None

    @property
    def num_stacked_codebooks(self) -> int:
        return self.num_audio_codebooks * self.frame_stacking_factor

    @property
    def codebook_vocab_size(self) -> int:
        return self.codebook_size + NUM_AUDIO_SPECIAL_TOKENS

    @property
    def audio_bos_id(self) -> int:
        if self.forced_audio_bos_id is not None:
            return self.forced_audio_bos_id
        else:
            return self.codebook_size + AUDIO_BOS_OFFSET

    @property
    def audio_eos_id(self) -> int:
        if self.forced_audio_eos_id is not None:
            return self.forced_audio_eos_id
        else:
            return self.codebook_size + AUDIO_EOS_OFFSET

    @classmethod
    def from_dict(cls, raw: dict) -> EasyMagpieTTSConfig:
        missing = [key for key in REQUIRED_TTS_KEYS if key not in raw]
        if missing:
            raise ValueError(f"EasyMagpie config is missing: {', '.join(missing)}")
        else:
            pass
        values = {key: int(raw[key]) for key in REQUIRED_TTS_KEYS}
        config = cls(
            **values,
            phoneme_confidence_unk_threshold=float(
                raw.get("phoneme_confidence_unk_threshold", 0.0)
            ),
            forced_audio_bos_id=optional_int(raw.get("forced_audio_bos_id")),
            forced_audio_eos_id=optional_int(raw.get("forced_audio_eos_id")),
        )
        if config.num_stacked_codebooks < 1 or config.codebook_size < 1:
            raise ValueError("EasyMagpie needs positive codebook dimensions")
        else:
            pass
        if config.embedding_dim != config.audio_embedding_dim:
            raise ValueError(
                "EasyMagpie needs matching text and audio embedding widths, got "
                f"{config.embedding_dim} and {config.audio_embedding_dim}"
            )
        else:
            pass
        return config


def optional_int(value: int | str | None) -> int | None:
    if value is None:
        return None
    else:
        return int(value)


def adapt_backbone_config(raw: dict) -> dict:
    """Return kwargs that SGLang's NemotronHConfig reads correctly.

    Legacy Mamba keys would otherwise land in kwargs while the canonical
    attribute silently keeps its default. ``raw`` is normally the ``to_dict()``
    of an already-built config, where the canonical key always holds that
    default, so the legacy value must win.
    """
    adapted = dict(raw)
    for legacy, canonical in LEGACY_BACKBONE_KEYS.items():
        if legacy in raw:
            adapted[canonical] = raw[legacy]
        else:
            pass
    # note (Yashwant Hayaran): the checkpoint stores one unindexed shared expert
    # per MoE layer, and NemotronHMoE reads n_shared_experts without a default.
    adapted.setdefault("n_shared_experts", 1)
    # The backbone vocabulary is a dummy continue/stop pair. The inherited
    # transformers EOS of 2 is outside it and could never stop a request.
    eos_token_id = adapted.get("eos_token_id")
    vocab_size = int(adapted.get("vocab_size", 0))
    if vocab_size == 2 and not (
        isinstance(eos_token_id, int) and 0 <= eos_token_id < vocab_size
    ):
        adapted["eos_token_id"] = 1
    else:
        pass
    return adapted


def adapt_head_weight(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Squeeze kernel-1 Conv1d local-transformer weights into Linear shape."""
    if (
        name.startswith("local_transformer.")
        and ".conv.weight" in name
        and tensor.ndim == 3
        and tensor.shape[-1] == 1
    ):
        return tensor.squeeze(-1)
    else:
        return tensor


def partition_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    head_weights: list[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield backbone weights with the SGLang prefix; collect head weights.

    The split must happen before SGLang's loader runs: its substring rename of
    embeddings to embed_tokens would corrupt audio_embeddings and
    phoneme_embeddings. Codec weights are dropped; the codec stage owns them.
    """
    for name, tensor in weights:
        if name.startswith(CHECKPOINT_BACKBONE_PREFIX):
            suffix = name[len(CHECKPOINT_BACKBONE_PREFIX) :]
            yield SGLANG_BACKBONE_PREFIX + suffix, tensor
        elif name.startswith(CODEC_WEIGHT_PREFIX):
            continue
        else:
            head_weights.append((name, adapt_head_weight(name, tensor)))


__all__ = [
    "EasyMagpieTTSConfig",
    "adapt_backbone_config",
    "adapt_head_weight",
    "partition_weights",
]
