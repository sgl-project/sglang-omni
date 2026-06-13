# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Zyphra ZONOS2 TTS pipeline skeleton."""

from __future__ import annotations

_ZONOS2_NOT_IMPLEMENTED = (
    "ZONOS2 runtime support is not implemented yet. This package currently "
    "registers the pipeline shape for issue #775; the text frontend, speaker "
    "conditioning, multi-codebook MoE decode, and DAC vocoder still need to be "
    "ported before serving Zyphra/ZONOS2."
)


def _raise_not_implemented(*args, **kwargs):
    raise NotImplementedError(_ZONOS2_NOT_IMPLEMENTED)


def create_text_frontend_executor(*args, **kwargs):
    """Create the ZONOS2 text normalization/tokenization stage."""
    return _raise_not_implemented(*args, **kwargs)


def create_speaker_embedding_executor(*args, **kwargs):
    """Create the ZONOS2 speaker/reference conditioning stage."""
    return _raise_not_implemented(*args, **kwargs)


def create_sglang_tts_engine_executor(*args, **kwargs):
    """Create the ZONOS2 multi-codebook MoE decode stage."""
    return _raise_not_implemented(*args, **kwargs)


def create_dac_vocoder_executor(*args, **kwargs):
    """Create the ZONOS2 DAC vocoder stage."""
    return _raise_not_implemented(*args, **kwargs)
