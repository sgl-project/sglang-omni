# SPDX-License-Identifier: Apache-2.0
"""Vendored Ming TTS talker components."""

from .audio_vae.modeling_audio_vae import AudioVAE
from .configuration_bailing_talker import (
    MingOmniTalkerConfig,
    normalize_max_conc,
    resolve_talker_max_conc,
)
from .modeling_ming_omni_talker import MingOmniTalker, SpkembExtractor

__all__ = [
    "MingOmniTalker",
    "MingOmniTalkerConfig",
    "SpkembExtractor",
    "AudioVAE",
    "normalize_max_conc",
    "resolve_talker_max_conc",
]
