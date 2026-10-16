# SPDX-License-Identifier: Apache-2.0
"""The libraries the HTTP server decodes and encodes audio with.

The speech routes import them on first use. A pipeline that serves audio
loads them before its stages start, so no request pays for the import, and a
pipeline that does not serve audio never loads them.
"""

from __future__ import annotations

import importlib

AUDIO_STACK_MODULES = ("torch", "torchaudio", "av", "soundfile")
REALTIME_STACK_MODULES = ("silero_vad",)


def preload_audio_stack(*, realtime: bool) -> None:
    """Import the audio libraries and resolve the platform the decoders ask."""
    from sglang_omni import platforms

    for name in AUDIO_STACK_MODULES:
        importlib.import_module(name)
    if realtime:
        for name in REALTIME_STACK_MODULES:
            importlib.import_module(name)
    else:
        pass
    platforms.get_current_platform()
