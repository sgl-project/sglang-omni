# SPDX-License-Identifier: Apache-2.0
"""Voice preset manifest loading for the Ming-Omni talker."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)

BUNDLED_VOICE_DIR = Path(__file__).resolve().parent.parent / "voices"


class VoicePreset(TypedDict):
    prompt_text: str
    prompt_wav_path: str


def load_voice_presets(
    talker_dir: str, default_voice: str | None
) -> dict[str, VoicePreset]:
    """Load and validate voice presets, resolving each prompt wav to an absolute path.

    Reads talker_dir/data/voice_name.json, whose wav paths are relative to talker_dir.
    Checkpoints without that manifest fall back to the presets bundled with sglang-omni.
    """
    checkpoint_manifest_path = Path(talker_dir) / "data" / "voice_name.json"
    if checkpoint_manifest_path.is_file():
        manifest_path = checkpoint_manifest_path
        wav_root = Path(talker_dir)
    else:
        manifest_path = BUNDLED_VOICE_DIR / "voice_name.json"
        wav_root = BUNDLED_VOICE_DIR
        logger.info(
            f"No voice manifest at {checkpoint_manifest_path}; using the bundled presets in {manifest_path}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Voice manifest {manifest_path} must be a JSON object")
    else:
        pass
    if default_voice is not None and default_voice not in manifest:
        raise ValueError(
            f"Default voice {default_voice!r} not found in {manifest_path}; available presets: {sorted(manifest)}. Install a manifest with this voice at {checkpoint_manifest_path}."
        )
    else:
        pass

    voice_presets: dict[str, VoicePreset] = {}
    for voice_name, entry in manifest.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"Voice preset {voice_name!r} in {manifest_path} must be a JSON object"
            )
        else:
            pass
        prompt_text = entry.get("prompt_text")
        relative_wav_path = entry.get("prompt_wav_path")
        if not isinstance(prompt_text, str):
            raise ValueError(
                f"Voice preset {voice_name!r} in {manifest_path} is missing prompt_text"
            )
        elif not isinstance(relative_wav_path, str):
            raise ValueError(
                f"Voice preset {voice_name!r} in {manifest_path} is missing prompt_wav_path"
            )
        else:
            pass
        wav_path = wav_root / relative_wav_path
        if not wav_path.is_file():
            raise FileNotFoundError(
                f"Voice preset {voice_name!r} in {manifest_path} references missing prompt wav {wav_path}"
            )
        else:
            pass
        voice_presets[voice_name] = VoicePreset(
            prompt_text=prompt_text, prompt_wav_path=str(wav_path)
        )
    return voice_presets
