# SPDX-License-Identifier: Apache-2.0
"""What a conversation opens with: a voice, and a role in text.

PersonaPlex ships 18 voices as .pt files inside voices.tgz. Each holds
the fused input rows the reference stepped through while "listening" to that
voice, so a packaged voice costs no codec pass. A voice may also be a
recording, which is loudness-normalised and Mimi-encoded like the reference
does. The role prompt is SentencePiece text between <system> tags.
"""

from __future__ import annotations

import math
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sglang_omni.models.personaplex.architecture import (
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    SYSTEM_TAG,
    TEXT_MARKER_IDS,
)
from sglang_omni.models.personaplex.timeline import voice_tail_codes_from_cache

TEXT_TOKENIZER_NAME = "tokenizer_spm_32k_3.model"
VOICES_ARCHIVE_NAME = "voices.tgz"
VOICES_DIR_NAME = "voices"
VOICE_PROMPT_TARGET_LUFS = -24.0
_VOICE_SUFFIXES = (".pt", ".wav", ".flac", ".mp3", ".ogg")

DEFAULT_TEXT_PROMPT = (
    "You are a wise and friendly teacher. Answer questions or provide advice "
    "in a clear and engaging way."
)
DEFAULT_VOICE = "NATF2"


def load_text_tokenizer(model_dir: str | Path):
    import sentencepiece

    path = Path(model_dir) / TEXT_TOKENIZER_NAME
    if not path.is_file():
        raise FileNotFoundError(f"PersonaPlex text tokenizer missing: {path}")
    return sentencepiece.SentencePieceProcessor(model_file=str(path))


def wrap_system_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith(SYSTEM_TAG) and cleaned.endswith(SYSTEM_TAG):
        return cleaned
    return f"{SYSTEM_TAG} {cleaned} {SYSTEM_TAG}"


def tokenize_text_prompt(tokenizer, text: str | None) -> list[int]:
    if not text or not text.strip():
        return []
    return [int(i) for i in tokenizer.encode(wrap_system_tags(text))]


def decode_text(tokenizer, token_ids: list[int]) -> str:
    """Spoken words only; the frame-locked markers (PAD, EPAD, BOS, EOS) are dropped."""
    spoken = [int(i) for i in token_ids if int(i) not in TEXT_MARKER_IDS]
    return tokenizer.decode(spoken) if spoken else ""


@dataclass
class VoicePrompt:
    """Either embeddings (packaged voice) or waveform (a recording)."""

    frames: int
    embeddings: torch.Tensor | None = None
    tail_codes: torch.Tensor | None = None
    waveform: torch.Tensor | None = None


def _unpack_voices(archive: Path, parent: Path) -> Path:
    """parent/voices, unpacked from archive unless it already exists.

    The archive is unpacked into a staging folder and renamed into place, so an
    interrupted unpack never leaves a partial voices/ behind.
    """
    target = parent / VOICES_DIR_NAME
    if target.is_dir():
        return target
    staging = Path(tempfile.mkdtemp(prefix=f".{VOICES_DIR_NAME}-", dir=parent))
    try:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=staging, filter="data")
        unpacked = staging / VOICES_DIR_NAME
        if not unpacked.is_dir():
            raise RuntimeError(
                f"{archive} did not contain a {VOICES_DIR_NAME}/ directory"
            )
        try:
            unpacked.rename(target)
        except OSError:
            if not target.is_dir():
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def voices_dir(model_dir: str | Path) -> Path:
    """The extracted voices/ folder, unpacking voices.tgz on first use.

    It is unpacked next to the checkpoint, or into the temp directory when the
    checkpoint folder cannot be written (permissions, a read-only mount).
    """
    model_dir = Path(model_dir)
    extracted = model_dir / VOICES_DIR_NAME
    if extracted.is_dir():
        return extracted
    archive = model_dir / VOICES_ARCHIVE_NAME
    if not archive.is_file():
        raise FileNotFoundError(
            f"no {VOICES_DIR_NAME}/ or {VOICES_ARCHIVE_NAME} under {model_dir}; "
            "pass a voice prompt path instead of a voice name"
        )
    try:
        return _unpack_voices(archive, model_dir)
    except OSError:
        fallback = (
            Path(tempfile.gettempdir())
            / f"sglang-omni-personaplex-{archive.stat().st_ino}"
        )
        fallback.mkdir(exist_ok=True)
        return _unpack_voices(archive, fallback)


def resolve_voice_path(model_dir: str | Path, voice: str) -> Path:
    """A voice is a file path, or the name of a packaged voice (NATF2)."""
    direct = Path(voice).expanduser()
    if direct.is_file():
        return direct
    folder = voices_dir(model_dir)
    candidates = [
        folder / voice,
        *(folder / f"{voice}{suffix}" for suffix in _VOICE_SUFFIXES),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    available = sorted(p.stem for p in folder.glob("*.pt"))
    raise FileNotFoundError(f"unknown voice {voice!r}; packaged voices: {available}")


def normalize_loudness(
    waveform: np.ndarray, sample_rate: int, target_lufs: float
) -> np.ndarray:
    try:
        import pyloudnorm
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "a recorded voice prompt needs `pip install pyloudnorm` for the "
            "reference loudness normalisation; packaged .pt voices do not"
        ) from exc
    meter = pyloudnorm.Meter(sample_rate)
    loudness = meter.integrated_loudness(waveform)
    return pyloudnorm.normalize.loudness(waveform, loudness, target_lufs)


def pad_to_whole_frames(waveform: torch.Tensor) -> torch.Tensor:
    remainder = waveform.shape[-1] % SAMPLES_PER_FRAME
    if remainder:
        waveform = torch.nn.functional.pad(waveform, (0, SAMPLES_PER_FRAME - remainder))
    return waveform


def load_voice_prompt(path: str | Path, *, load_audio) -> VoicePrompt:
    """load_audio(path) must return the recording as [channels, samples]
    float at 24 kHz; only the first channel is the voice."""
    path = Path(path)
    if path.suffix == ".pt":
        saved = torch.load(path, map_location="cpu", weights_only=True)
        embeddings = saved["embeddings"]
        embeddings = embeddings.reshape(embeddings.shape[0], -1).to(torch.float32)
        frames = int(embeddings.shape[0]) + 1
        return VoicePrompt(
            frames=frames,
            embeddings=embeddings,
            tail_codes=voice_tail_codes_from_cache(
                saved["cache"].to(torch.long), frames
            ),
        )
    channels = np.asarray(load_audio(str(path)), dtype=np.float32)
    mono = channels[0] if channels.ndim == 2 else channels
    mono = normalize_loudness(
        mono.astype(np.float64), SAMPLE_RATE, VOICE_PROMPT_TARGET_LUFS
    )
    waveform = pad_to_whole_frames(torch.as_tensor(mono, dtype=torch.float32))
    return VoicePrompt(
        frames=math.ceil(waveform.shape[-1] / SAMPLES_PER_FRAME), waveform=waveform
    )


__all__ = [
    "DEFAULT_TEXT_PROMPT",
    "DEFAULT_VOICE",
    "VoicePrompt",
    "decode_text",
    "load_text_tokenizer",
    "load_voice_prompt",
    "normalize_loudness",
    "pad_to_whole_frames",
    "resolve_voice_path",
    "tokenize_text_prompt",
    "voices_dir",
    "wrap_system_tags",
]
