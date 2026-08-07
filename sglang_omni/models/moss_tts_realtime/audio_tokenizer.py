# SPDX-License-Identifier: Apache-2.0
"""MOSS-Audio-Tokenizer loader and reference encoder."""

from __future__ import annotations

import base64
import logging
import re
from functools import cache
from pathlib import Path
from typing import Any

import torch

from sglang_omni.models.moss_tts_realtime.payload_types import N_CODEBOOKS, SAMPLE_RATE
from sglang_omni.preprocessing.audio import AudioMediaIO

logger = logging.getLogger(__name__)

DEFAULT_CODEC_MODEL = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
_DATA_URI_RE = re.compile(
    r"^data:(?P<media_type>audio/[^;,]+);base64,(?P<data>.+)$",
    re.DOTALL,
)


@cache
def load_moss_realtime_audio_tokenizer(
    model_path: str = DEFAULT_CODEC_MODEL,
    *,
    device: str = "cuda:0",
) -> Any:
    """Load one process-local codec instance per model and device."""
    from transformers import AutoModel

    logger.info(
        "Loading MOSS-TTS-Realtime audio tokenizer from %s on %s",
        model_path,
        device,
    )
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.float32,
    )
    return model.eval().to(device)


def load_reference_audio(reference: Any) -> torch.Tensor:
    """Decode a path, data URI, bytes, array, or tensor to mono 24 kHz PCM."""
    if isinstance(reference, torch.Tensor):
        waveform = reference.detach().to("cpu", torch.float32).reshape(-1)
        return waveform
    media = AudioMediaIO(target_sr=SAMPLE_RATE)
    if isinstance(reference, (bytes, bytearray)):
        audio, _ = media.load_bytes(bytes(reference))
    elif isinstance(reference, str):
        match = _DATA_URI_RE.match(reference)
        if match is not None:
            audio, _ = media.load_bytes(base64.b64decode(match.group("data")))
        else:
            audio, _ = media.load_file(Path(reference).expanduser())
    else:
        audio = reference
    return torch.as_tensor(audio, dtype=torch.float32).reshape(-1)


@torch.inference_mode()
def encode_reference_audio(codec: Any, reference: Any) -> torch.Tensor:
    waveform = load_reference_audio(reference)
    device = next(codec.parameters()).device
    encoded = codec.encode(
        waveform.reshape(1, 1, -1).to(device),
        num_quantizers=N_CODEBOOKS,
        return_dict=True,
    )
    codes = getattr(encoded, "audio_codes", None)
    if codes is None and isinstance(encoded, dict):
        codes = encoded.get("audio_codes")
    if not isinstance(codes, torch.Tensor):
        raise TypeError("MOSS audio tokenizer returned no reference codes")
    if codes.ndim != 3 or int(codes.shape[0]) < N_CODEBOOKS:
        raise RuntimeError(
            f"MOSS audio tokenizer returned invalid codes {tuple(codes.shape)}"
        )
    return (
        codes[:N_CODEBOOKS, 0]
        .transpose(0, 1)
        .detach()
        .to(device="cpu", dtype=torch.long)
        .contiguous()
    )


__all__ = [
    "DEFAULT_CODEC_MODEL",
    "encode_reference_audio",
    "load_moss_realtime_audio_tokenizer",
    "load_reference_audio",
]
