# SPDX-License-Identifier: Apache-2.0
"""EasyMagpie preset voices.

Each voice is a fixed block of prompt rows. The engine keeps every voice on
the GPU in one table, so requests carry only the voice name and its frame
count instead of the rows themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

SPEAKER_SUBDIR = "speaker_embeddings"


def load_speaker_embeddings(
    checkpoint: Path, embedding_dim: int
) -> dict[str, torch.Tensor]:
    """Load every preset voice as a [frames, embedding_dim] float16 tensor."""
    voices = {}
    for path in sorted((checkpoint / SPEAKER_SUBDIR).glob("*.pt")):
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(loaded, dict):
            embedding = loaded["speaker_encoding"]
        else:
            embedding = loaded
        if embedding.ndim != 2 or embedding.shape[1] != embedding_dim:
            raise ValueError(
                f"EasyMagpie speaker embedding {path} must be [frames, {embedding_dim}]"
            )
        else:
            pass
        voices[path.stem] = embedding.detach().to(torch.float16)
    if not voices:
        raise ValueError(
            f"No EasyMagpie voices found under {checkpoint / SPEAKER_SUBDIR}"
        )
    else:
        pass
    return voices


@dataclass
class SpeakerTable:
    """Every voice's rows back to back, with each voice's first row and length."""

    rows: torch.Tensor
    spans: dict[str, tuple[int, int]]

    @classmethod
    def from_voices(
        cls,
        voices: dict[str, torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SpeakerTable:
        spans, start = {}, 0
        for name, embedding in voices.items():
            spans[name] = (start, int(embedding.shape[0]))
            start += int(embedding.shape[0])
        rows = torch.cat(list(voices.values()), dim=0).to(device=device, dtype=dtype)
        return cls(rows=rows, spans=spans)


__all__ = ["SPEAKER_SUBDIR", "SpeakerTable", "load_speaker_embeddings"]
