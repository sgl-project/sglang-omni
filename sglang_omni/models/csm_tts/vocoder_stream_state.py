# SPDX-License-Identifier: Apache-2.0
"""Per-request streaming state for the CSM vocoder and its codebook contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from sglang_omni.models.csm_tts.utils import CODEBOOK_VOCAB, NUM_CODEBOOKS
from sglang_omni.scheduling.streaming_vocoder import (
    INITIAL_CODEC_CHUNK_FRAMES_PARAM,
    resolve_initial_codec_chunk_frames,
)


@dataclass
class CsmVocoderStreamState:
    """Code rows received so far and how many frames were already vocoded."""

    code_rows: list[torch.Tensor] = field(default_factory=list)
    frames_emitted: int = 0
    num_codebooks: int | None = None
    codebook_size: int | None = None
    initial_codec_chunk_frames: int | None = None


def as_contract_int(raw: object, default: int, field_name: str, source: str) -> int:
    """One codebook-contract integer out of a wire mapping."""
    if raw is None:
        return default
    elif isinstance(raw, bool):
        raise TypeError(f"CSM {source} {field_name} must be an integer, got {raw!r}")
    elif isinstance(raw, int):
        return raw
    elif isinstance(raw, str) and raw.isdigit():
        return int(raw)
    else:
        raise TypeError(f"CSM {source} {field_name} must be an integer, got {raw!r}")


def latch_stream_contract(
    request_id: str,
    state: CsmVocoderStreamState,
    contract: Mapping[str, object],
    *,
    source: str,
) -> None:
    """Pin the codebook contract for one stream and reject any later change."""
    num_codebooks = as_contract_int(
        contract.get("num_codebooks"),
        state.num_codebooks or NUM_CODEBOOKS,
        "num_codebooks",
        source,
    )
    codebook_size = as_contract_int(
        contract.get("codebook_size"),
        state.codebook_size or CODEBOOK_VOCAB,
        "codebook_size",
        source,
    )
    if num_codebooks != NUM_CODEBOOKS or codebook_size != CODEBOOK_VOCAB:
        raise ValueError(
            f"CSM {source} for {request_id!r} has invalid contract "
            f"num_codebooks={num_codebooks}, codebook_size={codebook_size}; "
            f"expected {NUM_CODEBOOKS}/{CODEBOOK_VOCAB}"
        )
    else:
        pass
    if state.num_codebooks is not None and state.num_codebooks != num_codebooks:
        raise ValueError(
            f"CSM stream num_codebooks changed for {request_id!r}: "
            f"{state.num_codebooks} -> {num_codebooks}"
        )
    else:
        pass
    if state.codebook_size is not None and state.codebook_size != codebook_size:
        raise ValueError(
            f"CSM stream codebook_size changed for {request_id!r}: "
            f"{state.codebook_size} -> {codebook_size}"
        )
    else:
        pass
    state.num_codebooks = num_codebooks
    state.codebook_size = codebook_size


def latch_initial_codec_chunk_frames(
    state: CsmVocoderStreamState,
    params: Mapping[str, object] | None,
    *,
    steady_chunk_frames: int,
) -> None:
    """Pin the shorter first chunk the caller asked for, else keep the steady stride."""
    if params is not None and INITIAL_CODEC_CHUNK_FRAMES_PARAM in params:
        state.initial_codec_chunk_frames = resolve_initial_codec_chunk_frames(
            params, steady_chunk_frames=steady_chunk_frames
        )
    elif state.initial_codec_chunk_frames is None:
        state.initial_codec_chunk_frames = 0
    else:
        pass


def latch_stream_metadata(
    request_id: str,
    state: CsmVocoderStreamState,
    metadata: Mapping[str, str | int | bool] | None,
    *,
    steady_chunk_frames: int,
) -> None:
    """Validate one chunk's metadata and fold its contract into the stream state."""
    if not isinstance(metadata, Mapping):
        latch_stream_contract(request_id, state, {}, source="stream metadata defaults")
        return
    else:
        pass
    if metadata.get("modality") not in (None, "audio_codes"):
        raise ValueError(
            f"CSM stream chunk modality must be audio_codes, got "
            f"{metadata.get('modality')!r}"
        )
    else:
        pass
    if metadata.get("stream") is not True:
        raise RuntimeError(
            f"CSM stream chunk for {request_id!r} must include "
            "metadata['stream'] == True"
        )
    else:
        pass
    latch_stream_contract(request_id, state, metadata, source="stream metadata")
    if INITIAL_CODEC_CHUNK_FRAMES_PARAM in metadata:
        latch_initial_codec_chunk_frames(
            state, metadata, steady_chunk_frames=steady_chunk_frames
        )
    else:
        pass


__all__ = [
    "CsmVocoderStreamState",
    "latch_initial_codec_chunk_frames",
    "latch_stream_contract",
    "latch_stream_metadata",
]
