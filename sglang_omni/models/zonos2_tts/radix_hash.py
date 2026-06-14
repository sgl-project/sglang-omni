# SPDX-License-Identifier: Apache-2.0
"""Capture-safe GPU radix-key hash for ZONOS2 TTS generated frames.

The scheduler appends one radix-cache token id per generated frame to a
request's KV chain, and the radix tree keys on those ids. For ZONOS2's
multi-codebook output, the first codebook token alone is insufficient as a
radix key because different requests may generate the same first codebook
token. We hash the full multi-channel row (n_codebooks audio codes + text
column) so that a radix match implies identical audio content.

Prompt rows are hashed once, off the decode hot path, by
``build_row_cache_key_ids`` (host-side blake2b); that call never runs inside
a CUDA-graph capture region. The *generated*-row key, by contrast, is
computed every decode step on a device tensor, so it uses a fixed-coefficient
polynomial hash entirely in int64 torch ops -- no host sync, CUDA-graph
capturable.

Adapted from sglang_omni/models/moss_tts_local/radix_hash.py.
"""

from __future__ import annotations

import hashlib

import torch

# The hash space must stay below the special-token band. The scheduler
# finishes any request whose generated id crosses the vocab boundary
# (``Req._check_vocab_boundary_finish``), so a real (continuing) audio frame
# must never land in or above the band. We use the same constant as
# moss_tts_local for consistency.
RADIX_HASH_SPACE = 151643

# Polynomial-hash constants.
# _MOD is the Mersenne prime 2**31 - 1. With the accumulator and every channel
# value reduced below _MOD (< 2**31) and _BASE < _MOD, each Horner step
# ``acc * _BASE + v`` stays below 2**31 * 2**31 = 2**62, comfortably inside
# signed int64 (max 2**63 - 1). So the int64 ops never overflow and the result
# is bit-reproducible on CPU and GPU.
_MOD = 2147483647  # 2**31 - 1, Mersenne prime M31
_BASE = 1000000007  # 1e9 + 7, prime, < _MOD


def poly_row_hash(rows: torch.Tensor) -> torch.Tensor:
    """Fixed-coefficient polynomial hash of each row, in ``[0, _MOD)``.

    ``rows`` is ``[B, C]`` integer. Returns ``[B]`` int64 on ``rows.device``.
    Pure elementwise int64 torch ops (mul / add / remainder) over a static
    channel count -- no host sync, CUDA-graph capturable.
    """
    if rows.ndim != 2:
        raise ValueError(f"rows must be 2-D [B, C], got shape {tuple(rows.shape)}")
    work = rows.to(torch.int64)
    acc = torch.zeros(work.shape[0], dtype=torch.int64, device=work.device)
    # Static trip count (one frame = a fixed number of channels): the loop
    # unrolls into a fixed op sequence at capture time.
    for channel in range(work.shape[1]):
        # Reduce defensively in case a caller passes a raw id >= _MOD.
        value = torch.remainder(work[:, channel], _MOD)
        acc = torch.remainder(acc * _BASE + value, _MOD)
    return acc


def folded_hash_coefficients(
    width: int,
    *,
    device: torch.device | str,
    hash_space: int = RADIX_HASH_SPACE,
) -> torch.Tensor:
    """Return coefficients for a one-reduction folded row hash.

    The decode loop only needs a stable radix key inside ``hash_space``; it
    does not need the intermediate M31 polynomial value. Precomputing the
    polynomial coefficients modulo ``hash_space`` lets the hot path hash a full
    row with one multiply/sum/remainder instead of one remainder per channel.
    """
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    base = _BASE % int(hash_space)
    coeffs = [0] * width
    power = 1
    for idx in range(width - 1, -1, -1):
        coeffs[idx] = power
        power = (power * base) % int(hash_space)
    return torch.tensor(coeffs, dtype=torch.int64, device=device)


def folded_row_hash(
    rows: torch.Tensor,
    coeffs: torch.Tensor,
    *,
    hash_space: int = RADIX_HASH_SPACE,
) -> torch.Tensor:
    """Fast full-row hash already folded into ``[0, hash_space)``."""
    if rows.ndim != 2:
        raise ValueError(f"rows must be 2-D [B, C], got shape {tuple(rows.shape)}")
    if coeffs.ndim != 1 or int(coeffs.shape[0]) != int(rows.shape[1]):
        raise ValueError(
            "coeffs must be 1-D with one coefficient per row channel "
            f"(got {tuple(coeffs.shape)} for rows {tuple(rows.shape)})"
        )
    work = rows.to(torch.int64)
    folded = (torch.remainder(work, hash_space) * coeffs.view(1, -1)).sum(dim=1)
    return torch.remainder(folded, hash_space)


def gpu_radix_row_hash(
    rows: torch.Tensor,
    eoa_mask: torch.Tensor,
    eoa_id: int,
    *,
    hash_space: int = RADIX_HASH_SPACE,
    coeffs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Capture-safe radix token ids for a batch of generated ZONOS2 frames.

    Args:
        rows: [B, frame_width] int64 (n_codebooks audio codes + text column).
        eoa_mask: [B] bool -- True for frames where EOS was detected (any
            codebook emitted eoa_id).
        eoa_id: The end-of-audio token id.
        hash_space: The modular space for folding hash values.

    Returns:
        [B] int64 -- radix cache token ids. Continuing frames get a key in
        ``[0, hash_space)``; EOS rows get the raw ``eoa_id`` so the existing
        eos detection still fires.
    """
    if coeffs is None:
        folded = torch.remainder(poly_row_hash(rows), hash_space)
    else:
        folded = folded_row_hash(rows, coeffs, hash_space=hash_space)
    eoa_val = torch.full_like(folded, eoa_id)
    return torch.where(eoa_mask, eoa_val, folded)


def build_row_cache_key_ids(rows: torch.Tensor) -> list[int]:
    """Build stable radix-cache token ids for ZONOS2 multi-channel prompt rows.

    This is the host-side (CPU) version used during request building for prompt
    rows. It uses blake2b for high-quality hashing since it only runs once per
    request, not on the decode hot path.

    Args:
        rows: [seq_len, frame_width] tensor of prompt rows.

    Returns:
        List of int64 hash values, one per row, suitable as radix cache keys.
    """
    rows = rows.detach().to(dtype=torch.long, device="cpu")
    key_ids: list[int] = []
    for row in rows:
        digest = hashlib.blake2b(row.numpy().tobytes(), digest_size=8).digest()
        key_ids.append(int.from_bytes(digest, "little") & ((1 << 63) - 1))
    return key_ids
