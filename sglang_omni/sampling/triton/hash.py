# SPDX-License-Identifier: Apache-2.0
"""Triton hashing primitives for seeded GPU sampling."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def rotl32(x, r: tl.constexpr) -> tl.uint32:
    x = x.to(tl.uint64)
    return ((x << r) | (x >> (32 - r))) & 0xFFFFFFFF


@triton.jit
def fmix32(h: tl.uint32) -> tl.uint32:
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


@triton.jit
def murmur3_mix(h: tl.uint32, k: tl.uint32) -> tl.uint32:
    k = (k * 0xCC9E2D51) & 0xFFFFFFFF
    k = rotl32(k, 15)
    k = (k * 0x1B873593) & 0xFFFFFFFF
    h ^= k
    h = rotl32(h, 13)
    h = (h * 5 + 0xE6546B64) & 0xFFFFFFFF
    return h


@triton.jit
def murmur_hash_seed_position_key(seed, position, key) -> tl.uint32:
    """Hash one sampling key from seed, generation position, and token id."""

    seed = seed.to(tl.uint64)
    h: tl.uint32 = 0
    h = murmur3_mix(h, (seed & 0xFFFFFFFF).to(tl.uint32))
    h = murmur3_mix(h, ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32))
    h = murmur3_mix(h, position.to(tl.uint32))
    h = murmur3_mix(h, key.to(tl.uint32))
    h ^= 16
    return fmix32(h)


__all__ = [
    "fmix32",
    "murmur3_mix",
    "murmur_hash_seed_position_key",
    "rotl32",
]
