# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the capture-safe generated-row radix hash.

Imports only ``torch`` and the ``radix_hash`` module (no model/runtime deps),
so the whole file runs on CPU. These cover the *key layer* of the two-layer
verification rubric in ``docs/design/gpu_radix_hash.md``; the GPU bit-identity
*output layer* rerun is tracked separately (PENDING-GPU).
"""

from __future__ import annotations

import torch

from sglang_omni.models.moss_tts_local.radix_hash import (
    _BASE,
    _MOD,
    RADIX_HASH_SPACE,
    gpu_radix_row_hash,
    poly_row_hash,
)

_N_CHANNELS = 13  # text channel + 12 RVQ codes (n_vq = 12)
_END_ID = 151670  # audio_end_token_id: in the special band (>= RADIX_HASH_SPACE)
_SLOT_ID = 151646  # audio_assistant_slot_token_id: text channel of a continuing frame


def _continuing_rows(codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[B, 13] rows: slot id in channel 0, ``codes`` in channels 1..12."""
    b = codes.shape[0]
    rows = torch.empty((b, _N_CHANNELS), dtype=torch.long)
    rows[:, 0] = _SLOT_ID
    rows[:, 1:] = codes
    next_text = torch.full((b,), _SLOT_ID, dtype=torch.long)
    return rows, next_text


def _ref_poly(values: list[int]) -> int:
    """Pure-Python bignum reference for the Horner hash (no int64 overflow)."""
    acc = 0
    for v in values:
        acc = (acc * _BASE + (v % _MOD)) % _MOD
    return acc


def test_matches_python_reference():
    """The int64 torch hash equals an exact bignum reference (no overflow)."""
    codes = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], dtype=torch.long)
    rows, next_text = _continuing_rows(codes)
    raw = int(poly_row_hash(rows)[0])
    ref = _ref_poly([_SLOT_ID, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    assert raw == ref
    folded = int(gpu_radix_row_hash(rows, next_text, _END_ID)[0])
    assert folded == ref % RADIX_HASH_SPACE


def test_deterministic_across_repeats():
    """Same input hashed N times is identical (determinism / property ①)."""
    torch.manual_seed(0)
    codes = torch.randint(0, 1024, (32, 12))
    rows, next_text = _continuing_rows(codes)
    first = gpu_radix_row_hash(rows, next_text, _END_ID)
    for _ in range(8):
        again = gpu_radix_row_hash(rows.clone(), next_text.clone(), _END_ID)
        assert torch.equal(first, again)


def test_no_collisions_on_adjacent_and_permuted_rows():
    """Adjacent (+/-1) and permuted rows hash distinctly (property ②)."""
    base = torch.arange(10, 130, 10, dtype=torch.long)  # 12 distinct codes
    variants = [base.clone()]
    for c in range(12):  # single-channel neighbours
        up = base.clone()
        up[c] += 1
        variants.append(up)
        dn = base.clone()
        dn[c] -= 1
        variants.append(dn)
    torch.manual_seed(1)
    for _ in range(8):  # permutations of the same codes (order sensitivity)
        variants.append(base[torch.randperm(12)])
    variants.extend(torch.randint(0, 2048, (40, 12)))  # distinct random block
    codes = torch.stack(variants, dim=0)
    rows, next_text = _continuing_rows(codes)

    # Test real collisions only: dedup identical rows (same input is not a clash).
    rows = torch.unique(rows, dim=0)
    next_text = torch.full((rows.shape[0],), _SLOT_ID, dtype=torch.long)

    raw = poly_row_hash(rows)
    folded = gpu_radix_row_hash(rows, next_text, _END_ID)
    assert torch.unique(raw).numel() == raw.numel(), "raw poly hash collided"
    assert torch.unique(folded).numel() == folded.numel(), "folded key collided"


def test_eos_rows_keep_audio_end_id():
    """EOS rows return the raw audio_end id; others fold below the band (③)."""
    torch.manual_seed(3)
    codes = torch.randint(0, 1024, (5, 12))
    rows, next_text = _continuing_rows(codes)
    for i in (1, 3):
        next_text[i] = _END_ID
        rows[i, 0] = _END_ID
    keys = gpu_radix_row_hash(rows, next_text, _END_ID)
    assert int(keys[1]) == _END_ID
    assert int(keys[3]) == _END_ID
    for i in (0, 2, 4):
        assert 0 <= int(keys[i]) < RADIX_HASH_SPACE


def test_continuing_keys_within_hash_space():
    """Continuing-frame keys are in [0, RADIX_HASH_SPACE) (domain / ④)."""
    torch.manual_seed(2)
    codes = torch.randint(0, 4096, (256, 12))
    rows, next_text = _continuing_rows(codes)
    keys = gpu_radix_row_hash(rows, next_text, _END_ID)
    assert int(keys.min()) >= 0
    assert int(keys.max()) < RADIX_HASH_SPACE


def test_output_dtype_and_device_follow_input():
    """dtype is int64 and device follows the input rows (⑤)."""
    codes = torch.randint(0, 1024, (4, 12))
    rows, next_text = _continuing_rows(codes)
    keys = gpu_radix_row_hash(rows, next_text, _END_ID)
    assert keys.dtype == torch.int64
    assert keys.device == rows.device
    raw = poly_row_hash(rows)
    assert raw.dtype == torch.int64
    assert raw.device == rows.device


def test_accepts_non_int64_input_dtype():
    """int32 rows are handled (the hash casts to int64 internally)."""
    codes = torch.randint(0, 1024, (4, 12), dtype=torch.int32)
    rows = torch.empty((4, _N_CHANNELS), dtype=torch.int32)
    rows[:, 0] = _SLOT_ID
    rows[:, 1:] = codes
    next_text = torch.full((4,), _SLOT_ID, dtype=torch.int32)
    keys = gpu_radix_row_hash(rows, next_text, _END_ID)
    assert keys.dtype == torch.int64
    assert int(keys.min()) >= 0 and int(keys.max()) < RADIX_HASH_SPACE
