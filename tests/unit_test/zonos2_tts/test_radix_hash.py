import pytest
import torch

from sglang_omni.models.zonos2_tts.radix_hash import (
    RADIX_HASH_SPACE,
    folded_hash_coefficients,
    folded_row_hash,
    gpu_radix_row_hash,
)


def test_folded_row_hash_is_stable_and_full_row_sensitive():
    rows = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 6],
            [6, 4, 3, 2, 1],
        ],
        dtype=torch.long,
    )
    coeffs = folded_hash_coefficients(rows.shape[1], device=rows.device)

    first = folded_row_hash(rows, coeffs)
    second = folded_row_hash(rows.clone(), coeffs.clone())

    assert torch.equal(first, second)
    assert int(first[0]) != int(first[1])
    assert int(first[0]) != int(first[2])
    assert bool(((first >= 0) & (first < RADIX_HASH_SPACE)).all())


def test_gpu_radix_row_hash_uses_eoa_mask_with_folded_coeffs():
    rows = torch.tensor(
        [
            [10, 20, 30],
            [11, 21, 31],
        ],
        dtype=torch.long,
    )
    coeffs = folded_hash_coefficients(rows.shape[1], device=rows.device)
    eoa_id = 1024
    keys = gpu_radix_row_hash(
        rows,
        torch.tensor([False, True]),
        eoa_id,
        coeffs=coeffs,
    )

    assert int(keys[1]) == eoa_id
    assert 0 <= int(keys[0]) < RADIX_HASH_SPACE


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_folded_row_hash_matches_cpu_on_cuda():
    rows = torch.tensor(
        [
            [1024, 7, 42, 99],
            [5, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    cpu_coeffs = folded_hash_coefficients(rows.shape[1], device="cpu")
    cuda_rows = rows.cuda()
    cuda_coeffs = folded_hash_coefficients(rows.shape[1], device=cuda_rows.device)

    assert torch.equal(
        folded_row_hash(rows, cpu_coeffs),
        folded_row_hash(cuda_rows, cuda_coeffs).cpu(),
    )
