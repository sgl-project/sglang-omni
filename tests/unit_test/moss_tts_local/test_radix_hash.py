# SPDX-License-Identifier: Apache-2.0
"""Generated-row radix hash properties, CUDA equivalence, and graph replay."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.moss_tts_local.radix_hash import (
    _BASE,
    _MOD,
    RADIX_HASH_SPACE,
    build_rows_and_radix_token_ids,
    gpu_radix_row_hash,
    poly_row_hash,
)

N_CHANNELS = 13  # text channel + 12 RVQ codes (n_vq = 12)
END_ID = 151670  # audio_end_token_id: in the special band (>= RADIX_HASH_SPACE)
SLOT_ID = 151646  # audio_assistant_slot_token_id: text channel of a continuing frame

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def continuing_rows(codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[B, 13] rows: slot id in channel 0, ``codes`` in channels 1..12."""
    b = codes.shape[0]
    rows = torch.empty((b, N_CHANNELS), dtype=torch.long)
    rows[:, 0] = SLOT_ID
    rows[:, 1:] = codes
    next_text = torch.full((b,), SLOT_ID, dtype=torch.long)
    return rows, next_text


def ref_poly(values: list[int]) -> int:
    """Pure-Python bignum reference for the Horner hash (no int64 overflow)."""
    acc = 0
    for v in values:
        acc = (acc * _BASE + (v % _MOD)) % _MOD
    return acc


def test_matches_python_reference():
    """The int64 torch hash equals an exact bignum reference (no overflow)."""
    codes = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], dtype=torch.long)
    rows, next_text = continuing_rows(codes)
    raw = int(poly_row_hash(rows)[0])
    ref = ref_poly([SLOT_ID, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    assert raw == ref
    folded = int(gpu_radix_row_hash(rows, next_text, END_ID)[0])
    assert folded == ref % RADIX_HASH_SPACE


def test_deterministic_across_repeats():
    """Same input hashed N times is identical (determinism / property ①)."""
    torch.manual_seed(0)
    codes = torch.randint(0, 1024, (32, 12))
    rows, next_text = continuing_rows(codes)
    first = gpu_radix_row_hash(rows, next_text, END_ID)
    for _ in range(8):
        again = gpu_radix_row_hash(rows.clone(), next_text.clone(), END_ID)
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
    rows, next_text = continuing_rows(codes)

    # Test real collisions only: dedup identical rows (same input is not a clash).
    rows = torch.unique(rows, dim=0)
    next_text = torch.full((rows.shape[0],), SLOT_ID, dtype=torch.long)

    raw = poly_row_hash(rows)
    folded = gpu_radix_row_hash(rows, next_text, END_ID)
    assert torch.unique(raw).numel() == raw.numel(), "raw poly hash collided"
    assert torch.unique(folded).numel() == folded.numel(), "folded key collided"


def test_eos_rows_keep_audio_end_id():
    """EOS rows return the raw audio_end id; others fold below the band (③)."""
    torch.manual_seed(3)
    codes = torch.randint(0, 1024, (5, 12))
    rows, next_text = continuing_rows(codes)
    for i in (1, 3):
        next_text[i] = END_ID
        rows[i, 0] = END_ID
    keys = gpu_radix_row_hash(rows, next_text, END_ID)
    assert int(keys[1]) == END_ID
    assert int(keys[3]) == END_ID
    for i in (0, 2, 4):
        assert 0 <= int(keys[i]) < RADIX_HASH_SPACE


def test_continuing_keys_within_hash_space():
    """Continuing-frame keys are in [0, RADIX_HASH_SPACE) (domain / ④)."""
    torch.manual_seed(2)
    codes = torch.randint(0, 4096, (256, 12))
    rows, next_text = continuing_rows(codes)
    keys = gpu_radix_row_hash(rows, next_text, END_ID)
    assert int(keys.min()) >= 0
    assert int(keys.max()) < RADIX_HASH_SPACE


def test_output_dtype_and_device_follow_input():
    """dtype is int64 and device follows the input rows (⑤)."""
    codes = torch.randint(0, 1024, (4, 12))
    rows, next_text = continuing_rows(codes)
    keys = gpu_radix_row_hash(rows, next_text, END_ID)
    assert keys.dtype == torch.int64
    assert keys.device == rows.device
    raw = poly_row_hash(rows)
    assert raw.dtype == torch.int64
    assert raw.device == rows.device


def test_accepts_non_int64_input_dtype():
    """int32 rows are handled (the hash casts to int64 internally)."""
    codes = torch.randint(0, 1024, (4, 12), dtype=torch.int32)
    rows = torch.empty((4, N_CHANNELS), dtype=torch.int32)
    rows[:, 0] = SLOT_ID
    rows[:, 1:] = codes
    next_text = torch.full((4,), SLOT_ID, dtype=torch.int32)
    keys = gpu_radix_row_hash(rows, next_text, END_ID)
    assert keys.dtype == torch.int64
    assert int(keys.min()) >= 0 and int(keys.max()) < RADIX_HASH_SPACE


def test_build_rows_and_ids_cpu_matches_split_reference():
    stop = torch.tensor([0, 1, 2], dtype=torch.long)
    codes = torch.tensor([[1, 2], [-1, _MOD + 3], [7, 8]], dtype=torch.long)
    rows, ids = build_rows_and_radix_token_ids(stop, codes, SLOT_ID, END_ID)
    expected_rows = torch.tensor(
        [[SLOT_ID, 1, 2], [END_ID, -1, _MOD + 3], [END_ID, 7, 8]],
        dtype=torch.long,
    )
    expected_ids = torch.tensor(
        [
            ref_poly(row) % RADIX_HASH_SPACE if row[0] != END_ID else END_ID
            for row in expected_rows.tolist()
        ],
        dtype=torch.int64,
    )
    assert torch.equal(rows, expected_rows)
    assert torch.equal(ids, expected_ids)


@pytest.mark.accelerator
@requires_cuda
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "shape", [(0, 13), (4, 0), (1, 1), (3, 7), (16, 13), (129, 33)]
)
@pytest.mark.parametrize("strided", [False, True])
def test_cuda_matches_python_reference(shape, dtype, strided):
    batch, channels = shape
    rows = torch.randint(-4096, 4096, shape, dtype=dtype)
    limits = torch.iinfo(dtype)
    boundaries = [limits.min, -_MOD, -1, 0, _MOD - 1, _MOD, limits.max]
    for index, value in enumerate(boundaries[: rows.numel()]):
        rows.view(-1)[index] = value
    text = torch.full((batch,), SLOT_ID, dtype=dtype)
    text[::3] = END_ID
    hash_space = 1009
    expected = torch.tensor(
        [
            END_ID if token == END_ID else ref_poly(row) % hash_space
            for row, token in zip(rows.tolist(), text.tolist())
        ],
        dtype=torch.int64,
    )
    if strided:
        storage = torch.empty((batch, channels * 2), device="cuda", dtype=dtype)
        device_rows = storage[:, ::2]
        device_rows.copy_(rows)
    else:
        device_rows = rows.cuda()
    text_storage = torch.empty((batch, 2), device="cuda", dtype=dtype)
    device_text = text_storage[:, 1]
    device_text.copy_(text)
    actual = gpu_radix_row_hash(device_rows, device_text, END_ID, hash_space=hash_space)
    assert actual.dtype == torch.int64
    assert actual.device == device_rows.device
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.accelerator
@requires_cuda
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_cuda_row_builder_tail_matches_python_reference(dtype, strided):
    batch = 129
    channels = N_CHANNELS - 1
    codes = (torch.arange(batch * channels) % 1024).reshape(batch, channels)
    stop = (torch.arange(batch) + 1) % 3
    text = torch.where(stop == 0, SLOT_ID, END_ID)
    expected_rows = torch.cat((text[:, None], codes), dim=1)
    expected_ids = torch.tensor(
        [
            END_ID if row[0] == END_ID else ref_poly(row) % RADIX_HASH_SPACE
            for row in expected_rows.tolist()
        ],
        dtype=torch.int64,
    )
    if strided:
        code_storage = torch.empty((batch, channels * 2), device="cuda", dtype=dtype)
        device_codes = code_storage[:, ::2]
        device_codes.copy_(codes)
        stop_storage = torch.empty(batch * 2, device="cuda", dtype=dtype)
        device_stop = stop_storage[::2]
        device_stop.copy_(stop)
    else:
        device_codes = codes.to(device="cuda", dtype=dtype)
        device_stop = stop.to(device="cuda", dtype=dtype)

    rows, ids = build_rows_and_radix_token_ids(
        device_stop, device_codes, SLOT_ID, END_ID
    )

    assert rows.dtype == ids.dtype == torch.int64
    assert rows.device == ids.device == device_codes.device
    assert torch.equal(rows.cpu(), expected_rows)
    assert torch.equal(ids.cpu(), expected_ids)


@pytest.mark.accelerator
@requires_cuda
def test_cuda_graph_replays_new_rows_and_eos():
    rows = torch.zeros((16, N_CHANNELS), device="cuda", dtype=torch.int64)
    text = rows[:, 0]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        gpu_radix_row_hash(rows, text, END_ID)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gpu_radix_row_hash(rows, text, END_ID)
    for step in range(3):
        host_rows = torch.randint(0, 4096, rows.shape, dtype=torch.int64)
        host_rows[:, 0] = SLOT_ID
        host_rows[step::3, 0] = END_ID
        expected = gpu_radix_row_hash(host_rows, host_rows[:, 0], END_ID)
        rows.copy_(host_rows)
        graph.replay()
        assert torch.equal(actual.cpu(), expected)


@pytest.mark.accelerator
@requires_cuda
def test_cuda_graph_replays_row_builder():
    stop = torch.zeros(16, device="cuda", dtype=torch.int64)
    codes = torch.zeros((16, N_CHANNELS - 1), device="cuda", dtype=torch.int64)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        build_rows_and_radix_token_ids(stop, codes, SLOT_ID, END_ID)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_rows, actual_ids = build_rows_and_radix_token_ids(
            stop, codes, SLOT_ID, END_ID
        )
    for step in range(3):
        host_stop = torch.arange(16, dtype=torch.int64) % 3
        host_codes = torch.randint(0, 1024, codes.shape, dtype=torch.int64)
        host_rows = torch.empty((16, N_CHANNELS), dtype=torch.int64)
        host_rows[:, 0] = torch.where(
            host_stop == 0,
            torch.full((16,), SLOT_ID, dtype=torch.int64),
            torch.full((16,), END_ID, dtype=torch.int64),
        )
        host_rows[:, 1:] = host_codes
        expected_ids = torch.tensor(
            [
                (
                    END_ID
                    if row[0] == END_ID
                    else ref_poly(row.tolist()) % RADIX_HASH_SPACE
                )
                for row in host_rows
            ],
            dtype=torch.int64,
        )
        stop.copy_(host_stop)
        codes.copy_(host_codes)
        graph.replay()
        assert torch.equal(actual_rows.cpu(), host_rows)
        assert torch.equal(actual_ids.cpu(), expected_ids)
