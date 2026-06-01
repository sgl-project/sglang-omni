# SPDX-License-Identifier: Apache-2.0
"""Real-model parity tests for _forward_batch (GPU-gated on CODE2WAV_PARITY_MODEL_PATH)."""
from __future__ import annotations

import math
import os

import pytest
import torch

from sglang_omni.models.qwen3_omni.components.code2wav_executor import (
    _DEFAULT_NUM_QUANTIZERS,
    _Code2WavStreamingExecutor,
    load_code2wav_model,
)

_MODEL_PATH_ENV = "CODE2WAV_PARITY_MODEL_PATH"


def _skip_if_no_real_model():
    if not torch.cuda.is_available():
        pytest.skip("Real code2wav parity needs CUDA")
    path = os.environ.get(_MODEL_PATH_ENV)
    if not path:
        pytest.skip(f"{_MODEL_PATH_ENV} not set (path to Qwen3-Omni model snapshot)")
    if not os.path.isdir(path):
        pytest.skip(f"{_MODEL_PATH_ENV}={path!r} is not a directory")
    return path


@pytest.fixture(scope="module")
def real_model():
    path = _skip_if_no_real_model()
    model = load_code2wav_model(path, device="cuda:0", dtype="bfloat16")
    model.eval()
    return model


@pytest.fixture(scope="module")
def warmed_executor(real_model):
    exe = _Code2WavStreamingExecutor(
        real_model,
        device="cuda:0",
        stream_chunk_size=10,
        left_context_size=25,
        max_batch_size=4,
        warmup=True,
    )
    return exe


def _rms(x: torch.Tensor) -> float:
    return float(x.pow(2).mean().sqrt().item())


def _db_fs(rms: float) -> float:
    return 20 * math.log10(rms) if rms > 0 else -200.0


def test_output_length_lookup_matches_real_forward(real_model, warmed_executor):
    assert warmed_executor._output_len_by_seq, "warmup did not populate lookup"

    num_q = _DEFAULT_NUM_QUANTIZERS
    pad = 0
    with torch.no_grad():
        for seq_len, recorded in warmed_executor._output_len_by_seq.items():
            dummy = torch.full(
                (1, num_q, seq_len),
                fill_value=pad,
                dtype=torch.long,
                device="cuda:0",
            )
            out = real_model(dummy)
            actual = int(out.shape[-1])
            assert actual == recorded, (
                f"seq_len={seq_len}: recorded {recorded} != fresh forward "
                f"actual {actual}"
            )
            naive = seq_len * warmed_executor._total_upsample
            assert actual < naive, (
                f"seq_len={seq_len}: actual {actual} should be < naive {naive} "
                f"(evidence the old formula was over-slicing)"
            )


def test_output_length_offset_is_architecture_constant(real_model, warmed_executor):
    offsets = {
        sl * warmed_executor._total_upsample - actual
        for sl, actual in warmed_executor._output_len_by_seq.items()
    }
    assert len(offsets) == 1, f"offset expected to be constant; got {sorted(offsets)}"
    offset = offsets.pop()
    assert warmed_executor._output_len_offset == offset


@pytest.mark.parametrize(
    "short_len,long_len",
    [(10, 35), (20, 35), (30, 35), (35, 70)],
)
def test_short_request_tail_has_no_padding_bleed(
    real_model,
    warmed_executor,
    short_len: int,
    long_len: int,
):
    num_q = _DEFAULT_NUM_QUANTIZERS
    pad = 0
    torch.manual_seed(42)

    short = torch.randint(
        0, 1024, (1, num_q, short_len), dtype=torch.long, device="cuda"
    )
    long_ = torch.randint(
        0, 1024, (1, num_q, long_len), dtype=torch.long, device="cuda"
    )

    with torch.no_grad():
        single_short = real_model(short).detach().float()[0, 0]
        single_long = real_model(long_).detach().float()[0, 0]

        padded = torch.full(
            (2, num_q, long_len),
            fill_value=pad,
            dtype=torch.long,
            device="cuda:0",
        )
        padded[0, :, :short_len] = short[0]
        padded[1, :, :long_len] = long_[0]
        batched = real_model(padded).detach().float()

    s_len = single_short.shape[-1]
    l_len = single_long.shape[-1]
    b_short = batched[0, 0, :s_len]
    b_long = batched[1, 0, :l_len]

    for window in (32, 64, 128, 256, 512):
        if window >= s_len or window >= l_len:
            continue
        short_tail_rms = _rms((single_short - b_short)[-window:])
        long_tail_rms = _rms((single_long - b_long)[-window:])
        assert short_tail_rms <= long_tail_rms * 4.0 + 1e-4, (
            f"short_len={short_len}, long_len={long_len}, window={window}: "
            f"short_tail_rms={short_tail_rms:.2e} ({_db_fs(short_tail_rms):.1f} dBFS) "
            f"vs long baseline {long_tail_rms:.2e} "
            f"({_db_fs(long_tail_rms):.1f} dBFS) — would indicate real "
            f"padding bleed beyond bf16 batch noise floor"
        )
        _ABS_CEILING = 10 ** (-60 / 20)
        assert short_tail_rms < _ABS_CEILING, (
            f"short_len={short_len}, long_len={long_len}, window={window}: "
            f"short_tail_rms={short_tail_rms:.2e} "
            f"({_db_fs(short_tail_rms):.1f} dBFS) exceeds absolute ceiling "
            f"-60 dBFS — low-amplitude correlated artifact in short tail"
        )
