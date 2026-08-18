# SPDX-License-Identifier: Apache-2.0
"""The cached resample path must stay bit-identical to torchaudio's."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torchaudio = pytest.importorskip("torchaudio")

from sglang_omni.utils.audio import _cached_resample, _resample_kernel


@pytest.mark.parametrize(
    "orig,target",
    [(8000, 16000), (44100, 16000), (22050, 16000), (48000, 16000), (16000, 24000)],
)
def test_matches_torchaudio_bit_exactly(orig, target):
    rng = np.random.default_rng(0)
    wav = torch.from_numpy(rng.standard_normal(orig, dtype=np.float32))
    expected = torchaudio.functional.resample(wav, orig, target)
    got = _cached_resample(wav, orig, target, None)
    assert got.shape == expected.shape
    assert torch.equal(got, expected)


def test_kwargs_are_honoured_and_keyed():
    rng = np.random.default_rng(1)
    wav = torch.from_numpy(rng.standard_normal(16000, dtype=np.float32))
    kwargs = {"lowpass_filter_width": 16}
    expected = torchaudio.functional.resample(wav, 16000, 8000, **kwargs)
    got = _cached_resample(wav, 16000, 8000, kwargs)
    assert torch.equal(got, expected)
    # A different option must not reuse the first kernel.
    other = _cached_resample(wav, 16000, 8000, {"lowpass_filter_width": 64})
    assert not torch.equal(got, other)


def test_kernel_is_reused_across_calls():
    _resample_kernel.cache_clear()
    rng = np.random.default_rng(2)
    wav = torch.from_numpy(rng.standard_normal(16000, dtype=np.float32))
    for _ in range(5):
        _cached_resample(wav, 44100, 16000, None)
    info = _resample_kernel.cache_info()
    assert info.misses == 1
    assert info.hits == 4
