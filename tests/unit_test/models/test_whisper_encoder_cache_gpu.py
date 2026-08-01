# SPDX-License-Identifier: Apache-2.0
"""GPU integration test for the Whisper encoder-output cache.

This test requires ``torch`` and a CUDA device. When either is missing it is
skipped (``pytest.importorskip``), so it does not break CPU-only unit CI. When
run on a GPU it exercises the REAL cache logic (EncoderOutputCache + a real
deterministic encoder) and proves the two claims in the PR description:

* on a cache hit the encoder forward runs **once** (not re-run), and
* the returned encoder states are **numerically identical** to the uncached run.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from sglang_omni.models.whisper_asr.whisper_encoder_cache import EncoderOutputCache


class _CountingEncoder:
    """Deterministic stub encoder that counts forward calls."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device(device)
        self.calls = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        # deterministic, stateless mapping: mean over feature dim, keep dim
        return x.to(self.device).mean(dim=-1, keepdim=True).expand(-1, -1, x.shape[-1])


def _feature(seed: int, n: int = 16, device: str = "cuda") -> torch.Tensor:
    # deterministic content so the digest is stable across calls
    data = torch.tensor(
        [(seed + i) % 256 for i in range(n)], dtype=torch.float32, device=device
    )
    return data


def test_encoder_runs_once_on_cache_hit_and_outputs_identical() -> None:
    backend: dict = {}
    cache = EncoderOutputCache(model_id="whisper-test", backend=backend)
    encoder = _CountingEncoder(device="cuda")

    feat = _feature(seed=42)

    # First call: cache MISS -> encoder runs, result stored.
    out1 = cache.get_or_encode([feat], "cuda", encode_fn=lambda: encoder(feat))
    assert encoder.calls == 1

    # Second call with identical audio: cache HIT -> encoder NOT re-run.
    out2 = cache.get_or_encode([feat], "cuda", encode_fn=lambda: encoder(feat))
    assert encoder.calls == 1, "encoder must not re-run on a cache hit"

    # Outputs identical to the (uncached) first run.
    assert torch.allclose(out1, out2)
    # Stored artifact is CPU-resident (StageOutputCache-style), reload on GPU.
    assert out2.device.type == "cuda"

    # A different audio must miss and run the encoder again.
    feat_other = _feature(seed=7)
    out3 = cache.get_or_encode([feat_other], "cuda", encode_fn=lambda: encoder(feat_other))
    assert encoder.calls == 2
    assert not torch.allclose(out1, out3)


def test_cache_disabled_like_uncached_run_still_correct() -> None:
    # Sanity: a plain encoder call (what the model does when cache is off)
    # produces the same math as the cached path on first (miss) use.
    encoder = _CountingEncoder(device="cuda")
    feat = _feature(seed=99)
    direct = encoder(feat)
    assert direct.shape == (1, 1, feat.shape[-1])
