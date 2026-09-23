# SPDX-License-Identifier: Apache-2.0
"""Check request seeds on the scheduler-owned sampling batch."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.sampling.seed import SAMPLING_SEED_MASK, derive_sampling_seed


def _req(seed, request_id="req"):
    sp = SimpleNamespace(sampling_seed=seed)
    return SimpleNamespace(
        request_id=request_id,
        data=SimpleNamespace(req=SimpleNamespace(sampling_params=sp)),
    )


def _batch(sampling_seed=None, *, top_p=False, min_p=False):
    return SimpleNamespace(
        sampling_info=SimpleNamespace(
            device="cpu",
            sampling_seed=sampling_seed,
            need_top_p_sampling=top_p,
            need_top_k_sampling=False,
            need_min_p_sampling=min_p,
        )
    )


def test_mixed_batch_seeds_and_unseeded_batch():
    runner = object.__new__(ModelRunner)
    batch = _batch()
    requests = [_req(42, "seeded"), _req(None, "unseeded")]
    runner.install_sampling_seeds(batch, requests)
    assert batch.sampling_info.sampling_seed.tolist() == [
        42,
        derive_sampling_seed("sglang-omni-unseeded-row", "unseeded"),
    ]
    assert requests[1].data.req.sampling_params.sampling_seed is None

    unseeded = _batch()
    runner.install_sampling_seeds(unseeded, [_req(None)])
    assert unseeded.sampling_info.sampling_seed is None


def test_existing_sampling_info_seed_is_reused():
    runner = object.__new__(ModelRunner)
    batch = _batch()
    requests = [_req(7), _req(11)]
    runner.install_sampling_seeds(batch, requests)
    seed_tensor = batch.sampling_info.sampling_seed
    with patch("sglang_omni.model_runner.base.torch.tensor") as make:
        runner.install_sampling_seeds(batch, requests)
    assert batch.sampling_info.sampling_seed is seed_tensor
    make.assert_not_called()


def test_mixed_merge_repairs_seed_shape():
    runner = object.__new__(ModelRunner)
    batch = _batch()
    runner.install_sampling_seeds(batch, [_req(7)])
    previous = batch.sampling_info.sampling_seed
    requests = [_req(7), _req(None, "new")]
    runner.install_sampling_seeds(batch, requests)
    assert batch.sampling_info.sampling_seed.tolist() == [
        7,
        derive_sampling_seed("sglang-omni-unseeded-row", "new"),
    ]
    assert previous.tolist() == [7]


def test_unseeded_row_reverts_after_seeded_row_finishes():
    runner = object.__new__(ModelRunner)
    batch = _batch()
    runner.install_sampling_seeds(batch, [_req(7), _req(None, "remaining")])
    runner.install_sampling_seeds(batch, [_req(None, "remaining")])
    assert batch.sampling_info.sampling_seed is None


def test_preserves_preinstalled_seed():
    runner = object.__new__(ModelRunner)
    preset = torch.tensor([1, 2])
    batch = _batch(preset)
    runner.install_sampling_seeds(batch, [_req(7), _req(11)])
    assert batch.sampling_info.sampling_seed is preset


@pytest.mark.parametrize("seed", [-1, 1 << 80])
def test_normalizes_explicit_seed_once(seed):
    runner = object.__new__(ModelRunner)
    request = _req(seed)
    batch = _batch()
    runner.install_sampling_seeds(batch, [request])
    assert request.data.req.sampling_params.sampling_seed == seed & SAMPLING_SEED_MASK
    assert batch.sampling_info.sampling_seed.tolist() == [seed & SAMPLING_SEED_MASK]


def test_seeded_sampling_validation(monkeypatch):
    runner = object.__new__(ModelRunner)
    with pytest.raises(ValueError, match="min_p"):
        runner.install_sampling_seeds(_batch(min_p=True), [_req(42)])

    monkeypatch.setattr(
        "sglang_omni.model_runner.base.current_sglang_sampling_backend",
        lambda: "flashinfer",
    )
    with pytest.raises(ValueError, match="flashinfer"):
        runner.install_sampling_seeds(_batch(top_p=True), [_req(42)])
