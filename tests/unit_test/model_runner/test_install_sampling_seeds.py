# SPDX-License-Identifier: Apache-2.0
"""Base-runner _install_sampling_seeds: wire a request seed to the sampler."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.sampling.seed import SAMPLING_SEED_MASK, derive_sampling_seed


def _runner():
    runner = object.__new__(ModelRunner)
    runner.device = torch.device("cpu")
    runner._sampling_seed_cache = None
    return runner


def _req(seed, request_id="req"):
    sp = SimpleNamespace(sampling_seed=seed)
    return SimpleNamespace(
        request_id=request_id,
        data=SimpleNamespace(req=SimpleNamespace(sampling_params=sp)),
    )


def _fb(sampling_seed=None, *, top_p=False, top_k=False, min_p=False):
    return SimpleNamespace(
        sampling_info=SimpleNamespace(
            device="cpu",
            sampling_seed=sampling_seed,
            need_top_p_sampling=top_p,
            need_top_k_sampling=top_k,
            need_min_p_sampling=min_p,
        )
    )


def test_installs_per_row_seeds_and_noops_without_one():
    runner = _runner()
    # seeded rows -> per-row int64 seed tensor; mixed unseeded rows get a
    # rank-shared fallback derived from the request id.
    fb = _fb()
    requests = [_req(42, "seeded"), _req(None, "unseeded")]
    runner._install_sampling_seeds(fb, requests)
    ss = fb.sampling_info.sampling_seed
    assert isinstance(ss, torch.Tensor) and ss.dtype == torch.long
    assert int(ss[0]) == 42 and ss.shape == (2,)
    assert int(ss[1]) == derive_sampling_seed("sglang-omni-unseeded-row", "unseeded")
    assert requests[1].data.req.sampling_params.sampling_seed is None
    # no seed anywhere -> left unseeded (preserves random sampling)
    fb2 = _fb()
    runner._install_sampling_seeds(fb2, [_req(None), _req(None)])
    assert fb2.sampling_info.sampling_seed is None


def test_does_not_clobber_subclass_installed_seed():
    runner = _runner()
    preset = torch.tensor([1, 2, 3])
    fb = _fb(sampling_seed=preset)
    runner._install_sampling_seeds(fb, [_req(42), _req(42), _req(42)])
    assert fb.sampling_info.sampling_seed is preset


def test_preinstalled_seed_requires_sampling_mode_contract():
    runner = _runner()
    preset = torch.tensor([1, 2])
    fb = SimpleNamespace(sampling_info=SimpleNamespace(sampling_seed=preset))
    with pytest.raises(AttributeError):
        runner._install_sampling_seeds(fb, [_req(42), _req(42)])


def test_unseeded_row_in_seeded_batch_uses_rank_shared_fallback():
    runner = _runner()
    requests = [_req(42, "seeded"), _req(None, "unseeded")]
    fb = _fb()
    runner._install_sampling_seeds(fb, requests)
    fallback = int(fb.sampling_info.sampling_seed[1])
    assert requests[1].data.req.sampling_params.sampling_seed is None

    fb_next = _fb()
    runner._install_sampling_seeds(fb_next, requests)
    assert int(fb_next.sampling_info.sampling_seed[1]) == fallback
    assert requests[1].data.req.sampling_params.sampling_seed is None


def test_rejects_seeded_min_p_before_upstream_sampler():
    runner = _runner()
    with pytest.raises(ValueError, match="min_p"):
        runner._install_sampling_seeds(_fb(min_p=True), [_req(42)])


def test_rejects_seeded_flashinfer_top_p_before_upstream_sampler(monkeypatch):
    monkeypatch.setattr(
        "sglang_omni.model_runner.base._current_sglang_sampling_backend",
        lambda: "flashinfer",
    )
    runner = _runner()
    with pytest.raises(ValueError, match="flashinfer"):
        runner._install_sampling_seeds(_fb(top_p=True), [_req(42)])


def test_allows_seeded_pytorch_top_p(monkeypatch):
    monkeypatch.setattr(
        "sglang_omni.model_runner.base._current_sglang_sampling_backend",
        lambda: "pytorch",
    )
    runner = _runner()
    fb = _fb(top_p=True)
    runner._install_sampling_seeds(fb, [_req(42)])
    assert int(fb.sampling_info.sampling_seed[0]) == 42


def test_reuses_seed_tensor_across_steps_and_new_requests():
    """Explicit seed values, not request identity, own the reusable tensor."""
    runner = _runner()
    first = _fb()
    runner._install_sampling_seeds(first, [_req(7, "finished"), _req(101)])
    with patch(
        "sglang_omni.model_runner.base.torch.tensor", wraps=torch.tensor
    ) as make:
        following = _fb()
        runner._install_sampling_seeds(following, [_req(7, "new"), _req(101)])
    assert following.sampling_info.sampling_seed is first.sampling_info.sampling_seed
    make.assert_not_called()


@pytest.mark.parametrize("seeds", [[101, 7], [7], [7, 102], [7, 101, 42]])
def test_changed_rows_do_not_overwrite_an_inflight_tensor(seeds):
    runner = _runner()
    first = _fb()
    runner._install_sampling_seeds(first, [_req(7), _req(101)])
    following = _fb()
    runner._install_sampling_seeds(following, [_req(seed) for seed in seeds])
    assert following.sampling_info.sampling_seed.tolist() == seeds
    assert (
        following.sampling_info.sampling_seed is not first.sampling_info.sampling_seed
    )
    assert first.sampling_info.sampling_seed.tolist() == [7, 101]


def test_unseeded_request_replacement_changes_only_derived_row():
    runner = _runner()
    first, following = _fb(), _fb()
    runner._install_sampling_seeds(first, [_req(42), _req(None, "old")])
    replacement = [_req(42), _req(None, "new")]
    runner._install_sampling_seeds(following, replacement)
    assert following.sampling_info.sampling_seed.tolist() == [
        42,
        derive_sampling_seed("sglang-omni-unseeded-row", "new"),
    ]
    assert first.sampling_info.sampling_seed[1] == derive_sampling_seed(
        "sglang-omni-unseeded-row", "old"
    )
    assert replacement[1].data.req.sampling_params.sampling_seed is None


@pytest.mark.parametrize("seed", [-1, 1 << 80])
def test_normalized_seed_is_reused(seed):
    runner = _runner()
    request = _req(seed)
    first, following = _fb(), _fb()
    runner._install_sampling_seeds(first, [request])
    assert request.data.req.sampling_params.sampling_seed == seed & SAMPLING_SEED_MASK
    runner._install_sampling_seeds(following, [request])
    assert following.sampling_info.sampling_seed is first.sampling_info.sampling_seed


def test_cache_hit_does_not_bypass_sampler_validation(monkeypatch):
    runner = _runner()
    runner._install_sampling_seeds(_fb(), [_req(42)])
    with pytest.raises(ValueError, match="min_p"):
        runner._install_sampling_seeds(_fb(min_p=True), [_req(42)])
    monkeypatch.setattr(
        "sglang_omni.model_runner.base._current_sglang_sampling_backend",
        lambda: "flashinfer",
    )
    with pytest.raises(ValueError, match="flashinfer"):
        runner._install_sampling_seeds(_fb(top_k=True), [_req(42)])


def test_worker_device_is_part_of_cache_key():
    """Simulate worker-device changes while keeping allocations CPU-only."""
    runner = _runner()
    first, following = _fb(), _fb()
    runner.device = torch.device("cuda:0")
    runner._install_sampling_seeds(first, [_req(42)])
    runner.device = torch.device("cuda:1")
    runner._install_sampling_seeds(following, [_req(42)])
    assert (
        following.sampling_info.sampling_seed is not first.sampling_info.sampling_seed
    )
    assert following.sampling_info.sampling_seed.tolist() == [42]


def test_independent_runners_preserve_rank_shared_seed_values():
    """Check TP fallback determinism without claiming a distributed GPU test."""
    left, right = _fb(), _fb()
    _runner()._install_sampling_seeds(left, [_req(42), _req(None, "same-request")])
    _runner()._install_sampling_seeds(right, [_req(42), _req(None, "same-request")])
    assert torch.equal(
        left.sampling_info.sampling_seed, right.sampling_info.sampling_seed
    )


def test_cache_retains_only_latest_tensor():
    import gc
    import weakref

    runner = _runner()
    first = _fb()
    runner._install_sampling_seeds(first, [_req(1)])
    retired = weakref.ref(first.sampling_info.sampling_seed)
    del first
    for seed in range(2, 20):
        runner._install_sampling_seeds(_fb(), [_req(seed)])
    gc.collect()
    assert retired() is None


def test_failed_replacement_preserves_the_previous_entry():
    runner = _runner()
    first = _fb()
    runner._install_sampling_seeds(first, [_req(1)])
    with (
        patch(
            "sglang_omni.model_runner.base.torch.tensor",
            side_effect=RuntimeError("allocation failed"),
        ),
        pytest.raises(RuntimeError, match="allocation failed"),
    ):
        runner._install_sampling_seeds(_fb(), [_req(2)])
    following = _fb()
    runner._install_sampling_seeds(following, [_req(1)])
    assert following.sampling_info.sampling_seed is first.sampling_info.sampling_seed
