# SPDX-License-Identifier: Apache-2.0
"""Decode-launch host-traffic contract for HiggsTTSModelRunner.

``_populate_cg_buffers`` runs at launch time while the previous step's forward
is still in flight, so any ``.cpu()`` / ``.tolist()`` on sampling_info tensors
is a stream-syncing D2H that serializes the async-decode overlap. These tests
pin the contract: sampling params reach the CG buffers device-side (no host
materialization), and the per-step row-indices rebuild is skipped when batch
membership is unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.higgs_tts.model_runner import HiggsTTSModelRunner
from sglang_omni.models.higgs_tts.sampler import K_MAX

_POOL = 6
_PAD = 5
_NCB = 3


class _NoHostReadbackTensor(torch.Tensor):
    """Tensor whose host-materialization entry points fail the test."""

    def cpu(self, *args, **kwargs):
        raise RuntimeError("host readback (cpu) on the decode launch path")

    def tolist(self):
        raise RuntimeError("host readback (tolist) on the decode launch path")

    def numpy(self, *args, **kwargs):
        raise RuntimeError("host readback (numpy) on the decode launch path")

    def item(self):
        raise RuntimeError("host readback (item) on the decode launch path")

    def __float__(self):
        raise RuntimeError("host readback (float) on the decode launch path")

    def __iter__(self):
        raise RuntimeError("host readback (iter) on the decode launch path")

    def to(self, *args, **kwargs):
        if any(str(a) == "cpu" for a in args) or str(kwargs.get("device")) == "cpu":
            raise RuntimeError("host readback (to cpu) on the decode launch path")
        return super().to(*args, **kwargs)


def _guarded(data, dtype):
    return torch.tensor(data, dtype=dtype).as_subclass(_NoHostReadbackTensor)


def _sampling_info(temps, top_ps, top_ks):
    return SimpleNamespace(
        temperatures=_guarded(temps, torch.float32).unsqueeze(-1),
        top_ps=_guarded(top_ps, torch.float32),
        top_ks=_guarded(top_ks, torch.long),
    )


def _make_runner(n_reqs=3):
    pool = SimpleNamespace(
        delay_count=torch.zeros(_POOL, dtype=torch.int32),
        eoc_countdown=torch.full((_POOL,), -1, dtype=torch.int32),
        generation_done=torch.zeros(_POOL, dtype=torch.bool),
        last_codes=torch.zeros((_POOL, _NCB), dtype=torch.long),
        seeds=torch.zeros(_POOL, dtype=torch.long),
        step_count=torch.zeros(_POOL, dtype=torch.long),
    )

    def _pool_reset_row(row):
        pool.delay_count[row] = 0
        pool.eoc_countdown[row] = -1
        pool.generation_done[row] = False
        pool.last_codes[row].zero_()
        pool.seeds[row] = 0
        pool.step_count[row] = 0

    pool.reset_row = _pool_reset_row

    acquire_calls: list[str] = []
    rid_to_row = {f"req{i}": i for i in range(n_reqs)}

    def _acquire(rid):
        acquire_calls.append(rid)
        return rid_to_row[rid]

    _make_runner.rid_to_row = rid_to_row

    runner = object.__new__(HiggsTTSModelRunner)
    runner._outbox = None
    runner._vocoder_target = "vocoder"
    runner._async_enabled = True
    runner._staging_slot = 0
    runner._host_staging_buffers = []
    runner._logprob_host_buffers = None
    runner._logprob_slot = 0
    runner._async_query_hit = 0
    runner._async_query_miss = 0
    runner.model = SimpleNamespace(
        _cg_row_indices=torch.zeros(_POOL, dtype=torch.long),
        _cg_temperature=torch.ones(_POOL, dtype=torch.float32),
        _cg_top_p=torch.ones(_POOL, dtype=torch.float32),
        _cg_top_k_buf=torch.full((_POOL,), K_MAX, dtype=torch.long),
        _cg_active_delay_count=torch.zeros(_POOL, dtype=torch.int32),
        _cg_active_eoc_countdown=torch.zeros(_POOL, dtype=torch.int32),
        _cg_active_generation_done=torch.zeros(_POOL, dtype=torch.bool),
        _cg_active_last_codes=torch.zeros((_POOL, _NCB), dtype=torch.long),
        _cg_active_seeds=torch.zeros(_POOL, dtype=torch.long),
        _cg_active_step_count=torch.zeros(_POOL, dtype=torch.long),
        _sampler_pool=pool,
        _padding_row=_PAD,
        _row_epoch=0,
        acquire_row=_acquire,
    )
    reqs = [SimpleNamespace(request_id=f"req{i}") for i in range(n_reqs)]
    return runner, reqs, acquire_calls


def test_decode_populate_reads_sampling_info_device_side_only():
    runner, reqs, _ = _make_runner()
    # Poison the buffers so the padding-tail assertions can't pass vacuously.
    runner.model._cg_temperature[:] = -9.0
    runner.model._cg_top_p[:] = -9.0
    runner.model._cg_top_k_buf[:] = -9
    fb = SimpleNamespace(
        batch_size=4,
        sampling_info=_sampling_info([0.7, 1.3, 0.9], [0.95, 0.8, 1.0], [-1, 5, 0]),
    )
    runner._populate_cg_buffers(fb, reqs)
    model = runner.model
    assert torch.allclose(
        model._cg_temperature[:3], torch.tensor([0.7, 1.3, 0.9], dtype=torch.float32)
    )
    assert torch.allclose(
        model._cg_top_p[:3], torch.tensor([0.95, 0.8, 1.0], dtype=torch.float32)
    )
    # top_k outside (0, K_MAX) (sentinels -1 / 0) normalizes to K_MAX = no-op.
    assert model._cg_top_k_buf[:4].tolist() == [K_MAX, 5, K_MAX, K_MAX]
    # Padding slot (position 3 of bs=4) holds neutral values.
    assert float(model._cg_temperature[3]) == 1.0
    assert float(model._cg_top_p[3]) == 1.0


def test_populate_without_sampling_info_uses_defaults():
    runner, reqs, _ = _make_runner()
    runner.model._cg_temperature[:] = 0.123
    runner.model._cg_top_k_buf[:] = 7
    fb = SimpleNamespace(batch_size=4, sampling_info=None)
    runner._populate_cg_buffers(fb, reqs)
    model = runner.model
    assert model._cg_temperature[:4].tolist() == [1.0] * 4
    assert model._cg_top_p[:4].tolist() == [1.0] * 4
    assert model._cg_top_k_buf[:4].tolist() == [K_MAX] * 4


def test_done_row_guard_keeps_position_params_but_padding_state():
    """The lookahead overrun guard redirects a done row's STATE gathers to the
    padding row; its sampling params still come from its batch position
    (legacy behavior)."""
    runner, reqs, _ = _make_runner()
    pool = runner.model._sampler_pool
    pool.generation_done[1] = True
    pool.last_codes[1] = 42
    fb = SimpleNamespace(
        batch_size=3,
        sampling_info=_sampling_info([0.7, 1.3, 0.9], [0.95, 0.8, 1.0], [7, 5, 2]),
    )
    runner._populate_cg_buffers(fb, reqs, is_lookahead=True)
    model = runner.model
    assert float(model._cg_temperature[1]) == pytest.approx(1.3)
    assert int(model._cg_top_k_buf[1]) == 5
    assert int(model._cg_row_indices[1]) == _PAD
    # State gather went through the padding row (zeros), not row 1's 42s.
    assert model._cg_active_last_codes[1].tolist() == [0, 0, 0]


def test_rows_cache_skips_reacquire_on_unchanged_membership():
    runner, reqs, acquire_calls = _make_runner()
    fb = SimpleNamespace(
        batch_size=4,
        sampling_info=_sampling_info([0.7, 1.3, 0.9], [0.95, 0.8, 1.0], [7, 5, 2]),
    )
    runner._populate_cg_buffers(fb, reqs)
    n_first = len(acquire_calls)
    runner._populate_cg_buffers(fb, reqs)
    assert len(acquire_calls) == n_first, "unchanged membership must not re-acquire"
    assert runner.model._cg_row_indices[:4].tolist() == [0, 1, 2, _PAD]


def test_rows_cache_restores_rows_mutated_by_guard():
    """An overrun step mutates ``_cg_row_indices`` in place (done row ->
    padding); the next populate with unchanged membership must restore the
    true rows before applying its own guard."""
    runner, reqs, _ = _make_runner()
    fb = SimpleNamespace(
        batch_size=3,
        sampling_info=_sampling_info([0.7, 1.3, 0.9], [0.95, 0.8, 1.0], [7, 5, 2]),
    )
    runner._populate_cg_buffers(fb, reqs, is_lookahead=True)
    pool = runner.model._sampler_pool
    pool.generation_done[1] = True
    runner._populate_cg_buffers(fb, reqs, is_lookahead=True)
    assert int(runner.model._cg_row_indices[1]) == _PAD
    pool.generation_done[1] = False
    runner._populate_cg_buffers(fb, reqs, is_lookahead=True)
    assert runner.model._cg_row_indices[:3].tolist() == [0, 1, 2]


def test_rows_cache_invalidated_on_row_release_epoch():
    """Same request-id list + bs can reappear after a row was released and
    remapped (request-id reuse); the release-side epoch must invalidate the
    cache so stale row mappings are never restored."""
    runner, reqs, _ = _make_runner()
    rid_to_row = _make_runner.rid_to_row
    fb = SimpleNamespace(
        batch_size=4,
        sampling_info=_sampling_info([0.7, 1.3, 0.9], [0.95, 0.8, 1.0], [7, 5, 2]),
    )
    runner._populate_cg_buffers(fb, reqs)
    assert runner.model._cg_row_indices[:3].tolist() == [0, 1, 2]
    # req1's row is released and a same-named request later lands on row 4.
    rid_to_row["req1"] = 4
    runner.model._row_epoch += 1
    runner._populate_cg_buffers(fb, reqs)
    assert runner.model._cg_row_indices[:3].tolist() == [0, 4, 2]


def test_empty_sampling_tensors_fall_back_to_defaults():
    """Zero-length sampling attrs behave like absent ones. For temperatures /
    top_ps this matches the removed path's falsy fallback; for top_ks the old
    path raised a shape mismatch (unreachable in serving), defaults are the
    deliberate replacement."""
    runner, reqs, _ = _make_runner()
    fb = SimpleNamespace(
        batch_size=4,
        sampling_info=SimpleNamespace(
            temperatures=torch.zeros((0, 1), dtype=torch.float32),
            top_ps=torch.zeros(0, dtype=torch.float32),
            top_ks=torch.zeros(0, dtype=torch.long),
        ),
    )
    runner._populate_cg_buffers(fb, reqs)
    model = runner.model
    assert model._cg_temperature[:4].tolist() == [1.0] * 4
    assert model._cg_top_p[:4].tolist() == [1.0] * 4
    assert model._cg_top_k_buf[:4].tolist() == [K_MAX] * 4


def test_list_sampling_attrs_still_supported():
    runner, reqs, _ = _make_runner()
    fb = SimpleNamespace(
        batch_size=3,
        sampling_info=SimpleNamespace(
            temperatures=[0.5, 0.6, 0.7], top_ps=[0.9, 0.8, 0.7], top_ks=[1, 2, 3]
        ),
    )
    runner._populate_cg_buffers(fb, reqs)
    assert torch.allclose(
        runner.model._cg_temperature[:3],
        torch.tensor([0.5, 0.6, 0.7], dtype=torch.float32),
    )
    assert runner.model._cg_top_k_buf[:3].tolist() == [1, 2, 3]


def test_rows_cache_invalidated_on_membership_change():
    runner, reqs, acquire_calls = _make_runner()
    fb = SimpleNamespace(
        batch_size=4,
        sampling_info=_sampling_info([0.7, 1.3, 0.9], [0.95, 0.8, 1.0], [7, 5, 2]),
    )
    runner._populate_cg_buffers(fb, reqs)
    survivors = [reqs[0], reqs[2]]
    fb2 = SimpleNamespace(
        batch_size=2,
        sampling_info=_sampling_info([0.7, 0.9], [0.95, 1.0], [7, 2]),
    )
    n_before = len(acquire_calls)
    runner._populate_cg_buffers(fb2, survivors)
    assert len(acquire_calls) > n_before
    assert runner.model._cg_row_indices[:2].tolist() == [0, 2]
    assert torch.allclose(
        runner.model._cg_temperature[:2],
        torch.tensor([0.7, 0.9], dtype=torch.float32),
    )


def test_model_release_row_bumps_epoch():
    """The model-side half of the cache invariant: release_row itself must
    bump ``_row_epoch`` (mutation-kill for the bump in model.py)."""
    from sglang_omni.models.higgs_tts.model import HiggsTTSModel

    stub = SimpleNamespace(
        _rid_to_row={"A": 3},
        _free_rows=[],
        _row_epoch=0,
        _output_codes={},
    )
    HiggsTTSModel.release_row(stub, "A")
    assert stub._row_epoch == 1
    assert stub._free_rows == [3]
    # Releasing an unknown id changes no mapping and must not bump.
    HiggsTTSModel.release_row(stub, "ghost")
    assert stub._row_epoch == 1
