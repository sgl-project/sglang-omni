# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the sync-free decode-launch cache
(``HiggsTTSModelRunner._populate_cg_buffers`` +
``SGLANG_OMNI_SYNCFREE_LAUNCH``).

Covers: hit skips the sampling-param extraction/upload; every
composition-change shape (admission, finish/removal, reorder, padded-bs
change, pool-row re-allocation) misses; the lookahead done-row guard's
padding redirect persists across hits exactly like the rebuild-every-step
baseline; per-step pool gathers still run on hits; flag off == always
rebuild. All structures are CPU mocks — no CUDA, no real model.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.higgs_tts.model_runner import (
    HiggsTTSModelRunner,
    _syncfree_launch_enabled,
)
from sglang_omni.models.higgs_tts.sampler import K_MAX, NO_SEED

POOL = 8  # pool size incl. the reserved padding row
N_CB = 4
PAD_ROW = POOL - 1


def _make_pool():
    pool = SimpleNamespace(
        delay_count=torch.zeros(POOL, dtype=torch.int32),
        eoc_countdown=torch.full((POOL,), -1, dtype=torch.int32),
        generation_done=torch.zeros(POOL, dtype=torch.bool),
        last_codes=torch.zeros(POOL, N_CB, dtype=torch.long),
        seeds=torch.full((POOL,), NO_SEED, dtype=torch.long),
        step_count=torch.zeros(POOL, dtype=torch.long),
    )

    def reset_row(row: int) -> None:
        pool.delay_count[row] = 0
        pool.eoc_countdown[row] = -1
        pool.generation_done[row] = False
        pool.last_codes[row].zero_()
        pool.seeds[row] = NO_SEED
        pool.step_count[row] = 0

    pool.reset_row = reset_row
    return pool


class _FakeModel:
    """Mimics the HiggsTTSModel surface _populate_cg_buffers touches."""

    def __init__(self) -> None:
        self._sampler_pool = _make_pool()
        self._padding_row = PAD_ROW
        self._rid_to_row: dict[str, int] = {}
        self._free_rows = list(range(PAD_ROW))
        self._cg_row_indices = torch.zeros(POOL, dtype=torch.long)
        self._cg_temperature = torch.ones(POOL, dtype=torch.float32)
        self._cg_top_p = torch.ones(POOL, dtype=torch.float32)
        self._cg_top_k_buf = torch.full((POOL,), K_MAX, dtype=torch.long)
        self._cg_active_delay_count = torch.zeros(POOL, dtype=torch.int32)
        self._cg_active_eoc_countdown = torch.full((POOL,), -1, dtype=torch.int32)
        self._cg_active_generation_done = torch.zeros(POOL, dtype=torch.bool)
        self._cg_active_last_codes = torch.zeros(POOL, N_CB, dtype=torch.long)
        self._cg_active_seeds = torch.full((POOL,), NO_SEED, dtype=torch.long)
        self._cg_active_step_count = torch.zeros(POOL, dtype=torch.long)

    def acquire_row(self, rid: str) -> int:
        row = self._rid_to_row.get(rid)
        if row is not None:
            return row
        row = self._free_rows.pop()
        self._rid_to_row[rid] = row
        self._sampler_pool.reset_row(row)
        return row

    def release_row(self, rid: str) -> None:
        row = self._rid_to_row.pop(rid, None)
        if row is not None:
            self._free_rows.append(row)


def _make_runner(model, *, enabled=True, async_enabled=False):
    runner = object.__new__(HiggsTTSModelRunner)
    runner.model = model
    runner._async_enabled = async_enabled
    runner._syncfree_launch = enabled
    runner._cg_launch_key = None
    calls = {"extract": 0}
    orig = HiggsTTSModelRunner._extract_decode_sampling_params

    def counting_extract(forward_batch, n_real):
        calls["extract"] += 1
        return orig(forward_batch, n_real)

    runner._extract_decode_sampling_params = counting_extract
    return runner, calls


def _reqs(*rids):
    return [SimpleNamespace(request_id=r) for r in rids]


def _fb(bs, temps, top_ps, top_ks):
    return SimpleNamespace(
        batch_size=bs,
        sampling_info=SimpleNamespace(
            temperatures=torch.tensor(temps, dtype=torch.float32),
            top_ps=torch.tensor(top_ps, dtype=torch.float32),
            top_ks=torch.tensor(top_ks, dtype=torch.long),
        ),
    )


AB_FB = ([0.7, 0.9], [0.95, 0.8], [50, 0])  # top_k=0 -> None -> K_MAX


def test_hit_skips_extract_and_keeps_buffers():
    model = _FakeModel()
    runner, calls = _make_runner(model)
    fb = _fb(2, *AB_FB)
    reqs = _reqs("a", "b")

    runner._populate_cg_buffers(fb, reqs)
    assert calls["extract"] == 1
    snap = {
        "rows": model._cg_row_indices.clone(),
        "temp": model._cg_temperature.clone(),
        "top_p": model._cg_top_p.clone(),
        "top_k": model._cg_top_k_buf.clone(),
    }
    assert snap["temp"][:2].tolist() == pytest.approx([0.7, 0.9])
    assert snap["top_p"][:2].tolist() == pytest.approx([0.95, 0.8])
    assert snap["top_k"][:2].tolist() == [50, K_MAX]

    for _ in range(3):
        runner._populate_cg_buffers(fb, reqs)
    assert calls["extract"] == 1  # all hits
    assert torch.equal(model._cg_row_indices, snap["rows"])
    assert torch.equal(model._cg_temperature, snap["temp"])
    assert torch.equal(model._cg_top_p, snap["top_p"])
    assert torch.equal(model._cg_top_k_buf, snap["top_k"])


def test_pool_gathers_still_run_on_hit():
    model = _FakeModel()
    runner, _ = _make_runner(model)
    fb = _fb(2, *AB_FB)
    reqs = _reqs("a", "b")
    runner._populate_cg_buffers(fb, reqs)
    row_a = model._rid_to_row["a"]
    # Note (Yueying Li): simulate the previous step's scatter mutating pool state.
    model._sampler_pool.delay_count[row_a] = 5
    model._sampler_pool.step_count[row_a] = 7
    runner._populate_cg_buffers(fb, reqs)  # hit
    assert int(model._cg_active_delay_count[0]) == 5
    assert int(model._cg_active_step_count[0]) == 7


def test_miss_on_admission_finish_reorder_bs_and_rows():
    model = _FakeModel()
    runner, calls = _make_runner(model)

    fb_a = _fb(1, [0.7], [0.95], [50])
    runner._populate_cg_buffers(fb_a, _reqs("a"))
    assert calls["extract"] == 1

    # Admission: [a] -> [a, b]
    fb_ab = _fb(2, *AB_FB)
    runner._populate_cg_buffers(fb_ab, _reqs("a", "b"))
    assert calls["extract"] == 2
    assert model._cg_temperature[:2].tolist() == pytest.approx([0.7, 0.9])

    # Reorder: [a, b] -> [b, a] (order-sensitive key)
    fb_ba = _fb(2, [0.9, 0.7], [0.8, 0.95], [0, 50])
    runner._populate_cg_buffers(fb_ba, _reqs("b", "a"))
    assert calls["extract"] == 3
    assert model._cg_temperature[:2].tolist() == pytest.approx([0.9, 0.7])
    assert model._cg_top_k_buf[:2].tolist() == [K_MAX, 50]

    # Finish/removal: [b, a] -> [b]
    fb_b = _fb(1, [0.9], [0.8], [0])
    runner._populate_cg_buffers(fb_b, _reqs("b"))
    assert calls["extract"] == 4
    assert model._cg_row_indices[0] == model._rid_to_row["b"]

    # Padded-bs change with same composition: bs 1 -> 3
    fb_b_pad = _fb(3, [0.9], [0.8], [0])
    runner._populate_cg_buffers(fb_b_pad, _reqs("b"))
    assert calls["extract"] == 5
    assert model._cg_row_indices[1:3].tolist() == [PAD_ROW, PAD_ROW]
    assert model._cg_temperature[1:3].tolist() == [1.0, 1.0]
    assert model._cg_top_k_buf[1:3].tolist() == [K_MAX, K_MAX]

    # Note (Yueying Li): row re-allocation under an identical rid tuple: retract-like release,
    # another request steals the row, then "b" is re-admitted.
    old_row = model._rid_to_row["b"]
    model.release_row("b")
    assert model.acquire_row("z") == old_row  # steals b's row
    runner._populate_cg_buffers(fb_b_pad, _reqs("b"))  # same rids, same bs
    assert calls["extract"] == 6  # row change -> miss
    assert model._cg_row_indices[0] == model._rid_to_row["b"]
    assert model._rid_to_row["b"] != old_row


def test_flag_off_always_rebuilds():
    model = _FakeModel()
    runner, calls = _make_runner(model, enabled=False)
    fb = _fb(2, *AB_FB)
    reqs = _reqs("a", "b")
    for _ in range(3):
        runner._populate_cg_buffers(fb, reqs)
    assert calls["extract"] == 3


def test_lookahead_guard_redirect_persists_and_matches_baseline():
    def run_steps(enabled):
        model = _FakeModel()
        runner, calls = _make_runner(model, enabled=enabled, async_enabled=True)
        fb = _fb(2, *AB_FB)
        reqs = _reqs("a", "b")
        rows_per_step = []
        runner._populate_cg_buffers(fb, reqs, is_lookahead=True)
        rows_per_step.append(model._cg_row_indices[:2].tolist())
        # Note (Yueying Li): request "b" finishes via EOC on-GPU: the step's scatter marks its
        # pool row done while the host has not yet filtered it out.
        model._sampler_pool.generation_done[model._rid_to_row["b"]] = True
        for _ in range(2):  # overrun steps, composition unchanged
            runner._populate_cg_buffers(fb, reqs, is_lookahead=True)
            rows_per_step.append(model._cg_row_indices[:2].tolist())
        # Host drops "b": composition changes -> rebuild.
        model.release_row("b")
        fb_a = _fb(1, [0.7], [0.95], [50])
        runner._populate_cg_buffers(fb_a, _reqs("a"), is_lookahead=True)
        rows_per_step.append(model._cg_row_indices[:1].tolist())
        return rows_per_step, model._rid_to_row["a"], calls["extract"]

    cached_rows, row_a, cached_extracts = run_steps(enabled=True)
    baseline_rows, _, baseline_extracts = run_steps(enabled=False)

    assert cached_rows == baseline_rows  # bit-identical row routing
    assert cached_rows[1] == [row_a, PAD_ROW]  # guard redirected done row
    assert cached_rows[2] == [row_a, PAD_ROW]  # redirect sticky across hits
    assert cached_rows[3] == [row_a]  # rebuild after composition change
    assert cached_extracts == 2  # initial + post-removal rebuild
    assert baseline_extracts == 4  # every step


def test_rid_reuse_on_same_row_misses():
    # Note (Yueying Li): client-supplied rids may be reused after a request finishes; LIFO row
    # recycling then reproduces the exact (rid, row, bs) key of the finished
    # request. A fresh acquisition must force a rebuild or the new request
    # inherits the old params and a stale padding-row redirect.
    model = _FakeModel()
    runner, calls = _make_runner(model, async_enabled=True)
    fb_old = _fb(1, [0.3], [0.5], [10])
    runner._populate_cg_buffers(fb_old, _reqs("x"), is_lookahead=True)
    assert calls["extract"] == 1
    row = model._rid_to_row["x"]

    # "x" EOC-finishes on-GPU; overrun step redirects its slot to padding.
    model._sampler_pool.generation_done[row] = True
    runner._populate_cg_buffers(fb_old, _reqs("x"), is_lookahead=True)
    assert calls["extract"] == 1  # hit, as in baseline
    assert int(model._cg_row_indices[0]) == PAD_ROW

    # Host drops "x"; LIFO hands the same row to the reincarnated rid "x".
    model.release_row("x")
    model._sampler_pool.generation_done[row] = False
    fb_new = _fb(1, [0.9], [0.8], [50])
    runner._populate_cg_buffers(fb_new, _reqs("x"), is_lookahead=True)
    assert model._rid_to_row["x"] == row  # same row: key would collide
    assert calls["extract"] == 2  # fresh acquisition -> MISS
    assert int(model._cg_row_indices[0]) == row  # redirect cleared
    assert float(model._cg_temperature[0]) == pytest.approx(0.9)
    assert int(model._cg_top_k_buf[0]) == 50


def test_env_flag_parsing(monkeypatch):
    monkeypatch.delenv("SGLANG_OMNI_SYNCFREE_LAUNCH", raising=False)
    assert _syncfree_launch_enabled() is True  # default ON
    monkeypatch.setenv("SGLANG_OMNI_SYNCFREE_LAUNCH", "0")
    assert _syncfree_launch_enabled() is False
    monkeypatch.setenv("SGLANG_OMNI_SYNCFREE_LAUNCH", "1")
    assert _syncfree_launch_enabled() is True
    monkeypatch.setenv("SGLANG_OMNI_SYNCFREE_LAUNCH", "bogus")
    with pytest.raises(ValueError):
        _syncfree_launch_enabled()
