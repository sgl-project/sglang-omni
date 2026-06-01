# SPDX-License-Identifier: Apache-2.0
"""Parity tests for HiggsTTSModelRunner's async-decode (one-step lookahead) path.

The async decode path splits the synchronous collect into two halves:

  - ``post_decode_launch``  : GPU pack + non-blocking D2H into a pinned host
                              buffer, plus a no-host-sync publish of
                              ``result.next_token_ids`` from on-GPU codes.
  - ``post_decode_resolve`` : host-side collect over the already-copied snapshot.

Both halves are claimed (model_runner.py docstrings) to "mirror the tail of
``_collect_step_outputs_cg``", i.e. to be semantically identical to the
synchronous path. These tests pin that claim: for an identical initial CG-buffer
state, the async path and the sync ``_collect_step_outputs_cg`` must produce the
same ``data.output_codes`` / ``data.generation_done`` / ``req.finished_reason`` /
``result.next_token_ids``.

CPU-only: ``BaseModelRunner._next_host_staging`` allocates a ``pin_memory=True``
host buffer (needs a CUDA context). We monkeypatch it to a plain CPU buffer so
the collect logic runs on a CPU-only box. A separate CUDA-guarded test exercises
the real pinned ping-pong buffer to confirm the monkeypatch is faithful.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.higgs_tts.model_runner import HiggsTTSModelRunner
from sglang_omni.models.higgs_tts.utils import EOC_ID


def _build_runner(
    *,
    codes_BN,
    was_done,
    active_generation_done,
    is_chunked,
    finished,
    async_enabled=False,
    n_codebooks=3,
):
    """Build a HiggsTTSModelRunner over a SimpleNamespace model, mirroring
    test_pipeline.py's CG fixtures. Each call returns a FRESH, independent
    fixture: ``_decode_pack_gpu`` scatters shadow state back into the sampler
    pool (mutating the model), so sync vs async parity must compare two
    independent fixtures seeded identically, never one model run twice.
    """
    n = len(codes_BN)
    runner = object.__new__(HiggsTTSModelRunner)
    runner._outbox = None
    runner._vocoder_target = "vocoder"
    # async-decode base-runner state (normally set in BaseModelRunner.__init__)
    runner._async_enabled = async_enabled
    runner._staging_slot = 0
    runner._host_staging_buffers = []
    runner._async_query_hit = 0
    runner._async_query_miss = 0
    runner.model = SimpleNamespace(
        _cg_row_indices=torch.arange(n),
        _cg_active_delay_count=torch.zeros(n, dtype=torch.int32),
        _cg_active_eoc_countdown=torch.zeros(n, dtype=torch.int32),
        _cg_active_generation_done=torch.tensor(active_generation_done),
        _cg_active_last_codes=torch.zeros((n, n_codebooks), dtype=torch.long),
        _cg_was_done=torch.tensor(was_done),
        _cg_codes_BN=torch.tensor(codes_BN),
        _cg_collect_staging=torch.zeros((n, n_codebooks + 2), dtype=torch.long),
        _sampler_pool=SimpleNamespace(
            delay_count=torch.zeros(n, dtype=torch.int32),
            eoc_countdown=torch.zeros(n, dtype=torch.int32),
            generation_done=torch.zeros(n, dtype=torch.bool),
            last_codes=torch.zeros((n, n_codebooks), dtype=torch.long),
        ),
    )
    reqs = [
        SimpleNamespace(
            is_chunked=is_chunked[i], finished_reason=None, finished=finished[i]
        )
        for i in range(n)
    ]
    datas = [
        SimpleNamespace(req=reqs[i], output_codes=[], generation_done=False)
        for i in range(n)
    ]
    sched = [SimpleNamespace(request_id=f"req{i}", data=datas[i]) for i in range(n)]
    result = SimpleNamespace(
        logits_output=SimpleNamespace(next_token_logits=torch.zeros(n, 4))
    )
    forward_batch = SimpleNamespace(batch_size=n)
    return runner, sched, result, forward_batch, reqs, datas


def _snapshot(reqs, datas, result):
    """Capture the four observable outputs for cross-path comparison."""
    return {
        "output_codes": [[c.tolist() for c in d.output_codes] for d in datas],
        "generation_done": [d.generation_done for d in datas],
        "finished_reason": [
            None if r.finished_reason is None else r.finished_reason.to_json()
            for r in reqs
        ],
        "next_token_ids": result.next_token_ids.tolist(),
    }


# A 4-row mixed batch: row0 chunked, row1 was-done(skip), row2 active(not done),
# row3 active(EOC done). Same shape as test_higgs_model_runner_collect_cg_mixed_batch.
_MIXED = dict(
    codes_BN=[[1, 1, 1], [7, 8, 9], [20, 1, 2], [EOC_ID, 3, 4]],
    was_done=[False, True, False, False],
    active_generation_done=[False, True, False, True],
    is_chunked=[1, 0, 0, 0],
    finished=[lambda: False, lambda: False, lambda: False, lambda: False],
)


def _patch_cpu_host_staging(monkeypatch):
    """Strip the pin_memory (CUDA) requirement from ``_next_host_staging``; the
    collect logic is dtype/shape identical on a plain CPU buffer. The CUDA-guarded
    test below proves this monkeypatch does not mask a pinned-path-specific bug.
    """
    monkeypatch.setattr(
        HiggsTTSModelRunner,
        "_next_host_staging",
        lambda self, device_staging: torch.empty(
            device_staging.shape, dtype=device_staging.dtype, device="cpu"
        ),
    )


def _run_sync(**kw):
    runner, sched, result, fb, reqs, datas = _build_runner(async_enabled=False, **kw)
    runner._collect_step_outputs_cg(result, fb, sched)
    return _snapshot(reqs, datas, result)


def _run_async(monkeypatch, **kw):
    runner, sched, result, fb, reqs, datas = _build_runner(async_enabled=True, **kw)
    _patch_cpu_host_staging(monkeypatch)
    host_buf = runner.post_decode_launch(result, fb, sched)
    # The base runner records a CUDA event here; CPU-only we just hand the
    # already-copied snapshot straight to resolve.
    runner.post_decode_resolve(host_buf, result, fb, None, sched)
    return _snapshot(reqs, datas, result)


def test_async_matches_sync_mixed_batch(monkeypatch):
    """Core parity: async (launch+resolve) == sync collect on a mixed batch."""
    sync = _run_sync(**_MIXED)
    asy = _run_async(monkeypatch, **_MIXED)

    # Lock the expected sync values first (regression anchor independent of async).
    assert sync["output_codes"] == [[], [], [[20, 1, 2]], [[EOC_ID, 3, 4]]]
    assert sync["generation_done"] == [False, False, False, True]
    assert sync["finished_reason"] == [
        None,
        None,
        None,
        {"type": "stop", "matched": EOC_ID},
    ]
    assert sync["next_token_ids"] == [0, 0, 20, EOC_ID]

    # Then assert the async path is byte-for-byte the same on all four outputs.
    assert asy == sync


def test_async_next_token_ids_published_at_launch(monkeypatch):
    """AC2: post_decode_launch publishes next_token_ids from on-GPU codes
    (``_cg_codes_BN[:, 0].clamp_min(0)``) WITHOUT a host sync, before resolve
    runs. clamp_min keeps STOP_CODE(-1) rows in embed range.
    """
    kw = dict(
        codes_BN=[[5, 1, 2], [-1, 3, 4]],  # row1 cb0 = -1 (STOP) -> clamp to 0
        was_done=[False, False],
        active_generation_done=[False, True],
        is_chunked=[0, 0],
        finished=[lambda: False, lambda: False],
    )
    runner, sched, result, fb, reqs, datas = _build_runner(async_enabled=True, **kw)
    _patch_cpu_host_staging(monkeypatch)
    runner.post_decode_launch(result, fb, sched)
    # Published immediately at launch, from clamp_min(0) of codebook-0.
    assert result.next_token_ids.tolist() == [5, 0]
    assert result.next_token_ids.dtype == torch.long


def test_async_resolve_overrun_guard_skips_finished_row(monkeypatch):
    """AC3: under lookahead, a row already finished() at an earlier step gets
    one wasted forward; resolve must skip its append (no overrun token leak,
    no double-collect), matching the sync path's finished() skip.
    """
    # row0 active, row1 already finished() (the overrun row).
    kw = dict(
        codes_BN=[[11, 1, 2], [22, 3, 4]],
        was_done=[False, False],
        active_generation_done=[False, False],
        is_chunked=[0, 0],
        finished=[lambda: False, lambda: True],
    )
    sync = _run_sync(**kw)
    asy = _run_async(monkeypatch, **kw)
    # row1 finished -> skipped in BOTH paths: no codes, cb0 reported as 0.
    assert sync["output_codes"] == [[[11, 1, 2]], []]
    assert sync["next_token_ids"] == [11, 0]
    assert asy == sync


def test_async_matches_sync_bs1_active(monkeypatch):
    """Parity at bs=1 (the size that the scheduler routes to the SYNC fast path
    via async_decode_min_batch_size=2). The collect itself must still agree, so
    a future default-flip can't silently change bs=1 behavior.
    """
    kw = dict(
        codes_BN=[[9, 8, 7]],
        was_done=[False],
        active_generation_done=[False],
        is_chunked=[0],
        finished=[lambda: False],
    )
    sync = _run_sync(**kw)
    asy = _run_async(monkeypatch, **kw)
    assert sync["output_codes"] == [[[9, 8, 7]]]
    assert sync["next_token_ids"] == [9]
    assert asy == sync


def test_async_matches_sync_bs1_eoc(monkeypatch):
    """bs=1 where the single row finishes via EOC: generation_done + finish
    reason must propagate identically through the async path."""
    kw = dict(
        codes_BN=[[EOC_ID, 1, 2]],
        was_done=[False],
        active_generation_done=[True],
        is_chunked=[0],
        finished=[lambda: False],
    )
    sync = _run_sync(**kw)
    asy = _run_async(monkeypatch, **kw)
    assert sync["generation_done"] == [True]
    assert sync["finished_reason"] == [{"type": "stop", "matched": EOC_ID}]
    assert asy == sync


def _pick_free_cuda_device(min_free_mib: int = 512) -> str | None:
    """Return the first CUDA device with at least ``min_free_mib`` free, else
    None. Avoids OOM on shared boxes where some GPUs already host a server."""
    if not torch.cuda.is_available():
        return None
    for i in range(torch.cuda.device_count()):
        try:
            free, _ = torch.cuda.mem_get_info(i)
        except Exception:
            continue
        if free // (1024 * 1024) >= min_free_mib:
            return f"cuda:{i}"
    return None


def test_async_real_pinned_path_matches_sync():
    """CUDA-guarded: run the async path through the REAL _next_host_staging
    (pinned host buffer + non-blocking copy on a CUDA model) and confirm it
    still equals the sync collect. Proves the CPU monkeypatch above is faithful.
    Picks a GPU with free memory (the box may already host a TTS server).
    """
    dev = _pick_free_cuda_device()
    if dev is None:
        pytest.skip("no CUDA device with free memory for pinned D2H test")
    # pin_memory uses the DEFAULT CUDA context (device 0). On a shared box where
    # device 0 already hosts a server, that allocation OOMs even though `dev` is
    # free. Pin the default device to the free GPU for this test's lifetime.
    torch.cuda.set_device(dev)

    def build(async_enabled):
        n = 4
        runner = object.__new__(HiggsTTSModelRunner)
        runner._outbox = None
        runner._vocoder_target = "vocoder"
        runner._async_enabled = async_enabled
        runner._staging_slot = 0
        runner._host_staging_buffers = []
        runner._async_query_hit = 0
        runner._async_query_miss = 0
        runner.model = SimpleNamespace(
            _cg_row_indices=torch.arange(n, device=dev),
            _cg_active_delay_count=torch.zeros(n, dtype=torch.int32, device=dev),
            _cg_active_eoc_countdown=torch.zeros(n, dtype=torch.int32, device=dev),
            _cg_active_generation_done=torch.tensor(
                [False, True, False, True], device=dev
            ),
            _cg_active_last_codes=torch.zeros((n, 3), dtype=torch.long, device=dev),
            _cg_was_done=torch.tensor([False, True, False, False], device=dev),
            _cg_codes_BN=torch.tensor(
                [[1, 1, 1], [7, 8, 9], [20, 1, 2], [EOC_ID, 3, 4]], device=dev
            ),
            _cg_collect_staging=torch.zeros((n, 3 + 2), dtype=torch.long, device=dev),
            _sampler_pool=SimpleNamespace(
                delay_count=torch.zeros(n, dtype=torch.int32, device=dev),
                eoc_countdown=torch.zeros(n, dtype=torch.int32, device=dev),
                generation_done=torch.zeros(n, dtype=torch.bool, device=dev),
                last_codes=torch.zeros((n, 3), dtype=torch.long, device=dev),
            ),
        )
        reqs = [
            SimpleNamespace(is_chunked=c, finished_reason=None, finished=lambda: False)
            for c in (1, 0, 0, 0)
        ]
        datas = [
            SimpleNamespace(req=reqs[i], output_codes=[], generation_done=False)
            for i in range(n)
        ]
        sched = [SimpleNamespace(request_id=f"req{i}", data=datas[i]) for i in range(n)]
        result = SimpleNamespace(
            logits_output=SimpleNamespace(
                next_token_logits=torch.zeros(n, 4, device=dev)
            )
        )
        fb = SimpleNamespace(batch_size=n)
        return runner, sched, result, fb, reqs, datas

    r_s, sc_s, res_s, fb_s, rq_s, dt_s = build(False)
    r_s._collect_step_outputs_cg(res_s, fb_s, sc_s)
    sync = _snapshot(rq_s, dt_s, res_s)

    r_a, sc_a, res_a, fb_a, rq_a, dt_a = build(True)
    host_buf = r_a.post_decode_launch(res_a, fb_a, sc_a)
    torch.cuda.synchronize()  # stand in for the base runner's recorded event
    r_a.post_decode_resolve(host_buf, res_a, fb_a, None, sc_a)
    asy = _snapshot(rq_a, dt_a, res_a)

    assert asy == sync
