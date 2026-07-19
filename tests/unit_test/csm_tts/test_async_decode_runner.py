# SPDX-License-Identifier: Apache-2.0
"""Parity tests for CsmTTSModelRunner's async-decode (one-step lookahead) path.

The async path splits the synchronous collect into two halves:

  - ``post_decode_launch``  : GPU pack + non-blocking D2H into a pinned host
                              buffer, plus a no-host-sync publish of
                              ``result.next_token_ids`` from on-GPU codes.
  - ``post_decode_resolve`` : host-side collect over the already-copied snapshot.

Both halves must be semantically identical to the synchronous
``_collect_step_outputs_cg``: for an identical initial CG-buffer state, they must
produce the same ``data.output_frames`` / ``data.generation_done`` /
``req.finished_reason`` / ``result.next_token_ids``.

Async decode is dormant by default (``async_decode_min_batch_size=2``); these
tests pin the parity contract so a future default-flip cannot silently change
behavior. CPU-only: ``_next_host_staging`` allocates a ``pin_memory=True`` buffer
(needs a CUDA context), so it is monkeypatched to a plain CPU buffer; a separate
CUDA-guarded test exercises the real pinned path."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.csm_tts.model_runner import CsmTTSModelRunner
from sglang_omni.models.csm_tts.utils import CODEBOOK_EOS, NUM_CODEBOOKS


def _row(cb0: int) -> list[int]:
    """A full 32-codebook frame with codebook-0 = ``cb0`` (rest = 1)."""
    return [cb0] + [1] * (NUM_CODEBOOKS - 1)


def _build_runner(
    *,
    codes_BN,
    was_done,
    active_generation_done,
    is_chunked,
    finished,
    async_enabled=False,
    device="cpu",
):
    """Fresh CsmTTSModelRunner over a SimpleNamespace model, mirroring the CG
    fixtures. Each call returns an independent fixture: ``_decode_pack_gpu``
    scatters shadow state back into the sampler pool, so sync-vs-async parity
    must compare two independently seeded fixtures, never one model run twice."""
    n = len(codes_BN)
    runner = object.__new__(CsmTTSModelRunner)
    runner._outbox = None
    runner._vocoder_target = "vocoder"
    runner._async_enabled = async_enabled
    runner._staging_slot = 0
    runner._host_staging_buffers = []
    runner._async_query_hit = 0
    runner._async_query_miss = 0
    runner.model = SimpleNamespace(
        _cg_row_indices=torch.arange(n, device=device),
        _cg_active_generation_done=torch.tensor(active_generation_done, device=device),
        _cg_active_last_codes=torch.zeros(
            (n, NUM_CODEBOOKS), dtype=torch.long, device=device
        ),
        _cg_was_done=torch.tensor(was_done, device=device),
        _cg_codes_BN=torch.tensor(codes_BN, dtype=torch.long, device=device),
        _cg_collect_staging=torch.zeros(
            (n, NUM_CODEBOOKS + 2), dtype=torch.long, device=device
        ),
        _sampler_pool=SimpleNamespace(
            generation_done=torch.zeros(n, dtype=torch.bool, device=device),
            last_codes=torch.zeros((n, NUM_CODEBOOKS), dtype=torch.long, device=device),
            frames_emitted=torch.zeros(n, dtype=torch.int32, device=device),
        ),
    )
    reqs = [
        SimpleNamespace(
            is_chunked=is_chunked[i], finished_reason=None, finished=finished[i]
        )
        for i in range(n)
    ]
    datas = [
        SimpleNamespace(
            req=reqs[i], output_frames=[], generation_done=False, stream_metadata=None
        )
        for i in range(n)
    ]
    sched = [SimpleNamespace(request_id=f"req{i}", data=datas[i]) for i in range(n)]
    result = SimpleNamespace(
        logits_output=SimpleNamespace(
            next_token_logits=torch.zeros(n, 4, device=device)
        )
    )
    forward_batch = SimpleNamespace(batch_size=n)
    return runner, sched, result, forward_batch, reqs, datas


def _snapshot(reqs, datas, result):
    return {
        "output_frames": [[c.tolist() for c in d.output_frames] for d in datas],
        "generation_done": [d.generation_done for d in datas],
        "finished_reason": [
            None if r.finished_reason is None else r.finished_reason.matched
            for r in reqs
        ],
        "next_token_ids": result.next_token_ids.tolist(),
    }


# 4-row mixed batch: row0 chunked, row1 was-done (skip), row2 active (not done),
# row3 active newly-done (EOS this frame).
_MIXED = dict(
    codes_BN=[_row(1), _row(7), _row(20), _row(5)],
    was_done=[False, True, False, False],
    active_generation_done=[False, True, False, True],
    is_chunked=[1, 0, 0, 0],
    finished=[lambda: False] * 4,
)


def _patch_cpu_host_staging(monkeypatch):
    monkeypatch.setattr(
        CsmTTSModelRunner,
        "_next_host_staging",
        lambda self, dev_staging: torch.empty(
            dev_staging.shape, dtype=dev_staging.dtype, device="cpu"
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
    runner.post_decode_resolve(host_buf, result, fb, None, sched)
    return _snapshot(reqs, datas, result)


def test_async_matches_sync_mixed_batch(monkeypatch):
    """Core parity: async (launch + resolve) == sync collect on a mixed batch."""
    sync = _run_sync(**_MIXED)
    asy = _run_async(monkeypatch, **_MIXED)
    # Regression anchor on the sync path, independent of async.
    assert sync["output_frames"] == [[], [], [_row(20)], [_row(5)]]
    assert sync["generation_done"] == [False, False, False, True]
    assert sync["finished_reason"] == [None, None, None, CODEBOOK_EOS]
    assert sync["next_token_ids"] == [0, 0, 20, 5]
    # Async must be byte-for-byte identical on all four outputs.
    assert asy == sync


def test_async_next_token_ids_published_at_launch(monkeypatch):
    """post_decode_launch publishes next_token_ids from on-GPU codebook-0
    (``_cg_codes_BN[:, 0].clamp_min(0)``) before resolve; clamp keeps a
    STOP_CODE (-1) row in embed range."""
    kw = dict(
        codes_BN=[_row(5), [-1] * NUM_CODEBOOKS],
        was_done=[False, True],
        active_generation_done=[False, True],
        is_chunked=[0, 0],
        finished=[lambda: False, lambda: False],
    )
    runner, sched, result, fb, reqs, datas = _build_runner(async_enabled=True, **kw)
    _patch_cpu_host_staging(monkeypatch)
    runner.post_decode_launch(result, fb, sched)
    assert result.next_token_ids.tolist() == [5, 0]
    assert result.next_token_ids.dtype == torch.long


def test_async_matches_sync_bs1_active(monkeypatch):
    """Parity at bs=1 (the size the scheduler routes to the SYNC fast path);
    the collect must still agree so a future default-flip is safe."""
    kw = dict(
        codes_BN=[_row(9)],
        was_done=[False],
        active_generation_done=[False],
        is_chunked=[0],
        finished=[lambda: False],
    )
    sync = _run_sync(**kw)
    asy = _run_async(monkeypatch, **kw)
    assert sync["output_frames"] == [[_row(9)]]
    assert sync["next_token_ids"] == [9]
    assert asy == sync


def test_async_matches_sync_bs1_eos(monkeypatch):
    """bs=1 finishing via EOS: generation_done + finish reason propagate
    identically through the async path."""
    kw = dict(
        codes_BN=[_row(0)],
        was_done=[False],
        active_generation_done=[True],
        is_chunked=[0],
        finished=[lambda: False],
    )
    sync = _run_sync(**kw)
    asy = _run_async(monkeypatch, **kw)
    assert sync["generation_done"] == [True]
    assert sync["finished_reason"] == [CODEBOOK_EOS]
    assert asy == sync


def _free_cuda_device(min_free_mib: int = 512):
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
    """CUDA-guarded: run the async path through the REAL pinned ``_next_host_staging``
    (non-blocking D2H on a CUDA model) and confirm it still equals sync. Proves
    the CPU monkeypatch above does not mask a pinned-path bug."""
    dev = _free_cuda_device()
    if dev is None:
        pytest.skip("no CUDA device with free memory for the pinned D2H test")
    torch.cuda.set_device(dev)
    sync = _run_sync(device=dev, **_MIXED)

    runner, sched, result, fb, reqs, datas = _build_runner(
        async_enabled=True, device=dev, **_MIXED
    )
    host_buf = runner.post_decode_launch(result, fb, sched)
    torch.cuda.synchronize()  # stand in for the base runner's recorded event
    runner.post_decode_resolve(host_buf, result, fb, None, sched)
    assert _snapshot(reqs, datas, result) == sync
