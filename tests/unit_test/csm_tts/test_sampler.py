# SPDX-License-Identifier: Apache-2.0
"""Unit tests for csm_tts.sampler (CPU, no GPU) plus the
frame-0-EOS zero-decode lifecycle at the runner level (sglang stubbed
via conftest on machines without it)."""

from __future__ import annotations

import queue
from types import SimpleNamespace

import torch

from sglang_omni.models.csm_tts import sampler
from sglang_omni.models.csm_tts.utils import (
    CODEBOOK_EOS,
    K_MAX,
    NUM_CODEBOOKS,
    STOP_CODE,
)

torch.manual_seed(0)


def test_step_reference_vs_frame_finalize_direct_parity() -> None:
    """Per-row ``step_reference`` vs batched ``frame_finalize_direct`` across
    phases: running / eos_now / done-freeze / mixed batch."""
    for trial in range(50):
        bsz = int(torch.randint(1, 9, ()).item())
        codes = torch.randint(0, 2051, (bsz, 32), dtype=torch.int64)
        for r in range(bsz):
            mode = r % 4
            if mode == 1:
                codes[r] = 0  # all-32-zero EOS
            elif mode == 2:
                codes[r] = 0
                codes[r, 31] = int(torch.randint(1, 2048, ()).item())  # cb31-only
        done = torch.rand(bsz) < 0.4
        b_out, b_new, b_was = sampler.frame_finalize_direct(codes, done)
        for r in range(bsz):
            r_out, r_new, r_was = sampler.step_reference(codes[r], bool(done[r]))
            assert torch.equal(b_out[r], r_out), f"codes trial {trial} row {r}"
            assert bool(b_new[r]) == r_new, f"new_done trial {trial} row {r}"
            assert bool(b_was[r]) == r_was, f"was_done trial {trial} row {r}"


def test_eos_uses_cb0_to_cb30_only() -> None:
    """EOS covers codebooks 0..30 only — a frame with cb0..30 == 0 and
    cb31 != 0 STILL stops (HF-exact stop(31) semantics); any nonzero inside
    cb0..30 does not."""
    not_done = torch.zeros(4, dtype=torch.bool)
    codes = torch.zeros(4, 32, dtype=torch.int64)
    codes[1, 31] = 7  # EOS despite cb31 != 0
    codes[2, 5] = 9  # NOT EOS (cb5 nonzero)
    codes[3] = torch.arange(1, 33)  # NOT EOS
    out, new_done, was_done = sampler.frame_finalize_direct(codes, not_done)
    assert new_done.tolist() == [True, True, False, False]
    assert was_done.tolist() == [False] * 4
    # Non-frozen rows pass codes through — the EOS frame itself is recorded
    # (HF ``sequences`` parity; the vocoder never receives it).
    assert torch.equal(out, codes)


def test_stop_sentinel_freeze() -> None:
    """Done rows emit STOP_CODE=-1 and stay frozen on subsequent frames; the
    ``was_done`` snapshot survives the caller scattering ``new_done`` back
    into the same buffer."""
    done = torch.tensor([True, True, False])
    codes = torch.randint(1, 2048, (3, 32), dtype=torch.int64)
    out, new_done, was_done = sampler.frame_finalize_direct(codes, done)
    assert (out[:2] == STOP_CODE).all()
    assert torch.equal(out[2], codes[2])
    assert new_done.tolist() == [True, True, False]
    assert was_done.tolist() == [True, True, False]

    # all-zero frame finishes the running row; frozen rows stay frozen
    out, new_done, _ = sampler.frame_finalize_direct(
        torch.zeros(3, 32, dtype=torch.int64), new_done
    )
    assert new_done.tolist() == [True, True, True]
    assert (out[:2] == STOP_CODE).all() and (out[2] == 0).all()

    # was_done must not alias the live done buffer (model.py scatters into it)
    buf = torch.tensor([False, True])
    _, nd, wd = sampler.frame_finalize_direct(
        torch.zeros(2, 32, dtype=torch.int64), buf
    )
    buf.copy_(nd)
    assert wd.tolist() == [False, True]


def test_greedy_short_circuit_determinism() -> None:
    """temp=0 / top_k=1 rows short-circuit branchlessly to argmax over RAW
    logits — bitwise deterministic across calls, immune to top_p."""
    B = 5
    zeros_t = torch.zeros(B)
    neutral_k = torch.full((B,), K_MAX, dtype=torch.int64)
    for step in range(32):  # cb0 + 31 depth steps
        logits = torch.randn(B, K_MAX, dtype=torch.float32) * 4.0
        out = sampler.sample_codes_batched(logits, zeros_t, neutral_k)
        assert torch.equal(out, logits.argmax(dim=-1)), f"step {step}"
        assert out.dtype == torch.int64

    logits = torch.randn(B, K_MAX)
    a = sampler.sample_codes_batched(logits, zeros_t, neutral_k)
    b = sampler.sample_codes_batched(logits, zeros_t, neutral_k)
    assert torch.equal(a, b)

    # top_k == 1 at temp > 0 also short-circuits to argmax
    hot_t = torch.full((B,), 0.9)
    one_k = torch.ones(B, dtype=torch.int64)
    out = sampler.sample_codes_batched(logits, hot_t, one_k)
    assert torch.equal(out, logits.argmax(dim=-1))

    # greedy rows unaffected by a tight top_p
    out = sampler.sample_codes_batched(
        logits, zeros_t, neutral_k, torch.full((B,), 0.1)
    )
    assert torch.equal(out, logits.argmax(dim=-1))


def test_fixed_shape_topk_filter_vs_naive() -> None:
    """Fixed-shape ``topk(K_MAX)`` + per-row k-th-value gather keeps sampled
    codes inside each row's naive top-k support for mixed per-row k; greedy
    rows in the same batch still equal argmax."""
    B = 5
    logits = torch.randn(B, K_MAX, dtype=torch.float32) * 4.0
    mix_t = torch.tensor([0.0, 0.9, 0.0, 0.9, 0.9])
    mix_k = torch.tensor([K_MAX, 50, 1, 50, 7], dtype=torch.int64)
    for _ in range(20):
        out = sampler.sample_codes_batched(logits, mix_t, mix_k)
        assert torch.equal(out[[0, 2]], logits.argmax(dim=-1)[[0, 2]])
        for r, k in ((1, 50), (3, 50), (4, 7)):
            scaled = logits[r] / 0.9
            kth = scaled.topk(k).values[-1]
            assert scaled[out[r]] >= kth, f"row {r} escaped top-{k} support"
        assert ((out >= 0) & (out < K_MAX)).all()


def test_sampler_state_pool_reset_row_and_staging_width() -> None:
    """Pool-row state shapes/dtypes + ``reset_row`` + the packed D2H staging
    width contract."""
    pool = sampler.CsmBatchedSamplerState(4, device="cpu")
    assert pool.last_codes.shape == (4, NUM_CODEBOOKS)
    assert pool.last_codes.dtype == torch.int64
    assert pool.generation_done.dtype == torch.bool
    assert pool.frames_emitted.dtype == torch.int32
    pool.last_codes[1] = 5
    pool.generation_done[1] = True
    pool.frames_emitted[1] = 99
    pool.reset_row(1)
    assert (pool.last_codes[1] == 0).all()
    assert not bool(pool.generation_done[1])
    assert int(pool.frames_emitted[1]) == 0
    assert sampler.STAGING_WIDTH == NUM_CODEBOOKS + 2 == 34


class _FakeReq:
    def __init__(self) -> None:
        self.is_chunked = 0
        self.finished_reason = None
        self.output_ids: list[int] = []

    def finished(self) -> bool:
        return self.finished_reason is not None


def test_forced_frame0_eos_zero_decode_lifecycle() -> None:
    """A frame-0 EOS sampled AT PREFILL finishes the
    request at prefill — the EOS frame is recorded in ``output_frames`` (HF
    ``sequences`` parity), a finish reason is set (KV release trigger), and
    NOTHING is emitted to the vocoder. A sibling non-EOS request in the same
    collect IS streamed (proves the empty outbox is not vacuous)."""
    from sglang_omni.models.csm_tts.model_runner import CsmTTSModelRunner

    eos_frame = torch.zeros(NUM_CODEBOOKS, dtype=torch.long)
    eos_frame[31] = 7  # cb31 excluded from the EOS check
    normal_frame = torch.arange(1, NUM_CODEBOOKS + 1, dtype=torch.long)

    fake_model = SimpleNamespace(
        _rid_to_row={"r-eos": 0, "r-run": 1},
        _output_codes={"r-eos": [eos_frame], "r-run": [normal_frame]},
        _sampler_pool=SimpleNamespace(generation_done=torch.tensor([True, False])),
    )
    tp_worker = SimpleNamespace(
        gpu_id=0, model_runner=SimpleNamespace(model=fake_model)
    )
    runner = CsmTTSModelRunner(tp_worker, output_processor=None)
    outbox: queue.Queue = queue.Queue()
    runner.set_stream_outbox(outbox)

    metadata = {"modality": "audio_codes", "stream": True}
    reqs = []
    datas = []
    for rid in ("r-eos", "r-run"):
        req = _FakeReq()
        data = SimpleNamespace(
            req=req,
            output_frames=[],
            stream_metadata=metadata,
            generation_done=False,
        )
        reqs.append(SimpleNamespace(request_id=rid, data=data))
        datas.append(data)

    result = SimpleNamespace(
        logits_output=SimpleNamespace(next_token_logits=torch.zeros(2, 4)),
        next_token_ids=None,
    )
    runner._collect_step_outputs(result, reqs)

    eos_data, run_data = datas
    # EOS frame recorded but the request finishes at prefill...
    assert len(eos_data.output_frames) == 1
    assert torch.equal(eos_data.output_frames[0], eos_frame)
    assert eos_data.generation_done is True
    assert eos_data.req.finished_reason is not None
    assert eos_data.req.finished_reason.matched == CODEBOOK_EOS
    # ...while the running request keeps going and streams its frame.
    assert run_data.generation_done is False
    assert run_data.req.finished_reason is None

    emitted = []
    while not outbox.empty():
        emitted.append(outbox.get_nowait())
    assert [m.request_id for m in emitted] == ["r-run"]  # EOS never streamed
    assert emitted[0].target == "vocoder"
    assert torch.equal(emitted[0].data, normal_frame)

    # cb0 published per row (EOS row contributes its cb0 = 0)
    assert result.next_token_ids.tolist() == [0, 1]


def test_top_p_nucleus_keeps_top_token_and_excludes_tail() -> None:
    """On sampled rows (temperature>0, top_k=K_MAX) the top_p filter keeps the
    highest-prob token and never samples from the excluded low-prob tail."""
    torch.manual_seed(1234)
    V, B = 16, 4
    temp = torch.ones(B)
    topk = torch.full((B,), K_MAX, dtype=torch.long)
    # (a) one dominant token + tiny p -> nucleus collapses to the argmax.
    logits = torch.full((B, V), -10.0)
    logits[:, 3] = 10.0
    top_p = torch.full((B,), 0.5)
    for _ in range(64):
        out = sampler.sample_codes_batched(logits, temp, topk, top_p)
        assert torch.all(out == 3), out.tolist()
    # (b) flat top-3 (logit 10) over a -10 tail -> samples stay within {0,1,2}.
    logits = torch.full((B, V), -10.0)
    logits[:, :3] = 10.0
    top_p = torch.full((B,), 0.9)
    seen: set[int] = set()
    for _ in range(200):
        out = sampler.sample_codes_batched(logits, temp, topk, top_p)
        seen.update(out.tolist())
    assert seen and all(t < 3 for t in seen), f"tail token sampled: {sorted(seen)}"
