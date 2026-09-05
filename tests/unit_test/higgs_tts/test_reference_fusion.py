# SPDX-License-Identifier: Apache-2.0
"""Unit tests for reference-space voice fusion (``fusion_reference.py``).

Pure CPU. WORLD-morph math tests run only where pyworld is installed
(``pytest.importorskip``); the orchestrator state machine is tested against a
mock scheduler with the codec/morph heavy path patched out, and never imports
the sglang-dependent ``request_builders`` module (its lazy import inside the
retry path is satisfied with a stub module).
"""

from __future__ import annotations

import sys
import time
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.higgs_tts import fusion_reference as fr
from sglang_omni.models.higgs_tts.text_tokenizer import (
    AUDIO_PLACEHOLDER_ID,
    HiggsTokenizerAdapter,
)
from sglang_omni.models.higgs_tts.utils import apply_delay_pattern

# --- cache key ---------------------------------------------------------------


def _refs(seed=0, n=2, frames=90):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        raw = torch.randint(0, 1024, (frames, 8), generator=g)
        out.append({"codes_delayed": apply_delay_pattern(raw).tolist()})
    return out


def test_cache_key_is_stable_and_input_sensitive():
    refs = _refs(seed=1)
    k1 = fr.fused_reference_cache_key(refs, [0.5, 0.5], "cal")
    assert k1 == fr.fused_reference_cache_key(refs, [0.5, 0.5], "cal")
    assert k1 != fr.fused_reference_cache_key(refs, [0.6, 0.4], "cal")
    assert k1 != fr.fused_reference_cache_key(refs, [0.5, 0.5], "other cal")
    assert k1 != fr.fused_reference_cache_key(_refs(seed=2), [0.5, 0.5], "cal")
    assert k1 != fr.fused_reference_cache_key(list(reversed(refs)), [0.5, 0.5], "cal")


# --- DTW ---------------------------------------------------------------------


def test_dtw_map_identity_for_identical_sequences():
    feats = np.random.default_rng(0).normal(size=(40, 8))
    out = fr.dtw_map(feats, feats)
    assert out.shape == (40,)
    assert (out == np.arange(40)).all()


def test_dtw_map_monotonic_for_stretched_sequence():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(30, 8))
    stretched = np.repeat(base, 2, axis=0)  # B is A at half speed
    out = fr.dtw_map(base, stretched)
    assert (np.diff(out) >= 0).all()
    assert out[0] <= 1 and out[-1] >= stretched.shape[0] - 2


# --- prompt parts ------------------------------------------------------------


class _StubTok:
    def get_added_vocab(self):
        return {
            "<|tts|>": 1,
            "<|ref_audio|>": 2,
            "<|text|>": 3,
            "<|audio|>": 4,
            "<|ref_text|>": 5,
        }

    def encode(self, text, add_special_tokens=False):
        return [100 + (ord(c) % 50) for c in text]


@pytest.mark.parametrize("reference_text", [None, "校准句"])
@pytest.mark.parametrize("num_ref_tokens", [1, 7, 300])
def test_prompt_parts_reassemble_to_build_prompt(reference_text, num_ref_tokens):
    adapter = HiggsTokenizerAdapter(_StubTok())
    prefix, suffix = adapter.build_prompt_parts(
        "目标文本", reference_text=reference_text
    )
    assembled = prefix + [AUDIO_PLACEHOLDER_ID] * num_ref_tokens + suffix
    assert assembled == adapter.build_prompt(
        "目标文本", num_ref_tokens=num_ref_tokens, reference_text=reference_text
    )


# --- orchestrator state machine ---------------------------------------------


class _MockScheduler:
    def __init__(self):
        self._fusion_group_members = {}
        self._aborted_request_ids = set()
        self.enqueued = []
        self.errors = []
        self.aborted = []

    def _enqueue_built_request(self, payload, pending_stream_done, req_data):
        self.enqueued.append(req_data)

    def _emit_request_error(self, request_id, error):
        self.errors.append((request_id, error))

    def abort(self, request_id):
        self.aborted.append(request_id)


class _FakeCodec:
    def decode(self, raw_TN):
        return torch.zeros(raw_TN.shape[0] * 320, dtype=torch.float32)

    def encode_reference(self, wav, sample_rate=24000):
        return torch.randint(0, 1024, (80, 8))


def _cal_row_result(rid, frames=90, finish="stop"):
    g = torch.Generator().manual_seed(hash(rid) % (2**31))
    delayed = apply_delay_pattern(torch.randint(0, 1024, (frames, 8), generator=g))
    return SimpleNamespace(
        req=SimpleNamespace(rid=rid),
        finish_reason=finish,
        output_codes=[delayed[i] for i in range(delayed.shape[0])],
    )


def _bound_orchestrator(monkeypatch):
    orch = fr.FusionReferenceOrchestrator()
    sched = _MockScheduler()
    orch.bind(sched, "/nonexistent/ckpt")
    monkeypatch.setattr(orch, "_codec", lambda: _FakeCodec())
    # Run the worker path inline so tests are deterministic.
    monkeypatch.setattr(orch._executor, "submit", lambda fn, *a, **k: fn(*a, **k))
    return orch, sched


def _patch_world(monkeypatch, cal_f0=150.0, anchor_f0=150.0):
    """Stub the WORLD analysis stack: the gate measures calibration wavs via
    ``world_f0``, the anchor via ``median_f0``, and the morph consumes
    ``world_extract`` (whose harvest pass is short-circuited by ``f0_t``)."""
    monkeypatch.setattr(
        fr,
        "world_f0",
        lambda wav, fs=24000: (np.full(8, float(cal_f0)), np.zeros(8)),
    )
    monkeypatch.setattr(fr, "median_f0", lambda wav, fs=24000: anchor_f0)
    monkeypatch.setattr(
        fr, "world_extract", lambda w, fs=24000, f0_t=None: ("f0", "sp", "ap")
    )


def _register(
    orch,
    request_id="req1",
    n=2,
    refs=None,
    fp_of_slot=None,
    make_fallback_request=None,
):
    refs = refs if refs is not None else _refs(seed=7, n=n)
    fps = (
        fp_of_slot if fp_of_slot is not None else [fr.ref_fingerprint(r) for r in refs]
    )
    ref_of_fp = {fp: ref for fp, ref in zip(fps, refs)}
    built = {}

    def make_real(delayed_rows):
        built["rows"] = delayed_rows
        return SimpleNamespace(req=SimpleNamespace(rid=request_id), real=True)

    rows = [
        fr._CalRow(fp=fp, seed_idx=0, rid=f"{request_id}#cal{fp[:8]}r0")
        for fp in dict.fromkeys(fps)
    ]
    orch.register_group(
        request_id=request_id,
        payload=SimpleNamespace(request_id=request_id),
        cache_key="key-" + request_id,
        fp_of_slot=fps,
        ref_of_fp=ref_of_fp,
        weights=[1.0 / len(fps)] * len(fps),
        cal_text="cal",
        make_real_request=make_real,
        cal_rows=rows,
        make_fallback_request=make_fallback_request,
    )
    return rows, built


def test_happy_path_builds_and_enqueues_real_request(monkeypatch):
    orch, sched = _bound_orchestrator(monkeypatch)
    _patch_world(monkeypatch)
    monkeypatch.setattr(
        fr, "_fuse_world_entries", lambda entries, fs=24000: np.zeros(24000)
    )
    rows, built = _register(orch)

    orch.on_internal_done("req1", rows[0].fp, _cal_row_result(rows[0].rid))
    assert not sched.enqueued  # one calibration row is not enough
    orch.on_internal_done("req1", rows[1].fp, _cal_row_result(rows[1].rid))

    assert built["rows"], "hybrid delayed rows must reach make_real_request"
    assert len(sched.enqueued) == 1 and getattr(sched.enqueued[0], "real", False)
    assert not sched.errors
    # group + abort entry fully cleaned up
    assert "req1" not in orch._groups
    assert "req1" not in sched._fusion_group_members
    # gate-validated calibration codes are now cached per voice
    assert orch.cal_cache_get(rows[0].fp, "cal") is not None
    assert orch.cal_cache_get(rows[1].fp, "cal") is not None


def test_short_real_reference_passes_the_length_floor(monkeypatch):
    """A legitimate ~2 s reference is only ~50 frames at the codec's real
    25 Hz rate — the length floor must reject garbage, not short references.
    (Regression: an erroneous 75-frame floor rejected real 53-frame refs.)"""
    orch, sched = _bound_orchestrator(monkeypatch)
    _patch_world(monkeypatch)
    monkeypatch.setattr(
        fr, "_fuse_world_entries", lambda entries, fs=24000: np.zeros(24000)
    )
    rows, built = _register(
        orch, request_id="req2", refs=_refs(seed=11, n=2, frames=53)
    )
    orch.on_internal_done("req2", rows[0].fp, _cal_row_result(rows[0].rid, frames=53))
    orch.on_internal_done("req2", rows[1].fp, _cal_row_result(rows[1].rid, frames=53))
    assert built.get("rows"), "53-frame references must build successfully"
    assert not sched.errors


def test_duplicate_references_deduplicate_to_distinct_voices(monkeypatch):
    """12 slots of only 2 distinct voices: 2 calibration rows drive the whole
    build, and the morph sees all 12 weighted slots."""
    orch, sched = _bound_orchestrator(monkeypatch)
    _patch_world(monkeypatch)
    seen_entries = {}

    def fake_fuse(entries, fs=24000):
        seen_entries["n"] = len(entries)
        return np.zeros(24000)

    monkeypatch.setattr(fr, "_fuse_world_entries", fake_fuse)
    two = _refs(seed=13, n=2)
    refs12 = [two[i % 2] for i in range(12)]
    fps12 = [fr.ref_fingerprint(r) for r in refs12]
    assert len(set(fps12)) == 2
    rows, built = _register(orch, request_id="req3", refs=refs12, fp_of_slot=fps12)
    assert len(rows) == 2, "only distinct voices get calibration rows"
    orch.on_internal_done("req3", rows[0].fp, _cal_row_result(rows[0].rid))
    orch.on_internal_done("req3", rows[1].fp, _cal_row_result(rows[1].rid))
    assert built.get("rows")
    assert seen_entries["n"] == 12, "morph must weight all 12 slots"
    assert not sched.errors


def test_member_registration_excludes_client_facing_id(monkeypatch):
    """The admission gate withholds groups whose members aren't all queued;
    ``request_id`` never has a queue row, so it must not be a member."""
    orch, sched = _bound_orchestrator(monkeypatch)
    rows, _ = _register(orch)
    members = sched._fusion_group_members["req1"]
    assert members == {r.rid for r in rows}
    assert "req1" not in members


def test_aborted_calibration_row_fails_group_without_client_error(monkeypatch):
    orch, sched = _bound_orchestrator(monkeypatch)
    rows, built = _register(orch)
    orch.on_internal_done(
        "req1", rows[0].fp, _cal_row_result(rows[0].rid, finish="abort")
    )
    assert "rows" not in built
    assert not sched.enqueued
    assert not sched.errors  # abort came from the client; no extra error
    assert rows[1].rid in sched.aborted  # sibling cascade
    assert "req1" not in orch._groups


def test_empty_calibration_output_emits_error(monkeypatch):
    orch, sched = _bound_orchestrator(monkeypatch)
    rows, _ = _register(orch)
    result = _cal_row_result(rows[0].rid)
    result.output_codes = []
    orch.on_internal_done("req1", rows[0].fp, result)
    assert sched.errors and sched.errors[0][0] == "req1"
    assert not sched.enqueued


def test_f0_gate_failure_retries_with_next_seed(monkeypatch):
    orch, sched = _bound_orchestrator(monkeypatch)
    # ref0's calibration comes back an octave off its anchor once, then fine.
    # The gate measures calibration wavs via world_f0; the retry round must
    # re-measure ONLY the failed voice, so exactly 3 world_f0 calls happen
    # (cal0 fail, cal1 pass, cal0-retry pass) — a 4th raises StopIteration.
    cal_track = iter([300.0, 100.0, 100.0])
    monkeypatch.setattr(
        fr,
        "world_f0",
        lambda wav, fs=24000: (np.full(8, next(cal_track)), np.zeros(8)),
    )
    monkeypatch.setattr(fr, "median_f0", lambda wav, fs=24000: 100.0)
    monkeypatch.setattr(
        fr, "world_extract", lambda w, fs=24000, f0_t=None: ("f0", "sp", "ap")
    )
    monkeypatch.setattr(
        fr, "_fuse_world_entries", lambda entries, fs=24000: np.zeros(24000)
    )
    retry_builds = []
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.higgs_tts.request_builders",
        types.SimpleNamespace(
            build_calibration_request=lambda **kw: (
                retry_builds.append(kw),
                SimpleNamespace(req=SimpleNamespace(rid=kw["rid"]), retry=True),
            )[1]
        ),
    )
    rows, built = _register(orch)
    orch.on_internal_done("req1", rows[0].fp, _cal_row_result(rows[0].rid))
    orch.on_internal_done("req1", rows[1].fp, _cal_row_result(rows[1].rid))

    assert retry_builds and retry_builds[0]["seed"] == fr.CAL_SEEDS[1]
    assert len(sched.enqueued) == 1 and getattr(sched.enqueued[0], "retry", False)
    retry_rid = retry_builds[0]["rid"]
    assert retry_rid in sched._fusion_group_members["req1"]

    orch.on_internal_done("req1", rows[0].fp, _cal_row_result(retry_rid))
    assert built["rows"]
    assert any(getattr(r, "real", False) for r in sched.enqueued)


def test_f0_gate_exhaustion_degrades_to_fallback_when_available(monkeypatch):
    """Auto-split builds must not hard-fail a request plain trimming would
    have served: when every calibration seed fails the F0 gate, the group
    degrades to cloning from the highest-weight raw segment."""
    orch, sched = _bound_orchestrator(monkeypatch)
    # Every calibration read is an octave off its anchor, on every seed.
    monkeypatch.setattr(
        fr, "world_f0", lambda wav, fs=24000: (np.full(8, 300.0), np.zeros(8))
    )
    monkeypatch.setattr(fr, "median_f0", lambda wav, fs=24000: 100.0)
    monkeypatch.setattr(
        fr, "world_extract", lambda w, fs=24000, f0_t=None: ("f0", "sp", "ap")
    )
    retry_builds = []
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.higgs_tts.request_builders",
        types.SimpleNamespace(
            build_calibration_request=lambda **kw: (
                retry_builds.append(kw),
                SimpleNamespace(req=SimpleNamespace(rid=kw["rid"]), retry=True),
            )[1]
        ),
    )
    fallback_built = {}

    def make_fallback(rows):
        fallback_built["rows"] = rows
        return SimpleNamespace(req=SimpleNamespace(rid="req9"), fallback=True)

    refs = _refs(seed=17, n=2)
    rows, built = _register(
        orch, request_id="req9", refs=refs, make_fallback_request=make_fallback
    )

    orch.on_internal_done("req9", rows[0].fp, _cal_row_result(rows[0].rid))
    orch.on_internal_done("req9", rows[1].fp, _cal_row_result(rows[1].rid))
    for _ in range(2):  # seed 1 and seed 2 retry rounds, both voices
        rids = [kw["rid"] for kw in retry_builds[-2:]]
        for rid, row in zip(rids, rows):
            orch.on_internal_done("req9", row.fp, _cal_row_result(rid))

    assert "rows" not in built, "hybrid build must not run after exhaustion"
    assert fallback_built["rows"] == refs[0]["codes_delayed"]
    assert any(getattr(r, "fallback", False) for r in sched.enqueued)
    assert not sched.errors
    assert "req9" not in orch._groups
    assert "req9" not in sched._fusion_group_members


def test_calibration_reads_codes_from_the_preallocated_buffer(monkeypatch):
    """The model runner writes decoded codes into ``output_code_buffer``, not
    the ``output_codes`` list, for every request that carries the buffer
    fields -- which is every ordinary request. A calibration row read only
    through the list therefore looked empty, and every cold fusion build
    failed with "produced no codes"."""
    orch, sched = _bound_orchestrator(monkeypatch)
    _patch_world(monkeypatch)
    monkeypatch.setattr(
        fr, "_fuse_world_entries", lambda entries, fs=24000: np.zeros(24000)
    )
    rows, built = _register(orch, request_id="reqbuf")

    def buffered_result(rid, frames=90):
        """A finished row shaped the way the model runner leaves it."""
        result = _cal_row_result(rid, frames=frames)
        stacked = torch.stack(result.output_codes, dim=0)
        result.output_codes = []  # runner never appends here
        result.output_code_buffer = stacked
        result.output_code_count = stacked.shape[0]
        result.num_codebooks = stacked.shape[1]
        return result

    orch.on_internal_done("reqbuf", rows[0].fp, buffered_result(rows[0].rid))
    orch.on_internal_done("reqbuf", rows[1].fp, buffered_result(rows[1].rid))

    assert built.get("rows"), "buffer-backed calibration must drive the build"
    assert not sched.errors


def test_client_abort_before_finalize_drops_the_build(monkeypatch):
    orch, sched = _bound_orchestrator(monkeypatch)
    _patch_world(monkeypatch)
    monkeypatch.setattr(
        fr, "_fuse_world_entries", lambda entries, fs=24000: np.zeros(24000)
    )
    rows, built = _register(orch)
    sched._aborted_request_ids.add("req1")
    orch.on_internal_done("req1", rows[0].fp, _cal_row_result(rows[0].rid))
    orch.on_internal_done("req1", rows[1].fp, _cal_row_result(rows[1].rid))
    assert not sched.enqueued
    assert "req1" not in orch._groups


def test_expired_build_is_swept_with_an_error(monkeypatch):
    orch, sched = _bound_orchestrator(monkeypatch)
    rows, _ = _register(orch)
    with orch._lock:
        orch._groups["req1"].deadline = time.monotonic() - 1
    orch._sweep_expired()
    assert sched.errors and sched.errors[0][0] == "req1"
    assert "req1" not in orch._groups


def test_codec_falls_back_to_cpu_when_gpu_load_fails(monkeypatch):
    orch = fr.FusionReferenceOrchestrator()
    orch._checkpoint_dir = "/ckpt"
    calls = []

    def fake_loader(path, device, dtype):
        calls.append((device, dtype))
        if device == "cuda":
            raise RuntimeError("no VRAM headroom")
        return "cpu-codec"

    monkeypatch.setattr(fr, "get_or_load_codec", fake_loader)
    assert orch._codec() == "cpu-codec"
    assert calls == [("cuda", "bfloat16"), ("cpu", "float32")]
    # sticky: never retries the GPU after the first failure
    assert orch._codec() == "cpu-codec"
    assert calls[-1] == ("cpu", "float32") and len(calls) == 3


def test_cache_roundtrip_and_eviction():
    orch = fr.FusionReferenceOrchestrator()
    for i in range(fr._CACHE_MAX_ENTRIES + 5):
        orch._cache_put(f"k{i}", [[i]])
    assert orch.cache_get("k0") is None  # evicted
    newest = f"k{fr._CACHE_MAX_ENTRIES + 4}"
    assert orch.cache_get(newest) == [[fr._CACHE_MAX_ENTRIES + 4]]


# --- WORLD morph math (requires pyworld) ------------------------------------


def _voiced_tone(f0_hz, seconds=1.2, fs=24000):
    t = np.arange(int(seconds * fs)) / fs
    # A pulse-ish periodic signal with harmonics so WORLD tracks it robustly.
    x = np.zeros_like(t)
    for k in (1, 2, 3, 4):
        x += (0.5 / k) * np.sin(2 * np.pi * f0_hz * k * t)
    return (x * 0.5).astype(np.float64)


def test_morph_f0_lands_at_log_weighted_point():
    pytest.importorskip("pyworld")
    low, high = _voiced_tone(100.0), _voiced_tone(200.0)
    for alpha, expect in ((0.0, 100.0), (0.5, 141.4), (1.0, 200.0)):
        hybrid = fr.build_fused_reference([low, high], [1 - alpha, alpha])
        f0 = fr.median_f0(hybrid.astype(np.float64))
        assert f0 is not None
        assert abs(np.log(f0 / expect)) < np.log(1.08), (alpha, f0)


def test_morph_weight_monotonicity():
    pytest.importorskip("pyworld")
    low, high = _voiced_tone(100.0), _voiced_tone(200.0)
    f0s = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        hybrid = fr.build_fused_reference([low, high], [1 - alpha, alpha])
        f0s.append(fr.median_f0(hybrid.astype(np.float64)))
    assert all(a < b for a, b in zip(f0s, f0s[1:])), f0s
