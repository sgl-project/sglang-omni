# SPDX-License-Identifier: Apache-2.0
"""Reference-space voice fusion (zero training), built INSIDE the engine stage.

Instead of blending per-step output distributions across sibling rows (see
``fusion.py``, kept as the ``logits`` research mode), this module builds ONE
hybrid-timbre reference from N weighted reference voices and then serves the
request as a completely ordinary single-reference clone:

1. **Calibration**: each reference voice clones the same fixed calibration
   sentence (engine-internal requests, fixed seed + F0 quality gate), giving
   N same-content / different-timbre readings.
2. **WORLD morph**: DTW-align the readings on WORLD spectral-envelope
   features, then weight-interpolate log-F0 / log spectral envelope /
   aperiodicity and resynthesize a hybrid reference waveform.
3. **Serve**: codec-encode the hybrid waveform into an ordinary reference and
   enqueue the real request as a standard single-reference clone. The hybrid
   reference is cached per (reference codes, weights, algo version) so only
   the first request of a combination pays the build cost.

Why reference-space: per-step distribution pooling was live-verified to be
seed-bimodal (an intermediate-timbre frame is a low-probability tail of BOTH
experts, and AR hysteresis locks the register within a few frames — see
``docs/voice_fusion_design.md`` 第五阶段). Moving fusion into the reference
makes the intermediate timbre the MODE of the model's speaker posterior, and
single-reference cloning has no bimodality mechanism at all. Live E1 run:
8/8 mixed clones landed inside [0.35, 0.65] on the log-F0 axis (old
mechanism: ~15/16 locked to an endpoint) with strictly weight-monotonic
output across alpha ∈ {0, 0.25, 0.5, 0.75, 1}.

Everything here runs in the tts_engine stage process. Heavy work (CPU codec
decode/encode, pyworld analysis/synthesis) happens on a single worker thread;
scheduler-thread callbacks only collect codes and advance the state machine.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from sglang_omni.models.higgs_tts.utils import (
    apply_delay_pattern,
    collected_output_codes,
    get_or_load_codec,
)
from sglang_omni.utils.codec_delay import reverse_delay_pattern

logger = logging.getLogger(__name__)

# --- Configuration -----------------------------------------------------------

FUSION_MODE_ENV = "HIGGS_FUSION_MODE"  # "reference" (default) | "logits"
CAL_TEXT_ENV = "HIGGS_FUSION_CAL_TEXT"

# Calibration sentence: natural register, reasonably phoneme-rich, ~8 s read.
# Must stay stable across releases: it is part of the cache key and of the
# hybrid reference's transcript.
DEFAULT_CAL_TEXT = (
    "今天天气不错，我们在花园里散步，聊起了旅行、音乐和美食，每个人都很开心。"
)

ALGO_VERSION = "ref-fusion-v1"

# Calibration sampling: fixed, user-independent (part of determinism + cache
# identity). Matches the live-validated E1 protocol.
CAL_SEEDS = (1234, 5678, 424242)
CAL_TEMPERATURE = 0.8
CAL_TOP_P = 0.8
CAL_TOP_K = 30
CAL_MAX_NEW_TOKENS = 768  # codec is 25 Hz → ~30 s cap; calibration reads are ~8 s

# F0 quality gate: a calibration clone must sit within x1.35 of its own
# reference voice's F0 median, else retry with the next seed.
_GATE_LOG_RATIO = math.log(1.35)

# A build that hasn't finished within this window is failed + swept (covers
# rows silently removed from waiting_queue by abort, which never reach
# ``stream_output``).
BUILD_DEADLINE_S = 300.0

_CACHE_MAX_ENTRIES = 64

_WORLD_FRAME_PERIOD_MS = 5.0
_SAMPLE_RATE = 24_000

# Anchor-F0 measurement window: ~10 s of delayed rows at the codec's real
# 25 Hz rate (+7 delay-pattern rows). Median F0 is insensitive to duration
# past this -- measured on one voice, the median over 5 s and over 20 s differ
# by ln 0.02, against a gate of ln 1.35 -- while harvest's cost is linear in
# it, so a longer window buys no gate precision at real expense.
_ANCHOR_MAX_DELAYED_ROWS = 10 * 25 + 7

# WORLD analysis is CPU-bound C code that releases the GIL (measured: four
# concurrent harvest calls finish in 1.2x the time of one), and the calls a
# build makes are independent per voice. Running them on this pool instead of
# in sequence is what keeps a multi-voice build from costing K times a single
# one -- harvest dominates the whole build, at roughly 0.28 s per second of
# audio analysed.
_WORLD_POOL = ThreadPoolExecutor(
    max_workers=int(os.environ.get("HIGGS_WORLD_THREADS", "8")),
    thread_name_prefix="higgs-world",
)


def fusion_mode() -> str:
    mode = os.environ.get(FUSION_MODE_ENV, "reference").strip().lower()
    if mode not in ("reference", "logits"):
        raise ValueError(
            f"{FUSION_MODE_ENV} must be 'reference' or 'logits', got {mode!r}"
        )
    return mode


def calibration_text() -> str:
    return os.environ.get(CAL_TEXT_ENV) or DEFAULT_CAL_TEXT


# --- WORLD-domain morph (numpy only; pyworld imported lazily) ----------------


def _pyworld():
    try:
        import pyworld
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "reference-space voice fusion requires the 'pyworld' package "
            "(WORLD vocoder analysis/synthesis) in the tts_engine environment"
        ) from exc
    return pyworld


def world_f0(wav: np.ndarray, fs: int = _SAMPLE_RATE) -> tuple[np.ndarray, np.ndarray]:
    """harvest + stonemask F0 track with its time axis.

    harvest is by far the most expensive WORLD stage; callers that need both
    an F0 gate check and a full ``world_extract`` should run this once and
    pass the result on via ``world_extract(..., f0_t=...)``.
    """
    pw = _pyworld()
    x = np.ascontiguousarray(wav, dtype=np.float64)
    f0, t = pw.harvest(x, fs, frame_period=_WORLD_FRAME_PERIOD_MS)
    return pw.stonemask(x, f0, t, fs), t


def world_extract(
    wav: np.ndarray,
    fs: int = _SAMPLE_RATE,
    f0_t: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """mono float waveform → (f0, spectral envelope, aperiodicity).

    ``f0_t`` short-circuits the harvest/stonemask pass with a precomputed
    ``world_f0`` result for the same waveform.
    """
    pw = _pyworld()
    x = np.ascontiguousarray(wav, dtype=np.float64)
    if f0_t is None:
        f0, t = world_f0(x, fs)
    else:
        f0, t = f0_t
    sp = pw.cheaptrick(x, f0, t, fs)
    ap = pw.d4c(x, f0, t, fs)
    return f0, sp, ap


def median_f0(wav: np.ndarray, fs: int = _SAMPLE_RATE) -> float | None:
    """Median voiced F0 via WORLD harvest; None when fully unvoiced."""
    f0, _ = world_f0(wav, fs)
    voiced = f0[f0 > 0]
    if voiced.size == 0:
        return None
    return float(np.median(voiced))


def dtw_features(sp: np.ndarray, num_bands: int = 32) -> np.ndarray:
    """Per-frame alignment features: band-pooled, mean-centered log envelope."""
    lsp = np.log(sp + 1e-12)
    frames, bins = lsp.shape
    edges = np.linspace(0, bins, num_bands + 1).astype(int)
    feats = np.stack(
        [lsp[:, a:b].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1
    )
    return feats - feats.mean(axis=0, keepdims=True)


def dtw_map(
    feats_a: np.ndarray, feats_b: np.ndarray, band: int | None = None
) -> np.ndarray:
    """DTW-align B onto A's frame axis: returns ``map_ab[T_a] -> B index``.

    ``band`` is a slope-normalized Sakoe-Chiba radius: row ``i`` only fills
    cells within ``band`` of the diagonal point ``i * tb / ta``. Calibration
    readings share the exact same text, so their alignment hugs the diagonal
    and the band cuts the pure-python DP from O(ta*tb) to O(ta*band). If the
    banded pass fails to connect (pathological drift), it falls back to the
    full matrix.
    """
    cost = np.sqrt(
        np.maximum(
            (feats_a**2).sum(1)[:, None]
            + (feats_b**2).sum(1)[None, :]
            - 2.0 * feats_a @ feats_b.T,
            0.0,
        )
    )
    ta, tb = cost.shape
    if band is None:
        # Floor of 64 decimated frames (= ~5 s of drift allowance at the
        # 20 ms decimated rate) absorbs local pauses/speed differences
        # between same-text readings without falling back to the full matrix.
        band = max(64, int(0.15 * tb) + abs(tb - ta))
    acc = np.full((ta + 1, tb + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, ta + 1):
        center = int(round((i - 1) * tb / max(ta, 1))) + 1
        lo = max(1, center - band)
        hi = min(tb, center + band)
        cost_row = cost[i - 1]
        prev_row = acc[i - 1]
        cur_row = acc[i]
        for j in range(lo, hi + 1):
            cur_row[j] = cost_row[j - 1] + min(
                prev_row[j], cur_row[j - 1], prev_row[j - 1]
            )
    if not np.isfinite(acc[ta, tb]):  # band too tight for this pair: redo full
        return dtw_map(feats_a, feats_b, band=max(ta, tb))
    i, j = ta, tb
    path: list[tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        _, i, j = min(
            (acc[i - 1, j - 1], i - 1, j - 1),
            (acc[i - 1, j], i - 1, j),
            (acc[i, j - 1], i, j - 1),
        )
    path.reverse()
    buckets: dict[int, list[int]] = {}
    for a, b in path:
        buckets.setdefault(a, []).append(b)
    out = np.zeros(ta, dtype=np.int64)
    last = 0
    for a in range(ta):
        if a in buckets:
            last = int(np.median(buckets[a]))
        out[a] = last
    return out


# DTW cost is a pure-python O(Ta*Tb) DP; at the 5 ms WORLD frame rate an ~8 s
# read is ~1600 frames (2.6M cells per pairwise merge). Aligning on a 4x
# time-decimated feature track (20 ms) cuts that 16x; 20 ms alignment jitter
# is far below what the envelope-averaging morph can perceive.
_DTW_DECIMATE = 4


def _morph_pair(
    world_a: tuple[np.ndarray, np.ndarray, np.ndarray],
    world_b: tuple[np.ndarray, np.ndarray, np.ndarray],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blend B into A's time axis with share ``alpha`` for B (E1-validated).

    log-F0 weighted mean where both are voiced; a lone-voiced frame keeps its
    contour shifted by the global (median) pitch ratio so the register still
    lands at the blended point; log-envelope weighted mean; linear
    aperiodicity.
    """
    f0_a, sp_a, ap_a = world_a
    f0_b, sp_b, ap_b = world_b
    feats_a, feats_b = dtw_features(sp_a), dtw_features(sp_b)
    d = _DTW_DECIMATE
    map_ds = dtw_map(feats_a[::d], feats_b[::d])
    map_ab = np.minimum(np.repeat(map_ds * d, d)[: len(feats_a)], len(feats_b) - 1)
    if len(map_ab) < len(feats_a):  # decimated track shorter than full track
        map_ab = np.pad(map_ab, (0, len(feats_a) - len(map_ab)), mode="edge")
    f0_bw, sp_bw, ap_bw = f0_b[map_ab], sp_b[map_ab], ap_b[map_ab]

    voiced_a, voiced_b = f0_a > 0, f0_bw > 0
    gm_a = float(np.median(f0_a[f0_a > 0]))
    gm_b = float(np.median(f0_b[f0_b > 0]))

    f0_m = np.zeros_like(f0_a)
    both = voiced_a & voiced_b
    f0_m[both] = np.exp((1 - alpha) * np.log(f0_a[both]) + alpha * np.log(f0_bw[both]))
    only_a = voiced_a & ~voiced_b
    f0_m[only_a] = f0_a[only_a] * (gm_b / gm_a) ** alpha
    only_b = (~voiced_a) & voiced_b
    f0_m[only_b] = f0_bw[only_b] * (gm_a / gm_b) ** (1 - alpha)

    sp_m = np.exp((1 - alpha) * np.log(sp_a + 1e-16) + alpha * np.log(sp_bw + 1e-16))
    ap_m = np.clip((1 - alpha) * ap_a + alpha * ap_bw, 0.001, 0.999)
    return f0_m, sp_m, ap_m


def _fuse_world_entries(
    entries: list[dict],
    fs: int = _SAMPLE_RATE,
) -> np.ndarray:
    """``[{"world": (f0, sp, ap), "weight": float}, ...]`` → hybrid waveform.

    Reduces pairwise (always merging the two smallest current weights,
    Huffman-style) so every step goes through the validated binary blend; the
    time axis converges to the largest-weight member's. Entries may share
    ``world`` objects (duplicate references) — they are never mutated.
    """
    pw = _pyworld()
    entries = [dict(e) for e in entries]
    while len(entries) > 1:
        entries.sort(key=lambda e: e["weight"])
        low, high = entries.pop(0), entries.pop(0)
        # Base (time axis) = the heavier of the pair; alpha = lighter's share.
        alpha = low["weight"] / (low["weight"] + high["weight"])
        merged = _morph_pair(high["world"], low["world"], alpha)
        entries.append({"world": merged, "weight": low["weight"] + high["weight"]})

    f0_m, sp_m, ap_m = entries[0]["world"]
    wav = pw.synthesize(
        np.ascontiguousarray(f0_m),
        np.ascontiguousarray(sp_m),
        np.ascontiguousarray(ap_m),
        fs,
        _WORLD_FRAME_PERIOD_MS,
    )
    peak = max(float(np.abs(wav).max()), 1e-8)
    return (wav / peak * 0.9).astype(np.float32)


def build_fused_reference(
    cal_wavs: list[np.ndarray],
    weights: list[float],
    fs: int = _SAMPLE_RATE,
) -> np.ndarray:
    """N same-content calibration waveforms + weights → hybrid waveform.

    N == 2 is exactly the live-validated E1 morph.
    """
    if len(cal_wavs) != len(weights) or len(cal_wavs) < 2:
        raise ValueError(
            f"need >= 2 calibration waveforms with matching weights, got "
            f"{len(cal_wavs)} / {len(weights)}"
        )
    entries = [
        {"world": world_extract(w, fs), "weight": float(wt)}
        for w, wt in zip(cal_wavs, weights)
    ]
    return _fuse_world_entries(entries, fs)


# --- Cache -------------------------------------------------------------------


def ref_fingerprint(ref: dict[str, Any]) -> str:
    """Content fingerprint of ONE reference's delayed code sequence.

    Keys the per-reference calibration/anchor caches AND deduplicates repeated
    references inside a single fusion request (12 slots of 2 distinct voices
    cost 2 calibration syntheses, not 12).
    """
    # Row-major little-endian uint16 — byte-identical to the historical
    # per-value ``int(c).to_bytes(2, "little")`` loop, so keys survive.
    buf = np.asarray(ref["codes_delayed"], dtype="<u2").tobytes()
    return hashlib.blake2b(buf, digest_size=16).hexdigest()


def fused_reference_cache_key(
    refs: list[dict[str, Any]], weights: list[float], cal_text: str
) -> str:
    """Content key over the N reference code sequences + weights + algo id."""
    h = hashlib.blake2b(digest_size=16)
    h.update(ALGO_VERSION.encode())
    h.update(cal_text.encode())
    for ref, weight in zip(refs, weights):
        h.update(b"|ref|")
        h.update(f"{weight:.6f}".encode())
        h.update(ref_fingerprint(ref).encode())
    return h.hexdigest()


# --- Engine-side orchestrator ------------------------------------------------


@dataclass
class _CalRow:
    fp: str  # reference fingerprint this calibration row serves
    seed_idx: int
    rid: str


@dataclass
class _BuildGroup:
    request_id: str
    payload: Any
    cache_key: str
    fp_of_slot: list[str]  # per-slot reference fingerprint (duplicates allowed)
    ref_of_fp: dict[str, dict[str, Any]]  # fp -> one ref dict for that voice
    weights: list[float]
    cal_text: str
    make_real_request: Callable[[list[list[int]]], Any]
    deadline: float
    pending: dict[str, _CalRow] = field(default_factory=dict)
    collected: dict[str, torch.Tensor] = field(default_factory=dict)  # fp -> delayed
    seed_used: dict[str, int] = field(default_factory=dict)
    # fp -> (cal_wav, (f0, t)) for voices that already passed the F0 gate, so
    # a seed retry re-decodes/re-measures only the voices that failed.
    gated: dict[str, tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]] = field(
        default_factory=dict
    )
    # Auto-split long-reference builds: degrade to plain single-segment
    # cloning instead of failing the request when the F0 gate exhausts all
    # calibration seeds. None for ordinary client-provided fusions.
    make_fallback_request: Callable[[list[list[int]]], Any] | None = None
    failed: bool = False

    @property
    def unique_fps(self) -> list[str]:
        return list(dict.fromkeys(self.fp_of_slot))


class FusionReferenceOrchestrator:
    """Per-engine-process build coordinator for reference-space fusion.

    Lives on the Higgs model object (see ``get_orchestrator``); bound to the
    OmniScheduler in ``HiggsTtsEngineBuilder.post_scheduler_setup``. Scheduler
    thread: ``on_internal_done`` (collect + advance). Worker thread: codec
    decode/encode + WORLD morph + real-request enqueue.
    """

    def __init__(self) -> None:
        self._scheduler: Any | None = None
        self._checkpoint_dir: str | None = None
        self._lock = threading.Lock()
        self._groups: dict[str, _BuildGroup] = {}
        self._cache: dict[str, list[list[int]]] = {}
        self._cache_order: list[str] = []
        # Per-reference caches, keyed by ref fingerprint: gate-validated
        # calibration codes (per cal_text) and the reference's own median F0.
        # These survive across weight combinations and across requests, so a
        # voice's calibration synthesis is only ever paid once per process.
        self._cal_cache: dict[tuple[str, str], torch.Tensor] = {}
        self._anchor_cache: dict[str, float | None] = {}
        self._anchor_futures: dict[str, Any] = {}
        self._codec_device: str | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="higgs-ref-fusion"
        )

    # -- wiring --

    def bind(self, scheduler: Any, checkpoint_dir: str) -> None:
        self._scheduler = scheduler
        self._checkpoint_dir = checkpoint_dir

    @property
    def is_bound(self) -> bool:
        return self._scheduler is not None

    # -- cache --

    def cache_get(self, key: str) -> list[list[int]] | None:
        with self._lock:
            return self._cache.get(key)

    def _cache_put(self, key: str, delayed_rows: list[list[int]]) -> None:
        with self._lock:
            if key not in self._cache:
                self._cache_order.append(key)
                while len(self._cache_order) > _CACHE_MAX_ENTRIES:
                    evicted = self._cache_order.pop(0)
                    self._cache.pop(evicted, None)
            self._cache[key] = delayed_rows

    def cal_cache_get(self, fp: str, cal_text: str) -> torch.Tensor | None:
        with self._lock:
            return self._cal_cache.get((fp, cal_text))

    def _cal_cache_put(self, fp: str, cal_text: str, delayed: torch.Tensor) -> None:
        with self._lock:
            self._cal_cache[(fp, cal_text)] = delayed

    # -- build lifecycle --

    def register_group(
        self,
        *,
        request_id: str,
        payload: Any,
        cache_key: str,
        fp_of_slot: list[str],
        ref_of_fp: dict[str, dict[str, Any]],
        weights: list[float],
        cal_text: str,
        make_real_request: Callable[[list[list[int]]], Any],
        cal_rows: list[_CalRow],
        pre_collected: dict[str, torch.Tensor] | None = None,
        make_fallback_request: Callable[[list[list[int]]], Any] | None = None,
    ) -> None:
        group = _BuildGroup(
            request_id=request_id,
            payload=payload,
            cache_key=cache_key,
            fp_of_slot=fp_of_slot,
            ref_of_fp=ref_of_fp,
            weights=weights,
            cal_text=cal_text,
            make_real_request=make_real_request,
            make_fallback_request=make_fallback_request,
            deadline=time.monotonic() + BUILD_DEADLINE_S,
        )
        self._sweep_expired()
        for fp, delayed in (pre_collected or {}).items():
            group.collected[fp] = delayed
            group.seed_used[fp] = 0
        for row in cal_rows:
            group.pending[row.rid] = row
            group.seed_used[row.fp] = row.seed_idx
        with self._lock:
            self._groups[request_id] = group
        # Abort entry point: a client abort of ``request_id`` must cascade
        # into the in-flight calibration rows. The set deliberately does NOT
        # contain ``request_id`` itself — the atomic-admission gate withholds
        # any group whose members aren't all present in the waiting queue, and
        # ``request_id`` never has a queue row of its own during the build.
        # (Calibration rows skip that gate entirely via
        # ``fusion_skip_atomic_admission``; this entry is abort-cascade only.)
        self._scheduler._fusion_group_members[request_id] = {
            row.rid for row in cal_rows
        }

        # Start the anchor analyses now: they depend only on the reference
        # codes, and the calibration reads they will be compared against are
        # about to spend seconds generating on the GPU.
        for fp, ref in ref_of_fp.items():
            self.prefetch_anchor(fp, ref)

    def make_done_callback(self, request_id: str, fp: str) -> Callable[[Any], None]:
        def _done(req_data: Any) -> None:
            self.on_internal_done(request_id, fp, req_data)

        return _done

    def on_internal_done(self, request_id: str, fp: str, req_data: Any) -> None:
        """Scheduler-thread callback for a finished calibration row."""
        self._sweep_expired()
        with self._lock:
            group = self._groups.get(request_id)
        if group is None or group.failed:
            return
        group.pending.pop(req_data.req.rid, None)

        if (req_data.finish_reason or "").lower() == "abort":
            self._fail(group, RuntimeError("calibration row aborted"), emit=False)
            return
        delayed = collected_output_codes(req_data).cpu()
        if delayed.shape[0] == 0:
            self._fail(
                group,
                RuntimeError(
                    f"calibration synthesis for voice {fp[:8]} produced no codes"
                ),
            )
            return
        group.collected[fp] = delayed
        if group.pending or any(f not in group.collected for f in group.unique_fps):
            return
        self._executor.submit(self._finalize_build, group)

    # -- heavy path (worker thread) --

    def _codec(self):
        assert self._checkpoint_dir is not None
        # GPU codec on the engine's device, in the exact (device, dtype)
        # configuration the production vocoder/audio_encoder stages already
        # run — ~1 GB extra VRAM in the engine process, loaded once on the
        # first cold build. Kernel-level interleaving with AR decode is safe
        # (shared default stream serializes) and only costs the cold path a
        # few ms of contention. Falls back to the documented-stable fp32 CPU
        # path when the GPU load fails (e.g. no VRAM headroom).
        if self._codec_device is None:
            try:
                codec = get_or_load_codec(self._checkpoint_dir, "cuda", "bfloat16")
                self._codec_device = "cuda"
                return codec
            except Exception:
                logger.exception(
                    "reference-fusion GPU codec load failed; falling back to CPU fp32"
                )
                self._codec_device = "cpu"
        if self._codec_device == "cuda":
            return get_or_load_codec(self._checkpoint_dir, "cuda", "bfloat16")
        return get_or_load_codec(self._checkpoint_dir, "cpu", "float32")

    def _delayed_to_wav(self, delayed_LN: torch.Tensor) -> np.ndarray:
        raw = reverse_delay_pattern(delayed_LN, allow_short=True)
        raw = raw[(raw < 1024).all(dim=1)]
        # Sanity floor only — reject empty/garbage sequences, not short-but-real
        # references. The codec runs at 25 Hz (hop 960 @ 24 kHz; the "75 Hz"
        # note elsewhere in this package is a conservative cap, not the real
        # rate), so a legitimate ~2 s reference is only ~50 frames.
        if raw.shape[0] < 20:
            raise RuntimeError(
                f"decoded reference is too short ({raw.shape[0]} frames)"
            )
        # .float(): the GPU codec decodes in bf16, which numpy can't represent.
        return self._codec().decode(raw).float().numpy().astype(np.float64)

    def _finalize_build(self, group: _BuildGroup) -> None:
        try:
            self._finalize_build_inner(group)
        except Exception as exc:  # noqa: BLE001 - single failure funnel
            logger.exception("reference-fusion build failed for %s", group.request_id)
            self._fail(group, exc)

    def prefetch_anchor(self, fp: str, ref: dict[str, Any]) -> None:
        """Start this voice's anchor F0 now, off the critical path.

        The anchor depends only on the reference codes, which are known as
        soon as the group is registered -- while the calibration synthesis it
        will be compared against is still generating on the GPU. Computing it
        there instead of in ``_finalize_build_inner`` hides a harvest pass per
        voice behind work that was going to happen anyway.

        Only the harvest goes to the pool. The codec decode ahead of it stays
        on the caller's thread: it is milliseconds of GPU work, and issuing it
        from N pool threads instead puts N concurrent allocations on the
        device at the exact moment the engine is running the calibration
        generations this is meant to overlap with -- which is enough, at eight
        voices, to take the engine out of memory.
        """
        with self._lock:
            if fp in self._anchor_cache or fp in self._anchor_futures:
                return
        anchor_wav = self._anchor_wav(ref)
        with self._lock:
            if fp in self._anchor_cache or fp in self._anchor_futures:
                return
            self._anchor_futures[fp] = _WORLD_POOL.submit(
                self._anchor_from_wav, fp, anchor_wav
            )

    def _anchor_wav(self, ref: dict[str, Any]) -> np.ndarray:
        return self._delayed_to_wav(
            torch.tensor(
                ref["codes_delayed"][:_ANCHOR_MAX_DELAYED_ROWS], dtype=torch.long
            )
        )

    def _anchor_from_wav(self, fp: str, anchor_wav: np.ndarray) -> float | None:
        value = median_f0(anchor_wav)
        with self._lock:
            self._anchor_cache[fp] = value
            self._anchor_futures.pop(fp, None)
        return value

    def _anchor_f0(self, fp: str, ref: dict[str, Any]) -> float | None:
        with self._lock:
            if fp in self._anchor_cache:
                return self._anchor_cache[fp]
        # The anchor is a coarse median-F0 register check against a ln(1.35)
        # gate; ~10 s of audio pins the median closely enough for that (the
        # median over 5 s and over 20 s of one voice differ by ln 0.02, under
        # a tenth of the gate) and this decode+harvest is the only cold-build
        # cost that scales with reference length — so truncate.
        return self._anchor_from_wav(fp, self._anchor_wav(ref))

    def _await_anchor(self, fp: str, ref: dict[str, Any]) -> float | None:
        """The prefetched anchor, computing it here if no prefetch ran."""
        with self._lock:
            if fp in self._anchor_cache:
                return self._anchor_cache[fp]
            future = self._anchor_futures.get(fp)
        if future is not None:
            return future.result()
        return self._anchor_f0(fp, ref)

    def _finalize_build_inner(self, group: _BuildGroup) -> None:
        scheduler = self._scheduler
        if group.request_id in scheduler._aborted_request_ids:
            self._drop(group)
            return

        # Gate + decode once per DISTINCT voice; duplicate slots share it, and
        # ``group.gated`` carries voices that already passed on a previous
        # round so a seed retry re-measures only the ones that failed. The F0
        # track computed for the gate is reused by ``world_extract`` below —
        # harvest (the dominant WORLD cost) runs once per calibration wav.
        retry_rows: list[tuple[str, int]] = []  # (fp, next_seed_idx)

        # Decode on this thread (the codec is GPU-side and the decodes are
        # milliseconds), then run every voice's harvest concurrently. The gate
        # verdicts below stay in ``unique_fps`` order regardless, so which
        # voice finishes analysing first cannot change which seed is retried
        # or which failure is reported.
        ungated = [fp for fp in group.unique_fps if fp not in group.gated]
        cal_wavs = {fp: self._delayed_to_wav(group.collected[fp]) for fp in ungated}
        cal_futures = {
            fp: _WORLD_POOL.submit(world_f0, wav) for fp, wav in cal_wavs.items()
        }

        for fp in group.unique_fps:
            if fp in group.gated:
                continue
            cal_wav = cal_wavs[fp]
            f0, t = cal_futures[fp].result()
            voiced = f0[f0 > 0]
            cal_f0 = float(np.median(voiced)) if voiced.size else None
            anchor_f0 = self._await_anchor(fp, group.ref_of_fp[fp])
            if cal_f0 is None or anchor_f0 is None:
                deviation = None
            else:
                deviation = abs(math.log(cal_f0 / anchor_f0))
            if deviation is None or deviation > _GATE_LOG_RATIO:
                next_seed = group.seed_used[fp] + 1
                if next_seed < len(CAL_SEEDS):
                    retry_rows.append((fp, next_seed))
                    logger.warning(
                        "reference-fusion %s: calibration for voice %s failed "
                        "the F0 gate (cal=%s anchor=%s), retrying with seed #%d",
                        group.request_id,
                        fp[:8],
                        cal_f0,
                        anchor_f0,
                        next_seed,
                    )
                    continue
                if group.make_fallback_request is not None:
                    logger.warning(
                        "reference-fusion %s: voice %s failed the F0 gate on "
                        "all %d seeds (last: cal_f0=%s anchor_f0=%s); "
                        "degrading to single-segment cloning",
                        group.request_id,
                        fp[:8],
                        len(CAL_SEEDS),
                        cal_f0,
                        anchor_f0,
                    )
                    self._serve_fallback(group)
                    return
                raise RuntimeError(
                    f"calibration for voice {fp[:8]} failed the F0 quality "
                    f"gate on all {len(CAL_SEEDS)} seeds "
                    f"(last: cal_f0={cal_f0}, anchor_f0={anchor_f0})"
                )
            group.gated[fp] = (cal_wav, (f0, t))

        if retry_rows:
            self._enqueue_retries(group, retry_rows)
            return

        for fp in group.unique_fps:
            self._cal_cache_put(fp, group.cal_text, group.collected[fp])

        # cheaptrick + d4c, one pass per voice. Cheaper than harvest but not
        # free (about a tenth of it), and independent per voice like the rest.
        extract_futures = {
            fp: _WORLD_POOL.submit(world_extract, wav, f0_t=f0_t)
            for fp, (wav, f0_t) in group.gated.items()
        }
        world_by_fp = {fp: fut.result() for fp, fut in extract_futures.items()}
        entries = [
            {"world": world_by_fp[fp], "weight": w}
            for fp, w in zip(group.fp_of_slot, group.weights)
        ]
        hybrid = _fuse_world_entries(entries)
        codes_TN = self._codec().encode_reference(
            torch.from_numpy(hybrid), sample_rate=_SAMPLE_RATE
        )
        delayed_rows = apply_delay_pattern(codes_TN).tolist()
        self._cache_put(group.cache_key, delayed_rows)

        if group.request_id in scheduler._aborted_request_ids:
            self._drop(group)
            return
        real_req_data = group.make_real_request(delayed_rows)
        self._drop(group)
        scheduler._enqueue_built_request(group.payload, False, real_req_data)
        logger.info(
            "reference-fusion %s: hybrid reference built (%d frames, %d distinct "
            "voices over %d slots), real request enqueued",
            group.request_id,
            len(delayed_rows),
            len(group.unique_fps),
            len(group.fp_of_slot),
        )

    def _serve_fallback(self, group: _BuildGroup) -> None:
        """Serve an auto-split build as a plain single-reference clone of the
        highest-weight raw segment (no calibration, no morph)."""
        scheduler = self._scheduler
        if group.request_id in scheduler._aborted_request_ids:
            self._drop(group)
            return
        best_slot = max(range(len(group.weights)), key=lambda i: group.weights[i])
        rows = group.ref_of_fp[group.fp_of_slot[best_slot]]["codes_delayed"]
        real_req_data = group.make_fallback_request(rows)
        self._drop(group)
        scheduler._enqueue_built_request(group.payload, False, real_req_data)

    def _enqueue_retries(
        self, group: _BuildGroup, retry_rows: list[tuple[str, int]]
    ) -> None:
        # Import here: request_builders imports this module at load time.
        from sglang_omni.models.higgs_tts.request_builders import (
            build_calibration_request,
        )

        # Register all retries as pending before enqueueing any, so a
        # lightning-fast completion can't observe a half-updated group. Retry
        # rows never enter the atomic-admission registry; they are only added
        # to the client-facing abort entry.
        pending_requests = []
        for fp, seed_idx in retry_rows:
            group.collected.pop(fp, None)
            group.seed_used[fp] = seed_idx
            rid = f"{group.request_id}#cal{fp[:8]}r{seed_idx}"
            row = _CalRow(fp=fp, seed_idx=seed_idx, rid=rid)
            group.pending[rid] = row
            pending_requests.append(
                build_calibration_request(
                    ref=group.ref_of_fp[fp],
                    rid=rid,
                    seed=CAL_SEEDS[seed_idx],
                    done_callback=self.make_done_callback(group.request_id, fp),
                )
            )
            members = self._scheduler._fusion_group_members.get(group.request_id)
            if members is not None:
                members.add(rid)
        for req_data in pending_requests:
            self._scheduler._enqueue_built_request(group.payload, False, req_data)

    # -- failure / cleanup --

    def _drop(self, group: _BuildGroup) -> None:
        with self._lock:
            self._groups.pop(group.request_id, None)
        members = self._scheduler._fusion_group_members.pop(group.request_id, None)
        if members is not None:
            for rid in members:
                self._scheduler._fusion_group_members.pop(rid, None)

    def _fail(self, group: _BuildGroup, exc: Exception, *, emit: bool = True) -> None:
        with self._lock:
            if group.failed:
                return
            group.failed = True
        if emit:
            self._scheduler._emit_request_error(group.request_id, exc)
        # Cascade-kill any still-pending calibration rows for this group.
        for rid in list(group.pending):
            try:
                self._scheduler.abort(rid)
            except Exception:  # noqa: BLE001 - best-effort cleanup
                logger.exception("failed to abort calibration row %s", rid)
        self._drop(group)

    def _sweep_expired(self) -> None:
        now = time.monotonic()
        with self._lock:
            expired = [g for g in self._groups.values() if now > g.deadline]
        for group in expired:
            self._fail(
                group,
                RuntimeError(
                    f"reference-fusion build timed out after {BUILD_DEADLINE_S:.0f}s"
                ),
            )


_ORCHESTRATOR_ATTR = "_higgs_fusion_reference_orchestrator"


def get_orchestrator(model: Any) -> FusionReferenceOrchestrator:
    """Engine-process singleton, keyed on the Higgs model object."""
    orch = getattr(model, _ORCHESTRATOR_ATTR, None)
    if orch is None:
        orch = FusionReferenceOrchestrator()
        setattr(model, _ORCHESTRATOR_ATTR, orch)
    return orch


__all__ = [
    "ALGO_VERSION",
    "CAL_MAX_NEW_TOKENS",
    "CAL_SEEDS",
    "CAL_TEMPERATURE",
    "CAL_TOP_K",
    "CAL_TOP_P",
    "DEFAULT_CAL_TEXT",
    "FusionReferenceOrchestrator",
    "build_fused_reference",
    "calibration_text",
    "dtw_features",
    "dtw_map",
    "fused_reference_cache_key",
    "fusion_mode",
    "get_orchestrator",
    "median_f0",
    "world_extract",
    "world_f0",
]
