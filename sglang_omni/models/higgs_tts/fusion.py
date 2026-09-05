# SPDX-License-Identifier: Apache-2.0
"""Voice timbre fusion — output-distribution blending for Higgs TTS.

A *fusion request* conditions one synthesis on ``N`` reference voices at once.
Each reference is prefilled into its own KV context as a separate *sibling* row,
and at every AR decode step the sibling rows' per-codebook output distributions
are blended by weight **before** sampling. All siblings then sample the *same*
multi-codebook frame (shared seed), so their ``N`` KV contexts evolve in
lock-step and decode the same audio; only the group *leader* row is emitted.

This module holds the pure, ``sgl_kernel``-free pieces of the mechanism — no
torch/sglang engine dependency, so they are unit-testable standalone:

- :func:`fuse_group_logits` — weighted **log-linear pool** (product-of-experts
  / weighted geometric mean, not a probability-space arithmetic mean — see
  that function's docstring for why the arithmetic-mean version this module
  shipped with first caused a real, measured bimodal "randomly sounds like
  just one reference voice" failure at near-equal blend weights) across group
  members, returned as logits ready to feed the standard sampler, plus a
  per-row ``is_grouped`` mask the caller MUST use to keep singleton rows
  sampling at their real (unfolded) temperature — see the "greedy" warning
  below.
- :func:`fuse_group_generation_done` — "any sibling done ⇒ all done" barrier so
  group members terminate on the same step.
- :class:`FusionRegistry` — thread-safe bookkeeping of which requests belong to
  which fusion group, at what weight, who the leader is, and which groups were
  caught split at prefill and must be aborted even if they later look complete.

Both are CUDA-Graph friendly: fixed-shape ``scatter_add_`` / advanced-index ops,
no host-side control flow. They are identity no-ops for the default case where
every row is its own singleton group (``group_id[i] == i``, ``weight == 1``).

Caller contract — do not fold temperature into the sampler call unconditionally:
``fuse_group_logits`` pre-applies ``temperature_B`` for grouped rows so the
blend happens in the same temperature-scaled space the sampler will use, and
the caller then samples the returned logits at ``temperature=1`` for those
rows. But the sampler's greedy short-circuit (:func:`sampler._sample_independent`
and its batched counterpart) is keyed on the *temperature it receives*, not on
the logits — it decides ``temperature <= _GREEDY_TEMP_THRESHOLD`` before ever
looking at the logits. If the caller passes ``temperature=1`` for EVERY row
(grouped or not, as a blanket simplification), a plain non-fusion request with
``temperature=0`` silently loses its argmax short-circuit: it becomes a
``multinomial`` draw over a near-one-hot distribution instead of a
deterministic ``argmax``, which both breaks determinism and burns global RNG
state that a truly-greedy row was never supposed to touch. The returned
``is_grouped`` mask is exactly what the caller needs to avoid this:
singleton rows must keep sampling at their real ``temperature_B``, and only
grouped rows get temperature folded away.
"""

from __future__ import annotations

import math
import threading

import torch

# Floor for the per-group weight-sum denominator so an all-zero-weight group
# (a caller bug, never a real weight config) divides by a tiny positive number
# instead of zero, rather than propagating inf/nan into the pooled logits.
_LOG_FLOOR = 1e-30

# Cap for FusionRegistry.update_delta's trajectory-feedback tilt: at most a
# 2x effective-weight swing in either direction, so a runaway
# integral-controller update can't overwhelm the user's own nominal weight
# the way an unclamped correction could.
_DELTA_MAX = math.log(2.0)

# Entropy-matching Newton solve (see _solve_entropy_matching_gamma): a fixed
# iteration count keeps the op shape-static/CUDA-Graph-safe (no host-side
# convergence check). Measured convergence: in the moderate-entropy regime
# (H ~2.5-6.5 nats) 3 iterations is already exact to float precision, but in
# the sharp/confident regime a real decode step actually lives in (H ~0.2-1.8
# nats — a member's distribution well after it has locked onto a specific
# codec token) 3 iterations leaves a real residual (measured 0.1-0.35 nats);
# 5 closes that to 1e-4-1e-5 nats. Gamma is clamped to
# [1/_ENTROPY_MATCH_GAMMA_CLAMP, _ENTROPY_MATCH_GAMMA_CLAMP] so a
# near-degenerate (near-zero-variance) distribution can't blow up the scale
# to something numerically wild; the clamp value is a "confidence fine-tunes,
# blend weight decides" cap, not a tight bound — a member is still allowed to
# end up noticeably sharper or flatter than its groupmates' average, just not
# unboundedly so.
_ENTROPY_MATCH_ITERS = 5
_ENTROPY_MATCH_GAMMA_CLAMP = 3.0


def _solve_entropy_matching_gamma(
    logits_BNV: torch.Tensor,
    group_id_B: torch.Tensor,
    norm_weight_B: torch.Tensor,
) -> torch.Tensor:
    """Per-(row, codebook) scale ``gamma`` so that
    ``H(softmax(gamma · logits))`` matches the row's group's
    weighted-average entropy — i.e. every member of a fusion group is
    rescaled to a common confidence level *before* being weighted-summed in
    :func:`fuse_group_logits`, so the nominal blend weight (not whichever
    member happened to be more "confident"/peaked) is what actually decides
    the pooled outcome. See that function's docstring for why this
    correction is needed: log-linear pooling is precision-weighted, not
    weight-only, by construction.

    Solved via a fixed ``_ENTROPY_MATCH_ITERS``-step Newton iteration (no
    host-side convergence check, so this stays a shape-static, CUDA-Graph-safe
    tensor op) using the closed-form derivative of softmax entropy w.r.t. a
    uniform logit-scale factor: for ``p(gamma) = softmax(gamma · z)`` and
    ``H(gamma) = -Σ_v p_v(gamma)·log p_v(gamma)``, standard exponential-family
    differentiation gives ``dH/dgamma = -gamma · Var_p(z)`` (``Var_p`` is the
    variance of the *original*, unscaled logits ``z`` under the *current*
    ``p(gamma)``) — independently re-derivable from
    ``H(gamma) = -gamma·E_p[z] + logZ(gamma)`` and ``dlogZ/dgamma = E_p[z]``.
    """
    B, N, V = logits_BNV.shape
    device = logits_BNV.device
    gid = group_id_B

    log_p0 = torch.log_softmax(logits_BNV, dim=-1)
    base_entropy_BN = -(log_p0.exp() * log_p0).sum(-1)  # [B, N]

    idx_BN = gid.view(B, 1).expand(B, N)
    target_entropy_gN = torch.zeros(B, N, dtype=torch.float32, device=device)
    target_entropy_gN.scatter_add_(
        0, idx_BN, base_entropy_BN * norm_weight_B.view(B, 1)
    )
    target_entropy_BN = target_entropy_gN.index_select(0, gid)

    gamma_BN = torch.ones(B, N, dtype=torch.float32, device=device)
    for _ in range(_ENTROPY_MATCH_ITERS):
        z = logits_BNV * gamma_BN.unsqueeze(-1)
        log_p = torch.log_softmax(z, dim=-1)
        p = log_p.exp()
        entropy_BN = -(p * log_p).sum(-1)
        mean_z = (p * logits_BNV).sum(-1)
        mean_z2 = (p * logits_BNV.square()).sum(-1)
        var_z = (mean_z2 - mean_z.square()).clamp_min(1e-6)
        dH_dgamma = (-gamma_BN * var_z).clamp(max=-1e-6)
        gamma_BN = gamma_BN - (entropy_BN - target_entropy_BN) / dH_dgamma
        gamma_BN = gamma_BN.clamp(
            1.0 / _ENTROPY_MATCH_GAMMA_CLAMP, _ENTROPY_MATCH_GAMMA_CLAMP
        )

    return gamma_BN


class FusionRegistry:
    """Thread-safe registry of which in-flight requests belong to which
    voice-fusion group, at what blend weight, and whether they are the
    group's audio-emitting leader.

    Written by the scheduler's request-build thread (:meth:`set`) and read
    every decode step by the GPU-worker thread (:meth:`is_follower`,
    :meth:`expected_size`, ...). ``_lock`` guards every access so a decode
    step can never observe a half-registered group (which would otherwise
    spuriously trip a group-completeness check). The lock is held only for
    cheap dict ops, never across a GPU forward.

    Pure Python (no torch/sglang dependency) so it is unit-testable standalone
    — see ``test_voice_fusion.py``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._group_of: dict[str, str] = {}
        self._weight_of: dict[str, float] = {}
        self._leader: dict[str, bool] = {}
        # Trajectory-level integral-feedback correction (see
        # ``docs/voice_fusion_design.md``'s "AR 滞后" section): entropy
        # matching only corrects the marginal per-step distribution, which
        # can't reach the autoregressive lock-in that happens once shared
        # context already favors one voice. This tracks a persistent, slowly
        # updated per-request weight tilt (log-scale, so
        # ``effective_weight = weight * exp(delta)``) driven by each step's
        # actual per-member log-likelihood of the emitted frame — unlike
        # ``_weight_of`` (set once at group registration and never touched
        # again), this is written every decode step by the GPU-worker thread
        # itself (see ``HiggsTTSModelRunner._decode_collect_host``), not just
        # read by it.
        self._delta: dict[str, float] = {}
        # group_ids that were ever caught split at prefill (see mark_poisoned)
        # and must be aborted the next time they reach decode, even if by
        # then every member happens to be present again ("healed"). A group
        # that split at prefill sampled at least one unblended frame per
        # present member before the split was noticed (fuse_group_logits has
        # no Req handle at that layer, so it can isolate but not abort) — its
        # KV contexts have already permanently diverged from what a real
        # fused decode would have produced, so "looks complete now" must not
        # be treated as "safe to resume fusing."
        self._poisoned: set[str] = set()
        # Cheap, lock-free "is any fusion request live right now" signal for
        # the hot (non-fusion) path: a server with zero fusion traffic must
        # not pay a per-decode-step lock acquisition + dict-comprehension tax
        # just to learn that, yes, it's still zero. Maintained under
        # ``_lock`` (write side, cheap int increment/decrement) but read
        # without it — a reader can observe a value that's one register/clear
        # stale, which only means an all-singleton step occasionally still
        # takes the (harmless, correct) fusion-aware path, never the reverse.
        self._active_count = 0

    def set(
        self, req_id: str, group_id: str | None, weight: float, *, is_leader: bool
    ) -> None:
        """Register ``req_id`` as a member of voice-fusion group ``group_id``.

        ``group_id is None`` clears any fusion membership (normal request).
        Idempotent: re-registering the same ``req_id`` overwrites in place (no
        double-counting), so a retry that reuses a request id can't inflate
        the group.
        """
        with self._lock:
            if group_id is None:
                old_gid = self._group_of.pop(req_id, None)
                if old_gid is not None:
                    self._active_count -= 1
                self._weight_of.pop(req_id, None)
                self._leader.pop(req_id, None)
                self._delta.pop(req_id, None)
                if old_gid is not None and old_gid not in self._group_of.values():
                    # Last member of this group just cleared — drop its
                    # poisoned marker too, or it would leak forever (group
                    # ids are never reused once every member has released).
                    self._poisoned.discard(old_gid)
                return
            if req_id not in self._group_of:
                self._active_count += 1
            self._group_of[req_id] = group_id
            self._weight_of[req_id] = float(weight)
            self._leader[req_id] = bool(is_leader)
            self._delta[req_id] = 0.0  # fresh group membership starts untilted

    def update_delta(self, req_id: str, new_delta: float) -> None:
        """Overwrite ``req_id``'s trajectory-feedback tilt (see ``_delta``'s
        docstring), clamped to ``±_DELTA_MAX`` here (not left to the caller)
        so every write path — today just
        ``HiggsTTSModelRunner._decode_collect_host``, but any future one too
        — automatically gets the same bound an unbounded integral controller
        would otherwise be able to run away past, the same "confidence
        fine-tunes, weight decides" philosophy as
        ``_ENTROPY_MATCH_GAMMA_CLAMP``.

        Called every decode step for a registered fusion member. A no-op for
        an id that isn't (or is no longer) currently registered — checked
        explicitly rather than writing unconditionally, or a caller racing a
        concurrent ``set(req_id, None, ...)`` clear (or a plain bug passing a
        stale/unknown rid) would leave a ``_delta`` entry with nothing left
        to ever clear it, leaking one dict entry per such call for the
        lifetime of the server.
        """
        with self._lock:
            if req_id in self._group_of:
                clamped = max(-_DELTA_MAX, min(_DELTA_MAX, float(new_delta)))
                self._delta[req_id] = clamped

    def get_delta(self, req_id: str) -> float:
        """Current trajectory-feedback tilt for ``req_id`` (0.0 if unset or
        not a fusion member)."""
        with self._lock:
            return self._delta.get(req_id, 0.0)

    def mark_poisoned(self, group_id: str) -> None:
        """Flag ``group_id`` as having sampled at least one unblended frame
        while split — it must be aborted the next time it reaches decode,
        even if it looks complete by then (see ``_poisoned``'s docstring)."""
        with self._lock:
            self._poisoned.add(group_id)

    def is_poisoned(self, group_id: str) -> bool:
        with self._lock:
            return group_id in self._poisoned

    def has_any(self) -> bool:
        """Lock-free, best-effort "is any fusion request registered right now".

        For the overwhelmingly common non-fusion server, this lets the decode
        hot path skip the fusion bookkeeping (buffer population, per-row
        follower checks) entirely without ever taking the lock. See
        ``_active_count``'s docstring for the staleness tradeoff this makes.
        """
        return self._active_count > 0

    def expected_size(self, group_id: str) -> int:
        """Number of currently-registered members of ``group_id`` (0 if none).

        Derived live from membership (not a separate counter) so it can never
        drift out of sync with the actual registry on retries or partial
        cleanup: a reused request id overwrites its own entry rather than
        incrementing a count.
        """
        with self._lock:
            return sum(1 for g in self._group_of.values() if g == group_id)

    def snapshot(
        self, req_ids: list[str]
    ) -> tuple[dict[str, str], dict[str, float], dict[str, float]]:
        """Atomic snapshot of (group_id, weight, delta) for the given req_ids.

        Taken under the lock so a decode step sees a consistent view of every
        row's membership even if a concurrent register/clear is in flight.
        Returns ``(group_of, weight_of, delta_of)`` restricted to req_ids
        that are fusion members; non-members are absent from all three
        dicts. ``delta_of`` is the trajectory-feedback tilt (see ``_delta``'s
        docstring) — the caller combines it with ``weight_of`` as
        ``weight * exp(delta)`` to get the *effective* blend weight fed into
        this step's pool; the raw nominal weight in ``weight_of`` is never
        itself mutated by the correction.
        """
        with self._lock:
            group_of = {r: self._group_of[r] for r in req_ids if r in self._group_of}
            weight_of = {r: self._weight_of.get(r, 1.0) for r in group_of}
            delta_of = {r: self._delta.get(r, 0.0) for r in group_of}
        return group_of, weight_of, delta_of

    def is_leader(self, req_id: str) -> bool:
        """True iff ``req_id`` is a fusion member and the group's output leader."""
        with self._lock:
            return self._leader.get(req_id, True)

    def is_follower(self, req_id: str) -> bool:
        """True iff ``req_id`` is a fusion member that is NOT the leader (its
        decoded codes duplicate the leader's and must not be emitted)."""
        with self._lock:
            return req_id in self._group_of and not self._leader.get(req_id, True)


def fuse_group_logits(
    logits_BNV: torch.Tensor,
    group_id_B: torch.Tensor,
    weight_B: torch.Tensor,
    *,
    temperature_B: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Blend per-codebook output distributions within each fusion group.

    Args:
        logits_BNV: raw head logits, shape ``[B, N, V]`` (B rows, N codebooks,
            V codec vocab).
        group_id_B: ``[B]`` int. Rows sharing a value are one fusion group.
            For a normal request each row is its own group (``group_id[i] == i``),
            making the blend an identity (up to a constant log shift that does
            not affect argmax / multinomial sampling).
        weight_B: ``[B]`` float blend weight per row. Weights need not be
            normalized; only their within-group ratio matters.
        temperature_B: optional ``[B]`` float. When given, the *blend* is
            computed at ``softmax(logits / temperature)`` so grouped rows fuse
            in the same temperature-scaled space the sampler will use.
            ``None`` blends raw-logit softmax (temperature applied later by
            the caller for every row, grouped or not).

    Returns:
        ``(logits_out, is_grouped_B, matched_logits_BNV)``:

        - ``logits_out``: ``[B, N, V]``. Grouped rows carry the blended
          log-distribution (temperature already folded in — sample these at
          ``temperature=1``). Singleton rows carry their **raw, untouched**
          ``logits_BNV`` — the caller must apply their real ``temperature_B``
          when sampling them, exactly as it would with fusion disabled
          entirely. Do NOT apply temperature to singleton rows a second time:
          this function does not pre-divide them, precisely so the caller's
          one real division is the only one that happens.
        - ``is_grouped_B``: ``[B]`` bool, true for rows in a real (size > 1)
          group. The caller MUST sample grouped rows at ``temperature=1`` (the
          blend already applied ``temperature_B``) but sample singleton rows at
          their **real** ``temperature_B`` — folding every row to 1
          unconditionally defeats the sampler's greedy short-circuit for
          ordinary (non-fusion) requests. See the module docstring.
        - ``matched_logits_BNV``: ``[B, N, V]``, each row's OWN (still
          per-member, not yet group-pooled) temperature-scaled logits after
          entropy-matching rescaling — i.e. exactly what this function feeds
          into the weighted sum, one step before ``scatter_add_`` pools it
          across the group. This is what the trajectory-feedback controller
          (``FusionRegistry.update_delta``,
          ``mean_log_likelihood_of_sampled_frame``) must measure each member's
          own opinion on, NOT the true raw (unmatched) ``logits_BNV``: raw
          per-member logits differ in sharpness by construction (that's the
          whole reason entropy-matching exists), so a controller servoing
          "equalize raw log-likelihood across members" is actually servoing
          raw-space KL-divergence equidistance, which sits far from the
          nominal blend ratio whenever members differ in sharpness — this was
          a real, measured bug (see ``docs/voice_fusion_design.md``'s "AR
          滞后" section): the controller converged to a fixed, systematically
          wrong set-point instead of the intended 50/50 center, and got
          WORSE (not better) as the integral gain increased, because a larger
          gain just reaches that wrong fixed point faster.

    The blend is a weighted **log-linear pool** (product-of-experts / weighted
    geometric mean) over members ``i`` of each group ``g``:
    ``blended_logits_g = Σ_i w_i · (logits_i / T)``, group weights renormalized
    to sum to 1. Because group membership is expressed purely through
    ``scatter_add_`` + advanced indexing, the op is shape-static and safe
    inside a captured CUDA graph.

    Why log-linear (product-of-experts / AND) and not probability-space
    arithmetic averaging (mixture-of-experts / OR): an arithmetic mean of two
    reference voices' ``softmax`` distributions is bimodal whenever the voices
    disagree (it has almost no mass on tokens *neither* voice individually
    favors), so at each decode step sampling from it can only pick a
    voice-A-like or voice-B-like token, never something acoustically between
    the two. Autoregressive coherence then makes whichever mode gets sampled
    at the first few (register-establishing) steps self-reinforcing for the
    rest of the utterance — confirmed by live-inference measurement, not just
    theory: two references at equal weight and independent random seeds
    produced a clean **bimodal** split (every run landed close to voice A's or
    voice B's own pitch, none intermediate), while a heavily skewed weight
    (e.g. 0.9/0.1) was stable and consistent across seeds — ruling out "weight
    is ignored" and pointing squarely at probability-space averaging being
    unable to represent a compromise token in the first place. A weighted
    geometric mean does not have this failure mode: it concentrates mass
    exactly on tokens *both* experts assign reasonable probability to, which
    is what a blended-timbre frame actually is.
    ``softmax(Σ_i w_i · logits_i/T)`` is *exactly* proportional to
    ``Π_i softmax(logits_i/T)^{w_i}`` (the weighted geometric mean of the
    per-member distributions) — the two differ only by a per-group additive
    log-constant that ``softmax``/``argmax``/top-k/top-p are all invariant to
    — so pooling the temperature-scaled logits directly is both mathematically
    exact and simpler than the old probability-space route (no ``softmax`` /
    ``log`` round-trip, no ``_LOG_FLOOR`` needed on the grouped path).

    Singleton-group rows (the entire non-fusion batch) are returned as
    ``logits_BNV`` **unchanged** — bit-identical to what the sampler would
    have received without fusion — rather than pre-divided by temperature,
    which the caller already does downstream (dividing twice would silently
    sharpen/dull every ordinary request's sampling — this was a real bug
    caught in review, not a hypothetical). The singleton-vs-blended choice is
    a per-row ``torch.where`` (tensor op, no host branch), so it stays
    CUDA-Graph-safe in a mixed fusion/non-fusion batch.
    """
    if logits_BNV.ndim != 3:
        raise ValueError(f"logits_BNV must be [B, N, V], got {tuple(logits_BNV.shape)}")
    B, N, V = logits_BNV.shape
    device = logits_BNV.device

    raw_logits = logits_BNV.float()
    logits = raw_logits
    if temperature_B is not None:
        safe_temp = temperature_B.to(device).clamp_min(1e-5).view(B, 1, 1)
        logits = logits / safe_temp

    gid = group_id_B.to(device=device, dtype=torch.long)
    w = weight_B.to(device=device, dtype=torch.float32)

    # Per-group member count + weight sum (both via scatter_add_, CG-safe).
    ones = torch.ones(B, dtype=torch.float32, device=device)
    group_count = torch.zeros(B, dtype=torch.float32, device=device)
    group_count.scatter_add_(0, gid, ones)
    group_weight_sum = torch.zeros(B, dtype=torch.float32, device=device)
    group_weight_sum.scatter_add_(0, gid, w)

    # Per-group weight normalization: divide each row's weight by its group's
    # total, so the pooled log-distribution stays a properly weighted mean.
    norm_w = w / group_weight_sum[gid].clamp_min(_LOG_FLOOR)  # [B]

    # Entropy-matched rescaling: log-linear pooling is precision-weighted, not
    # weight-only — a member whose (temperature-scaled) distribution happens
    # to be sharper (lower entropy) than its groupmates' pulls the pooled
    # result toward itself regardless of the nominal blend weight (confirmed
    # via live inference: two distinct reference voices at nominal 0.5/0.5
    # still landed close to whichever one had the sharper per-step
    # distribution in 7/8 independent runs, not a stable 50/50 compromise).
    # Rescale each member's logits by a per-(row, codebook) factor so every
    # member enters the pool at its GROUP's weighted-average entropy first —
    # only then does the nominal weight decide the outcome, as intended.
    gamma = _solve_entropy_matching_gamma(logits, gid, norm_w)
    matched_logits = logits * gamma.unsqueeze(-1)

    weighted = matched_logits * norm_w.view(B, 1, 1)  # [B, N, V]
    idx = gid.view(B, 1, 1).expand(B, N, V)
    fused = torch.zeros_like(logits)
    fused.scatter_add_(0, idx, weighted)  # group g accumulates its members

    # Broadcast each group's pooled logits back onto all its member rows.
    blended_logits = fused.index_select(0, gid)

    # Rows in a real (size > 1) group get the blended log-probs (temperature
    # already folded in); singleton rows get their exact RAW logits back —
    # the caller applies the real per-row temperature exactly once, downstream
    # — so non-fusion decoding is bit-identical to baseline. Per-row select —
    # no host branch.
    is_grouped_B = group_count.index_select(0, gid) > 1.5
    logits_out = torch.where(is_grouped_B.view(B, 1, 1), blended_logits, raw_logits)
    return logits_out, is_grouped_B, matched_logits


def fuse_group_generation_done(
    generation_done_B: torch.Tensor,
    group_id_B: torch.Tensor,
) -> torch.Tensor:
    """ "Any sibling done ⇒ all done" group barrier.

    Returns a ``[B]`` bool where a row is done iff *any* member of its fusion
    group is done. For singleton groups this is an identity. Keeping group
    members' ``generation_done`` synchronized makes them terminate on the same
    AR step, so their sibling KV contexts never desynchronize.
    """
    device = generation_done_B.device
    gid = group_id_B.to(device=device, dtype=torch.long)
    done_f = generation_done_B.to(torch.float32)
    group_any = torch.zeros(
        generation_done_B.shape[0], dtype=torch.float32, device=device
    )
    group_any.scatter_add_(0, gid, done_f)
    return group_any.index_select(0, gid) > 0


def mean_log_likelihood_of_sampled_frame(
    observation_logits_BNV: torch.Tensor,
    temperature_B: torch.Tensor,
    codes_BN: torch.Tensor,
) -> torch.Tensor:
    """Each row's own mean-over-codebooks log-probability of the frame that
    was ACTUALLY sampled this step — "how much does this row's own,
    un-pooled distribution endorse the emitted trajectory". This is the
    per-step observation the trajectory-feedback controller
    (``FusionRegistry.update_delta``) is driven by: entropy-matched log-linear
    pooling (see ``fuse_group_logits``) only corrects the per-step *marginal*
    distribution, which can't reach the autoregressive lock-in that happens
    once the shared sampled context already favors one group member over
    another — measuring each member's likelihood of the trajectory itself,
    integrated over many steps, is what can.

    Args:
        observation_logits_BNV: each row's own, still per-member (not
            group-pooled) logits, evaluated at the group's shared
            entropy-matched confidence scale — i.e. ``fuse_group_logits``'s
            ``matched_logits_BNV`` return, NOT its true raw (pre-entropy-match)
            head logits. Using the true raw per-member logits here was a
            real, measured bug: two members always differ in native
            sharpness (that's exactly what entropy-matching corrects for the
            *blend*), so measuring raw log-likelihood servos raw-space
            KL-divergence equidistance between members instead of the
            intended "match the nominal blend ratio" — a fixed,
            systematically wrong set-point that a larger integral gain
            reaches *faster*, not one it corrects (see
            ``docs/voice_fusion_design.md``'s "AR 滞后" section). Using the
            group-*pooled* consensus logits instead would be wrong in the
            opposite direction — that would just measure how much a row
            agrees with a consensus it already voted on, not its independent
            opinion.
        temperature_B: ``[B]`` per-row temperature. Pass ``torch.ones_like``
            when ``observation_logits_BNV`` is already temperature-scaled
            (true of ``matched_logits_BNV``, which folds in ``temperature_B``
            before entropy-matching) — dividing again would double-apply it.
        codes_BN: ``[B, N]`` long, the codebook tokens actually sampled this
            step (post-fusion — the whole point is measuring each row's own
            opinion of the group's shared, already-decided outcome). Clamped
            to ``>= 0`` before use: a row already ``generation_done`` at step
            entry decodes ``STOP_CODE`` (``-1``) for every codebook (see
            ``sampler.py``), and an ungathered ``-1`` index is out of bounds
            for ``V`` (a device-side assert under CUDA, fatal to the whole
            engine) — every other consumer of these codes
            (``model_runner.py``'s ``post_decode_launch``/
            ``_decode_step_logprobs``) clamps for the exact same reason. A
            clamped row's own likelihood is never actually used: the
            fusion-group "any done ⇒ all done" barrier means a done row's
            group is done too, so this observation has no live group left to
            update by the time the caller aggregates it.

    Returns:
        ``[B]`` float32. No grouping/weighting applied here — purely a
        per-row observation; the caller aggregates it per group (see
        ``docs/voice_fusion_design.md``).
    """
    device = observation_logits_BNV.device
    safe_temp = temperature_B.to(device=device).clamp_min(1e-5).view(-1, 1, 1)
    log_p_BNV = torch.log_softmax(observation_logits_BNV.float() / safe_temp, dim=-1)
    codes_BN1 = codes_BN.to(device=device, dtype=torch.long).clamp_min(0).unsqueeze(-1)
    ell_BN = log_p_BNV.gather(-1, codes_BN1).squeeze(-1)
    return ell_BN.mean(dim=-1)


__all__ = [
    "FusionRegistry",
    "fuse_group_logits",
    "fuse_group_generation_done",
    "mean_log_likelihood_of_sampled_frame",
]
