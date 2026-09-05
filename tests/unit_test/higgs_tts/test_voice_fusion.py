# SPDX-License-Identifier: Apache-2.0
"""Unit tests for voice-fusion blend ops (pure torch, no sglang engine)."""

import math

import pytest
import torch

from sglang_omni.models.higgs_tts.fusion import (
    FusionRegistry,
    _solve_entropy_matching_gamma,
    fuse_group_generation_done,
    fuse_group_logits,
    mean_log_likelihood_of_sampled_frame,
)


def _singleton_groups(B):
    return torch.arange(B, dtype=torch.long), torch.ones(B, dtype=torch.float32)


def _entropy(logits_NV):
    log_p = torch.log_softmax(logits_NV, dim=-1)
    return -(log_p.exp() * log_p).sum(-1)


def _permuted_same_entropy(logits_NV, seed):
    """A same-shape logits tensor with independently-assigned values but
    IDENTICAL per-row entropy to ``logits_NV`` (permuting the vocab axis
    leaves each row's entropy exactly unchanged) — lets a test isolate "is
    the pooling formula correct" from entropy-matching's rescaling (which is
    a no-op, gamma == 1, only when group members already share one entropy).
    """
    V = logits_NV.shape[-1]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(V, generator=g)
    return logits_NV[:, perm]


def test_singleton_is_sampling_identity():
    """Each row its own group → blended log-probs sample-equivalent to raw logits.

    log(softmax(logits)) differs from logits only by a per-row constant, which
    leaves argmax and multinomial(softmax(.)) unchanged.
    """
    torch.manual_seed(0)
    B, N, V = 4, 8, 1026
    logits = torch.randn(B, N, V)
    gid, w = _singleton_groups(B)
    out, is_grouped, _ = fuse_group_logits(logits, gid, w)
    assert not is_grouped.any()
    # argmax preserved per (row, codebook)
    assert torch.equal(out.argmax(-1), logits.argmax(-1))
    # softmax preserved (log-prob == log-softmax up to fp error)
    torch.testing.assert_close(
        out.softmax(-1), logits.float().softmax(-1), atol=1e-5, rtol=1e-4
    )


def test_singleton_is_byte_identical_to_raw_logits_regardless_of_temperature():
    """A singleton row returns the RAW logits *exactly* (byte-identical), no
    matter what ``temperature_B`` is passed — never pre-divided.

    This is a regression guard for a real bug caught in review: the caller
    (``model.py``) applies the row's real temperature exactly once, itself,
    downstream, using the ``is_grouped_B`` mask this function returns. If this
    function also divided singleton rows by temperature before returning them,
    every ordinary (non-fusion) request would get sampled at ``T²`` instead of
    ``T`` — silently sharper or duller than requested, no crash, no test
    failure unless this exact contract is pinned down.
    """
    torch.manual_seed(7)
    B, N, V = 3, 8, 1026
    logits = torch.randn(B, N, V)
    gid, w = _singleton_groups(B)
    # temperature == 1: out must equal the raw logits (division by 1 is moot
    # either way, so this alone would NOT catch the T² bug).
    out1, is_grouped1, _ = fuse_group_logits(
        logits, gid, w, temperature_B=torch.ones(B)
    )
    assert not is_grouped1.any()
    assert torch.equal(out1, logits.float())
    # arbitrary per-row temperature != 1: out must STILL equal the raw logits,
    # not logits / T. This is the case the T² bug would actually break.
    temp = torch.tensor([0.7, 1.0, 1.5])
    out2, is_grouped2, _ = fuse_group_logits(logits, gid, w, temperature_B=temp)
    assert not is_grouped2.any()
    assert torch.equal(out2, logits.float())


def test_mixed_batch_singleton_rows_byte_identical_to_raw_logits():
    """In a batch mixing a fused group with singletons, the singleton rows
    stay byte-identical to the RAW logits (not logits/T) while the fused rows
    blend — same T² regression guard as above, in a mixed batch."""
    torch.manual_seed(8)
    N, V = 8, 1026
    logits = torch.randn(4, N, V)
    gid = torch.tensor([0, 0, 2, 3], dtype=torch.long)  # rows 0,1 fused; 2,3 alone
    w = torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32)
    temp = torch.tensor([1.0, 1.0, 0.8, 1.3])
    out, is_grouped, _ = fuse_group_logits(logits, gid, w, temperature_B=temp)
    assert is_grouped.tolist() == [True, True, False, False]
    # singleton rows: exact raw logits, untouched by temperature
    assert torch.equal(out[2], logits[2].float())
    assert torch.equal(out[3], logits[3].float())
    # fused rows: blended (not equal to either raw row)
    assert not torch.equal(out[0], logits[0].float())


def test_two_member_equal_weight_is_geometric_mean():
    """A 2-row group at 0.5/0.5 yields the weighted geometric mean (product-of-
    experts / log-linear pool) of the two softmaxes, NOT their arithmetic mean.

    This is the fix for a real, measured bug: pooling in probability space
    (arithmetic mean) is bimodal whenever the two references disagree, so
    autoregressive sampling from it locks onto one reference or the other
    (confirmed live: equal-weight fusion of two distinct voices split cleanly
    into "sounds like A" / "sounds like B" across random seeds, never
    intermediate). Log-linear pooling concentrates mass on tokens BOTH
    references find plausible instead.

    Both rows are constructed to share one entropy (see
    ``_permuted_same_entropy``) so entropy-matching is a no-op (gamma == 1)
    here and this test isolates the pooling formula itself; entropy-matching's
    own rescaling behavior is covered separately below.
    """
    torch.manual_seed(1)
    N, V = 8, 1026
    row0 = torch.randn(N, V)
    row1 = _permuted_same_entropy(row0, seed=101)
    logits = torch.stack([row0, row1])
    gid = torch.tensor([0, 0], dtype=torch.long)
    w = torch.tensor([0.5, 0.5], dtype=torch.float32)
    out, is_grouped, _ = fuse_group_logits(logits, gid, w)
    assert is_grouped.all()

    geometric_mean = (logits[0].softmax(-1) ** 0.5) * (logits[1].softmax(-1) ** 0.5)
    geometric_mean = geometric_mean / geometric_mean.sum(-1, keepdim=True)
    arithmetic_mean = 0.5 * logits[0].softmax(-1) + 0.5 * logits[1].softmax(-1)

    # both rows carry the same pooled distribution
    torch.testing.assert_close(out[0].softmax(-1), geometric_mean, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(out[1].softmax(-1), out[0].softmax(-1))
    # and it must NOT be the old (buggy) arithmetic mean
    assert not torch.allclose(out[0].softmax(-1), arithmetic_mean, atol=1e-3)


# --------------------------------------------------------------------------- #
# _solve_entropy_matching_gamma — log-linear pooling is precision-weighted,
# not weight-only (a real, measured failure: two distinct reference voices at
# NOMINAL 0.5/0.5 landed close to whichever one had the sharper per-step
# distribution in 7/8 independent live-inference runs, not a stable
# compromise). This rescales each group member to its group's
# weighted-average entropy BEFORE pooling, so the nominal weight is what
# actually decides the outcome.
# --------------------------------------------------------------------------- #
def test_entropy_matching_is_near_identity_for_already_matched_entropy():
    """Two rows that already share one entropy need no correction: gamma
    should solve to (very close to) 1 for both."""
    torch.manual_seed(20)
    N, V = 8, 1026
    row0 = torch.randn(N, V)
    row1 = _permuted_same_entropy(row0, seed=201)
    logits = torch.stack([row0, row1])
    gid = torch.tensor([0, 0], dtype=torch.long)
    norm_w = torch.tensor([0.5, 0.5], dtype=torch.float32)
    gamma = _solve_entropy_matching_gamma(logits, gid, norm_w)
    torch.testing.assert_close(gamma, torch.ones_like(gamma), atol=0.02, rtol=0)


def test_entropy_matching_sharpens_the_flatter_member_toward_the_target():
    """A flatter (higher-entropy) member gets a gamma > 1 correction
    (sharpened up toward the group's weighted-average entropy) — the
    mechanism that stops a flatter voice from being drowned out by a
    groupmate's naturally sharper distribution regardless of nominal weight.

    Scale ratio (1.4x) is modeled on the live-inference finding that even a
    mild sharpness asymmetry (~1.25x logit amplitude) is enough to flip which
    reference dominates an equal-weight pool — this is the realistic
    magnitude the fixed 3-iteration Newton solve needs to handle well, not an
    adversarially extreme case (see the separate clamp test for that).
    """
    torch.manual_seed(21)
    N, V = 8, 1026
    sharp = torch.randn(N, V) * 1.4  # larger logit spread -> lower entropy
    flat = torch.randn(N, V) * 1.0
    logits = torch.stack([sharp, flat])
    gid = torch.tensor([0, 0], dtype=torch.long)
    norm_w = torch.tensor([0.5, 0.5], dtype=torch.float32)

    assert (_entropy(sharp) < _entropy(flat)).all()

    gamma = _solve_entropy_matching_gamma(logits, gid, norm_w)
    # sharp member (index 0) should be relaxed toward the target (gamma < 1);
    # flat member (index 1) should be sharpened up (gamma > 1).
    assert (gamma[0] < 1.0).all()
    assert (gamma[1] > 1.0).all()

    # after applying gamma, the two members' entropies should converge much
    # closer together than they started (not necessarily exact in 3 Newton
    # steps, but the gap should shrink substantially for a realistic
    # asymmetry magnitude).
    pre_gap = (_entropy(sharp) - _entropy(flat)).abs()
    post_sharp = _entropy(sharp * gamma[0].unsqueeze(-1))
    post_flat = _entropy(flat * gamma[1].unsqueeze(-1))
    post_gap = (post_sharp - post_flat).abs()
    assert bool((post_gap < pre_gap * 0.15).all())


def test_entropy_matching_respects_group_boundaries():
    """A row's gamma only depends on its OWN group's target entropy, not on
    unrelated rows/groups in the same batch."""
    torch.manual_seed(22)
    N, V = 8, 1026
    a = torch.randn(N, V) * 4.0
    b = torch.randn(N, V) * 0.25
    c = torch.randn(N, V) * 4.0  # same scale as `a`, different (unrelated) group
    logits = torch.stack([a, b, c])
    gid = torch.tensor([0, 0, 1], dtype=torch.long)  # a,b grouped; c its own group
    norm_w = torch.tensor([0.5, 0.5, 1.0], dtype=torch.float32)
    gamma = _solve_entropy_matching_gamma(logits, gid, norm_w)
    # c is a singleton group -> its own target IS its own entropy -> no-op.
    torch.testing.assert_close(gamma[2], torch.ones_like(gamma[2]), atol=0.02, rtol=0)


def test_entropy_matching_gamma_stays_within_clamp_for_extreme_asymmetry():
    """An extreme sharpness mismatch must not blow up gamma to something
    numerically wild — it's bounded by the documented clamp."""
    torch.manual_seed(23)
    N, V = 8, 1026
    extremely_sharp = torch.randn(N, V) * 50.0
    extremely_flat = torch.randn(N, V) * 0.01
    logits = torch.stack([extremely_sharp, extremely_flat])
    gid = torch.tensor([0, 0], dtype=torch.long)
    norm_w = torch.tensor([0.5, 0.5], dtype=torch.float32)
    gamma = _solve_entropy_matching_gamma(logits, gid, norm_w)
    assert torch.isfinite(gamma).all()
    assert bool((gamma >= 1.0 / 3.0 - 1e-6).all())
    assert bool((gamma <= 3.0 + 1e-6).all())


def test_three_member_group_is_weighted_geometric_mean():
    """The log-linear pool generalizes to N > 2 members with arbitrary
    weights, not just the equal-weight 2-member case. All three rows share
    one entropy (see ``_permuted_same_entropy``) so entropy-matching is a
    no-op here, isolating the pooling formula."""
    torch.manual_seed(9)
    N, V = 8, 1026
    row0 = torch.randn(N, V)
    row1 = _permuted_same_entropy(row0, seed=102)
    row2 = _permuted_same_entropy(row0, seed=103)
    logits = torch.stack([row0, row1, row2])
    gid = torch.tensor([0, 0, 0], dtype=torch.long)
    w = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32)
    out, is_grouped, _ = fuse_group_logits(logits, gid, w)
    assert is_grouped.all()

    probs = [logits[i].softmax(-1) for i in range(3)]
    weights = [0.5, 0.3, 0.2]
    expected = probs[0] ** weights[0] * probs[1] ** weights[1] * probs[2] ** weights[2]
    expected = expected / expected.sum(-1, keepdim=True)
    for i in range(3):
        torch.testing.assert_close(out[i].softmax(-1), expected, atol=1e-4, rtol=1e-3)


def test_weight_ratio_only():
    """Unnormalized weights blend by ratio: [3,1] == [0.75,0.25]."""
    torch.manual_seed(2)
    N, V = 8, 1026
    logits = torch.randn(2, N, V)
    gid = torch.tensor([0, 0], dtype=torch.long)
    out_raw, _, _ = fuse_group_logits(logits, gid, torch.tensor([3.0, 1.0]))
    out_norm, _, _ = fuse_group_logits(logits, gid, torch.tensor([0.75, 0.25]))
    torch.testing.assert_close(
        out_raw.softmax(-1), out_norm.softmax(-1), atol=1e-6, rtol=1e-5
    )


def test_fused_rows_sample_identically_with_shared_seed():
    """Two fused rows + same seed draw the same multi-codebook frame."""
    torch.manual_seed(3)
    N, V = 8, 1026
    logits = torch.randn(2, N, V)
    gid = torch.tensor([0, 0], dtype=torch.long)
    fused, _, _ = fuse_group_logits(logits, gid, torch.tensor([0.5, 0.5]))
    g0 = torch.Generator().manual_seed(42)
    g1 = torch.Generator().manual_seed(42)
    s0 = fused[0].softmax(-1).multinomial(1, generator=g0)
    s1 = fused[1].softmax(-1).multinomial(1, generator=g1)
    assert torch.equal(s0, s1)


def test_mixed_batch_groups_and_singletons():
    """A batch mixing a 2-row group and a singleton blends only within group.
    Rows 0,1 share one entropy so entropy-matching is a no-op here."""
    torch.manual_seed(4)
    N, V = 8, 1026
    row0 = torch.randn(N, V)
    row1 = _permuted_same_entropy(row0, seed=104)
    row2 = torch.randn(N, V)
    logits = torch.stack([row0, row1, row2])
    gid = torch.tensor([0, 0, 2], dtype=torch.long)  # rows 0,1 grouped; row 2 alone
    w = torch.tensor([0.5, 0.5, 1.0], dtype=torch.float32)
    out, is_grouped, _ = fuse_group_logits(logits, gid, w)
    assert is_grouped.tolist() == [True, True, False]
    expected01 = (logits[0].softmax(-1) ** 0.5) * (logits[1].softmax(-1) ** 0.5)
    expected01 = expected01 / expected01.sum(-1, keepdim=True)
    torch.testing.assert_close(out[0].softmax(-1), expected01, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(out[1].softmax(-1), expected01, atol=1e-4, rtol=1e-3)
    # singleton untouched (sample-equivalent)
    assert torch.equal(out[2].argmax(-1), logits[2].argmax(-1))


def test_generation_done_barrier():
    """Any done in a group ⇒ all done; singletons untouched."""
    gid = torch.tensor([0, 0, 0, 3], dtype=torch.long)
    done = torch.tensor([False, True, False, False])
    out = fuse_group_generation_done(done, gid)
    assert out.tolist() == [True, True, True, False]


def test_generation_done_singletons_identity():
    done = torch.tensor([True, False, True])
    gid = torch.arange(3, dtype=torch.long)
    assert torch.equal(fuse_group_generation_done(done, gid), done)


# --------------------------------------------------------------------------- #
# mean_log_likelihood_of_sampled_frame — the per-step observation driving
# FusionRegistry's trajectory-feedback delta (entropy matching alone corrects
# only the marginal distribution per step, not the AR lock-in across steps).
# --------------------------------------------------------------------------- #
def test_mean_log_likelihood_matches_hand_computed_value():
    torch.manual_seed(30)
    B, N, V = 2, 3, 8
    logits = torch.randn(B, N, V)
    temp = torch.ones(B)
    codes = torch.randint(0, V, (B, N))

    ell = mean_log_likelihood_of_sampled_frame(logits, temp, codes)

    expected = torch.zeros(B)
    for b in range(B):
        lp = torch.log_softmax(logits[b], dim=-1)
        per_codebook = torch.stack([lp[n, codes[b, n]] for n in range(N)])
        expected[b] = per_codebook.mean()
    torch.testing.assert_close(ell, expected, atol=1e-5, rtol=1e-4)


def test_mean_log_likelihood_is_highest_for_the_argmax_frame():
    """A row's own likelihood of the frame it would itself have picked
    (argmax at every codebook) must be its highest possible value — sanity
    check that this is really measuring "does this row endorse the sampled
    frame", not something inverted."""
    torch.manual_seed(31)
    N, V = 4, 16
    logits = torch.randn(1, N, V)
    own_argmax = logits[0].argmax(-1).unsqueeze(0)  # [1, N]
    other = torch.randint(0, V, (1, N))
    temp = torch.ones(1)

    ell_own = mean_log_likelihood_of_sampled_frame(logits, temp, own_argmax)
    ell_other = mean_log_likelihood_of_sampled_frame(logits, temp, other)
    assert bool((ell_own >= ell_other).all())


def test_mean_log_likelihood_respects_temperature():
    """A hotter temperature flattens the distribution, pulling every
    non-degenerate frame's log-likelihood toward the uniform value
    -log(V) — including the argmax frame's, which must therefore DROP as
    temperature rises (it starts above uniform and is squeezed toward it)."""
    torch.manual_seed(32)
    N, V = 4, 16
    logits = torch.randn(1, N, V) * 3.0  # sharp distribution
    codes = logits[0].argmax(-1).unsqueeze(0)

    ell_cold = mean_log_likelihood_of_sampled_frame(logits, torch.tensor([0.5]), codes)
    ell_hot = mean_log_likelihood_of_sampled_frame(logits, torch.tensor([5.0]), codes)
    assert bool((ell_hot < ell_cold).all())


def test_mean_log_likelihood_clamps_stop_code_instead_of_crashing():
    """A row already ``generation_done`` at step entry decodes STOP_CODE
    (-1) for every codebook (see sampler.py) -- gathering a raw -1 index is
    out of bounds and, unclamped, would be a fatal device-side assert under
    CUDA inside a captured graph. Must not raise, and (harmlessly, since a
    done row's fusion group is done too and this observation is never
    consumed) just clamps to codebook 0 instead."""
    torch.manual_seed(33)
    B, N, V = 2, 3, 8
    logits = torch.randn(B, N, V)
    temp = torch.ones(B)
    codes = torch.full((B, N), -1, dtype=torch.long)

    ell = mean_log_likelihood_of_sampled_frame(logits, temp, codes)

    assert torch.isfinite(ell).all()
    expected = mean_log_likelihood_of_sampled_frame(
        logits, temp, torch.zeros((B, N), dtype=torch.long)
    )
    torch.testing.assert_close(ell, expected)


# --------------------------------------------------------------------------- #
# Regression test for a real, measured bug in the trajectory-feedback
# controller (round 4): the delta observation must be measured on
# ``fuse_group_logits``'s ``matched_logits`` (entropy-matched, still
# per-member), NOT the true raw per-member logits. Two members that differ
# in native sharpness (any two distinct real reference voices) produce a
# systematically nonzero raw-observation gap even at a genuine nominal
# 50/50 pool -- confirmed live (the controller converged to a fixed WRONG
# voice, and got there FASTER, not more centered, as the integral gain
# increased) and by a standalone closed-loop simulation. See
# docs/voice_fusion_design.md's "AR 滞后" section.
# --------------------------------------------------------------------------- #
def test_delta_observation_on_raw_logits_is_biased_toward_the_sharper_member():
    """Sanity check that the bug is real and reproducible: averaged over many
    independent (sharp, flat) trials, measuring each member's OWN raw logits'
    likelihood of the group's actually-sampled shared frame gives the sharper
    member a systematically higher score than the flatter one, even though
    the pool was requested at equal weight."""
    torch.manual_seed(40)
    N, V = 8, 1026
    n_trials = 100
    gid = torch.tensor([0, 0], dtype=torch.long)
    norm_w = torch.tensor([0.5, 0.5], dtype=torch.float32)
    ones2 = torch.ones(2)

    raw_gaps = []
    for _ in range(n_trials):
        sharp = torch.randn(N, V) * 1.4
        flat = torch.randn(N, V) * 1.0
        logits = torch.stack([sharp, flat])
        out, _, _ = fuse_group_logits(logits, gid, norm_w)
        codes = out[0].argmax(-1).unsqueeze(0).expand(2, -1)

        ell_raw = mean_log_likelihood_of_sampled_frame(logits, ones2, codes)
        raw_gaps.append((ell_raw[0] - ell_raw[1]).item())

    raw_gap_mean = sum(raw_gaps) / n_trials
    # Consistent sign, not just noise: the sharper member (index 0) is
    # systematically less "surprised" by the shared frame.
    assert raw_gap_mean > 0.1


def test_delta_observation_on_matched_logits_is_not_biased_by_sharpness():
    """The fix: measuring on ``matched_logits`` instead of raw per-member
    logits removes the sharpness bias the test above demonstrates -- averaged
    over the same kind of (sharp, flat) trials, the matched-logits gap must
    sit much closer to zero than the raw-logits gap does."""
    torch.manual_seed(40)
    N, V = 8, 1026
    n_trials = 100
    gid = torch.tensor([0, 0], dtype=torch.long)
    norm_w = torch.tensor([0.5, 0.5], dtype=torch.float32)
    ones2 = torch.ones(2)

    raw_gaps = []
    matched_gaps = []
    for _ in range(n_trials):
        sharp = torch.randn(N, V) * 1.4
        flat = torch.randn(N, V) * 1.0
        logits = torch.stack([sharp, flat])
        out, _, matched = fuse_group_logits(logits, gid, norm_w)
        codes = out[0].argmax(-1).unsqueeze(0).expand(2, -1)

        ell_raw = mean_log_likelihood_of_sampled_frame(logits, ones2, codes)
        raw_gaps.append((ell_raw[0] - ell_raw[1]).item())

        ell_matched = mean_log_likelihood_of_sampled_frame(matched, ones2, codes)
        matched_gaps.append((ell_matched[0] - ell_matched[1]).item())

    raw_gap_mean = sum(raw_gaps) / n_trials
    matched_gap_mean = sum(matched_gaps) / n_trials
    assert abs(matched_gap_mean) < abs(raw_gap_mean) * 0.5


def test_temperature_applied_before_blend():
    """temperature_B scales each row's logits before the log-linear pool.
    Rows share one entropy (post temperature-scaling) so entropy-matching is
    a no-op here, isolating the temperature/pooling interaction."""
    torch.manual_seed(5)
    N, V = 8, 1026
    row0 = torch.randn(N, V)
    row1 = _permuted_same_entropy(row0, seed=105)
    logits = torch.stack([row0, row1])
    gid = torch.tensor([0, 0], dtype=torch.long)
    w = torch.tensor([0.5, 0.5])
    temp = torch.tensor([2.0, 2.0])
    out, _, _ = fuse_group_logits(logits, gid, w, temperature_B=temp)
    p0 = (logits[0] / 2.0).softmax(-1)
    p1 = (logits[1] / 2.0).softmax(-1)
    expected = (p0**0.5) * (p1**0.5)
    expected = expected / expected.sum(-1, keepdim=True)
    torch.testing.assert_close(out[0].softmax(-1), expected, atol=1e-4, rtol=1e-3)


# --------------------------------------------------------------------------- #
# Regression guard for the Linus-review BLOCKING-1 finding: a caller that
# unconditionally sets ``sampler_temperature = 1`` after calling
# ``fuse_group_logits`` silently defeats the sampler's greedy short-circuit for
# every ordinary (non-fusion) request. These tests exercise the *actual*
# invariant that matters — the sampled codes, not an intermediate tensor — so
# they fail loudly if a future caller reintroduces the bug.
# --------------------------------------------------------------------------- #
def _greedy_sample(logits_NV: torch.Tensor) -> torch.Tensor:
    """Mirrors ``sampler._sample_independent``'s greedy branch: plain argmax."""
    return logits_NV.argmax(dim=-1)


def _wrong_caller_sample(
    logits_NV: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """The BLOCKING-1 bug, reproduced directly: blend with temperature folded
    in, then unconditionally resample at temperature=1 regardless of whether
    the row was ever actually grouped. For a singleton row this multinomial-
    samples a near-one-hot distribution instead of taking the argmax."""
    gid = torch.arange(logits_NV.shape[0], dtype=torch.long)
    w = torch.ones(logits_NV.shape[0])
    temp = torch.full((logits_NV.shape[0],), 1e-5)  # requested: greedy
    blended, _, _ = fuse_group_logits(
        logits_NV.unsqueeze(1), gid, w, temperature_B=temp
    )
    probs = blended.squeeze(1).softmax(dim=-1)
    return probs.multinomial(num_samples=1, generator=generator).squeeze(-1)


def _correct_caller_sample(logits_NV: torch.Tensor) -> torch.Tensor:
    """The fixed contract: singleton rows sample at their real temperature
    (here ~0 → greedy), so they must go through argmax exactly like baseline
    and must NOT touch the RNG at all."""
    gid = torch.arange(logits_NV.shape[0], dtype=torch.long)
    w = torch.ones(logits_NV.shape[0])
    temp = torch.full((logits_NV.shape[0],), 1e-5)
    blended, is_grouped, _ = fuse_group_logits(
        logits_NV.unsqueeze(1), gid, w, temperature_B=temp
    )
    assert not is_grouped.any()  # every row here is a singleton
    sampler_temp = torch.where(is_grouped, torch.ones_like(temp), temp)
    # Mirror the batched sampler's greedy short-circuit: temperature<=threshold
    # rows go through argmax — no multinomial call, no RNG consumed.
    assert bool((sampler_temp <= 1e-5).all())
    return blended.squeeze(1).argmax(-1)


def test_singleton_greedy_sampling_matches_baseline_not_the_blocking1_bug():
    """A plain (non-fusion) request with temperature=0 must decode by argmax,
    byte-identical to the no-fusion baseline, and must not consume any RNG
    state (greedy is deterministic and mustn't perturb other rows' draws in
    the same batch). The BLOCKING-1 bug — a caller that unconditionally
    resamples at temperature=1 after the blend — breaks BOTH properties for
    every ordinary request, fusion or not: it becomes a multinomial draw
    (RNG-consuming, only *probabilistically* matching argmax) instead of a
    deterministic, RNG-free argmax.
    """
    torch.manual_seed(11)
    B, V = 4, 1026
    logits = torch.randn(B, V)
    baseline = _greedy_sample(logits)

    # The buggy caller path takes a torch.Generator only because it MUST
    # consume RNG state (multinomial) — the fixed path below takes none.
    # Confirm bug reproduction: same seed, but it's a real sampling call.
    g = torch.Generator().manual_seed(123)
    state_before = g.get_state().clone()
    _wrong_caller_sample(logits, g)
    state_after_wrong = g.get_state()
    assert not torch.equal(state_before, state_after_wrong), (
        "sanity check: the buggy always-temperature=1 caller pattern is "
        "expected to consume RNG state via multinomial — if it doesn't, this "
        "helper no longer reproduces BLOCKING-1"
    )

    # The fix: singleton rows must sample at their real temperature, matching
    # baseline exactly and touching no RNG at all (deterministic argmax).
    correct = _correct_caller_sample(logits)
    assert torch.equal(correct, baseline)


def test_singleton_nongreedy_sampling_is_not_scaled_by_temperature_twice():
    """A plain (non-fusion) request at an ORDINARY temperature (neither ~0 nor
    exactly 1) must be sampled at its real T, not T². This is the regression
    the greedy test above cannot catch: argmax is scale-invariant, so a caller
    that fed a pre-divided ``logits/T`` back into a second ``/T`` division
    would still pick the same argmax — the greedy test passes either way.
    Softmax is not scale-invariant, so this test exercises the actual caller
    contract (``fuse_group_logits`` then ``torch.where(is_grouped, 1, T)``
    then divide-by-T once, exactly as ``model.py``'s CG/eager paths do) end
    to end, and compares the resulting *distribution* against the no-fusion
    baseline.
    """
    torch.manual_seed(13)
    B, V = 5, 1026
    logits = torch.randn(B, V)
    temp = torch.tensor([0.7, 1.3, 0.9, 1.1, 0.5])  # never 1, never ~0

    gid = torch.arange(B, dtype=torch.long)
    w = torch.ones(B)
    blended, is_grouped, _ = fuse_group_logits(
        logits.unsqueeze(1), gid, w, temperature_B=temp
    )
    assert not is_grouped.any()
    sampler_temp = torch.where(is_grouped, torch.ones_like(temp), temp)
    # The real caller's one and only division by temperature.
    actual_probs = (blended.squeeze(1) / sampler_temp.view(B, 1)).softmax(dim=-1)

    baseline_probs = (logits / temp.view(B, 1)).softmax(dim=-1)
    torch.testing.assert_close(actual_probs, baseline_probs, atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------------- #
# FusionRegistry — the engine-side bookkeeping backing HiggsTTSModel's
# set_fusion_group/has_any_fusion/is_fusion_follower/... (Linus-review
# MAJOR-4: the non-fusion hot path must skip fusion work at zero cost). Pure
# Python, no torch/sglang dependency, so the counter/registry logic itself is
# directly unit-testable here rather than only indirectly via the model.
# --------------------------------------------------------------------------- #
def test_registry_starts_empty():
    reg = FusionRegistry()
    assert reg.has_any() is False
    assert reg.expected_size("g0") == 0
    assert reg.is_leader("nope") is True  # default: a non-member is its own leader
    assert reg.is_follower("nope") is False


def test_registry_register_marks_has_any_and_expected_size():
    reg = FusionRegistry()
    reg.set("r0", "g0", 0.5, is_leader=True)
    reg.set("r1", "g0", 0.5, is_leader=False)
    assert reg.has_any() is True
    assert reg.expected_size("g0") == 2
    assert reg.is_leader("r0") is True
    assert reg.is_follower("r0") is False
    assert reg.is_leader("r1") is False
    assert reg.is_follower("r1") is True


def test_registry_expected_size_only_counts_matching_group():
    reg = FusionRegistry()
    reg.set("r0", "g0", 1.0, is_leader=True)
    reg.set("r1", "g0", 1.0, is_leader=False)
    reg.set("r2", "g1", 1.0, is_leader=True)
    assert reg.expected_size("g0") == 2
    assert reg.expected_size("g1") == 1
    assert reg.expected_size("g-missing") == 0


def test_registry_clear_one_member_keeps_has_any_true_for_the_rest():
    reg = FusionRegistry()
    reg.set("r0", "g0", 0.5, is_leader=True)
    reg.set("r1", "g0", 0.5, is_leader=False)
    reg.set("r0", None, 1.0, is_leader=True)  # clear leader only
    assert reg.has_any() is True  # r1 still registered
    assert reg.expected_size("g0") == 1
    assert reg.is_follower("r0") is False  # no longer a member at all


def test_registry_clear_last_member_resets_has_any():
    reg = FusionRegistry()
    reg.set("r0", "g0", 0.5, is_leader=True)
    reg.set("r1", "g0", 0.5, is_leader=False)
    reg.set("r0", None, 1.0, is_leader=True)
    reg.set("r1", None, 1.0, is_leader=False)
    assert reg.has_any() is False
    assert reg.expected_size("g0") == 0


def test_registry_reregistering_same_req_id_does_not_inflate_active_count():
    """Idempotent re-registration (e.g. a retry reusing a request id) must
    overwrite in place, not double-count — else ``has_any`` could get stuck
    True after every member is cleared once."""
    reg = FusionRegistry()
    reg.set("r0", "g0", 0.5, is_leader=True)
    reg.set("r0", "g0", 0.9, is_leader=True)  # re-register same id, same group
    reg.set("r0", None, 1.0, is_leader=True)  # single clear must fully zero it out
    assert reg.has_any() is False


def test_registry_clear_of_never_registered_id_is_a_no_op():
    reg = FusionRegistry()
    reg.set("ghost", None, 1.0, is_leader=True)
    assert reg.has_any() is False


def test_registry_reused_id_after_clear_and_reregister_has_correct_count():
    """Register → clear → register again on the same req_id (id reuse across
    requests) must leave the registry in exactly the single-member state, not
    drift the active count from stale increments/decrements."""
    reg = FusionRegistry()
    reg.set("r0", "g0", 0.5, is_leader=True)
    reg.set("r0", None, 1.0, is_leader=True)
    reg.set("r0", "g1", 0.7, is_leader=False)
    assert reg.has_any() is True
    assert reg.expected_size("g0") == 0
    assert reg.expected_size("g1") == 1
    assert reg.is_follower("r0") is True


def test_registry_snapshot_restricted_to_members():
    reg = FusionRegistry()
    reg.set("r0", "g0", 0.7, is_leader=True)
    reg.set("r1", "g0", 0.3, is_leader=False)
    group_of, weight_of, delta_of = reg.snapshot(["r0", "r1", "not-a-member"])
    assert group_of == {"r0": "g0", "r1": "g0"}
    assert weight_of == {"r0": pytest.approx(0.7), "r1": pytest.approx(0.3)}
    assert delta_of == {"r0": pytest.approx(0.0), "r1": pytest.approx(0.0)}


def test_registry_snapshot_empty_for_all_non_members():
    reg = FusionRegistry()
    group_of, weight_of, delta_of = reg.snapshot(["a", "b"])
    assert group_of == {}
    assert weight_of == {}
    assert delta_of == {}


def test_registry_poisoned_group_starts_unpoisoned():
    reg = FusionRegistry()
    assert reg.is_poisoned("g0") is False


def test_registry_mark_poisoned_is_visible():
    reg = FusionRegistry()
    reg.set("r0", "g0", 1.0, is_leader=True)
    reg.set("r1", "g0", 1.0, is_leader=False)
    reg.mark_poisoned("g0")
    assert reg.is_poisoned("g0") is True
    # A different group is unaffected.
    assert reg.is_poisoned("g1") is False


def test_registry_poisoned_flag_survives_partial_clear():
    """A group poisoned while split (some members not yet registered/already
    cleared) must stay poisoned as long as ANY member is still registered —
    clearing one member must not erase the poison for the survivors."""
    reg = FusionRegistry()
    reg.set("r0", "g0", 1.0, is_leader=True)
    reg.set("r1", "g0", 1.0, is_leader=False)
    reg.mark_poisoned("g0")
    reg.set("r0", None, 1.0, is_leader=True)  # r0 clears (e.g. aborted)
    assert reg.is_poisoned("g0") is True  # r1 still registered


def test_registry_poisoned_flag_clears_when_last_member_clears():
    """Once every member of a poisoned group has cleared, the poison marker
    must not leak forever — group ids are never reused once fully released."""
    reg = FusionRegistry()
    reg.set("r0", "g0", 1.0, is_leader=True)
    reg.mark_poisoned("g0")
    reg.set("r0", None, 1.0, is_leader=True)
    assert reg.is_poisoned("g0") is False


# --------------------------------------------------------------------------- #
# FusionRegistry delta — the trajectory-level integral-feedback tilt, on top
# of entropy matching, for the AR-hysteresis lock-in that per-step marginal
# correction alone can't reach (see docs/voice_fusion_design.md).
# --------------------------------------------------------------------------- #
def test_registry_delta_starts_at_zero_for_a_fresh_member():
    reg = FusionRegistry()
    reg.set("r0", "g0", 0.5, is_leader=True)
    assert reg.get_delta("r0") == pytest.approx(0.0)


def test_registry_delta_unset_member_is_zero():
    reg = FusionRegistry()
    assert reg.get_delta("ghost") == pytest.approx(0.0)


def test_registry_update_delta_is_visible_via_get_delta():
    reg = FusionRegistry()
    reg.set("r0", "g0", 0.5, is_leader=True)
    reg.update_delta("r0", 0.2)
    assert reg.get_delta("r0") == pytest.approx(0.2)


def test_registry_update_delta_is_clamped():
    reg = FusionRegistry()
    reg.set("r0", "g0", 0.5, is_leader=True)
    reg.update_delta("r0", 10.0)
    assert reg.get_delta("r0") == pytest.approx(math.log(2.0))
    reg.update_delta("r0", -10.0)
    assert reg.get_delta("r0") == pytest.approx(-math.log(2.0))


def test_registry_update_delta_is_a_no_op_for_an_unregistered_id():
    """Writing a delta for an id that was never (or is no longer)
    registered must not leak a dict entry — there would be nothing left to
    ever clear it."""
    reg = FusionRegistry()
    reg.update_delta("ghost", 0.3)
    assert reg.get_delta("ghost") == pytest.approx(0.0)
    assert "ghost" not in reg._delta


def test_registry_delta_resets_to_zero_on_fresh_registration():
    """Re-registering a request id (e.g. id reuse, or a retry) starts its
    tilt fresh at 0 rather than carrying over a stale value from a
    previous, unrelated group membership."""
    reg = FusionRegistry()
    reg.set("r0", "g0", 0.5, is_leader=True)
    reg.update_delta("r0", 0.3)
    reg.set("r0", None, 1.0, is_leader=True)
    reg.set("r0", "g1", 0.5, is_leader=True)
    assert reg.get_delta("r0") == pytest.approx(0.0)


def test_registry_delta_cleared_when_member_clears():
    reg = FusionRegistry()
    reg.set("r0", "g0", 0.5, is_leader=True)
    reg.update_delta("r0", 0.3)
    reg.set("r0", None, 1.0, is_leader=True)
    assert "r0" not in reg._delta
