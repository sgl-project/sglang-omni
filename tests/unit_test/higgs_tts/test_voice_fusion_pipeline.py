# SPDX-License-Identifier: Apache-2.0
"""Pipeline-level tests for Higgs voice fusion: preprocessing detection,
builder fan-out, and group lifecycle.

These import the request/scheduler layer, which pulls in ``sglang`` — so they
run only in a full sglang-omni environment (Linux + sgl_kernel), not in a bare
torch venv. The pure-tensor blend math is covered separately by
``test_voice_fusion.py`` (no sglang import), which runs anywhere.
"""

# ruff: noqa: E402 -- imports below the importorskip guards are intentional:
# they must not run at all when sglang/torch aren't installed.

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sglang")

from sglang.srt.managers.schedule_batch import FINISH_ABORT

from sglang_omni.models.higgs_tts.fusion import FusionRegistry
from sglang_omni.models.higgs_tts.model_runner import HiggsTTSModelRunner
from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.models.higgs_tts.request_builders import build_fusion_sibling_requests
from sglang_omni.models.higgs_tts.stages import _fusion_ref_entries


# --------------------------------------------------------------------------- #
# Fusion-request detection (_fusion_ref_entries)
# --------------------------------------------------------------------------- #
def test_detect_requires_two_weighted_refs():
    # < 2 refs → not fusion
    assert (
        _fusion_ref_entries({"references": [{"audio_path": "a.wav", "weight": 1.0}]})
        is None
    )
    # 2 refs but no weight → legacy single-ref path, not fusion
    assert (
        _fusion_ref_entries(
            {"references": [{"audio_path": "a.wav"}, {"audio_path": "b.wav"}]}
        )
        is None
    )


def test_detect_two_weighted_refs():
    specs = _fusion_ref_entries(
        {
            "references": [
                {"audio_path": "a.wav", "weight": 0.7, "text": "hi"},
                {"audio_path": "b.wav", "weight": 0.3},
            ]
        }
    )
    assert specs is not None
    assert len(specs) == 2
    assert specs[0]["weight"] == 0.7
    assert specs[0]["audio"] == "a.wav"
    assert specs[0]["reference_text"] == "hi"
    assert specs[1]["weight"] == 0.3


def test_detect_rejects_negative_weight():
    with pytest.raises(ValueError, match="weight"):
        _fusion_ref_entries(
            {
                "references": [
                    {"audio_path": "a.wav", "weight": -1.0},
                    {"audio_path": "b.wav", "weight": 0.5},
                ]
            }
        )


def test_detect_pre_encoded_codes():
    specs = _fusion_ref_entries(
        {
            "references": [
                {"reference_codes": [[1, 2, 3, 4, 5, 6, 7, 8]], "weight": 0.5},
                {"reference_codes": [[8, 7, 6, 5, 4, 3, 2, 1]], "weight": 0.5},
            ]
        }
    )
    assert specs is not None
    assert specs[0]["codes"] == [[1, 2, 3, 4, 5, 6, 7, 8]]
    assert specs[0]["audio"] is None


# --------------------------------------------------------------------------- #
# Builder fan-out (build_fusion_sibling_requests)
# --------------------------------------------------------------------------- #
def _fusion_state(n: int) -> HiggsTtsState:
    """A fusion state with ``n`` pre-built sibling refs (prompt + delayed codes)."""
    refs = []
    for i in range(n):
        refs.append(
            {
                "codes_delayed": [[i % 1024] * 8 for _ in range(3 + i)],
                "weight": 1.0,
                "prompt_token_ids": [10, 20, -100, -100, -100, 30, 40],
                "reference_text": None,
            }
        )
    return HiggsTtsState(
        prompt_token_ids=[],
        fusion_refs=refs,
        target_text="hello",
        num_codebooks=8,
        codebook_size=1026,
        max_new_tokens=256,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
    )


def test_fanout_produces_n_siblings():
    leader = build_fusion_sibling_requests(_fusion_state(3), request_id="rid-x")
    followers = leader.fusion_siblings
    assert followers is not None
    assert len(followers) == 2  # leader + 2 followers = 3
    group = [leader, *followers]
    assert all(s.fusion_group_id == "rid-x" for s in group)


def test_fanout_leader_and_followers():
    leader = build_fusion_sibling_requests(_fusion_state(3), request_id="rid-y")
    group = [leader, *(leader.fusion_siblings or [])]
    assert leader.fusion_is_leader is True
    assert sum(1 for s in group if s.fusion_is_leader) == 1
    assert [s.fusion_is_leader for s in group[1:]] == [False, False]


def test_fanout_shares_one_seed():
    leader = build_fusion_sibling_requests(_fusion_state(3), request_id="rid-z")
    group = [leader, *(leader.fusion_siblings or [])]
    seeds = {s.req.sampling_params.sampling_seed for s in group}
    assert len(seeds) == 1  # all siblings share one concrete seed
    assert next(iter(seeds)) is not None


def test_fanout_distinct_rids():
    leader = build_fusion_sibling_requests(_fusion_state(4), request_id="rid-w")
    group = [leader, *(leader.fusion_siblings or [])]
    rids = [s.req.rid for s in group]
    assert rids[0] == "rid-w"
    assert len(set(rids)) == 4  # all distinct


def test_fanout_weights_preserved():
    state = _fusion_state(2)
    state.fusion_refs[0]["weight"] = 0.7
    state.fusion_refs[1]["weight"] = 0.3
    leader = build_fusion_sibling_requests(state, request_id="rid-v")
    group = [leader, *(leader.fusion_siblings or [])]
    assert group[0].fusion_weight == pytest.approx(0.7)
    assert group[1].fusion_weight == pytest.approx(0.3)


def test_fanout_rejects_single_ref():
    with pytest.raises(ValueError, match=">= 2"):
        build_fusion_sibling_requests(_fusion_state(1), request_id="rid-u")


# --------------------------------------------------------------------------- #
# HiggsTTSModelRunner._populate_fusion_buffers — the decode-time
# group-completeness guard (Linus-review BLOCKING-3: a group split mid-decode
# by a KV-pressure retract must isolate just that group's present rows rather
# than abort the whole batch). Constructed the same way as
# test_async_decode_runner.py: bypass the heavy __init__ and stub only what
# this method touches, backed by a real ``FusionRegistry`` so the registry
# side of the interaction is exercised for real, not re-mocked.
# --------------------------------------------------------------------------- #
def _build_populate_buffers_runner(bs: int):
    registry = FusionRegistry()
    runner = object.__new__(HiggsTTSModelRunner)
    runner._fusion_buffers_dirty = False
    runner.model = SimpleNamespace(
        has_any_fusion=registry.has_any,
        fusion_membership_snapshot=registry.snapshot,
        expected_fusion_group_size=registry.expected_size,
        is_fusion_group_poisoned=registry.is_poisoned,
        get_fusion_delta=registry.get_delta,
        update_fusion_delta=registry.update_delta,
        _cg_fusion_group=torch.arange(bs, dtype=torch.long),
        _cg_fusion_weight=torch.ones(bs, dtype=torch.float32),
    )
    return runner, registry


def _fake_requests(n: int):
    """``n`` fake sched_req-like objects with default ids r0..r{n-1}; override
    ``.request_id`` on the returned objects for custom ids. ``req.finished()``
    mirrors real sglang's semantics off the SAME ``finished_reason`` a test
    may set later (e.g. via ``_populate_fusion_buffers``'s abort path) — a
    live closure over ``reqs[i]``, not a snapshot, so it stays correct
    regardless of when a test mutates ``finished_reason``."""
    reqs = [SimpleNamespace(finished_reason=None) for _ in range(n)]
    for r in reqs:
        r.finished = lambda r=r: r.finished_reason is not None
    datas = [SimpleNamespace(req=reqs[i]) for i in range(n)]
    requests = [SimpleNamespace(request_id=f"r{i}", data=datas[i]) for i in range(n)]
    return requests, reqs


def test_populate_buffers_no_fusion_registered_is_a_no_op():
    """Zero fusion traffic ever registered: early-exits before touching the
    registry at all (the MAJOR-4 hot-path guarantee)."""
    runner, _registry = _build_populate_buffers_runner(3)
    requests, reqs = _fake_requests(3)
    runner._populate_fusion_buffers(requests, bs=3, n_real=3)
    assert torch.equal(runner.model._cg_fusion_group, torch.arange(3))
    assert torch.equal(runner.model._cg_fusion_weight, torch.ones(3))
    assert all(r.finished_reason is None for r in reqs)
    assert runner._fusion_buffers_dirty is False


def test_populate_buffers_intact_group_blends():
    """Both members of a 2-row group present → batch-local group id anchors
    on the first member's row, weights carried through, no abort."""
    runner, registry = _build_populate_buffers_runner(3)
    requests, reqs = _fake_requests(3)
    registry.set("r0", "gid-a", 0.6, is_leader=True)
    registry.set("r1", "gid-a", 0.4, is_leader=False)
    runner._populate_fusion_buffers(requests, bs=3, n_real=3)
    assert runner.model._cg_fusion_group.tolist() == [0, 0, 2]
    assert runner.model._cg_fusion_weight.tolist() == pytest.approx([0.6, 0.4, 1.0])
    assert all(r.finished_reason is None for r in reqs)
    assert runner._fusion_buffers_dirty is True


def test_populate_buffers_applies_trajectory_feedback_delta_to_weight():
    """The effective weight fed into the blend is nominal_weight *
    exp(delta), not the raw nominal weight — see FusionRegistry's delta
    docstring. A fresh registration has delta=0 (test above pins that as an
    identity); this pins the case where a persistent tilt has accumulated."""
    runner, registry = _build_populate_buffers_runner(2)
    requests, reqs = _fake_requests(2)
    registry.set("r0", "gid-a", 0.5, is_leader=True)
    registry.set("r1", "gid-a", 0.5, is_leader=False)
    registry.update_delta("r0", 0.3)
    registry.update_delta("r1", -0.3)
    runner._populate_fusion_buffers(requests, bs=2, n_real=2)
    assert runner.model._cg_fusion_weight.tolist() == pytest.approx(
        [0.5 * math.exp(0.3), 0.5 * math.exp(-0.3)]
    )


def test_populate_buffers_split_group_isolates_present_row_and_aborts():
    """A group expecting 2 members but only 1 present this step (the sibling
    was retracted) must: demote the present row to a standalone singleton (no
    blend), abort its request, and leave an unrelated row in the same batch
    untouched."""
    runner, registry = _build_populate_buffers_runner(3)
    requests, reqs = _fake_requests(2)
    requests[1].request_id = "other"
    registry.set("r0", "gid-a", 0.5, is_leader=True)
    registry.set("sib1", "gid-a", 0.5, is_leader=False)  # absent from this batch
    runner._populate_fusion_buffers(requests, bs=3, n_real=2)
    assert runner.model._cg_fusion_group.tolist() == [0, 1, 2]  # r0 demoted to own slot
    assert runner.model._cg_fusion_weight.tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert isinstance(reqs[0].finished_reason, FINISH_ABORT)
    assert reqs[1].finished_reason is None  # unrelated row untouched


def test_populate_buffers_poisoned_group_aborted_even_once_healed():
    """A group poisoned by an earlier prefill-time split (model.py::
    _batch_local_fusion) must be aborted the first time it reaches decode
    with ALL members present — looking complete is not enough once poisoned,
    because it already sampled an unblended frame per member before the
    split was caught, permanently desyncing their KV contexts."""
    runner, registry = _build_populate_buffers_runner(2)
    requests, reqs = _fake_requests(2)
    registry.set("r0", "gid-a", 0.5, is_leader=True)
    registry.set("r1", "gid-a", 0.5, is_leader=False)
    registry.mark_poisoned("gid-a")
    runner._populate_fusion_buffers(requests, bs=2, n_real=2)
    # Both present, counts match (2/2) - an unpoisoned group would blend.
    # A poisoned one must still be isolated + aborted despite looking intact.
    assert runner.model._cg_fusion_group.tolist() == [0, 1]
    assert runner.model._cg_fusion_weight.tolist() == pytest.approx([1.0, 1.0])
    assert isinstance(reqs[0].finished_reason, FINISH_ABORT)
    assert isinstance(reqs[1].finished_reason, FINISH_ABORT)


def test_populate_buffers_unpoisoned_intact_group_still_blends():
    """Sanity check for the test above: an otherwise-identical intact group
    that was NEVER poisoned must blend normally, not get aborted."""
    runner, registry = _build_populate_buffers_runner(2)
    requests, reqs = _fake_requests(2)
    registry.set("r0", "gid-a", 0.5, is_leader=True)
    registry.set("r1", "gid-a", 0.5, is_leader=False)
    runner._populate_fusion_buffers(requests, bs=2, n_real=2)
    assert runner.model._cg_fusion_group.tolist() == [0, 0]
    assert all(r.finished_reason is None for r in reqs)


def test_populate_buffers_never_overwrites_an_already_finished_request():
    """A row already finished for another reason keeps that reason — the
    split-group abort must not clobber a pre-existing finish."""
    runner, registry = _build_populate_buffers_runner(2)
    requests, reqs = _fake_requests(1)
    registry.set("r0", "gid-a", 0.5, is_leader=True)
    registry.set("sib1", "gid-a", 0.5, is_leader=False)
    sentinel = object()
    reqs[0].finished_reason = sentinel
    runner._populate_fusion_buffers(requests, bs=2, n_real=1)
    assert reqs[0].finished_reason is sentinel


def test_populate_buffers_dirty_flag_resets_after_fusion_clears():
    """Once every fusion member is cleared, the next call's scan finds
    nothing to group — buffers reset to all-singleton and the dirty flag
    drops, so the call after that takes the zero-cost early-exit again."""
    runner, registry = _build_populate_buffers_runner(2)
    requests, _reqs = _fake_requests(2)
    registry.set("r0", "gid-a", 0.5, is_leader=True)
    registry.set("r1", "gid-a", 0.5, is_leader=False)
    runner._populate_fusion_buffers(requests, bs=2, n_real=2)
    assert runner._fusion_buffers_dirty is True

    registry.set("r0", None, 1.0, is_leader=True)
    registry.set("r1", None, 1.0, is_leader=True)
    runner._populate_fusion_buffers(requests, bs=2, n_real=2)
    assert runner.model._cg_fusion_group.tolist() == [0, 1]
    assert runner.model._cg_fusion_weight.tolist() == pytest.approx([1.0, 1.0])
    assert runner._fusion_buffers_dirty is False


def test_populate_buffers_clean_reset_scrubs_stale_slots_beyond_a_shrunk_batch():
    """Regression guard: a dirty-to-clean transition at a SMALLER bs than the
    fusion step that dirtied the buffer must not leave stale non-identity
    values at slots >= the smaller bs — those buffers are fixed pool_size and
    never resized per step, so a later batch growing back into those slots
    with brand-new, unrelated traffic would otherwise silently reuse the
    finished group's leftover group/weight values and get blended together.
    """
    runner, registry = _build_populate_buffers_runner(4)  # pool_size=4

    # Step 1, bs=4: a real fusion group at slots 2,3; slots 0,1 ordinary.
    requests4, _reqs4 = _fake_requests(4)
    registry.set("r2", "gid-a", 0.6, is_leader=True)
    registry.set("r3", "gid-a", 0.4, is_leader=False)
    runner._populate_fusion_buffers(requests4, bs=4, n_real=4)
    assert runner.model._cg_fusion_group.tolist() == [0, 1, 2, 2]
    assert runner.model._cg_fusion_weight.tolist() == pytest.approx(
        [1.0, 1.0, 0.6, 0.4]
    )
    assert runner._fusion_buffers_dirty is True

    # Step 2, bs=2: the fusion group finished and the batch shrank down to
    # just the two ordinary rows — a normal, common sequence of events, not
    # an edge case. This is the dirty-to-clean transition.
    registry.set("r2", None, 1.0, is_leader=True)
    registry.set("r3", None, 1.0, is_leader=True)
    requests2, _reqs2 = _fake_requests(2)
    runner._populate_fusion_buffers(requests2, bs=2, n_real=2)
    assert runner._fusion_buffers_dirty is False
    # The bug: without a full-buffer reset, slots 2,3 would still read
    # [2, 2] / [0.6, 0.4] here, left over from step 1's write to [:4].
    assert runner.model._cg_fusion_group.tolist() == [0, 1, 2, 3]
    assert runner.model._cg_fusion_weight.tolist() == pytest.approx(
        [1.0, 1.0, 1.0, 1.0]
    )

    # Step 3, bs=4 again: traffic grows back with brand-new, unrelated
    # requests reoccupying slots 2,3 - no fusion registered at all this time.
    # has_any_fusion() and _fusion_buffers_dirty are both False, so this call
    # takes the early-return and trusts the buffer as-is.
    requests4b, _reqs4b = _fake_requests(4)
    runner._populate_fusion_buffers(requests4b, bs=4, n_real=4)
    assert runner.model._cg_fusion_group.tolist() == [0, 1, 2, 3]
    assert runner.model._cg_fusion_weight.tolist() == pytest.approx(
        [1.0, 1.0, 1.0, 1.0]
    )


# --------------------------------------------------------------------------- #
# HiggsTTSModelRunner._update_fusion_deltas — the CG-decode-path host-side
# half of the trajectory-feedback controller (see FusionRegistry's _delta
# docstring): given each row's own pre-fusion log-likelihood of the frame the
# group just sampled together, nudges each member's persistent delta toward
# closing the gap with its group's nominal-weighted-average opinion.
# --------------------------------------------------------------------------- #
def test_update_fusion_deltas_pushes_the_dominant_members_delta_down():
    """The member whose own distribution liked the sampled frame MORE than
    the group's weighted average gets its delta DECREASED (dampening its
    effective weight next step); the other gets INCREASED."""
    runner, registry = _build_populate_buffers_runner(2)
    requests, _reqs = _fake_requests(2)
    registry.set("r0", "gid-a", 0.5, is_leader=True)
    registry.set("r1", "gid-a", 0.5, is_leader=False)
    # r0's own distribution strongly endorses the sampled frame (-0.1 nat);
    # r1's barely does (-5.0 nat) -- r0 is "winning" this step.
    ell = torch.tensor([-0.1, -5.0])

    runner._update_fusion_deltas(requests, ell)

    assert registry.get_delta("r0") < 0.0
    assert registry.get_delta("r1") > 0.0


def test_update_fusion_deltas_is_symmetric_for_equal_likelihoods():
    """If both members equally endorse the sampled frame, there's no gap to
    correct -- delta stays at 0 for both."""
    runner, registry = _build_populate_buffers_runner(2)
    requests, _reqs = _fake_requests(2)
    registry.set("r0", "gid-a", 0.5, is_leader=True)
    registry.set("r1", "gid-a", 0.5, is_leader=False)
    ell = torch.tensor([-2.0, -2.0])

    runner._update_fusion_deltas(requests, ell)

    assert registry.get_delta("r0") == pytest.approx(0.0)
    assert registry.get_delta("r1") == pytest.approx(0.0)


def test_update_fusion_deltas_is_a_no_op_for_non_fusion_requests():
    runner, registry = _build_populate_buffers_runner(2)
    requests, _reqs = _fake_requests(2)
    ell = torch.tensor([-1.0, -3.0])

    runner._update_fusion_deltas(requests, ell)  # must not raise

    assert registry.get_delta("r0") == pytest.approx(0.0)
    assert registry.get_delta("r1") == pytest.approx(0.0)


def test_update_fusion_deltas_skips_a_group_with_only_one_row_present():
    """A group's expected size is 2 but only one member is in THIS batch
    (e.g. the other hasn't reached decode yet) -- nothing meaningful to
    aggregate against, so its delta is left untouched rather than compared
    against itself."""
    runner, registry = _build_populate_buffers_runner(1)
    requests, _reqs = _fake_requests(1)
    registry.set("r0", "gid-a", 0.5, is_leader=True)
    registry.set("sib1", "gid-a", 0.5, is_leader=False)  # absent from this batch
    ell = torch.tensor([-1.0])

    runner._update_fusion_deltas(requests, ell)

    assert registry.get_delta("r0") == pytest.approx(0.0)
