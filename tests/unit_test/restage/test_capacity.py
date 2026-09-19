import math

import pytest

from sglang_omni.restage.capacity import (
    SharingDiscount,
    StageCapacity,
    Workload,
    footprint_gib,
    kappa_lower,
    plan_utility,
    pool_requests,
)

WORKLOAD = Workload(context_tokens=250, audio_seconds=4.5)
FULL = {"engine": 0.9, "tail": 0.9}


def capacity(name, throughput, *, kv=None, weights=1.0, provenance="MEASURED test"):
    return StageCapacity(
        process=name,
        throughput=throughput,
        provenance=provenance,
        weights_gib=weights,
        kv_bytes_per_token=kv,
        delta_s=0.05,
    )


CAPS = {"engine": capacity("engine", 30.0), "tail": capacity("tail", 20.0)}


def utility(
    assignments, flows, *, fractions=FULL, mode="timeslice", caps=CAPS, discounts=None
):
    return plan_utility(
        assignments,
        flows,
        caps,
        WORKLOAD,
        gpu_mem_gib=80.0,
        fractions=fractions,
        sharing_mode=mode,
        discounts=discounts,
    )


def test_kappa_rises_with_throughput_and_falls_with_stall():
    fast = kappa_lower(40.0, 0.05, 4.5)
    slow = kappa_lower(10.0, 0.05, 4.5)
    stalled = kappa_lower(40.0, 0.5, 4.5)
    assert fast > slow
    assert fast > stalled


def test_pool_requests_uses_weights_split_by_tp():
    cap = capacity("engine", 20.0, kv=20480, weights=10.0)
    single = pool_requests(cap, WORKLOAD, 0.5, 80.0)
    tp2 = pool_requests(cap, WORKLOAD, 0.5, 80.0, tp_size=2)
    assert tp2 > single > 0
    assert pool_requests(cap, WORKLOAD, 0.1, 80.0) == 0.0
    assert pool_requests(capacity("tail", 5.0), WORKLOAD, 0.5, 80.0) is None


def test_footprint_adds_kappa_sized_pool_only_for_kv_holders():
    tail = capacity("tail", 5.0, weights=1.0)
    engine = capacity("engine", 5.0, kv=20480, weights=1.0)
    assert footprint_gib(tail, WORKLOAD) == pytest.approx(3.0)
    assert footprint_gib(engine, WORKLOAD) > 3.0


def test_one_pipeline_on_one_gpu_serializes_its_stages():
    # Note (Jiaxin Deng): the calibration derives stage throughputs from busy
    # shares, so one flow on one GPU must reproduce the pipeline throughput.
    result = utility(
        {"engine": ((0,),), "tail": ((0,),)}, {"engine": (0,), "tail": (0,)}
    )
    assert result.utility == pytest.approx(1 / (1 / 30 + 1 / 20))
    assert result.binding == "gpu0"
    assert result.provenance == "MEASURED"


def test_split_is_bounded_by_the_slower_dedicated_stage():
    result = utility(
        {"engine": ((0,),), "tail": ((1,),)}, {"engine": (0,), "tail": (1,)}
    )
    assert result.utility == pytest.approx(20.0)
    assert result.binding == "gpu1"
    replicated_tail = utility(
        {"engine": ((0,),), "tail": ((1,), (2,))}, {"engine": (0,), "tail": (1, 2)}
    )
    assert replicated_tail.utility == pytest.approx(30.0)
    assert replicated_tail.binding == "gpu0"


def test_competing_flows_retain_the_sharing_discount():
    colocated = utility(
        {"engine": ((0,), (0,)), "tail": ((0,), (0,))},
        {"engine": (0, 1), "tail": (0, 1)},
    )
    assert colocated.utility == pytest.approx(2 * 0.58 / (1 / 30 + 1 / 20))
    consolidated = utility(
        {"engine": ((0,), (1,)), "tail": ((2,), (2,))},
        {"engine": (0, 1), "tail": (0, 1)},
        fractions={"engine": 0.9, "tail": 0.45},
        mode="mps",
    )
    assert consolidated.utility == pytest.approx(2 * 20.0 * 0.85)
    assert consolidated.binding == "gpu2"
    assert [row.capacity for row in consolidated.gpus] == [1.0, 1.0, pytest.approx(1.7)]


def test_prior_discount_marks_the_plan_predicted_until_measured_at_that_fan_in():
    assignments = {"engine": ((0,), (0,)), "tail": ((0,), (0,))}
    flows = {"engine": (0, 1), "tail": (0, 1)}
    prior = utility(assignments, flows, mode="mps")
    assert prior.provenance == "PREDICTED"
    assert prior.sharing["mps@2"].provenance.startswith("PRIOR")
    other_fan_in = {"mps@3": SharingDiscount(0.9, "MEASURED this model c8")}
    assert (
        utility(assignments, flows, mode="mps", discounts=other_fan_in).provenance
        == "PREDICTED"
    )
    measured = utility(
        assignments,
        flows,
        mode="mps",
        discounts={"mps@2": SharingDiscount(0.7, "MEASURED this model c8")},
    )
    assert measured.provenance == "MEASURED"
    assert measured.utility == pytest.approx(2 * 0.7 / (1 / 30 + 1 / 20))
    with pytest.raises(ValueError):
        SharingDiscount(1.2, "MEASURED")


def test_pool_bound_caps_a_starved_kv_pool():
    caps = {"engine": capacity("engine", 30.0, kv=4 * 1024 * 1024, weights=10.0)}
    starved = plan_utility(
        {"engine": ((0,),)},
        {"engine": (0,)},
        caps,
        WORKLOAD,
        gpu_mem_gib=16.0,
        fractions={"engine": 0.7},
    )
    roomy = plan_utility(
        {"engine": ((0,),)},
        {"engine": (0,)},
        caps,
        WORKLOAD,
        gpu_mem_gib=80.0,
        fractions={"engine": 0.9},
    )
    assert starved.binding == "pool:engine"
    assert starved.utility < roomy.utility == pytest.approx(30.0)


def test_predicted_inputs_and_tp_mark_the_plan_predicted():
    prior = {"engine": capacity("engine", 30.0, provenance="PRIOR registry")}
    assert (
        plan_utility(
            {"engine": ((0,),)},
            {"engine": (0,)},
            prior,
            WORKLOAD,
            gpu_mem_gib=80.0,
            fractions={"engine": 0.9},
        ).provenance
        == "PREDICTED"
    )
    measured = {"engine": capacity("engine", 30.0)}
    tp = plan_utility(
        {"engine": ((0, 1),)},
        {"engine": (0,)},
        measured,
        WORKLOAD,
        gpu_mem_gib=80.0,
        fractions={"engine": 0.9},
    )
    assert tp.provenance == "PREDICTED"
    assert tp.utility == pytest.approx(30.0)


def test_rejects_mismatched_inputs():
    caps = {"engine": capacity("engine", 30.0)}
    with pytest.raises(ValueError):
        plan_utility(
            {"tail": ((0,),)},
            {"tail": (0,)},
            caps,
            WORKLOAD,
            gpu_mem_gib=80.0,
            fractions={},
        )
    with pytest.raises(ValueError):
        plan_utility(
            {"engine": ((0,),)},
            {"engine": (0, 1)},
            caps,
            WORKLOAD,
            gpu_mem_gib=80.0,
            fractions={"engine": 0.9},
        )
    with pytest.raises(ValueError):
        plan_utility(
            {"engine": ((0,),)},
            {"engine": (0,)},
            caps,
            WORKLOAD,
            gpu_mem_gib=80.0,
            fractions={"engine": 0.9},
            sharing_mode="green",
        )
    with pytest.raises(ValueError):
        capacity("engine", math.nan)
