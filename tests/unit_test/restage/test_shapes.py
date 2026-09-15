import pytest

from sglang_omni.restage.shapes import (
    ProcessShape,
    instance_fractions,
    residency_shapes,
)

ENGINE = ProcessShape("engine", footprint_gib=10.0, backbone=True)
TAIL = ProcessShape("tail", footprint_gib=3.0)


def names(shapes):
    return sorted(s.name for s in shapes)


def test_single_process_pipeline_only_replicates():
    shapes = residency_shapes([ENGINE], [0, 1], gpu_mem_gib=80.0)
    assert names(shapes) == [
        "colocate_x2_mps",
        "colocate_x2_timeslice",
        "colocate_x3_mps",
        "colocate_x3_timeslice",
        "colocate_x4_mps",
        "colocate_x4_timeslice",
        "replicate_x1",
        "replicate_x2",
    ]
    by_name = {s.name: s for s in shapes}
    assert by_name["replicate_x1"].idle_devices == (1,)
    assert by_name["replicate_x2"].assignments == {"engine": ((0,), (1,))}
    assert by_name["colocate_x2_mps"].assignments == {
        "engine": ((0,), (0,), (1,), (1,))
    }


def test_two_process_pipeline_covers_all_four_families():
    shapes = residency_shapes([ENGINE, TAIL], [0, 1, 2], gpu_mem_gib=80.0)
    got = names(shapes)
    assert "replicate_x3" in got
    assert "split_backbone1_tails2" in got
    assert "split_backbone2_tails1" in got
    assert "consolidate_backbone2_tails2_timeslice" in got
    assert "consolidate_backbone2_tails2_mps" in got
    assert not any(name.startswith("tp") for name in got)
    consolidated = next(
        s for s in shapes if s.name == "consolidate_backbone2_tails2_mps"
    )
    assert consolidated.assignments == {"engine": ((0,), (1,)), "tail": ((2,), (2,))}
    assert consolidated.sharing_mode == "mps"
    split = next(s for s in shapes if s.name == "split_backbone1_tails2")
    assert split.assignments == {"engine": ((0,),), "tail": ((1,), (2,))}
    assert split.sharing_mode == "dedicated"


def test_colocation_stops_at_the_memory_wall():
    big = ProcessShape("engine", footprint_gib=30.0, backbone=True)
    shapes = residency_shapes([big, TAIL], [0], gpu_mem_gib=80.0)
    assert names(shapes) == ["colocate_x2_mps", "colocate_x2_timeslice", "replicate_x1"]


def test_pipeline_too_large_for_one_gpu_replicates_pairs():
    big = ProcessShape("engine", footprint_gib=60.0, backbone=True)
    tail = ProcessShape("tail", footprint_gib=30.0)
    shapes = residency_shapes([big, tail], [0, 1, 2, 3], gpu_mem_gib=80.0)
    got = names(shapes)
    assert "replicate_pairs_x1" in got and "replicate_pairs_x2" in got
    assert not any(
        name.startswith("replicate_x") or name.startswith("colocate") for name in got
    )


def test_backbone_over_one_gpu_forces_tensor_parallel():
    huge = ProcessShape("engine", footprint_gib=120.0, backbone=True)
    shapes = residency_shapes([huge, TAIL], [0, 1, 2, 3, 4, 5], gpu_mem_gib=80.0)
    got = names(shapes)
    assert got == [
        "tp2_x1_tails1",
        "tp2_x1_tails2",
        "tp2_x1_tails3",
        "tp2_x1_tails4",
        "tp2_x2_tails1",
    ]
    two = next(s for s in shapes if s.name == "tp2_x2_tails1")
    assert two.assignments == {"engine": ((0, 1), (3, 4)), "tail": ((2,), (5,))}


def test_declared_tp_is_kept():
    tp_engine = ProcessShape("engine", footprint_gib=10.0, backbone=True, tp_size=2)
    shapes = residency_shapes([tp_engine, TAIL], [0, 1, 2], gpu_mem_gib=80.0)
    assert names(shapes) == ["tp2_x1_tails1"]


def test_rejections():
    with pytest.raises(ValueError):
        residency_shapes([TAIL], [0], gpu_mem_gib=80.0)
    with pytest.raises(ValueError):
        residency_shapes([ENGINE, TAIL], [0, 0], gpu_mem_gib=80.0)
    with pytest.raises(ValueError):
        residency_shapes(
            [ProcessShape("engine", footprint_gib=1000.0, backbone=True)],
            [0, 1],
            gpu_mem_gib=80.0,
        )


def test_fractions_fill_dedicated_gpus_and_split_shared_ones():
    processes = {"engine": ENGINE, "tail": TAIL}
    shapes = {
        s.name: s for s in residency_shapes([ENGINE, TAIL], [0, 1, 2], gpu_mem_gib=80.0)
    }
    split = instance_fractions(
        shapes["split_backbone1_tails2"], processes, gpu_mem_gib=80.0
    )
    assert split == {"engine": 0.9, "tail": 0.9}
    consolidated = instance_fractions(
        shapes["consolidate_backbone2_tails2_mps"], processes, gpu_mem_gib=80.0
    )
    assert consolidated == {"engine": 0.9, "tail": 0.45}
    colocated = instance_fractions(
        shapes["colocate_x2_mps"], processes, gpu_mem_gib=80.0
    )
    assert colocated["engine"] + colocated["tail"] == pytest.approx(0.45, abs=0.002)
