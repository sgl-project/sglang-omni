# SPDX-License-Identifier: Apache-2.0
"""Wiring tests: config field, plan attach, child-process affinity apply."""

import logging
import os

import pytest
from pydantic import ValidationError

from sglang_omni.config.schema import PlacementConfig
from sglang_omni.cpu_alloc.__main__ import format_blocks, plan_replica_blocks
from sglang_omni.cpu_alloc.allocator import CpuAllocationPlan, ProcessCpuAssignment
from sglang_omni.cpu_alloc.topology import discover_topology
from sglang_omni.pipeline.mp_runner import _attach_cpu_plan
from sglang_omni.pipeline.stage_workers import (
    StageGroup,
    StageLaunchConfig,
    StageWorkerProcessSpec,
    _apply_cpu_affinity,
    _patched_spawn_env,
    _seeded_spawn_affinity,
)


class TestPlacementConfigField:
    def test_default_off(self):
        assert PlacementConfig().cpu_allocator == "off"

    @pytest.mark.parametrize("mode", ["off", "static", "auto"])
    def test_valid_modes(self, mode):
        assert PlacementConfig(cpu_allocator=mode).cpu_allocator == mode

    @pytest.mark.parametrize("mode", ["dynamic", "on", ""])
    def test_invalid_mode_rejected(self, mode):
        with pytest.raises(ValidationError):
            PlacementConfig(cpu_allocator=mode)


def make_group(process_name):
    spec = StageWorkerProcessSpec(
        process_name=process_name,
        stage_specs=[StageLaunchConfig(stage_name="s")],
    )
    return StageGroup(process_name, [spec])


class TestAttachCpuPlan:
    def test_attaches_matching_assignments(self):
        groups = [make_group("pipeline"), make_group("vocoder")]
        plan = CpuAllocationPlan(
            assignments={"pipeline": ProcessCpuAssignment("pipeline", (0, 8), True, 0)},
            shared_pools={},
            events=(),
        )
        _attach_cpu_plan(groups, plan)
        assert groups[0].process_specs[0].cpu_ids == (0, 8)
        assert groups[1].process_specs[0].cpu_ids is None

    def test_none_plan_is_noop(self):
        groups = [make_group("pipeline")]
        _attach_cpu_plan(groups, None)
        assert groups[0].process_specs[0].cpu_ids is None


class TestApplyCpuAffinity:
    def _spec(self, cpu_ids):
        return StageWorkerProcessSpec(
            process_name="p",
            stage_specs=[StageLaunchConfig(stage_name="s")],
            cpu_ids=cpu_ids,
        )

    def test_none_does_not_touch_affinity(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            os, "sched_setaffinity", lambda *a: calls.append(a), raising=False
        )
        _apply_cpu_affinity(self._spec(None), logging.getLogger("t"))
        assert calls == []

    def test_pins_to_cpu_ids(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            os, "sched_setaffinity", lambda *a: calls.append(a), raising=False
        )
        _apply_cpu_affinity(self._spec((0, 8)), logging.getLogger("t"))
        assert calls == [(0, (0, 8))]

    def test_oserror_is_nonfatal(self, monkeypatch):
        def boom(*a):
            raise OSError("denied")

        monkeypatch.setattr(os, "sched_setaffinity", boom, raising=False)
        _apply_cpu_affinity(self._spec((0,)), logging.getLogger("t"))


class TestSeededSpawnAffinity:
    def test_seeds_and_restores(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False
        )
        monkeypatch.setattr(
            os,
            "sched_setaffinity",
            lambda pid, cpus: calls.append(set(cpus)),
            raising=False,
        )
        with _seeded_spawn_affinity((0, 8)):
            assert calls == [{0, 8}]
        assert calls == [{0, 8}, {0, 1, 2, 3}]

    def test_none_is_noop(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            os, "sched_setaffinity", lambda *a: calls.append(a), raising=False
        )
        with _seeded_spawn_affinity(None):
            pass
        assert calls == []

    def test_seed_failure_still_restores_nothing(self, monkeypatch):
        def boom(pid, cpus):
            raise OSError("denied")

        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)
        monkeypatch.setattr(os, "sched_setaffinity", boom, raising=False)
        with _seeded_spawn_affinity((0,)):
            pass


class TestOmpCapUnderCpuPlan:
    def _spec(self, cpu_ids, omp):
        stage = StageLaunchConfig(stage_name="s", env_defaults={"OMP_NUM_THREADS": omp})
        return StageWorkerProcessSpec(
            process_name="p", stage_specs=[stage], cpu_ids=cpu_ids
        )

    def test_caps_omp_default_to_plan_width(self, monkeypatch):
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        with _patched_spawn_env(self._spec((0, 8), "16")):
            assert os.environ["OMP_NUM_THREADS"] == "2"

    def test_narrower_declaration_kept(self, monkeypatch):
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        with _patched_spawn_env(self._spec((0, 1, 8, 9), "3")):
            assert os.environ["OMP_NUM_THREADS"] == "3"

    def test_no_plan_leaves_default(self, monkeypatch):
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        with _patched_spawn_env(self._spec(None, "16")):
            assert os.environ["OMP_NUM_THREADS"] == "16"

    def test_explicit_env_wins_over_cap(self, monkeypatch):
        monkeypatch.setenv("OMP_NUM_THREADS", "12")
        with _patched_spawn_env(self._spec((0, 8), "16")):
            assert os.environ["OMP_NUM_THREADS"] == "12"


class TestPlanReplicaBlocks:
    @pytest.fixture
    def topology(self, dual_node_sysfs):
        return discover_topology(range(16), sysfs_root=dual_node_sysfs)

    def test_blocks_are_whole_cores_on_gpu_node(self, topology):
        result = plan_replica_blocks(
            topology, replicas=3, gpu_numa_node=1, server_share=0.75
        )
        assert result["numa_node"] == 1
        assert result["server_blocks"] == [[4, 12], [5, 13], [6, 14]]
        assert result["client_cpus"] == [7, 15]

    def test_blocks_disjoint_and_cover_share(self, topology):
        result = plan_replica_blocks(
            topology, replicas=2, gpu_numa_node=0, server_share=0.75
        )
        flat = [cpu for block in result["server_blocks"] for cpu in block]
        assert len(flat) == len(set(flat))
        assert set(flat) | set(result["client_cpus"]) == {0, 1, 2, 3, 8, 9, 10, 11}

    def test_unresolved_node_uses_universe(self, topology):
        result = plan_replica_blocks(
            topology, replicas=2, gpu_numa_node=None, server_share=0.75
        )
        assert result["numa_node"] is None
        assert len(result["server_blocks"]) == 2

    def test_too_many_replicas_raises(self, topology):
        with pytest.raises(ValueError, match="physical cores"):
            plan_replica_blocks(
                topology, replicas=5, gpu_numa_node=1, server_share=0.75
            )

    @pytest.mark.parametrize("bad_share", [0.0, 1.0, -0.5])
    def test_bad_share_raises(self, topology, bad_share):
        with pytest.raises(ValueError, match="server_share"):
            plan_replica_blocks(
                topology, replicas=1, gpu_numa_node=0, server_share=bad_share
            )

    def test_format_blocks_is_shell_ready(self, topology):
        # autodp.sh consumes this value verbatim as CORE_BLOCKS.
        result = plan_replica_blocks(
            topology, replicas=3, gpu_numa_node=1, server_share=0.75
        )
        assert format_blocks(result) == "4,12 5,13 6,14"
