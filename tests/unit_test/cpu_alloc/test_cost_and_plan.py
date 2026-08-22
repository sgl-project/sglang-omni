# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from sglang_omni.config.placement import StagePlacement, StagePlacementPlan
from sglang_omni.config.topology import ProcessGroupPlacement, ProcessTopologyPlan
from sglang_omni.cpu_alloc.cost import StageCpuCost, resolve_stage_cpu_costs
from sglang_omni.cpu_alloc.pipeline_plan import (
    SERVING_PARENT_PROCESS,
    build_pipeline_cpu_plan,
)
from sglang_omni.cpu_alloc.topology import discover_topology


def make_config(stage_names, costs):
    class FakeConfig:
        stages = [SimpleNamespace(name=name) for name in stage_names]

        @classmethod
        def stage_cpu_costs(cls):
            return costs

    return FakeConfig()


class TestStageCpuCost:
    def test_valid_declarations(self):
        config = make_config(
            ["ar", "voc"],
            {
                "ar": {"host_class": "serial-loop", "exclusive_cores": 2},
                "voc": {"host_class": "gpu-bound"},
            },
        )
        costs = resolve_stage_cpu_costs(config)
        assert costs["ar"] == StageCpuCost("serial-loop", exclusive_cores=2)
        assert costs["voc"].host_class == "gpu-bound"

    def test_empty_default(self):
        assert resolve_stage_cpu_costs(make_config(["ar"], {})) == {}

    def test_unknown_stage_raises(self):
        config = make_config(["ar"], {"nope": {"host_class": "gpu-bound"}})
        with pytest.raises(ValueError, match="unknown stage 'nope'"):
            resolve_stage_cpu_costs(config)

    def test_bad_host_class_raises(self):
        config = make_config(["ar"], {"ar": {"host_class": "banana"}})
        with pytest.raises(ValueError, match="host_class"):
            resolve_stage_cpu_costs(config)

    def test_unknown_keys_raise(self):
        config = make_config(["ar"], {"ar": {"host_class": "gpu-bound", "width": 3}})
        with pytest.raises(ValueError, match="unknown keys"):
            resolve_stage_cpu_costs(config)

    def test_serial_loop_rejects_zero_cores(self):
        with pytest.raises(ValueError, match="exclusive_cores"):
            StageCpuCost("serial-loop", exclusive_cores=0)


class TestBuildPipelineCpuPlan:
    def _plans(self):
        placement_plan = StagePlacementPlan(
            stages={
                "tts_engine": StagePlacement("tts_engine", (0,), 1, 0.5),
                "vocoder": StagePlacement("vocoder", (0,), 1, 0.3),
            },
            gpus={},
        )
        process_plan = ProcessTopologyPlan(
            groups=(
                ProcessGroupPlacement("pipeline", ("tts_engine",), 0),
                ProcessGroupPlacement("vocoder", ("vocoder",), 0),
            ),
            stage_to_process={"tts_engine": "pipeline", "vocoder": "vocoder"},
            tp_stage_to_processes={},
        )
        return placement_plan, process_plan

    def test_plan_pins_serial_stage_and_shares_rest(self, dual_node_sysfs):
        topology = discover_topology(range(16), sysfs_root=dual_node_sysfs)
        config = make_config(
            ["tts_engine", "vocoder"],
            {"tts_engine": {"host_class": "serial-loop"}},
        )
        placement_plan, process_plan = self._plans()
        plan = build_pipeline_cpu_plan(
            config,
            placement_plan=placement_plan,
            process_plan=process_plan,
            topology=topology,
        )
        pipeline = plan.assignments["pipeline"]
        assert pipeline.exclusive and len(pipeline.cpu_ids) == 2
        vocoder = plan.assignments["vocoder"]
        assert not vocoder.exclusive
        assert not set(vocoder.cpu_ids) & set(pipeline.cpu_ids)

    def test_the_serving_parent_sits_on_the_granted_cores(self, dual_node_sysfs):
        # It is on every request path, so the shared pool is the wrong place
        # for it: sharing the pool with colocated load measured 32 vs 127 QPS.
        topology = discover_topology(range(16), sysfs_root=dual_node_sysfs)
        config = make_config(
            ["tts_engine", "vocoder"],
            {
                "tts_engine": {"host_class": "serial-loop", "exclusive_cores": 2},
                "vocoder": {"host_class": "gpu-bound"},
            },
        )
        placement_plan, process_plan = self._plans()
        plan = build_pipeline_cpu_plan(
            config,
            placement_plan=placement_plan,
            process_plan=process_plan,
            topology=topology,
        )
        parent = plan.assignments[SERVING_PARENT_PROCESS]
        granted = plan.assignments["pipeline"]
        assert granted.exclusive
        assert parent.cpu_ids == granted.cpu_ids
        assert not set(parent.cpu_ids) & set(plan.shared_pools[None])

    def test_a_lone_serial_stage_keeps_the_whole_node(self, dual_node_sysfs):
        # Nobody lives in the pool here, so holding cores back for it would
        # take them from the only two processes there are.
        topology = discover_topology(range(4), sysfs_root=dual_node_sysfs)
        config = make_config(
            ["tts_engine"],
            {"tts_engine": {"host_class": "serial-loop", "exclusive_cores": 2}},
        )
        placement_plan, process_plan = self._plans()
        process_plan = ProcessTopologyPlan(
            groups=(ProcessGroupPlacement("pipeline", ("tts_engine",), 0),),
            stage_to_process={"tts_engine": "pipeline"},
            tp_stage_to_processes={},
        )
        plan = build_pipeline_cpu_plan(
            config,
            placement_plan=placement_plan,
            process_plan=process_plan,
            topology=topology,
        )
        granted = plan.assignments["pipeline"]
        assert set(granted.cpu_ids) == set(topology.universe)
        assert plan.assignments[SERVING_PARENT_PROCESS].cpu_ids == granted.cpu_ids

    def test_no_declarations_returns_none(self, dual_node_sysfs):
        topology = discover_topology(range(16), sysfs_root=dual_node_sysfs)
        config = make_config(["tts_engine", "vocoder"], {})
        placement_plan, process_plan = self._plans()
        assert (
            build_pipeline_cpu_plan(
                config,
                placement_plan=placement_plan,
                process_plan=process_plan,
                topology=topology,
            )
            is None
        )

    def test_tp_ranks_get_disjoint_grants_spread_across_nodes(self, dual_node_sysfs):
        topology = discover_topology(range(16), sysfs_root=dual_node_sysfs)
        config = make_config(["thinker"], {"thinker": {"host_class": "serial-loop"}})
        placement_plan = StagePlacementPlan(
            stages={"thinker": StagePlacement("thinker", (0, 1), 2, 0.8)},
            gpus={},
        )
        process_plan = ProcessTopologyPlan(
            groups=(),
            stage_to_process={},
            tp_stage_to_processes={"thinker": ("thinker_tp0", "thinker_tp1")},
        )
        plan = build_pipeline_cpu_plan(
            config,
            placement_plan=placement_plan,
            process_plan=process_plan,
            topology=topology,
        )
        tp0 = plan.assignments["thinker_tp0"]
        tp1 = plan.assignments["thinker_tp1"]
        assert tp0.exclusive and tp1.exclusive
        assert not set(tp0.cpu_ids) & set(tp1.cpu_ids)
        assert {tp0.numa_node, tp1.numa_node} == {0, 1}

    @pytest.mark.parametrize("as_property", [True, False])
    def test_replicated_process_stays_in_shared_pool(
        self, dual_node_sysfs, as_property
    ):
        # One grant per logical group would be inherited by every replica and
        # stop being exclusive, so replicated groups keep today's behavior.
        topology = discover_topology(range(16), sysfs_root=dual_node_sysfs)
        config = make_config(
            ["tts_engine", "vocoder"],
            {"tts_engine": {"host_class": "serial-loop"}},
        )
        placement_plan, process_plan = self._plans()

        names = ("pipeline",)
        processes = SimpleNamespace(
            replicated_process_names=names if as_property else (lambda: names)
        )
        replicated_plan = SimpleNamespace(
            groups=process_plan.groups,
            stage_to_process=process_plan.stage_to_process,
            tp_stage_to_processes={},
            processes=processes,
        )
        plan = build_pipeline_cpu_plan(
            config,
            placement_plan=placement_plan,
            process_plan=replicated_plan,
            topology=topology,
        )
        assert not plan.assignments["pipeline"].exclusive
        assert plan.assignments["pipeline"].cpu_ids == plan.shared_pools[None]
