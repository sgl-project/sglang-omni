# SPDX-License-Identifier: Apache-2.0
"""Unit tests for process-level replicas."""

import pytest

from sglang_omni.config.placement import build_stage_placement_plan
from sglang_omni.config.schema import (
    PipelineConfig,
    PlacementConfig,
    ProcessConfig,
    StageConfig,
)
from sglang_omni.config.topology import compile_logical_processes
from sglang_omni.pipeline.replicas import (
    ReplicaTopology,
    RoundRobinBindingPolicy,
    assign_replica_bindings,
    expand_replica_stages,
    parse_replica_instance_name,
    replica_instance_name,
    validate_device_assignment,
)


def build_stage_config(name: str, **kwargs) -> StageConfig:
    defaults = dict(factory_path="pkg.mod.create", terminal=True, process=name)
    defaults.update(kwargs)
    return StageConfig(name=name, **defaults)


def build_pipeline_config(stages: list[StageConfig], **kwargs) -> PipelineConfig:
    kwargs.setdefault("model_path", "m")
    kwargs.setdefault(
        "placement", PlacementConfig(require_memory_fraction_for_colocation=False)
    )
    return PipelineConfig(stages=stages, **kwargs)


def expand(config: PipelineConfig):
    plan, stages = compile_logical_processes(config)
    expanded, topology = expand_replica_stages(stages, plan)
    return plan, expanded, topology


class TestInstanceNaming:
    def test_round_trip(self):
        name = replica_instance_name("talker_ar", 1)
        assert name == "talker_ar@r1"
        assert parse_replica_instance_name(name) == ("talker_ar", 1)

    def test_plain_name_passthrough(self):
        assert parse_replica_instance_name("thinker") == ("thinker", None)

    def test_non_numeric_suffix_is_not_replica(self):
        assert parse_replica_instance_name("stage@rx") == ("stage@rx", None)


class TestReplicaDevices:
    def test_gpu_process_replica_requires_devices(self):
        with pytest.raises(ValueError, match="requires replica_devices"):
            compile_logical_processes(
                build_pipeline_config(
                    [build_stage_config("s", process="p", gpu=1)],
                    processes={"p": ProcessConfig(num_replicas=2)},
                )
            )

    def test_cpu_process_must_not_declare_devices(self):
        with pytest.raises(ValueError, match="must not declare replica_devices"):
            compile_logical_processes(
                build_pipeline_config(
                    [build_stage_config("s", process="p")],
                    processes={
                        "p": ProcessConfig(num_replicas=2, replica_devices="0,1")
                    },
                )
            )

    def test_cpu_process_replicates_without_devices(self):
        _, expanded, topology = expand(
            build_pipeline_config(
                [build_stage_config("s", process="p")],
                processes={"p": ProcessConfig(num_replicas=2)},
            )
        )
        assert [stage.gpu for stage in expanded] == [None, None]
        assert topology.to_dict() == {"s": ["s@r0", "s@r1"]}

    def test_device_count_must_match_replicas_times_tp(self):
        with pytest.raises(ValueError, match="expected 4"):
            compile_logical_processes(
                build_pipeline_config(
                    [
                        build_stage_config(
                            "thinker", tp_size=2, gpu=[0, 1], process=None
                        )
                    ],
                    processes={
                        "thinker": ProcessConfig(
                            num_replicas=2, replica_devices="0,1,2"
                        )
                    },
                )
            )

    def test_replicas_may_share_one_device(self):
        _, expanded, _ = expand(
            build_pipeline_config(
                [build_stage_config("s", process="p", gpu=0)],
                processes={"p": ProcessConfig(num_replicas=2, replica_devices=[0, 0])},
            )
        )
        assert [stage.gpu for stage in expanded] == [0, 0]

    def test_tp_replica_group_requires_unique_devices(self):
        with pytest.raises(ValueError, match="unique GPU ids"):
            compile_logical_processes(
                build_pipeline_config(
                    [
                        build_stage_config(
                            "thinker", tp_size=2, gpu=[0, 1], process=None
                        )
                    ],
                    processes={
                        "thinker": ProcessConfig(
                            num_replicas=2, replica_devices=[0, 0, 2, 3]
                        )
                    },
                )
            )


class TestProcessExpansion:
    def test_no_replicas_is_identity(self):
        config = build_pipeline_config(
            [build_stage_config("a"), build_stage_config("b")]
        )
        _, expanded, topology = expand(config)

        assert [stage.name for stage in expanded] == ["a", "b"]
        assert not topology
        assert topology.to_dict() == {}

    def test_whole_process_is_copied_with_one_index(self):
        config = build_pipeline_config(
            [
                build_stage_config(
                    "decode", terminal=False, next="postprocess", process="tail"
                ),
                build_stage_config("postprocess", process="tail"),
            ],
            processes={"tail": ProcessConfig(num_replicas=2)},
        )
        _, expanded, topology = expand(config)

        by_name = {stage.name: stage for stage in expanded}
        assert set(by_name) == {
            "decode@r0",
            "decode@r1",
            "postprocess@r0",
            "postprocess@r1",
        }
        assert by_name["decode@r0"].process == "tail@r0"
        assert by_name["postprocess@r0"].process == "tail@r0"
        assert by_name["decode@r1"].process == "tail@r1"
        assert by_name["postprocess@r1"].process == "tail@r1"
        assert topology.to_dict() == {
            "decode": ["decode@r0", "decode@r1"],
            "postprocess": ["postprocess@r0", "postprocess@r1"],
        }

    def test_expansion_keeps_logical_wiring_and_assigns_devices(self):
        config = build_pipeline_config(
            [
                build_stage_config(
                    "talker_ar",
                    terminal=False,
                    next="code2wav",
                    stream_to=["code2wav"],
                    gpu=1,
                    process="talker_ar",
                ),
                build_stage_config("code2wav", process="code2wav"),
            ],
            processes={
                "talker_ar": ProcessConfig(num_replicas=2, replica_devices="1,2")
            },
        )
        _, expanded, topology = expand(config)

        names = [stage.name for stage in expanded]
        assert names == ["talker_ar@r0", "talker_ar@r1", "code2wav"]
        r0, r1 = expanded[0], expanded[1]
        assert (r0.gpu, r1.gpu) == (1, 2)
        assert r0.next == "code2wav" and r0.stream_to == ["code2wav"]
        assert topology.to_dict() == {"talker_ar": ["talker_ar@r0", "talker_ar@r1"]}

    def test_cpu_stage_in_mixed_process_stays_on_host(self):
        config = build_pipeline_config(
            [
                build_stage_config(
                    "normalize", terminal=False, next="encode", process="front"
                ),
                build_stage_config("encode", process="front", gpu=0),
            ],
            processes={"front": ProcessConfig(num_replicas=2, replica_devices=[4, 5])},
        )
        _, expanded, _ = expand(config)

        by_name = {stage.name: stage for stage in expanded}
        assert by_name["normalize@r0"].gpu is None
        assert by_name["normalize@r1"].gpu is None
        assert by_name["encode@r0"].gpu == 4
        assert by_name["encode@r1"].gpu == 5

    def test_tp_process_expands_by_whole_rank_group(self):
        config = build_pipeline_config(
            [build_stage_config("thinker", tp_size=2, gpu=[0, 1], process=None)],
            processes={
                "thinker": ProcessConfig(num_replicas=2, replica_devices=[0, 1, 2, 3])
            },
        )
        _, expanded, topology = expand(config)

        assert [stage.name for stage in expanded] == ["thinker@r0", "thinker@r1"]
        assert [stage.gpu for stage in expanded] == [[0, 1], [2, 3]]
        assert [stage.process for stage in expanded] == ["thinker@r0", "thinker@r1"]
        assert topology.to_dict() == {"thinker": ["thinker@r0", "thinker@r1"]}

    def test_config_order_is_preserved_inside_a_replica(self):
        config = build_pipeline_config(
            [
                build_stage_config("a", terminal=False, next="b", process="p"),
                build_stage_config("b", terminal=False, next="c", process="p"),
                build_stage_config("c", process="p"),
            ],
            processes={"p": ProcessConfig(num_replicas=2)},
        )
        _, expanded, _ = expand(config)

        assert [stage.name for stage in expanded] == [
            "a@r0",
            "a@r1",
            "b@r0",
            "b@r1",
            "c@r0",
            "c@r1",
        ]
        for replica_id in (0, 1):
            in_process = [
                stage.name for stage in expanded if stage.process == f"p@r{replica_id}"
            ]
            assert in_process == [
                f"a@r{replica_id}",
                f"b@r{replica_id}",
                f"c@r{replica_id}",
            ]


class TestValidateDeviceAssignment:
    def test_valid_ids_pass(self):
        _, expanded, _ = expand(
            build_pipeline_config(
                [build_stage_config("s", gpu=1, process="p")],
                processes={"p": ProcessConfig(num_replicas=2, replica_devices="1,2")},
            )
        )
        validate_device_assignment(expanded, device_count=4)

    def test_out_of_range_id_raises(self):
        _, expanded, _ = expand(
            build_pipeline_config(
                [build_stage_config("s", gpu=3, process="p")],
                processes={"p": ProcessConfig(num_replicas=2, replica_devices="3,4")},
            )
        )
        with pytest.raises(ValueError, match="GPU id 4"):
            validate_device_assignment(expanded, device_count=4)

    def test_cpu_stages_are_skipped(self):
        validate_device_assignment([build_stage_config("s")], device_count=0)

    def test_unknown_device_count_skips_range_check(self):
        validate_device_assignment([build_stage_config("s", gpu=7)], device_count=None)


class TestReplicaTopology:
    def topo(self) -> ReplicaTopology:
        config = build_pipeline_config(
            [
                build_stage_config("talker_ar", gpu=1, process="talker_ar"),
                build_stage_config("code2wav", gpu=1, process="code2wav"),
                build_stage_config("thinker", process="thinker"),
            ],
            processes={
                "talker_ar": ProcessConfig(num_replicas=2, replica_devices="1,2"),
                "code2wav": ProcessConfig(num_replicas=2, replica_devices="1,2"),
            },
        )
        _, _, topology = expand(config)
        return topology

    def test_resolve_and_logical_name(self):
        topo = self.topo()
        assert topo.resolve("talker_ar", 1) == "talker_ar@r1"
        assert topo.logical_name("talker_ar@r1") == "talker_ar"
        assert topo.logical_name("thinker") == "thinker"

    def test_resolve_out_of_range(self):
        with pytest.raises(ValueError, match="has 2 replicas"):
            self.topo().resolve("talker_ar", 5)

    def test_resolve_unreplicated(self):
        topo = self.topo()
        assert topo.resolve("thinker", 0) == "thinker"
        with pytest.raises(ValueError, match="not replicated"):
            topo.resolve("thinker", 1)

    def test_instances(self):
        topo = self.topo()
        assert topo.instances("code2wav") == ("code2wav@r0", "code2wav@r1")
        assert topo.instances("thinker") == ("thinker",)

    def test_unregistered_suffix_name_is_not_normalized(self):
        assert self.topo().logical_name("other@r0") == "other@r0"

    def test_dict_round_trip(self):
        topo = self.topo()
        restored = ReplicaTopology.from_dict(topo.to_dict())
        assert restored == topo
        assert not ReplicaTopology.from_dict(None)


class TestBinding:
    def plan(self, **processes):
        config = build_pipeline_config(
            [
                build_stage_config(
                    "decode", terminal=False, next="postprocess", process="tail"
                ),
                build_stage_config("postprocess", process="tail"),
                build_stage_config("thinker", process="thinker"),
            ],
            processes=processes,
        )
        plan, _ = compile_logical_processes(config)
        return plan

    def test_round_robin_cycles_per_process(self):
        policy = RoundRobinBindingPolicy()
        picks = [policy.bind("tail", 2, f"req{i}") for i in range(4)]
        assert picks == [0, 1, 0, 1]
        assert policy.bind("thinker", 3, "reqx") == 0

    def test_one_choice_projects_onto_every_member_stage(self):
        plan = self.plan(tail=ProcessConfig(num_replicas=2))
        policy = RoundRobinBindingPolicy()

        first = assign_replica_bindings(plan, policy, "req0")
        second = assign_replica_bindings(plan, policy, "req1")

        assert first == {"decode": 0, "postprocess": 0}
        assert second == {"decode": 1, "postprocess": 1}

    def test_processes_choose_independently(self):
        plan = self.plan(
            tail=ProcessConfig(num_replicas=3),
            thinker=ProcessConfig(num_replicas=2),
        )
        policy = RoundRobinBindingPolicy()
        bindings = [assign_replica_bindings(plan, policy, f"req{i}") for i in range(6)]

        assert [b["decode"] for b in bindings] == [0, 1, 2, 0, 1, 2]
        assert [b["postprocess"] for b in bindings] == [0, 1, 2, 0, 1, 2]
        assert [b["thinker"] for b in bindings] == [0, 1, 0, 1, 0, 1]

    def test_equal_count_stream_processes_advance_in_lockstep(self):
        config = build_pipeline_config(
            [
                build_stage_config(
                    "talker_ar",
                    terminal=False,
                    next="code2wav",
                    stream_to=["code2wav"],
                    process="talker",
                ),
                build_stage_config("code2wav", process="codec"),
            ],
            processes={
                "talker": ProcessConfig(num_replicas=2),
                "codec": ProcessConfig(num_replicas=2),
            },
        )
        plan, _ = compile_logical_processes(config)
        policy = RoundRobinBindingPolicy()

        bindings = [assign_replica_bindings(plan, policy, f"req{i}") for i in range(6)]

        assert [
            (binding["talker_ar"], binding["code2wav"]) for binding in bindings
        ] == [(0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (1, 1)]

    def test_unreplicated_plan_binds_none(self):
        assert (
            assign_replica_bindings(self.plan(), RoundRobinBindingPolicy(), "r") is None
        )

    def test_out_of_range_policy_choice_is_rejected(self):
        class BadPolicy:
            def bind(self, process_name, num_replicas, request_id):
                return num_replicas

        plan = self.plan(tail=ProcessConfig(num_replicas=2))
        with pytest.raises(ValueError, match="selected replica 2"):
            assign_replica_bindings(plan, BadPolicy(), "req")


class TestEntryProcessReplicas:
    def test_entry_process_can_be_replicated(self):
        config = build_pipeline_config(
            [
                build_stage_config(
                    "normalize", terminal=False, next="sink", process="front"
                ),
                build_stage_config("sink", process="sink"),
            ],
            processes={"front": ProcessConfig(num_replicas=2)},
        )
        plan, expanded, topology = expand(config)

        assert config.resolved_entry_stage == "normalize"
        assert topology.instances("normalize") == ("normalize@r0", "normalize@r1")
        assert assign_replica_bindings(plan, RoundRobinBindingPolicy(), "req") == {
            "normalize": 0
        }


class TestColocatedReplicaRejection:
    def test_colocated_rejects_replicated_process(self):
        from sglang_omni.models.qwen3_omni.config import (
            Qwen3OmniSpeechColocatedPipelineConfig,
        )

        config_data = Qwen3OmniSpeechColocatedPipelineConfig(
            model_path="m"
        ).model_dump()
        config_data["processes"] = {
            "talker_ar": {
                "num_replicas": 2,
                "replica_devices": [0, 0],
            }
        }
        config = Qwen3OmniSpeechColocatedPipelineConfig(**config_data)

        with pytest.raises(ValueError, match="does not support process replicas"):
            build_placement(config)


def build_placement(config: PipelineConfig):
    _, expanded, topology = expand(config)
    return build_stage_placement_plan(
        config,
        stages_cfg=expanded,
        replica_instances=topology.replicas,
    )


def qwen_speech_replica_config(talker_devices: list[int]) -> PipelineConfig:
    from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig

    config_data = Qwen3OmniSpeechPipelineConfig(model_path="m").model_dump()
    thinker = next(
        stage for stage in config_data["stages"] if stage["name"] == "thinker"
    )
    thinker["gpu"] = [0, 1]
    thinker["tp_size"] = 2
    config_data["processes"] = {
        "talker_ar": {
            "num_replicas": 2,
            "replica_devices": talker_devices,
        }
    }
    return Qwen3OmniSpeechPipelineConfig(**config_data)


class TestQwenReplicaPlacementPolicy:
    def test_accepts_single_talker_overlapping_thinker_tp_rank(self):
        from sglang_omni.config.placement import StagePlacement, StagePlacementPlan
        from sglang_omni.models.qwen3_omni.placement import Qwen3OmniPlacementPolicy

        config = build_pipeline_config(
            [
                build_stage_config(name)
                for name in (
                    "preprocessing",
                    "image_encoder",
                    "audio_encoder",
                    "thinker",
                    "decode",
                    "talker_ar",
                    "code2wav",
                )
            ]
        )
        plan = StagePlacementPlan(
            stages={
                "thinker": StagePlacement("thinker", (0, 1), 2, None),
                "talker_ar": StagePlacement("talker_ar", (1,), 1, None),
            },
            gpus={},
        )

        Qwen3OmniPlacementPolicy().validate(config, plan)

    def test_rejects_talker_replica_overlapping_thinker_tp_rank(self):
        config = qwen_speech_replica_config([1, 2])

        with pytest.raises(ValueError, match="talker_ar@r0"):
            build_placement(config)

    def test_accepts_talker_replicas_disjoint_from_thinker_tp(self):
        config = qwen_speech_replica_config([2, 3])

        plan = build_placement(config)

        assert [
            (placement.stage_name, placement.gpu_ids)
            for placement in plan.instances_of("talker_ar")
        ] == [
            ("talker_ar@r0", (2,)),
            ("talker_ar@r1", (3,)),
        ]
        assert [
            (placement.stage_name, placement.gpu_ids)
            for placement in plan.instances_of("thinker")
        ] == [("thinker", (0, 1))]


class TestRemovedStageLevelReplicaConfig:
    def test_stage_num_replicas_is_rejected(self):
        with pytest.raises(ValueError, match="num_replicas") as exc_info:
            build_stage_config("s", num_replicas=2)
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_stage_replica_devices_is_rejected(self):
        with pytest.raises(ValueError, match="replica_devices") as exc_info:
            build_stage_config("s", replica_devices="0,1")
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_fused_stages_is_rejected(self):
        with pytest.raises(ValueError, match="fused_stages") as exc_info:
            PipelineConfig(
                model_path="m",
                stages=[
                    build_stage_config("a", terminal=False, next="b", process="p"),
                    build_stage_config("b", process="p"),
                ],
                fused_stages=[["a", "b"]],
            )
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_stage_overrides_reject_replica_keys(self):
        from sglang_omni.config.sources import patches_from_stages_mapping

        config = build_pipeline_config([build_stage_config("code2wav")])
        with pytest.raises(Exception, match="num_replicas"):
            patches_from_stages_mapping(
                {"code2wav": {"num_replicas": 3}},
                type(config),
                [stage.name for stage in config.stages],
                origin="test",
            )


class TestReservedStageNames:
    def test_reserved_instance_suffix_rejected(self):
        with pytest.raises(ValueError, match="reserved"):
            PipelineConfig(model_path="m", stages=[build_stage_config("foo@r0")])

    def test_non_numeric_suffix_allowed(self):
        PipelineConfig(model_path="m", stages=[build_stage_config("foo@rx")])


class TestRuntimeOverridesOnReplicas:
    def build_pipeline_config(self) -> PipelineConfig:
        from sglang_omni.config import FactoryArgs

        return build_pipeline_config(
            [
                build_stage_config("src", terminal=False, next="gen", process="src"),
                build_stage_config(
                    "gen",
                    gpu=1,
                    process="gen",
                    factory=FactoryArgs(max_seq_len=4096),
                ),
            ],
            processes={"gen": ProcessConfig(num_replicas=2, replica_devices="1,2")},
        )

    def test_replica_instances_inherit_logical_overrides(self):
        from sglang_omni.config.runtime import resolve_stage_typed_kwargs

        config = self.build_pipeline_config()
        _, expanded, topology = expand(config)
        assert topology.instances("gen") == ("gen@r0", "gen@r1")

        for stage_cfg in expanded:
            if stage_cfg.name.startswith("gen@r"):
                args = resolve_stage_typed_kwargs(stage_cfg)
                assert (
                    args.get("max_seq_len") == 4096
                ), f"{stage_cfg.name} lost the value configured for 'gen'"

    def test_unreplicated_stage_does_not_borrow_overrides(self):
        from sglang_omni.config.runtime import resolve_stage_typed_kwargs

        config = self.build_pipeline_config()
        src = {s.name: s for s in config.stages}["src"]
        assert "max_seq_len" not in resolve_stage_typed_kwargs(src)


class TestReceiveSideLogicalNames:
    """A replica sends its instance name; fan-in and streams expect logical ones."""

    def payload(self, request_id: str = "req"):
        from sglang_omni.proto import OmniRequest, StagePayload

        return StagePayload(
            request_id=request_id, request=OmniRequest(inputs="x"), data={}
        )

    def build_stage_config(self, handler, **kwargs):
        from tests.unit_test.pipeline.helpers import make_stage

        return make_stage(name="aggregate", input_handler=handler, **kwargs)

    def test_fan_in_accepts_a_replicated_upstream(self):
        import asyncio

        from sglang_omni.pipeline.stage.input import AggregatedInput

        merged: list[list[str]] = []

        def merge(inputs):
            merged.append(sorted(inputs))
            return self.payload()

        stage = self.build_stage_config(
            AggregatedInput(sources={"a", "b"}, merge=merge),
            replica_topology={"a": ["a@r0", "a@r1"]},
        )

        async def run() -> None:
            await stage.receive_local_payload("req", "a@r0", self.payload())
            await stage.receive_local_payload("req", "b", self.payload())

        asyncio.run(run())

        assert merged == [["a", "b"]]

    def test_wait_for_fn_sees_the_logical_source(self):
        import asyncio

        from sglang_omni.pipeline.stage.input import AggregatedInput

        seen: list[str] = []

        def wait_for_fn(request_id, from_stage, data):
            seen.append(from_stage)
            return ["a", "b"]

        stage = self.build_stage_config(
            AggregatedInput(
                sources={"a", "b"},
                merge=lambda inputs: self.payload(),
                expected_sources_fn=wait_for_fn,
            ),
            replica_topology={"a": ["a@r0", "a@r1"]},
        )

        asyncio.run(stage.receive_local_payload("req", "a@r1", self.payload()))

        assert seen == ["a"]

    def test_unregistered_replica_suffix_passes_through(self):
        import asyncio

        from sglang_omni.pipeline.stage.input import AggregatedInput

        merged: list[list[str]] = []

        def merge(inputs):
            merged.append(sorted(inputs))
            return self.payload()

        stage = self.build_stage_config(
            AggregatedInput(sources={"other@r0"}, merge=merge),
            replica_topology={"a": ["a@r0"]},
        )

        asyncio.run(stage.receive_local_payload("req", "other@r0", self.payload()))

        assert merged == [["other@r0"]]

    def test_stream_chunks_reach_the_scheduler_with_logical_source(self):
        import asyncio

        from sglang_omni.pipeline.stage.stream_queue import StreamQueue
        from tests.unit_test.pipeline.helpers import make_stage

        stage = make_stage(
            name="vocoder",
            can_accept_stream_before_payload=True,
            replica_topology={"engine": ["engine@r0", "engine@r1"]},
        )
        stage.stream_queue = StreamQueue()

        async def run() -> None:
            await stage.receive_local_stream_chunk(
                "req", "engine@r1", chunk_id=0, data={"pcm": 1}
            )

        asyncio.run(run())

        message = stage.scheduler.inbox.get_nowait()
        assert message.type == "stream_chunk"
        assert message.data.from_stage == "engine"


class TestSingleReplicaDeviceOverride:
    def test_replica_devices_override_gpu_without_replicating(self):
        config = build_pipeline_config(
            [build_stage_config("s", process="p", gpu=1)],
            processes={"p": ProcessConfig(num_replicas=1, replica_devices=[7])},
        )
        _, expanded, topology = expand(config)

        assert [(stage.name, stage.gpu, stage.process) for stage in expanded] == [
            ("s", 7, "p")
        ]
        assert topology.to_dict() == {}

    def test_tp_process_single_replica_device_override(self):
        config = build_pipeline_config(
            [build_stage_config("thinker", tp_size=2, gpu=[0, 1], process=None)],
            processes={
                "thinker": ProcessConfig(num_replicas=1, replica_devices=[4, 5])
            },
        )
        _, expanded, _ = expand(config)

        assert [(stage.name, stage.gpu) for stage in expanded] == [("thinker", [4, 5])]
