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


def _stage(name: str, **kwargs) -> StageConfig:
    defaults = dict(factory_path="pkg.mod.create", terminal=True, process=name)
    defaults.update(kwargs)
    return StageConfig(name=name, **defaults)


def _config(stages: list[StageConfig], **kwargs) -> PipelineConfig:
    kwargs.setdefault("model_path", "m")
    kwargs.setdefault(
        "placement", PlacementConfig(require_memory_fraction_for_colocation=False)
    )
    return PipelineConfig(stages=stages, **kwargs)


def _expand(config: PipelineConfig):
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
                _config(
                    [_stage("s", process="p", gpu=1)],
                    processes={"p": ProcessConfig(num_replicas=2)},
                )
            )

    def test_cpu_process_must_not_declare_devices(self):
        with pytest.raises(ValueError, match="must not declare replica_devices"):
            compile_logical_processes(
                _config(
                    [_stage("s", process="p")],
                    processes={
                        "p": ProcessConfig(num_replicas=2, replica_devices="0,1")
                    },
                )
            )

    def test_cpu_process_replicates_without_devices(self):
        _, expanded, topology = _expand(
            _config(
                [_stage("s", process="p")],
                processes={"p": ProcessConfig(num_replicas=2)},
            )
        )
        assert [stage.gpu for stage in expanded] == [None, None]
        assert topology.to_dict() == {"s": ["s@r0", "s@r1"]}

    def test_device_count_must_match_replicas_times_tp(self):
        with pytest.raises(ValueError, match="expected 4"):
            compile_logical_processes(
                _config(
                    [_stage("thinker", tp_size=2, gpu=[0, 1], process=None)],
                    processes={
                        "thinker": ProcessConfig(
                            num_replicas=2, replica_devices="0,1,2"
                        )
                    },
                )
            )

    def test_replicas_may_share_one_device(self):
        _, expanded, _ = _expand(
            _config(
                [_stage("s", process="p", gpu=0)],
                processes={"p": ProcessConfig(num_replicas=2, replica_devices=[0, 0])},
            )
        )
        assert [stage.gpu for stage in expanded] == [0, 0]

    def test_tp_replica_group_requires_unique_devices(self):
        with pytest.raises(ValueError, match="unique GPU ids"):
            compile_logical_processes(
                _config(
                    [_stage("thinker", tp_size=2, gpu=[0, 1], process=None)],
                    processes={
                        "thinker": ProcessConfig(
                            num_replicas=2, replica_devices=[0, 0, 2, 3]
                        )
                    },
                )
            )


class TestProcessExpansion:
    def test_no_replicas_is_identity(self):
        config = _config([_stage("a"), _stage("b")])
        _, expanded, topology = _expand(config)

        assert [stage.name for stage in expanded] == ["a", "b"]
        assert not topology
        assert topology.to_dict() == {}

    def test_whole_process_is_copied_with_one_index(self):
        config = _config(
            [
                _stage("decode", terminal=False, next="postprocess", process="tail"),
                _stage("postprocess", process="tail"),
            ],
            processes={"tail": ProcessConfig(num_replicas=2)},
        )
        _, expanded, topology = _expand(config)

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
        config = _config(
            [
                _stage(
                    "talker_ar",
                    terminal=False,
                    next="code2wav",
                    stream_to=["code2wav"],
                    gpu=1,
                    process="talker_ar",
                ),
                _stage("code2wav", process="code2wav"),
            ],
            processes={
                "talker_ar": ProcessConfig(num_replicas=2, replica_devices="1,2")
            },
        )
        _, expanded, topology = _expand(config)

        names = [stage.name for stage in expanded]
        assert names == ["talker_ar@r0", "talker_ar@r1", "code2wav"]
        r0, r1 = expanded[0], expanded[1]
        assert (r0.gpu, r1.gpu) == (1, 2)
        assert r0.next == "code2wav" and r0.stream_to == ["code2wav"]
        assert topology.to_dict() == {"talker_ar": ["talker_ar@r0", "talker_ar@r1"]}

    def test_cpu_stage_in_mixed_process_stays_on_host(self):
        config = _config(
            [
                _stage("normalize", terminal=False, next="encode", process="front"),
                _stage("encode", process="front", gpu=0),
            ],
            processes={"front": ProcessConfig(num_replicas=2, replica_devices=[4, 5])},
        )
        _, expanded, _ = _expand(config)

        by_name = {stage.name: stage for stage in expanded}
        assert by_name["normalize@r0"].gpu is None
        assert by_name["normalize@r1"].gpu is None
        assert by_name["encode@r0"].gpu == 4
        assert by_name["encode@r1"].gpu == 5

    def test_tp_process_expands_by_whole_rank_group(self):
        config = _config(
            [_stage("thinker", tp_size=2, gpu=[0, 1], process=None)],
            processes={
                "thinker": ProcessConfig(num_replicas=2, replica_devices=[0, 1, 2, 3])
            },
        )
        _, expanded, topology = _expand(config)

        assert [stage.name for stage in expanded] == ["thinker@r0", "thinker@r1"]
        assert [stage.gpu for stage in expanded] == [[0, 1], [2, 3]]
        assert [stage.process for stage in expanded] == ["thinker@r0", "thinker@r1"]
        assert topology.to_dict() == {"thinker": ["thinker@r0", "thinker@r1"]}

    def test_config_order_is_preserved_inside_a_replica(self):
        config = _config(
            [
                _stage("a", terminal=False, next="b", process="p"),
                _stage("b", terminal=False, next="c", process="p"),
                _stage("c", process="p"),
            ],
            processes={"p": ProcessConfig(num_replicas=2)},
        )
        _, expanded, _ = _expand(config)

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
        _, expanded, _ = _expand(
            _config(
                [_stage("s", gpu=1, process="p")],
                processes={"p": ProcessConfig(num_replicas=2, replica_devices="1,2")},
            )
        )
        validate_device_assignment(expanded, device_count=4)

    def test_out_of_range_id_raises(self):
        _, expanded, _ = _expand(
            _config(
                [_stage("s", gpu=3, process="p")],
                processes={"p": ProcessConfig(num_replicas=2, replica_devices="3,4")},
            )
        )
        with pytest.raises(ValueError, match="GPU id 4"):
            validate_device_assignment(expanded, device_count=4)

    def test_cpu_stages_are_skipped(self):
        validate_device_assignment([_stage("s")], device_count=0)

    def test_unknown_device_count_skips_range_check(self):
        validate_device_assignment([_stage("s", gpu=7)], device_count=None)


class TestReplicaTopology:
    def _topo(self) -> ReplicaTopology:
        config = _config(
            [
                _stage("talker_ar", gpu=1, process="talker_ar"),
                _stage("code2wav", gpu=1, process="code2wav"),
                _stage("thinker", process="thinker"),
            ],
            processes={
                "talker_ar": ProcessConfig(num_replicas=2, replica_devices="1,2"),
                "code2wav": ProcessConfig(num_replicas=2, replica_devices="1,2"),
            },
        )
        _, _, topology = _expand(config)
        return topology

    def test_resolve_and_logical_name(self):
        topo = self._topo()
        assert topo.resolve("talker_ar", 1) == "talker_ar@r1"
        assert topo.logical_name("talker_ar@r1") == "talker_ar"
        assert topo.logical_name("thinker") == "thinker"

    def test_resolve_out_of_range(self):
        with pytest.raises(ValueError, match="has 2 replicas"):
            self._topo().resolve("talker_ar", 5)

    def test_resolve_unreplicated(self):
        topo = self._topo()
        assert topo.resolve("thinker", 0) == "thinker"
        with pytest.raises(ValueError, match="not replicated"):
            topo.resolve("thinker", 1)

    def test_instances(self):
        topo = self._topo()
        assert topo.instances("code2wav") == ("code2wav@r0", "code2wav@r1")
        assert topo.instances("thinker") == ("thinker",)

    def test_unregistered_suffix_name_is_not_normalized(self):
        assert self._topo().logical_name("other@r0") == "other@r0"

    def test_dict_round_trip(self):
        topo = self._topo()
        restored = ReplicaTopology.from_dict(topo.to_dict())
        assert restored == topo
        assert not ReplicaTopology.from_dict(None)


class TestBinding:
    def _plan(self, **processes):
        config = _config(
            [
                _stage("decode", terminal=False, next="postprocess", process="tail"),
                _stage("postprocess", process="tail"),
                _stage("thinker", process="thinker"),
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
        plan = self._plan(tail=ProcessConfig(num_replicas=2))
        policy = RoundRobinBindingPolicy()

        first = assign_replica_bindings(plan, policy, "req0")
        second = assign_replica_bindings(plan, policy, "req1")

        assert first == {"decode": 0, "postprocess": 0}
        assert second == {"decode": 1, "postprocess": 1}

    def test_processes_choose_independently(self):
        plan = self._plan(
            tail=ProcessConfig(num_replicas=3),
            thinker=ProcessConfig(num_replicas=2),
        )
        policy = RoundRobinBindingPolicy()
        bindings = [assign_replica_bindings(plan, policy, f"req{i}") for i in range(6)]

        assert [b["decode"] for b in bindings] == [0, 1, 2, 0, 1, 2]
        assert [b["postprocess"] for b in bindings] == [0, 1, 2, 0, 1, 2]
        assert [b["thinker"] for b in bindings] == [0, 1, 0, 1, 0, 1]

    def test_equal_count_stream_processes_advance_in_lockstep(self):
        config = _config(
            [
                _stage(
                    "talker_ar",
                    terminal=False,
                    next="code2wav",
                    stream_to=["code2wav"],
                    process="talker",
                ),
                _stage("code2wav", process="codec"),
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
            assign_replica_bindings(self._plan(), RoundRobinBindingPolicy(), "r")
            is None
        )

    def test_out_of_range_policy_choice_is_rejected(self):
        class BadPolicy:
            def bind(self, process_name, num_replicas, request_id):
                return num_replicas

        plan = self._plan(tail=ProcessConfig(num_replicas=2))
        with pytest.raises(ValueError, match="selected replica 2"):
            assign_replica_bindings(plan, BadPolicy(), "req")


class TestEntryProcessReplicas:
    def test_entry_process_can_be_replicated(self):
        config = _config(
            [
                _stage("normalize", terminal=False, next="sink", process="front"),
                _stage("sink", process="sink"),
            ],
            processes={"front": ProcessConfig(num_replicas=2)},
        )
        plan, expanded, topology = _expand(config)

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
            _build_placement(config)


def _build_placement(config: PipelineConfig):
    _, expanded, topology = _expand(config)
    return build_stage_placement_plan(
        config,
        stages_cfg=expanded,
        replica_instances=topology.replicas,
    )


def _qwen_speech_replica_config(talker_devices: list[int]) -> PipelineConfig:
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

        config = _config(
            [
                _stage(name)
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
        config = _qwen_speech_replica_config([1, 2])

        with pytest.raises(ValueError, match="talker_ar@r0"):
            _build_placement(config)

    def test_accepts_talker_replicas_disjoint_from_thinker_tp(self):
        config = _qwen_speech_replica_config([2, 3])

        plan = _build_placement(config)

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
            _stage("s", num_replicas=2)
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_stage_replica_devices_is_rejected(self):
        with pytest.raises(ValueError, match="replica_devices") as exc_info:
            _stage("s", replica_devices="0,1")
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_fused_stages_is_rejected(self):
        with pytest.raises(ValueError, match="fused_stages") as exc_info:
            PipelineConfig(
                model_path="m",
                stages=[
                    _stage("a", terminal=False, next="b", process="p"),
                    _stage("b", process="p"),
                ],
                fused_stages=[["a", "b"]],
            )
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_stage_overrides_reject_replica_keys(self):
        from sglang_omni.config.sources import patches_from_stages_mapping

        config = _config([_stage("code2wav")])
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
            PipelineConfig(model_path="m", stages=[_stage("foo@r0")])

    def test_non_numeric_suffix_allowed(self):
        PipelineConfig(model_path="m", stages=[_stage("foo@rx")])


class TestRuntimeOverridesOnReplicas:
    def _config(self) -> PipelineConfig:
        from sglang_omni.config import FactoryArgs

        return _config(
            [
                _stage("src", terminal=False, next="gen", process="src"),
                _stage(
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

        config = self._config()
        _, expanded, topology = _expand(config)
        assert topology.instances("gen") == ("gen@r0", "gen@r1")

        for stage_cfg in expanded:
            if stage_cfg.name.startswith("gen@r"):
                args = resolve_stage_typed_kwargs(stage_cfg)
                assert (
                    args.get("max_seq_len") == 4096
                ), f"{stage_cfg.name} lost the value configured for 'gen'"

    def test_unreplicated_stage_does_not_borrow_overrides(self):
        from sglang_omni.config.runtime import resolve_stage_typed_kwargs

        config = self._config()
        src = {s.name: s for s in config.stages}["src"]
        assert "max_seq_len" not in resolve_stage_typed_kwargs(src)


class TestReceiveSideLogicalNames:
    """A replica sends its instance name; fan-in and streams expect logical ones."""

    def _payload(self, request_id: str = "req"):
        from sglang_omni.proto import OmniRequest, StagePayload

        return StagePayload(
            request_id=request_id, request=OmniRequest(inputs="x"), data={}
        )

    def _stage(self, handler, **kwargs):
        from tests.unit_test.pipeline.helpers import make_stage

        return make_stage(name="aggregate", input_handler=handler, **kwargs)

    def test_fan_in_accepts_a_replicated_upstream(self):
        import asyncio

        from sglang_omni.pipeline.stage.input import AggregatedInput

        merged: list[list[str]] = []

        def merge(inputs):
            merged.append(sorted(inputs))
            return self._payload()

        stage = self._stage(
            AggregatedInput(sources={"a", "b"}, merge=merge),
            replica_topology={"a": ["a@r0", "a@r1"]},
        )

        async def run() -> None:
            await stage.receive_local_payload("req", "a@r0", self._payload())
            await stage.receive_local_payload("req", "b", self._payload())

        asyncio.run(run())

        assert merged == [["a", "b"]]

    def test_wait_for_fn_sees_the_logical_source(self):
        import asyncio

        from sglang_omni.pipeline.stage.input import AggregatedInput

        seen: list[str] = []

        def wait_for_fn(request_id, from_stage, data):
            seen.append(from_stage)
            return ["a", "b"]

        stage = self._stage(
            AggregatedInput(
                sources={"a", "b"},
                merge=lambda inputs: self._payload(),
                expected_sources_fn=wait_for_fn,
            ),
            replica_topology={"a": ["a@r0", "a@r1"]},
        )

        asyncio.run(stage.receive_local_payload("req", "a@r1", self._payload()))

        assert seen == ["a"]

    def test_unregistered_replica_suffix_passes_through(self):
        import asyncio

        from sglang_omni.pipeline.stage.input import AggregatedInput

        merged: list[list[str]] = []

        def merge(inputs):
            merged.append(sorted(inputs))
            return self._payload()

        stage = self._stage(
            AggregatedInput(sources={"other@r0"}, merge=merge),
            replica_topology={"a": ["a@r0"]},
        )

        asyncio.run(stage.receive_local_payload("req", "other@r0", self._payload()))

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
        stage._stream_queue = StreamQueue()

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
        config = _config(
            [_stage("s", process="p", gpu=1)],
            processes={"p": ProcessConfig(num_replicas=1, replica_devices=[7])},
        )
        _, expanded, topology = _expand(config)

        assert [(stage.name, stage.gpu, stage.process) for stage in expanded] == [
            ("s", 7, "p")
        ]
        assert topology.to_dict() == {}

    def test_tp_process_single_replica_device_override(self):
        config = _config(
            [_stage("thinker", tp_size=2, gpu=[0, 1], process=None)],
            processes={
                "thinker": ProcessConfig(num_replicas=1, replica_devices=[4, 5])
            },
        )
        _, expanded, _ = _expand(config)

        assert [(stage.name, stage.gpu) for stage in expanded] == [("thinker", [4, 5])]
