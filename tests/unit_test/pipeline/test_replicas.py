# SPDX-License-Identifier: Apache-2.0
"""Unit tests for runtime-level stage replicas."""

import pytest

from sglang_omni.config.schema import PipelineConfig, StageConfig
from sglang_omni.pipeline.replicas import (
    ReplicaTopology,
    RoundRobinBindingPolicy,
    assign_replica_bindings,
    expand_replica_stages,
    parse_replica_instance_name,
    replica_instance_name,
    split_replica_devices,
    validate_device_assignment,
)


def _stage(name: str, **kwargs) -> StageConfig:
    defaults = dict(factory="pkg.mod.create", terminal=True, process=name)
    defaults.update(kwargs)
    return StageConfig(name=name, **defaults)


class TestInstanceNaming:
    def test_round_trip(self):
        name = replica_instance_name("talker_ar", 1)
        assert name == "talker_ar@r1"
        assert parse_replica_instance_name(name) == ("talker_ar", 1)

    def test_plain_name_passthrough(self):
        assert parse_replica_instance_name("thinker") == ("thinker", None)

    def test_non_numeric_suffix_is_not_replica(self):
        assert parse_replica_instance_name("stage@rx") == ("stage@rx", None)


class TestSplitReplicaDevices:
    def test_pool_mode_tp1(self):
        assert split_replica_devices(
            "1,2", stage_name="s", num_replicas=2, tp_size=1
        ) == [[1], [2]]

    def test_pool_mode_tp2(self):
        assert split_replica_devices(
            "0,1,2,3", stage_name="s", num_replicas=2, tp_size=2
        ) == [[0, 1], [2, 3]]

    def test_non_contiguous_pool(self):
        assert split_replica_devices(
            "1,5", stage_name="s", num_replicas=2, tp_size=1
        ) == [[1], [5]]

    def test_same_gpu_pool(self):
        assert split_replica_devices(
            "0,0", stage_name="s", num_replicas=2, tp_size=1
        ) == [[0], [0]]

    def test_partial_list_raises(self):
        with pytest.raises(ValueError, match="expected 4"):
            split_replica_devices("0", stage_name="s", num_replicas=4, tp_size=1)
        with pytest.raises(ValueError, match="expected 4"):
            split_replica_devices("0,1", stage_name="s", num_replicas=2, tp_size=2)

    def test_list_input(self):
        assert split_replica_devices(
            [1, 2], stage_name="s", num_replicas=2, tp_size=1
        ) == [[1], [2]]

    def test_none_for_cpu_stage(self):
        assert split_replica_devices(
            None, stage_name="s", num_replicas=3, tp_size=1
        ) == [None, None, None]

    def test_invalid_length_raises(self):
        with pytest.raises(ValueError, match="replica_devices has 3"):
            split_replica_devices("0,1,2", stage_name="s", num_replicas=2, tp_size=2)


class TestExpandReplicaStages:
    def test_no_replicas_is_identity(self):
        stages = [_stage("a"), _stage("b")]
        expanded, topo = expand_replica_stages(stages)
        assert expanded == stages
        assert not topo
        assert topo.to_dict() == {}

    def test_expansion_names_gpus_processes(self):
        stages = [
            _stage(
                "talker_ar",
                terminal=False,
                next="code2wav",
                stream_to=["code2wav"],
                gpu=1,
                num_replicas=2,
                replica_devices="1,2",
            ),
            _stage("code2wav"),
        ]
        expanded, topo = expand_replica_stages(stages)
        names = [s.name for s in expanded]
        assert names == ["talker_ar@r0", "talker_ar@r1", "code2wav"]
        r0, r1 = expanded[0], expanded[1]
        assert (r0.gpu, r1.gpu) == (1, 2)
        assert r0.process == "talker_ar@r0"
        assert r0.num_replicas == 1 and r0.replica_devices is None
        # Wiring keeps logical names.
        assert r0.next == "code2wav" and r0.stream_to == ["code2wav"]
        assert topo.to_dict() == {"talker_ar": ["talker_ar@r0", "talker_ar@r1"]}

    def test_gpu_stage_requires_explicit_replica_devices(self):
        stages = [_stage("s", gpu=1, num_replicas=2)]
        with pytest.raises(ValueError, match="replica_devices"):
            expand_replica_stages(stages)

    def test_cpu_stage_needs_no_devices(self):
        stages = [_stage("s", num_replicas=2)]
        expanded, topo = expand_replica_stages(stages)
        assert [s.gpu for s in expanded] == [None, None]
        assert topo.to_dict() == {"s": ["s@r0", "s@r1"]}


class TestValidateDeviceAssignment:
    def test_valid_ids_pass(self):
        stages, _ = expand_replica_stages(
            [_stage("s", gpu=1, num_replicas=2, replica_devices="1,2")]
        )
        validate_device_assignment(stages, device_count=4)

    def test_out_of_range_id_raises(self):
        stages, _ = expand_replica_stages(
            [_stage("s", gpu=3, num_replicas=2, replica_devices="3,4")]
        )
        with pytest.raises(ValueError, match="GPU id 4"):
            validate_device_assignment(stages, device_count=4)

    def test_negative_id_raises(self):
        with pytest.raises(ValueError, match="GPU id -1"):
            validate_device_assignment([_stage("s", gpu=-1)], device_count=4)

    def test_duplicate_id_within_tp_group_raises(self):
        with pytest.raises(ValueError, match="duplicate"):
            validate_device_assignment(
                [_stage("s", gpu=[0, 0], tp_size=2)], device_count=4
            )

    def test_cpu_stages_are_skipped(self):
        validate_device_assignment([_stage("s")], device_count=0)

    def test_unknown_device_count_skips_range_check(self):
        validate_device_assignment([_stage("s", gpu=7)], device_count=None)


class TestReplicaTopology:
    def _topo(self) -> ReplicaTopology:
        _, topo = expand_replica_stages(
            [
                _stage("talker_ar", num_replicas=2, replica_devices="1,2", gpu=1),
                _stage("code2wav", num_replicas=2, replica_devices="1,2", gpu=1),
                _stage("thinker"),
            ]
        )
        return topo

    def test_resolve_and_logical_name(self):
        topo = self._topo()
        assert topo.resolve("talker_ar", 1) == "talker_ar@r1"
        assert topo.resolve_bound("talker_ar", {"talker_ar": 1}) == "talker_ar@r1"
        assert topo.resolve_bound("thinker", None) == "thinker"
        assert topo.logical_name("talker_ar@r1") == "talker_ar"
        assert topo.logical_name("thinker") == "thinker"

    def test_resolve_bound_requires_replica_binding(self):
        with pytest.raises(ValueError, match="no replica binding"):
            self._topo().resolve_bound("talker_ar", None)

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
    def test_round_robin_cycles_per_stage(self):
        policy = RoundRobinBindingPolicy()
        picks = [policy.bind("talker_ar", 2, f"req{i}") for i in range(4)]
        assert picks == [0, 1, 0, 1]
        assert policy.bind("code2wav", 3, "reqx") == 0

    def test_assign_bindings(self):
        _, topo = expand_replica_stages(
            [
                _stage("talker_ar", num_replicas=2, replica_devices="1,2", gpu=1),
                _stage("code2wav", num_replicas=2, replica_devices="1,2", gpu=1),
            ]
        )
        policy = RoundRobinBindingPolicy()
        first = assign_replica_bindings(topo, policy, "req0")
        second = assign_replica_bindings(topo, policy, "req1")
        assert first == {"talker_ar": 0, "code2wav": 0}
        assert second == {"talker_ar": 1, "code2wav": 1}

    def test_empty_topology_binds_none(self):
        assert (
            assign_replica_bindings(ReplicaTopology(), RoundRobinBindingPolicy(), "r")
            is None
        )


_SPEECH_STAGES = (
    "preprocessing",
    "image_encoder",
    "audio_encoder",
    "mm_aggregate",
    "thinker",
    "decode",
    "talker_ar",
    "code2wav",
)


def _speech_config(config_cls_name: str | None = None) -> PipelineConfig:
    cls = (
        type(config_cls_name, (PipelineConfig,), {})
        if config_cls_name
        else PipelineConfig
    )
    return cls(
        model_path="m",
        stages=[
            _stage(
                name,
                **(
                    {"num_replicas": 2, "replica_devices": "1,2", "gpu": 1}
                    if name == "talker_ar"
                    else {}
                ),
            )
            for name in _SPEECH_STAGES
        ],
    )


class TestColocatedReplicaRejection:
    _SPEECH_STAGES = _SPEECH_STAGES

    def test_colocated_rejects_replicated_stage(self):
        from sglang_omni.models.qwen3_omni.placement import Qwen3OmniPlacementPolicy

        colocated_cls = type(
            "Qwen3OmniSpeechColocatedPipelineConfig", (PipelineConfig,), {}
        )
        config = colocated_cls(
            model_path="m",
            stages=[
                _stage(
                    name,
                    **(
                        {"num_replicas": 2, "replica_devices": "0,1", "gpu": 0}
                        if name == "talker_ar"
                        else {}
                    ),
                )
                for name in self._SPEECH_STAGES
            ],
        )
        with pytest.raises(ValueError, match="does not support stage replicas"):
            Qwen3OmniPlacementPolicy().validate(config, plan=None)


class TestPlacementLogicalView:
    def _plan(self, talker_r0_gpu: int) -> "StagePlacementPlan":
        from sglang_omni.config.placement import StagePlacement, StagePlacementPlan

        def placement(name: str, gpu: int) -> StagePlacement:
            return StagePlacement(
                stage_name=name,
                gpu_ids=(gpu,),
                tp_size=1,
                total_gpu_memory_fraction=None,
            )

        return StagePlacementPlan(
            stages={
                "thinker": placement("thinker", 0),
                "talker_ar@r0": placement("talker_ar@r0", talker_r0_gpu),
                "talker_ar@r1": placement("talker_ar@r1", 2),
            },
            gpus={},
            replica_instances={"talker_ar": ("talker_ar@r0", "talker_ar@r1")},
        )

    def test_instances_of(self):
        plan = self._plan(talker_r0_gpu=1)
        assert [p.stage_name for p in plan.instances_of("talker_ar")] == [
            "talker_ar@r0",
            "talker_ar@r1",
        ]
        assert [p.stage_name for p in plan.instances_of("thinker")] == ["thinker"]
        assert plan.instances_of("preprocessing") == []

    def test_policy_catches_replica_sharing_gpu_with_thinker(self):
        from sglang_omni.models.qwen3_omni.placement import Qwen3OmniPlacementPolicy

        with pytest.raises(ValueError, match="talker_ar@r0"):
            Qwen3OmniPlacementPolicy().validate(
                _speech_config(), self._plan(talker_r0_gpu=0)
            )

    def test_policy_passes_disjoint_replicas(self):
        from sglang_omni.models.qwen3_omni.placement import Qwen3OmniPlacementPolicy

        Qwen3OmniPlacementPolicy().validate(
            _speech_config(), self._plan(talker_r0_gpu=1)
        )


class TestStageOverrides:
    def _manager(self):
        pytest.importorskip("transformers")
        from sglang_omni.config import manager

        return manager

    def test_replica_fields_pass_through(self):
        manager = self._manager()
        config = _speech_config()
        overridden = manager._apply_stage_overrides(
            config,
            {"code2wav": {"num_replicas": 3, "replica_devices": "1,2,3"}},
        )
        stage = {s.name: s for s in overridden.stages}["code2wav"]
        assert stage.num_replicas == 3
        assert stage.replica_devices == "1,2,3"

    def test_unsupported_key_still_rejected(self):
        manager = self._manager()
        with pytest.raises(ValueError, match="unsupported keys"):
            manager._apply_stage_overrides(_speech_config(), {"code2wav": {"gpu": 3}})


class TestSchemaValidation:
    def test_num_replicas_must_be_positive(self):
        with pytest.raises(ValueError, match="num_replicas >= 1"):
            _stage("s", num_replicas=0)

    def test_entry_stage_can_be_replicated(self):
        config = PipelineConfig(
            model_path="m",
            stages=[
                _stage("entry", terminal=False, next="sink", num_replicas=2),
                _stage("sink"),
            ],
        )
        assert config.resolved_entry_stage == "entry"

    def test_fused_group_cannot_include_replicated_stage(self):
        with pytest.raises(ValueError, match="cannot include replicated"):
            PipelineConfig(
                model_path="m",
                stages=[
                    _stage("a", terminal=False, next="b"),
                    _stage("b", terminal=False, next="c", num_replicas=2),
                    _stage("c"),
                ],
                fused_stages=[["a", "b"]],
            )


class TestUnequalReplicaCounts:
    def test_bindings_cycle_independently_and_resolve(self):
        _, topo = expand_replica_stages(
            [
                _stage("talker_ar", num_replicas=3, replica_devices="1,2,3", gpu=1),
                _stage("code2wav", num_replicas=2, replica_devices="1,2", gpu=1),
            ]
        )
        policy = RoundRobinBindingPolicy()
        bindings = [assign_replica_bindings(topo, policy, f"req{i}") for i in range(6)]
        assert [b["talker_ar"] for b in bindings] == [0, 1, 2, 0, 1, 2]
        assert [b["code2wav"] for b in bindings] == [0, 1, 0, 1, 0, 1]
        for b in bindings:
            assert (
                topo.resolve("talker_ar", b["talker_ar"])
                == f"talker_ar@r{b['talker_ar']}"
            )
            assert (
                topo.resolve("code2wav", b["code2wav"]) == f"code2wav@r{b['code2wav']}"
            )


class TestReservedStageNames:
    def test_reserved_instance_suffix_rejected(self):
        with pytest.raises(ValueError, match="reserved"):
            PipelineConfig(model_path="m", stages=[_stage("foo@r0")])

    def test_non_numeric_suffix_allowed(self):
        PipelineConfig(model_path="m", stages=[_stage("foo@rx")])


class TestRuntimeOverridesOnReplicas:
    def _config(self) -> PipelineConfig:
        return PipelineConfig(
            model_path="m",
            stages=[
                _stage("src", terminal=False, next="gen"),
                _stage("gen", num_replicas=2, replica_devices="1,2", gpu=1),
            ],
            runtime_overrides={"gen": {"max_seq_len": 4096}},
        )

    def test_replica_instances_inherit_logical_overrides(self):
        from sglang_omni.config.runtime import resolve_stage_static_factory_args

        config = self._config()
        expanded, topo = expand_replica_stages(list(config.stages))
        assert topo.instances("gen") == ("gen@r0", "gen@r1")

        for stage_cfg in expanded:
            if stage_cfg.name.startswith("gen@r"):
                args = resolve_stage_static_factory_args(stage_cfg, config)
                assert (
                    args.get("max_seq_len") == 4096
                ), f"{stage_cfg.name} lost the override configured for 'gen'"

    def test_unreplicated_stage_does_not_borrow_overrides(self):
        from sglang_omni.config.runtime import resolve_stage_static_factory_args

        config = self._config()
        src = {s.name: s for s in config.stages}["src"]
        assert "max_seq_len" not in resolve_stage_static_factory_args(src, config)
