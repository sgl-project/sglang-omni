# SPDX-License-Identifier: Apache-2.0
"""CPU worker processes are spread across NUMA nodes; the binding unit is the process."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.platforms import cpu_numa
from sglang_omni.platforms.cpu import CPUOmniPlatform

_BIND = cpu_numa.THREADS_BIND_ENV
_MAP = cpu_numa.NUMA_MAP_ENV


@pytest.fixture(autouse=True)
def fresh_assignment():
    """Placement is recorded per parent process; tests must not inherit it."""
    cpu_numa.reset_assignment()
    yield
    cpu_numa.reset_assignment()


@pytest.fixture
def nodes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A fixed four-node topology, so tests do not depend on the host."""
    topology = ["0-9", "10-19", "20-29", "30-39"]
    monkeypatch.setattr(cpu_numa, "numa_nodes", lambda: dict(enumerate(topology)))
    return topology


def _stage(name: str, tp_size: int = 1, tp_rank: int = 0):
    return SimpleNamespace(
        stage_name=name,
        tp_size=tp_size,
        tp_rank=tp_rank,
        gpu_id=0,
        allow_child_processes=False,
    )


def _bind(platform, process_name, stage_names, env, tp_size: int = 1, tp_rank: int = 0):
    specs = [_stage(n, tp_size, tp_rank) for n in stage_names]
    return platform.get_process_placement_env(process_name, specs, env=env)


# --- automatic spreading ------------------------------------------------


def test_separate_processes_land_on_separate_nodes(nodes) -> None:
    """The whole point: two processes must not share a node, or they each
    budget a full node's memory against the same node.
    """
    platform = CPUOmniPlatform()

    masks = [
        _bind(platform, f"proc{i}", [f"stage{i}"], env={})[_BIND]
        for i in range(len(nodes))
    ]

    assert masks == nodes
    assert len(set(masks)) == len(nodes)


def test_more_processes_than_nodes_wrap(nodes) -> None:
    """A pipeline wider than the host must still start; the extra processes
    share nodes again, which is the pre-existing behaviour, not a new failure.
    """
    platform = CPUOmniPlatform()

    masks = [
        _bind(platform, f"proc{i}", [f"stage{i}"], env={})[_BIND]
        for i in range(len(nodes) + 2)
    ]

    assert masks[len(nodes) :] == nodes[:2]


def test_stages_sharing_a_process_share_its_node(nodes) -> None:
    """MOSS-TTS and Voxtral run all three stages in one process; the binding
    covers the process, so they cannot be split and must not be double-counted
    as separate placements.
    """
    platform = CPUOmniPlatform()

    first = _bind(platform, "pipeline", ["pre", "engine", "vocoder"], env={})
    second = _bind(platform, "other", ["extra"], env={})

    assert first[_BIND] == nodes[0]
    assert second[_BIND] == nodes[1]


def test_asking_twice_for_one_process_is_stable(nodes) -> None:
    platform = CPUOmniPlatform()

    first = _bind(platform, "pipeline", ["a"], env={})
    second = _bind(platform, "pipeline", ["a"], env={})

    assert first == second


# --- user-supplied map --------------------------------------------------


def test_a_user_map_places_the_named_stage(nodes) -> None:
    platform = CPUOmniPlatform()

    mask = _bind(platform, "p", ["talker"], env={_MAP: "talker=2"})[_BIND]

    assert mask == nodes[2]


def test_a_user_map_entry_beyond_the_node_count_is_rejected(nodes) -> None:
    platform = CPUOmniPlatform()

    with pytest.raises(ValueError, match="NUMA node"):
        _bind(platform, "p", ["talker"], env={_MAP: f"talker={len(nodes)}"})


def test_unnamed_processes_still_spread_around_a_user_entry(nodes) -> None:
    """A partial map must not collapse the rest onto one node."""
    platform = CPUOmniPlatform()

    pinned = _bind(platform, "b", ["talker"], env={_MAP: "talker=3"})[_BIND]
    auto = _bind(platform, "a", ["thinker"], env={_MAP: "talker=3"})[_BIND]

    assert pinned == nodes[3]
    assert auto != pinned


def test_auto_placement_keeps_off_nodes_the_map_reserves(nodes) -> None:
    """A stage placed later on its mapped node must not find an auto worker there."""
    platform = CPUOmniPlatform()
    env = {_MAP: "talker=1"}

    auto = [_bind(platform, f"p{i}", [f"s{i}"], env=env)[_BIND] for i in range(3)]
    talker = _bind(platform, "talker", ["talker"], env=env)[_BIND]

    assert nodes[1] not in auto
    assert talker == nodes[1]


def test_splitting_one_process_across_nodes_is_refused(nodes) -> None:
    """The mask binds the process, so honouring both entries is impossible.
    Saying so beats silently binding to whichever stage came first.
    """
    platform = CPUOmniPlatform()

    with pytest.raises(cpu_numa.NumaPlacementConflict, match="split one process"):
        _bind(platform, "pipeline", ["pre", "vocoder"], env={_MAP: "pre=0,vocoder=2"})


def test_agreeing_entries_in_one_process_are_fine(nodes) -> None:
    platform = CPUOmniPlatform()

    mask = _bind(
        platform, "pipeline", ["pre", "vocoder"], env={_MAP: "pre=2,vocoder=2"}
    )

    assert mask[_BIND] == nodes[2]


@pytest.mark.parametrize("raw", ["thinker", "thinker=x", "thinker=-1", "=0"])
def test_a_malformed_map_is_rejected_with_its_own_text(raw: str) -> None:
    with pytest.raises(ValueError, match=_MAP):
        cpu_numa.parse_numa_map(raw)


def test_an_empty_map_is_not_an_error() -> None:
    assert cpu_numa.parse_numa_map(None) == {}
    assert cpu_numa.parse_numa_map("  ") == {}


# --- deferrals ----------------------------------------------------------


def test_an_operator_supplied_mask_is_never_second_guessed(nodes) -> None:
    """Overriding the binding is how someone tunes a deployment; replacing
    their mask would make the knob useless.
    """
    platform = CPUOmniPlatform()

    assert _bind(platform, "p", ["s"], env={_BIND: "0-3"}) == {}


def test_each_tp_rank_process_gets_its_own_node(nodes) -> None:
    platform = CPUOmniPlatform()

    rank0 = _bind(platform, "thinker_tp0", ["thinker"], env={}, tp_size=2, tp_rank=0)
    rank1 = _bind(platform, "thinker_tp1", ["thinker"], env={}, tp_size=2, tp_rank=1)

    assert rank0 != rank1
    assert all(_BIND in env for env in (rank0, rank1))


def test_a_tp_group_takes_the_least_used_nodes(nodes) -> None:
    platform = CPUOmniPlatform()

    _bind(platform, "pre", ["pre"], env={})
    _bind(platform, "img", ["img"], env={})
    ranks = [
        _bind(platform, f"thinker_tp{r}", ["thinker"], env={}, tp_size=2, tp_rank=r)[
            _BIND
        ]
        for r in range(2)
    ]

    assert ranks == [nodes[2], nodes[3]]


def test_a_user_map_lists_one_node_per_tp_rank(nodes) -> None:
    platform = CPUOmniPlatform()
    env = {_MAP: "thinker=1+3"}

    ranks = [
        _bind(platform, f"thinker_tp{r}", ["thinker"], env=env, tp_size=2, tp_rank=r)[
            _BIND
        ]
        for r in range(2)
    ]

    assert ranks == [nodes[1], nodes[3]]


def test_a_user_map_with_the_wrong_rank_count_is_rejected(nodes) -> None:
    with pytest.raises(ValueError, match="tp_size"):
        _bind(
            CPUOmniPlatform(),
            "thinker_tp0",
            ["thinker"],
            env={_MAP: "thinker=0+1+2"},
            tp_size=2,
        )


def test_an_unreadable_topology_degrades_to_the_previous_behaviour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failing to read sysfs must not break startup — it leaves SGLang on the
    node-0 default that predates this hook.
    """
    monkeypatch.setattr(cpu_numa, "numa_nodes", lambda: {})

    assert CPUOmniPlatform().get_process_placement_env("p", [_stage("s")], env={}) == {}


def test_a_process_with_no_stages_is_left_alone(nodes) -> None:
    assert CPUOmniPlatform().get_process_placement_env("p", [], env={}) == {}


# --- wiring -------------------------------------------------------------


def test_every_process_asks_the_platform_not_just_tp_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The spawn-time hook used to short-circuit unless a stage had tp_size > 1.

    That suits accelerators, which place by device index and need no env below
    TP, but it left CPU — whose only placement channel *is* env — unable to say
    anything at all.
    """
    from sglang_omni.pipeline import stage_workers

    asked: list[tuple[str, list[str]]] = []

    class _Platform:
        def is_cpu(self):
            return True

        def get_stage_process_env(self, spec, env=None):
            raise AssertionError("the per-stage hook is for TP stages only")

        def get_process_placement_env(self, process_name, stage_specs, env=None):
            asked.append((process_name, [s.stage_name for s in stage_specs]))
            return {"MARKER": process_name}

    monkeypatch.setattr(stage_workers, "current_platform", _Platform())

    spec = SimpleNamespace(
        process_name="pipeline",
        stage_specs=[_stage("pre"), _stage("vocoder")],
    )

    assert stage_workers.get_worker_process_env(spec) == {"MARKER": "pipeline"}
    assert asked == [("pipeline", ["pre", "vocoder"])]


def test_a_tp_stage_still_must_own_its_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Relaxing the gate must not relax the invariant behind it: a TP stage's
    device remap and NCCL settings assume it is the sole tenant.
    """
    from sglang_omni.pipeline import stage_workers

    monkeypatch.setattr(
        stage_workers,
        "current_platform",
        SimpleNamespace(
            is_cpu=lambda: True,
            get_stage_process_env=lambda spec, env=None: {},
            get_process_placement_env=lambda p, s, env=None: {},
        ),
    )

    spec = SimpleNamespace(
        process_name="pipeline",
        stage_specs=[_stage("thinker", tp_size=2), _stage("decode")],
    )

    with pytest.raises(AssertionError, match="must own their OS process"):
        stage_workers.get_worker_process_env(spec)


def test_accelerators_are_unaffected() -> None:
    """The new hook defaults to {} so CUDA and XPU behaviour cannot shift."""
    from sglang_omni.platforms.cuda import CUDAOmniPlatform
    from sglang_omni.platforms.xpu import XPUOmniPlatform

    for cls in (CUDAOmniPlatform, XPUOmniPlatform):
        platform = cls.__new__(cls)
        assert platform.get_process_placement_env("p", [_stage("s")], env={}) == {}


def test_bound_cpus_reads_the_first_rank_range() -> None:
    assert cpu_numa.bound_cpus({_BIND: "0-9"}) == set(range(10))
    assert len(cpu_numa.bound_cpus({_BIND: "0-3,8|16-31"})) == 5
    assert cpu_numa.bound_cpus({}) is None
    assert cpu_numa.bound_cpus({_BIND: "all"}) is None


def _spawn_spec(env_defaults, written_env=frozenset(), cpu_threads=None):
    from sglang_omni.pipeline.stage_workers import StageWorkerProcessSpec

    stage = SimpleNamespace(
        stage_name="pre",
        tp_size=1,
        tp_rank=0,
        gpu_id=0,
        allow_child_processes=False,
        env_defaults=env_defaults,
        written_env=written_env,
    )
    return StageWorkerProcessSpec(
        process_name="p", stage_specs=[stage], cpu_threads=cpu_threads
    )


def _spawned_omp_threads(monkeypatch, spec) -> str:
    import os

    from sglang_omni.pipeline import stage_workers

    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setattr(stage_workers, "current_platform", CPUOmniPlatform())
    monkeypatch.setattr(stage_workers, "get_platform_spec", lambda _p: "cpu")
    monkeypatch.setenv(_BIND, "0-9")
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(40)))
    with stage_workers.patched_spawn_env(spec):
        return os.environ["OMP_NUM_THREADS"]


def test_host_sized_omp_default_is_capped_to_the_bound_node(monkeypatch) -> None:
    spec = _spawn_spec({"OMP_NUM_THREADS": "192"})

    assert _spawned_omp_threads(monkeypatch, spec) == "10"


def test_written_omp_value_beats_the_node_cap(monkeypatch) -> None:
    spec = _spawn_spec(
        {"OMP_NUM_THREADS": "192"}, written_env=frozenset({"OMP_NUM_THREADS"})
    )

    assert _spawned_omp_threads(monkeypatch, spec) == "192"


def test_cpu_plan_fallback_is_capped_to_the_bound_node(monkeypatch) -> None:
    spec = _spawn_spec({}, cpu_threads=48)

    assert _spawned_omp_threads(monkeypatch, spec) == "10"


def test_derived_omp_default_is_capped_to_the_quota_share(monkeypatch) -> None:
    spec = _spawn_spec({"OMP_NUM_THREADS": "28"}, cpu_threads=4)

    assert _spawned_omp_threads(monkeypatch, spec) == "4"


def test_omp_cap_counts_only_the_cpus_the_launcher_allows(monkeypatch) -> None:
    import os

    spec = _spawn_spec({"OMP_NUM_THREADS": "192"})
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(8, 40)))
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

    from sglang_omni.pipeline import stage_workers

    monkeypatch.setattr(stage_workers, "current_platform", CPUOmniPlatform())
    monkeypatch.setattr(stage_workers, "get_platform_spec", lambda _p: "cpu")
    monkeypatch.setenv(_BIND, "4-11")
    with stage_workers.patched_spawn_env(spec):
        assert os.environ["OMP_NUM_THREADS"] == "4"


def test_omp_cap_falls_back_to_the_launcher_mask_when_the_bind_is_disallowed(
    monkeypatch,
) -> None:
    import os

    spec = _spawn_spec({"OMP_NUM_THREADS": "192"})
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(20, 26)))
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

    from sglang_omni.pipeline import stage_workers

    monkeypatch.setattr(stage_workers, "current_platform", CPUOmniPlatform())
    monkeypatch.setattr(stage_workers, "get_platform_spec", lambda _p: "cpu")
    monkeypatch.setenv(_BIND, "0-9")
    with stage_workers.patched_spawn_env(spec):
        assert os.environ["OMP_NUM_THREADS"] == "6"


def test_node_cap_never_raises_an_affinity_capped_default(monkeypatch) -> None:
    spec = _spawn_spec({"OMP_NUM_THREADS": "192"})
    spec.cpu_affinity = frozenset(range(4))

    assert _spawned_omp_threads(monkeypatch, spec) == "4"


def test_a_bind_list_configured_on_the_stage_is_not_overwritten(
    monkeypatch, nodes
) -> None:
    from sglang_omni.pipeline import stage_workers

    monkeypatch.setattr(stage_workers, "current_platform", CPUOmniPlatform())
    spec = _spawn_spec({})

    assert stage_workers.get_worker_process_env(spec, {_BIND: "0-3"}) == {}


def test_a_stage_owning_child_processes_is_still_placed(monkeypatch, nodes) -> None:
    from sglang_omni.pipeline import stage_workers

    monkeypatch.setattr(stage_workers, "current_platform", CPUOmniPlatform())
    spec = _spawn_spec({})
    spec.stage_specs[0].allow_child_processes = True

    assert stage_workers.get_worker_process_env(spec) == {_BIND: nodes[0]}


def test_map_entries_are_lscpu_node_ids_not_list_offsets(monkeypatch) -> None:
    monkeypatch.setattr(
        cpu_numa, "numa_nodes", lambda: {0: "0-9", 2: "20-29", 5: "50-59"}
    )
    platform = CPUOmniPlatform()

    assert _bind(platform, "p", ["talker"], {_MAP: "talker=5"})[_BIND] == "50-59"
    with pytest.raises(ValueError, match="NUMA node"):
        _bind(platform, "q", ["thinker"], {_MAP: "thinker=1"})


def test_an_explicit_bind_list_counts_against_auto_placement(nodes) -> None:
    platform = CPUOmniPlatform()

    assert _bind(platform, "pinned", ["a"], {_BIND: "0-9"}) == {}

    assert _bind(platform, "auto", ["b"], {})[_BIND] == nodes[1]


def test_the_spawning_thread_runs_on_the_bound_node(monkeypatch) -> None:
    import os

    from sglang_omni.pipeline import stage_workers

    calls: list[set[int]] = []
    monkeypatch.setattr(stage_workers, "current_platform", CPUOmniPlatform())
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(40)))
    monkeypatch.setattr(os, "sched_setaffinity", lambda _pid, cpus: calls.append(cpus))
    monkeypatch.setenv(_BIND, "4-6")

    with stage_workers.spawn_affinity(_spawn_spec({})):
        pass

    assert calls == [{4, 5, 6}, set(range(40))]


def test_a_worker_without_sglang_is_still_moved_onto_its_node(monkeypatch) -> None:
    import os

    applied: list[set[int]] = []
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(40)))
    monkeypatch.setattr(
        os, "sched_setaffinity", lambda _pid, cpus: applied.append(set(cpus))
    )
    monkeypatch.setattr(cpu_numa, "get_cpu_ids_by_node", lambda: [])
    monkeypatch.delenv(cpu_numa.LAUNCHER_CPUS_ENV, raising=False)
    monkeypatch.setenv(_BIND, "4-6")

    cpu_numa.apply_bound_affinity()

    assert applied == [{4, 5, 6}]


def test_affinity_is_left_alone_when_the_topology_cannot_be_cached(
    monkeypatch,
) -> None:
    import os

    def unreadable():
        raise RuntimeError("no lscpu")

    applied: list[set[int]] = []
    monkeypatch.setattr(cpu_numa, "get_cpu_ids_by_node", unreadable)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(40)))
    monkeypatch.setattr(
        os, "sched_setaffinity", lambda _pid, cpus: applied.append(cpus)
    )
    monkeypatch.delenv(cpu_numa.LAUNCHER_CPUS_ENV, raising=False)
    monkeypatch.setenv(_BIND, "0-3")

    cpu_numa.apply_bound_affinity()

    assert applied == []


def test_cpu_workers_use_the_first_bind_entry() -> None:
    """CPU engines always hand SGLang gpu_id 0, whatever the stage's configured id."""
    assert cpu_numa.bound_cpus({_BIND: "0-3|4-35"}) == {0, 1, 2, 3}


def test_the_child_caches_the_topology_under_the_launchers_mask(monkeypatch) -> None:
    import os

    masks: list[set[int]] = []
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {4, 5, 6})
    monkeypatch.setattr(
        os, "sched_setaffinity", lambda _pid, cpus: masks.append(set(cpus))
    )
    monkeypatch.setattr(
        cpu_numa, "get_cpu_ids_by_node", lambda: masks.append({-1}) or []
    )
    monkeypatch.setenv(_BIND, "4-6")
    monkeypatch.setenv(cpu_numa.LAUNCHER_CPUS_ENV, "0-39")

    cpu_numa.apply_bound_affinity()

    assert masks == [set(range(40)), {-1}, {4, 5, 6}]
    assert cpu_numa.LAUNCHER_CPUS_ENV not in os.environ


def test_an_unbound_worker_never_reads_the_launcher_affinity(monkeypatch) -> None:
    import os

    from sglang_omni.pipeline import stage_workers

    def unavailable(_pid):
        raise AttributeError("no sched_getaffinity on this OS")

    monkeypatch.setattr(
        stage_workers, "current_platform", SimpleNamespace(is_cpu=lambda: False)
    )
    monkeypatch.setattr(os, "sched_getaffinity", unavailable)

    with stage_workers.spawn_affinity(_spawn_spec({})):
        pass


def test_the_child_sees_the_launchers_full_mask_during_spawn(monkeypatch) -> None:
    import os

    from sglang_omni.pipeline import stage_workers

    monkeypatch.setattr(stage_workers, "current_platform", CPUOmniPlatform())
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(8)))
    monkeypatch.setattr(os, "sched_setaffinity", lambda _pid, _cpus: None)
    monkeypatch.setenv(_BIND, "4-6")
    monkeypatch.delenv(cpu_numa.LAUNCHER_CPUS_ENV, raising=False)

    with stage_workers.spawn_affinity(_spawn_spec({})):
        assert os.environ[cpu_numa.LAUNCHER_CPUS_ENV] == "0,1,2,3,4,5,6,7"

    assert cpu_numa.LAUNCHER_CPUS_ENV not in os.environ


def test_bound_affinity_never_leaves_the_launchers_allocation(monkeypatch) -> None:
    import os

    applied: list[set[int]] = []
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {4, 5, 6, 7})
    monkeypatch.setattr(
        os, "sched_setaffinity", lambda _pid, cpus: applied.append(set(cpus))
    )
    monkeypatch.setattr(cpu_numa, "get_cpu_ids_by_node", lambda: [])
    monkeypatch.setenv(_BIND, "6-9")
    monkeypatch.setenv(cpu_numa.LAUNCHER_CPUS_ENV, "4-7")

    cpu_numa.apply_bound_affinity()

    assert applied[-1] == {6, 7}


def test_replica_instances_use_the_logical_stage_map_entry(nodes) -> None:
    platform = CPUOmniPlatform()
    env = {_MAP: "talker=2,talker@r1=3"}

    replicas = [
        _bind(platform, f"talker@r{i}", [f"talker@r{i}"], env)[_BIND] for i in range(2)
    ]

    assert replicas == [nodes[2], nodes[3]]


def test_binding_hook_widens_the_mask_to_the_bound_range(monkeypatch) -> None:
    import os

    applied: list[set[int]] = []
    calls: list[str] = []
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(40)))
    monkeypatch.setattr(
        os, "sched_setaffinity", lambda _pid, cpus: applied.append(cpus)
    )
    monkeypatch.setattr(
        cpu_numa.sglang_bootstrap,
        "_init_cpu_threads_env",
        lambda **kwargs: calls.append("pinned"),
    )
    monkeypatch.setattr(
        cpu_numa.sglang_bootstrap, cpu_numa.WIDEN_HOOK_MARKER, False, raising=False
    )

    assert cpu_numa.install_binding_hook()
    assert not cpu_numa.install_binding_hook()
    cpu_numa.sglang_bootstrap._init_cpu_threads_env(
        local_omp_cpuid="0-3", dist_init_method=None
    )

    assert calls == ["pinned"]
    assert applied == [{0, 1, 2, 3}]


def _install_hook(monkeypatch, pinned: list) -> None:
    monkeypatch.setattr(
        cpu_numa.sglang_bootstrap,
        "_init_cpu_threads_env",
        lambda **kwargs: pinned.append(kwargs["local_omp_cpuid"]),
    )
    monkeypatch.setattr(
        cpu_numa.sglang_bootstrap, cpu_numa.WIDEN_HOOK_MARKER, False, raising=False
    )
    cpu_numa.install_binding_hook()


def test_binding_hook_pins_only_the_thread_budget(monkeypatch, nodes) -> None:
    import os

    pinned: list = []
    applied: list[set[int]] = []
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(40)))
    monkeypatch.setattr(
        os, "sched_setaffinity", lambda _pid, cpus: applied.append(cpus)
    )
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    _install_hook(monkeypatch, pinned)

    cpu_numa.sglang_bootstrap._init_cpu_threads_env(local_omp_cpuid="0-9")
    cpu_numa.sglang_bootstrap._init_cpu_threads_env(local_omp_cpuid=list(range(8)))
    monkeypatch.setenv("OMP_NUM_THREADS", "0")
    cpu_numa.sglang_bootstrap._init_cpu_threads_env(local_omp_cpuid="0-3")

    assert pinned == ["0,1,2,3", [0, 1, 2, 3], "0"]
    assert applied == [set(range(10)), set(range(8)), set(range(4))]


def test_binding_hook_keeps_a_multi_node_bind_whole(monkeypatch, nodes) -> None:
    import os

    pinned: list = []
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(40)))
    monkeypatch.setattr(os, "sched_setaffinity", lambda _pid, cpus: None)
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    _install_hook(monkeypatch, pinned)

    cpu_numa.sglang_bootstrap._init_cpu_threads_env(local_omp_cpuid="0-19")

    assert pinned == [",".join(map(str, range(20)))]


def test_binding_hook_never_pins_outside_the_allowed_cpus(monkeypatch) -> None:
    import os

    pinned: list = []
    applied: list[set[int]] = []
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1})
    monkeypatch.setattr(
        os, "sched_setaffinity", lambda _pid, cpus: applied.append(cpus)
    )
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    _install_hook(monkeypatch, pinned)

    cpu_numa.sglang_bootstrap._init_cpu_threads_env(local_omp_cpuid="0-3")
    assert pinned == ["0,1"]

    with pytest.raises(RuntimeError, match="outside the CPUs"):
        cpu_numa.sglang_bootstrap._init_cpu_threads_env(local_omp_cpuid="8-9")


def test_non_cpu_platforms_never_reach_the_cpu_hooks(monkeypatch) -> None:
    import os

    from sglang_omni.pipeline import stage_workers

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("CPU-only hook called on an accelerator")

    monkeypatch.setattr(
        stage_workers,
        "current_platform",
        SimpleNamespace(
            is_cpu=lambda: False,
            get_stage_process_env=lambda spec, env=None: {"STAGE": "1"},
            get_process_placement_env=_forbidden,
            get_process_cpu_mask=_forbidden,
            prepare_worker_process=_forbidden,
        ),
    )
    monkeypatch.setattr(stage_workers, "get_platform_spec", lambda _p: "gpu")

    plain = SimpleNamespace(
        process_name="p", stage_specs=[_stage("a")], cpu_threads=None
    )
    tp = SimpleNamespace(
        process_name="p", stage_specs=[_stage("a", tp_size=2)], cpu_threads=None
    )
    assert stage_workers.get_worker_process_env(plain) == {}
    assert stage_workers.get_worker_process_env(tp) == {"STAGE": "1"}

    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    spec = _spawn_spec({"OMP_NUM_THREADS": "192"})
    with stage_workers.patched_spawn_env(spec):
        assert os.environ["OMP_NUM_THREADS"] == "192"


def test_node_core_count_is_the_smallest_node(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cpu_numa, "numa_nodes", lambda: {0: "0-31,128-159", 1: "32-47"})

    assert cpu_numa.node_core_count() == 16


def test_cpu_thread_plan_uses_the_node_budget_not_the_host_share(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.pipeline import mp_runner
    from sglang_omni.pipeline.stage_workers import (
        StageLaunchConfig,
        StageWorkerProcessSpec,
    )

    monkeypatch.setattr(mp_runner, "effective_cpu_count", lambda: 384)
    monkeypatch.setattr(
        mp_runner,
        "current_platform",
        SimpleNamespace(
            get_process_thread_budget=lambda: 32,
            plan_process_placement=lambda _specs: None,
        ),
    )
    specs = [
        StageWorkerProcessSpec(f"p{i}", [StageLaunchConfig(f"s{i}")]) for i in range(7)
    ]

    mp_runner.apply_cpu_thread_plan([SimpleNamespace(process_specs=specs)])

    assert {spec.cpu_threads for spec in specs} == {32}


def test_planning_a_pipeline_forgets_earlier_placements(nodes) -> None:
    platform = CPUOmniPlatform()
    assert _bind(platform, "old", ["old"], {})[_BIND] == nodes[0]

    platform.plan_process_placement([])

    assert _bind(platform, "new", ["new"], {})[_BIND] == nodes[0]


def test_planning_reserves_nodes_a_later_worker_pins_explicitly(
    monkeypatch, nodes
) -> None:
    platform = CPUOmniPlatform()
    monkeypatch.delenv(_BIND, raising=False)
    pinned = _spawn_spec({_BIND: "0-9"})
    pinned.process_name = "pinned"

    platform.plan_process_placement([_spawn_spec({}), pinned])

    assert _bind(platform, "auto", ["a"], {})[_BIND] == nodes[1]


def test_planning_reserves_nodes_another_workers_numa_map_names(
    monkeypatch, nodes
) -> None:
    monkeypatch.delenv(_BIND, raising=False)
    monkeypatch.delenv(_MAP, raising=False)
    platform = CPUOmniPlatform()
    mapped = _spawn_spec({_MAP: "pinned=0"})
    mapped.process_name = "pinned"

    platform.plan_process_placement([_spawn_spec({}), mapped])

    assert _bind(platform, "auto", ["a"], {})[_BIND] == nodes[1]


def test_a_numa_map_beside_an_explicit_bind_reserves_nothing(
    monkeypatch, nodes
) -> None:
    monkeypatch.delenv(_BIND, raising=False)
    platform = CPUOmniPlatform()
    pinned = _spawn_spec({_MAP: "pinned=0", _BIND: "20-29"})
    pinned.process_name = "pinned"

    platform.plan_process_placement([pinned])

    assert _bind(platform, "auto", ["a"], {})[_BIND] == nodes[0]


def test_numa_nodes_use_the_node_column_not_cache_blocks(monkeypatch) -> None:
    import os
    import subprocess

    lscpu = "\n".join(
        ["# CPU,Core,Socket,Node"]
        + [f"{cpu},{cpu},0,{cpu // 4}" for cpu in range(8)]
        + [f"{cpu + 8},{cpu},0,{cpu // 4}" for cpu in range(8)]
    )
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: lscpu)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(16)))

    assert cpu_numa.numa_nodes() == {0: "0,1,2,3", 1: "4,5,6,7"}


def test_planning_gives_tp_groups_nodes_before_single_process_stages(
    monkeypatch, nodes
) -> None:
    from sglang_omni.pipeline.stage_workers import StageWorkerProcessSpec

    monkeypatch.delenv(_BIND, raising=False)
    platform = CPUOmniPlatform()

    def process(name: str, tp_size: int = 1, tp_rank: int = 0):
        stage = _stage(name.split("_")[0], tp_size, tp_rank)
        stage.env_defaults = {}
        return StageWorkerProcessSpec(process_name=name, stage_specs=[stage])

    singles = [process(name) for name in ("pre", "img", "audio")]
    ranks = [process(f"thinker_tp{r}", tp_size=2, tp_rank=r) for r in range(2)]
    platform.plan_process_placement(singles + ranks)

    single_masks = [
        _bind(platform, spec.process_name, [spec.stage_specs[0].stage_name], {})[_BIND]
        for spec in singles
    ]
    rank_masks = [
        _bind(platform, f"thinker_tp{r}", ["thinker"], {}, tp_size=2, tp_rank=r)[_BIND]
        for r in range(2)
    ]

    assert rank_masks == nodes[:2]
    assert single_masks == [nodes[2], nodes[3], nodes[0]]


def test_thread_plan_node_budget_respects_the_cpu_quota(monkeypatch) -> None:
    from sglang_omni.pipeline import mp_runner
    from sglang_omni.pipeline.stage_workers import (
        StageLaunchConfig,
        StageWorkerProcessSpec,
    )

    monkeypatch.setattr(mp_runner, "effective_cpu_count", lambda: 28)
    monkeypatch.setattr(
        mp_runner,
        "current_platform",
        SimpleNamespace(
            get_process_thread_budget=lambda: 64,
            plan_process_placement=lambda _specs: None,
        ),
    )
    specs = [
        StageWorkerProcessSpec(f"p{i}", [StageLaunchConfig(f"s{i}")]) for i in range(7)
    ]

    mp_runner.apply_cpu_thread_plan([SimpleNamespace(process_specs=specs)])

    assert {spec.cpu_threads for spec in specs} == {4}
