# SPDX-License-Identifier: Apache-2.0
import pytest

from sglang_omni.cpu_alloc.allocator import ProcessCpuDemand, allocate
from sglang_omni.cpu_alloc.topology import discover_topology


@pytest.fixture
def topology(dual_node_sysfs):
    return discover_topology(range(16), sysfs_root=dual_node_sysfs)


def demand(name, node=0, serial=0):
    return ProcessCpuDemand(process_name=name, numa_node=node, exclusive_cores=serial)


class TestAllocate:
    def test_exclusive_gets_whole_physical_cores(self, topology):
        plan = allocate(topology, [demand("ar", serial=1), demand("shared")])
        ar = plan.assignments["ar"]
        assert ar.exclusive
        assert ar.cpu_ids == (0, 8)  # both SMT siblings of one core
        assert plan.events == ()

    def test_shared_process_gets_node_remainder(self, topology):
        plan = allocate(topology, [demand("ar", serial=1), demand("shared")])
        shared = plan.assignments["shared"]
        assert not shared.exclusive
        assert shared.cpu_ids == (1, 2, 3, 9, 10, 11)
        assert 0 not in shared.cpu_ids and 8 not in shared.cpu_ids

    def test_numa_anchoring(self, topology):
        plan = allocate(
            topology,
            [demand("a", node=0, serial=1), demand("b", node=1, serial=1), demand("s")],
        )
        assert plan.assignments["a"].cpu_ids == (0, 8)
        assert plan.assignments["b"].cpu_ids == (4, 12)
        assert plan.assignments["a"].numa_node == 0
        assert plan.assignments["b"].numa_node == 1

    def test_unknown_numa_stays_shared_with_event(self, topology):
        # A guessed anchor could pin a GPU process to the wrong socket, so an
        # unresolvable node keeps today's shared behavior instead.
        plan = allocate(topology, [demand("a", node=7, serial=1)])
        assert not plan.assignments["a"].exclusive
        assert any("node 7" in event for event in plan.events)

    def test_unanchored_demands_spread_across_nodes(self, topology):
        plan = allocate(
            topology,
            [demand("a", node=None, serial=1), demand("b", node=None, serial=1)],
        )
        assert plan.assignments["a"].exclusive
        assert plan.assignments["b"].exclusive
        assert {
            plan.assignments["a"].numa_node,
            plan.assignments["b"].numa_node,
        } == {0, 1}

    def test_demand_over_node_budget_moves_to_shared(self, topology):
        # Node 0 has 4 physical cores; 1 stays shared, so the budget is 3.
        plan = allocate(topology, [demand("big", serial=4), demand("small", serial=1)])
        assert not plan.assignments["big"].exclusive
        assert plan.assignments["small"].exclusive
        assert any("moved to the shared pool" in e for e in plan.events)

    def test_overflow_degrades_and_is_logged(self, topology):
        # One core is held for the shared tenant, so 3 of the 5 fit.
        plan = allocate(
            topology,
            [demand(f"s{i}", serial=1) for i in range(5)] + [demand("shared")],
        )
        exclusive = [n for n, a in plan.assignments.items() if a.exclusive]
        shared = [n for n, a in plan.assignments.items() if not a.exclusive]
        assert len(exclusive) == 3 and len(shared) == 3
        assert any("moved to the shared pool" in e for e in plan.events)

    def test_exclusive_grants_are_disjoint(self, topology):
        plan = allocate(
            topology,
            [demand("a", serial=1), demand("b", serial=1), demand("c", serial=1)],
        )
        seen: set[int] = set()
        for assignment in plan.assignments.values():
            if assignment.exclusive:
                assert not (seen & set(assignment.cpu_ids))
                seen.update(assignment.cpu_ids)

    def test_pool_is_reserved_only_when_a_tenant_exists(self, topology):
        with_tenant = allocate(topology, [demand("ar", serial=1), demand("frontend")])
        assert with_tenant.shared_pools[0]

        # Nobody would use the leftover, so the holder takes the whole node.
        alone = allocate(topology, [demand("ar", serial=1)])
        assert alone.shared_pools[0] == ()
        assert len(alone.assignments["ar"].cpu_ids) == 8

    def test_deterministic(self, topology):
        demands = [demand("b", serial=1), demand("a", serial=1), demand("z")]
        first = allocate(topology, demands)
        second = allocate(topology, list(reversed(demands)))
        assert first.to_dict() == second.to_dict()

    def test_duplicate_names_raise(self, topology):
        with pytest.raises(ValueError, match="Duplicate"):
            allocate(topology, [demand("a"), demand("a")])

    def test_negative_demand_raises(self, topology):
        with pytest.raises(ValueError, match="must be >= 0"):
            ProcessCpuDemand(process_name="a", numa_node=0, exclusive_cores=-1)

    def test_no_anchor_shared_gets_union(self, topology):
        plan = allocate(topology, [demand("cpuonly", node=None)])
        assert plan.assignments["cpuonly"].cpu_ids == tuple(range(16))

    def test_to_dict_shape(self, topology):
        plan = allocate(topology, [demand("ar", serial=1), demand("frontend")])
        data = plan.to_dict()
        assert data["assignments"]["ar"] == {"cpus": "0,8", "exclusive": True}
        assert "0" in data["shared_pools"] and "None" in data["shared_pools"]


class TestDemotionKeepsExclusivityHonest:
    def test_demoted_process_gets_a_real_mask(self, topology):
        # It must not fall through with an empty mask: the caller skips empty
        # masks, so the process would keep the whole cpuset and land on the
        # cores the survivors were promised.
        plan = allocate(topology, [demand("big", serial=4), demand("small", serial=1)])
        big = plan.assignments["big"]
        small = plan.assignments["small"]
        assert big.cpu_ids
        assert not set(big.cpu_ids) & set(small.cpu_ids)

    def test_demoted_process_gets_a_mask_when_every_node_is_taken(self, topology):
        # Both nodes are fully claimed, so the demoted process has nowhere to
        # fall back to unless the pool is reserved for it.
        plan = allocate(
            topology,
            [
                demand("a", node=0, serial=4),
                demand("b", node=1, serial=4),
                demand("c", node=0, serial=1),
            ],
        )
        owned = {
            cpu for a in plan.assignments.values() if a.exclusive for cpu in a.cpu_ids
        }
        shared = [a for a in plan.assignments.values() if not a.exclusive]
        assert shared, "the case is only interesting when somebody is demoted"
        for a in shared:
            assert a.cpu_ids, f"{a.process_name} got an empty mask"
            assert not set(a.cpu_ids) & owned

    def test_every_demoted_process_stays_off_exclusive_cores(self, topology):
        plan = allocate(
            topology,
            [demand(f"s{i}", serial=1) for i in range(5)] + [demand("shared")],
        )
        owned = {
            cpu for a in plan.assignments.values() if a.exclusive for cpu in a.cpu_ids
        }
        for name, a in plan.assignments.items():
            if a.exclusive:
                continue
            assert a.cpu_ids, f"{name} got an empty mask"
            assert not set(a.cpu_ids) & owned, f"{name} overlaps an exclusive grant"

    def test_a_demotion_frees_the_capacity_it_asked_for(self, topology):
        # "huge" cannot fit anywhere; the cores it was tried against must stay
        # available to the demands behind it.
        plan = allocate(
            topology,
            [demand("huge", node=None, serial=9)]
            + [demand(f"s{i}", node=None, serial=3) for i in range(2)],
        )
        assert not plan.assignments["huge"].exclusive
        assert all(plan.assignments[f"s{i}"].exclusive for i in range(2))


class TestDeclaredCoresAreCountedInCores:
    def test_smt_siblings_do_not_double_the_count(self, topology):
        # The starvation trigger compares this against a stage's declaration,
        # so counting cpu ids would read every declaration as twice as large.
        plan = allocate(topology, [demand("a", serial=2), demand("shared")])
        assert len(plan.assignments["a"].cpu_ids) == 4
        assert plan.exclusive_physical_cores == 2

    def test_a_demoted_demand_is_not_counted(self, topology):
        plan = allocate(topology, [demand("big", serial=9), demand("shared")])
        assert not plan.assignments["big"].exclusive
        assert plan.exclusive_physical_cores == 0
