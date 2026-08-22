# SPDX-License-Identifier: Apache-2.0
"""CLI behavior the DP launchers depend on.

Kept out of ``test_integration.py`` so it runs without the serving stack: the
planner only needs topology discovery.
"""

import pytest

from sglang_omni.cpu_alloc import __main__ as cli
from sglang_omni.cpu_alloc.topology import discover_topology


@pytest.fixture
def planner(monkeypatch, dual_node_sysfs):
    monkeypatch.setattr(
        cli,
        "discover_topology",
        lambda: discover_topology(range(16), sysfs_root=dual_node_sysfs),
    )
    monkeypatch.setattr(cli, "gpu_numa_nodes", lambda ids: {list(ids)[0]: 1})
    return cli


class TestPlannerRefusalIsDistinct:
    def test_capacity_refusal_exits_3(self, planner, capsys):
        # autodp.sh only falls back to its NUMA-blind split when the planner is
        # absent, so a refusal must be told apart from a crash.
        code = planner.main(
            ["plan", "--replicas", "5", "--gpu-id", "0", "--format", "blocks"]
        )
        assert code == 3
        assert "physical cores" in capsys.readouterr().err

    def test_success_still_exits_0(self, planner, capsys):
        code = planner.main(
            ["plan", "--replicas", "3", "--gpu-id", "0", "--format", "blocks"]
        )
        assert code == 0
        assert capsys.readouterr().out.strip() == "4,12 5,13 6,14"
