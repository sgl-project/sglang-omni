# SPDX-License-Identifier: Apache-2.0
"""A tensor-parallel decode graph is only right if its all-reduces replay.

When oneCCL fails to record an all-reduce inside an XPU graph, the replay is a no-op:
each rank keeps its own partial sum, so a TP>1 model answers its first token from the
eager prefill and garbage after it. This checks the collective itself, in seconds and
with no checkpoint, on as many cards as a tensor-parallel row uses.

    pytest tests/test_ci/xpu_model/test_collectives.py -m xpu_serving -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_ci.xpu_model.conftest import free_port

pytestmark = [pytest.mark.accelerator, pytest.mark.xpu_serving, pytest.mark.xpu_tp]

REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE = "tests.test_ci.xpu_model.captured_all_reduce"


def probe(cards: list[int]) -> dict:
    env = dict(os.environ, ZE_AFFINITY_MASK=",".join(str(card) for card in cards))
    proc = subprocess.run(
        [sys.executable, "-m", PROBE, str(len(cards)), str(free_port())],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    assert (
        proc.returncode == 0 and lines
    ), f"the probe exited {proc.returncode}:\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.mark.parametrize("world_size", [2, 4])
def test_an_all_reduce_captured_in_an_xpu_graph_sums_on_replay(
    world_size: int, xpu_host: None, claim_cards
) -> None:
    result = probe(claim_cards(world_size))
    want = [world_size * (world_size + 1) / 2]

    def wrong(phase: str) -> dict[str, list[float]]:
        return {
            f"rank {rank} {size}": held[size][phase]
            for rank, held in enumerate(result["ranks"])
            for size in held
            if held[size][phase] != want
        }

    assert not wrong(
        "eager"
    ), f"an eager all-reduce did not sum to {want[0]}: {wrong('eager')}"
    assert not wrong("replay"), (
        f"an all-reduce captured in an XPU graph replayed without summing (want {want[0]}): "
        f"{wrong('replay')}. oneCCL {result['oneccl']} ({result['libccl']}) did not record it, "
        "so every TP>1 decode graph on this stack gives wrong output. oneCCL 2022.1.1, the "
        "version torch 2.14.0+xpu pins, drops it; 2022.1.2 records it."
    )
