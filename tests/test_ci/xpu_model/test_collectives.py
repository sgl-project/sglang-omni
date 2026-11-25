# SPDX-License-Identifier: Apache-2.0
"""An all-reduce captured in an XPU graph must sum on replay, or every TP>1 decode graph answers wrong.

pytest tests/test_ci/xpu_model/test_collectives.py -m xpu_serving -v
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.test_ci.xpu_model.captured_all_reduce import ProbeResult
from tests.test_ci.xpu_model.conftest import ClaimCards, free_port

pytestmark = [pytest.mark.accelerator, pytest.mark.xpu_serving, pytest.mark.xpu_tp]

REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE = "tests.test_ci.xpu_model.captured_all_reduce"


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def stop_group(proc: subprocess.Popen[str]) -> None:
    """End the launcher and every rank in its process group, SIGTERM first, before the cards are released."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        proc.poll()
        if not group_alive(proc.pid):
            break
        else:
            os.killpg(proc.pid, sig)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            proc.poll()
            if not group_alive(proc.pid):
                break
            else:
                time.sleep(0.5)
    proc.wait()


def probe(cards: list[int]) -> ProbeResult:
    env = dict(os.environ, ZE_AFFINITY_MASK=",".join(str(card) for card in cards))
    proc = subprocess.Popen(
        [sys.executable, "-m", PROBE, str(len(cards)), str(free_port())],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"the probe did not finish within 600 s on cards {cards}: a collective hung"
        )
    finally:
        stop_group(proc)
    lines = [line for line in stdout.splitlines() if line.startswith("RESULT ")]
    assert (
        proc.returncode == 0 and lines
    ), f"the probe exited {proc.returncode}:\n{stdout[-2000:]}\n{stderr[-4000:]}"
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.mark.parametrize("world_size", [2, 4])
def test_an_all_reduce_captured_in_an_xpu_graph_sums_on_replay(
    world_size: int, xpu_host: None, claim_cards: ClaimCards
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
