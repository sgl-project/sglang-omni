# SPDX-License-Identifier: Apache-2.0
"""Regression test for the async-decode gate, and the home of its methodology.

The S1-S5 ON/OFF gate is a GPU integration harness (server launcher, dump and
counter hooks, concurrent driver, ABAB runner): it needs a live serving stack and
GPUs, so it cannot run in CI and lives as operational tooling off-tree. This
module is the in-repo source of truth for the gate: it absorbs the CPU-testable
essence (the scenario table, the comparison and counter logic) and records the
methodology those scenarios implement.

Methodology
-----------
Two-tier criteria. The gate compares the vocoder-entry ``audio_codes`` of an ON
arm (``--decode-mode async``) against an OFF arm (sync), per request, keyed by
seed. Only S1 is checked for exact bit-identity: it is ``bs=1`` with
``min_batch_size=1``, so it is both deterministic (a single request has no
batch-composition variance) and lookahead-forced, which is exactly why it is the
load-bearing anchor and the test that caught the real bs=1 stop-boundary bug. The
multi-request scenarios (S2 steady batch, S3a/S3b finish transitions, S4
retraction, S5 rep-penalty routing) are structural: no crash, every seed
produces a dump, the lookahead engaged, every request finished.

Why concurrency precludes multi-request bit-identity. Engaging the lookahead
needs a real batch, which needs requests sent concurrently; but then per-step
batch composition is timing-dependent, so the CUDA-graph bucket trajectory varies
run to run, perturbing the low bits of each request's hidden states, which the
binary stop head amplifies into different stop frames (different frame counts).
So exact bit-identity and real-concurrency lookahead coverage are mutually
exclusive for multi-request scenarios; only ``bs=1`` (S1) gets both.

OFF-vs-OFF empirical basis. This is not a lookahead defect: an OFF-vs-OFF control
(two pure-sync concurrent runs of the same seeds, zero lookahead) already differs
in frame count and values (e.g. seed 1002: 49 vs 36 frames; seed 1000: same frame
count, different values). A lookahead defect would instead show ON != OFF while
OFF-A == OFF-B. The raw drift dumps are retained off-tree for review.

Counter methodology. Concurrency is necessary but not self-evident, so every ON
arm also asserts the lookahead hit/miss counter (written from inside
execute_resolve): every lookahead scenario must show ``hit+miss > 0`` and the
rep-penalty scenario must show ``0`` (its batch routes wholly to sync). A
bit-identity match without engagement is not accepted, because a sequential
driver kept the server at ``bs=1`` (below ``async_decode_min_batch_size`` for
every scenario but S1), silently ran the ON arm through the sync fast path, and
"proved" bit-identity of sync against sync.

This test locks three things a refactor must not silently break: the criteria
tiering, that the rep-penalty scenario actually carries a penalty request (else
its ``counter == 0`` check is vacuous), and that the comparison flags shape AND
value drift while the counter assertion distinguishes engaged from sync-routed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pytest


@dataclass
class Req:
    text: str
    seed: int
    max_new_tokens: int | None = None
    repetition_penalty: float = 1.0


@dataclass
class Scenario:
    key: str
    criterion: str  # "exact" | "structural"
    requests: list[Req] = field(default_factory=list)
    expect_lookahead: bool = True  # False only for the rep-penalty sync route


_TEXT = "The quick brown fox jumps over the lazy dog."

# The scenario contract (mirrors the operational driver's SCENARIOS table). S1 is
# the only exact scenario: bs=1 with min_batch_size=1 is both deterministic and
# lookahead-forced. S2-S5 are structural; S5 expects no lookahead (sync routing).
SCENARIOS: dict[str, Scenario] = {
    "S1": Scenario("S1", "exact", [Req(_TEXT, 1234, 64)]),
    "S2": Scenario("S2", "structural", [Req(_TEXT, 1000 + i, 48) for i in range(4)]),
    "S3a": Scenario(
        "S3a",
        "structural",
        [Req(_TEXT, 2000 + i, 64) for i in range(6)] + [Req(_TEXT, 2099, 16)],
    ),
    "S3b": Scenario("S3b", "structural", [Req(_TEXT, 3000, 64), Req(_TEXT, 3001, 32)]),
    "S4": Scenario("S4", "structural", [Req(_TEXT, 4000 + i, 64) for i in range(4)]),
    "S5": Scenario(
        "S5",
        "structural",
        [Req(_TEXT, 5000, 48, 1.0), Req(_TEXT, 5001, 48, 1.3)],
        expect_lookahead=False,
    ),
}


def compare_exact(off_dir: str, on_dir: str, rids: list[str]) -> list[str]:
    """Return the rids whose ON/OFF audio_codes differ (empty == pass)."""
    diffs = []
    for rid in rids:
        a = np.load(f"{off_dir}/{rid}.npy")
        b = np.load(f"{on_dir}/{rid}.npy")
        if a.shape != b.shape or not np.array_equal(a, b):
            diffs.append(rid)
    return diffs


def read_counters(counter_file: str) -> tuple[int, int]:
    """(hit, miss) from the gate counter dump, or (0, 0) if absent. Absent is the
    meaningful zero: the dump is written only from inside execute_resolve, so a
    scenario that never engages lookahead leaves no file."""
    if not counter_file or not os.path.exists(counter_file):
        return 0, 0
    with open(counter_file) as f:
        parts = f.read().split()
    if len(parts) != 2:
        raise ValueError(f"invalid counter file {counter_file!r}: {parts!r}")
    return int(parts[0]), int(parts[1])


def counters_ok(sc: Scenario, counter_file: str) -> bool:
    hit, miss = read_counters(counter_file)
    total = hit + miss
    return total > 0 if sc.expect_lookahead else total == 0


# --------------------------------------------------------------------------- #


def test_only_s1_is_exact_rest_structural():
    assert SCENARIOS["S1"].criterion == "exact"
    for key in ("S2", "S3a", "S3b", "S4", "S5"):
        assert SCENARIOS[key].criterion == "structural", key


def test_only_s5_routes_sync():
    # The lookahead counter is the engagement guarantee: every lookahead scenario
    # must engage (>0); only the rep-penalty scenario must not (==0).
    assert SCENARIOS["S5"].expect_lookahead is False
    for key in ("S1", "S2", "S3a", "S3b", "S4"):
        assert SCENARIOS[key].expect_lookahead is True, key


def test_s5_actually_carries_a_rep_penalty_request():
    # If the penalty did not reach a request, S5 would lookahead-batch instead of
    # routing sync, and the counter==0 check would be vacuous.
    penalties = [r.repetition_penalty for r in SCENARIOS["S5"].requests]
    assert any(p != 1.0 for p in penalties)


@pytest.mark.parametrize("key", list(SCENARIOS))
def test_scenario_seeds_are_unique(key):
    # Dumps are keyed by seed (concurrency-safe), so seeds must not collide.
    seeds = [r.seed for r in SCENARIOS[key].requests]
    assert len(seeds) == len(set(seeds))


def test_compare_exact_flags_shape_and_value_drift(tmp_path):
    off = tmp_path / "off"
    on = tmp_path / "on"
    off.mkdir()
    on.mkdir()
    base = np.arange(36 * 12, dtype=np.int64).reshape(36, 12)
    np.save(off / "1.npy", base)
    np.save(on / "1.npy", base.copy())  # identical
    np.save(off / "2.npy", base)
    np.save(on / "2.npy", base[:-1])  # shorter -> shape drift
    np.save(off / "3.npy", base)
    v = base.copy()
    v[10, 5] += 1
    np.save(on / "3.npy", v)  # one value differs

    assert compare_exact(str(off), str(on), ["1"]) == []
    assert compare_exact(str(off), str(on), ["1", "2", "3"]) == ["2", "3"]


def test_counter_assertion_engaged_vs_sync(tmp_path):
    f = tmp_path / "counters.txt"
    s2, s5 = SCENARIOS["S2"], SCENARIOS["S5"]

    # No file: engaged scenario fails, sync-routed scenario passes.
    assert read_counters(str(f)) == (0, 0)
    assert counters_ok(s2, str(f)) is False
    assert counters_ok(s5, str(f)) is True

    # Lookahead ran: engaged passes, sync-routed fails.
    f.write_text("39 0")
    assert read_counters(str(f)) == (39, 0)
    assert counters_ok(s2, str(f)) is True
    assert counters_ok(s5, str(f)) is False


def test_malformed_counter_file_fails(tmp_path):
    f = tmp_path / "counters.txt"
    f.write_text("39")

    with pytest.raises(ValueError, match="invalid counter file"):
        read_counters(str(f))
