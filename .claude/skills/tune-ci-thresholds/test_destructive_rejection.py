"""Exercise tune.py's destructive-rejection control flow without GPUs.

Only `_run_shared` (the pytest launcher) and the GPU gate are stubbed; the
detection, marking, replenishment, restart, and cap logic all run for real.
"""
import argparse
import importlib.util
import json
import pathlib
import shutil
import sys
import types

REPO = pathlib.Path("/data/chenyang/sglang-omni")
TUNE = REPO / ".claude/skills/tune-ci-thresholds/tune.py"
SB = REPO / ".tune-runs/_loop_test"


def load():
    spec = importlib.util.spec_from_file_location("tunemod", TUNE)
    m = importlib.util.module_from_spec(spec)
    sys.modules["tunemod"] = m
    spec.loader.exec_module(m)
    return m


def make_args(out, repeats=5, no_reject=False, stages="mmsu_speed"):
    return argparse.Namespace(
        model="omni", stages=stages, repeats=repeats, output_dir=str(out),
        resume=False, skip_precheck=True, stages_yaml=None, max_passes=3,
        no_destructive_rejection=no_reject, venv_python=None, host_profile=None,
    )


def run_scenario(name, bad_rounds, repeats=5, no_reject=False, expect=None):
    """bad_rounds: set of round indices whose synthetic metrics are destroyed."""
    t = load()
    if SB.exists():
        shutil.rmtree(SB)
    SB.mkdir(parents=True)

    calls = []

    def fake_run_shared(test_path, stage_keys, all_stages, out, k, py, total,
                        gpus_needed, extra_args, host=None):
        calls.append(k)
        if len(calls) > 40:
            raise RuntimeError("runaway loop — control flow does not terminate")
        for sk in stage_keys:
            stage = all_stages[sk]
            metrics = {}
            for i, mk in enumerate(stage["metrics"]):
                base = 10.0 + i
                metrics[mk] = base * 0.25 if k in bad_rounds else base + (k % 3) * 0.01
            (out / sk).mkdir(parents=True, exist_ok=True)
            (out / sk / f"run{k}.json").write_text(json.dumps({
                "status": "ok", "reason": "", "metrics": metrics,
                "sample_counts": {"total": stage.get("expected_samples") or 10,
                                  "ok": stage.get("expected_samples") or 10},
                "git_sha": t.git_info()["sha"], "attempts": 1,
            }, indent=2))
        return 0

    t._run_shared = fake_run_shared
    t._ensure_gpus_free = lambda *a, **kw: True
    t._stage_gpus = lambda *a, **kw: 1

    cfg = t.load_model_config("omni")
    args = make_args(SB, repeats=repeats, no_reject=no_reject)
    buf = []
    real_print = print
    try:
        rc = t._run_cmd_inner(args, cfg, "python", "test", SB)
    except RuntimeError as e:
        print(f"  !! {e}")
        return False

    stages = t._load_yaml(t.stages_path("omni"))
    plan = json.loads((SB / "plan.json").read_text())
    sk = plan["stages"][0]
    rounds = t._round_indices(SB, [sk])
    dest = [k for k in rounds if t._run_json_destructive(SB / sk / f"run{k}.json")]
    clean = t._clean_rounds(SB, sk, stages[sk])
    got = dict(rounds=rounds, destructive=dest, clean=len(clean), rc=rc)
    ok = all(got[k] == v for k, v in (expect or {}).items())
    print(f"  {name}")
    print(f"    rounds={rounds} destructive={dest} clean={len(clean)} rc={rc}")
    if expect:
        print(f"    expected {expect} -> {'PASS' if ok else 'FAIL'}")
    return ok


def run_resume_scenario():
    """Kill the run right after a round is condemned, then resume.

    This is the shape a 2-hour session teardown produces: the mark is on disk
    but its replacement rounds were never launched.
    """
    t = load()
    if SB.exists():
        shutil.rmtree(SB)
    SB.mkdir(parents=True)
    stages = t._load_yaml(t.stages_path("omni"))
    cfg = t.load_model_config("omni")

    # --- phase 1: 5 rounds, r3 broken, killed before replenishment ---
    launched = []

    def fake_a(test_path, stage_keys, all_stages, out, k, py, total, gpus_needed,
               extra_args, host=None):
        launched.append(k)
        for sk in stage_keys:
            st = all_stages[sk]
            m = {mk: (10.0 + i) * (0.25 if k == 3 else 1.0) + (0 if k == 3 else k * 0.01)
                 for i, mk in enumerate(st["metrics"])}
            (out / sk).mkdir(parents=True, exist_ok=True)
            (out / sk / f"run{k}.json").write_text(json.dumps({
                "status": "ok", "reason": "", "metrics": m,
                "sample_counts": {"total": st.get("expected_samples") or 10,
                                  "ok": st.get("expected_samples") or 10},
                "git_sha": t.git_info()["sha"], "attempts": 1}, indent=2))
        # Simulate the teardown: die as soon as the 5 baseline rounds exist.
        if len(launched) >= 5:
            raise KeyboardInterrupt("session teardown")
        return 0

    t._run_shared = fake_a
    t._ensure_gpus_free = lambda *a, **kw: True
    t._stage_gpus = lambda *a, **kw: 1
    try:
        t._run_cmd_inner(make_args(SB), cfg, "python", "test", SB)
    except KeyboardInterrupt:
        pass
    # Mark r3 the way the loop would have, then stop — replacements never ran.
    sk0 = "mmsu_speed"
    unit = [sk0]
    ev = t._detect_destructive_rounds(SB, unit, stages)
    t._mark_destructive_rounds(SB, unit, ev)
    before = dict(rounds=t._round_indices(SB, unit),
                  dest=[k for k in t._round_indices(SB, unit)
                        if t._run_json_destructive(SB / sk0 / f"run{k}.json")],
                  clean=len(t._clean_rounds(SB, sk0, stages[sk0])))
    print(f"  after teardown: {before}")

    # --- phase 2: resume ---
    t2 = load()

    def fake_b(test_path, stage_keys, all_stages, out, k, py, total, gpus_needed,
               extra_args, host=None):
        for sk in stage_keys:
            st = all_stages[sk]
            m = {mk: (10.0 + i) + k * 0.01 for i, mk in enumerate(st["metrics"])}
            (out / sk / f"run{k}.json").write_text(json.dumps({
                "status": "ok", "reason": "", "metrics": m,
                "sample_counts": {"total": st.get("expected_samples") or 10,
                                  "ok": st.get("expected_samples") or 10},
                "git_sha": t2.git_info()["sha"], "attempts": 1}, indent=2))
        return 0

    t2._run_shared = fake_b
    t2._ensure_gpus_free = lambda *a, **kw: True
    t2._stage_gpus = lambda *a, **kw: 1
    args = make_args(SB); args.resume = True
    rc = t2._run_cmd_inner(args, t2.load_model_config("omni"), "python", "test", SB)
    rounds = t2._round_indices(SB, unit)
    dest = [k for k in rounds if t2._run_json_destructive(SB / sk0 / f"run{k}.json")]
    clean = len(t2._clean_rounds(SB, sk0, stages[sk0]))
    ok = clean >= 5 and dest == [3] and rc == 0
    print(f"  after resume:   rounds={rounds} destructive={dest} clean={clean} rc={rc}")
    print(f"  -> {'PASS' if ok else 'FAIL (resume did not replenish)'}")
    shutil.rmtree(SB, ignore_errors=True)
    return ok



def _writer(t, bad=(), skip=()):
    """Fake pytest: writes clean metrics unless the round is in `bad`."""
    def w(test_path, stage_keys, all_stages, out, k, py, total, gpus_needed,
          extra_args, host=None):
        if k in skip:
            return 0                      # simulate a round that never landed
        for sk in stage_keys:
            st = all_stages[sk]
            m = {mk: (10.0 + i) * (0.25 if k in bad else 1.0)
                     + (0 if k in bad else k * 0.01)
                 for i, mk in enumerate(st["metrics"])}
            (out / sk).mkdir(parents=True, exist_ok=True)
            (out / sk / f"run{k}.json").write_text(json.dumps({
                "status": "ok", "reason": "", "metrics": m,
                "sample_counts": {"total": st.get("expected_samples") or 10,
                                  "ok": st.get("expected_samples") or 10},
                "git_sha": t.git_info()["sha"], "attempts": 1}, indent=2))
        return 0
    return w


def _state(t, sk="mmsu_speed"):
    stages = t._load_yaml(t.stages_path("omni"))
    rounds = t._round_indices(SB, [sk])
    dest = [k for k in rounds if t._run_json_destructive(SB / sk / f"run{k}.json")]
    return rounds, dest, len(t._clean_rounds(SB, sk, stages[sk]))


def two_phase(name, phase1_bad, phase1_stop_after, phase2_bad=(), mutate=None,
              expect_rc=0, expect_min_clean=5):
    """Run, kill it partway, then resume into the same directory."""
    if SB.exists():
        shutil.rmtree(SB)
    SB.mkdir(parents=True)
    t = load()
    n = {"c": 0}
    base = _writer(t, bad=phase1_bad)

    def dying(*a, **kw):
        n["c"] += 1
        r = base(*a, **kw)
        if n["c"] >= phase1_stop_after:
            raise KeyboardInterrupt("teardown")
        return r

    t._run_shared = dying
    t._ensure_gpus_free = lambda *a, **kw: True
    t._stage_gpus = lambda *a, **kw: 1
    try:
        t._run_cmd_inner(make_args(SB), t.load_model_config("omni"),
                         "python", "test", SB)
    except KeyboardInterrupt:
        pass
    if mutate:
        mutate(t)
    r0, d0, c0 = _state(t)
    t2 = load()
    t2._run_shared = _writer(t2, bad=phase2_bad)
    t2._ensure_gpus_free = lambda *a, **kw: True
    t2._stage_gpus = lambda *a, **kw: 1
    args = make_args(SB); args.resume = True
    rc = t2._run_cmd_inner(args, t2.load_model_config("omni"), "python", "test", SB)
    r1, d1, c1 = _state(t2)
    ok = (rc == expect_rc) and (c1 >= expect_min_clean if expect_rc == 0 else True)
    print(f"  {name}")
    print(f"    killed at: rounds={r0} dest={d0} clean={c0}")
    print(f"    resumed  : rounds={r1} dest={d1} clean={c1} rc={rc}")
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    shutil.rmtree(SB, ignore_errors=True)
    return ok


def resume_wrong_commit():
    """Resuming onto a different commit must be refused, not silently mixed."""
    if SB.exists():
        shutil.rmtree(SB)
    SB.mkdir(parents=True)
    t = load()
    t._run_shared = _writer(t)
    t._ensure_gpus_free = lambda *a, **kw: True
    t._stage_gpus = lambda *a, **kw: 1
    t._run_cmd_inner(make_args(SB), t.load_model_config("omni"), "python", "test", SB)
    plan = json.loads((SB / "plan.json").read_text())
    plan["calibration_git_sha"] = "0" * 40
    plan["git_sha"] = "0" * 40
    (SB / "plan.json").write_text(json.dumps(plan, indent=2))
    t2 = load()
    t2._run_shared = _writer(t2)
    t2._ensure_gpus_free = lambda *a, **kw: True
    t2._stage_gpus = lambda *a, **kw: 1
    args = make_args(SB); args.resume = True
    rc = t2._run_cmd_inner(args, t2.load_model_config("omni"), "python", "test", SB)
    ok = rc != 0
    print(f"  resume across commits refused: rc={rc} -> {'PASS' if ok else 'FAIL (mixed commits!)'}")
    shutil.rmtree(SB, ignore_errors=True)
    return ok


def resume_no_flag_refused():
    """Re-entering an existing dir WITHOUT --resume must hard-error."""
    if SB.exists():
        shutil.rmtree(SB)
    SB.mkdir(parents=True)
    t = load()
    t._run_shared = _writer(t)
    t._ensure_gpus_free = lambda *a, **kw: True
    t._stage_gpus = lambda *a, **kw: 1
    t._run_cmd_inner(make_args(SB), t.load_model_config("omni"), "python", "test", SB)
    t2 = load()
    t2._run_shared = _writer(t2)
    t2._ensure_gpus_free = lambda *a, **kw: True
    t2._stage_gpus = lambda *a, **kw: 1
    rc = t2._run_cmd_inner(make_args(SB), t2.load_model_config("omni"),
                           "python", "test", SB)
    ok = rc != 0
    print(f"  re-entry without --resume refused: rc={rc} -> {'PASS' if ok else 'FAIL'}")
    shutil.rmtree(SB, ignore_errors=True)
    return ok



def unit_edge_cases():
    """Detector edge cases that never reach the control flow."""
    import math
    t = load()
    ok = True

    def chk(label, cond):
        nonlocal ok
        ok = ok and cond
        print(f"    {'PASS' if cond else 'FAIL'}  {label}")

    d = t._destructive_value_indices
    chk("clean sample flags nothing", not d([10., 10.1, 9.9, 10.05, 9.95]))
    chk("single outlier flagged", sorted(d([10., 10.1, 9.9, 10.05, 2.5])) == [4])
    chk("identical values flag nothing", not d([5.] * 5))
    chk("integer count 1,1,1,1,2 protected by floor",
        not d([1., 1., 1., 1., 2.], 1.0))
    chk("NaN condemned with a readable reason",
        d([1., 2., 3., 4., math.nan]).get(4, {}).get("reason") == "non-finite value")
    chk("inf condemned with a readable reason",
        d([1., 2., 3., 4., math.inf]).get(4, {}).get("reason") == "non-finite value")
    chk("below MIN_OBS declines to judge", not d([10., 2.5, 10.]))
    chk("degenerate 3v2 split detected",
        t._degenerate_series([2.5, 2.5, 2.5, 10., 10.1]) is not None)
    chk("mild bimodality is NOT degenerate",
        t._degenerate_series([4.74, 4.80, 5.00, 6.73, 6.79]) is None)
    chk("lone outlier is NOT degenerate",
        t._degenerate_series([10., 10.1, 9.9, 10.05, 2.5]) is None)
    return ok


def repeats_below_min_warns():
    """--repeats < MIN_OBS must disable rejection loudly, not silently."""
    import io, contextlib
    if SB.exists():
        shutil.rmtree(SB)
    SB.mkdir(parents=True)
    t = load()
    t._run_shared = _writer(t, bad={2})
    t._ensure_gpus_free = lambda *a, **kw: True
    t._stage_gpus = lambda *a, **kw: 1
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        t._run_cmd_inner(make_args(SB, repeats=3), t.load_model_config("omni"),
                         "python", "test", SB)
    out = buf.getvalue()
    warned = "rejection is INACTIVE" in out
    rounds, dest, clean = _state(t)
    ok = warned and dest == []
    print(f"    warned={warned} rounds={rounds} destructive={dest}")
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    shutil.rmtree(SB, ignore_errors=True)
    return ok



def merge_with_rejections():
    """merge-runs must carry destructive marks and per-stage effective N."""
    A = REPO / ".tune-runs/_merge_a"
    B = REPO / ".tune-runs/_merge_b"
    C = REPO / ".tune-runs/_merge_c"
    for d in (A, B, C):
        shutil.rmtree(d, ignore_errors=True)

    def build(where, stages, bad):
        where.mkdir(parents=True)
        t = load()
        t._run_shared = _writer(t, bad=bad)
        t._ensure_gpus_free = lambda *a, **kw: True
        t._stage_gpus = lambda *a, **kw: 1
        args = make_args(where, stages=stages)
        rc = t._run_cmd_inner(args, t.load_model_config("omni"), "python", "test", where)
        return t, rc

    ta, rca = build(A, "mmsu_speed", {3})      # partition A: one rejection
    tb, rcb = build(B, "mmmu_speed", set())    # partition B: clean
    t = load()
    rc = t.merge_runs([A, B], C)
    stages = t._load_yaml(t.stages_path("omni"))
    a_rounds = t._round_indices(C, ["mmsu_speed"])
    a_dest = [k for k in a_rounds if t._run_json_destructive(C / "mmsu_speed" / f"run{k}.json")]
    a_clean = len(t._clean_rounds(C, "mmsu_speed", stages["mmsu_speed"]))
    b_clean = len(t._clean_rounds(C, "mmmu_speed", stages["mmmu_speed"]))
    report_txt = (C / "report.md").read_text() if (C / "report.md").exists() else ""
    ok = (rc == 0 and a_dest == [3] and a_clean >= 5 and b_clean == 5
          and "Rejected destructive rounds" in report_txt)
    print(f"    A rounds={a_rounds} dest={a_dest} clean={a_clean}; B clean={b_clean}; rc={rc}")
    print(f"    merged report keeps rejection section: "
          f"{'Rejected destructive rounds' in report_txt}")
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    for d in (A, B, C):
        shutil.rmtree(d, ignore_errors=True)
    return ok



def rejection_survives_missing_sibling():
    """A rejected round must stay rejected even if one stage loses its file."""
    t = load()
    SBp = REPO / ".tune-runs/_probe_missing"
    shutil.rmtree(SBp, ignore_errors=True)
    stages = t._load_yaml(t.stages_path("omni"))
    sks = ["mmsu_accuracy", "mmsu_speed"]          # one unit, two stages
    for sk in sks:
        (SBp / sk).mkdir(parents=True)
    for k in range(1, 6):
        for sk in sks:
            st = stages[sk]
            m = {mk: (0.7 if "accuracy" in mk else 10.0) * (0.25 if k == 3 else 1.0)
                     + k * 0.001 for mk in st["metrics"]}
            (SBp / sk / f"run{k}.json").write_text(json.dumps({
                "status": "ok", "reason": "", "metrics": m,
                "sample_counts": {"total": st.get("expected_samples") or 10,
                                  "ok": st.get("expected_samples") or 10},
                "git_sha": "x", "attempts": 1}))
    ev = t._detect_destructive_rounds(SBp, sks, stages)
    t._mark_destructive_rounds(SBp, sks, ev)
    (SBp / sks[0] / "run3.json").unlink()
    still = t._round_rejected(SBp, sks, 3)
    refl = 3 in t._detect_destructive_rounds(SBp, sks, stages)
    ok = sorted(ev) == [3] and still and not refl
    print(f"    marked={sorted(ev)} still_rejected={still} re-flagged={refl}")
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    shutil.rmtree(SBp, ignore_errors=True)
    return ok


def resume_with_stage_subset():
    """Resuming a subset of stages must not disturb the others."""
    if SB.exists():
        shutil.rmtree(SB)
    SB.mkdir(parents=True)
    t = load()
    t._run_shared = _writer(t, bad={3})
    t._ensure_gpus_free = lambda *a, **kw: True
    t._stage_gpus = lambda *a, **kw: 1
    t._run_cmd_inner(make_args(SB, stages="mmsu_speed,mmmu_speed"),
                     t.load_model_config("omni"), "python", "test", SB)
    before = {sk: len(t._clean_rounds(SB, sk, t._load_yaml(t.stages_path("omni"))[sk]))
              for sk in ("mmsu_speed", "mmmu_speed")}
    t2 = load()
    t2._run_shared = _writer(t2)
    t2._ensure_gpus_free = lambda *a, **kw: True
    t2._stage_gpus = lambda *a, **kw: 1
    args = make_args(SB, stages="mmsu_speed"); args.resume = True
    rc = t2._run_cmd_inner(args, t2.load_model_config("omni"), "python", "test", SB)
    after = {sk: len(t2._clean_rounds(SB, sk, t2._load_yaml(t2.stages_path("omni"))[sk]))
             for sk in ("mmsu_speed", "mmmu_speed")}
    ok = rc == 0 and after["mmmu_speed"] == before["mmmu_speed"] and after["mmsu_speed"] >= 5
    print(f"    before={before} after={after} rc={rc}")
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    shutil.rmtree(SB, ignore_errors=True)
    return ok


if __name__ == "__main__":
    results = []
    print("=== 1. no destructive rounds: must stay at exactly `repeats` ===")
    results.append(run_scenario("clean run", set(),
                                expect=dict(destructive=[], clean=5,
                                            rounds=[1, 2, 3, 4, 5])))

    print("\n=== 2. one destructive round: reject + 2 replacements ===")
    results.append(run_scenario("n=1", {3},
                                expect=dict(destructive=[3], clean=6,
                                            rounds=[1, 2, 3, 4, 5, 6, 7])))

    print("\n=== 3. a replacement round is ALSO destructive (your example) ===")
    results.append(run_scenario("n=1 then 1 more", {3, 6},
                                expect=dict(destructive=[3, 6], clean=5,
                                            rounds=[1, 2, 3, 4, 5, 6, 7])))

    print("\n=== 4. escape hatch: --no-destructive-rejection keeps everything ===")
    results.append(run_scenario("escape hatch", {3}, no_reject=True,
                                expect=dict(destructive=[], clean=5,
                                            rounds=[1, 2, 3, 4, 5])))

    print("\n=== 5. n>=3 -> discard the block and restart once ===")
    results.append(run_scenario("n=3 restart", {1, 2, 3},
                                expect=dict(destructive=[1, 2, 3, 4, 5],
                                            clean=5,
                                            rounds=list(range(1, 11)))))

    print("\n=== 6. resume after teardown mid-replenishment ===")
    results.append(run_resume_scenario())

    print("\n=== 7. resume when the run was already complete (no-op) ===")
    results.append(two_phase("already complete", (), 99))

    print("\n=== 8. resume with only SOME replacement rounds on disk ===")
    results.append(two_phase("partial replacements", {3}, 6))

    print("\n=== 9. resume mid-restart (degenerate block discarded) ===")
    results.append(two_phase("mid-restart", {1, 2, 3}, 7))

    print("\n=== 10. resume across a different commit must be refused ===")
    results.append(resume_wrong_commit())

    print("\n=== 11. re-entry without --resume must be refused ===")
    results.append(resume_no_flag_refused())

    print("\n=== 12. detector edge cases ===")
    results.append(unit_edge_cases())

    print("\n=== 13. --repeats below MIN_OBS warns and disables ===")
    results.append(repeats_below_min_warns())

    print("\n=== 14. merge-runs carries rejections and mixed effective N ===")
    results.append(merge_with_rejections())

    print("\n=== 15. rejection survives a missing sibling stage file ===")
    results.append(rejection_survives_missing_sibling())

    print("\n=== 16. resume with a --stages subset ===")
    results.append(resume_with_stage_subset())

    print("\n" + "=" * 60)
    print(f"{sum(results)}/{len(results)} scenarios passed")
    shutil.rmtree(SB, ignore_errors=True)
    sys.exit(0 if all(results) else 1)
