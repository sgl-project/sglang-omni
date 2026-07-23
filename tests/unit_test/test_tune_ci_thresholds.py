import ast
import importlib.util
from pathlib import Path

TUNE_PATH = (
    Path(__file__).resolve().parents[2] / ".claude/skills/tune-ci-thresholds/tune.py"
)
SPEC = importlib.util.spec_from_file_location("tune_ci_thresholds", TUNE_PATH)
tune = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(tune)


def test_expected_samples_never_uses_concurrency():
    tree = ast.parse("CONCURRENCY = 16\nMAX_SAMPLES = 50\n")

    assert tune._expected_samples(tree, "accuracy", None, ["CONCURRENCY"]) is None
    assert (
        tune._expected_samples(tree, "accuracy", None, ["MAX_SAMPLES", "CONCURRENCY"])
        == 50
    )


def test_moss_td_stream_n_above_50_cer_max_is_fixed_not_calibrated():
    """Streaming n_above_50 stays hand-pinned; discover/apply must not target it."""
    assert "MOSS_TD_STREAM_N_ABOVE_50_CER_MAX" in tune._FIXED_THRESHOLD_SYMBOLS
    assert tune.match_metric("MOSS_TD_STREAM_N_ABOVE_50_CER_MAX", None) is None
    assert tune.match_metric("MOSS_TD_N_ABOVE_50_CER_MAX", None) == "n_above_50_pct_cer"


def test_configured_expected_samples_supports_group_override():
    assert tune._configured_expected_samples({"expected_samples": 50}, "speed") == 50
    assert (
        tune._configured_expected_samples(
            {"expected_samples": {"speed": 2000}}, "speed"
        )
        == 2000
    )


def test_gpu_cleanup_is_scoped_to_explicit_targets(monkeypatch, tmp_path):
    script = tmp_path / ".github/scripts/delete_gpu_process.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env bash\n")
    calls = []
    monkeypatch.setattr(tune, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        tune.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append((cmd, kwargs))
        or type("R", (), {"returncode": 0})(),
    )

    tune._kill_calibration_gpu_processes([3, 2, 3])

    assert len(calls) == 1
    cmd, kwargs = calls[0]
    assert cmd[-1] == "--kill-orphans"
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "2,3"


def test_metric_statistics_retains_outlier_and_reports_dispersion():
    stats = tune._metric_statistics([1.0, 1.0, 1.1, 1.1, 9.0], "max")

    assert stats["worst"] == 9.0
    assert stats["median"] == 1.1
    assert stats["range"] == 8.0
    assert stats["std"] > 0
    assert stats["cv"] > 0
    assert stats["outlier_runs"] == [5]


def test_wilson_interval_contains_observed_accuracy():
    low, high = tune._wilson_interval(80, 100)

    assert low < 0.8 < high


def test_merge_runs_combines_disjoint_strict_ready_partitions(monkeypatch, tmp_path):
    schema = tmp_path / "stages.yaml"
    schema.write_text("a: {}\nb: {}\n")
    run_a, run_b, merged = tmp_path / "a", tmp_path / "b", tmp_path / "merged"
    plans = []
    for run_dir, stage in ((run_a, "a"), (run_b, "b")):
        run_dir.mkdir()
        (run_dir / stage).mkdir()
        (run_dir / stage / "run1.json").write_text("{}")
        plan = {
            "model": "omni",
            "repeats": 1,
            "calibration_git_sha": "abc",
            "stages": [stage],
            "stages_yaml": str(schema),
        }
        plans.append(plan)
        (run_dir / "plan.json").write_text(tune.json.dumps(plan))

    ready_by_dir = {
        run_a: {"plan": plans[0], "stages_yaml": schema},
        run_b: {"plan": plans[1], "stages_yaml": schema},
    }
    monkeypatch.setattr(
        tune, "validate_run_ready", lambda run_dir: (ready_by_dir[run_dir], [])
    )
    monkeypatch.setattr(tune, "report", lambda run_dir: 0)

    assert tune.merge_runs([run_a, run_b], merged) == 0
    merged_plan = tune.json.loads((merged / "plan.json").read_text())
    assert merged_plan["stages"] == ["a", "b"]
    assert (merged / "a/run1.json").exists()
    assert (merged / "b/run1.json").exists()
