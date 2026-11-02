"""Omni CI orders its H100 jobs by data dependency, not by habit.

A job that waits for a stage whose output it never reads holds a runner idle
for nothing, so inside a suite every `needs` must be backed by an artifact the
job downloads. MPS stays last because it may rebuild the PR home that the other
suites share; see the DAG note in omni-ci.yaml.
"""

from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).parents[2] / ".github/workflows"
SUITES = {
    "pr-test": "test.yaml",
    "asr-ci": "test-asr-ci.yaml",
    "tts-ci": "test-tts-ci.yaml",
    "mps-ci": "test-mps-ci.yaml",
    "qwen3-omni-ci": "test-qwen3-omni-ci.yaml",
}
MODEL_SUITES = {"asr-ci", "tts-ci", "mps-ci", "qwen3-omni-ci"}


def load_jobs(name):
    return yaml.safe_load((WORKFLOWS / name).read_text())["jobs"]


def needs_of(job):
    needs = job.get("needs", [])
    if isinstance(needs, str):
        return {needs}
    return set(needs)


def uploads(job):
    names = set()
    for step in job.get("steps", []):
        uses = step.get("uses", "")
        options = step.get("with") or {}
        if uses.startswith("actions/upload-artifact"):
            names.add(options["name"])
        if uses == "./.github/actions/omni-post-stage" and options.get(
            "artifact-upload-name"
        ):
            names.add(options["artifact-upload-name"])
    return names


def downloads(job):
    return {
        step["with"]["name"]
        for step in job.get("steps", [])
        if step.get("uses", "").startswith("actions/download-artifact")
    }


def test_suite_jobs_wait_only_for_artifacts_they_read():
    for file in SUITES.values():
        jobs = load_jobs(file)
        for name, job in jobs.items():
            for dependency in needs_of(job):
                assert downloads(job) & uploads(
                    jobs[dependency]
                ), f"{file}: {name} waits for {dependency} but reads none of its artifacts"


def test_artifact_readers_wait_for_their_writers():
    for file in SUITES.values():
        jobs = load_jobs(file)
        writer = {
            artifact: name for name, job in jobs.items() for artifact in uploads(job)
        }
        for name, job in jobs.items():
            for artifact in downloads(job):
                assert writer[artifact] in needs_of(
                    job
                ), f"{file}: {name} reads {artifact} without waiting for {writer[artifact]}"


def test_model_suites_do_not_wait_for_each_other():
    jobs = load_jobs("omni-ci.yaml")
    for suite in MODEL_SUITES - {"mps-ci"}:
        assert not needs_of(jobs[suite]) & MODEL_SUITES, suite


def test_mps_runs_after_every_other_suite():
    jobs = load_jobs("omni-ci.yaml")
    assert set(SUITES) - {"mps-ci"} <= needs_of(jobs["mps-ci"])


def test_dispatch_cleanup_waits_for_every_suite():
    jobs = load_jobs("omni-ci.yaml")
    assert {"setup", *SUITES} <= needs_of(jobs["cleanup"])
