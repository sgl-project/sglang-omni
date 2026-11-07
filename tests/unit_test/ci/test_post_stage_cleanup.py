"""Post-stage cleanup exit policy and timeout diagnostics."""

import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def cleanup_step_script() -> str:
    action_path = REPO_ROOT / ".github/actions/omni-post-stage/action.yaml"
    action = yaml.safe_load(action_path.read_text())
    return next(
        step["run"]
        for step in action["runs"]["steps"]
        if step["name"] == "Kill GPU processes"
    )


def test_post_stage_cleanup_only_downgrades_memory_timeout(
    cleanup_step_script: str,
) -> None:
    assert "cleanup_status=$?" in cleanup_step_script
    assert 'if [ "${cleanup_status}" -eq 1 ]; then' in cleanup_step_script
    assert "::warning::Post-stage GPU cleanup did not reach" in cleanup_step_script
    assert 'elif [ "${cleanup_status}" -ne 0 ]; then' in cleanup_step_script
    assert 'exit "${cleanup_status}"' in cleanup_step_script


@pytest.mark.parametrize(
    "cleanup_status, expected_status",
    [(0, 0), (1, 0), (2, 2), (126, 126), (127, 127), (130, 130), (137, 137)],
)
def test_post_stage_cleanup_exit_status_policy(
    cleanup_step_script: str,
    tmp_path: Path,
    cleanup_status: int,
    expected_status: int,
) -> None:
    cleanup_command = "bash .github/scripts/delete_gpu_process.sh --kill-orphans"
    assert cleanup_step_script.count(cleanup_command) == 1
    simulated_script = cleanup_step_script.replace(
        cleanup_command, f"bash -c 'exit {cleanup_status}'"
    )
    result = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-e",
            "-o",
            "pipefail",
            "-c",
            simulated_script,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == expected_status
    assert ("::warning::" in result.stdout) is (cleanup_status == 1)


def test_gpu_cleanup_timeout_prints_diagnostics_before_failing() -> None:
    script_path = REPO_ROOT / ".github/scripts/delete_gpu_process.sh"
    script = script_path.read_text()

    assert "_print_gpu_cleanup_diagnostics" in script
    assert "Timed out waiting for GPU memory.used" in script
    assert script.index("_print_gpu_cleanup_diagnostics") < script.rindex(
        "Timed out waiting for GPU memory.used"
    )
