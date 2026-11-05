import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / ".github/scripts/resolve_ci_labels.py"


def run_resolver(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments], capture_output=True, text=True
    )


def test_run_all_ci_label_selects_nothing(tmp_path: Path):
    """The removed run-all-ci label once selected every stage; it must not run CI."""
    resolve = run_resolver(
        "resolve",
        "--labels-json",
        '["run-all-ci"]',
        "--seed",
        "1",
        "--github-output",
        str(tmp_path / "github_output"),
    )
    cancel_with_run_all = run_resolver(
        "workflows-to-cancel", "--labels-json", '["run-all-ci"]'
    )
    cancel_without_labels = run_resolver("workflows-to-cancel", "--labels-json", "[]")

    assert resolve.returncode == 1
    assert "No CI label found" in resolve.stdout
    assert cancel_with_run_all.stdout == cancel_without_labels.stdout
