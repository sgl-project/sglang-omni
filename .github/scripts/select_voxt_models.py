"""Picks the native models a Voxt Mac CI run tests, from its changed files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

MODEL_JOBS_PATH = Path("sglang_omni_mlx/native/ci/model_jobs.json")


def select_models(
    model_jobs: dict, changed_files: list[str], requested: str | None
) -> list[dict]:
    """Returns the model jobs to run; shared paths or a request for all select every model."""
    models = model_jobs["models"]
    if requested:
        names = {name.strip() for name in requested.split(",") if name.strip()}
        if "all" in names:
            return models
        else:
            unknown = names - {model["name"] for model in models}
            if unknown:
                raise SystemExit(f"unknown model jobs: {sorted(unknown)}")
            else:
                return [model for model in models if model["name"] in names]
    elif any(
        re.search(pattern, path)
        for pattern in model_jobs["shared_paths"]
        for path in changed_files
    ):
        return models
    else:
        return [
            model
            for model in models
            if any(
                re.search(pattern, path)
                for pattern in model["paths"]
                for path in changed_files
            )
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--changed-files", type=Path, required=True)
    parser.add_argument("--requested")
    parser.add_argument("--github-output", type=Path, required=True)
    arguments = parser.parse_args()
    model_jobs = json.loads(MODEL_JOBS_PATH.read_text())
    changed_files = json.loads(arguments.changed_files.read_text())
    selected = select_models(model_jobs, changed_files, arguments.requested)
    print("model jobs:", ", ".join(model["name"] for model in selected) or "none")
    with arguments.github_output.open("a") as output:
        output.write(f"model_matrix={json.dumps({'model': selected})}\n")
        output.write(f"has_models={'true' if selected else 'false'}\n")


if __name__ == "__main__":
    main()
