#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def load_stage(stage_id: str, registry_path: Path) -> dict:
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    for stage in data.get("stages", []):
        if stage["stage_id"] == stage_id:
            return stage
    raise SystemExit(f"Unknown stage_id: {stage_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-id", required=True)
    parser.add_argument(
        "--registry",
        default="ci-scheduler/stages.json",
        help="Path to stages.json from repository root.",
    )
    parser.add_argument(
        "--venv-name",
        default=os.environ.get("OMNI_VENV_NAME", "omni"),
        help="Virtualenv directory created by omni-setup.",
    )
    args = parser.parse_args()

    stage = load_stage(args.stage_id, Path(args.registry))
    commands = stage.get("commands") or []
    if not commands:
        raise SystemExit(f"Stage {args.stage_id} does not define commands")

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", os.getcwd())
    env.setdefault("OMNI_CI_STAGE_ID", args.stage_id)
    env.setdefault("OMNI_CI_STAGE_LABEL", stage["check_name"])

    for command in commands:
        shell_command = (
            "set -euo pipefail; "
            f"source {args.venv_name}/bin/activate; "
            "export PYTHONPATH=$PWD; "
            f"{command}"
        )
        print(f"=== running {args.stage_id}: {command}", flush=True)
        subprocess.run(["bash", "-lc", shell_command], check=True, env=env)

    return 0


if __name__ == "__main__":
    sys.exit(main())
