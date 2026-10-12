#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run native, model-free response regression without mocking platform imports."""

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if sys.platform != "linux":
        parser.error(
            "Run in Linux; this acceptance does not mock Unix/platform dependencies."
        )
    tests = [
        "tests/unit_test/client/test_usage.py",
        "tests/unit_test/client/test_completion_rollout.py",
        "tests/unit_test/serve/test_usage_response.py",
    ]
    if args.mode == "full":
        tests += [
            "tests/unit_test/serve/test_generate_rollout.py",
            "tests/unit_test/serve/test_openai_api.py",
            "tests/unit_test/serve/test_speech_protocol.py",
            "tests/unit_test/serve/test_openai_errors.py",
        ]
    missing = [test for test in tests if not (repo / test).is_file()]
    if missing:
        parser.error(
            "Missing tests; checkout/copy the response implementation first: "
            + ", ".join(missing)
        )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *tests,
        "--tb=short",
        f"--junitxml={out / 'junit.xml'}",
    ]
    metadata = {
        "python": sys.executable,
        "mode": args.mode,
        "repo": str(repo),
        "scope": "native Linux model-free regression; no GPU/NPU or model loading",
        "command": command,
    }
    for label, git_args in (
        ("commit", ["rev-parse", "HEAD"]),
        ("branch", ["branch", "--show-current"]),
        ("working_tree", ["status", "--short"]),
    ):
        try:
            result = subprocess.run(
                ["git", *git_args], cwd=repo, text=True, capture_output=True
            )
            metadata[label] = (
                result.stdout.strip() if result.returncode == 0 else "unavailable"
            )
        except FileNotFoundError:
            metadata[label] = "unavailable"
    with (out / "pytest.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        metadata["exit_code"] = process.wait()
    if (out / "junit.xml").is_file():
        root = ET.parse(out / "junit.xml").getroot()
        suites = list(root.iter("testsuite"))
        metadata["pytest_counts"] = {
            key: sum(int(suite.get(key, 0)) for suite in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
    metadata["passed"] = metadata["exit_code"] == 0
    (out / "summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
