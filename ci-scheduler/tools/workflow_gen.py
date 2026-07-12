#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compiler.generator import check_generated, generate_all  # noqa: E402
from compiler.graph import GENERATED_WORKFLOWS_DIR, compile_all  # noqa: E402
from compiler.loader import DEFAULT_SCHEDULER_ROOTS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate per-job scheduler workflows from .github/workflows DAG sources"
    )
    parser.add_argument(
        "--workflows-dir",
        type=Path,
        default=ROOT.parent / ".github" / "workflows",
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        default=list(DEFAULT_SCHEDULER_ROOTS),
        help="Root workflow filenames under --workflows-dir (default: omni-ci.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT.parent / Path(GENERATED_WORKFLOWS_DIR),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify generated workflows match compiler output",
    )
    args = parser.parse_args()

    graphs = compile_all(args.workflows_dir, roots=tuple(args.roots))
    if args.check:
        errors: list[str] = []
        for graph in graphs.values():
            errors.extend(check_generated(graph, args.output_dir))
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        print(f"OK: {len(graphs)} source workflow(s) match generated output")
        return 0

    written = 0
    for graph in graphs.values():
        paths = generate_all(graph, args.output_dir)
        written += len(paths)
        for path in paths:
            print(path)
    print(f"Generated {written} workflow(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
