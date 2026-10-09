#!/usr/bin/env bash
# Combine every finished repeat under RUN_ROOT into RUN_ROOT/RESULTS.md.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

"$SCORING_VENV/bin/python" "$SGLANG_OMNI_ROOT/benchmarks/duplex/fdb_v15/aggregate.py" \
    --run-root "$RUN_ROOT" --engine "$ENGINE_LABEL" --judge "$JUDGE"
