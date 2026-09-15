# SPDX-License-Identifier: Apache-2.0
"""Omni entrypoint for the GPU deep-dive backend.

Applies ``omni_source_shim`` so ``sglang_omni/`` frames survive the backend's
source-attribution allowlists, then delegates the whole CLI to the vendored
``llm-torch-profiler-analysis`` analyzer unchanged. Every backend flag works
here; see that skill for the flag list.

    python analyze_omni_profile.py --framework sglang \
        --mapping-input .profiling-runs/whisper/mapping \
        --formal-input .profiling-runs/whisper/formal \
        --output-dir .profiling-runs/whisper/report
"""

from __future__ import annotations

import sys

import omni_source_shim


def main(argv: list[str] | None = None) -> int:
    _, analyzer = omni_source_shim.apply()
    return analyzer.main(list(argv if argv is not None else sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
