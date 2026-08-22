# SPDX-License-Identifier: Apache-2.0
"""Shared pytest configuration for the unit-test tree.

Two jobs run this tree:
- unit-test: CPU only. The job sets ``OMNI_UNIT_CPU_ONLY=1`` and hides
  accelerators, and deselects ``gpu``-marked tests with ``-m "not gpu"``.
- gpu-test: runs ``-m gpu`` with one forked process per test so every test
  gets a fresh CUDA context.
"""

import os

import pytest

from tests.unit_test.accel import has_accelerator

_CPU_ONLY_ENV = "OMNI_UNIT_CPU_ONLY"

# Import-time guard so a workflow edit cannot silently hand the CPU-only
# unit-test job an accelerator again (hook-based checks fire too late when
# this conftest is loaded mid-collection).
if os.environ.get(_CPU_ONLY_ENV) == "1" and has_accelerator():
    raise RuntimeError(
        f"{_CPU_ONLY_ENV}=1 but an accelerator is visible; run the unit-test "
        'stage with accelerators hidden (e.g. CUDA_VISIBLE_DEVICES="").'
    )


def pytest_collection_modifyitems(config, items):
    if has_accelerator():
        return
    skip_gpu = pytest.mark.skip(reason="requires an accelerator (gpu marker)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
