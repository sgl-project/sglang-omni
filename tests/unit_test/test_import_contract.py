# SPDX-License-Identifier: Apache-2.0
"""sglang_omni modules must not import sgl_kernel at import time."""

import subprocess
import sys

import pytest


# note: scheduling.omni_scheduler joins this list once upstream sglang stops
# importing sgl_kernel at module level (via cache_move.py:10 and
# layers/quantization/utils.py:45) — see issue #2384.
@pytest.mark.parametrize("module", ["sglang_omni.scheduling.sglang_backend"])
def test_module_does_not_import_sgl_kernel(module):
    code = (
        "import sys; import "
        f"{module}; "
        "assert 'sgl_kernel' not in sys.modules, "
        f"'{module} eagerly imported sgl_kernel'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"importing {module} must not pull in sgl_kernel "
        "(SIGILLs on hosts without AVX-512-BF16):\n"
        f"{result.stderr}"
    )
