# SPDX-License-Identifier: Apache-2.0
"""Check that different KV cache growth steps preserve returned values."""

import subprocess
from pathlib import Path


def test_kv_cache_growth_and_trim(stub_server_binary: Path) -> None:
    subprocess.run(
        [str(stub_server_binary.with_name("kv_cache_test"))],
        check=True,
        capture_output=True,
        text=True,
    )
