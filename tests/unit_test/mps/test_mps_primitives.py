# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from sglang_omni.mps.state import validate_control_socket


def test_validate_control_socket_names_the_sun_path_limit() -> None:
    long_socket = Path("/state") / ("x" * 120) / "mps" / "pipe" / "control"
    with pytest.raises(ValueError, match="sun_path"):
        validate_control_socket(long_socket)
    validate_control_socket(Path("/short/mps/pipe/control"))
