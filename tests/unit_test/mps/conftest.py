# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the MPS unit tests."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def short_root():
    """State root under /tmp, because the daemon's control socket path is capped."""
    root = Path(tempfile.mkdtemp(prefix="mps-", dir="/tmp"))
    yield root
    shutil.rmtree(root, ignore_errors=True)
