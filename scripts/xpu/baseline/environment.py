# SPDX-License-Identifier: Apache-2.0
"""Which machine and software a baseline belongs to, and the card family whose recipe it serves."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import socket
import subprocess

import sglang
import torch

from scripts.xpu.baseline.models import REPO_ROOT
from tests.test_ci.xpu_model.model_table import DEVICE_FAMILY_ENV, device_family


def installed_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def environment_fingerprint(host_label: str = "") -> dict[str, str]:
    """The machine, card and software a baseline belongs to; host_label names the machine from inside a container."""
    device = (
        torch.xpu.get_device_properties(0).name
        if torch.xpu.is_available()
        else "unknown"
    )
    # oneCCL decides whether a TP>1 decode graph replays its all-reduces, so it is part of the environment.
    return {
        "host": host_label or socket.gethostname(),
        "device": device,
        "torch": torch.__version__,
        "sglang": sglang.__version__,
        "oneccl": installed_version("oneccl"),
    }


def card_family(fingerprint: dict[str, str]) -> str:
    """The recipe family of the fingerprinted card, read the way the serving lane reads it."""
    return device_family(fingerprint["device"], os.environ.get(DEVICE_FAMILY_ENV, ""))


def fingerprint_key(fingerprint: dict[str, str]) -> str:
    blob = json.dumps(fingerprint, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def recorded_from_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
