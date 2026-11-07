# SPDX-License-Identifier: Apache-2.0
"""Relay backends whose transport packages are not installed."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

# note (Minyi Tang): a fresh interpreter, because the relay package records
# backend availability once, when it is first imported.
SHM_RELAY_THEN_SELECTED_RELAY = """
import json, logging, sys

# A None entry fails the import as if the package were not installed.
sys.modules.update(mooncake=None, nixl=None)

records = []
class RecordCollector(logging.Handler):
    def emit(self, record):
        records.append(record)
logging.getLogger().addHandler(RecordCollector())
logging.getLogger().setLevel(logging.DEBUG)

from sglang_omni.relay.base import create_relay

create_relay("shm", engine_id="probe", device="cpu").close()
shm_warnings = [
    f"{record.levelname} {record.name}: {record.getMessage()}"
    for record in records
    if record.levelno >= logging.WARNING
]
selection_error = selection_cause = None
try:
    create_relay(sys.argv[1], engine_id="selected", device="cpu", slot_size_mb=1, credits=1)
except RuntimeError as error:
    selection_error, selection_cause = str(error), repr(error.__cause__)
print(json.dumps({
    "shm_warnings": shm_warnings,
    "selection_error": selection_error,
    "selection_cause": selection_cause,
}))
"""


@pytest.mark.parametrize("relay_type", ["mooncake", "nixl"])
def test_absent_transport_is_quiet_until_selected(relay_type: str) -> None:
    """An SHM-only install logged ERROR for absent Mooncake and NIXL, and selecting either did not name the missing package."""
    completed = subprocess.run(
        [sys.executable, "-c", SHM_RELAY_THEN_SELECTED_RELAY, relay_type],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result["shm_warnings"] == []
    assert f"the {relay_type} package" in result["selection_error"]
    assert result["selection_cause"].startswith("ModuleNotFoundError")
