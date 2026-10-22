#!/usr/bin/env python3
"""The design-tour fixture answers like the Mac's Local API: /v1/capabilities is the recorded
contract less the endpoints the fixture doesn't serve, and /v1/health has its fields with the
version the fixture was started with.

Usage: python3 -I tools/test-design-tour-fixture.py [path/to/design-tour-fixture.py]
Starts the fixture on loopback twice (default, then --version 1.16.0) in a temporary
folder under $TMPDIR, and fails within 10 s if it doesn't come up.
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

FIXTURE = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path(__file__).resolve().parent / "design-tour-fixture.py"
)
_spec = importlib.util.spec_from_file_location("design_tour_fixture", FIXTURE)
fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixture)
CONTRACT = fixture.CONTRACT


def get(extra_args, path):
    with tempfile.TemporaryDirectory() as folder:
        process = subprocess.Popen(
            [sys.executable, "-I", str(FIXTURE), "--output", folder, *extra_args],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            endpoint = Path(folder) / "endpoint.txt"
            deadline = time.time() + 10
            while not endpoint.exists() or not endpoint.read_text():
                if process.poll() is not None:
                    raise AssertionError(
                        f"fixture exited: {process.stderr.read().strip()}"
                    )
                if time.time() > deadline:
                    raise AssertionError("fixture did not start within 10 s")
                time.sleep(0.1)
            request = urllib.request.Request(
                endpoint.read_text() + path,
                headers={"Authorization": "Bearer design-review"},
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                return json.load(response)
        finally:
            process.terminate()
            process.wait(timeout=5)


failures = 0
# Upstream has Voxt/Mobile/contract beside Voxt/Mobile/tools; this repository has a copy
# beside the phone's tests; VOXT_CONTRACT_DIR names another.
with tempfile.TemporaryDirectory() as root:
    mobile = Path(root) / "Voxt/Mobile"
    here = mobile / "tools/design-tour-fixture.py"
    here.parent.mkdir(parents=True)
    copy = mobile / "ios/VoxtMobileKit/Tests/VoxtMobileKitTests/Fixtures/contract"
    found = {"copy": fixture.contract_folder({}, here)}
    (mobile / "contract").mkdir()
    found["shared"] = fixture.contract_folder({}, here)
    found["environment"] = fixture.contract_folder(
        {"VOXT_CONTRACT_DIR": "/elsewhere"}, here
    )
    expected = {
        "copy": copy,
        "shared": mobile / "contract",
        "environment": Path("/elsewhere"),
    }
    for name in expected:
        ok = found[name] == expected[name]
        print(("PASS" if ok else "FAIL"), "contract folder", name, "->", found[name])
        failures += 0 if ok else 1
recorded_health = json.loads((CONTRACT / "health.json").read_text())
for args, expected in [([], "design-fixture"), (["--version", "1.16.0"], "1.16.0")]:
    try:
        health = get(args, "/v1/health")
        ok = health == {**recorded_health, "version": expected}
        print(("PASS" if ok else "FAIL"), args, "health ->", health)
        failures += 0 if ok else 1
    except AssertionError as error:
        print("FAIL", args, "->", error)
        failures += 1
try:
    capabilities = get([], "/v1/capabilities")
    recorded = json.loads((CONTRACT / "capabilities.json").read_text())
    # The fixture lists only the endpoints it answers.
    recorded["endpoints"] = [
        e for e in recorded["endpoints"] if e not in ("corrections", "history/sync")
    ]
    ok = capabilities == recorded
    print(("PASS" if ok else "FAIL"), "capabilities ->", capabilities)
    failures += 0 if ok else 1
except AssertionError as error:
    print("FAIL capabilities ->", error)
    failures += 1
sys.exit(1 if failures else 0)
