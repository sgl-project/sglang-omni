"""The UI-test health fixture answers only on loopback and only to the fixture token."""

import json
import socket
import subprocess
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/connector-health-fixture.py"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class HealthFixtureTests(unittest.TestCase):
    def setUp(self):
        self.port = free_port()
        self.process = subprocess.Popen(
            ["python3", str(SCRIPT), str(self.port)], stdout=subprocess.PIPE, text=True
        )
        self.addCleanup(self.process.wait, 5)
        self.addCleanup(self.process.terminate)
        self.assertEqual(
            self.process.stdout.readline().strip(), f"listening 127.0.0.1:{self.port}"
        )

    def get(self, token):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/health",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.load(response)

    def test_fixture_token_gets_voxt_health(self):
        self.assertEqual(
            self.get("connector-owned-fixture"),
            (200, {"ok": True, "app": "Voxt", "version": "fixture"}),
        )

    def test_other_token_is_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.get("invalid-fixture-token")
        self.assertEqual(caught.exception.code, 401)


if __name__ == "__main__":
    unittest.main()
