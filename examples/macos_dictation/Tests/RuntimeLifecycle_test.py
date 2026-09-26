# SPDX-License-Identifier: Apache-2.0
"""Exercise launcher deadlines and isolated worker groups without model services."""

import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def worker(root, stubborn):
    if stubborn:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (root / "child").write_text(str(os.getpid()))
    while True:
        (root / "heartbeat").touch()
        time.sleep(0.02)


def backend(root, stubborn):
    # Match the backend's spawn/daemon worker topology, without importing MLX.
    child = multiprocessing.get_context("spawn").Process(
        target=worker, args=(root, stubborn), daemon=True
    )
    child.start()

    def stop(_signum, _frame):
        child.terminate()
        child.join(timeout=2)
        (root / "graceful").touch()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, signal.SIG_IGN if stubborn else stop)
    (root / "parent").write_text(str(os.getpid()))
    while True:
        time.sleep(0.02)


class RuntimeLifecycleTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="omni-lifecycle-test-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.env = dict(os.environ)
        self.env.pop("SGLANG_OMNI_STARTUP_TIMEOUT", None)

    def shell(self, code, *args):
        return subprocess.run(
            [
                "/bin/bash",
                "-c",
                code,
                "test",
                str(PROJECT / "local_runtime.sh"),
                sys.executable,
                *map(str, args),
            ],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=15,
        )

    def wait_for(self, predicate):
        deadline = time.monotonic() + 5
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(predicate(), "Timed out waiting for fixture state")

    @staticmethod
    def running(pid):
        # A killed orphan may remain a zombie briefly; it is no longer a worker.
        status = subprocess.run(
            ["ps", "-p", str(pid), "-o", "stat="],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return bool(status) and not status.startswith("Z")

    def test_asr_budget_tracks_backend_configuration(self):
        for configured, expected in [(None, 660), ("1200", 1260), ("600.5", 661)]:
            with self.subTest(configured=configured):
                if configured is not None:
                    self.env["SGLANG_OMNI_STARTUP_TIMEOUT"] = configured
                result = self.shell(
                    'source "$1"; dictation_python="$2"; asr_startup_timeout'
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(int(result.stdout), expected)

    def test_invalid_asr_budgets_fail(self):
        for value in ("0", "-1", "nan", "inf", "bad", ""):
            with self.subTest(value=value):
                self.env["SGLANG_OMNI_STARTUP_TIMEOUT"] = value
                result = self.shell(
                    'source "$1"; dictation_python="$2"; asr_startup_timeout'
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("positive finite", result.stderr)

    def test_slow_readiness_with_simulated_clock(self):
        # Run the real polling function with a deterministic clock, not a real
        # 400-second model load. Its deadline must still reject a hung backend.
        code = """source "$1"
dictation_python="$2"
budget="$(asr_startup_timeout)"
unset SECONDS
SECONDS=0
kill() { return 0; }
sleep() { SECONDS=$((SECONDS + 1)); }
ready_at="$3"
ready() { ((SECONDS >= ready_at)); }
wait_ready 123 ready ASR fixture.log "$budget"
"""
        for ready_at, success in [(400, True), (700, False)]:
            result = self.shell(code, ready_at)
            self.assertEqual(result.returncode == 0, success, result.stderr)
        self.env["SGLANG_OMNI_STARTUP_TIMEOUT"] = "1200"
        result = self.shell(code, 1000)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_ollama_keeps_its_separate_budget(self):
        result = self.shell(
            """source "$1"
dictation_logs="$3"
ollama_ready() { return 1; }
port_in_use() { return 1; }
find_ollama() { ollama_bin=fixture; }
start_owned() { last_owned_pid=123; }
wait_ready() { [[ "$1" == 123 && "$3" == Ollama && "$5" == 300 ]]; }
ensure_ollama
""",
            self.root,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def start_service(self, *, stubborn=False, shell_owner=False):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--backend",
            str(self.root),
            "stubborn" if stubborn else "normal",
        ]
        if shell_owner:
            command = [
                "/bin/bash",
                "-c",
                """set -euo pipefail
source "$1"
dictation_python="$2"
shift 2
trap cleanup_owned EXIT
trap 'exit 143' TERM
start_owned /dev/null "$@"
wait "$last_owned_pid"
""",
                "test",
                str(PROJECT / "local_runtime.sh"),
                sys.executable,
                *command,
            ]
        elif stubborn:
            # Shorten only test cleanup budgets to exercise escalation promptly.
            command = [
                sys.executable,
                "-c",
                "import sys; from service_process import run; "
                "sys.exit(run(sys.argv[1:], shutdown_timeout=0.2, descendant_timeout=0.2))",
                *command,
            ]
        else:
            command = [sys.executable, str(PROJECT / "service_process.py"), *command]
        owner = subprocess.Popen(
            command, cwd=PROJECT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        def cleanup():
            if owner.poll() is None:
                owner.terminate()
                try:
                    owner.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    owner.kill()
                    owner.wait()
            if (self.root / "parent").exists():
                try:
                    os.killpg(int((self.root / "parent").read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass

        self.addCleanup(cleanup)
        self.wait_for(
            lambda: (self.root / "parent").exists()
            and (self.root / "child").exists()
            and (self.root / "heartbeat").exists()
        )
        parent = int((self.root / "parent").read_text())
        child = int((self.root / "child").read_text())
        self.assertEqual(os.getpgid(child), parent)
        self.assertNotEqual(os.getpgrp(), parent)
        return owner, parent, child

    def test_parent_crash_cleans_workers_without_touching_other_processes(self):
        unrelated = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )
        self.addCleanup(unrelated.wait)
        self.addCleanup(unrelated.terminate)
        owner, parent, child = self.start_service()
        os.kill(parent, signal.SIGKILL)
        self.assertEqual(owner.wait(timeout=10), 137)
        self.wait_for(lambda: not self.running(child))
        self.assertIsNone(unrelated.poll())
        self.assertFalse((self.root / "graceful").exists())

    def test_shell_cleanup_allows_backend_graceful_shutdown(self):
        owner, parent, child = self.start_service(shell_owner=True)
        owner.terminate()
        self.assertEqual(owner.wait(timeout=10), 143)
        self.assertTrue((self.root / "graceful").exists())
        self.wait_for(lambda: not self.running(parent) and not self.running(child))

    def test_term_resistant_workers_are_killed_after_grace(self):
        owner, parent, child = self.start_service(stubborn=True)
        owner.terminate()
        self.assertEqual(owner.wait(timeout=5), 143)
        self.wait_for(lambda: not self.running(parent) and not self.running(child))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--backend":
        backend(Path(sys.argv[2]), sys.argv[3] == "stubborn")
    else:
        unittest.main()
