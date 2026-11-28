import argparse
import json
import os
import pathlib
import queue
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ASSETS = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(os.environ["CHROME_NATIVE_PROBE_DIR"])
HELPER = ASSETS.parents[1] / "helper/.build/release/voxtnew-helper"
OBS = os.environ["CHROME_AX_OBSERVER"]


class Helper:
    def __init__(self):
        with open(ROOT / "helper-stderr.log", "a") as helper_log:
            self.p = subprocess.Popen(
                [str(HELPER)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=helper_log,
                text=True,
                bufsize=1,
            )
        self.pending = {}
        self.seq = 0
        self.lock = threading.Lock()
        self.notices = []
        threading.Thread(target=self.read, daemon=True).start()
        self.call(
            "helper.hello",
            {"protocol": 1, "app": "voxt-chrome-validation", "version": "2.1.4"},
        )

    def read(self):
        for line in self.p.stdout:
            try:
                j = json.loads(line)
            except ValueError:
                continue
            if "id" in j:
                q = self.pending.get(j["id"])
                if q:
                    q.put(j)
            else:
                self.notices.append(j)
                with open(ROOT / "helper-events.jsonl", "a") as f:
                    f.write(json.dumps(j) + "\n")

    def call(self, method, params):
        with self.lock:
            self.seq += 1
            key = str(self.seq)
            q = queue.Queue()
            self.pending[key] = q
            self.p.stdin.write(
                json.dumps({"id": key, "method": method, "params": params}) + "\n"
            )
            self.p.stdin.flush()
        j = q.get(timeout=12)
        self.pending.pop(key, None)
        if "error" in j:
            raise RuntimeError(f'{method}: {j["error"]}')
        return j["result"]

    def stop(self):
        try:
            self.call("app.configure", {"manualAccessibility": []})
            time.sleep(0.5)
        finally:
            self.p.stdin.close()
            try:
                self.p.wait(3)
            except subprocess.TimeoutExpired:
                self.p.terminate()


def observe(pid):
    return json.loads(subprocess.check_output([OBS, str(pid)], text=True, timeout=3))


def seconds(s):
    p = s.split(":")
    return sum(float(v) * 60**i for i, v in enumerate(reversed(p)))


def usage(pid):
    txt = subprocess.check_output(
        ["ps", "-axo", "pid=,ppid=,time=,rss=,comm="], text=True
    )
    procs = {}
    for line in txt.splitlines():
        x = line.split(None, 4)
        if len(x) == 5:
            try:
                procs[int(x[0])] = {
                    "parent": int(x[1]),
                    "cpu": seconds(x[2]),
                    "rss": int(x[3]),
                }
            except ValueError:
                pass
    ids = {pid}
    for _ in range(5):
        ids |= {p for p, v in procs.items() if v["parent"] in ids}
    return {p: procs[p] for p in ids if p in procs}


def native_mode(pid, bundle, phase):
    key = str(time.time_ns())
    command = {"id": key, "pid": pid, "bundle": bundle, "phase": phase}
    temp = ROOT / "native-command.tmp"
    temp.write_text(json.dumps(command))
    temp.replace(ROOT / "native-command.json")
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        try:
            r = json.loads((ROOT / "native-response.json").read_text())
            if r["id"] == key:
                return r
        except (OSError, ValueError):
            pass
        time.sleep(0.025)
    raise TimeoutError("native production-code probe did not respond")


class Session:
    def __init__(self, pid, bundle="com.google.Chrome"):
        self.bundle = bundle
        self.pid = pid
        self.helper = None
        self.results = []
        self.inputs = []
        self.pollStop = threading.Event()
        self.initial = observe(pid)
        self.mark = None
        self.errors = []
        self.transitions = []

    def init(self):
        if self.helper:
            return
        self.initial = observe(self.pid)
        self.helper = Helper()
        self.permissions = self.helper.call("permissions.status", {})
        if not self.permissions.get("accessibility"):
            raise RuntimeError("test helper lacks existing Accessibility authorization")

    def mode(self, phase):
        self.pollStop.set()
        if hasattr(self, "thread"):
            self.thread.join(timeout=13)
        self.pollStop = threading.Event()
        if os.environ.get("CHROME_NATIVE_PROBE_DIR"):
            self.bridgeResult = native_mode(self.pid, self.bundle, phase)
        else:
            self.helper.call(
                "app.configure",
                {"manualAccessibility": [self.bundle] if phase == "active" else []},
            )
        if phase == "active":
            result = self.helper.call(
                "browser.url", {"pid": self.pid, "bundleId": self.bundle}
            )
            self.urlResult = {k: v for k, v in result.items() if k != "url"}

            def poll():
                for delay in [1, 2] + [5] * 100:
                    if self.pollStop.wait(delay):
                        break
                    try:
                        self.helper.call(
                            "browser.url", {"pid": self.pid, "bundleId": self.bundle}
                        )
                    except Exception as e:
                        self.errors.append(str(e))

            self.thread = threading.Thread(target=poll, daemon=True)
            self.thread.start()
        time.sleep(2.2)
        ax = observe(self.pid)
        expected = (
            True
            if phase == "active"
            else self.initial["AXEnhancedUserInterface"]["value"]
        )
        transition = {
            "phase": phase,
            "ax": ax,
            "expected": expected,
            "front": self.helper.call("app.frontmost", {}),
            "time": time.time(),
            "nativeBridge": getattr(self, "bridgeResult", None),
        }
        self.transitions.append(transition)
        self.save()
        if transition["front"].get("pid") != self.pid:
            raise RuntimeError("target browser is not foreground; exclude this run")
        if ax["AXEnhancedUserInterface"]["value"] != expected:
            raise RuntimeError(
                "AX transition did not match expected state: " + json.dumps(transition)
            )
        return transition

    def begin(self, data):
        self.mark = (time.monotonic(), usage(self.pid))
        return {"ok": True}

    def finish(self, data):
        now = time.monotonic()
        current = usage(self.pid)
        t, before = self.mark
        elapsed = now - t
        cpu = sum(
            max(0, v["cpu"] - before.get(p, {"cpu": 0})["cpu"])
            for p, v in current.items()
        )
        out = {
            **data,
            "wallSeconds": elapsed,
            "browserCPU": 100 * cpu / elapsed,
            "rssMiB": sum(v["rss"] for v in current.values()) / 1024,
            "ax": observe(self.pid),
            "timestamp": time.time(),
        }
        self.results.append(out)
        self.save()
        return out

    def save(self):
        (ROOT / f"{self.pid}-measurements.json").write_text(
            json.dumps(
                {
                    "pid": self.pid,
                    "initial": self.initial,
                    "results": self.results,
                    "inputs": self.inputs,
                    "errors": self.errors,
                    "transitions": self.transitions,
                },
                indent=2,
            )
        )

    def summary(self):
        return {
            "pid": self.pid,
            "initial": self.initial,
            "runs": len(self.results),
            "errors": self.errors,
            "permissions": getattr(self, "permissions", {}),
            "transitions": self.transitions,
        }


sessions = {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/probe") or self.path == "/":
            b = (ASSETS / "probe.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b)
        elif self.path == "/results":
            b = json.dumps({k: s.summary() for k, s in sessions.items()}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b)
        else:
            self.send_error(404)

    def do_POST(self):
        try:
            d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            tag = d.get("tag", "current")
            s = sessions[tag]
            if self.path == "/hello":
                j = {"pid": s.pid}
            elif self.path == "/initialize":
                s.init()
                j = s.summary()
            elif self.path == "/mode":
                j = s.mode(d["phase"])
            elif self.path == "/begin":
                j = s.begin(d)
            elif self.path == "/finish":
                j = s.finish(d)
            elif self.path == "/input":
                s.inputs.append(d["ms"])
                s.save()
                j = {"ok": True}
            elif self.path == "/summary":
                j = s.summary()
            else:
                raise ValueError("unknown method")
            status = 200
        except Exception as e:
            j = {"error": str(e)}
            status = 500
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(j).encode())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--port", type=int, default=18773)
    ap.add_argument("--bundle", default="com.google.Chrome")
    a = ap.parse_args()
    sessions["current"] = Session(a.pid, a.bundle)
    print(
        json.dumps({"listening": a.port, "initial": sessions["current"].initial}),
        flush=True,
    )
    try:
        ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
    finally:
        for s in sessions.values():
            s.pollStop.set()
            if hasattr(s, "thread"):
                s.thread.join(timeout=13)
            if s.helper:
                s.helper.stop()
