#!/usr/bin/env python3
"""Run the simulator's native Tailcat pairing test against the built Mac helper."""

import argparse
import http.server
import json
import os
import pathlib
import queue
import secrets
import subprocess
import tempfile
import threading
import urllib.parse
import urllib.request

PARSER = argparse.ArgumentParser()
PARSER.add_argument(
    "--red", action="store_true", help="return a virtual port the helper does not serve"
)
PARSER.add_argument(
    "--transport", action="store_true", help="run the native HTTP transport suite"
)
PARSER.add_argument(
    "--ui",
    action="store_true",
    help="run the simulator invitation and relaunch UI test",
)
PARSER.add_argument(
    "--phone-key", help="public key printed by the simulator's phone-key UI test"
)
OPTIONS = PARSER.parse_args()
if OPTIONS.ui and (OPTIONS.transport or not OPTIONS.phone_key):
    PARSER.error("--ui requires --phone-key and cannot be combined with --transport")
ROOT = pathlib.Path(__file__).resolve().parent.parent
DESTINATION = os.environ.get(
    "VOXT_SIMULATOR_DESTINATION", "platform=iOS Simulator,name=iPhone 17 Pro"
)
HELPER = ROOT / "build/voxt-tailcat-host/voxt-tailcat-host"
if not HELPER.is_file():
    PARSER.error("build the Mac helper first")

CONTROL_TOKEN = secrets.token_urlsafe(32)
HTTP_TOKEN = secrets.token_urlsafe(32)
PAYLOAD = b"swift-native-fixture" * 16384
CHILDREN = []
ADDRESSES = []
INVITATION = None
SLOW_STARTED = threading.Event()


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, _format, *_arguments):
        pass

    def send_json(self, code, body):
        encoded = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def authenticated(self):
        return self.headers.get("Authorization") == "Bearer " + HTTP_TOKEN

    def do_GET(self):
        if self.path == "/invitation" and OPTIONS.ui and INVITATION:
            body = INVITATION.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if not self.authenticated():
            self.send_json(401, {"error": "unauthorized"})
        elif self.path == "/slow-started":
            self.send_json(
                200 if SLOW_STARTED.is_set() else 425,
                {"started": SLOW_STARTED.is_set()},
            )
        elif self.path == "/v1/slow":
            SLOW_STARTED.set()
            threading.Event().wait(8)
            try:
                self.send_json(200, {"ok": True})
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/v1/health":
            self.send_json(
                200, {"ok": True, "app": "Voxt", "version": "native-fixture"}
            )
        else:
            self.send_json(404, {"error": "missing"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length < 0 or length > 400000:
            self.send_json(400, {"error": "body"})
            return
        body = self.rfile.read(length)
        if self.path == "/v1/dictations":
            if not self.authenticated():
                self.send_json(401, {"error": "unauthorized"})
            elif body != PAYLOAD:
                self.send_json(400, {"error": "body"})
            else:
                self.send_json(
                    200,
                    {
                        "id": "fixture",
                        "text": "fixture verified",
                        "rawText": "fixture verified",
                        "asrMs": 0,
                    },
                )
            return
        if (
            self.path != "/pair"
            or self.headers.get("Authorization") != "Bearer " + CONTROL_TOKEN
        ):
            self.send_json(401, {"error": "unauthorized"})
            return
        try:
            public_key = json.loads(body)["publicKey"]
            if not isinstance(public_key, str):
                raise ValueError("public key missing")
        except (ValueError, KeyError, TypeError):
            self.send_json(400, {"error": "request"})
            return
        child = subprocess.Popen(
            [str(HELPER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        CHILDREN.append(child)
        request = {
            "version": 1,
            "state": None,
            "allowedClients": [public_key],
            "localPort": self.server.server_port,
        }
        child.stdin.write(json.dumps(request).encode() + b"\n")
        child.stdin.flush()
        replies = queue.Queue(maxsize=1)
        threading.Thread(
            target=lambda: replies.put(child.stdout.readline()), daemon=True
        ).start()
        try:
            reply = json.loads(replies.get(timeout=25))
            if not reply.get("address") or reply.get("port") != 48931:
                raise ValueError("helper rejected the request")
        except (queue.Empty, ValueError, json.JSONDecodeError):
            self.send_json(503, {"error": "helper"})
            return
        ADDRESSES.append(reply["address"])
        self.send_json(
            200,
            {
                "address": reply["address"],
                "port": reply["port"] + int(OPTIONS.red),
                "token": HTTP_TOKEN,
            },
        )


SERVER = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
SERVER_THREAD = threading.Thread(target=SERVER.serve_forever, daemon=True)
SERVER_THREAD.start()
ENVIRONMENT = os.environ.copy()
if OPTIONS.ui:
    pair = urllib.request.Request(
        f"http://127.0.0.1:{SERVER.server_port}/pair",
        data=json.dumps({"publicKey": OPTIONS.phone_key}).encode(),
        headers={
            "Authorization": "Bearer " + CONTROL_TOKEN,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(pair, timeout=35) as response:
        details = json.load(response)
    INVITATION = "voxtmobile://pair?" + urllib.parse.urlencode(
        {
            "tailcat": details["address"],
            "port": details["port"],
            "token": details["token"],
            "client": OPTIONS.phone_key,
        }
    )
    ENVIRONMENT["TEST_RUNNER_VOXT_INVITATION_FIXTURE_URL"] = (
        f"http://127.0.0.1:{SERVER.server_port}/invitation"
    )
    COMMAND = [
        "xcodebuild",
        "test",
        "-project",
        "ios/VoxtMobile.xcodeproj",
        "-scheme",
        "VoxtMobile",
        "-destination",
        DESTINATION,
        "-derivedDataPath",
        "build/mobile-parity",
        "-parallel-testing-enabled",
        "NO",
        "-collect-test-diagnostics",
        "never",
        "-only-testing:VoxtMobileUITests/PreferencesUITests/testLiveTailcatInvitationFromLocalFixture",
    ]
elif OPTIONS.transport:
    ENVIRONMENT["VOXT_TAILCAT_FIXTURE_URL"] = f"http://127.0.0.1:{SERVER.server_port}"
    ENVIRONMENT["VOXT_FIXTURE_CONTROL_TOKEN"] = CONTROL_TOKEN
    COMMAND = [
        "swift",
        "test",
        "--package-path",
        "ios/VoxtTailcat",
        "--disable-keychain",
        "--disable-netrc",
        "--filter",
        "NativeTransportTests",
    ]
else:
    ENVIRONMENT["TEST_RUNNER_VOXT_TAILCAT_FIXTURE_URL"] = (
        f"http://127.0.0.1:{SERVER.server_port}"
    )
    ENVIRONMENT["TEST_RUNNER_VOXT_FIXTURE_CONTROL_TOKEN"] = CONTROL_TOKEN
    COMMAND = [
        "xcodebuild",
        "test",
        "-project",
        "ios/VoxtMobile.xcodeproj",
        "-scheme",
        "VoxtMobile",
        "-destination",
        DESTINATION,
        "-derivedDataPath",
        "build/mobile-parity",
        "-parallel-testing-enabled",
        "NO",
        "-collect-test-diagnostics",
        "never",
        "-only-testing:VoxtMobileTests/AppModelNativeTailcatTests",
    ]
kind = "ui" if OPTIONS.ui else "transport" if OPTIONS.transport else "mobile"
LOG = pathlib.Path(tempfile.gettempdir()) / (
    "voxt-tailcat-" + kind + "-host-" + ("red" if OPTIONS.red else "green") + ".log"
)
try:
    with LOG.open("w") as output:
        result = subprocess.run(
            COMMAND,
            cwd=ROOT,
            env=ENVIRONMENT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=480,
        )
        filtered = result.stdout.replace(CONTROL_TOKEN, "[control-token]").replace(
            HTTP_TOKEN, "[http-token]"
        )
        for address in ADDRESSES:
            filtered = filtered.replace(address, "[tailcat-address]")
        if INVITATION:
            filtered = filtered.replace(INVITATION, "[invitation]")
        output.write(filtered)
    print(
        "transport test exit:" if OPTIONS.transport else "simulator test exit:",
        result.returncode,
    )
    raise SystemExit(result.returncode)
finally:
    SERVER.shutdown()
    SERVER.server_close()
    for child in CHILDREN:
        if child.poll() is None:
            child.stdin.close()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
