#!/usr/bin/env python3
"""Test the isolated Mac UI invitation against the iPhone simulator without logging it."""

import argparse
import http.server
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import threading
import urllib.parse

ROOT = pathlib.Path(__file__).resolve().parent.parent
DESTINATION = os.environ.get(
    "VOXT_SIMULATOR_DESTINATION", "platform=iOS Simulator,name=iPhone 17 Pro"
)
PARSER = argparse.ArgumentParser()
PARSER.add_argument("--red", action="store_true")
PARSER.add_argument("--phone-key", required=True)
OPTIONS = PARSER.parse_args()

invitation = subprocess.check_output(["/usr/bin/pbpaste"], text=True).strip()
parsed = urllib.parse.urlsplit(invitation)
items = dict(urllib.parse.parse_qsl(parsed.query))
if (
    parsed.scheme != "voxtmobile"
    or parsed.netloc != "pair"
    or set(items) != {"tailcat", "port", "token", "client"}
):
    PARSER.error("Mac clipboard does not contain a Tailcat invitation")
if (
    items["client"] != OPTIONS.phone_key
    or not items["tailcat"].startswith("tc")
    or not items["token"]
):
    PARSER.error("Mac invitation does not match the simulator phone")
if OPTIONS.red:
    items["client"] = "nodekey:" + "a" * 64
    invitation = urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(items), "")
    )


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, _format, *_arguments):
        pass

    def do_GET(self):
        if self.path != "/invitation":
            self.send_error(404)
            return
        body = invitation.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
environment = os.environ.copy()
environment["TEST_RUNNER_VOXT_INVITATION_FIXTURE_URL"] = (
    f"http://127.0.0.1:{server.server_port}/invitation"
)
result_parent = pathlib.Path(tempfile.mkdtemp(prefix="voxt-live-invitation-"))
result_bundle = result_parent / "result.xcresult"
command = [
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
    "-resultBundlePath",
    str(result_bundle),
    "-only-testing:VoxtMobileUITests/PreferencesUITests/testLiveTailcatInvitationFromLocalFixture",
]
log = pathlib.Path(tempfile.gettempdir()) / (
    "voxt-live-invitation-" + ("red" if OPTIONS.red else "green") + ".log"
)
try:
    completed = subprocess.run(
        command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=480
    )
    output = completed.stdout + completed.stderr
    output = output.replace(invitation, "[invitation]").replace(
        items["token"], "[token]"
    )
    output = re.sub(r"voxtmobile://pair\?[^\s\"']+", "[invitation]", output)
    log.write_text(output)
    print("live invitation test exit:", completed.returncode, "log:", log)
    raise SystemExit(completed.returncode)
finally:
    server.shutdown()
    server.server_close()
    shutil.rmtree(result_parent, ignore_errors=True)
