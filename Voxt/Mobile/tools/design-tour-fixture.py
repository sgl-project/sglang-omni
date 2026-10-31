#!/usr/bin/env python3
"""Loopback-only synthetic server for DesignTourUITests; never uses a model or Mac host."""
import argparse
import base64
import json
import os
import threading
import time
import uuid
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def contract_folder(environ, here):
    """The Mac's recorded Local API replies: VOXT_CONTRACT_DIR if set, else
    Voxt/Mobile/contract next to this tools folder, else the copy beside the phone's tests.
    """
    if environ.get("VOXT_CONTRACT_DIR"):
        return Path(environ["VOXT_CONTRACT_DIR"])
    mobile = here.parent.parent
    if (mobile / "contract").is_dir():
        return mobile / "contract"
    return mobile / "ios/VoxtMobileKit/Tests/VoxtMobileKitTests/Fixtures/contract"


CONTRACT = contract_folder(os.environ, Path(__file__).resolve())
TEXT = "This is a design review dictation."
EDITED = "This is an edited design review dictation."


class Fixture(ThreadingHTTPServer):
    version = "design-fixture"
    delay = 4.0
    failure = False
    compose_count = 0
    last_options = {}
    lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/keyboard-test":
            data = b'<html><meta name="viewport" content="width=device-width"><label>Session test<textarea aria-label="Session test" rows="5"></textarea></label><label>Other field<textarea aria-label="Other field" rows="2">Different text</textarea></label></html>'
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if (
            self.path == "/stats"
            and self.headers.get("Authorization") == "Bearer design-review"
        ):
            return self.reply(200, {"composeCount": self.server.compose_count})
        if (
            self.path == "/last-options"
            and self.headers.get("Authorization") == "Bearer design-review"
        ):
            return self.reply(200, self.server.last_options)
        if (
            self.path == "/v1/compose/prompts"
            and self.headers.get("Authorization") == "Bearer design-review"
        ):
            return self.reply(
                200,
                {
                    "prompts": {
                        "dictate": "Clean up the transcribed speech.",
                        "edit": "Apply the instruction to the selected text.",
                        "translate": "Translate the transcribed speech into {{TARGET_LANGUAGE}}.",
                        "write": "Write the message.",
                    }
                },
            )
        if (
            self.path == "/v1/capabilities"
            and self.headers.get("Authorization") == "Bearer design-review"
        ):
            capabilities = json.loads((CONTRACT / "capabilities.json").read_text())
            # Corrections and history sync are optional; this fixture has neither.
            capabilities["endpoints"] = [
                e
                for e in capabilities["endpoints"]
                if e not in ("corrections", "history/sync")
            ]
            return self.reply(200, capabilities)
        if (
            self.path == "/v1/health"
            and self.headers.get("Authorization") == "Bearer design-review"
        ):
            health = json.loads((CONTRACT / "health.json").read_text())
            self.reply(200, {**health, "version": self.server.version})
        else:
            self.reply(401, {"error": "Unauthorized fixture request"})

    def do_POST(self):
        if self.headers.get("Authorization") != "Bearer design-review":
            return self.reply(401, {"error": "Unauthorized fixture request"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4_000_000:
                return self.reply(413, {"error": "Invalid fixture body size"})
            body = json.loads(self.rfile.read(length))
            if self.path == "/control":
                delay, failure = body["delay"], body["failure"]
                if (
                    not isinstance(delay, (float, int))
                    or not 0 <= delay <= 15
                    or type(failure) is not bool
                ):
                    return self.reply(400, {"error": "Invalid fixture controls"})
                with self.server.lock:
                    self.server.delay, self.server.failure = delay, failure
                return self.reply(200, {"ok": True})
            if self.path != "/v1/compose":
                return self.reply(404, {"error": "Unknown fixture route"})
            audio = base64.b64decode(body["audio"], validate=True)
            mode = body["options"]["mode"]
            if not audio.startswith(b"RIFF") or mode not in [
                "dictate",
                "translate",
                "edit",
                "write",
            ]:
                return self.reply(400, {"error": "Invalid fixture composition"})
            with self.server.lock:
                self.server.compose_count += 1
                self.server.last_options = {
                    "mode": mode,
                    "selectedText": body["options"].get("selectedText", ""),
                    "customPrompt": body["options"].get("customPrompt"),
                    "targetLanguage": body["options"].get("targetLanguage"),
                }
                delay, failure = self.server.delay, self.server.failure
            time.sleep(delay)
            if failure:
                return self.reply(503, {"error": "Design review: simulated failure."})
            text = EDITED if mode == "edit" else TEXT
            self.reply(
                200,
                {
                    "id": str(uuid.uuid4()),
                    "mode": mode,
                    "rawText": TEXT,
                    "text": text,
                    "asrMs": 0,
                    "enhanceMs": 0,
                },
            )
        except (ValueError, KeyError, TypeError):
            self.reply(400, {"error": "Malformed fixture request"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    # README images show a real-looking version; design reviews keep the default.
    parser.add_argument("--version", default="design-fixture")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output / "fixture.wav"), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 32000)
    server = Fixture(("127.0.0.1", 0), Handler)
    server.version = args.version
    (args.output / "endpoint.txt").write_text(f"http://127.0.0.1:{server.server_port}")
    print(
        f"Synthetic fixture listening on loopback port {server.server_port}", flush=True
    )
    server.serve_forever()
