#!/usr/bin/env python3
"""Loopback stand-in for Voxt's /v1/health, for Voxt Connector UI tests.

Usage: connector-health-fixture.py PORT. Accepts only the synthetic token below.
"""
import http.server
import json
import sys

TOKEN = "connector-owned-fixture"


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/v1/health":
            self.send_error(404)
        elif self.headers.get("Authorization") != f"Bearer {TOKEN}":
            self.send_error(401)
        else:
            body = json.dumps(
                {"ok": True, "app": "Voxt", "version": "fixture"}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *args):
        pass


port = int(sys.argv[1])
server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
print(f"listening 127.0.0.1:{port}", flush=True)
server.serve_forever()
