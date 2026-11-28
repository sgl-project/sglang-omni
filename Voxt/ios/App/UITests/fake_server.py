"""A fake OpenAI-compatible server for the app's UI tests (VoxtUITests).

Chat completions answer with a fixed English text (a JSON title/content
answer when the request asks for JSON: Rewrite writing from scratch);
transcriptions answer with a fixed Chinese sentence, and only with the key
the tests type ("test"). Every request is logged to stdout.

    python3 fake_server.py [port, default 18080]
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

ANSWER = "Hello, this is a test of Voxt on iOS."
SPEECH = "你好，这是 Voxt 在 iOS 上的测试。"


class Handler(BaseHTTPRequestHandler):
    def _send(self, body):
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        print("GET", self.path, flush=True)
        self._send({"object": "list", "data": [{"id": "fake", "object": "model"}]})

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path.endswith("/audio/transcriptions"):
            wav = raw.find(b"RIFF")
            print(
                "POST",
                self.path,
                f"{len(raw)} bytes, wav at {wav}, auth={self.headers.get('Authorization')!r}",
                flush=True,
            )
            if self.headers.get("Authorization") != "Bearer test":
                self.send_response(401)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self._send({"text": SPEECH})
            return
        body = json.loads(raw or b"{}")
        text = json.dumps(body, ensure_ascii=False)
        print("POST", self.path, text[:3000], flush=True)
        content = ANSWER
        if "response_format" in body or "Reply with a single JSON value" in text:
            content = json.dumps({"title": "Test title", "content": ANSWER})
        self._send(
            {
                "id": "c1",
                "object": "chat.completion",
                "model": body.get("model", "fake"),
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 9,
                    "total_tokens": 19,
                },
            }
        )

    def log_message(self, *args):
        pass


HTTPServer(
    ("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 18080), Handler
).serve_forever()
