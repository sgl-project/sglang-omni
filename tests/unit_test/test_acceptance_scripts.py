# SPDX-License-Identifier: Apache-2.0
"""Exercise acceptance clients against a local fake HTTP service, not a model."""

import base64
import io
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image


@pytest.mark.parametrize(
    "api,bad_usage,bad_stage,lifecycle",
    [
        ("chat", False, False, False),
        ("images", False, False, False),
        ("chat", True, False, False),
        ("images", True, False, False),
        ("images", False, True, False),
        ("images", False, False, True),
    ],
)
def test_live_acceptance_scripts_record_success_and_response_failure(
    tmp_path, api, bad_usage, bad_stage, lifecycle
):
    state = {"requests": 0, "running": 0}
    lock = threading.Lock()
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            names = [
                "sglang:e2e_request_latency_seconds_count",
                "sglang:num_requests_total",
                "sglang:time_to_first_token_seconds_count",
                "sglang:inter_token_latency_seconds_count",
                "sglang:prompt_tokens_total",
                "sglang:generation_tokens_total",
                "sglang_omni:generated_images_total",
            ]
            stage = "unknown" if bad_stage else "generate"
            with lock:
                completed = state["requests"]
                running = state["running"]
            body = "\n".join(
                f'{name}{{model_name="test",stage="{stage}"}} {completed}'
                for name in names
            )
            body += f'\nsglang:num_running_reqs{{model_name="test",stage="generate"}} {running}'
            body += f'\nsglang:num_queue_reqs{{model_name="test",stage="generate"}} {max(running - 1, 0)}\n'
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body.encode())

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if lifecycle and not payload:
                self.send_response(422)
                self.end_headers()
                return
            else:
                pass
            if lifecycle:
                with lock:
                    state["running"] += 1
                time.sleep(0.4)
                with lock:
                    state["running"] -= 1
            else:
                pass
            with lock:
                state["requests"] += 1
            self.send_response(200)
            self.end_headers()
            if api == "images":
                data = {
                    "created": 1,
                    "data": [{"b64_json": encoded, "output_format": "png"}],
                    "usage": {
                        "input_tokens": 10,
                        "total_tokens": 999 if bad_usage else 10,
                        "input_tokens_details": {"text_tokens": 10, "image_tokens": 0},
                        "reasoning_tokens": 0,
                        "image_count": 1,
                        "width": 32,
                        "height": 32,
                    },
                }
                self.wfile.write(json.dumps(data).encode())
                return
            usage = {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22}
            if bad_usage:
                usage["total_tokens"] = 999
            if not payload["stream"]:
                data = {
                    "id": "test-response",
                    "object": "chat.completion",
                    "choices": [{"finish_reason": "stop"}],
                    "usage": usage,
                }
                self.wfile.write(json.dumps(data).encode())
                return
            events = [
                {
                    "object": "chat.completion.chunk",
                    "choices": [{"finish_reason": "stop"}],
                    "usage": usage,
                }
            ]
            assert "stream_options" not in payload
            body = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
            self.wfile.write((body + "data: [DONE]\n\n").encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        repo = Path(__file__).resolve().parents[2]
        command = [
            sys.executable,
            str(repo / "scripts/verify_model_metrics.py"),
            "--api",
            api,
            "--base-url",
            f"http://127.0.0.1:{server.server_port}",
            "--model",
            "test",
            "--size",
            "32x32",
            "--out-dir",
            str(tmp_path),
            "--timeout",
            "10",
        ]
        if lifecycle:
            command[1] = str(repo / "scripts/verify_metrics_lifecycle.py")
            del command[2:4]
            command += [
                "--concurrency",
                "2",
                "--poll-seconds",
                "0.02",
                "--require-queue",
            ]
        else:
            pass
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
        )
        summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
        expected_failure = bad_usage or bad_stage
        assert result.returncode == (1 if expected_failure else 0), (
            result.stdout + result.stderr
        )
        assert summary["passed"] is (not expected_failure)
        if lifecycle:
            assert summary["max_running"] > 0
            assert summary["queue_transition_observed"]
            assert summary["rejected_http_status"] == 422
            assert (tmp_path / "samples.json").is_file()
        else:
            assert (tmp_path / "before.prom").is_file()
            assert (tmp_path / "after.prom").is_file()
            assert (tmp_path / "responses/summary.json").is_file()
        if bad_usage:
            assert summary["request_exit_code"] == 1
        else:
            assert all(value > 0 for value in summary["deltas"].values())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
