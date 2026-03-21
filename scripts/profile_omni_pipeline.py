#!/usr/bin/env python3
"""Profile the Qwen3-Omni Thinker+Talker speech pipeline.

One command to: start server -> wait for ready -> profile -> send requests -> collect traces.

Usage:
    # Minimal (2x A100 80GB):
    python scripts/profile_omni_pipeline.py

    # Custom:
    python scripts/profile_omni_pipeline.py \
        --model-path /local/Qwen3-Omni-30B-A3B-Instruct \
        --gpu-thinker 0 --gpu-talker 1 --gpu-code-predictor 1 \
        --num-requests 5 --output-dir ./my_profiles

    # Profile an already-running server (skip server launch):
    python scripts/profile_omni_pipeline.py --server-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("profile_omni")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DEFAULT_PORT = 8000
DEFAULT_NUM_REQUESTS = 3
DEFAULT_MAX_TOKENS = 128
DEFAULT_STARTUP_TIMEOUT = 600  # 10 min -- model loading is slow
DEFAULT_OUTPUT_DIR = "./omni_profiles"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Test prompts -- short, varied, exercises different generation lengths
TEST_PROMPTS = [
    "Count from one to ten slowly.",
    "Explain what a GPU is in two sentences.",
    "Say hello in English, Chinese, and Japanese.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Server
    p.add_argument("--server-url", type=str, default=None,
                    help="If set, skip server launch and profile this running server.")
    p.add_argument("--model-path", type=str, default=DEFAULT_MODEL)
    p.add_argument("--model-name", type=str, default="qwen3-omni",
                    help="Model name in chat completions requests.")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)

    # GPU placement (2x A100 default)
    p.add_argument("--gpu-thinker", type=int, default=0)
    p.add_argument("--gpu-talker", type=int, default=1)
    p.add_argument("--gpu-code-predictor", type=int, default=1)
    p.add_argument("--gpu-code2wav", type=int, default=0)

    # Profiling
    p.add_argument("--num-requests", type=int, default=DEFAULT_NUM_REQUESTS,
                    help="Number of requests to send during profiling.")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                    help="Where to store trace files.")
    p.add_argument("--run-id", type=str, default=None,
                    help="Custom run ID. Auto-generated if not set.")
    p.add_argument("--startup-timeout", type=int, default=DEFAULT_STARTUP_TIMEOUT)
    p.add_argument("--warmup-requests", type=int, default=1,
                    help="Requests to send BEFORE profiling (warm up CUDA graphs, etc).")
    p.add_argument("--text-only", action="store_true",
                    help="Profile text-only (no audio). Useful for isolating Thinker.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wait_for_server(base_url: str, timeout: int) -> None:
    """Poll GET /health until 200 or timeout."""
    log.info("Waiting for server at %s (timeout=%ds)...", base_url, timeout)
    start = time.time()
    last_error = ""
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{base_url}/health", timeout=2)
            if resp.status_code == 200:
                elapsed = time.time() - start
                log.info("Server ready (%.0fs)", elapsed)
                return
            last_error = f"HTTP {resp.status_code}"
        except requests.exceptions.ConnectionError:
            last_error = "connection refused"
        except requests.exceptions.Timeout:
            last_error = "timeout"
        time.sleep(2)
    raise TimeoutError(
        f"Server not ready after {timeout}s. Last error: {last_error}\n"
        f"Check server logs for startup failures."
    )


def send_chat_request(
    base_url: str,
    prompt: str,
    max_tokens: int,
    model_name: str = "qwen3-omni",
    text_only: bool = False,
) -> dict:
    """Send one chat completion request. Returns response JSON."""
    modalities = ["text"] if text_only else ["text", "audio"]
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
        "modalities": modalities,
    }
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def start_profiler(base_url: str, run_id: str | None = None) -> dict:
    """POST /start_profile. Returns {"run_id": ..., "trace_path_template": ...}."""
    body = {}
    if run_id:
        body["run_id"] = run_id
    resp = requests.post(f"{base_url}/start_profile", json=body, timeout=10)
    if resp.status_code == 404:
        raise RuntimeError(
            "/start_profile returned 404. The server was not started with "
            "SGLANG_TORCH_PROFILER_DIR set, or profiler routes are not mounted.\n"
            "If using --server-url, make sure the server has profiler support."
        )
    resp.raise_for_status()
    result = resp.json()
    log.info("Profiler started -- run_id=%s", result["run_id"])
    return result


def stop_profiler(base_url: str, run_id: str) -> dict:
    """POST /stop_profile. Returns {"run_id": ...}."""
    resp = requests.post(
        f"{base_url}/stop_profile",
        json={"run_id": run_id},
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()
    log.info("Profiler stopped -- run_id=%s", result["run_id"])
    return result


def launch_server(args: argparse.Namespace, profile_dir: str) -> subprocess.Popen:
    """Start the speech server as a subprocess. Returns the Popen handle."""
    env = os.environ.copy()
    env["SGLANG_TORCH_PROFILER_DIR"] = profile_dir

    server_script = str(REPO_ROOT / "examples" / "run_qwen3_omni_speech_server.py")

    cmd = [
        sys.executable,
        server_script,
        "--model-path", args.model_path,
        "--gpu-thinker", str(args.gpu_thinker),
        "--gpu-talker", str(args.gpu_talker),
        "--gpu-code-predictor", str(args.gpu_code_predictor),
        "--gpu-code2wav", str(args.gpu_code2wav),
        "--port", str(args.port),
    ]
    log.info("Launching server: %s", " ".join(cmd))

    # start_new_session=True puts server + all children in a new process group
    # so we can kill them all together later
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    return proc


def kill_server(proc: subprocess.Popen) -> None:
    """Stop the server and all its child processes (pipeline stages)."""
    if proc.poll() is not None:
        return  # already exited
    pgid = os.getpgid(proc.pid)
    log.info("Stopping server process group (pgid=%d)...", pgid)
    try:
        os.killpg(pgid, signal.SIGTERM)
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        log.warning("Server didn't stop after SIGTERM, sending SIGKILL...")
        os.killpg(pgid, signal.SIGKILL)
        proc.wait(timeout=5)
    except ProcessLookupError:
        pass  # already gone
    log.info("Server stopped.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Resolve paths ---
    profile_dir = str(Path(args.output_dir).resolve())
    os.makedirs(profile_dir, exist_ok=True)

    base_url = args.server_url or f"http://localhost:{args.port}"
    we_launched_server = args.server_url is None
    server_proc = None

    try:
        # --- 1. Start server (unless --server-url given) ---
        if we_launched_server:
            server_proc = launch_server(args, profile_dir)
        else:
            log.info("Using existing server at %s", base_url)
            log.info(
                "NOTE: Traces are stored on the SERVER side at "
                "$SGLANG_TORCH_PROFILER_DIR/{run_id}/. "
                "Make sure that env var was set when the server started."
            )

        # --- 2. Wait for server ready ---
        wait_for_server(base_url, args.startup_timeout)

        # --- 3. Warmup (pre-profile) ---
        if args.warmup_requests > 0:
            log.info("Warmup: sending %d request(s)...", args.warmup_requests)
            for i in range(args.warmup_requests):
                prompt = TEST_PROMPTS[i % len(TEST_PROMPTS)]
                result = send_chat_request(
                    base_url, prompt, args.max_tokens,
                    args.model_name, args.text_only,
                )
                text = result["choices"][0]["message"].get("content", "")
                log.info("  Warmup %d: %s...", i + 1, text[:60])

        # --- 4. Start profiler ---
        log.info("Starting profiler...")
        prof_info = start_profiler(base_url, args.run_id)
        run_id = prof_info["run_id"]

        # --- 5. Send profiled requests ---
        mode = "text-only" if args.text_only else "text+audio"
        log.info(
            "Sending %d profiled request(s) [mode=%s, max_tokens=%d]...",
            args.num_requests, mode, args.max_tokens,
        )

        for i in range(args.num_requests):
            prompt = TEST_PROMPTS[i % len(TEST_PROMPTS)]
            t0 = time.time()
            result = send_chat_request(
                base_url, prompt, args.max_tokens,
                args.model_name, args.text_only,
            )
            elapsed = time.time() - t0

            text = result["choices"][0]["message"].get("content", "")
            has_audio = result["choices"][0]["message"].get("audio") is not None
            log.info(
                "  [%d/%d] %.1fs | audio=%s | %s...",
                i + 1, args.num_requests, elapsed, has_audio, text[:50],
            )

        # --- 6. Stop profiler ---
        log.info("Stopping profiler...")
        stop_profiler(base_url, run_id)

        # --- 7. Report results ---
        # Wait for background gzip compression (server spawns `gzip -f` subprocess)
        log.info("Waiting for trace compression...")
        time.sleep(5)

        _report_traces(profile_dir, run_id)

    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
    except TimeoutError as e:
        log.error("%s", e)
        sys.exit(1)
    except RuntimeError as e:
        log.error("%s", e)
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        if server_proc and server_proc.poll() is not None:
            log.error(
                "Server process died (exit code %d). Check logs above.",
                server_proc.returncode,
            )
        else:
            log.error("Cannot connect to %s. Is the server running?", base_url)
        sys.exit(1)
    finally:
        if we_launched_server and server_proc:
            kill_server(server_proc)


def _report_traces(profile_dir: str, run_id: str) -> None:
    """Find and log trace file locations."""
    trace_dir = os.path.join(profile_dir, run_id)
    log.info("=" * 60)
    log.info("PROFILING COMPLETE")
    log.info("=" * 60)
    log.info("Run ID:     %s", run_id)
    log.info("Trace dir:  %s", trace_dir)

    if os.path.isdir(trace_dir):
        traces = sorted(Path(trace_dir).rglob("*.trace.json*"))
        log.info("Trace files (%d):", len(traces))
        for t in traces:
            size_mb = t.stat().st_size / 1024 / 1024
            log.info("  %s  (%.1f MB)", t, size_mb)
    else:
        log.warning("Trace dir not found at %s", trace_dir)
        log.warning("  Check SGLANG_TORCH_PROFILER_DIR was set when server started.")
        all_traces = sorted(Path(profile_dir).rglob("*.trace.json*"))
        if all_traces:
            log.info("  Found traces elsewhere:")
            for t in all_traces:
                log.info("    %s", t)

    log.info("")
    log.info("Open traces with:")
    log.info("  chrome://tracing  (paste file path)")
    log.info("  https://ui.perfetto.dev  (drag & drop)")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
