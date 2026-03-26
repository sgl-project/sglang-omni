import os
import signal
import subprocess
import sys
import time
from logging import Logger

import requests

logger = Logger(__name__)


def launch_server(args) -> subprocess.Popen:
    """Launch sglang-omni server as a subprocess and wait until healthy."""
    if not args.model_path:
        raise ValueError("--model-path is required when using --launch-server")

    host = args.host
    port = args.port
    base_url = f"http://{host}:{port}"

    cmd = [
        sys.executable,
        "-m",
        "sglang_omni.cli.cli",
        "serve",
        "--model-path",
        args.model_path,
        "--host",
        host,
        "--port",
        str(port),
    ]
    if args.config:
        cmd += ["--config", args.config]
    if args.relay_backend:
        cmd += ["--relay-backend", args.relay_backend]

    logger.info("Launching server: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )

    # Wait for health endpoint
    timeout = args.server_timeout
    for _ in range(timeout):
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"Server exited with code {proc.returncode}.\n{out}")
        try:
            resp = requests.get(f"{base_url}/health", timeout=2)
            if resp.status_code == 200 and "healthy" in resp.text:
                logger.info("Server is healthy.")
                return proc
        except requests.ConnectionError:
            pass
        time.sleep(1)

    kill_server(proc)
    raise TimeoutError(f"Server not healthy within {timeout}s")


def kill_server(proc: subprocess.Popen) -> None:
    """Gracefully terminate the server process group."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=10)
    except ProcessLookupError:
        pass
    logger.info("Server process terminated.")
