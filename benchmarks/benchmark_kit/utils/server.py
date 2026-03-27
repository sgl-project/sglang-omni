import json
import os
import signal
import subprocess
import sys
import time
from logging import Logger

import aiohttp
import requests

logger = Logger(__name__)


SSE_DATA_PREFIX = "data: "
SSE_DONE_MARKER = "data: [DONE]"


def parse_sse_event(line: str) -> dict | None:
    """Parse an SSE data line into a JSON dict, or None."""
    if not line.startswith(SSE_DATA_PREFIX) or line == SSE_DONE_MARKER:
        return None
    return json.loads(line[len(SSE_DATA_PREFIX) :])


async def iter_sse_lines(response: aiohttp.ClientResponse):
    """Yield decoded SSE lines from an aiohttp streaming response."""
    buffer = bytearray()
    async for chunk in response.content.iter_any():
        buffer.extend(chunk)
        while b"\n" in buffer:
            idx = buffer.index(b"\n")
            raw_line = bytes(buffer[:idx])
            del buffer[: idx + 1]
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line:
                yield line
    if buffer.strip():
        yield bytes(buffer).decode("utf-8", errors="replace").strip()


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


def wait_for_server(base_url: str, timeout: int = 1200) -> None:
    """Block until the server health endpoint returns 200."""
    import requests as requests_lib

    logger.info("Waiting for service at %s ...", base_url)
    start = time.time()
    while True:
        try:
            resp = requests_lib.get(f"{base_url}/health", timeout=1)
            if resp.status_code == 200:
                logger.info("Service is ready.")
                return
        except requests_lib.exceptions.RequestException:
            pass
        if time.time() - start > timeout:
            raise TimeoutError(f"Service at {base_url} not ready within {timeout}s")
        time.sleep(1)
