# SPDX-License-Identifier: Apache-2.0
"""Shared utilities: WAV duration, SSE parsing, service health check."""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path

import requests as requests_lib

logger = logging.getLogger(__name__)

STARTUP_TIMEOUT = 600
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GPU_CLEANUP_SCRIPT = REPO_ROOT / ".github/scripts/delete_gpu_process.sh"
GPU_IDLE_THRESHOLD_MB = 2048
GPU_IDLE_WAIT_SECONDS = 600
GPU_IDLE_POLL_SECONDS = 5
WAV_HEADER_SIZE = 44
SSE_DATA_PREFIX = "data: "
SSE_DONE_MARKER = "data: [DONE]"


@contextmanager
def disable_proxy() -> Generator[None, None, None]:
    """Temporarily disable proxy env vars for loopback requests."""
    proxy_vars = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    )
    saved_env = {k: os.environ[k] for k in proxy_vars if k in os.environ}
    for k in proxy_vars:
        os.environ.pop(k, None)
    try:
        yield
    finally:
        for k in proxy_vars:
            os.environ.pop(k, None)
        os.environ.update(saved_env)


def no_proxy_env() -> dict[str, str]:
    """Return a copy of os.environ with proxy variables removed, for subprocess use."""
    proxy_keys = {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
    return {k: v for k, v in os.environ.items() if k.lower() not in proxy_keys}


def server_log_file(tmp_path_factory, prefix: str = "server_logs") -> Path | None:
    """Capture server logs to a file on CI; stream to the terminal locally."""
    is_ci = os.environ.get("GITHUB_ACTIONS") == "true"
    if not is_ci:
        return None
    return tmp_path_factory.mktemp(prefix) / "server.log"


def stop_server(proc: subprocess.Popen) -> None:
    """Gracefully stop the server process group, tolerating already-dead processes."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, ChildProcessError):
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)
        except (ProcessLookupError, ChildProcessError):
            return


def wait_for_gpu_memory_release(
    *,
    memory_threshold_mb: int | None = None,
    wait_timeout_seconds: int | None = None,
    poll_seconds: int | None = None,
) -> None:
    """Kill orphan GPU processes and block until every GPU is below threshold."""
    if not GPU_CLEANUP_SCRIPT.exists():
        raise FileNotFoundError(f"GPU cleanup script missing: {GPU_CLEANUP_SCRIPT}")

    env = os.environ.copy()
    env["OMNI_CI_GPU_MEMORY_CLEAN_THRESHOLD_MB"] = str(
        memory_threshold_mb
        if memory_threshold_mb is not None
        else GPU_IDLE_THRESHOLD_MB
    )
    env["OMNI_CI_GPU_CLEAN_WAIT_SECONDS"] = str(
        wait_timeout_seconds
        if wait_timeout_seconds is not None
        else GPU_IDLE_WAIT_SECONDS
    )
    env["OMNI_CI_GPU_CLEAN_POLL_SECONDS"] = str(
        poll_seconds if poll_seconds is not None else GPU_IDLE_POLL_SECONDS
    )

    print(
        f"[gpu cleanup] running ensure_gpus_idle "
        f"(threshold={env['OMNI_CI_GPU_MEMORY_CLEAN_THRESHOLD_MB']} MiB)...",
        flush=True,
    )
    result = subprocess.run(
        ["bash", str(GPU_CLEANUP_SCRIPT), "--kill-orphans"],
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "GPU memory was not released after stopping the inference server. "
            f"delete_gpu_process.sh exit={result.returncode}"
        )


def wait_healthy(
    proc: subprocess.Popen,
    port: int,
    log_file: Path | None,
    timeout: int = STARTUP_TIMEOUT,
) -> None:
    """Wait for a server to report healthy, stopping it and raising on failure."""
    try:
        with disable_proxy():
            wait_for_service(
                f"http://localhost:{port}",
                timeout=timeout,
                server_process=proc,
                server_log_file=log_file,
                health_body_contains="healthy",
            )
    except Exception as exc:
        stop_server(proc)
        log_text = (
            log_file.read_text() if log_file is not None and log_file.exists() else ""
        )
        message = str(exc)
        if log_text and log_text not in message:
            message = f"{message}\n{log_text}"
        if isinstance(exc, TimeoutError):
            raise TimeoutError(message) from exc
        if isinstance(exc, RuntimeError):
            raise RuntimeError(message) from exc
        raise


def start_server_from_cmd(
    cmd: list[str],
    log_file: Path | None,
    port: int,
    timeout: int = STARTUP_TIMEOUT,
    env: dict[str, str] | None = None,
    tee: bool = False,
) -> subprocess.Popen:
    """Start a server from an arbitrary command and wait until healthy."""
    process_env = os.environ.copy()
    if env is not None:
        process_env.update(env)
    if log_file is None:
        proc = subprocess.Popen(
            cmd,
            env=process_env,
            start_new_session=True,
        )
    elif tee:
        # Tee (file + stdout): TP=2 fixture wants the file for grep + live
        # output for `pytest -s`. Pattern from sglang's popen_launch_server.
        log_handle = open(log_file, "w")
        try:
            proc = subprocess.Popen(
                cmd,
                env=process_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
                bufsize=1,
            )
        except Exception:
            log_handle.close()
            raise

        def _tee_stdout(src, sink) -> None:
            try:
                for line in iter(src.readline, ""):
                    sink.write(line)
                    sink.flush()
                    sys.stdout.write(line)
                    sys.stdout.flush()
            finally:
                src.close()
                sink.close()

        # log_handle ownership is handed to the thread; its finally closes it.
        threading.Thread(
            target=_tee_stdout,
            args=(proc.stdout, log_handle),
            daemon=True,
        ).start()
    else:
        with open(log_file, "w") as log_handle:
            proc = subprocess.Popen(
                cmd,
                env=process_env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    wait_healthy(proc, port, log_file, timeout=timeout)
    return proc


@contextmanager
def managed_omni_server(
    *,
    model_path: str,
    port: int,
    host: str,
    log_file: Path | None,
    max_running_requests: int | None = None,
    cuda_graph_max_bs: int | None = None,
    mem_fraction_static: float | None = None,
    timeout: int = STARTUP_TIMEOUT,
    wait_for_gpu_release: bool = True,
) -> Iterator[None]:
    """Start an ``sglang_omni.cli serve`` process and clean it up on exit."""
    _ensure_port_available(host, port)
    cmd = [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--model-path",
        model_path,
        "--port",
        str(port),
        "--host",
        host,
    ]
    if max_running_requests is not None:
        cmd.extend(["--max-running-requests", str(max_running_requests)])
    if cuda_graph_max_bs is not None:
        cmd.extend(["--cuda-graph-max-bs", str(cuda_graph_max_bs)])
    if mem_fraction_static is not None:
        cmd.extend(["--mem-fraction-static", str(mem_fraction_static)])
    logger.info(f"Starting server: {' '.join(cmd)}")
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    proc = start_server_from_cmd(cmd, log_file, port, timeout=timeout)
    try:
        yield
    finally:
        logger.info(f"Stopping server ({model_path})")
        stop_server(proc)
        if wait_for_gpu_release:
            wait_for_gpu_memory_release()


def _ensure_port_available(host: str, port: int) -> None:
    probe_host = "" if host in {"0.0.0.0", "::"} else host
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((probe_host, port))
    except OSError as exc:
        raise RuntimeError(
            f"Port {port} is already in use on {host}; pass a free --port."
        ) from exc


def get_wav_duration(wav_bytes: bytes) -> float:
    """Return PCM playback length in seconds for raw WAV bytes."""
    if len(wav_bytes) <= WAV_HEADER_SIZE:
        return 0.0
    sample_rate = struct.unpack_from("<I", wav_bytes, 24)[0]
    num_channels = struct.unpack_from("<H", wav_bytes, 22)[0]
    bits_per_sample = struct.unpack_from("<H", wav_bytes, 34)[0]
    if sample_rate == 0 or num_channels == 0 or bits_per_sample == 0:
        return 0.0
    bytes_per_sample = num_channels * bits_per_sample // 8
    pcm_size = len(wav_bytes) - WAV_HEADER_SIZE
    return pcm_size / (sample_rate * bytes_per_sample)


def parse_sse_event(line: str) -> dict | None:
    """Parse one Server-Sent Event (SSE) JSON line."""
    if not line.startswith(SSE_DATA_PREFIX) or line == SSE_DONE_MARKER:
        return None
    return json.loads(line[len(SSE_DATA_PREFIX) :])


def wait_for_service(
    base_url: str,
    timeout: int = 1200,
    *,
    server_process: subprocess.Popen | None = None,
    server_log_file: str | os.PathLike[str] | None = None,
    health_body_contains: str | None = None,
) -> None:
    """Wait for SGLang Omni Server to be ready."""
    logger.info(f"Waiting for service at {base_url} ...")
    start = time.time()
    while True:
        if server_process is not None:
            exit_code = server_process.poll()
            if exit_code is not None:
                log_text = ""
                if server_log_file is not None:
                    log_path = os.fspath(server_log_file)
                    if os.path.isfile(log_path):
                        with open(log_path) as f:
                            log_text = f.read()
                raise RuntimeError(f"Server exited with code {exit_code}.\n{log_text}")
        try:
            resp = requests_lib.get(f"{base_url}/health", timeout=1)
            if resp.status_code == 200 and (
                health_body_contains is None or health_body_contains in resp.text
            ):
                logger.info("Service is ready.")
                return
        except requests_lib.exceptions.RequestException as exc:
            logger.debug(f"Health check failed for {base_url}: {exc}")
        if time.time() - start > timeout:
            raise TimeoutError(f"Service at {base_url} not ready within {timeout}s")
        time.sleep(1)


def save_json_results(results: dict, output_dir: str, filename: str) -> str:
    """Write results as JSON to output_dir/filename and return the path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {path}")
    return path
