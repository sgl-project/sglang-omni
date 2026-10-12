# SPDX-License-Identifier: Apache-2.0
"""Client environment and server identity recorded with a benchmark run."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import TypedDict

import requests

_NO_PROXIES = {"http": None, "https": None}
_FINGERPRINT_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "HF_ENDPOINT",
    "TORCHINDUCTOR_CACHE_DIR",
    "OMP_NUM_THREADS",
    "SGLANG_TORCH_PROFILER_DIR",
)
_FINGERPRINT_PACKAGES = ("torch", "sglang", "sglang-omni", "transformers")


class GitFingerprint(TypedDict):
    sha: str | None
    branch: str | None
    dirty: bool


class EnvironmentFingerprint(TypedDict):
    captured_at: str
    hostname: str
    platform: str
    python: str
    git: GitFingerprint
    packages: dict[str, str | None]
    dependency_freeze_sha256: str | None
    gpus: str | None
    env: dict[str, str | None]


class ModelEnvironmentFingerprint(EnvironmentFingerprint):
    model_path: str
    model_revision: str | None


class ServerIdentity(TypedDict):
    url: str
    models: list[str] | None


class BenchmarkFingerprint(TypedDict):
    client: EnvironmentFingerprint | ModelEnvironmentFingerprint
    server: ServerIdentity


def _run_command(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=60, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _installed_package_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _cached_hf_revision(model_path: str) -> str | None:
    if os.path.sep in model_path and os.path.isdir(model_path):
        return None
    hf_home = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface"
    )
    repo_dir = os.path.join(hf_home, "hub", "models--" + model_path.replace("/", "--"))
    ref_main = os.path.join(repo_dir, "refs", "main")
    try:
        with open(ref_main, encoding="utf-8") as handle:
            return handle.read().strip() or None
    except OSError:
        return None


def collect_server_identity(base_url: str) -> ServerIdentity:
    """Record the URL and the model ids reported by /v1/models."""
    identity: ServerIdentity = {"url": base_url.rstrip("/"), "models": None}
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/v1/models",
            timeout=10,
            proxies=_NO_PROXIES,
        )
        response.raise_for_status()
        payload = response.json()
        model_ids: list[str] = []
        for entry in payload.get("data", []):
            model_id = entry.get("id")
            if isinstance(model_id, str) and model_id:
                model_ids.append(model_id)
        identity["models"] = model_ids
    except (requests.RequestException, ValueError):
        identity["models"] = None
    return identity


def collect_environment_fingerprint(
    model_path: str | None = None,
) -> EnvironmentFingerprint | ModelEnvironmentFingerprint:
    """Capture client code, dependency, and hardware identity.

    A missing tool yields None so fingerprinting never blocks a run.
    """
    pip_freeze = _run_command([sys.executable, "-m", "pip", "freeze"])
    fingerprint: EnvironmentFingerprint = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "git": {
            "sha": _run_command(["git", "rev-parse", "HEAD"]),
            "branch": _run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(_run_command(["git", "status", "--porcelain"]) or ""),
        },
        "packages": {
            package_name: _installed_package_version(package_name)
            for package_name in _FINGERPRINT_PACKAGES
        },
        "dependency_freeze_sha256": (
            hashlib.sha256(pip_freeze.encode("utf-8")).hexdigest()
            if pip_freeze
            else None
        ),
        "gpus": _run_command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "env": {
            env_name: os.environ.get(env_name) for env_name in _FINGERPRINT_ENV_KEYS
        },
    }
    if not model_path:
        return fingerprint
    return {
        **fingerprint,
        "model_path": model_path,
        "model_revision": _cached_hf_revision(model_path),
    }
