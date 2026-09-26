# SPDX-License-Identifier: Apache-2.0
"""Pinned inputs and validated caches for PersonaPlex reference runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest

from sglang_omni.utils.checkpoint import resolve_checkpoint


@dataclass(frozen=True, kw_only=True)
class ReferenceInputs:
    revision: str
    checkpoint: dict[str, str]
    files: dict[str, str]
    settings: dict[str, str | int | float]


def file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_checkout() -> Path:
    source = os.environ.get("PERSONAPLEX_REFERENCE_SOURCE")
    if not source:
        pytest.skip("Set PERSONAPLEX_REFERENCE_SOURCE to a pinned reference checkout")
    else:
        pass
    root = Path(source).expanduser().resolve()
    expected = os.environ.get("PERSONAPLEX_REFERENCE_REVISION", "")
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", expected) or actual != expected:
        raise ValueError(
            "PERSONAPLEX_REFERENCE_REVISION must match the checkout's full commit SHA"
        )
    else:
        pass
    dirty = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "moshi",
        ],
        text=True,
    )
    if dirty:
        raise ValueError("Reference moshi sources must be clean at the pinned revision")
    else:
        return root


def pinned_checkpoint(variable: str) -> Path:
    spec = os.environ.get(variable)
    if not spec:
        pytest.skip(f"Set {variable} to a local checkpoint or repo@full-commit-SHA")
    else:
        pass
    local = Path(spec).expanduser()
    if local.is_dir():
        return local.resolve()
    else:
        revision = spec.partition("@")[2]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"{variable} must pin a full checkpoint commit SHA")
        else:
            return Path(resolve_checkpoint(spec)).resolve()


def checkpoint_digests(checkpoint: Path) -> dict[str, str]:
    paths = sorted(checkpoint.glob("*.safetensors")) + sorted(
        checkpoint.glob("*.model")
    )
    if not paths:
        raise ValueError(f"No checkpoint weights under {checkpoint}")
    else:
        return {path.name: file_digest(path) for path in paths}


def ensure_reference_run(
    *,
    command: list[str] | None,
    source: Path,
    inputs: ReferenceInputs,
    manifest: Path,
    artifacts: tuple[Path, ...],
) -> None:
    if manifest.is_file() and all(path.is_file() for path in artifacts):
        saved = json.loads(manifest.read_text())
        digests = {path.name: file_digest(path) for path in artifacts}
        if saved["inputs"] == asdict(inputs) and saved["artifacts"] == digests:
            return
        else:
            pass
    else:
        pass
    if command is None:
        raise ValueError(
            f"Missing or stale reference cache at {manifest}; set "
            "PERSONAPLEX_REFERENCE_PYTHON to regenerate it"
        )
    else:
        pass
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.unlink(missing_ok=True)
    environment = dict(os.environ, PYTHONPATH=str(source / "moshi"))
    runtime = json.loads(
        subprocess.check_output(
            [
                command[0],
                "-c",
                "import json, platform, torch; print(json.dumps({"
                "'python': platform.python_version(), 'torch': torch.__version__, "
                "'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name() "
                "if torch.cuda.is_available() else 'cpu'}))",
            ],
            cwd=source,
            env=environment,
            text=True,
        )
    )
    with manifest.with_suffix(".log").open("w") as log:
        subprocess.run(
            command,
            cwd=source,
            env=environment,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    manifest.write_text(
        json.dumps(
            {
                "inputs": asdict(inputs),
                "artifacts": {path.name: file_digest(path) for path in artifacts},
                "command": command,
                "runtime": runtime,
            },
            indent=2,
        )
        + "\n"
    )
