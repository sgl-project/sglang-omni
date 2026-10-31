# SPDX-License-Identifier: Apache-2.0
"""Prepare an environment-local SGLang source overlay before serving imports."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sysconfig
import tempfile
from importlib.metadata import distribution
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel

PATCH_DIRECTORY = Path(__file__).with_name("patches")
RUNTIME_DIRECTORY = "sglang_omni_runtime"
ACTIVATION_FILE = "sglang_omni_runtime.pth"


class SourceFileVersions(BaseModel):
    source_sha256: str | None
    patched_sha256: str


class RuntimePatchSpec(BaseModel):
    base_revision: str
    patches: list[str]
    files: dict[str, SourceFileVersions]


def verify_files(
    package_root: Path, specification: RuntimePatchSpec, *, patched: bool
) -> None:
    mismatches: list[str] = []
    for relative_path, versions in specification.files.items():
        source_file = package_root / relative_path
        actual_sha256 = (
            hashlib.sha256(source_file.read_bytes()).hexdigest()
            if source_file.is_file()
            else None
        )
        expected_sha256 = versions.patched_sha256 if patched else versions.source_sha256
        if actual_sha256 != expected_sha256:
            mismatches.append(relative_path)
        else:
            pass
    if mismatches:
        raise RuntimeError(
            f"SGLang source does not match the {'patched' if patched else 'base'} "
            f"runtime at {specification.base_revision}: {', '.join(mismatches)}. "
            "Install the pinned base and run sgl-omni-prepare-runtime before serving."
        )
    else:
        pass


def prepare_runtime(
    source_root: Path,
    site_packages: Path,
    patch_directory: Path = PATCH_DIRECTORY,
) -> Path:
    """Copy and patch the pinned source, then activate it for new Python processes."""
    source_root = source_root.resolve()
    specification_bytes = (patch_directory / "runtime.json").read_bytes()
    specification = RuntimePatchSpec.model_validate_json(specification_bytes)
    verify_files(source_root, specification, patched=False)
    fingerprint = hashlib.sha256(specification_bytes)
    for source_file in sorted(source_root.rglob("*")):
        if (
            source_file.is_file()
            and "__pycache__" not in source_file.parts
            and source_file.suffix != ".pyc"
        ):
            fingerprint.update(str(source_file.relative_to(source_root)).encode())
            fingerprint.update(b"\0")
            fingerprint.update(hashlib.sha256(source_file.read_bytes()).digest())
        else:
            pass
    runtime_root = site_packages / RUNTIME_DIRECTORY / fingerprint.hexdigest()
    if runtime_root.exists():
        verify_files(runtime_root / "sglang", specification, patched=True)
    else:
        runtime_root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runtime_root.parent) as temporary:
            staging_root = Path(temporary) / "runtime"
            package_root = staging_root / "sglang"
            shutil.copytree(
                source_root,
                package_root,
                symlinks=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            for filename in specification.patches:
                patch_path = (patch_directory / filename).resolve()
                subprocess.run(
                    ["git", "apply", str(patch_path)],
                    cwd=package_root,
                    check=True,
                    text=True,
                    env={**os.environ, "GIT_CEILING_DIRECTORIES": str(staging_root)},
                )
            verify_files(package_root, specification, patched=True)
            staging_root.rename(runtime_root)

    activation = site_packages / ACTIVATION_FILE
    temporary_activation = activation.with_suffix(".tmp")
    temporary_activation.write_text(
        f"import sys; sys.path.insert(0, {str(runtime_root)!r})\n"
    )
    temporary_activation.replace(activation)
    return runtime_root / "sglang"


def verify_active_runtime(patch_directory: Path = PATCH_DIRECTORY) -> None:
    """Reject a missing or stale runtime before model allocation and graph capture."""
    module = importlib.util.find_spec("sglang")
    if module is None or module.origin is None:
        raise RuntimeError("Install SGLang and run sgl-omni-prepare-runtime first")
    else:
        specification = RuntimePatchSpec.model_validate_json(
            (patch_directory / "runtime.json").read_bytes()
        )
        verify_files(Path(module.origin).parent, specification, patched=True)


def prepare_command(
    source: Annotated[
        Path | None,
        typer.Option(
            help="Unmodified SGLang package directory; needed for editable installs."
        ),
    ] = None,
    check: Annotated[
        bool, typer.Option(help="Verify the active runtime without writing.")
    ] = False,
) -> None:
    if check:
        verify_active_runtime()
        typer.echo("Prepared SGLang runtime verified")
    else:
        source_root = (
            Path(distribution("sglang").locate_file("sglang"))
            if source is None
            else source
        )
        package_root = prepare_runtime(source_root, Path(sysconfig.get_path("purelib")))
        typer.echo(f"Prepared {package_root}; restart Python to use this runtime")


app = typer.Typer()
app.command()(prepare_command)
