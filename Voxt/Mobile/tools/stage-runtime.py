#!/usr/bin/env python3
"""Stage an already-built runtime; never build, execute, or sign either artifact.

Run before signing the helper and then the containing app. All temporary writes
stay inside the destination bundle.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path


def stage(app: Path, runtime: Path) -> None:
    if (
        not (app / "Contents/MacOS").is_dir()
        or not (app / "Contents/Info.plist").is_file()
    ):
        raise ValueError("App bundle is missing")
    if not runtime.is_file() or not os.access(runtime, os.X_OK):
        raise ValueError("Runtime is missing or not executable")
    helpers = app / "Contents/Helpers"
    if helpers.is_symlink():
        raise ValueError("Helpers must be a directory inside the bundle")
    helpers.mkdir(exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".runtime-", dir=helpers)
    os.close(descriptor)
    try:
        shutil.copyfile(runtime, temporary)
        os.chmod(temporary, 0o755)
        os.replace(temporary, helpers / "voxt-tailcat-host")
    finally:
        Path(temporary).unlink(missing_ok=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: stage-runtime.py APP RUNTIME")
    try:
        stage(Path(sys.argv[1]), Path(sys.argv[2]))
    except (OSError, ValueError) as error:
        raise SystemExit(str(error))
