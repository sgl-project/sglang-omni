# SPDX-License-Identifier: Apache-2.0
"""Standalone filesystem API server for the playground."""

from __future__ import annotations

import argparse
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

logger = logging.getLogger(__name__)

MEDIA_SUFFIXES = {
    "audio": {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".webm"},
    "image": {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"},
    "video": {".mp4", ".mov", ".avi", ".mkv", ".webm"},
}


def _default_fs_root() -> Path:
    # Default to container root when SGLANG_OMNI_FS_ROOT is unset.
    return Path("/")


def _default_browse_start_path() -> Path:
    # fs_api.py -> serve -> sglang_omni -> sglang-omni (parent of playground/)
    return Path(__file__).resolve().parents[2]


def _filesystem_root() -> Path:
    # Always allow full container filesystem access.
    return _default_fs_root()


def _browse_start_path() -> Path:
    configured = os.getenv("SGLANG_OMNI_FS_ROOT")
    if configured and configured.strip():
        return Path(configured).resolve()
    return _default_browse_start_path()


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_fs_path(root: Path, raw_path: str | None) -> Path:
    if raw_path is None or raw_path.strip() == "":
        candidate = root
    else:
        requested = Path(raw_path.strip())
        candidate = (
            requested.resolve()
            if requested.is_absolute()
            else (root / requested).resolve()
        )
    if not _is_within_root(candidate, root):
        raise HTTPException(
            status_code=403,
            detail=f"Path is outside allowed root: {root}",
        )
    return candidate


def _classify_media_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    for kind, suffixes in MEDIA_SUFFIXES.items():
        if suffix in suffixes:
            return kind
    return "other"


def create_fs_app() -> FastAPI:
    app = FastAPI(title="sglang-omni-fs", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> JSONResponse:
        root = _filesystem_root()
        browse_start = _browse_start_path()
        return JSONResponse(
            content={
                "status": "healthy",
                "root_path": str(root),
                "browse_start_path": str(browse_start),
                "root_exists": root.exists() and root.is_dir(),
            }
        )

    @app.get("/v1/fs/list")
    async def list_container_files(path: str | None = Query(default=None)) -> JSONResponse:
        root = _filesystem_root()
        current = _resolve_fs_path(root, path)
        if not root.exists() or not root.is_dir():
            raise HTTPException(
                status_code=500,
                detail=f"Configured filesystem root does not exist or is not a directory: {root}",
            )
        if not current.exists():
            raise HTTPException(status_code=404, detail=f"Path not found: {current}")
        if not current.is_dir():
            raise HTTPException(status_code=400, detail=f"Not a directory: {current}")

        entries: list[dict[str, Any]] = []
        for child in sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            try:
                resolved = child.resolve()
            except OSError:
                continue
            if not _is_within_root(resolved, root):
                continue
            is_dir = child.is_dir()
            item: dict[str, Any] = {
                "name": child.name,
                "path": str(resolved),
                "is_dir": is_dir,
                "kind": "dir" if is_dir else _classify_media_kind(child),
            }
            if not is_dir:
                try:
                    item["size"] = child.stat().st_size
                except OSError:
                    item["size"] = None
            entries.append(item)

        parent_path = str(current.parent) if current != root else None
        return JSONResponse(
            content={
                "root_path": str(root),
                "current_path": str(current),
                "parent_path": parent_path,
                "entries": entries,
            }
        )

    @app.get("/v1/fs/file")
    async def read_container_file(path: str = Query(..., min_length=1)) -> FileResponse:
        root = _filesystem_root()
        if not root.exists() or not root.is_dir():
            raise HTTPException(
                status_code=500,
                detail=f"Configured filesystem root does not exist or is not a directory: {root}",
            )
        file_path = _resolve_fs_path(root, path)
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        media_type, _ = mimetypes.guess_type(file_path.name)
        return FileResponse(
            path=file_path,
            media_type=media_type or "application/octet-stream",
            filename=file_path.name,
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone filesystem API server.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--log-level", type=str, default="info")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Default browse start path (sets SGLANG_OMNI_FS_ROOT).",
    )
    args = parser.parse_args()

    if args.root:
        os.environ["SGLANG_OMNI_FS_ROOT"] = args.root

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    uvicorn.run(create_fs_app(), host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
