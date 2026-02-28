import argparse
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filesystem browser API (/v1/fs/*)
# ---------------------------------------------------------------------------

_MEDIA_SUFFIXES = {
    "audio": {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".webm"},
    "image": {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"},
    "video": {".mp4", ".mov", ".avi", ".mkv", ".webm"},
}


def _fs_root() -> Path:
    """Filesystem root — always container root ``/``."""
    return Path("/")


def _fs_resolve(root: Path, raw_path: str | None) -> Path:
    if raw_path is None or raw_path.strip() == "":
        candidate = root
    else:
        requested = Path(raw_path.strip())
        candidate = (
            requested.resolve()
            if requested.is_absolute()
            else (root / requested).resolve()
        )
    try:
        candidate.relative_to(root)
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail=f"Path is outside allowed root: {root}",
        )
    return candidate


def _fs_classify(path: Path) -> str:
    suffix = path.suffix.lower()
    for kind, suffixes in _MEDIA_SUFFIXES.items():
        if suffix in suffixes:
            return kind
    return "other"


def _register_home(app: FastAPI) -> None:
    API_BASE = os.environ.get("SGLANG_OMNI_API_BASE", "")  # empty = use same origin

    @app.get("/")
    async def index():
        html = (FRONTEND_DIR / "index.html").read_text()
        if API_BASE:
            injection = f'<script>window.SGLANG_OMNI_API_BASE = "{API_BASE}";</script>'
            html = html.replace("<head>", f"<head>{injection}", 1)
        return HTMLResponse(html)


def _register_filesystem(app: FastAPI) -> None:
    @app.get("/v1/fs/list")
    async def list_files(
        path: str | None = Query(default=None),
    ) -> JSONResponse:
        """List directory contents for the file browser."""
        root = _fs_root()
        current = _fs_resolve(root, path)
        if not root.exists() or not root.is_dir():
            raise HTTPException(
                status_code=500,
                detail=f"Filesystem root does not exist: {root}",
            )
        if not current.exists():
            raise HTTPException(status_code=404, detail=f"Path not found: {current}")
        if not current.is_dir():
            raise HTTPException(status_code=400, detail=f"Not a directory: {current}")

        entries: list[dict[str, Any]] = []
        for child in sorted(
            current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
        ):
            try:
                resolved = child.resolve()
            except OSError:
                continue
            try:
                resolved.relative_to(root)
            except ValueError:
                continue
            is_dir = child.is_dir()
            item: dict[str, Any] = {
                "name": child.name,
                "path": str(resolved),
                "is_dir": is_dir,
                "kind": "dir" if is_dir else _fs_classify(child),
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
    async def read_file(path: str = Query(..., min_length=1)) -> FileResponse:
        """Download a file from the container filesystem."""
        root = _fs_root()
        if not root.exists() or not root.is_dir():
            raise HTTPException(
                status_code=500,
                detail=f"Filesystem root does not exist: {root}",
            )
        file_path = _fs_resolve(root, path)
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        media_type, _ = mimetypes.guess_type(file_path.name)
        return FileResponse(
            path=file_path,
            media_type=media_type or "application/octet-stream",
            filename=file_path.name,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SGLang-Omni Playground")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


app = FastAPI(title="sglang-omni-playground")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
_register_filesystem(app)
_register_home(app)

FRONTEND_DIR = Path(__file__).parent / "frontend"
assert FRONTEND_DIR.is_dir(), "Frontend directory does not exist"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True))
logger.info("Serving playground UI from %s", FRONTEND_DIR)

args = parse_args()
uvicorn.run(app, host="0.0.0.0", port=args.port)
