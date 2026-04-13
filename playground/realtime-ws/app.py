import argparse
import json
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

FRONTEND_DIR = Path(__file__).parent
assert FRONTEND_DIR.is_dir(), "Frontend directory does not exist"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SGLang-Omni Realtime WS Playground")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--api-base", type=str, default=None)
    return parser.parse_args()


def create_app(api_base: str | None) -> FastAPI:
    app = FastAPI(title="sglang-omni-realtime-ws-playground")
    app.state.api_base = (
        api_base or os.environ.get("SGLANG_OMNI_API_BASE", "http://localhost:8000")
    ).rstrip("/")

    @app.get("/")
    async def index() -> HTMLResponse:
        html = (FRONTEND_DIR / "index.html").read_text()
        injection = (
            "<script>"
            f"window.SGLANG_OMNI_API_BASE = {json.dumps(app.state.api_base)};"
            "</script>"
        )
        html = html.replace("<head>", f"<head>{injection}", 1)
        return HTMLResponse(html)

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "api_base": app.state.api_base})

    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True))
    return app


def main() -> None:
    args = parse_args()
    uvicorn.run(create_app(args.api_base), host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
