import argparse
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

FRONTEND_DIR = Path(__file__).parent
assert FRONTEND_DIR.is_dir(), "Frontend directory does not exist"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SGLang-Omni Realtime Playground")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--api-base", type=str, default=None)
    return parser.parse_args()


def create_app(api_base: str | None) -> FastAPI:
    app = FastAPI(title="sglang-omni-realtime-playground")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def index() -> HTMLResponse:
        html = (FRONTEND_DIR / "index.html").read_text()
        effective_api_base = api_base or os.environ.get(
            "SGLANG_OMNI_API_BASE", "http://localhost:8000"
        )
        injection = (
            f'<script>window.SGLANG_OMNI_API_BASE = "{effective_api_base}";</script>'
        )
        html = html.replace("<head>", f"<head>{injection}", 1)
        return HTMLResponse(html)

    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True))
    return app


def main() -> None:
    args = parse_args()
    app = create_app(args.api_base)
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
