import argparse
import json
import os
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

FRONTEND_DIR = Path(__file__).parent
assert FRONTEND_DIR.is_dir(), "Frontend directory does not exist"


def _load_ice_config() -> dict[str, object]:
    ice_urls = [
        value.strip()
        for value in os.environ.get("SGLANG_OMNI_ICE_URLS", "").split(",")
        if value.strip()
    ]
    return {
        "urls": ice_urls,
        "username": os.environ.get("SGLANG_OMNI_ICE_USERNAME") or None,
        "credential": os.environ.get("SGLANG_OMNI_ICE_CREDENTIAL") or None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SGLang-Omni Realtime Playground")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--api-base", type=str, default=None)
    return parser.parse_args()


def create_app(api_base: str | None) -> FastAPI:
    app = FastAPI(title="sglang-omni-realtime-playground")
    app.state.api_base = (
        api_base or os.environ.get("SGLANG_OMNI_API_BASE", "http://localhost:8000")
    ).rstrip("/")
    app.state.ice_config = _load_ice_config()

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
        injection = (
            "<script>"
            'window.SGLANG_OMNI_API_BASE = "";'
            f"window.SGLANG_OMNI_ICE_CONFIG = {json.dumps(app.state.ice_config)};"
            "</script>"
        )
        html = html.replace("<head>", f"<head>{injection}", 1)
        return HTMLResponse(html)

    async def _proxy(method: str, path: str, *, body: bytes | None = None) -> Response:
        target = f"{app.state.api_base}{path}"
        headers = {"content-type": "application/json"} if body else None
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                upstream = await client.request(
                    method, target, content=body, headers=headers
                )
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to reach realtime backend at {app.state.api_base}: {exc}",
            ) from exc

        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
        )

    @app.post("/v1/realtime/webrtc/offer")
    async def realtime_offer(request: Request) -> Response:
        return await _proxy(
            "POST", "/v1/realtime/webrtc/offer", body=await request.body()
        )

    @app.delete("/v1/realtime/sessions/{session_id}")
    async def close_session(session_id: str) -> Response:
        return await _proxy(
            "DELETE",
            f"/v1/realtime/sessions/{session_id}",
        )

    @app.get("/health")
    async def health() -> Response:
        return await _proxy("GET", "/health")

    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True))
    return app


def main() -> None:
    args = parse_args()
    app = create_app(args.api_base)
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
