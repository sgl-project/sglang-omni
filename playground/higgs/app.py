# SPDX-License-Identifier: Apache-2.0
"""FastAPI app for the Higgs Audio v3 TTS playground.

Serves a static HTML/CSS/JS frontend and proxies synthesis requests to a
running sglang-omni backend. Reference audio uploads are persisted to a
temporary directory and forwarded to the backend as local file paths.
"""

from __future__ import annotations

import argparse
import atexit
import logging
import shutil
import tempfile
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from playground.http_utils import register_playground_favicon

logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).parent / "frontend"

_UPLOAD_ROOT = Path(tempfile.mkdtemp(prefix="sglang-omni-higgs-uploads-"))
atexit.register(lambda: shutil.rmtree(_UPLOAD_ROOT, ignore_errors=True))


def _parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


async def _save_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "ref.wav").suffix or ".wav"
    fd, raw_path = tempfile.mkstemp(dir=_UPLOAD_ROOT, suffix=suffix)
    path = Path(raw_path)
    try:
        with path.open("wb") as f:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    finally:
        import os

        os.close(fd)
    return str(path)


def _build_payload(
    *,
    text: str,
    ref_audio_path: str | None,
    ref_audio_url: str | None,
    ref_text: str,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    max_new_tokens: int | None,
    seed: int | None,
    stream: bool,
) -> dict:
    if not text.strip():
        raise HTTPException(status_code=400, detail="Input text is empty.")

    payload: dict = {"input": text}
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    if max_new_tokens is not None:
        payload["max_new_tokens"] = max_new_tokens
    if seed is not None:
        payload["seed"] = seed
    if stream:
        payload["stream"] = True
        payload["response_format"] = "pcm"

    # Precedence: if the user supplies both an uploaded file and a URL,
    # the uploaded file wins. Documented in the UI as well.
    reference_source = ref_audio_path or ref_audio_url
    if reference_source:
        reference: dict = {"audio_path": reference_source}
        if ref_text.strip():
            reference["text"] = ref_text.strip()
        payload["references"] = [reference]

    return payload


def create_app(api_base: str) -> FastAPI:
    api_base = api_base.rstrip("/")
    app = FastAPI(title="sglang-omni-higgs-playground")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_playground_favicon(app, frontend_dir=FRONTEND_DIR)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        html = (FRONTEND_DIR / "index.html").read_text()
        # Cache-bust frontend assets by stamping each /static/<asset> URL with
        # the file's mtime — otherwise browsers happily serve a stale app.js
        # after we ship a UI fix.
        for asset in ("app.js", "styles.css"):
            path = FRONTEND_DIR / asset
            if path.exists():
                v = int(path.stat().st_mtime)
                html = html.replace(f"/static/{asset}", f"/static/{asset}?v={v}")
        return HTMLResponse(
            html, headers={"Cache-Control": "no-cache, must-revalidate"}
        )

    @app.get("/healthz")
    async def healthz() -> dict:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{api_base}/health")
                backend_ok = resp.status_code == 200
        except Exception:
            backend_ok = False
        return {"backend": "ok" if backend_ok else "down", "api_base": api_base}

    @app.post("/api/synthesize")
    async def synthesize(
        text: str = Form(...),
        ref_audio: UploadFile | None = File(None),
        ref_audio_url: str = Form(""),
        ref_text: str = Form(""),
        temperature: str = Form(""),
        top_p: str = Form(""),
        top_k: str = Form(""),
        max_new_tokens: str = Form(""),
        seed: str = Form(""),
    ) -> Response:
        ref_audio_path = None
        if ref_audio is not None and (ref_audio.filename or "").strip():
            ref_audio_path = await _save_upload(ref_audio)

        payload = _build_payload(
            text=text,
            ref_audio_path=ref_audio_path,
            ref_audio_url=ref_audio_url.strip() or None,
            ref_text=ref_text,
            temperature=_parse_optional_float(temperature),
            top_p=_parse_optional_float(top_p),
            top_k=_parse_optional_int(top_k),
            max_new_tokens=_parse_optional_int(max_new_tokens),
            seed=_parse_optional_int(seed),
            stream=False,
        )

        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(f"{api_base}/v1/audio/speech", json=payload)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Backend error: {exc}")

        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        return Response(
            content=resp.content,
            media_type=resp.headers.get("content-type", "audio/wav"),
        )

    @app.post("/api/synthesize/stream")
    async def synthesize_stream(
        text: str = Form(...),
        ref_audio: UploadFile | None = File(None),
        ref_audio_url: str = Form(""),
        ref_text: str = Form(""),
        temperature: str = Form(""),
        top_p: str = Form(""),
        top_k: str = Form(""),
        max_new_tokens: str = Form(""),
        seed: str = Form(""),
    ) -> StreamingResponse:
        ref_audio_path = None
        if ref_audio is not None and (ref_audio.filename or "").strip():
            ref_audio_path = await _save_upload(ref_audio)

        payload = _build_payload(
            text=text,
            ref_audio_path=ref_audio_path,
            ref_audio_url=ref_audio_url.strip() or None,
            ref_text=ref_text,
            temperature=_parse_optional_float(temperature),
            top_p=_parse_optional_float(top_p),
            top_k=_parse_optional_int(top_k),
            max_new_tokens=_parse_optional_int(max_new_tokens),
            seed=_parse_optional_int(seed),
            stream=True,
        )

        client = httpx.AsyncClient(timeout=None)
        try:
            backend_request = client.build_request(
                "POST", f"{api_base}/v1/audio/speech", json=payload
            )
            resp = await client.send(backend_request, stream=True)
        except httpx.HTTPError as exc:
            await client.aclose()
            raise HTTPException(status_code=502, detail=f"Backend error: {exc}")

        if resp.status_code != 200:
            err = await resp.aread()
            await resp.aclose()
            await client.aclose()
            raise HTTPException(
                status_code=resp.status_code,
                detail=err.decode("utf-8", "replace"),
            )

        async def proxy_audio():
            try:
                async for chunk in resp.aiter_raw():
                    if chunk:
                        yield chunk
            finally:
                await resp.aclose()
                await client.aclose()

        return StreamingResponse(
            proxy_audio(),
            media_type=resp.headers.get("content-type", "audio/pcm"),
            headers={
                "Cache-Control": "no-store",
                "X-Accel-Buffering": "no",
                "X-Sample-Rate": resp.headers.get("x-sample-rate", "24000"),
                "X-Channels": resp.headers.get("x-channels", "1"),
                "X-Bit-Depth": resp.headers.get("x-bit-depth", "16"),
            },
        )

    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR), html=False),
        name="static",
    )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Higgs Audio v3 TTS Playground")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = create_app(args.api_base)
    uvicorn.run(app, host=args.host, port=args.port)
