# SPDX-License-Identifier: Apache-2.0
"""Serve the streaming cascade demo with a transparent SGLang-Omni realtime connection."""

import argparse
import asyncio
import json
import logging
import os
import struct
from pathlib import Path

import numpy as np
from aiohttp import ClientError, ClientSession, ClientTimeout, WSMsgType, web

ROOT = Path(__file__).parent
FRONTEND = ROOT / "frontend"
# Note (ruoyu): the settings panel is the realtime page's module, served under this page's own path so its ./session.js import resolves to the cascade session.
SHARED_SETTINGS_MODULE = ROOT.parent / "realtime" / "frontend" / "orb" / "settings.js"
SHARED_MEDIA_GATE_MODULE = (
    ROOT.parent / "realtime" / "frontend" / "orb" / "reply_media_gate.js"
)
logger = logging.getLogger(__name__)

# SMPL-X (https://smpl-x.is.tue.mpg.de) is licensed for non-commercial research only, so no model data lives in this repository: given a local
# SMPLX_NEUTRAL_2020.npz (--smplx-model or SMPLX_MODEL_PATH), the page gets what linear blend skinning needs at /smplx/mesh.bin: the template
# vertices and triangles, the rest joints the joint regressor puts on the template (no shape), the kinematic tree, each vertex's 4 largest
# skinning weights renormalized, and the expression blend shapes (float16) of the vertices they move; no pose correctives. The blob is a uint32
# header length, a JSON header naming each array's dtype, shape and byte offset, padded to 4 bytes, then the arrays at 4-byte aligned offsets.
SKIN_INFLUENCES = 4
NUM_EXPRESSIONS = 100
# SMPL-X 2020 shapedirs: 300 shape directions, then 100 expression directions.
SHAPE_DIRECTIONS = 300


def smplx_mesh_arrays(model_path: str | Path) -> dict[str, np.ndarray]:
    """The arrays the browser skins the mesh with, from an SMPL-X .npz model."""
    model = np.load(model_path, allow_pickle=True)
    template = model["v_template"].astype(np.float32)
    weights = model["weights"].astype(np.float32)
    top = np.argsort(-weights, axis=1)[:, :SKIN_INFLUENCES]
    top_weights = np.take_along_axis(weights, top, axis=1)
    top_weights /= top_weights.sum(axis=1, keepdims=True)
    parents = model["kintree_table"][0].astype(np.int64)
    parents[0] = -1
    arrays = {
        "vertices": template,
        "faces": model["f"].astype(np.uint32),
        "joints": (model["J_regressor"].astype(np.float32) @ template).astype(
            np.float32
        ),
        "parents": parents.astype(np.int32),
        "skin_indices": top.astype(np.uint8),
        "skin_weights": top_weights.astype(np.float32),
    }
    shapedirs = model["shapedirs"]
    if shapedirs.shape[2] >= SHAPE_DIRECTIONS + NUM_EXPRESSIONS:
        expression = shapedirs[
            :, :, SHAPE_DIRECTIONS : SHAPE_DIRECTIONS + NUM_EXPRESSIONS
        ]
        moved = np.flatnonzero(np.abs(expression).max(axis=(1, 2)) > 1e-7)
        arrays["expression_vertices"] = moved.astype(np.uint32)
        arrays["expression_dirs"] = expression[moved].astype(np.float16)
    else:
        pass
    return arrays


def pack_arrays(arrays: dict[str, np.ndarray]) -> bytes:
    """One blob: uint32 header length, the JSON header, padding to 4 bytes, then each array at its offset (4-byte aligned)."""
    header: dict[str, dict] = {}
    offset = 0
    for name, array in arrays.items():
        header[name] = {
            "dtype": array.dtype.str.lstrip("<|="),
            "shape": list(array.shape),
            "offset": offset,
        }
        offset += -(-array.nbytes // 4) * 4
    text = json.dumps(header).encode()
    text += b" " * (-(len(text) + 4) % 4)
    body = bytearray(offset)
    for name, array in arrays.items():
        start = header[name]["offset"]
        data = (
            np.ascontiguousarray(array).astype(array.dtype.newbyteorder("<")).tobytes()
        )
        body[start : start + len(data)] = data
    return struct.pack("<I", len(text)) + text + bytes(body)


def smplx_mesh_blob(model_path: str | Path) -> bytes:
    return pack_arrays(smplx_mesh_arrays(model_path))


def create_app(api_base: str, smplx_model: str | None = None) -> web.Application:
    """smplx_model: a local SMPL-X .npz the page draws smplx_motion and smplh_flame_motion on as a mesh; without it, a stick figure."""
    app = web.Application()
    smplx_mesh: dict[str, bytes] = {}

    async def capabilities(request: web.Request) -> web.Response:
        try:
            async with ClientSession(timeout=ClientTimeout(total=10)) as client:
                async with client.get(
                    f"{api_base}/v1/realtime/capabilities"
                ) as response:
                    return web.Response(
                        body=await response.read(),
                        status=response.status,
                        content_type="application/json",
                    )
        except (ClientError, TimeoutError) as error:
            return web.json_response({"error": str(error)}, status=502)

    async def realtime(request: web.Request) -> web.WebSocketResponse:
        browser = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)
        await browser.prepare(request)
        try:
            async with ClientSession() as client:
                async with client.ws_connect(
                    f"{api_base}/v1/realtime", max_msg_size=16 * 1024 * 1024
                ) as upstream:

                    async def send_input() -> None:
                        async for message in browser:
                            if message.type == WSMsgType.TEXT:
                                await upstream.send_str(message.data)
                            else:
                                break

                    async def receive_output() -> None:
                        async for message in upstream:
                            if message.type == WSMsgType.TEXT:
                                await browser.send_str(message.data)
                            else:
                                break

                    tasks = [
                        asyncio.create_task(send_input()),
                        asyncio.create_task(receive_output()),
                    ]
                    try:
                        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        for task in tasks:
                            task.cancel()
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                    for result in results:
                        if isinstance(result, Exception):
                            raise result
        except (ClientError, TimeoutError, ConnectionError) as error:
            if not browser.closed:
                await browser.send_json(
                    {"type": "error", "error": {"message": str(error)}}
                )
        finally:
            await browser.close()
        return browser

    async def index(request: web.Request) -> web.FileResponse:
        return web.FileResponse(FRONTEND / "orb" / "index.html")

    async def smplx_mesh_asset(request: web.Request) -> web.Response:
        if not smplx_model:
            raise web.HTTPNotFound(text="no SMPL-X model: start with --smplx-model")
        elif "blob" not in smplx_mesh:
            smplx_mesh["blob"] = await asyncio.to_thread(smplx_mesh_blob, smplx_model)
        else:
            pass
        return web.Response(
            body=smplx_mesh["blob"],
            content_type="application/octet-stream",
            headers={"Cache-Control": "no-store"},
        )

    async def settings_module(request: web.Request) -> web.FileResponse:
        return web.FileResponse(SHARED_SETTINGS_MODULE)

    async def media_gate_module(request: web.Request) -> web.FileResponse:
        return web.FileResponse(SHARED_MEDIA_GATE_MODULE)

    app.router.add_get("/v1/realtime/capabilities", capabilities)
    app.router.add_get("/v1/realtime", realtime)
    app.router.add_get("/", index)

    app.router.add_get("/orb", index)
    app.router.add_get("/smplx/mesh.bin", smplx_mesh_asset)
    # Note (ruoyu): aiohttp matches routes in the order they are added, so this must come before the static mount.
    app.router.add_get("/static/orb/settings.js", settings_module)
    app.router.add_get("/static/orb/reply_media_gate.js", media_gate_module)
    app.router.add_static("/static/", FRONTEND)
    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--smplx-model",
        default=os.environ.get("SMPLX_MODEL_PATH"),
        help="a local SMPL-X .npz (e.g. SMPLX_NEUTRAL_2020.npz; non-commercial license, not shipped) to draw body motion as a mesh; default: $SMPLX_MODEL_PATH",
    )
    args = parser.parse_args()
    web.run_app(
        create_app(args.api_base.rstrip("/"), args.smplx_model),
        host=args.host,
        port=args.port,
    )
