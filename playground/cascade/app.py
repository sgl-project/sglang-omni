# SPDX-License-Identifier: Apache-2.0
"""Serve the streaming cascade demo with a transparent SGLang-Omni realtime connection."""

import argparse
import asyncio
import logging
from pathlib import Path

from aiohttp import ClientError, ClientSession, ClientTimeout, WSMsgType, web

ROOT = Path(__file__).parent
FRONTEND = ROOT / "frontend"
# Note (ruoyu): the settings panel is the realtime page's module, served under this page's own path so its ./session.js import resolves to the cascade session.
SHARED_SETTINGS_MODULE = ROOT.parent / "realtime" / "frontend" / "orb" / "settings.js"
logger = logging.getLogger(__name__)


def create_app(api_base: str) -> web.Application:
    app = web.Application()

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

    async def settings_module(request: web.Request) -> web.FileResponse:
        return web.FileResponse(SHARED_SETTINGS_MODULE)

    app.router.add_get("/v1/realtime/capabilities", capabilities)
    app.router.add_get("/v1/realtime", realtime)
    app.router.add_get("/", index)

    app.router.add_get("/orb", index)
    # Note (ruoyu): aiohttp matches routes in the order they are added, so this must come before the static mount.
    app.router.add_get("/static/orb/settings.js", settings_module)
    app.router.add_static("/static/", FRONTEND)
    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    web.run_app(create_app(args.api_base.rstrip("/")), host=args.host, port=args.port)
