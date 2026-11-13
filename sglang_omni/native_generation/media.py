# SPDX-License-Identifier: Apache-2.0
"""Serve the native SGLang media routes of a generation stage from that stage.

A pipeline whose generation stage runs native multimodal_gen lists
ROUTE_PREFIXES in that stage's ``http_routes``, and the stage's scheduler
returns a :class:`NativeMediaApp` from ``http_app()``. The routes then run in
the generation stage process, which already holds the native runtime, start
with the stage, and send requests straight to the native scheduler the stage
owns. The Omni HTTP process relays them (sglang_omni.serve.stage_routes) and
never imports native SGLang.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, suppress
from urllib.parse import unquote, urlparse

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.types import Receive, Scope, Send

from sglang_omni.native_generation import ROUTE_PREFIXES
from sglang_omni.preprocessing.resource_connector import (
    MediaPolicyError,
    MultiModalResourceConnector,
    get_global_resource_connector,
)
from sglang_omni.serve.stage_routes import server_port
from sglang_omni.utils.port_claim import claim_tcp_port

logger = logging.getLogger(__name__)

__all__ = ["ROUTE_PREFIXES", "NativeMediaApp", "native_media_server_kwargs"]

NATIVE_HOST = "127.0.0.1"
LOOPBACK_HOSTS = (NATIVE_HOST, "localhost")
# Fields of those routes that carry a media path or URL, besides the *_path
# request fields the served model declares. Action observations use image,
# images and video.
MEDIA_REFERENCE_FIELDS = frozenset(
    {
        "image",
        "images",
        "input_reference",
        "reference_url",
        "url",
        "url[]",
        "video",
        "video_url",
    }
)
# Keys of the image payloads that the native action routes decode in memory
# instead of opening a file.
INLINE_IMAGE_KEYS = frozenset({"b64_json", "base64", "data", "values"})
# Native names a saved upload after its file name, and an action upload after
# the request id too, so a separator there would leave the upload directory.
ACTION_UPLOAD_ID_FIELDS = frozenset({"request_id", "id"})
# Containers native SGLang merges into a request, as JSON objects or strings.
EXTRA_CONTAINERS = frozenset({"extra_args", "extra_body", "extra_json", "extra_params"})


def claim_native_port(avoid: set[int]) -> int:
    """A port claimed across processes for this one, outside ``avoid``.

    A port read free and released is not reserved: on a busy host any
    process, outgoing connections included, can take it before native's
    strict check. Claimed ports sit below the ephemeral range and stay held
    until this process exits, which outlives the native processes it starts.
    """
    while True:
        port = claim_tcp_port()
        if port not in avoid:
            return port
        else:
            pass


def claim_native_ports(count: int, avoid: set[int]) -> int:
    """The first of ``count`` consecutive ports claimed for this process.

    Claims count up from the bottom of their range, so a run cut short by a
    port another process holds starts again after it. Ports passed over stay
    claimed until this process exits.
    """
    run: list[int] = []
    while len(run) < count:
        port = claim_native_port(avoid)
        run = run + [port] if run and port == run[-1] + 1 else [port]
    return run[0]


def native_media_server_kwargs(
    overrides: dict[str, object] | None, *, output_dir: str
) -> dict[str, object]:
    """Settle the native server settings the media routes rely on.

    The native scheduler binds its request socket on loopback and unpickles
    what it receives, so it stays there whatever host the Omni server binds.
    Its ports are explicit and strict, so startup fails instead of native
    moving to another port. Native checks the HTTP port, one scheduler port
    per replica counting up from scheduler_port, and the master port. Those
    the overrides leave unset are claimed across processes, away from the set
    ones. The HTTP port is the Omni server's, which it binds only after its
    stages started, or a claimed one when no Omni server runs, since the stage
    serves the routes itself. Native routes keep uploads in a temporary
    directory for their request or video job instead of one that grows with
    every request, and save media under the stage's output directory.
    """
    kwargs = dict(overrides or {})
    if kwargs.get("host", NATIVE_HOST) not in LOOPBACK_HOSTS:
        raise ValueError(
            "The native generation scheduler listens on loopback only. Remove "
            "host from the generation stage's factory.server_args_overrides."
        )
    else:
        pass
    http_port = server_port()
    if http_port is not None:
        kwargs["port"] = http_port
    else:
        pass
    replicas = int(kwargs.get("dp_size") or 1)
    taken = {
        int(kwargs[name])
        for name in ("port", "master_port")
        if kwargs.get(name) is not None
    }
    if kwargs.get("scheduler_port") is None:
        kwargs["scheduler_port"] = claim_native_ports(replicas, taken)
    else:
        pass
    first = int(kwargs["scheduler_port"])
    taken.update(range(first, first + replicas))
    if "master_port" not in kwargs:
        kwargs["master_port"] = claim_native_port(taken)
        taken.add(kwargs["master_port"])
    else:
        pass
    if kwargs.get("port") is None:
        kwargs["port"] = claim_native_port(taken)
    else:
        pass
    kwargs.setdefault("input_save_path", None)
    kwargs.update(host=NATIVE_HOST, strict_ports=True, output_path=output_dir)
    return kwargs


class NativeMediaApp:
    """The native media routes of one generation stage, served from the stage.

    Only the image, video and action routers are served: native create_app
    also serves weights, memory, LoRA, rollout, mesh, realtime, server info
    and stats routes without Omni's admin key, and the action websocket stays
    out. The app reuses the native scheduler client and video job shutdown,
    and starts no native broker, since nothing in Omni sends offline jobs
    through one. Its lifespan starts the scheduler client and the prompt
    enhancer before the stage reports ready, so a failure to start fails the
    stage with its cause.

    A request in flight fails with 503 as soon as a native scheduler socket
    closes, which happens when its process exits, because native SGLang before
    worker_failure support would wait for the answer forever.
    """

    def __init__(self, server_args: object, request_fields: frozenset[str]) -> None:
        from sglang.multimodal_gen.runtime.entrypoints.action import api as action_api
        from sglang.multimodal_gen.runtime.entrypoints.openai import (
            image_api,
            video_api,
        )
        from sglang.multimodal_gen.runtime.server_args import set_global_server_args

        set_global_server_args(server_args)
        self.server_args = server_args
        self.media_fields = frozenset(f for f in request_fields if f.endswith("_path"))
        # Last monitor event of each native scheduler endpoint.
        self.scheduler_events: dict[bytes, int] = {}
        self.unavailable_reason: str | None = None
        self.unavailable = asyncio.Event()
        app = FastAPI(lifespan=self.lifespan)
        app.state.server_args = server_args
        app.state.prompt_enhancer = None
        police = [Depends(self.police_request)]
        app.include_router(image_api.router, dependencies=police)
        app.include_router(video_api.router, dependencies=police)
        if server_args.pipeline_config.supports_action_endpoint():
            actions = APIRouter()
            actions.routes.extend(
                route
                for route in action_api.router.routes
                if isinstance(route, APIRoute)
            )
            app.include_router(actions, dependencies=police)
        else:
            pass
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
        elif self.unavailable.is_set():
            await self.refuse(scope, receive, send, self.unavailable_reason)
        else:
            await self.serve_while_available(scope, receive, send)

    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncIterator[None]:
        from sglang.multimodal_gen.runtime.entrypoints.openai.prompt_enhancement import (
            PromptEnhancer,
        )
        from sglang.multimodal_gen.runtime.entrypoints.openai.video_api import (
            shutdown_video_jobs,
        )
        from sglang.multimodal_gen.runtime.scheduler_client import (
            async_scheduler_client,
        )

        async_scheduler_client.initialize(self.server_args)
        watch = enhancer = None
        # A failure before serving still releases what started.
        try:
            watch = asyncio.create_task(self.watch_scheduler())
            config = self.server_args.prompt_enhancer_config
            enhancer = None if config is None else PromptEnhancer.from_file(config)
            app.state.prompt_enhancer = enhancer
            yield
        finally:
            if watch is not None:
                watch.cancel()
                with suppress(asyncio.CancelledError):
                    await watch
            else:
                pass
            await shutdown_video_jobs()
            if enhancer is not None:
                await enhancer.close()
            else:
                pass
            async_scheduler_client.close()

    async def police_request(self, request: Request) -> None:
        """Apply the server media policy before a native route reads a request.

        Fields that name server paths are refused. Without a configured policy,
        media paths and URLs pass as on other routes. With one, local paths and
        file URLs must resolve inside the allowed directory, and remote URLs are
        refused while a domain allowlist is set, because native SGLang fetches
        them itself and follows redirects without checking each hop.
        """
        content_type = request.headers.get("content-type", "")
        if request.method not in ("PATCH", "POST", "PUT"):
            return
        elif "multipart/form-data" in content_type or "urlencoded" in content_type:
            fields: dict[str, list[object]] = {}
            for name, value in (await request.form()).multi_items():
                fields.setdefault(name, []).append(value)
                if isinstance(value, str):
                    if (
                        name in ACTION_UPLOAD_ID_FIELDS
                        and "/" in value
                        and request.url.path.startswith("/v1/actions/")
                    ):
                        raise refused_upload_name(name)
                    else:
                        pass
                elif "/" in (value.filename or ""):
                    raise refused_upload_name(name)
                else:
                    pass
            body: object = fields
        elif "msgpack" in content_type:
            import msgpack

            try:
                body = msgpack.unpackb(
                    await request.body(), raw=False, strict_map_key=False
                )
            except ValueError:
                return
        else:
            try:
                body = await request.json()
            except ValueError:
                return
            except RecursionError:
                raise nested_too_deeply() from None
        try:
            self.police_fields(
                body, inline_images=request.url.path.startswith("/v1/actions/")
            )
        except RecursionError:
            raise nested_too_deeply() from None

    def police_fields(self, value: object, inline_images: bool = False) -> None:
        if isinstance(value, dict):
            for name, item in value.items():
                name = str(name)
                if name in EXTRA_CONTAINERS:
                    item = parse_containers(item)
                else:
                    pass
                if name.endswith("_path") and name not in self.media_fields:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Native media requests cannot set {name}, "
                        "a path the server owns.",
                    )
                elif name in self.media_fields or name in MEDIA_REFERENCE_FIELDS:
                    police_media_value(name, item, inline_images)
                else:
                    pass
                self.police_fields(item, inline_images)
        elif isinstance(value, list):
            for item in value:
                self.police_fields(item, inline_images)
        else:
            pass

    def update_availability(self) -> None:
        import zmq

        if zmq.EVENT_DISCONNECTED in self.scheduler_events.values():
            self.unavailable_reason = "The native generation scheduler is unreachable"
        else:
            self.unavailable_reason = None
        if self.unavailable_reason is None:
            self.unavailable.clear()
        else:
            logger.error("Native media routes unavailable: %s", self.unavailable_reason)
            self.unavailable.set()

    async def watch_scheduler(self) -> None:
        """Follow the connection to each native scheduler socket.

        A scheduler process that exits closes its socket, and the socket
        monitor reports that at once, without polling. Reconnecting, which
        only a live scheduler accepts, makes the routes available again.
        """
        import zmq
        import zmq.asyncio
        from zmq.utils.monitor import recv_monitor_message

        context = zmq.asyncio.Context()
        probe = context.socket(zmq.DEALER)
        monitor = probe.get_monitor_socket(zmq.EVENT_CONNECTED | zmq.EVENT_DISCONNECTED)
        try:
            for endpoint in self.server_args.scheduler_endpoints:
                probe.connect(endpoint)
            while True:
                event = await recv_monitor_message(monitor)
                self.scheduler_events[event["endpoint"]] = event["event"]
                self.update_availability()
        finally:
            probe.disable_monitor()
            monitor.close(linger=0)
            probe.close(linger=0)
            context.term()

    @staticmethod
    async def refuse(scope: Scope, receive: Receive, send: Send, reason: str) -> None:
        await JSONResponse({"detail": reason}, status_code=503)(scope, receive, send)

    async def serve_while_available(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        started = False

        async def send_tracked(message: dict[str, object]) -> None:
            nonlocal started
            started = started or message["type"] == "http.response.start"
            await send(message)

        request = asyncio.ensure_future(self.app(scope, receive, send_tracked))
        unavailable = asyncio.ensure_future(self.unavailable.wait())
        try:
            await asyncio.wait(
                (request, unavailable), return_when=asyncio.FIRST_COMPLETED
            )
            reason = self.unavailable_reason or "Native generation stopped"
        finally:
            unavailable.cancel()
            if not request.done():
                request.cancel()
                with suppress(asyncio.CancelledError):
                    await request
            else:
                pass
        if not request.cancelled():
            request.result()
        elif not started:
            await self.refuse(scope, receive, send, reason)
        else:
            # The response already started, so the connection just closes.
            pass


def refused_upload_name(name: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=f"{name} refused: an upload name cannot contain '/'.",
    )


def nested_too_deeply() -> HTTPException:
    # Native may still decode what nests too deeply for this check.
    return HTTPException(
        status_code=400,
        detail="Native media request refused: it nests too deeply to check.",
    )


def parse_containers(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    elif isinstance(value, list):
        return [parse_containers(item) for item in value]
    else:
        return value


def media_references(
    value: object, inline_images: bool = False
) -> Iterator[tuple[str, bool]]:
    """Each string native SGLang may read a media reference from, and whether
    native can open it only as written.

    Native decodes JSON encoded request values, multipart fields once and
    model fields such as control_path once more, so a string also stands
    for every string it decodes to. JSON text never starts with a slash, a
    tilde or a scheme, so as written it can only be a relative path. With
    inline_images, image payloads that native decodes in memory hold no
    reference.
    """
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            yield value, False
            return
        yield value, True
        yield from decoded_references(decoded, top=True)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from media_references(item, inline_images)
    elif isinstance(value, dict):
        if inline_images and INLINE_IMAGE_KEYS.intersection(value):
            return
        else:
            pass
        for item in value.values():
            yield from media_references(item, inline_images)
    else:
        pass


def decoded_references(value: object, *, top: bool) -> Iterator[tuple[str, bool]]:
    """References in a value native decoded from a string.

    Native reads decoded strings as paths and turns any other decoded value,
    except a list of paths, into its text, as Cosmos3 lowering does for the
    items of control_path.
    """
    if isinstance(value, str):
        yield from media_references(value)
    elif value is not None:
        if not (top and isinstance(value, list)):
            yield str(value), True
        else:
            pass
        if isinstance(value, (dict, list)):
            items = value.values() if isinstance(value, dict) else value
            for item in items:
                yield from decoded_references(item, top=False)
        else:
            pass
    else:
        pass


def police_media_value(name: str, value: object, inline_images: bool) -> None:
    """Refuse a media field value that the server media policy does not allow.

    Without a policy every reference passes, as on other routes.
    """
    connector = get_global_resource_connector()
    if (
        connector.allowed_local_media_path is None
        and not connector.allowed_media_domains
    ):
        return
    else:
        pass
    for source, as_written in media_references(value, inline_images):
        police_media_reference(connector, name, source, as_written)


def police_media_reference(
    connector: MultiModalResourceConnector, name: str, source: str, as_written: bool
) -> None:
    """Refuse a media path or URL that the server media policy does not allow.

    Follows the rules main applies to request media: a bare path or file URL
    must sit in the allowed directory, here in every reading native SGLang
    may open it in. Inline media should use data URLs.
    """
    remote = source.lower().startswith(("http://", "https://"))
    file_url = source[:7].lower() == "file://"
    try:
        if remote and connector.allowed_media_domains:
            raise MediaPolicyError(
                "remote media is refused on native media routes while "
                "--allowed-media-domain is set, since native SGLang fetches "
                "it without checking each redirect. Send it as an upload or "
                "a data URL."
            )
        elif connector.allowed_local_media_path is not None:
            for path in local_readings(source, as_written or remote, file_url):
                connector.local_media_path(path)
        else:
            pass
    except ValueError as exc:
        # A media policy refusal, or a path that cannot be resolved at all,
        # such as one with a NUL character.
        hint = " Send local media as a path." if file_url else ""
        raise HTTPException(
            status_code=400, detail=f"{name} refused: {exc}{hint}"
        ) from exc


def local_readings(source: str, if_present: bool, file_url: bool) -> list[str]:
    """The local paths native SGLang may open for a media reference.

    Native expands the home directory of sampling path fields, while its
    prompt enhancer opens a path as written. Its loaders open a file URL as
    written, a path relative to the working directory, while srt opens the
    path the URL names. A URL, a data URL, or text that native opens only as
    written (if_present) names a local file only when that file exists, since
    its image loader tries an existing file first. Relative paths resolve
    against the server working directory, which the stage and native worker
    processes inherit.
    """
    if if_present or source.startswith("data:"):
        return [source] if os.path.lexists(source) else []
    else:
        readings = [source]
    if source.startswith("~"):
        readings.append(os.path.join(os.getcwd(), source))
    else:
        pass
    if file_url:
        readings.append(unquote(urlparse(source).path))
    else:
        pass
    return readings
