# SPDX-License-Identifier: Apache-2.0
"""Serve a chat stage through the native SGLang SRT Engine and chat service."""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import os
import tempfile
import threading
from concurrent.futures import Future
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse
from urllib.request import url2pathname

from sglang_omni.client.types import SamplingParams
from sglang_omni.preprocessing.resource_connector import (
    MediaPolicyError,
    get_global_resource_connector,
    resolve_local_file,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler
from sglang_omni.serve.openai_errors import generation_error_status_code

if TYPE_CHECKING:
    from sglang import Engine
    from sglang.srt.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        ChatCompletionResponse,
    )
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
    from starlette.responses import Response, StreamingResponse
else:
    pass

logger = logging.getLogger(__name__)

# Client errors carry this prefix, which the API maps to 400.
REQUEST_ERROR = "Native chat request rejected: "

# Omni leaves an unset sampling field out, except the stop lists it sends empty.
SAMPLING_PLACEHOLDERS = SamplingParams().to_dict()


def set_sampling_fields(values: dict) -> dict:
    """Drop the empty stop lists that the caller did not set.

    Native SGLang resolves an omitted field from the checkpoint generation
    config, so only the fields the caller set reach it.
    """
    return {
        key: value
        for key, value in values.items()
        if key not in SAMPLING_PLACEHOLDERS or value != SAMPLING_PLACEHOLDERS[key]
    }


async def police_media_reference(
    source: str, kind: str, max_bytes: int | None, fetched: list[Path]
) -> str:
    """Apply the server media policy to one chat media reference.

    These are main's rules for chat media: data URLs pass, remote URLs follow
    the domain allowlist on every redirect hop, file URLs need the allowed
    local directory and must resolve inside it, and bare paths are confined to
    that directory once it is set. Native SGLang applies no path policy and
    matches exact hosts only, so with an allowlist the stage fetches remote
    media itself into a temporary file, listed in fetched for the caller to
    remove, and hands native that file. A refused URL is the caller's error,
    and the shared error classifier decides other fetch failures, as it does
    for every other model. Native's image and audio loaders open a resolved
    local path as given, while its video loader parses a plain path as a URL
    and misreads a name that holds #, ? or %, so a resolved local video goes
    to native as a file URL, which it reads back exactly.
    """
    connector = get_global_resource_connector()
    url = urlparse(source)
    if url.scheme == "data":
        return source
    elif url.scheme in ("http", "https") and not connector.allowed_media_domains:
        connector.assert_url_allowed(source)
        return source
    elif url.scheme in ("http", "https"):
        try:
            data, _ = await connector.load_http_bytes_async(
                source, timeout=connector.connection.timeout, max_bytes=max_bytes
            )
        except MediaPolicyError:
            raise
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        handle, name = tempfile.mkstemp(prefix="sglang-omni-chat-media-")
        os.close(handle)
        fetched.append(Path(name))
        await asyncio.to_thread(Path(name).write_bytes, data)
        return Path(name).as_uri()
    elif url.scheme == "file" and connector.allowed_local_media_path is None:
        raise MediaPolicyError(
            "File URLs require --allowed-local-media-path on the server"
        )
    elif url.scheme == "file" and url.netloc not in ("", "localhost"):
        raise MediaPolicyError(f"File URL netloc is not supported: {url.netloc}")
    elif url.scheme == "file":
        resolved = resolve_local_file(
            url2pathname(url.path),
            allowed_local_media_path=connector.allowed_local_media_path,
        )
    elif url.scheme:
        raise MediaPolicyError(f"Unsupported media URL scheme: {url.scheme}")
    elif connector.allowed_local_media_path is None:
        return source
    else:
        resolved = resolve_local_file(
            source, allowed_local_media_path=connector.allowed_local_media_path
        )
    return resolved.as_uri() if kind == "video_url" else str(resolved)


async def police_chat_media(
    messages: list[dict], max_bytes: int | None, fetched: list[Path]
) -> None:
    """Police every image, video, and audio reference of the messages in place."""
    for message in messages:
        content = message.get("content")
        for part in content if isinstance(content, list) else ():
            kind = part.get("type") if isinstance(part, dict) else None
            if kind not in ("image_url", "video_url", "audio_url"):
                continue
            else:
                pass
            reference = part.get(kind)
            if isinstance(reference, dict) and isinstance(reference.get("url"), str):
                reference["url"] = await police_media_reference(
                    reference["url"], kind, max_bytes, fetched
                )
            elif isinstance(reference, str):
                part[kind] = await police_media_reference(
                    reference, kind, max_bytes, fetched
                )
            else:
                raise ValueError(f"{kind} needs a url")


def build_chat_fields(
    payload: StagePayload, model: str, fields: set[str], stage_name: str
) -> dict:
    """Translate an Omni request into native chat request fields.

    stage_params and stage_sampling entries under stage_name hold native
    options for this stage.
    """
    inputs = payload.request.inputs
    metadata = payload.request.metadata or {}
    if any(key.startswith("video_") or key == "use_audio_in_video" for key in metadata):
        raise ValueError("Use native chat content for media and its processing options")
    else:
        pass
    if isinstance(inputs, str):
        messages = [{"role": "user", "content": inputs}]
    elif isinstance(inputs, list) and all(isinstance(m, dict) for m in inputs):
        messages = deepcopy(inputs)
    elif isinstance(inputs, dict) and isinstance(inputs.get("messages"), list):
        messages = deepcopy(inputs["messages"])
        if not all(isinstance(message, dict) for message in messages):
            raise ValueError("Chat messages must be mappings")
        else:
            pass
        media = {key: inputs.get(key) for key in ("images", "videos", "audios")}
        if media["audios"]:
            raise ValueError(
                "Use native chat content for media and its processing options"
            )
        else:
            pass
        for key, kind in (("images", "image_url"), ("videos", "video_url")):
            values = media[key]
            if not values:
                continue
            else:
                pass
            if not isinstance(values, list) or not all(
                isinstance(v, str) for v in values
            ):
                raise ValueError("Media inputs must be lists of URLs or data URLs")
            else:
                pass
            message = next(
                (m for m in reversed(messages) if m.get("role") == "user"), None
            )
            if message is None:
                raise ValueError("Media requires a user message")
            else:
                pass
            content = message.get("content") or []
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            else:
                pass
            if not isinstance(content, list):
                raise ValueError("Media requires text or content parts")
            else:
                pass
            message["content"] = content + [
                {"type": kind, kind: {"url": v}} for v in values
            ]
    else:
        raise ValueError("Chat inputs must be text or chat messages")
    if not messages:
        raise ValueError("Chat requests require at least one message")
    else:
        pass
    params = set_sampling_fields(dict(payload.request.params))
    for name in ("stage_sampling", "stage_params"):
        stage_params = params.pop(name, None)
        if stage_params is None:
            continue
        else:
            pass
        if not isinstance(stage_params, dict):
            raise ValueError(f"{name} must be a mapping")
        else:
            pass
        stage_options = stage_params.get(stage_name)
        if stage_options is None:
            continue
        else:
            pass
        if not isinstance(stage_options, dict):
            raise ValueError(f"{name}.{stage_name} must be a mapping")
        elif name == "stage_sampling":
            stage_options = set_sampling_fields(stage_options)
        else:
            pass
        params.update(stage_options)
    if params.get("tools") or params.get("tool_choice"):
        raise ValueError("The Omni text result does not carry native tool calls")
    else:
        pass
    if "max_new_tokens" in params:
        params["max_completion_tokens"] = params.pop("max_new_tokens")
    else:
        pass
    if "return_logprob" in params:
        params["logprobs"] = params.pop("return_logprob")
    else:
        pass
    kwargs = {key: value for key, value in params.items() if key in fields}
    if kwargs.get("n", 1) != 1:
        raise ValueError("The native chat stage returns one completion per request")
    else:
        pass
    if kwargs.get("logprobs"):
        if kwargs.get("stream"):
            raise ValueError("Native token log probabilities require stream=false")
        else:
            pass
        kwargs["return_meta_info"] = True
        kwargs["return_token_ids"] = True
    else:
        pass
    if kwargs.get("stream"):
        # The stage result reports the final usage, as other Omni stages do.
        kwargs.setdefault("stream_options", {"include_usage": True})
    else:
        pass
    kwargs.update(model=model, messages=messages, rid=payload.request_id, n=1)
    return kwargs


def response_dict(response: ChatCompletionResponse | Response) -> dict:
    if hasattr(response, "body"):
        return json.loads(response.body)
    else:
        pass
    return response.model_dump()


def collect_response(response: ChatCompletionResponse | Response) -> dict:
    data = response_dict(response)
    if "error" in data or "choices" not in data:
        if getattr(response, "status_code", None) == 400:
            raise ValueError(f"{REQUEST_ERROR}{data.get('message', data)}")
        else:
            pass
        raise ValueError(str(data.get("error", data)))
    else:
        pass
    choice = data["choices"][0]
    return {
        "text": choice["message"].get("content") or "",
        "modality": "text",
        "token_ids": choice.get("response_token_ids"),
        "output_token_logprobs": (choice.get("meta_info") or {}).get(
            "output_token_logprobs"
        ),
        "usage": data.get("usage"),
        "finish_reason": choice.get("finish_reason") or "stop",
    }


def warm_up(
    engine: Engine,
    serving_chat: OpenAIServingChat,
    request_type: type[ChatCompletionRequest],
) -> None:
    """Send the request that sglang serve sends before it reports ready.

    Lazy kernel and encoder setup then runs at startup, as it does for sglang
    serve, instead of on the first client request. The native
    skip_server_warmup flag turns it off. The request runs on the thread that
    runs the engine loop, as every later native coroutine does.
    """
    if engine.server_args.skip_server_warmup:
        return
    else:
        pass
    model_config = engine.tokenizer_manager.model_config
    if model_config.is_image_understandable_model:
        from sglang.srt.entrypoints.http_server import (  # noqa: leading-underscore
            _get_vlm_warmup_image_base64 as warmup_image,
        )

        hf_config = model_config.hf_config
        image = warmup_image(
            {
                "architectures": getattr(hf_config, "architectures", None),
                "model_type": getattr(hf_config, "model_type", None),
            }
        )
        content = [
            {
                "type": "image_url",
                # Native SGLang decodes a plain base64 string as an image.
                "image_url": {"url": image},
            },
            {"type": "text", "text": "Describe the image."},
        ]
    else:
        content = "The capital city of France is"
    request = request_type(
        model=engine.tokenizer_manager.served_model_name,
        messages=[{"role": "user", "content": content}],
        max_completion_tokens=8,
        temperature=0.0,
    )
    response = asyncio.run_coroutine_threadsafe(
        serving_chat.handle_request(request, None), engine.loop
    ).result()
    if getattr(response, "status_code", 200) != 200:
        raise RuntimeError(
            f"The native chat warmup request failed: {response_dict(response)}"
        )
    else:
        pass


class NativeChatScheduler(ThreadedSimpleScheduler[StagePayload, StagePayload]):
    """Submit every request to the native engine loop and let SRT own admission.

    Each request becomes one coroutine on the Engine event loop. No worker
    thread is held per request, so the native scheduler's running and queue
    limits govern concurrency instead of a fixed adapter cap.
    """

    def __init__(
        self,
        engine: Engine,
        serving_chat: OpenAIServingChat,
        request_type: type[ChatCompletionRequest],
        stage_name: str,
        max_concurrency: int | None = None,
        device: int | None = None,
    ):
        self.engine = engine
        self.stage_name = stage_name
        self.device = device
        self.serving_chat = serving_chat
        self.request_type = request_type
        self.native_lock = threading.Lock()
        # Remote media fetched under a domain allowlist keeps native's size cap.
        max_media_mb = getattr(engine.server_args, "media_url_max_file_size_mb", 0)
        self.max_media_bytes = max_media_mb * 1024 * 1024 if max_media_mb else None
        self.closed = False
        self.ready = threading.Event()
        if engine.loop.is_running():
            raise ValueError("The native chat stage must own its Engine event loop")
        else:
            pass
        super().__init__(
            None, max_concurrency=max_concurrency, dispatch_fn=self.dispatch
        )
        self.requires_tp_work_fanout = False
        self.loop_thread = threading.Thread(
            target=self.run_native_loop, name="native-chat-tokenizer"
        )
        try:
            self.loop_thread.start()
            if not self.ready.wait(timeout=5):
                raise RuntimeError("The native tokenizer event loop did not start")
            else:
                pass
        except BaseException:
            if self.loop_thread.is_alive():
                self.engine.loop.call_soon_threadsafe(self.engine.loop.stop)
                self.loop_thread.join(timeout=5)
            else:
                pass
            raise

    def run_native_loop(self) -> None:
        if self.device is not None:
            # The current device is per thread, and this thread runs every
            # native tokenizer coroutine, including its device work.
            from sglang_omni.platforms import current_platform

            current_platform.set_device(self.device)
        else:
            pass
        asyncio.set_event_loop(self.engine.loop)
        self.engine.loop.call_soon(self.ready.set)
        self.engine.loop.run_forever()

    def dispatch(self, payload: StagePayload) -> Future[StagePayload]:
        with self.native_lock:
            if self.closed:
                cancelled: Future[StagePayload] = Future()
                cancelled.cancel()
                return cancelled
            else:
                pass
            return asyncio.run_coroutine_threadsafe(
                self.complete(payload), self.engine.loop
            )

    async def complete(self, payload: StagePayload) -> StagePayload:
        fetched: list[Path] = []
        try:
            return await self.run_native(payload, fetched)
        finally:
            # Native SGLang has read the fetched media once the request ends.
            for path in fetched:
                path.unlink(missing_ok=True)

    async def run_native(
        self, payload: StagePayload, fetched: list[Path]
    ) -> StagePayload:
        # A cancellation requested before the loop reached this coroutine is
        # delivered one loop step after its first step. Yield once so an
        # already aborted request never reaches the native engine.
        await asyncio.sleep(0)
        try:
            fields = build_chat_fields(
                payload,
                self.engine.tokenizer_manager.served_model_name,
                set(self.request_type.model_fields),
                self.stage_name,
            )
            await police_chat_media(fields["messages"], self.max_media_bytes, fetched)
            request = self.request_type(**fields)
        except ValueError as exc:
            raise ValueError(f"{REQUEST_ERROR}{exc}") from exc
        settled = False
        try:
            response = await self.serving_chat.handle_request(request, None)
            # A complete response means native has already settled the request.
            settled = not hasattr(response, "body_iterator")
            if settled:
                payload.data = collect_response(response)
            else:
                payload.data = await self.collect_stream(payload.request_id, response)
                settled = True
        finally:
            if not settled:
                # A cancelled or failed request must not keep running in
                # native SGLang, and only this coroutine aborts it.
                self.engine.tokenizer_manager.abort_request(payload.request_id)
            else:
                pass
        return payload

    async def collect_stream(
        self, request_id: str, response: StreamingResponse
    ) -> dict:
        text_parts = []
        usage = None
        finish_reason = "stop"
        try:
            async for frame in response.body_iterator:
                if isinstance(frame, bytes):
                    frame = frame.decode("utf-8")
                else:
                    pass
                data = frame.removeprefix("data:").strip()
                if data == "[DONE]" or not data:
                    continue
                else:
                    pass
                event = json.loads(data)
                if "error" in event:
                    raise RuntimeError(str(event["error"]))
                else:
                    pass
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {}).get("content") or ""
                    finish_reason = choice.get("finish_reason") or finish_reason
                    if delta:
                        text_parts.append(delta)
                        # The stage result owns completion and final usage.
                        self.outbox.put(
                            OutgoingMessage(
                                request_id,
                                "stream",
                                {"text": delta, "modality": "text"},
                            )
                        )
                    else:
                        pass
        finally:
            if hasattr(response.body_iterator, "aclose"):
                await response.body_iterator.aclose()
            else:
                pass
        return {
            "text": "".join(text_parts),
            "modality": "text",
            "usage": usage,
            "finish_reason": finish_reason,
        }

    def log_failure(self, request_id: str, exc: BaseException) -> None:
        # The API takes the status from the shared classifier, so a failure it
        # answers with 400 is the caller's error and logs one warning line.
        if generation_error_status_code(exc) == 400:
            logger.warning("Native chat request %s failed: %s", request_id, exc)
        else:
            super().log_failure(request_id, exc)

    async def cancel_native_tasks(self) -> None:
        self.engine.tokenizer_manager.abort_request(abort_all=True)
        tasks = [
            task for task in asyncio.all_tasks() if task is not asyncio.current_task()
        ]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def stop(self) -> None:
        super().stop()
        with self.native_lock:
            if self.closed:
                return
            else:
                pass
            self.closed = True
        with self.lock:
            futures = list(self.pending.values())
        for future in futures:
            future.cancel()
        try:
            asyncio.run_coroutine_threadsafe(
                self.cancel_native_tasks(), self.engine.loop
            ).result(timeout=10)
        finally:
            try:
                self.engine.shutdown()
            finally:
                self.engine.loop.call_soon_threadsafe(self.engine.loop.stop)
                self.loop_thread.join(timeout=5)
                if self.loop_thread.is_alive():
                    raise RuntimeError("The native tokenizer event loop did not stop")
                else:
                    pass
                self.engine.loop.close()


def native_chat_server_kwargs(
    model_path: str,
    gpu_id: int,
    overrides: dict | None,
    runtime_gpu_ids: list[int] | None = None,
    total_gpu_memory_fraction: float | None = None,
) -> dict:
    kwargs = dict(overrides or {})
    devices = list(runtime_gpu_ids) if runtime_gpu_ids is not None else [gpu_id]
    if not devices or devices[0] != gpu_id or min(devices) < 0:
        raise ValueError("Native chat placement must start with the stage GPU")
    else:
        pass
    step = devices[1] - devices[0] if len(devices) > 1 else 1
    if step <= 0 or devices != list(range(gpu_id, gpu_id + step * len(devices), step)):
        raise ValueError("Native SRT requires GPUs with one positive index stride")
    else:
        pass
    if kwargs.get("tp_size", len(devices)) != len(devices):
        raise ValueError("Native chat tensor parallel size must match Omni placement")
    else:
        pass
    for key in ("pp_size", "dp_size", "nnodes"):
        if kwargs.get(key, 1) != 1:
            raise ValueError("This stage owns one native SRT tensor parallel group")
        else:
            pass
    if any(
        key in kwargs
        for key in ("base_gpu_id", "gpu_id_step", "node_rank", "model_path")
    ):
        raise ValueError("Set the native chat model and GPU through Omni placement")
    else:
        pass
    if kwargs.get("reasoning_parser") or kwargs.get("tool_call_parser"):
        raise ValueError("The Omni text result requires native text output")
    else:
        pass
    kwargs.update(
        model_path=model_path,
        base_gpu_id=gpu_id,
        gpu_id_step=step,
        tp_size=len(devices),
    )
    # Native derives the context length from the checkpoint and the memory
    # fraction from the device. A placement budget for the stage replaces only
    # the latter, and an explicit engine.mem_fraction_static wins over both.
    if total_gpu_memory_fraction is None:
        pass
    elif kwargs.setdefault("mem_fraction_static", total_gpu_memory_fraction) != (
        total_gpu_memory_fraction
    ):
        logger.warning(
            "engine.mem_fraction_static=%s overrides the placement budget "
            "gpu_memory_fraction=%s of the stage",
            kwargs["mem_fraction_static"],
            total_gpu_memory_fraction,
        )
    else:
        pass
    # A pipeline stage shares its host with other stages, so the native
    # scheduler waits on its request sockets instead of spinning while idle.
    kwargs.setdefault("sleep_on_idle", True)
    return kwargs


def create_native_chat_scheduler(
    model_path: str,
    *,
    stage_name: str,
    device: str | None = None,
    gpu_id: int | None = None,
    max_concurrency: int | None = None,
    runtime_gpu_ids: list[int] | None = None,
    server_args_overrides: dict[str, object] | None = None,
    total_gpu_memory_fraction: float | None = None,
) -> NativeChatScheduler:
    from sglang_omni.utils.device import resolve_concrete_device

    concrete_device = resolve_concrete_device(device, gpu_id)
    if concrete_device.index is None:
        raise ValueError("The native chat stage requires an indexed accelerator")
    else:
        pass
    gpu_id = concrete_device.index
    if multiprocessing.current_process().daemon:
        raise RuntimeError("Native SRT requires allow_child_processes=true")
    else:
        pass
    from sglang import Engine
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

    from sglang_omni.platforms import current_platform

    kwargs = native_chat_server_kwargs(
        model_path,
        gpu_id,
        server_args_overrides,
        runtime_gpu_ids,
        total_gpu_memory_fraction,
    )
    engine = Engine(**kwargs)
    try:
        # The native tokenizer manager runs in this host process and works on
        # the serving GPU itself: the fast image processor, pinned frame
        # buffers and the GPU feature transport. Pin the host to that GPU only
        # now, as sglang serve does: a host CUDA context that exists while the
        # native scheduler measures free memory shrinks its KV pool.
        current_platform.set_device(gpu_id)
        service = OpenAIServingChat(engine.tokenizer_manager, engine.template_manager)
        scheduler = NativeChatScheduler(
            engine,
            service,
            ChatCompletionRequest,
            stage_name,
            max_concurrency,
            device=gpu_id,
        )
    except BaseException:
        engine.shutdown()
        raise
    # The first request off the main thread keeps native's crash handler, which
    # kills the process tree when a native child fails, a process signal
    # handler instead of a callback on the engine loop thread.
    try:
        warm_up(engine, service, ChatCompletionRequest)
    except BaseException:
        scheduler.stop()
        raise
    return scheduler
