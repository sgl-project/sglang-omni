# SPDX-License-Identifier: Apache-2.0
"""Serve Cosmos3 understanding through the native SRT Engine and chat service."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import threading
from concurrent.futures import Future
from copy import deepcopy
from typing import Any

from sglang_omni.admission import InvalidRequestError
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler


def build_chat_messages(inputs: Any) -> list[dict]:
    """Keep the native chat representation for text and multimodal content."""
    if isinstance(inputs, str):
        messages = [{"role": "user", "content": inputs}]
    elif isinstance(inputs, list) and all(isinstance(m, dict) for m in inputs):
        messages = deepcopy(inputs)
    elif isinstance(inputs, dict) and isinstance(inputs.get("messages"), list):
        messages = deepcopy(inputs["messages"])
        if not all(isinstance(message, dict) for message in messages):
            raise ValueError("Reasoner messages must be mappings")
        else:
            pass
        media = {key: inputs.get(key) for key in ("images", "videos", "audios")}
        if media["audios"] or any(key.startswith("video_") for key in inputs):
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
        raise ValueError("Reasoner inputs must be text or chat messages")
    if not messages:
        raise ValueError("Reasoner requests require at least one message")
    else:
        pass
    return messages


def build_chat_fields(payload: StagePayload, model: str, fields: set[str]) -> dict:
    messages = build_chat_messages(payload.request.inputs)
    params = dict(payload.request.params)
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
        reasoner_params = stage_params.get("reasoner")
        if reasoner_params is None:
            continue
        else:
            pass
        if not isinstance(reasoner_params, dict):
            raise ValueError("Reasoner stage parameters must be a mapping")
        else:
            pass
        params.update(reasoner_params)
    if params.get("tools") or params.get("tool_choice"):
        raise ValueError("The Omni text result does not carry native tool calls")
    else:
        pass
    if "max_new_tokens" in params:
        params["max_tokens"] = params.pop("max_new_tokens")
    else:
        pass
    if "return_logprob" in params:
        params["logprobs"] = params.pop("return_logprob")
    else:
        pass
    kwargs = {key: value for key, value in params.items() if key in fields}
    if kwargs.get("n", 1) != 1:
        raise ValueError("The Omni Reasoner returns one completion per request")
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
    kwargs.update(model=model, messages=messages, rid=payload.request_id, n=1)
    return kwargs


def response_dict(response: Any) -> dict:
    if isinstance(response, dict):
        return response
    else:
        pass
    if hasattr(response, "body"):
        return json.loads(response.body)
    else:
        pass
    return response.model_dump()


class NativeReasonerScheduler(ThreadedSimpleScheduler):
    """Submit every request to the native engine loop and let SRT own admission.

    Each request becomes one coroutine on the Engine event loop. No worker
    thread is held per request, so the native scheduler's running and queue
    limits govern concurrency instead of a fixed adapter cap.
    """

    def __init__(
        self,
        engine: Any,
        serving_chat: Any,
        request_type: Any,
        max_concurrency: int | None = None,
        device: int | None = None,
    ):
        self.engine = engine
        self.device = device
        self.serving_chat = serving_chat
        self.request_type = request_type
        self.native_lock = threading.Lock()
        self.closed = False
        self.ready = threading.Event()
        if engine.loop.is_running():
            raise ValueError("The Reasoner stage must own its native Engine event loop")
        else:
            pass
        super().__init__(
            None,
            max_concurrency=max_concurrency,
            abort_callback=self.abort_native,
            dispatch_fn=self.dispatch,
        )
        self.requires_tp_work_fanout = False
        self.loop_thread = threading.Thread(
            target=self.run_native_loop, name="cosmos3-native-tokenizer"
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

    def run_native_loop(self):
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

    def dispatch(self, payload: StagePayload) -> Future:
        with self.native_lock:
            if self.closed:
                cancelled: Future = Future()
                cancelled.cancel()
                return cancelled
            else:
                pass
            return asyncio.run_coroutine_threadsafe(
                self.complete(payload), self.engine.loop
            )

    async def complete(self, payload: StagePayload) -> StagePayload:
        # A cancellation requested before the loop reached this coroutine is
        # delivered one loop step after its first step. Yield once so an
        # already aborted request never reaches the native engine.
        await asyncio.sleep(0)
        try:
            fields = build_chat_fields(
                payload,
                self.engine.tokenizer_manager.served_model_name,
                set(self.request_type.model_fields),
            )
            request = self.request_type(**fields)
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc
        response = await self.serving_chat.handle_request(request, None)
        if hasattr(response, "body_iterator"):
            text_parts = []
            usage = None
            finish_reason = "stop"
            drained = False
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
                                    payload.request_id,
                                    "stream",
                                    {"text": delta, "modality": "text"},
                                )
                            )
                        else:
                            pass
                drained = True
                payload.data = {
                    "text": "".join(text_parts),
                    "modality": "text",
                    "usage": usage,
                    "finish_reason": finish_reason,
                }
            finally:
                if hasattr(response.body_iterator, "aclose"):
                    await response.body_iterator.aclose()
                else:
                    pass
                if not drained:
                    # note (Richard Wang): Direct consumers need no HTTP abort grace.
                    self.engine.tokenizer_manager.abort_request(payload.request_id)
                else:
                    pass
        else:
            data = response_dict(response)
            if "error" in data or "choices" not in data:
                if getattr(response, "status_code", None) == 400:
                    raise InvalidRequestError(str(data.get("message", data)))
                else:
                    pass
                raise ValueError(str(data.get("error", data)))
            else:
                pass
            choice = data["choices"][0]
            payload.data = {
                "text": choice["message"].get("content") or "",
                "modality": "text",
                "token_ids": choice.get("response_token_ids"),
                "output_token_logprobs": (choice.get("meta_info") or {}).get(
                    "output_token_logprobs"
                ),
                "usage": data.get("usage"),
                "finish_reason": choice.get("finish_reason") or "stop",
            }
        return payload

    def abort_native(self, request_id: str):
        with self.native_lock:
            if self.closed:
                return
            else:
                pass
            self.engine.loop.call_soon_threadsafe(
                self.engine.tokenizer_manager.abort_request, request_id
            )

    async def cancel_native_tasks(self):
        self.engine.tokenizer_manager.abort_request(abort_all=True)
        tasks = [
            task for task in asyncio.all_tasks() if task is not asyncio.current_task()
        ]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def stop(self):
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


def native_reasoner_kwargs(
    model_path: str,
    gpu_id: int,
    overrides: dict | None,
    runtime_gpu_ids: list[int] | None = None,
) -> dict:
    kwargs = dict(overrides or {})
    devices = list(runtime_gpu_ids) if runtime_gpu_ids is not None else [gpu_id]
    if not devices or devices[0] != gpu_id or min(devices) < 0:
        raise ValueError("Reasoner placement must start with the stage GPU")
    else:
        pass
    step = devices[1] - devices[0] if len(devices) > 1 else 1
    if step <= 0 or devices != list(range(gpu_id, gpu_id + step * len(devices), step)):
        raise ValueError("Native SRT requires GPUs with one positive index stride")
    else:
        pass
    if kwargs.get("tp_size", len(devices)) != len(devices):
        raise ValueError("Reasoner tensor parallel size must match Omni placement")
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
        raise ValueError("Set the Reasoner model and GPU through Omni placement")
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
    kwargs.setdefault("context_length", 8192)
    kwargs.setdefault("mem_fraction_static", 0.6)
    # A pipeline stage shares its host with other stages, so the native
    # scheduler waits on its request sockets instead of spinning while idle.
    kwargs.setdefault("sleep_on_idle", True)
    return kwargs


def create_reasoner_scheduler(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_concurrency: int | None = None,
    runtime_gpu_ids: list[int] | None = None,
    server_args_overrides: dict[str, Any] | None = None,
) -> NativeReasonerScheduler:
    from sglang_omni.utils.device import resolve_concrete_device

    concrete_device = resolve_concrete_device(device, gpu_id)
    if concrete_device.index is None:
        raise ValueError("Native Cosmos3 execution requires an indexed accelerator")
    else:
        pass
    gpu_id = concrete_device.index
    if multiprocessing.current_process().daemon:
        raise RuntimeError("Native SRT requires allow_child_processes=true")
    else:
        pass
    # The native tokenizer manager runs in this host process and works on the
    # serving GPU itself: the fast image processor, pinned frame buffers and
    # the GPU feature transport. The stage host no longer pins a device for
    # engines that own subprocesses, so pin it here, where the device work
    # is known, and operations that use the current device stay on it.
    from sglang_omni.platforms import current_platform

    current_platform.set_device(gpu_id)
    from sglang import Engine
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

    kwargs = native_reasoner_kwargs(
        model_path, gpu_id, server_args_overrides, runtime_gpu_ids
    )
    engine = Engine(**kwargs)
    try:
        service = OpenAIServingChat(engine.tokenizer_manager, engine.template_manager)
        return NativeReasonerScheduler(
            engine, service, ChatCompletionRequest, max_concurrency, device=gpu_id
        )
    except BaseException:
        engine.shutdown()
        raise
