# SPDX-License-Identifier: Apache-2.0
"""Serve Cosmos3 understanding through the native SRT Engine and chat service."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import threading
from concurrent.futures import CancelledError, Future
from copy import deepcopy
from typing import Any

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler


def build_chat_fields(payload: StagePayload, model: str, fields: set[str]) -> dict:
    inputs = payload.request.inputs
    if isinstance(inputs, str):
        messages = [{"role": "user", "content": inputs}]
    elif isinstance(inputs, list) and all(isinstance(m, dict) for m in inputs):
        messages = inputs
    elif isinstance(inputs, dict) and isinstance(inputs.get("messages"), list):
        messages = deepcopy(inputs["messages"])
        media = {key: inputs.get(key) for key in ("images", "videos", "audios")}
        if media["audios"] or any(key.startswith("video_") for key in inputs):
            raise ValueError(
                "Use native chat content for media and its processing options"
            )
        for key, kind in (("images", "image_url"), ("videos", "video_url")):
            values = media[key]
            if not values:
                continue
            if not isinstance(values, list) or not all(
                isinstance(v, str) for v in values
            ):
                raise ValueError("Media inputs must be lists of URLs or data URLs")
            message = next(
                (m for m in reversed(messages) if m.get("role") == "user"), None
            )
            if message is None:
                raise ValueError("Media requires a user message")
            content = message.get("content") or []
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            if not isinstance(content, list):
                raise ValueError("Media requires text or content parts")
            message["content"] = content + [
                {"type": kind, kind: {"url": v}} for v in values
            ]
    else:
        raise ValueError("Reasoner inputs must be text or chat messages")
    if not messages:
        raise ValueError("Reasoner requests require at least one message")
    params = dict(payload.request.params)
    for name in ("stage_sampling", "stage_params"):
        stage_params = params.pop(name, None)
        if stage_params is None:
            continue
        if not isinstance(stage_params, dict):
            raise ValueError(f"{name} must be a mapping")
        reasoner_params = stage_params.get("reasoner")
        if reasoner_params is None:
            continue
        if not isinstance(reasoner_params, dict):
            raise ValueError("Reasoner stage parameters must be a mapping")
        params.update(reasoner_params)
    if params.get("tools") or params.get("tool_choice"):
        raise ValueError("The Omni text result does not carry native tool calls")
    if "max_new_tokens" in params:
        params["max_tokens"] = params.pop("max_new_tokens")
    if "return_logprob" in params:
        params["logprobs"] = params.pop("return_logprob")
    kwargs = {key: value for key, value in params.items() if key in fields}
    if kwargs.get("n", 1) != 1:
        raise ValueError("The Omni Reasoner returns one completion per request")
    if kwargs.get("logprobs"):
        if kwargs.get("stream"):
            raise ValueError("Native token log probabilities require stream=false")
        kwargs["return_meta_info"] = True
        kwargs["return_token_ids"] = True
    kwargs.update(model=model, messages=messages, rid=payload.request_id, n=1)
    return kwargs


def _response_dict(response: Any) -> dict:
    if isinstance(response, dict):
        return response
    if hasattr(response, "body"):
        return json.loads(response.body)
    return response.model_dump()


class NativeReasonerScheduler(ThreadedSimpleScheduler):
    def __init__(
        self,
        engine: Any,
        serving_chat: Any,
        request_type: Any,
        max_concurrency: int = 8,
    ):
        self.engine = engine
        self.serving_chat = serving_chat
        self.request_type = request_type
        self._native_futures: dict[str, Future] = {}
        self._native_lock = threading.Lock()
        self._closed = False
        self._ready = threading.Event()
        if engine.loop.is_running():
            raise ValueError("The Reasoner stage must own its native Engine event loop")
        super().__init__(
            self._submit,
            max_concurrency=max_concurrency,
            abort_callback=self._abort_native,
        )
        self.requires_tp_work_fanout = False
        self._loop_thread = threading.Thread(
            target=self._run_native_loop, name="cosmos3-native-tokenizer"
        )
        try:
            self._loop_thread.start()
            if not self._ready.wait(timeout=5):
                raise RuntimeError("The native tokenizer event loop did not start")
        except BaseException:
            if self._loop_thread.is_alive():
                self.engine.loop.call_soon_threadsafe(self.engine.loop.stop)
                self._loop_thread.join(timeout=5)
            self._executor.shutdown(wait=True, cancel_futures=True)
            raise

    def _run_native_loop(self):
        asyncio.set_event_loop(self.engine.loop)
        self.engine.loop.call_soon(self._ready.set)
        self.engine.loop.run_forever()

    def _submit(self, payload: StagePayload) -> StagePayload:
        with self._lock, self._native_lock:
            if self._closed or payload.request_id not in self._pending:
                raise CancelledError()
            future = asyncio.run_coroutine_threadsafe(
                self._complete(payload), self.engine.loop
            )
            self._native_futures[payload.request_id] = future
        try:
            return future.result()
        finally:
            with self._native_lock:
                self._native_futures.pop(payload.request_id, None)

    async def _complete(self, payload: StagePayload) -> StagePayload:
        fields = build_chat_fields(
            payload,
            self.engine.tokenizer_manager.served_model_name,
            set(self.request_type.model_fields),
        )
        response = await self.serving_chat.handle_request(
            self.request_type(**fields), None
        )
        if hasattr(response, "body_iterator"):
            text_parts = []
            usage = None
            finish_reason = "stop"
            try:
                async for frame in response.body_iterator:
                    if isinstance(frame, bytes):
                        frame = frame.decode("utf-8")
                    data = frame.removeprefix("data:").strip()
                    if data == "[DONE]" or not data:
                        continue
                    event = json.loads(data)
                    if "error" in event:
                        raise RuntimeError(str(event["error"]))
                    usage = event.get("usage") or usage
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {}).get("content") or ""
                        if delta:
                            text_parts.append(delta)
                        finish_reason = choice.get("finish_reason") or finish_reason
                        self.outbox.put(
                            OutgoingMessage(
                                payload.request_id,
                                "stream",
                                {
                                    "text": delta,
                                    "modality": "text",
                                    "usage": usage,
                                    "finish_reason": choice.get("finish_reason"),
                                },
                            )
                        )
                payload.data = {
                    "text": "".join(text_parts),
                    "modality": "text",
                    "usage": usage,
                    "finish_reason": finish_reason,
                }
            finally:
                if hasattr(response.body_iterator, "aclose"):
                    await response.body_iterator.aclose()
                if response.background is not None:
                    await response.background()
        else:
            data = _response_dict(response)
            if "error" in data or "choices" not in data:
                raise ValueError(str(data.get("error", data)))
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

    def _abort_native(self, request_id: str):
        with self._native_lock:
            if self._closed:
                return
            future = self._native_futures.get(request_id)
        self.engine.loop.call_soon_threadsafe(
            self.engine.tokenizer_manager.abort_request, request_id
        )
        if future is not None:
            future.cancel()

    async def _cancel_native_tasks(self):
        self.engine.tokenizer_manager.abort_request(abort_all=True)
        tasks = [
            task for task in asyncio.all_tasks() if task is not asyncio.current_task()
        ]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def stop(self):
        super().stop()
        with self._native_lock:
            if self._closed:
                return
            self._closed = True
            futures = list(self._native_futures.values())
        for future in futures:
            future.cancel()
        try:
            asyncio.run_coroutine_threadsafe(
                self._cancel_native_tasks(), self.engine.loop
            ).result(timeout=10)
        finally:
            try:
                self.engine.shutdown()
            finally:
                self.engine.loop.call_soon_threadsafe(self.engine.loop.stop)
                self._loop_thread.join(timeout=5)
                if self._loop_thread.is_alive():
                    raise RuntimeError("The native tokenizer event loop did not stop")
                self._executor.shutdown(wait=True, cancel_futures=True)
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
    step = devices[1] - devices[0] if len(devices) > 1 else 1
    if step <= 0 or devices != list(range(gpu_id, gpu_id + step * len(devices), step)):
        raise ValueError("Native SRT requires GPUs with one positive index stride")
    if kwargs.get("tp_size", len(devices)) != len(devices):
        raise ValueError("Reasoner tensor parallel size must match Omni placement")
    for key in ("pp_size", "dp_size", "nnodes"):
        if kwargs.get(key, 1) != 1:
            raise ValueError("This stage owns one native SRT tensor parallel group")
    if any(
        key in kwargs
        for key in ("base_gpu_id", "gpu_id_step", "node_rank", "model_path")
    ):
        raise ValueError("Set the Reasoner model and GPU through Omni placement")
    if kwargs.get("reasoning_parser") or kwargs.get("tool_call_parser"):
        raise ValueError("The Omni text result requires native text output")
    kwargs.update(
        model_path=model_path,
        base_gpu_id=gpu_id,
        gpu_id_step=step,
        tp_size=len(devices),
    )
    kwargs.setdefault("context_length", 8192)
    kwargs.setdefault("mem_fraction_static", 0.6)
    return kwargs


def create_reasoner_scheduler(
    model_path: str,
    *,
    gpu_id: int = 0,
    max_concurrency: int = 8,
    runtime_gpu_ids: list[int] | None = None,
    server_args_overrides: dict[str, Any] | None = None,
) -> NativeReasonerScheduler:
    if multiprocessing.current_process().daemon:
        raise RuntimeError("Native SRT requires allow_child_processes=true")
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
            engine, service, ChatCompletionRequest, max_concurrency
        )
    except BaseException:
        engine.shutdown()
        raise
