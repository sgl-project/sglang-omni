# SPDX-License-Identifier: Apache-2.0
"""Adapt Omni payloads to the native SGLang generation API."""

from __future__ import annotations

import multiprocessing
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


def resolve_generation_options(
    request_params: dict[str, Any], inputs: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Lower common SDK seeds and native overrides with one precedence rule."""
    params = {}
    if request_params.get("seed") is not None:
        params["seed"] = request_params["seed"]
    params.update(inputs or {})
    stage_sampling = request_params.get("stage_sampling")
    if stage_sampling is not None:
        if not isinstance(stage_sampling, dict):
            raise ValueError("stage_sampling must be a mapping")
        generation = stage_sampling.get("generation", {})
        if not isinstance(generation, dict):
            raise ValueError("Generation stage sampling must be a mapping")
        if generation.get("seed") is not None:
            params["seed"] = generation["seed"]
    overrides = request_params.get("diffusion", {})
    if not isinstance(overrides, dict):
        raise ValueError("diffusion parameters must be a mapping")
    params.update(overrides)
    stage_params = request_params.get("stage_params")
    if stage_params is not None:
        if not isinstance(stage_params, dict):
            raise ValueError("stage_params must be a mapping")
        generation = stage_params.get("generation", {})
        if not isinstance(generation, dict):
            raise ValueError("Generation stage parameters must be a mapping")
        params.update(generation)
    return params


def build_sampling_params(payload: StagePayload, output_dir: str) -> dict[str, Any]:
    inputs = payload.request.inputs
    if isinstance(inputs, str):
        inputs = {"prompt": inputs}
    elif not isinstance(inputs, dict):
        raise ValueError("Generation inputs must be a prompt or native sampling fields")
    params = resolve_generation_options(payload.request.params, inputs)
    if not isinstance(params.get("prompt"), str) or not params["prompt"].strip():
        raise ValueError("Generation requires a nonempty prompt")
    if payload.request.params.get("stream"):
        raise ValueError("Use the native realtime video API for streaming generation")
    if params.get("prompt_file_path"):
        raise ValueError("Submit prompts individually through the Omni client")
    params.update(
        save_output=True,
        return_file_paths_only=True,
        output_path=output_dir,
        output_file_name=uuid4().hex,
    )
    return params


class NativeGenerationScheduler(SimpleScheduler):
    """Keep scheduler and pipeline execution in the native SGLang runtime."""

    def __init__(self, generator: Any, output_dir: str):
        self.generator = generator
        self.output_dir = output_dir
        self._native_requests: dict[str, _GenerationRequest] = {}
        super().__init__(
            self._generate,
            abort_callback=self._cancel_native,
            shutdown_callback=generator.shutdown,
        )
        self.requires_tp_work_fanout = False

    def _next_message(self):
        for process in self.generator.local_scheduler_process or []:
            if not process.is_alive():
                raise RuntimeError(
                    f"Native generation worker {process.pid} exited "
                    f"with code {process.exitcode}"
                )
        return super()._next_message()

    @staticmethod
    def _remove_request_directory(directory: Path) -> None:
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            pass

    def _cancel_native(self, request_id: str) -> None:
        directory = None
        with self._abort_lock:
            request = self._native_requests.get(request_id)
            if request is not None:
                request.cancelled.set()
                if not request.running and not request.delivering:
                    self._native_requests.pop(request_id)
                    directory = request.directory
        if directory is not None:
            self._remove_request_directory(directory)

    def _emit_result(self, request_id, result, outbox) -> None:
        directory = None
        with self._abort_lock:
            request = self._native_requests.get(request_id)
            if request_id in self._aborted:
                self._aborted.discard(request_id)
                if request is not None and request.result is result:
                    self._native_requests.pop(request_id)
                    directory = request.directory
            else:
                super()._emit_result(request_id, result, outbox)
        if directory is not None:
            self._remove_request_directory(directory)

    def claim_result(self, result: StagePayload, *, terminal: bool) -> bool:
        if not terminal:
            raise ValueError(
                "Saved native media requires terminal delivery; "
                "use inline media for inter-stage routing"
            )
        with self._abort_lock:
            request = self._native_requests.get(result.request_id)
            if request is None or request.result is not result:
                return False
            request.delivering = True
            return True

    def release_result(self, result: StagePayload, *, delivered: bool) -> None:
        directory = None
        with self._abort_lock:
            request = self._native_requests.get(result.request_id)
            if request is not None and request.result is result:
                self._native_requests.pop(result.request_id)
                if not delivered:
                    directory = request.directory
        if directory is not None:
            self._remove_request_directory(directory)

    def stop(self) -> None:
        directories = []
        with self._abort_lock:
            for request_id, request in list(self._native_requests.items()):
                request.cancelled.set()
                if not request.running and not request.delivering:
                    self._native_requests.pop(request_id)
                    directories.append(request.directory)
        try:
            super().stop()
        finally:
            for directory in directories:
                self._remove_request_directory(directory)

    def _generate(self, payload: StagePayload) -> StagePayload:
        params = build_sampling_params(payload, self.output_dir)
        with self._abort_lock:
            if payload.request_id in self._aborted:
                return payload
            request = _GenerationRequest(
                threading.Event(),
                Path(tempfile.mkdtemp(prefix="request_", dir=self.output_dir)),
            )
            self._native_requests[payload.request_id] = request
        params["output_path"] = str(request.directory)
        kwargs = {"sampling_params_kwargs": params}
        if (
            getattr(self.generator, "supports_cancellation", False)
            and params.get("num_outputs_per_prompt", 1) == 1
        ):
            kwargs["cancellation_event"] = request.cancelled
        succeeded = False
        try:
            results = self.generator.generate(**kwargs)
            if results is None:
                raise RuntimeError(
                    "Native generation failed without producing an output"
                )
            if not isinstance(results, list):
                results = [results]
            if not results or any(not result.output_file_path for result in results):
                raise RuntimeError("Native generation returned no saved media")
            payload.data = {
                "media": [
                    {
                        "path": result.output_file_path,
                        "size": result.size,
                        "prompt": result.prompt,
                        "generation_time": result.generation_time,
                        "peak_memory_mb": result.peak_memory_mb,
                        "metrics": result.metrics,
                    }
                    for result in results
                ],
                "finish_reason": "stop",
            }
            succeeded = True
            return payload
        finally:
            with self._abort_lock:
                request.running = False
                request.result = payload if succeeded else None
                cleanup = not succeeded or request.cancelled.is_set()
                if cleanup and self._native_requests.get(payload.request_id) is request:
                    self._native_requests.pop(payload.request_id)
            if cleanup:
                self._remove_request_directory(request.directory)


@dataclass
class _GenerationRequest:
    cancelled: threading.Event
    directory: Path
    running: bool = True
    delivering: bool = False
    result: StagePayload | None = None


def native_server_kwargs(
    model_path: str,
    gpu_id: int,
    overrides: dict[str, Any] | None,
    runtime_gpu_ids: list[int] | None = None,
) -> dict[str, Any]:
    kwargs = dict(overrides or {})
    if kwargs.get("scheduler_rpc_timeout") is not None:
        raise ValueError(
            "Native generation cannot use scheduler RPC deadlines before work settles"
        )
    devices = list(runtime_gpu_ids) if runtime_gpu_ids is not None else [gpu_id]
    if not devices or devices[0] != gpu_id or len(set(devices)) != len(devices):
        raise ValueError("Native GPU placement must start with the stage GPU")
    if any(gpu < 0 for gpu in devices):
        raise ValueError("Native GPU ids must be nonnegative")
    if kwargs.get("num_gpus", len(devices)) != len(devices):
        raise ValueError("Native GPU count must match Omni placement")
    if kwargs.get("tp_size", len(devices)) != len(devices):
        raise ValueError("Native tensor parallel size must match Omni placement")
    if kwargs.get("nnodes", 1) != 1:
        raise ValueError("This stage owns GPUs on one host")
    if any(key in kwargs for key in ("gpu_ids", "base_gpu_id", "model_path")):
        raise ValueError("Set the generation model and GPUs through Omni placement")
    kwargs.update(
        model_path=model_path,
        num_gpus=len(devices),
        tp_size=len(devices),
    )
    if runtime_gpu_ids is None:
        kwargs["base_gpu_id"] = gpu_id
    else:
        kwargs["gpu_ids"] = devices
    return kwargs


def resolved_native_server_kwargs(
    model_path: str,
    gpu_id: int,
    overrides: dict[str, Any] | None,
    runtime_gpu_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Resolve one immutable checkpoint while preserving its public model name."""
    kwargs = native_server_kwargs(model_path, gpu_id, overrides, runtime_gpu_ids)
    if "@" in model_path and not Path(model_path).is_dir():
        from sglang_omni.utils.checkpoint import resolve_checkpoint

        kwargs["model_path"] = resolve_checkpoint(model_path)
    kwargs.setdefault("served_model_name", model_path.partition("@")[0])
    return kwargs


def create_generation_scheduler(
    model_path: str,
    *,
    gpu_id: int = 0,
    output_dir: str = "outputs",
    runtime_gpu_ids: list[int] | None = None,
    server_args_overrides: dict[str, Any] | None = None,
) -> NativeGenerationScheduler:
    if multiprocessing.current_process().daemon:
        raise RuntimeError("Native generation requires allow_child_processes=true")
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    kwargs = resolved_native_server_kwargs(
        model_path, gpu_id, server_args_overrides, runtime_gpu_ids
    )
    output_dir = str(Path(output_dir).resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    generator = DiffGenerator.from_server_args(ServerArgs.from_kwargs(**kwargs))
    try:
        return NativeGenerationScheduler(generator, output_dir)
    except BaseException:
        generator.shutdown()
        raise
