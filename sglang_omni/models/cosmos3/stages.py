# SPDX-License-Identifier: Apache-2.0
"""Adapt Omni payloads to the native SGLang generation API."""

from __future__ import annotations

import json
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


def build_sampling_params(
    payload: StagePayload, output_dir: str, *, is_edge: bool = False
) -> dict[str, Any]:
    inputs = payload.request.inputs
    if isinstance(inputs, str):
        params = {"prompt": inputs}
    elif isinstance(inputs, dict):
        params = dict(inputs)
    else:
        raise ValueError("Generation inputs must be a prompt or native sampling fields")
    overrides = payload.request.params.get("diffusion", {})
    if not isinstance(overrides, dict):
        raise ValueError("diffusion parameters must be a mapping")
    params.update(overrides)
    stage_params = payload.request.params.get("stage_params")
    if stage_params is not None:
        if not isinstance(stage_params, dict):
            raise ValueError("stage_params must be a mapping")
        generation = stage_params.get("generation", {})
        if not isinstance(generation, dict):
            raise ValueError("Generation stage parameters must be a mapping")
        params.update(generation)
    if is_edge and params.get("action_mode") is not None:
        params["action_mode"] = str(params["action_mode"]).strip().lower()
    action_request = is_edge and params.get("action_mode") in (
        "policy",
        "inverse_dynamics",
        "forward_dynamics",
    )
    if action_request:
        params.setdefault("prompt", "")
    if not isinstance(params.get("prompt"), str) or (
        not action_request and not params["prompt"].strip()
    ):
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
        self._watch_stopped = threading.Event()
        self._native_failure: RuntimeError | None = None
        super().__init__(
            self._generate,
            abort_callback=self._cancel_native,
            shutdown_callback=generator.shutdown,
        )
        self.requires_tp_work_fanout = False

    def _check_native_workers(self) -> None:
        for process in self.generator.local_scheduler_process or []:
            if not process.is_alive():
                if self._watch_stopped.is_set():
                    return
                raise RuntimeError(
                    f"Native generation worker {process.pid} exited "
                    f"with code {process.exitcode}"
                )

    def _watch_native_workers(self) -> None:
        while not self._watch_stopped.wait(0.1):
            try:
                self._check_native_workers()
            except RuntimeError as exc:
                if self._watch_stopped.is_set():
                    return
                self._native_failure = exc
                # Stop the runtime to unblock an in-flight scheduler RPC
                super().stop()
                return

    def start(self) -> None:
        watcher = threading.Thread(
            target=self._watch_native_workers,
            name="cosmos3-worker-watch",
            daemon=True,
        )
        watcher.start()
        try:
            super().start()
            if self._native_failure is not None:
                raise self._native_failure
        finally:
            self._watch_stopped.set()
            watcher.join()

    def _next_message(self):
        if self._native_failure is not None:
            raise self._native_failure
        self._check_native_workers()
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
                if not request.running:
                    self._native_requests.pop(request_id)
                    directory = request.directory
        if directory is not None:
            self._remove_request_directory(directory)

    def _emit_result(self, request_id, result, outbox) -> None:
        directory = None
        with self._abort_lock:
            request = self._native_requests.pop(request_id, None)
            if request_id in self._aborted:
                self._aborted.discard(request_id)
                if request is not None:
                    directory = request.directory
            else:
                super()._emit_result(request_id, result, outbox)
        if directory is not None:
            self._remove_request_directory(directory)

    def stop(self) -> None:
        self._watch_stopped.set()
        with self._abort_lock:
            for request in self._native_requests.values():
                request.cancelled.set()
        super().stop()

    def _generate(self, payload: StagePayload) -> StagePayload:
        native_args = getattr(self.generator, "server_args", None)
        is_edge = getattr(
            getattr(native_args, "pipeline_config", None), "is_edge", False
        )
        params = build_sampling_params(payload, self.output_dir, is_edge=is_edge)
        action_output = is_edge and params.get("action_mode") in (
            "policy",
            "inverse_dynamics",
        )
        if action_output:
            from sglang_omni.models.cosmos3.action import action_sampling_kwargs

            params = action_sampling_kwargs(params, native_args)
        elif (
            is_edge
            and params.get("num_frames") != 1
            and not any(
                params.get(name)
                for name in ("enable_frame_interpolation", "enable_upscaling")
            )
        ):
            # Avoid the native CUDA video writer's buffer lifetime risk
            params.update(
                return_file_paths_only=False,
                return_frames=True,
            )
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
            and not action_output
        ):
            kwargs["cancellation_event"] = request.cancelled
        succeeded = False
        try:
            if action_output:
                from sglang.multimodal_gen.runtime.entrypoints.action.protocol import (
                    action_generation_response,
                )

                output = self.generator.generate_action(**kwargs)
                response = action_generation_response(
                    output, self.generator.server_args
                )
                path = request.directory / "action.json"
                path.write_text(json.dumps(response), encoding="utf-8")
                payload.data = {
                    "media": [{"path": str(path), "modality": "action"}],
                    "finish_reason": "stop",
                }
                succeeded = True
                return payload
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
                cleanup = not succeeded or request.cancelled.is_set()
                if cleanup:
                    self._native_requests.pop(payload.request_id, None)
            if cleanup:
                self._remove_request_directory(request.directory)


@dataclass
class _GenerationRequest:
    cancelled: threading.Event
    directory: Path
    running: bool = True


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
    if kwargs.get("nnodes", 1) != 1:
        raise ValueError("This stage owns GPUs on one host")
    if any(key in kwargs for key in ("gpu_ids", "base_gpu_id", "model_path")):
        raise ValueError("Set the generation model and GPUs through Omni placement")
    kwargs.update(model_path=model_path, num_gpus=len(devices))
    if runtime_gpu_ids is None:
        kwargs["base_gpu_id"] = gpu_id
    else:
        kwargs["gpu_ids"] = devices
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

    from sglang_omni.utils.checkpoint import resolve_checkpoint

    kwargs = native_server_kwargs(
        model_path, gpu_id, server_args_overrides, runtime_gpu_ids
    )
    kwargs["model_path"] = resolve_checkpoint(model_path)
    kwargs.setdefault("served_model_name", model_path.partition("@")[0])
    output_dir = str(Path(output_dir).resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    generator = DiffGenerator.from_server_args(ServerArgs.from_kwargs(**kwargs))
    try:
        return NativeGenerationScheduler(generator, output_dir)
    except BaseException:
        generator.shutdown()
        raise
