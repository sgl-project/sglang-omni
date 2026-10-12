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

from sglang_omni.models.cosmos3.checkpoint import resolve_native_checkpoint
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


def resolve_generation_options(
    request_params: dict[str, Any], inputs: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Lower common SDK seeds and native overrides with one precedence rule."""
    params = {}
    if request_params.get("seed") is not None:
        params["seed"] = request_params["seed"]
    else:
        pass
    params.update(inputs or {})
    stage_sampling = request_params.get("stage_sampling")
    if stage_sampling is not None:
        if not isinstance(stage_sampling, dict):
            raise ValueError("stage_sampling must be a mapping")
        else:
            pass
        generation = stage_sampling.get("generation", {})
        if not isinstance(generation, dict):
            raise ValueError("Generation stage sampling must be a mapping")
        else:
            pass
        if generation.get("seed") is not None:
            params["seed"] = generation["seed"]
        else:
            pass
    else:
        pass
    overrides = request_params.get("diffusion", {})
    if not isinstance(overrides, dict):
        raise ValueError("diffusion parameters must be a mapping")
    else:
        pass
    params.update(overrides)
    stage_params = request_params.get("stage_params")
    if stage_params is not None:
        if not isinstance(stage_params, dict):
            raise ValueError("stage_params must be a mapping")
        else:
            pass
        generation = stage_params.get("generation", {})
        if not isinstance(generation, dict):
            raise ValueError("Generation stage parameters must be a mapping")
        else:
            pass
        params.update(generation)
    else:
        pass
    return params


def build_sampling_params(payload: StagePayload, output_dir: str) -> dict[str, Any]:
    inputs = payload.request.inputs
    if isinstance(inputs, str):
        inputs = {"prompt": inputs}
    elif not isinstance(inputs, dict):
        raise ValueError("Generation inputs must be a prompt or native sampling fields")
    else:
        pass
    params = resolve_generation_options(payload.request.params, inputs)
    action_mode = params.get("action_mode")
    if action_mode is not None:
        action_mode = str(action_mode).strip().lower()
        if action_mode not in ("policy", "inverse_dynamics", "forward_dynamics"):
            raise ValueError(f"Unsupported action_mode={action_mode!r}")
        else:
            pass
        params["action_mode"] = action_mode
        params.setdefault("prompt", "")
    else:
        pass
    if not isinstance(params.get("prompt"), str) or (
        action_mode is None and not params["prompt"].strip()
    ):
        raise ValueError("Generation requires a nonempty prompt")
    else:
        pass
    if (
        action_mode in ("policy", "inverse_dynamics")
        and params.get("num_outputs_per_prompt", 1) != 1
    ):
        raise ValueError("Action generation returns one action sequence per request")
    else:
        pass
    if payload.request.params.get("stream"):
        raise ValueError("Use the native realtime video API for streaming generation")
    else:
        pass
    if params.get("prompt_file_path"):
        raise ValueError("Submit prompts individually through the Omni client")
    else:
        pass
    params.update(
        save_output=True,
        return_file_paths_only=True,
        output_path=output_dir,
        output_file_name=uuid4().hex,
    )
    return params


class NativeGenerationScheduler(SimpleScheduler):
    """Keep scheduler and pipeline execution in the native SGLang runtime."""

    def __init__(
        self,
        generator: Any,  # noqa: ANN401 - Native runtime types are loaded at factory time.
        output_dir: str,
    ):
        self.generator = generator
        self.output_dir = output_dir
        self.native_requests: dict[str, GenerationRequest] = {}
        super().__init__(
            self.generate,
            abort_callback=self.cancel_native,
            shutdown_callback=generator.shutdown,
        )
        self.requires_tp_work_fanout = False

    def next_message(self):
        for process in self.generator.local_scheduler_process or []:
            if not process.is_alive():
                raise RuntimeError(
                    f"Native generation worker {process.pid} exited "
                    f"with code {process.exitcode}"
                )
            else:
                pass
        return super().next_message()

    @staticmethod
    def remove_request_directory(directory: Path) -> None:
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            pass

    def cancel_native(self, request_id: str) -> None:
        directory = None
        with self.abort_lock:
            request = self.native_requests.get(request_id)
            if request is not None:
                request.cancelled.set()
                if not request.running and not request.delivering:
                    self.native_requests.pop(request_id)
                    directory = request.directory
                else:
                    pass
            else:
                pass
        if directory is not None:
            self.remove_request_directory(directory)
        else:
            pass

    def emit_result(self, request_id, result, outbox) -> None:
        directory = None
        with self.abort_lock:
            request = self.native_requests.get(request_id)
            if request_id in self.aborted:
                self.aborted.discard(request_id)
                if request is not None and request.result is result:
                    self.native_requests.pop(request_id)
                    directory = request.directory
                else:
                    pass
            else:
                super().emit_result(request_id, result, outbox)
        if directory is not None:
            self.remove_request_directory(directory)
        else:
            pass

    def claim_result(self, result: StagePayload, *, terminal: bool) -> bool:
        if not terminal:
            raise ValueError(
                "Saved native media requires terminal delivery; "
                "use inline media for inter-stage routing"
            )
        else:
            pass
        with self.abort_lock:
            request = self.native_requests.get(result.request_id)
            if request is None or request.result is not result:
                return False
            else:
                pass
            request.delivering = True
            return True

    def release_result(self, result: StagePayload, *, delivered: bool) -> None:
        directory = None
        with self.abort_lock:
            request = self.native_requests.get(result.request_id)
            if request is not None and request.result is result:
                self.native_requests.pop(result.request_id)
                if not delivered:
                    directory = request.directory
                else:
                    pass
            else:
                pass
        if directory is not None:
            self.remove_request_directory(directory)
        else:
            pass

    def stop(self) -> None:
        directories = []
        with self.abort_lock:
            for request_id, request in list(self.native_requests.items()):
                request.cancelled.set()
                if not request.running and not request.delivering:
                    self.native_requests.pop(request_id)
                    directories.append(request.directory)
                else:
                    pass
        try:
            super().stop()
        finally:
            for directory in directories:
                self.remove_request_directory(directory)

    def generate(self, payload: StagePayload) -> StagePayload:
        params = build_sampling_params(payload, self.output_dir)
        action_output = params.get("action_mode") in ("policy", "inverse_dynamics")
        if action_output:
            # Use native SDK sampling fields and normalization. HTTP observation
            # envelopes remain owned by the mounted native action endpoint.
            params.update(save_output=False, return_file_paths_only=False)
        else:
            pass
        with self.abort_lock:
            if payload.request_id in self.aborted:
                return payload
            else:
                pass
            request = GenerationRequest(
                threading.Event(),
                Path(tempfile.mkdtemp(prefix="request_", dir=self.output_dir)),
            )
            self.native_requests[payload.request_id] = request
        params["output_path"] = str(request.directory)
        kwargs = {"sampling_params_kwargs": params}
        if (
            getattr(self.generator, "supports_cancellation", False)
            and params.get("num_outputs_per_prompt", 1) == 1
            and not action_output
        ):
            kwargs["cancellation_event"] = request.cancelled
        else:
            pass
        succeeded = False
        try:
            if action_output:
                from sglang.multimodal_gen.runtime.entrypoints.action.protocol import (
                    action_generation_response,
                )

                # Native generate_action has no cancellation_event contract.
                # Wait for it to settle before the existing finally block can
                # remove cancelled outputs or release request ownership.
                output = self.generator.generate_action(**kwargs)
                response = action_generation_response(
                    output, self.generator.server_args
                )
                path = request.directory / "action.json"
                path.write_text(json.dumps(response, allow_nan=False), encoding="utf-8")
                payload.data = {
                    "media": [{"path": str(path), "modality": "action"}],
                    "finish_reason": "stop",
                }
                succeeded = True
                return payload
            else:
                pass
            results = self.generator.generate(**kwargs)
            if results is None:
                raise RuntimeError(
                    "Native generation failed without producing an output"
                )
            else:
                pass
            if not isinstance(results, list):
                results = [results]
            else:
                pass
            if not results or any(not result.output_file_path for result in results):
                raise RuntimeError("Native generation returned no saved media")
            else:
                pass
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
            with self.abort_lock:
                request.running = False
                request.result = payload if succeeded else None
                cleanup = not succeeded or request.cancelled.is_set()
                if cleanup and self.native_requests.get(payload.request_id) is request:
                    self.native_requests.pop(payload.request_id)
                else:
                    pass
            if cleanup:
                self.remove_request_directory(request.directory)
            else:
                pass


@dataclass
class GenerationRequest:
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
    else:
        pass
    devices = list(runtime_gpu_ids) if runtime_gpu_ids is not None else [gpu_id]
    if not devices or devices[0] != gpu_id or len(set(devices)) != len(devices):
        raise ValueError("Native GPU placement must start with the stage GPU")
    else:
        pass
    if any(gpu < 0 for gpu in devices):
        raise ValueError("Native GPU ids must be nonnegative")
    else:
        pass
    if kwargs.get("num_gpus", len(devices)) != len(devices):
        raise ValueError("Native GPU count must match Omni placement")
    else:
        pass
    if kwargs.get("nnodes", 1) != 1:
        raise ValueError("This stage owns GPUs on one host")
    else:
        pass
    if any(key in kwargs for key in ("gpu_ids", "base_gpu_id", "model_path")):
        raise ValueError("Set the generation model and GPUs through Omni placement")
    else:
        pass
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
    else:
        pass
    kwargs = native_server_kwargs(
        model_path, gpu_id, server_args_overrides, runtime_gpu_ids
    )
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    kwargs = resolve_native_checkpoint(kwargs)
    output_dir = str(Path(output_dir).resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    generator = DiffGenerator.from_server_args(ServerArgs.from_kwargs(**kwargs))
    try:
        return NativeGenerationScheduler(generator, output_dir)
    except BaseException:
        generator.shutdown()
        raise
