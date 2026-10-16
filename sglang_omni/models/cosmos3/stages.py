# SPDX-License-Identifier: Apache-2.0
"""Adapt Omni payloads to the native SGLang generation API."""

from __future__ import annotations

import logging
import multiprocessing
import multiprocessing.connection
import os
import queue
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse
from uuid import uuid4

from sglang_omni.native_generation import cancellation
from sglang_omni.preprocessing.resource_connector import (
    MediaPolicyError,
    get_global_resource_connector,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

logger = logging.getLogger(__name__)

REQUEST_ERROR = "Native generation request"

# SamplingParams fields that native SGLang's image and video routes accept from
# clients (ImageGenerationsRequest and VideoGenerationsRequest in the
# multimodal_gen OpenAI protocol). Server paths such as output_path,
# prompt_path, perf_dump_path and model paths, profiling switches and raw
# diffusers kwargs stay with the server. Each model adds the fields it declares
# for those routes, see model_request_fields.
COMMON_REQUEST_FIELDS = frozenset(
    {
        "attention_backend_override",
        "cache_dit_params",
        "cfg_gate_step",
        "enable_cache_dit",
        "enable_frame_interpolation",
        "enable_teacache",
        "enable_upscaling",
        "flow_shift",
        "fps",
        "frame_interpolation_exp",
        "frame_interpolation_scale",
        "generator_device",
        "guidance_scale",
        "guidance_scale_2",
        "height",
        "image_path",
        "max_sequence_length",
        "negative_prompt",
        "num_frames",
        "num_inference_steps",
        "num_outputs_per_prompt",
        "output_compression",
        "output_quality",
        "progressive_delta",
        "progressive_levels",
        "progressive_mode",
        "prompt",
        "quality",
        "seed",
        "skip_softmax_params",
        "task_type",
        "true_cfg_scale",
        "upscaling_scale",
        "video_path",
        "width",
    }
)

# Native image and video routes clamp n to this many outputs per prompt.
# Native SGLang keeps the bound inline, so it is restated here.
NATIVE_MAX_OUTPUTS_PER_PROMPT = 10


def model_request_fields(server_args: object) -> frozenset[str]:
    """Fields the served model declares for native image and video requests."""
    from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
        get_sampling_request_extra_fields,
        resolve_sampling_params_cls,
    )

    sampling_params_cls = resolve_sampling_params_cls(server_args)
    return get_sampling_request_extra_fields(
        sampling_params_cls, "image"
    ) | get_sampling_request_extra_fields(sampling_params_cls, "video")


def resolve_generation_options(
    request_params: dict[str, object], inputs: dict[str, object] | None = None
) -> dict[str, object]:
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
            raise ValueError(f"{REQUEST_ERROR} stage_sampling must be a mapping")
        else:
            pass
        generation = stage_sampling.get("generation", {})
        if not isinstance(generation, dict):
            raise ValueError(
                f"{REQUEST_ERROR} stage_sampling generation must be a mapping"
            )
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
        raise ValueError(f"{REQUEST_ERROR} diffusion must be a mapping")
    else:
        pass
    params.update(overrides)
    stage_params = request_params.get("stage_params")
    if stage_params is not None:
        if not isinstance(stage_params, dict):
            raise ValueError(f"{REQUEST_ERROR} stage_params must be a mapping")
        else:
            pass
        generation = stage_params.get("generation", {})
        if not isinstance(generation, dict):
            raise ValueError(
                f"{REQUEST_ERROR} stage_params generation must be a mapping"
            )
        else:
            pass
        params.update(generation)
    else:
        pass
    return params


def build_sampling_params(
    payload: StagePayload,
    output_dir: str,
    request_fields: frozenset[str] = COMMON_REQUEST_FIELDS,
) -> dict[str, object]:
    """Lower a request to native sampling fields that clients may set."""
    inputs = payload.request.inputs
    if isinstance(inputs, str):
        inputs = {"prompt": inputs}
    elif not isinstance(inputs, dict):
        raise ValueError(
            f"{REQUEST_ERROR} inputs must be a prompt or native sampling fields"
        )
    else:
        pass
    params = resolve_generation_options(payload.request.params, inputs)
    refused = sorted(set(params) - request_fields)
    if refused:
        raise ValueError(
            f"{REQUEST_ERROR} cannot set {', '.join(refused)}. Clients may set "
            "the fields that the native image and video routes accept for this "
            "model, and the server owns paths it writes or reads on its own."
        )
    else:
        pass
    if not isinstance(params.get("prompt"), str) or not params["prompt"].strip():
        raise ValueError(f"{REQUEST_ERROR} requires a nonempty prompt")
    else:
        pass
    if payload.request.params.get("stream"):
        raise ValueError(
            f"{REQUEST_ERROR} cannot stream. Use the native realtime video API "
            "for streaming generation"
        )
    else:
        pass
    outputs = params.get("num_outputs_per_prompt", 1)
    if (
        not isinstance(outputs, int)
        or isinstance(outputs, bool)
        or not 1 <= outputs <= NATIVE_MAX_OUTPUTS_PER_PROMPT
    ):
        raise ValueError(
            f"{REQUEST_ERROR} num_outputs_per_prompt must be an integer from 1 "
            f"to {NATIVE_MAX_OUTPUTS_PER_PROMPT}, as on the native routes"
        )
    else:
        pass
    params.update(
        save_output=True,
        return_file_paths_only=True,
        output_path=output_dir,
        output_file_name=uuid4().hex,
    )
    return params


def police_media_inputs(
    params: dict[str, object], request_fields: frozenset[str], directory: Path
) -> list[Path]:
    """Apply the server media policy to client media paths and URLs.

    Without a configured policy the native loaders read them as before. With
    one, local paths and file URLs must sit in the allowed directory, and
    http(s) URLs are fetched through the connector, which checks the domain on
    every redirect, into the request directory. Returns the fetched files.
    """
    connector = get_global_resource_connector()
    if (
        connector.allowed_local_media_path is None
        and not connector.allowed_media_domains
    ):
        return []
    else:
        pass
    fetched: list[Path] = []

    def police(name: str, source: object) -> str:
        if not isinstance(source, str):
            raise ValueError(f"{REQUEST_ERROR} {name} must be a path or URL")
        elif source.startswith("data:"):
            return source
        elif source.startswith(("http://", "https://")):
            data, _ = connector.load_http_bytes(
                source, timeout=connector.connection.timeout, max_bytes=None
            )
            target = (
                directory / f"input_{uuid4().hex}{Path(urlparse(source).path).suffix}"
            )
            target.write_bytes(data)
            fetched.append(target)
            return str(target)
        elif source.startswith("file://"):
            return str(connector.local_media_path(unquote(urlparse(source).path)))
        else:
            return str(connector.local_media_path(source))

    for name in sorted(params):
        if name not in request_fields or not name.endswith("_path"):
            continue
        else:
            pass
        value = params[name]
        try:
            if isinstance(value, list):
                params[name] = [police(name, item) for item in value]
            else:
                params[name] = police(name, value)
        except MediaPolicyError as exc:
            raise ValueError(f"{REQUEST_ERROR} {name} refused: {exc}") from exc
    return fetched


class NativeGenerationScheduler(SimpleScheduler):
    """Keep scheduler and pipeline execution in the native SGLang runtime.

    Native generation of each request runs in its own workspace directory
    under output_dir, which the stage removes once generation settles, and
    the saved media then sit in output_dir. Saved media belong to whoever
    holds the result. The stage removes them while it still holds them: when
    the request fails, is aborted before routing hands the completion to the
    transport, or that hand off fails. After the transport accepted the
    completion, the coordinator owns the media and removes them if it drops
    the completion.
    """

    def __init__(
        self,
        generator: object,
        output_dir: str,
        request_fields: frozenset[str] = COMMON_REQUEST_FIELDS,
        *,
        cancellation_mode: str | None = None,
    ):
        """``cancellation_mode`` is "plugin" when the Omni plugin runs in the
        native worker, else None."""
        self.generator = generator
        self.output_dir = output_dir
        self.request_fields = request_fields
        self.cancellation_mode = cancellation_mode
        self.native_requests: dict[str, GenerationRequest] = {}
        self.workers = list(generator.local_scheduler_process or [])
        self.worker_failure: RuntimeError | None = None
        self.closing = False
        super().__init__(
            self.generate,
            abort_callback=self.cancel_native,
            shutdown_callback=generator.shutdown,
        )
        self.requires_tp_work_fanout = False
        if self.workers:
            threading.Thread(
                target=self.watch_workers, name="native-worker-watch", daemon=True
            ).start()
        else:
            pass

    @staticmethod
    def worker_exit(process: multiprocessing.process.BaseProcess) -> RuntimeError:
        return RuntimeError(
            f"Native generation worker {process.pid} exited "
            f"with code {process.exitcode}"
        )

    def watch_workers(self) -> None:
        """Fail the stage on a native worker exit while idle, without polling."""
        sentinels = {process.sentinel: process for process in self.workers}
        ready = multiprocessing.connection.wait(list(sentinels))
        if not self.closing and self.worker_failure is None:
            self.worker_failure = self.worker_exit(sentinels[ready[0]])
            logger.error("%s, stopping the generation stage", self.worker_failure)
            # Wake a scheduler loop that waits for work, so it raises the failure.
            self.inbox.put(IncomingMessage("", "wakeup"))
        else:
            pass

    def next_message(self) -> IncomingMessage | None:
        if self.worker_failure is not None:
            raise self.worker_failure
        else:
            pass
        return super().next_message()

    def run_native(self, kwargs: dict[str, object]) -> object:
        """Run native generation on a helper thread and wait for it or a worker exit.

        Native generation blocks until its scheduler answers, which never
        happens once a worker died, so waiting on the worker sentinels fails the
        request at once. The stage then stops on its next scheduler iteration.
        """
        reader, writer = multiprocessing.Pipe(duplex=False)
        outcome: dict[str, object] = {}

        def run() -> None:
            try:
                outcome["result"] = self.generator.generate(**kwargs)
            except BaseException as exc:
                outcome["error"] = exc
            finally:
                writer.close()

        threading.Thread(target=run, name="native-generation", daemon=True).start()
        sentinels = {process.sentinel: process for process in self.workers}
        try:
            ready = multiprocessing.connection.wait([reader, *sentinels])
        finally:
            reader.close()
        if reader not in ready:
            failure = self.worker_exit(sentinels[ready[0]])
            if self.worker_failure is None:
                self.worker_failure = failure
            else:
                pass
            raise failure
        elif "error" in outcome:
            raise outcome["error"]
        else:
            return outcome.get("result")

    def signal_native_cancellation(self, request: GenerationRequest) -> None:
        """Mark a running request cancelled. Call with abort_lock held, so the
        marker lands before generate removes the request workspace."""
        try:
            (request.directory / cancellation.CANCELLED_MARKER).touch()
        except FileNotFoundError:
            pass

    @staticmethod
    def remove_request_directory(directory: Path) -> None:
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            pass

    @staticmethod
    def remove_outputs(request: GenerationRequest) -> None:
        for path in request.outputs:
            path.unlink(missing_ok=True)

    def cancel_native(self, request_id: str) -> None:
        discarded = None
        with self.abort_lock:
            request = self.native_requests.get(request_id)
            if request is None:
                # Nothing held, or the transport accepted the completion and
                # the coordinator owns its media.
                return
            else:
                pass
            request.cancelled.set()
            if not request.running and not request.delivering:
                self.native_requests.pop(request_id)
                discarded = request
            elif request.running and self.cancellation_mode == "plugin":
                self.signal_native_cancellation(request)
            else:
                pass
        if discarded is not None:
            self.remove_outputs(discarded)
        else:
            pass

    def emit_result(
        self,
        request_id: str,
        result: object,
        outbox: queue.Queue[OutgoingMessage],
    ) -> None:
        discarded = None
        with self.abort_lock:
            request = self.native_requests.get(request_id)
            if request_id in self.aborted:
                self.aborted.discard(request_id)
                if request is not None and request.result is result:
                    self.native_requests.pop(request_id)
                    discarded = request
                else:
                    pass
            else:
                super().emit_result(request_id, result, outbox)
        if discarded is not None:
            self.remove_outputs(discarded)
        else:
            pass

    def claim_result(self, result: object, *, terminal: bool) -> bool:
        if not isinstance(result, StagePayload):
            return False
        elif not terminal:
            raise ValueError(
                "Saved native media requires terminal delivery. "
                "Use inline media for inter-stage routing"
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

    def release_result(self, result: object, *, delivered: bool) -> None:
        if not isinstance(result, StagePayload):
            return
        else:
            pass
        discarded = None
        with self.abort_lock:
            request = self.native_requests.get(result.request_id)
            if request is not None and request.result is result:
                self.native_requests.pop(result.request_id)
                # An accepted completion may already be with a caller, even
                # after an abort, so its media leave the stage's hands.
                discarded = None if delivered else request
            else:
                pass
        if discarded is not None:
            self.remove_outputs(discarded)
        else:
            pass

    def stop(self) -> None:
        discarded = []
        self.closing = True
        with self.abort_lock:
            for request_id, request in list(self.native_requests.items()):
                request.cancelled.set()
                if not request.running and not request.delivering:
                    self.native_requests.pop(request_id)
                    discarded.append(request)
                elif request.running and self.cancellation_mode == "plugin":
                    self.signal_native_cancellation(request)
                else:
                    pass
        try:
            super().stop()
        finally:
            for request in discarded:
                self.remove_outputs(request)

    def generate(self, payload: StagePayload) -> StagePayload:
        params = build_sampling_params(payload, self.output_dir, self.request_fields)
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
        armed = None
        if self.cancellation_mode == "plugin":
            armed = request.directory / cancellation.ARMED_MARKER
        else:
            pass
        succeeded = False
        try:
            if armed is not None:
                armed.touch()
            else:
                pass
            police_media_inputs(params, self.request_fields, request.directory)
            results = self.run_native(kwargs)
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
            for result in results:
                # Native names each output after the request's unique
                # output_file_name, so the name is free in output_dir.
                source = Path(result.output_file_path)
                target = Path(self.output_dir) / source.name
                os.replace(source, target)
                request.outputs.append(target)
            payload.data = {
                "media": [
                    {
                        "path": str(path),
                        "size": result.size,
                        "prompt": result.prompt,
                        "generation_time": result.generation_time,
                        "peak_memory_mb": result.peak_memory_mb,
                        "metrics": result.metrics,
                    }
                    for result, path in zip(results, request.outputs, strict=True)
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
            self.remove_request_directory(request.directory)
            if cleanup:
                self.remove_outputs(request)
            else:
                pass


@dataclass
class GenerationRequest:
    cancelled: threading.Event
    directory: Path
    running: bool = True
    delivering: bool = False
    result: StagePayload | None = None
    outputs: list[Path] = field(default_factory=list)


def native_server_kwargs(
    model_path: str,
    gpu_id: int,
    overrides: dict[str, object] | None,
    runtime_gpu_ids: list[int] | None = None,
) -> dict[str, object]:
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
    # The checkpoint names a pipeline class that only native SGLang defines,
    # so the auto backend's diffusers fallback could never load it. Asking
    # for the native backend lets this process name the pipeline without
    # importing it. An explicit backend override still wins.
    kwargs.setdefault("backend", "sglang")
    if runtime_gpu_ids is None:
        kwargs["base_gpu_id"] = gpu_id
    else:
        kwargs["gpu_ids"] = devices
    return kwargs


def create_generation_scheduler(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    output_dir: str = "outputs",
    runtime_gpu_ids: list[int] | None = None,
    server_args_overrides: dict[str, object] | None = None,
) -> NativeGenerationScheduler:
    from sglang_omni.utils.device import resolve_concrete_device

    concrete_device = resolve_concrete_device(device, gpu_id)
    if concrete_device.index is None:
        raise ValueError("Native Cosmos3 execution requires an indexed accelerator")
    else:
        pass
    gpu_id = concrete_device.index
    if multiprocessing.current_process().daemon:
        raise RuntimeError("Native generation requires allow_child_processes=true")
    else:
        pass
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )
    from sglang.multimodal_gen.runtime.platforms.plugins import apply_plugin_hooks
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    kwargs = native_server_kwargs(
        model_path, gpu_id, server_args_overrides, runtime_gpu_ids
    )
    output_dir = str(Path(output_dir).resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Native applies plugin hooks once per process, ahead of its server
    # arguments as DiffGenerator.from_pretrained does. Applying them here,
    # outside the activation below, keeps this host out of the hooks it arms
    # for its native workers.
    apply_plugin_hooks()
    server_args = ServerArgs.from_kwargs(**kwargs)
    handshake = (
        tempfile.mkdtemp(prefix="sglang-omni-native-")
        if kwargs["num_gpus"] == 1
        else None
    )
    try:
        with cancellation.activation_environment(handshake):
            generator = DiffGenerator.from_server_args(server_args)
        if handshake is not None and os.path.exists(
            cancellation.handshake_path(handshake, 0)
        ):
            mode = "plugin"
        else:
            mode = None
    finally:
        if handshake is not None:
            shutil.rmtree(handshake, ignore_errors=True)
        else:
            pass
    if mode is None:
        logger.warning(
            "%s An aborted request keeps the generation stage busy until its "
            "native generation finishes.",
            degraded_cancellation_reason(kwargs["num_gpus"]),
        )
    else:
        pass
    try:
        return NativeGenerationScheduler(
            generator,
            output_dir,
            COMMON_REQUEST_FIELDS | model_request_fields(generator.server_args),
            cancellation_mode=mode,
        )
    except BaseException:
        generator.shutdown()
        raise


def degraded_cancellation_reason(num_gpus: object) -> str:
    if num_gpus != 1:
        return (
            "Cooperative generation cancellation covers one native GPU, and "
            f"this stage runs {num_gpus}."
        )
    else:
        return (
            "Cooperative generation cancellation did not start in the native "
            "worker, whose log names the reason. List "
            f"{cancellation.PLUGIN_NAME} in SGLANG_PLUGINS when that variable is set."
        )
