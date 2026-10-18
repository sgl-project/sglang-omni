# SPDX-License-Identifier: Apache-2.0
"""Adapt Omni payloads to the native SGLang generation API."""

from __future__ import annotations

import base64
import copy
import hashlib
import heapq
import itertools
import logging
import math
import multiprocessing
import multiprocessing.connection
import os
import queue
import shutil
import stat
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse, urlunparse
from uuid import uuid4

from sglang_omni.native_generation import cancellation
from sglang_omni.native_generation.media import (
    NativeMediaApp,
    native_media_server_kwargs,
)
from sglang_omni.preprocessing.resource_connector import (
    MediaPolicyError,
    MultiModalResourceConnector,
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

# Largest saved output a stage with inline_media returns as bytes.
INLINE_MEDIA_LIMIT_BYTES = 8 * 1024 * 1024
# Request metadata key of the order a stage with inline_media serves its
# queued requests in: a number or a list of numbers, smallest first, such as
# a deadline and an admission index. Requests without one, and equal ones,
# keep arrival order.
DEADLINE_METADATA_KEY = "generation_deadline"
INLINE_MEDIA_TYPES = {
    ".png": ("image", "image/png"),
    ".jpg": ("image", "image/jpeg"),
    ".jpeg": ("image", "image/jpeg"),
    ".webp": ("image", "image/webp"),
    ".mp4": ("video", "video/mp4"),
}


def data_url(mime: str, content: bytes) -> str:
    """Base64 data URL (RFC 2397) carrying content of this media type."""
    header = urlunparse(("data", "", mime, "base64,", "", ""))
    return header + base64.b64encode(content).decode("ascii")


def inline_media(path: str, directory: Path) -> dict[str, object]:
    """Read one bounded saved output while its request directory is owned."""
    target = Path(path)
    if not target.resolve(strict=True).is_relative_to(directory.resolve(strict=True)):
        raise ValueError("Native inline output is outside its owned request directory")
    elif target.suffix.lower() not in INLINE_MEDIA_TYPES:
        raise ValueError("Unsupported native inline media type")
    else:
        pass
    descriptor = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        size = os.fstat(stream.fileno())
        if not stat.S_ISREG(size.st_mode) or not 0 < size.st_size <= (
            INLINE_MEDIA_LIMIT_BYTES
        ):
            raise ValueError("Native inline media exceeds its regular file byte limit")
        else:
            pass
        content = stream.read(INLINE_MEDIA_LIMIT_BYTES + 1)
    if not 0 < len(content) <= INLINE_MEDIA_LIMIT_BYTES:
        raise ValueError("Native inline media exceeds its byte limit")
    else:
        pass
    kind, mime = INLINE_MEDIA_TYPES[target.suffix.lower()]
    return {
        "kind": kind,
        "mime_type": mime,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "url": data_url(mime, content),
    }


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


def media_policy_configured(connector: MultiModalResourceConnector) -> bool:
    return connector.allowed_local_media_path is not None or bool(
        connector.allowed_media_domains
    )


def check_media_source(
    connector: MultiModalResourceConnector, name: str, source: object
) -> str:
    """Apply the server media policy to one client path or URL without reading it.

    Returns what a loader may read: a data URL as is, a local path or file URL
    resolved inside the allowed directory, or an allowed http(s) URL as is,
    which its loader fetches under the same policy.
    """
    if not isinstance(source, str):
        raise ValueError(f"{REQUEST_ERROR} {name} must be a path or URL")
    elif source.startswith("data:"):
        return source
    elif source.startswith(("http://", "https://")):
        connector.assert_url_allowed(source)
        return source
    elif source.startswith("file://"):
        return str(connector.local_media_path(unquote(urlparse(source).path)))
    else:
        return str(connector.local_media_path(source))


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
    if not media_policy_configured(connector):
        return []
    else:
        pass
    fetched: list[Path] = []

    def police(name: str, source: object) -> str:
        if isinstance(source, str) and source.startswith(("http://", "https://")):
            data, _ = connector.load_http_bytes(
                source, timeout=connector.connection.timeout, max_bytes=None
            )
            target = (
                directory / f"input_{uuid4().hex}{Path(urlparse(source).path).suffix}"
            )
            target.write_bytes(data)
            fetched.append(target)
            return str(target)
        else:
            return check_media_source(connector, name, source)

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
    the completion. With inline_media the stage instead returns each output
    as bytes read from the workspace, for a receiver that is another stage,
    so no saved media leave the workspace. It then also serves queued
    requests by the deadline in their metadata and reports the time each
    request occupied it as service_s.
    """

    def __init__(
        self,
        generator: object,
        output_dir: str,
        request_fields: frozenset[str] = COMMON_REQUEST_FIELDS,
        *,
        cancellation_mode: str | None = None,
        inline_media: bool = False,
    ):
        """``cancellation_mode`` is "plugin" when the Omni plugin runs in the
        native worker, else None."""
        self.generator = generator
        self.output_dir = output_dir
        self.request_fields = request_fields
        self.cancellation_mode = cancellation_mode
        self.inline_media = inline_media
        self.queued: list[tuple[tuple[float, ...], int, IncomingMessage]] = []
        self.arrivals = itertools.count()
        self.native_requests: dict[str, GenerationRequest] = {}
        self.workers = list(generator.local_scheduler_process or [])
        self.worker_failure: RuntimeError | None = None
        self.closing = False
        # Set when native generation still runs its first request cold once
        # the stage serves, see start.
        self.warm_in_background = False
        super().__init__(
            self.generate,
            abort_callback=self.cancel_native,
            shutdown_callback=self.shutdown_native,
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

    def shutdown_native(self) -> None:
        """Shut native generation down, waiting for its workers only while
        they live.

        Native shutdown asks the scheduler to stop and waits for its answer,
        up to its own timeout. A worker that already exited, as on a signal
        to the whole process group, never answers, so the stage stops waiting
        once every worker exited, and the rest of the native shutdown, which
        only joins those workers and closes the client, finishes on its own
        thread.
        """
        reader, writer = multiprocessing.Pipe(duplex=False)
        failure: list[BaseException] = []

        def run() -> None:
            try:
                self.generator.shutdown()
            except BaseException as exc:
                failure.append(exc)
            finally:
                writer.close()

        threading.Thread(target=run, name="native-shutdown", daemon=True).start()
        waiting = [reader, *(process.sentinel for process in self.workers)]
        try:
            while reader in waiting and len(waiting) > 1:
                for ready in multiprocessing.connection.wait(waiting):
                    waiting.remove(ready)
            if reader in waiting and not self.workers:
                multiprocessing.connection.wait([reader])
            else:
                pass
        finally:
            reader.close()
        if failure:
            raise failure[0]
        else:
            pass

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
        elif not self.inline_media:
            return super().next_message()
        elif not self.queued:
            # Wait for work as SimpleScheduler does.
            message = super().next_message()
            if message is None:
                return None
            else:
                self.enqueue_by_deadline(message)
        else:
            pass
        while True:
            try:
                self.enqueue_by_deadline(self.inbox.get_nowait())
            except queue.Empty:
                break
        return heapq.heappop(self.queued)[2]

    def enqueue_by_deadline(self, message: IncomingMessage) -> None:
        payload = message.data
        value = (
            payload.request.metadata.get(DEADLINE_METADATA_KEY)
            if isinstance(payload, StagePayload)
            else None
        )
        key = tuple(value) if isinstance(value, (list, tuple)) else (value,)
        if not key or not all(
            type(part) in (int, float) and not math.isnan(part) for part in key
        ):
            key = (math.inf,)
        else:
            pass
        heapq.heappush(self.queued, (key, next(self.arrivals), message))

    def run_native(
        self,
        kwargs: dict[str, object],
        target: Callable[..., object] | None = None,
    ) -> object:
        """Run native generation, or target, on a helper thread and wait for it
        or a worker exit.

        Native generation blocks until its scheduler answers, which never
        happens once a worker died, so waiting on the worker sentinels fails the
        request at once. The stage then stops on its next scheduler iteration.
        """
        reader, writer = multiprocessing.Pipe(duplex=False)
        outcome: dict[str, object] = {}

        def run() -> None:
            try:
                outcome["result"] = (target or self.generator.generate)(**kwargs)
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

    def warm_up(self, directory: str | None = None) -> None:
        """Run native's synthetic server warmup, before the stage reports ready.

        These are the requests, shapes and residency probe that native's own
        server sends before it reports ready, sent through the scheduler client
        every request of this stage uses. Native fails open without explicit
        warmup resolutions, the only case this runs in, so a failed warmup is
        logged and startup continues, while a worker exit fails the start. The
        warmup image goes to a directory inside ``directory`` (the temporary
        directory by default) removed afterwards, because native writes it to
        a new temporary directory it never removes when input_save_path is
        None.
        """
        from sglang.multimodal_gen.runtime.scheduler_client import sync_scheduler_client
        from sglang.multimodal_gen.runtime.server_warmup import run_sync_client_warmup

        with tempfile.TemporaryDirectory(
            prefix="sglang-omni-native-warmup-", dir=directory
        ) as inputs:
            server_args = copy.copy(self.generator.server_args)
            server_args.input_save_path = inputs
            kwargs = {
                "server_args": server_args,
                "forward": sync_scheduler_client.forward,
            }
            try:
                self.run_native(kwargs, run_sync_client_warmup)
            except Exception:
                if self.worker_failure is not None:
                    raise
                else:
                    logger.warning(
                        "Synthetic server warmup failed, continuing startup",
                        exc_info=True,
                    )

    def start(self) -> None:
        """Serve requests, first starting the background warmup when due.

        Without native's server warmup before readiness, the first native
        request would pay one time costs, such as loading kernels, that no
        later request pays. The stage instead sends one step of native's
        synthetic warmup request while it already serves, so readiness never
        waits for it. A request that arrives meanwhile queues behind that step
        in the native scheduler and then runs warm.
        """
        if self.warm_in_background:
            self.warm_in_background = False
            threading.Thread(
                target=self.warm_up_in_background, name="native-warmup", daemon=True
            ).start()
        else:
            pass
        super().start()

    def warm_up_in_background(self) -> None:
        from sglang.multimodal_gen.runtime.scheduler_client import sync_scheduler_client
        from sglang.multimodal_gen.runtime.server_warmup import (
            build_client_warmup_reqs,
            prepare_warmup_image_path,
            should_include_warmup_image,
        )

        with tempfile.TemporaryDirectory(
            prefix="sglang-omni-native-warmup-", dir=self.output_dir
        ) as inputs:
            server_args = copy.copy(self.generator.server_args)
            server_args.input_save_path = inputs
            try:
                image = (
                    prepare_warmup_image_path(server_args)
                    if should_include_warmup_image(
                        server_args, server_based_warmup=True
                    )
                    else None
                )
                for req in build_client_warmup_reqs(
                    server_args, warmup_input_path=image, step_limit=1
                ):
                    if self.closing:
                        return
                    else:
                        pass
                    error = sync_scheduler_client.forward(req).error
                    if error is not None:
                        raise RuntimeError(error)
                    else:
                        pass
            except Exception:
                if not self.closing:
                    logger.warning(
                        "Background warmup failed, the first request runs cold",
                        exc_info=True,
                    )
                else:
                    pass

    def http_app(self) -> NativeMediaApp:
        """Native SGLang's media routes, served from this stage's process."""
        return NativeMediaApp(self.generator.server_args, self.request_fields)

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
        elif self.inline_media:
            # Inline results own nothing on disk.
            return True
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
        started = time.monotonic() if self.inline_media else 0.0
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
            paths = [Path(result.output_file_path) for result in results]
            if not self.inline_media:
                for index, source in enumerate(paths):
                    # Native names each output after the request's unique
                    # output_file_name, so the name is free in output_dir.
                    paths[index] = Path(self.output_dir) / source.name
                    os.replace(source, paths[index])
                    request.outputs.append(paths[index])
            else:
                pass
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
                    for result, path in zip(results, paths, strict=True)
                ],
                "finish_reason": "stop",
            }
            if self.inline_media:
                for item in payload.data["media"]:
                    item.update(inline_media(item.pop("path"), request.directory))
                payload.data["service_s"] = time.monotonic() - started
            else:
                pass
            succeeded = True
            return payload
        finally:
            with self.abort_lock:
                request.running = False
                request.result = payload if succeeded else None
                cleanup = (
                    self.inline_media or not succeeded or request.cancelled.is_set()
                )
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
    inline_media: bool = False,
) -> NativeGenerationScheduler:
    """Start native generation for this stage.

    The native settings the media routes rely on are settled here, in the
    stage process, before the native runtime starts, and the scheduler serves
    those routes from this process. When the native server arguments select
    native's server warmup, the stage runs it before it reports ready, as
    native's own server does, with its input under output_dir."""
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
    from sglang.multimodal_gen.runtime.server_warmup import (
        is_realtime_serving,
        should_run_explicit_client_warmup,
        should_run_synthetic_server_warmup,
        supports_synthetic_warmup,
    )

    output_dir = str(Path(output_dir).resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    kwargs = native_server_kwargs(
        model_path,
        gpu_id,
        native_media_server_kwargs(server_args_overrides, output_dir=output_dir),
        runtime_gpu_ids,
    )
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
    scheduler = None
    try:
        scheduler = NativeGenerationScheduler(
            generator,
            output_dir,
            COMMON_REQUEST_FIELDS | model_request_fields(generator.server_args),
            cancellation_mode=mode,
            inline_media=inline_media,
        )
        # With explicit warmup resolutions DiffGenerator already warmed up.
        if should_run_explicit_client_warmup(generator.server_args):
            pass
        elif should_run_synthetic_server_warmup(generator.server_args):
            # A stage stopped during the warmup runs no finalizer, so its
            # input sits with the request workspaces of this stage.
            scheduler.warm_up(output_dir)
        else:
            scheduler.warm_in_background = supports_synthetic_warmup(
                generator.server_args
            ) and not is_realtime_serving(generator.server_args)
        return scheduler
    except BaseException:
        if scheduler is None:
            generator.shutdown()
        else:
            # Stopping marks it closing, so its worker watch takes the exit
            # that follows as a stop.
            scheduler.stop()
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
