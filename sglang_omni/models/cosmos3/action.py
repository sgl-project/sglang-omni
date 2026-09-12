# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import math
import os
import shutil
import tempfile
from dataclasses import fields
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from sglang.multimodal_gen.configs.sample.cosmos3 import Cosmos3SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.action.api import (
    _multipart_action_payload,
    _prefer_numpy_output,
    _response_format,
    _wants_msgpack,
)
from sglang.multimodal_gen.runtime.entrypoints.action.cosmos3 import (
    build_cosmos3_action_sampling_params,
)
from sglang.multimodal_gen.runtime.entrypoints.action.protocol import (
    _action_request_to_observation,
    action_generation_response,
    action_raw_response,
    pack_msgpack,
    unpack_msgpack,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.srt.utils.json_response import orjson_response

from sglang_omni.pipeline.mp_runner import _finish_despite_cancellation

logger = logging.getLogger(__name__)


class _ActionSamplingParams(Cosmos3SamplingParams):
    def __init__(self, **kwargs):
        self._action_fields = set(kwargs)
        self._explicit_fields = set(kwargs)
        super().__init__(**kwargs)


def build_action_sampling_params(payload: dict[str, Any], server_args):
    if not isinstance(payload, dict):
        raise ValueError("Action requests must be a JSON object")
    observation = _action_request_to_observation(payload)
    options = {**observation, **dict(payload.get("parameters") or {})}
    if float(options.get("sound_duration") or 0) != 0 or any(
        str(options.get(name, "false")).lower() in ("true", "1")
        for name in ("generate_sound", "enable_sound")
    ):
        raise ValueError("Cosmos3 Edge does not support sound generation")
    sampling = build_cosmos3_action_sampling_params(
        payload, observation, server_args, _ActionSamplingParams
    )
    interval = options.get("guidance_interval")
    if interval is not None:
        if isinstance(interval, str):
            interval = json.loads(interval)
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError("guidance_interval must contain two timesteps")
        interval = tuple(float(value) for value in interval)
        if (
            not all(math.isfinite(value) for value in interval)
            or interval[0] > interval[1]
        ):
            raise ValueError("guidance_interval must contain finite, ordered timesteps")
        sampling.guidance_interval = interval
        sampling._action_fields.add("guidance_interval")
    sampling._explicit_fields.update(options)
    sampling._validate()
    return sampling


def action_sampling_kwargs(params: dict[str, Any], server_args) -> dict[str, Any]:
    sampling = build_action_sampling_params(params, server_args)
    kwargs = {
        field.name: getattr(sampling, field.name)
        for field in fields(sampling)
        if field.init
    }
    kwargs.update(
        {
            name: value
            for name, value in params.items()
            if name not in sampling._action_fields and name != "action_horizon"
        }
    )
    return kwargs


router = APIRouter()


async def _forward_action(request: Request, native_request):
    async def forward():
        request.state.action_input_settled = False
        output = await async_scheduler_client.forward(native_request)
        request.state.action_input_settled = True
        return output

    failure = getattr(request.app.state, "scheduler_failure", None)
    stop_runtime = getattr(request.app.state, "stop_runtime", None)
    if failure is not None and failure.done():
        failure.result()
        raise RuntimeError("Generation runtime stopped")
    if failure is None or stop_runtime is None:
        return await forward()

    task = asyncio.create_task(forward())
    done, _ = await asyncio.wait((task, failure), return_when=asyncio.FIRST_COMPLETED)
    if task in done:
        return task.result()

    try:
        await stop_runtime()
    except BaseException:
        await _finish_despite_cancellation(task)
        raise
    request.state.action_input_settled = True
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    failure.result()
    raise RuntimeError("Generation runtime stopped")


@router.post("/v1/actions/generations", include_in_schema=False)
async def create_action_generation(request: Request):
    server_args = request.app.state.server_args
    content_type = request.headers.get("content-type", "").lower()
    multipart = "multipart/form-data" in content_type
    uploads_dir = getattr(server_args, "input_save_path", None)
    temporary = multipart and uploads_dir is None
    if temporary:
        uploads_dir = tempfile.mkdtemp(prefix="sglang_")
    request.state.action_input_settled = True
    try:
        if multipart:
            if not temporary:
                os.makedirs(uploads_dir, exist_ok=True)
            payload = await _multipart_action_payload(request, uploads_dir)
        elif "msgpack" in content_type:
            payload = unpack_msgpack(await request.body())
        else:
            payload = await request.json()
        sampling = build_action_sampling_params(payload, server_args)
        wants_msgpack = _wants_msgpack(request)
        if wants_msgpack:
            _prefer_numpy_output(payload)
        response_format = _response_format(payload)
        native_request = prepare_request(server_args, sampling)
        task = asyncio.create_task(_forward_action(request, native_request))
        await _finish_despite_cancellation(task)
        output = task.result()
        if getattr(output, "error", None):
            raise RuntimeError(output.error)
        if not output.output:
            raise RuntimeError("action policy returned no output")
        response = (
            action_raw_response(output.output[0], preserve_numpy=wants_msgpack)
            if response_format == "raw"
            else action_generation_response(
                output.output[0], server_args, preserve_numpy=wants_msgpack
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temporary:
            if request.state.action_input_settled:
                shutil.rmtree(uploads_dir, ignore_errors=True)
            else:
                logger.error(
                    "Action input not settled; retaining uploads at %s", uploads_dir
                )
    if wants_msgpack:
        return Response(
            content=pack_msgpack(response), media_type="application/msgpack"
        )
    return orjson_response(response)
