# SPDX-License-Identifier: Apache-2.0
"""Minimal OpenAI-compatible chat adapter."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from sglang_omni.gateway import Gateway, GenerateRequest, Message, SamplingParams


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    request_id: str | None = None


def create_app(gateway: Gateway) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        request_id = req.request_id or str(uuid.uuid4())
        response_id = f"chatcmpl-{request_id}"
        created = int(time.time())

        gen_req = _build_generate_request(req)

        if req.stream:

            async def event_stream():
                role_sent = False
                async for chunk in gateway.generate(gen_req, request_id=request_id):
                    delta: dict[str, Any] = {}
                    emit = False
                    if not role_sent:
                        delta["role"] = "assistant"
                        role_sent = True
                        emit = True
                    if chunk.text:
                        delta["content"] = chunk.text
                        emit = True
                    if chunk.finish_reason is not None:
                        emit = True

                    if not emit:
                        continue

                    payload = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model or "",
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": chunk.finish_reason,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        result = None
        async for chunk in gateway.generate(gen_req, request_id=request_id):
            result = chunk
        if result is None:
            raise HTTPException(status_code=500, detail="No response from gateway")

        response = {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": req.model or "",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": result.finish_reason or "stop",
                }
            ],
        }
        if result.usage is not None:
            response["usage"] = result.usage.to_dict()

        return JSONResponse(response)

    return app


def _build_generate_request(req: ChatCompletionRequest) -> GenerateRequest:
    stop: list[str] = []
    if isinstance(req.stop, str):
        stop = [req.stop]
    elif isinstance(req.stop, list):
        stop = req.stop

    sampling = SamplingParams(
        temperature=req.temperature if req.temperature is not None else 1.0,
        top_p=req.top_p if req.top_p is not None else 1.0,
        stop=stop,
    )

    messages = [Message(role=m.role, content=m.content) for m in req.messages]

    return GenerateRequest(
        model=req.model,
        messages=messages,
        sampling=sampling,
        stream=req.stream,
        max_tokens=req.max_tokens,
    )
