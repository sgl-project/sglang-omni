# SPDX-License-Identifier: Apache-2.0
"""Decode Ming function calls and relay completed calls to streaming chat clients."""

import json
from dataclasses import dataclass
from functools import cached_property

import torch
from pydantic import JsonValue
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.ling3_detector import Ling3Detector
from transformers import PreTrainedTokenizerBase

from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.scheduling.types import RequestOutput


@dataclass(kw_only=True)
class MingAudioRequest(SGLangARRequestData):
    pending_call_tokens: list[int] | None = None
    streamed_calls: int = 0


def parse_tool_reply(
    text: str, tools: list[Tool], request_id: str, first_index: int
) -> tuple[str, list[dict[str, JsonValue]]]:
    parsed = Ling3Detector().detect_and_parse(text, tools)
    calls = [
        {
            "id": f"call_{request_id}_{index}",
            "type": "function",
            "function": {"name": call.name, "arguments": call.parameters},
        }
        for index, call in enumerate(parsed.calls, start=first_index)
    ]
    return (parsed.normal_text if calls else text), calls


class MingAudioDecodeScheduler(StreamingSimpleScheduler):
    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        assert isinstance(item.data, torch.Tensor)
        calls = json.loads(bytes(item.data.tolist()))
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data={"tool_calls": calls},
                metadata={"modality": "text"},
            )
        ]


class MingToolCallStream:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    @cached_property
    def start_call_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids("<tool_call>")

    @cached_property
    def end_call_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids("</tool_call>")

    def __call__(
        self, request_id: str, request: MingAudioRequest, output: RequestOutput
    ) -> list[OutgoingMessage]:
        payload = request.stage_payload
        if (
            not payload.request.metadata.get("tools")
            or not payload.request.params.get("stream")
            or output.data is None
            or request.req.inflight_middle_chunks
        ):
            return []
        else:
            token = int(output.data)
        if token == self.start_call_id:
            request.pending_call_tokens = [token]
        elif request.pending_call_tokens is None:
            return []
        else:
            request.pending_call_tokens.append(token)
        if token != self.end_call_id:
            return []
        else:
            text = self.tokenizer.decode(
                request.pending_call_tokens, skip_special_tokens=False
            )
            request.pending_call_tokens = None
            tools = [
                Tool.model_validate(tool) for tool in payload.request.metadata["tools"]
            ]
            _, calls = parse_tool_reply(text, tools, request_id, request.streamed_calls)
            request.streamed_calls += len(calls)
            if calls:
                encoded_calls = torch.tensor(
                    list(json.dumps(calls, ensure_ascii=False).encode()),
                    dtype=torch.uint8,
                )
                return [
                    OutgoingMessage(
                        request_id=request_id,
                        type="stream",
                        target="decode",
                        data=encoded_calls,
                    )
                ]
            else:
                return []
