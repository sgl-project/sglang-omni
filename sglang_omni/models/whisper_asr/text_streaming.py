# SPDX-License-Identifier: Apache-2.0
"""Emit Whisper text deltas without changing the final decoded transcript."""

import time

from transformers import WhisperTokenizer

from sglang_omni.models.whisper_asr.request_builders import WhisperASRRequestData
from sglang_omni.scheduling.exact_text_streaming import ExactTextStreamState
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.types import RequestOutput


class ExactWhisperStreamOutputBuilder:
    def __init__(
        self, tokenizer: WhisperTokenizer, interval_seconds: float = 0.05
    ) -> None:
        if interval_seconds < 0:
            raise ValueError("Whisper stream interval must be nonnegative")
        else:
            pass
        self.tokenizer: WhisperTokenizer = tokenizer
        self.interval_seconds: float = interval_seconds

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def __call__(
        self,
        request_id: str,
        request_data: WhisperASRRequestData,
        request_output: RequestOutput,
    ) -> list[OutgoingMessage]:
        request = request_data.req
        payload = request_data.stage_payload
        if request is None or payload is None:
            return []
        elif request.inflight_middle_chunks > 0:
            return []
        elif not (payload.request.params or {}).get("stream", False):
            return []
        elif request_data.detect_language or request_data.segment_timestamps:
            return []
        else:
            pass
        if request_data.exact_text_stream_state is None:
            request_data.exact_text_stream_state = ExactTextStreamState(
                decoder=self.decode, interval_seconds=self.interval_seconds
            )
        else:
            pass
        token_id = None if request_output.data is None else int(request_output.data)
        delta = request_data.exact_text_stream_state.step(
            token_id,
            terminal=request_output.finished or request.finished(),
            now_seconds=time.perf_counter(),
        )
        if not delta:
            return []
        else:
            return [
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    target=None,
                    data={"text": delta, "modality": "text", "stage_name": "asr"},
                    metadata={"modality": "text", "token_id": token_id},
                )
            ]

    def flush(
        self, request_id: str, request_data: WhisperASRRequestData
    ) -> list[OutgoingMessage]:
        return self(
            request_id,
            request_data,
            RequestOutput(request_id=request_id, data=None, finished=True),
        )
