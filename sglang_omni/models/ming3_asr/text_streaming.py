# SPDX-License-Identifier: Apache-2.0
"""Stream stable Ming ASR text and reconcile it with the final transcript."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

from sglang_omni.models.ming3_asr.preprocessing import extract_transcript
from sglang_omni.scheduling.exact_text_streaming import ExactTextStreamState
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import RequestOutput


@dataclass(kw_only=True)
class Ming3ASRRequestData(SGLangARRequestData):
    language: str | None = None
    text_stream_state: ExactTextStreamState | None = None
    is_stream_finished: bool = False


class Ming3ASRStreamOutputBuilder:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, *, interval_seconds: float
    ) -> None:
        if not math.isfinite(interval_seconds) or interval_seconds < 0:
            raise ValueError("Ming ASR stream interval must be finite and nonnegative")
        else:
            pass
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.interval_seconds: float = interval_seconds

    def __call__(
        self,
        request_id: str,
        request_data: Ming3ASRRequestData,
        request_output: RequestOutput,
    ) -> list[OutgoingMessage]:
        request = request_data.req
        payload = request_data.stage_payload
        if request is None or payload is None:
            return []
        elif request.inflight_middle_chunks > 0 or request_data.is_stream_finished:
            return []
        elif not payload.request.params.get("stream", False):
            return []
        elif request_output.data is None:
            return []
        else:
            pass
        if request_data.text_stream_state is None:
            language = request_data.language

            def decode(token_ids: list[int]) -> str:
                generated_text = self.tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                return extract_transcript(
                    generated_text, language=language, is_final=False
                )

            request_data.text_stream_state = ExactTextStreamState(
                decoder=decode, interval_seconds=self.interval_seconds
            )
        else:
            pass
        token_id = int(request_output.data)
        delta = request_data.text_stream_state.step(
            token_id, terminal=False, now_seconds=time.perf_counter()
        )
        return self.messages(request_id, delta, token_id)

    def flush(
        self, request_id: str, request_data: Ming3ASRRequestData
    ) -> list[OutgoingMessage]:
        payload = request_data.stage_payload
        if payload is None or request_data.is_stream_finished:
            return []
        elif not payload.request.params.get("stream", False):
            return []
        else:
            pass
        generated_text = self.tokenizer.decode(
            request_data.output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        transcript = extract_transcript(generated_text, language=request_data.language)
        state = request_data.text_stream_state
        emitted_text = state.emitted_text if state is not None else ""
        if not transcript.startswith(emitted_text):
            raise RuntimeError("Final Ming ASR transcript changed an emitted prefix")
        else:
            pass
        request_data.is_stream_finished = True
        return self.messages(request_id, transcript[len(emitted_text) :], None)

    @staticmethod
    def messages(
        request_id: str, delta: str, token_id: int | None
    ) -> list[OutgoingMessage]:
        if not delta:
            return []
        else:
            return [
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    target=None,
                    data={"text": delta, "modality": "text", "stage_name": "thinker"},
                    metadata={"modality": "text", "token_id": token_id},
                )
            ]
