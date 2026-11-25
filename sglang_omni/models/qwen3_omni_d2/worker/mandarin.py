# SPDX-License-Identifier: Apache-2.0
"""Mandarin response mode for the isolated D2 inference environment."""

from copy import copy

import torch
from d2.prompts import encode_system_prompt
from d2_qwen.inference.runtime import Runtime
from d2_qwen.inference.session import D2LiveSession
from mandarin_tokens import MandarinOpening
from pydantic import JsonValue

MANDARIN_PROMPT = "你是一位普通话语音助手。即使问题很短，也用有主语和谓语的完整中文句子，让听众单凭声音就能听懂。在回答中自然提到问题的主题，不只说孤立的一个字、字母或数字。说一次，答完就安静等待。"


class MandarinRuntime(Runtime):
    def metadata(self) -> dict[str, JsonValue]:
        return {**super().metadata(), "system_prompt": MANDARIN_PROMPT}

    def _load(self) -> None:
        super()._load()
        tokenizer = self.model.processor.tokenizer
        self.prompt = encode_system_prompt(tokenizer, MANDARIN_PROMPT)
        self.max_context_frames = (
            (3 * self.capacity_frames - len(self.prompt)) // (3 * self.k)
        ) * self.k
        if self.max_context_frames < self.k:
            raise ValueError(
                "D2 KV budget cannot hold the Mandarin prompt and one interval"
            )
        else:
            self.opening = MandarinOpening(
                tokenizer.backend_tokenizer,
                set(tokenizer.all_special_ids)
                | {self.model.pad_id, self.model.response_id, self.model.interrupt_id},
                self.model.response_id,
                self.device,
            )

    def _create_session(self) -> D2LiveSession:
        session = super()._create_session()
        generated_text = session.macro_session.generated_text
        model = self.model

        def mask_text_logits(
            logits: torch.Tensor,
            *,
            speech_open: torch.Tensor,
            text_done: torch.Tensor,
            previous_text_id: torch.Tensor,
        ) -> torch.Tensor:
            masked = model.mask_text_logits(
                logits,
                speech_open=speech_open,
                text_done=text_done,
                previous_text_id=previous_text_id,
            )
            return self.opening.mask(masked, generated_text)

        # Note (cuzimi): Both Thinker graph replay and eager decoding call the model mask before the Talker consumes the selected token. A shallow copy shares weights and keeps this session's history out of the resident model.
        session_model = copy(model)
        session_model.mask_text_logits = mask_text_logits
        session.macro_session.model = session_model
        return session
