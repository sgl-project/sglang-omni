# SPDX-License-Identifier: Apache-2.0
"""The resident D2 model and its captured graphs, serving one conversation at a time."""

import logging

import torch

from sglang_omni.admission import ContextExhaustedError
from sglang_omni.models.qwen3_omni_d2.config import NATIVE_FRAME_MS, D2Language
from sglang_omni.models.qwen3_omni_d2.mandarin import MANDARIN_PROMPT, MandarinOpening
from sglang_omni.models.qwen3_omni_d2.release import load_d2_model, resolve_release
from sglang_omni.models.qwen3_omni_d2.session import D2Frame, D2Session
from sglang_omni.utils.device import device_guard

logger = logging.getLogger(__name__)

CONVERSATION_PROMPT = (
    "You are a helpful spoken conversational assistant. "
    "Respond naturally when the user finishes speaking."
)
# Environment audio, delayed self audio and text each take one Thinker token per frame.
THINKER_TOKENS_PER_FRAME = 3


class D2Runtime:
    """Load the release once; each conversation reuses the same fixed caches and graphs."""

    def __init__(
        self,
        model_path: str,
        device: str,
        kv_budget: int,
        seed: int,
        language: D2Language,
    ) -> None:
        self.device = torch.device(device)
        with torch.no_grad(), device_guard(self.device):
            model = load_d2_model(resolve_release(model_path), self.device)
            prompt = MANDARIN_PROMPT if language == "zh" else CONVERSATION_PROMPT
            prompt_token_ids = model.tokenizer.encode(
                f"<|im_start|>system\n{prompt}<|im_end|>\n", add_special_tokens=False
            )
            interval_tokens = THINKER_TOKENS_PER_FRAME * model.interval_frames
            capacity_frames = kv_budget // interval_tokens * model.interval_frames
            self.max_frames = (
                (THINKER_TOKENS_PER_FRAME * capacity_frames - len(prompt_token_ids))
                // interval_tokens
                * model.interval_frames
            )
            if self.max_frames < model.interval_frames:
                raise ValueError(
                    "D2 KV budget cannot hold the system prompt and one interval"
                )
            elif language == "zh":
                tokenizer = model.tokenizer
                opening = MandarinOpening(
                    tokenizer.backend_tokenizer,
                    model.special_token_ids,
                    model.response_token_id,
                    self.device,
                )
            else:
                opening = None
            self.session = D2Session(
                model, capacity_frames, prompt_token_ids, seed, opening
            )
        self.interval_frames = model.interval_frames
        logger.info(
            f"Qwen3-Omni D2 ready: {model.interval_frames * NATIVE_FRAME_MS} ms interval, "
            f"{self.max_frames} frames per conversation, language={language}"
        )

    def open_session(self) -> None:
        with torch.no_grad(), device_guard(self.device):
            self.session.reset()

    def step(self, pcm16: bytes) -> list[D2Frame]:
        """Advance one interval; the fixed Thinker cache bounds the conversation length."""
        if self.session.frames + self.interval_frames > self.max_frames:
            raise ContextExhaustedError(
                f"{ContextExhaustedError.CODE}: D2 session reached its fixed context limit"
            )
        else:
            with torch.no_grad(), device_guard(self.device):
                return self.session.step(pcm16)
