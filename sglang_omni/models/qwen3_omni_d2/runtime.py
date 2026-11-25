# SPDX-License-Identifier: Apache-2.0
"""The resident Qwen3-Omni D2 model and its captured graphs, serving one conversation at a time."""

import logging

import torch

from sglang_omni.admission import ContextExhaustedError
from sglang_omni.models.qwen3_omni_d2.components.mandarin import (
    MANDARIN_PROMPT,
    MandarinOpening,
)
from sglang_omni.models.qwen3_omni_d2.components.model import (
    load_release_model,
    resolve_release,
)
from sglang_omni.models.qwen3_omni_d2.components.thinker import THINKER_TOKENS_PER_FRAME
from sglang_omni.models.qwen3_omni_d2.config import (
    INPUT_SAMPLE_RATE_HZ,
    NATIVE_FRAME_MS,
    Qwen3OmniD2Language,
)
from sglang_omni.models.qwen3_omni_d2.session import (
    Qwen3OmniD2Frame,
    Qwen3OmniD2Session,
)
from sglang_omni.proto.streams import PCM16_BYTES_PER_SAMPLE
from sglang_omni.utils.device import device_guard

logger = logging.getLogger(__name__)

CONVERSATION_PROMPT = (
    "You are a helpful spoken conversational assistant. "
    "Respond naturally when the user finishes speaking."
)


class Qwen3OmniD2Runtime:
    """Load the release once; each conversation reuses the same fixed caches and graphs."""

    def __init__(
        self,
        model_path: str,
        device: str,
        kv_budget: int,
        seed: int,
        language: Qwen3OmniD2Language,
    ) -> None:
        self.device = torch.device(device)
        with torch.no_grad(), device_guard(self.device):
            model = load_release_model(resolve_release(model_path), self.device)
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
            self.session = Qwen3OmniD2Session(
                model, capacity_frames, prompt_token_ids, seed, opening
            )
        self.interval_frames = model.interval_frames
        logger.info(
            f"Qwen3-Omni D2 ready: {model.interval_frames * NATIVE_FRAME_MS} ms interval, "
            f"{self.max_frames} frames per conversation, language={language}"
        )

    def warm_up(self) -> None:
        """Replay silence until both audio streams reach their captured graphs.

        Each startup interval attends over a new history length, and the first
        use of every length is several times slower than later ones.
        """
        startup_intervals = (
            self.session.audio_encoders[0].past_token_limit // self.interval_frames + 2
        )
        silence = bytes(
            self.interval_frames
            * NATIVE_FRAME_MS
            * INPUT_SAMPLE_RATE_HZ
            // 1000
            * PCM16_BYTES_PER_SAMPLE
        )
        self.open_session()
        for _ in range(min(startup_intervals, self.max_frames // self.interval_frames)):
            self.step(silence)

    def open_session(self) -> None:
        with torch.no_grad(), device_guard(self.device):
            self.session.reset()

    def step(self, pcm16: bytes) -> list[Qwen3OmniD2Frame]:
        """Advance one interval; the fixed Thinker cache bounds the conversation length."""
        if self.session.frames + self.interval_frames > self.max_frames:
            raise ContextExhaustedError(
                f"{ContextExhaustedError.CODE}: D2 session reached its fixed context limit"
            )
        else:
            with torch.no_grad(), device_guard(self.device):
                return self.session.step(pcm16)
