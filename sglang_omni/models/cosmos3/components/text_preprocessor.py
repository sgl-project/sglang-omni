# SPDX-License-Identifier: Apache-2.0
"""Text-only preprocessing for Cosmos3-Nano."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from sglang_omni.models.cosmos3.payload_types import Cosmos3PipelineState
from sglang_omni.preprocessing.text import ensure_chat_template
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)

DEFAULT_MAX_NEW_TOKENS = 2048
_MEDIA_INPUT_KEYS = ("images", "videos", "audios")


def load_cosmos3_tokenizer(model_path: str) -> Any:
    """Load tokenizer assets without hydrating the checkpoint weights."""

    local_only = Path(model_path).exists()
    if not local_only:
        try:
            return AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
        except (OSError, ValueError):
            pass
    return AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=local_only,
    )


def _normalize_text_messages(inputs: Any) -> list[dict[str, str]]:
    if isinstance(inputs, dict):
        populated_media = [key for key in _MEDIA_INPUT_KEYS if inputs.get(key)]
        if populated_media:
            raise ValueError(
                "Cosmos3 text preprocessing does not support media inputs yet: "
                + ", ".join(populated_media)
            )
        if "messages" not in inputs:
            raise ValueError("Cosmos3 text preprocessing expects a messages field")
        inputs = inputs["messages"]

    if not isinstance(inputs, list) or not inputs:
        raise ValueError(
            "Cosmos3 text preprocessing expects a non-empty list of chat messages"
        )

    messages: list[dict[str, str]] = []
    for index, message in enumerate(inputs):
        if not isinstance(message, dict):
            raise TypeError(f"Message {index} must be a dict with role/content")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role:
            raise ValueError(f"Message {index} must have a non-empty string role")
        if not isinstance(content, str):
            raise TypeError(
                f"Message {index} content must be a string; multimodal content "
                "will be added in a later Cosmos3 stage"
            )
        messages.append({"role": role, "content": content})
    return messages


def _reject_media_inputs(container: Any) -> None:
    if not isinstance(container, dict):
        return
    populated_media = [key for key in _MEDIA_INPUT_KEYS if container.get(key)]
    if populated_media:
        raise ValueError(
            "Cosmos3 text preprocessing does not support media inputs yet: "
            + ", ".join(populated_media)
        )


def _flatten_single_batch(value: Any, *, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 2:
        if tensor.shape[0] != 1:
            raise ValueError(f"Tokenizer returned batched {name}; expected one prompt")
        tensor = tensor[0]
    if tensor.ndim != 1:
        raise ValueError(
            f"Tokenizer returned invalid {name} shape {tuple(tensor.shape)}"
        )
    return tensor.to(dtype=torch.long)


def validate_prompt_seq_len(
    input_ids: torch.Tensor,
    *,
    max_seq_len: int | None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    request_id: str | None = None,
) -> None:
    """Apply the same prompt budget contract as the existing AR pipelines."""

    if max_seq_len is None:
        return
    prompt_len = int(input_ids.numel())
    if prompt_len >= max_seq_len:
        logger.info(
            "rejecting request %s: prompt %d tokens >= max_seq_len %d",
            request_id,
            prompt_len,
            max_seq_len,
        )
        raise ValueError(
            f"The input ({prompt_len} tokens) is longer than the model's "
            f"context length ({max_seq_len} tokens)."
        )

    total_tokens = prompt_len + int(max_new_tokens)
    if total_tokens >= max_seq_len:
        logger.info(
            "rejecting request %s: prompt %d + max_new_tokens %d >= max_seq_len %d",
            request_id,
            prompt_len,
            int(max_new_tokens),
            max_seq_len,
        )
        raise ValueError(
            "Requested token count exceeds the model's maximum context length "
            f"of {max_seq_len} tokens. You requested {prompt_len} prompt tokens "
            f"and {int(max_new_tokens)} completion tokens."
        )


class Cosmos3TextPreprocessor:
    """Build a tokenizer-only Cosmos3 pipeline state on CPU."""

    def __init__(
        self,
        model_path: str,
        max_seq_len: int | None = None,
        *,
        tokenizer: Any | None = None,
    ) -> None:
        self.model_path = model_path
        self.max_seq_len = max_seq_len
        self.tokenizer = (
            tokenizer if tokenizer is not None else load_cosmos3_tokenizer(model_path)
        )
        ensure_chat_template(self.tokenizer, model_path=model_path)
        if not getattr(self.tokenizer, "chat_template", None):
            raise ValueError(
                f"Tokenizer for {model_path!r} does not define a chat template"
            )

    def _tokenize_messages(self, inputs: Any) -> tuple[torch.Tensor, torch.Tensor, str]:
        messages = _normalize_text_messages(inputs)
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = _flatten_single_batch(encoded["input_ids"], name="input_ids")
        raw_attention_mask = encoded.get("attention_mask")
        attention_mask = (
            torch.ones_like(input_ids)
            if raw_attention_mask is None
            else _flatten_single_batch(raw_attention_mask, name="attention_mask")
        )
        if attention_mask.shape != input_ids.shape:
            raise ValueError("Tokenizer returned mismatched input_ids/attention_mask")
        return input_ids, attention_mask, prompt_text

    @staticmethod
    def _use_pretokenized_inputs(inputs: Any) -> bool:
        return (
            isinstance(inputs, list)
            and bool(inputs)
            and all(isinstance(token_id, int) for token_id in inputs)
        )

    def __call__(self, payload: StagePayload) -> StagePayload:
        # The OpenAI adapter places top-level media fields in metadata rather
        # than request.inputs. Keep the first Cosmos3 slice strictly text-only
        # regardless of which entry point constructed the OmniRequest.
        _reject_media_inputs(payload.request.metadata)
        inputs = payload.request.inputs
        if self._use_pretokenized_inputs(inputs):
            input_ids = torch.tensor(inputs, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            prompt_text = ""
        else:
            input_ids, attention_mask, prompt_text = self._tokenize_messages(inputs)

        max_new_tokens = payload.request.params.get(
            "max_new_tokens", DEFAULT_MAX_NEW_TOKENS
        )
        if max_new_tokens is None:
            max_new_tokens = DEFAULT_MAX_NEW_TOKENS
        validate_prompt_seq_len(
            input_ids,
            max_seq_len=self.max_seq_len,
            max_new_tokens=int(max_new_tokens),
            request_id=payload.request_id,
        )

        state = Cosmos3PipelineState(
            prompt={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "prompt_text": prompt_text,
            },
            stream_state={"token_ids": [], "text": ""},
        )
        payload.data = state.to_dict()
        payload.request.inputs = None
        return payload


__all__ = [
    "DEFAULT_MAX_NEW_TOKENS",
    "Cosmos3TextPreprocessor",
    "load_cosmos3_tokenizer",
    "validate_prompt_seq_len",
]
