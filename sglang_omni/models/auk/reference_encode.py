# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Tencent. All rights reserved.
# Derived from Tencent-Hunyuan/AuK; see LICENSE for the MIT permission notice.
"""AuK conditioning encoder: frozen Qwen2.5-Omni-3B Thinker."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import NamedTuple, TypedDict

import torch

from sglang_omni.models.auk.constants import NO_PROMPT_AUDIO_MARKER
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import ReplayableGraph

logger = logging.getLogger(__name__)


class ChatMessage(TypedDict):
    role: str
    content: list[dict[str, str | None]]


class CapturedConditioning(NamedTuple):
    graph: ReplayableGraph
    token_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    hidden_states: torch.Tensor


def build_messages(instruction: str, has_reference_audio: bool) -> list[ChatMessage]:
    """Build the single-turn ChatML message list AuK is trained on."""
    text = instruction
    if not has_reference_audio and not text.endswith(NO_PROMPT_AUDIO_MARKER):
        text = text + NO_PROMPT_AUDIO_MARKER
    else:
        pass

    content: list[dict[str, str | None]] = [{"type": "text", "text": text}]
    if has_reference_audio:
        content.append({"type": "audio", "audio": None})
    else:
        pass
    return [{"role": "user", "content": content}]


class AuKConditionEncoder:
    """Frozen Qwen2.5-Omni Thinker used as AuK's instruction/reference encoder."""

    def __init__(
        self,
        model_path: str,
        *,
        weight_dtype: torch.dtype,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        from transformers import (
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        class AuKThinker(Qwen2_5OmniThinkerForConditionalGeneration):
            _keys_to_ignore_on_load_unexpected = [
                *(
                    Qwen2_5OmniThinkerForConditionalGeneration._keys_to_ignore_on_load_unexpected  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
                    or []
                ),
                r"^(talker|token2wav)\.",
            ]

        self.model_path = model_path
        self.device = torch.device(device)
        self.dtype = dtype
        self.text_graphs: dict[tuple[int, torch.dtype | None], CapturedConditioning] = (
            {}
        )

        logger.info(
            "AuK: loading Qwen2.5-Omni Thinker from %s; "
            "checkpoint talker/token2wav branches are unused",
            model_path,
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        model = AuKThinker.from_pretrained(model_path, torch_dtype=dtype)
        model.visual = None
        model.lm_head = torch.nn.Identity()
        model.requires_grad_(False)
        model.eval()
        self.model = model.to(device=self.device, dtype=weight_dtype)

    @property
    def num_hidden_layers(self) -> int:
        return int(self.model.config.text_config.num_hidden_layers)

    @torch.inference_mode()
    def capture_text_graphs(
        self, token_lengths: Sequence[int], *, compute_dtype: torch.dtype
    ) -> None:
        """Capture exact single-request text lengths before serving starts."""
        if any(length < 1 for length in token_lengths):
            raise ValueError("AuK conditioning graph token lengths must be positive")
        else:
            pass
        backend = current_platform.get_device_graph_backend(self.device)
        if not token_lengths or backend is None:
            return
        else:
            pass
        module = torch.get_device_module(self.device)
        stream = module.Stream(device=self.device)
        stream.wait_stream(module.current_stream(self.device))
        pool = module.graph_pool_handle()
        for token_length in sorted(set(token_lengths)):
            token_ids = torch.zeros(
                (1, token_length), device=self.device, dtype=torch.long
            )
            attention_mask = torch.ones_like(token_ids)
            position_ids = (
                torch.arange(token_length, device=self.device)[None, None, :]
                .expand(3, 1, -1)
                .contiguous()
            )

            def encode_text() -> torch.Tensor:
                outputs = self.model(
                    input_ids=token_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_hidden_states=True,
                    use_cache=False,
                )
                return torch.stack(outputs.hidden_states, dim=1)

            stream.wait_stream(module.current_stream(self.device))
            with (
                module.stream(stream),
                torch.autocast(
                    self.device.type,
                    dtype=compute_dtype,
                    enabled=compute_dtype != torch.float32,
                    cache_enabled=False,
                ),
            ):
                for _ in range(3):
                    encode_text()
                stream.synchronize()
                with backend.capture(pool=pool, stream=stream) as graph:
                    hidden_states = encode_text()
            autocast_dtype = compute_dtype if compute_dtype != torch.float32 else None
            self.text_graphs[token_length, autocast_dtype] = CapturedConditioning(
                graph, token_ids, attention_mask, position_ids, hidden_states
            )
        module.current_stream(self.device).wait_stream(stream)
        module.synchronize(self.device)
        logger.info(f"AuK conditioning: captured {len(self.text_graphs)} text lengths")

    @torch.no_grad()
    def encode(self, messages, audio):
        return self.encode_batch([messages], [audio])[0]

    @torch.no_grad()
    def encode_batch(self, messages, audios):
        """Encode padded requests together, returning only each request's valid tokens."""
        formatted = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        kwargs = dict(text=formatted, padding=True, return_tensors="pt")
        references = [audio for audio in audios if audio is not None]
        if references:
            kwargs["audio"] = references
        else:
            pass
        inputs = self.processor(**kwargs)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        autocast_dtype = (
            torch.get_autocast_dtype(self.device.type)
            if torch.is_autocast_enabled(self.device.type)
            else None
        )
        captured = (
            self.text_graphs.get((inputs["input_ids"].shape[1], autocast_dtype))
            if len(messages) == 1
            and not references
            and inputs.keys() == {"input_ids", "attention_mask"}
            else None
        )
        if captured is None:
            outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
            hidden = torch.stack(outputs.hidden_states, dim=1)
        else:
            captured.token_ids.copy_(inputs["input_ids"])
            captured.attention_mask.copy_(inputs["attention_mask"])
            captured.graph.replay()
            hidden = captured.hidden_states
        masks = inputs["attention_mask"].bool()
        return [(item[:, mask], mask[mask]) for item, mask in zip(hidden, masks)]
