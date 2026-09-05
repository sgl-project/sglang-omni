# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o preprocessing: chat template + media feature extraction.

Multimodal requests run through the checkpoint's remote-code
``MiniCPMOProcessor`` (slice_mode image slicing, whisper mel extraction), which
also returns the ``image_bound`` / ``audio_bounds`` index ranges that mark the
``<unk>`` placeholder runs inside ``input_ids``. Those bounds drive embedding
injection in the thinker stage.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor, AutoTokenizer

from sglang_omni.models.minicpm_o.components.audio_chunking import (
    drop_zero_token_chunks,
    trim_zero_token_tail,
)
from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.preprocessing.audio import (
    compute_audio_cache_key,
    ensure_audio_list_async,
)
from sglang_omni.preprocessing.image import (
    compute_image_cache_key,
    ensure_image_list_async,
)
from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)

IMAGE_PLACEHOLDER = "<image>./</image>"
AUDIO_PLACEHOLDER = "<audio>./</audio>"


def _resolve_local_model_dir(model_path: str) -> str:
    if Path(model_path).exists():
        return model_path
    return str(resolve_model_path(model_path, local_files_only=False))


def _first_batch_item(value: Any) -> Any:
    """Unwrap the batch dimension of a processor output (batch size is 1)."""
    if isinstance(value, list):
        return value[0] if value else None
    if isinstance(value, torch.Tensor):
        return value[0]
    return value


class MiniCPMOPreprocessor:
    def __init__(
        self,
        model_path: str,
        *,
        speech_enabled: bool = False,
    ):
        local_dir = _resolve_local_model_dir(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_dir, trust_remote_code=True
        )
        # Lazy: the remote-code processor pulls in whisper feature extraction;
        # text-only deployments never need it.
        self._local_dir = local_dir
        self._processor = None
        # Speech pipelines render the tts chat template for audio-output
        # requests so the thinker emits a <|tts_bos|>...<|tts_eos|> span for
        # the talker; the remote code's chat() does the same via
        # use_tts_template before generating speech.
        self._speech_enabled = speech_enabled

    def _use_tts_template(self, payload: StagePayload) -> bool:
        from sglang_omni.models.minicpm_o.request_builders import (
            should_generate_audio_output,
        )

        return self._speech_enabled and should_generate_audio_output(payload)

    @property
    def processor(self):
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                self._local_dir, trust_remote_code=True
            )
        return self._processor

    def _audio_pool_step(self) -> int:
        """The processor's placeholder math and the encoder's pooling share
        ``audio_pool_step``; read it from the processor so the two stay
        consistent (checkpoint default 5)."""
        return int(getattr(self.processor, "pool_step", 5))

    async def __call__(self, payload: StagePayload) -> StagePayload:
        # Event names mirror qwen3_omni's preprocessor so the profiler's stage
        # breakdown attributes CPU preprocessing time for both models.
        _emit_event(
            request_id=payload.request_id,
            stage=None,
            event_name="preprocess_start",
        )
        try:
            return await self._call_impl(payload)
        finally:
            _emit_event(
                request_id=payload.request_id,
                stage=None,
                event_name="preprocess_end",
            )

    async def _call_impl(self, payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs
        raw_images = None
        raw_audios = None
        if isinstance(inputs, dict):
            messages = inputs.get("messages", [])
            raw_images = inputs.get("images")
            raw_audios = inputs.get("audio") or inputs.get("audios")
        else:
            messages = inputs

        if raw_images or raw_audios:
            return await self._preprocess_multimodal(
                payload, messages, raw_images=raw_images, raw_audios=raw_audios
            )

        if (
            isinstance(messages, list)
            and messages
            and all(isinstance(token, int) for token in messages)
        ):
            # Pre-tokenized prompt ids (rollout path): use them verbatim so
            # serving tokens match the caller's exactly.
            prompt_text = ""
            input_ids = torch.tensor(messages, dtype=torch.long)
        else:
            prompt_text = self._render_chat_template(
                messages, use_tts_template=self._use_tts_template(payload)
            )
            encoded = self.tokenizer(prompt_text, return_tensors="pt")
            input_ids = encoded["input_ids"][0].to(dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        state = MiniCPMOPipelineState(
            prompt={
                "prompt_text": prompt_text,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            stream_state={"token_ids": [], "text": ""},
        )
        payload.data = state.to_dict()
        # Downstream projections consume the canonical state.
        payload.request.inputs = None
        return payload

    def _render_chat_template(
        self, messages: Any, *, use_tts_template: bool = False
    ) -> str:
        if isinstance(messages, str):
            return messages
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            use_tts_template=use_tts_template,
        )

    def _messages_with_media_placeholders(
        self,
        messages: list[dict[str, Any]],
        *,
        num_images: int,
        num_audios: int,
    ) -> list[dict[str, Any]]:
        """Inject media placeholders into the last user message.

        Mirrors the remote code's ``chat()``: placeholder markers and the text
        are joined with newlines, placeholders first.
        """
        result: list[dict[str, Any]] = []
        for i, msg in enumerate(messages):
            if i == len(messages) - 1 and msg.get("role", "user") == "user":
                parts = (
                    [IMAGE_PLACEHOLDER] * num_images
                    + [AUDIO_PLACEHOLDER] * num_audios
                    + [str(msg.get("content", ""))]
                )
                result.append({**msg, "content": "\n".join(parts)})
            else:
                result.append(msg)
        return result

    async def _preprocess_multimodal(
        self,
        payload: StagePayload,
        messages: Any,
        *,
        raw_images: Any,
        raw_audios: Any,
    ) -> StagePayload:
        image_cache_key = compute_image_cache_key(raw_images)
        audio_cache_key = compute_audio_cache_key(raw_audios)

        # Media decode is I/O + CPU bound; fetch both modalities concurrently
        # (qwen3_omni preprocessor parity).
        images, audios = await asyncio.gather(
            ensure_image_list_async(raw_images),
            ensure_audio_list_async(raw_audios, target_sr=16000),
        )
        # A clip a few samples past a 30 s boundary would get a whole extra
        # whisper window that yields zero tokens (Daily-Omni's 30 s and 60 s
        # clips all do); trimming it changes neither the placeholder count nor
        # the encoder rows, and saves the STFT here and a full window there.
        pool_step = self._audio_pool_step()
        audios = [
            trim_zero_token_tail(np.asarray(audio), pool_step=pool_step)
            for audio in audios
        ]

        if isinstance(messages, list) and not (
            messages and all(isinstance(token, int) for token in messages)
        ):
            messages = self._messages_with_media_placeholders(
                messages, num_images=len(images), num_audios=len(audios)
            )
        prompt_text = self._render_chat_template(
            messages,
            use_tts_template=bool(audios) or self._use_tts_template(payload),
        )

        processed = self.processor(
            prompt_text,
            images=[images] if images else None,
            audios=[audios] if audios else None,
            return_tensors="pt",
        )

        input_ids = processed["input_ids"][0].to(dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        mm_inputs: dict[str, Any] = {}
        encoder_inputs: dict[str, dict[str, Any]] = {}
        if images:
            image_bound = _first_batch_item(processed["image_bound"])
            # pixel_values is nested [batch][image] with one tensor per slice;
            # flatten to a slice list — bound ranges appear in the same order.
            pixel_values = [
                slice_tensor
                for per_image in processed["pixel_values"][0]
                for slice_tensor in (
                    per_image if isinstance(per_image, list) else [per_image]
                )
            ]
            tgt_sizes = _first_batch_item(processed["tgt_sizes"])
            mm_inputs["image"] = {"bounds": image_bound, "cache_key": image_cache_key}
            encoder_inputs["image_encoder"] = {
                "pixel_values": pixel_values,
                "tgt_sizes": tgt_sizes,
                "cache_key": image_cache_key,
            }
        if audios:
            audio_bounds = _first_batch_item(processed["audio_bounds"])
            audio_feature_lens = _first_batch_item(processed["audio_feature_lens"])
            # Belt and braces for paths the trim above cannot reach (e.g. a
            # clip shorter than one window whose tail still rounds to zero).
            audio_features, audio_feature_lens = drop_zero_token_chunks(
                processed["audio_features"], audio_feature_lens, pool_step=pool_step
            )
            mm_inputs["audio"] = {"bounds": audio_bounds, "cache_key": audio_cache_key}
            if int(audio_feature_lens.numel()) > 0:
                encoder_inputs["audio_encoder"] = {
                    "audio_features": audio_features,
                    "audio_feature_lens": audio_feature_lens,
                    "cache_key": audio_cache_key,
                }

        state = MiniCPMOPipelineState(
            prompt={
                "prompt_text": prompt_text,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            mm_inputs=mm_inputs,
            encoder_inputs=encoder_inputs,
            stream_state={"token_ids": [], "text": ""},
        )
        payload.data = state.to_dict()
        payload.request.inputs = None
        for key in ("audios", "images", "videos"):
            payload.request.metadata.pop(key, None)
        return payload
