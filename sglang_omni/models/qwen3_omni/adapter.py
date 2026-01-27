# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni adapter for the generic omni pipeline glue."""

from __future__ import annotations

from typing import Any, Iterable

import torch

from sglang_omni.executors import FrontendExecutor
from sglang_omni.models.omni_adapter import (
    FrontendOutput,
    OmniAdapter,
    OmniEvent,
    ThinkerOutput,
)
from sglang_omni.models.omni_generic import (
    create_adapter_aggregate_executor,
    create_adapter_decode_executor,
)
from sglang_omni.models.qwen3_omni.frontend import Qwen3OmniFrontend
from sglang_omni.proto import OmniRequest, StagePayload

IMAGE_STAGE = "image_encoder"
AUDIO_STAGE = "audio_encoder"


def _as_tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype) if dtype is not None else value
    try:
        return torch.as_tensor(value, dtype=dtype)
    except Exception:
        return None


def _non_empty(tensor: torch.Tensor | None) -> bool:
    return isinstance(tensor, torch.Tensor) and tensor.numel() > 0


class Qwen3OmniAdapter(OmniAdapter):
    """Adapter that centralizes Qwen3-Omni pipeline policy."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.name = f"qwen3_omni::{model_id}"
        self.modalities = ("text", "image", "audio")
        self._frontend = Qwen3OmniFrontend(model_id=model_id)
        self.tokenizer = self._frontend.tokenizer
        self._eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

    def build_frontend(
        self,
        *,
        request_inputs: Any,
        request_params: dict[str, Any],
    ) -> FrontendOutput:
        request = OmniRequest(inputs=request_inputs, params=request_params)
        payload = StagePayload(request_id="adapter", request=request, data={})
        payload = self._frontend(payload)
        data = payload.data if isinstance(payload.data, dict) else {}

        prompt = data.get("prompt", {})
        mm_inputs = data.get("mm_inputs", {})
        engine_inputs = data.get("engine_inputs", {})
        adapter_state = {
            "engine_inputs": engine_inputs,
            "stream_state": {"token_ids": [], "text": ""},
        }

        frontend_out: FrontendOutput = {
            "prompt": prompt,
            "mm_inputs": mm_inputs,
            "adapter_state": adapter_state,
        }
        return frontend_out

    def build_encoder_inputs(
        self,
        *,
        frontend_out: FrontendOutput,
    ) -> dict[str, dict[str, Any]]:
        adapter_state = frontend_out.get("adapter_state", {})
        engine_inputs = (
            adapter_state.get("engine_inputs", {})
            if isinstance(adapter_state, dict)
            else {}
        )
        if not isinstance(engine_inputs, dict):
            engine_inputs = {}

        image_inputs = engine_inputs.get(IMAGE_STAGE, {})
        audio_inputs = engine_inputs.get(AUDIO_STAGE, {})

        image_inputs = image_inputs if isinstance(image_inputs, dict) else {}
        audio_inputs = audio_inputs if isinstance(audio_inputs, dict) else {}

        pixel_values = image_inputs.get("pixel_values")
        image_grid_thw = image_inputs.get("image_grid_thw")
        if not isinstance(pixel_values, torch.Tensor) or not isinstance(
            image_grid_thw, torch.Tensor
        ):
            image_inputs = {
                "_skip": True,
                "_result": {
                    "image_embeds": torch.empty(0),
                    "image_grid_thw": torch.empty((0, 3), dtype=torch.long),
                    "image_token_counts": torch.empty(0, dtype=torch.long),
                },
            }

        input_features = audio_inputs.get("input_features")
        if not isinstance(input_features, torch.Tensor):
            audio_inputs = {
                "_skip": True,
                "_result": {
                    "audio_embeds": torch.empty(0),
                    "audio_feature_lengths": torch.empty(0, dtype=torch.long),
                    "audio_output_lengths": torch.empty(0, dtype=torch.long),
                },
            }

        return {
            IMAGE_STAGE: image_inputs,
            AUDIO_STAGE: audio_inputs,
        }

    def merge_for_thinker(
        self,
        *,
        frontend_out: FrontendOutput,
        encoder_outs: dict[str, Any],
    ) -> dict[str, Any]:
        mm_inputs = frontend_out.get("mm_inputs", {})
        mm_image = mm_inputs.get("image", {}) if isinstance(mm_inputs, dict) else {}
        mm_audio = mm_inputs.get("audio", {}) if isinstance(mm_inputs, dict) else {}

        image_out = (
            encoder_outs.get(IMAGE_STAGE, {}) if isinstance(encoder_outs, dict) else {}
        )
        audio_out = (
            encoder_outs.get(AUDIO_STAGE, {}) if isinstance(encoder_outs, dict) else {}
        )

        image_embeds = (
            _as_tensor(image_out.get("image_embeds"))
            if isinstance(image_out, dict)
            else None
        )
        audio_embeds = (
            _as_tensor(audio_out.get("audio_embeds"))
            if isinstance(audio_out, dict)
            else None
        )

        image_grid_thw = _as_tensor(
            (
                image_out.get("image_grid_thw")
                if isinstance(image_out, dict)
                and image_out.get("image_grid_thw") is not None
                else mm_image.get("image_grid_thw")
            ),
            dtype=torch.long,
        )
        feature_attention_mask = _as_tensor(
            mm_audio.get("feature_attention_mask"),
            dtype=torch.long,
        )
        audio_feature_lengths = _as_tensor(
            (
                audio_out.get("audio_feature_lengths")
                if isinstance(audio_out, dict)
                and audio_out.get("audio_feature_lengths") is not None
                else mm_audio.get("audio_feature_lengths")
            ),
            dtype=torch.long,
        )

        thinker_model_inputs: dict[str, Any] = {}
        if _non_empty(image_embeds):
            thinker_model_inputs["image_embeds"] = image_embeds
        if _non_empty(audio_embeds):
            thinker_model_inputs["audio_embeds"] = audio_embeds
        if _non_empty(image_grid_thw):
            thinker_model_inputs["image_grid_thw"] = image_grid_thw
        if _non_empty(feature_attention_mask):
            thinker_model_inputs["feature_attention_mask"] = feature_attention_mask
        if _non_empty(audio_feature_lengths):
            thinker_model_inputs["audio_feature_lengths"] = audio_feature_lengths

        return {"model_inputs": thinker_model_inputs}

    def prune_frontend_for_thinker(
        self,
        *,
        frontend_out: FrontendOutput,
        encoder_outs: dict[str, Any],
    ) -> FrontendOutput:
        mm_inputs = frontend_out.get("mm_inputs", {})
        mm_image = mm_inputs.get("image", {}) if isinstance(mm_inputs, dict) else {}
        mm_audio = mm_inputs.get("audio", {}) if isinstance(mm_inputs, dict) else {}

        image_out = (
            encoder_outs.get(IMAGE_STAGE, {}) if isinstance(encoder_outs, dict) else {}
        )
        audio_out = (
            encoder_outs.get(AUDIO_STAGE, {}) if isinstance(encoder_outs, dict) else {}
        )

        image_grid_thw = _as_tensor(
            (
                image_out.get("image_grid_thw")
                if isinstance(image_out, dict)
                and image_out.get("image_grid_thw") is not None
                else mm_image.get("image_grid_thw")
            ),
            dtype=torch.long,
        )
        audio_feature_lengths = _as_tensor(
            (
                audio_out.get("audio_feature_lengths")
                if isinstance(audio_out, dict)
                and audio_out.get("audio_feature_lengths") is not None
                else mm_audio.get("audio_feature_lengths")
            ),
            dtype=torch.long,
        )

        adapter_state = frontend_out.get("adapter_state", {})
        stream_state = (
            adapter_state.get("stream_state", {})
            if isinstance(adapter_state, dict)
            else {}
        )

        pruned: FrontendOutput = {
            "prompt": frontend_out.get("prompt", {}),
            "mm_inputs": {
                "image": {"image_grid_thw": image_grid_thw},
                "audio": {"audio_feature_lengths": audio_feature_lengths},
            },
            "adapter_state": {
                "stream_state": (
                    stream_state
                    if isinstance(stream_state, dict)
                    else {"token_ids": [], "text": ""}
                ),
            },
        }
        return pruned

    def decode_events(
        self,
        *,
        thinker_out: ThinkerOutput,
        frontend_out: FrontendOutput,
        step: int,
    ) -> Iterable[OmniEvent]:
        del step
        output_ids = thinker_out.get("output_ids", [])
        if not isinstance(output_ids, list) or not output_ids:
            return []

        adapter_state = frontend_out.setdefault("adapter_state", {})
        stream_state = adapter_state.setdefault(
            "stream_state", {"token_ids": [], "text": ""}
        )
        token_ids = stream_state.setdefault("token_ids", [])
        prev_text = str(stream_state.setdefault("text", ""))

        is_final = bool(thinker_out.get("is_final"))

        if is_final:
            tokens = [
                int(t)
                for t in output_ids
                if self._eos_token_id is None or int(t) != int(self._eos_token_id)
            ]
            text = (
                self.tokenizer.decode(tokens, skip_special_tokens=True)
                if tokens
                else ""
            )
            stream_state["token_ids"] = tokens
            stream_state["text"] = text
            return [
                OmniEvent(
                    type="text_final",
                    modality="text",
                    payload={"text": text},
                    is_final=True,
                )
            ]

        token_id = int(output_ids[-1])
        if self._eos_token_id is not None and token_id == int(self._eos_token_id):
            text = str(stream_state.get("text", ""))
            return [
                OmniEvent(
                    type="text_final",
                    modality="text",
                    payload={"text": text},
                    is_final=True,
                )
            ]

        token_ids.append(token_id)
        decoded = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        if decoded.startswith(prev_text):
            delta = decoded[len(prev_text) :]
        else:
            delta = decoded
        stream_state["text"] = decoded

        if not delta:
            return []
        return [
            OmniEvent(
                type="text_delta",
                modality="text",
                payload={"text": delta},
                is_final=False,
            )
        ]


def create_aggregate_executor(*, adapter_name: str) -> FrontendExecutor:
    return create_adapter_aggregate_executor(adapter_name)


def create_decode_executor(model_id: str, *, adapter_name: str) -> FrontendExecutor:
    del model_id
    return create_adapter_decode_executor(adapter_name)
