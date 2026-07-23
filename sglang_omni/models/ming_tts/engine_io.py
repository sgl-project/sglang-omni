# SPDX-License-Identifier: Apache-2.0
"""SGLang engine I/O adapters for Ming-Omni-TTS."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from sglang_omni.models.ming_tts.payload_types import (
    MingTTSState,
    load_ming_tts_state,
    store_ming_tts_state,
)
from sglang_omni.models.ming_tts.tokenizer import MingTTSTokenizerBundle
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData


@dataclass
class MingTTSSGLangRequestData(SGLangARRequestData):
    """CPU scheduler state; recurrent GPU tensors live in the model runner."""

    enforce_request_limits: bool = True
    state: MingTTSState | None = None
    audio_patch_token_id: int = 0
    audio_eos_token_id: int = 0
    engine_start_s: float = 0.0
    generated_latents: torch.Tensor | None = None
    generated_last_chunk: list[bool] = field(default_factory=list)
    stop_step: int | None = None


def make_ming_tts_scheduler_adapters(
    *,
    model: Any,
    tokenizer: MingTTSTokenizerBundle,
    reset_request: Callable[[str], None],
    owns_acoustic_result: bool = True,
):
    """Build StagePayload <-> SGLang request adapters for Ming-Omni-TTS."""

    def request_builder(payload: StagePayload) -> MingTTSSGLangRequestData:
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        def config_value(config: Any, field: str) -> Any:
            if config is None:
                return None
            if isinstance(config, dict):
                return config.get(field)
            return getattr(config, field, None)

        state = load_ming_tts_state(payload)
        input_ids_list = [int(token_id) for token_id in (state.input_ids or [])]

        vocab_size = None
        for owner in (
            getattr(model, "config", None),
            getattr(model, "model_config", None),
            getattr(model, "hf_text_config", None),
            model,
        ):
            value = config_value(owner, "vocab_size")
            if value is not None:
                vocab_size = int(value)
                break
            llm_config = config_value(owner, "llm_config")
            value = config_value(llm_config, "vocab_size")
            if value is not None:
                vocab_size = int(value)
                break
        if vocab_size is None:
            vocab_size = int(len(tokenizer.tokenizer))

        sampling_params = SamplingParams(
            max_new_tokens=int(state.max_decode_steps),
            temperature=0.0,
            stop_token_ids=[int(tokenizer.special.end_of_audio)],
        )
        sampling_params.normalize(None)
        sampling_params.verify(vocab_size)

        requires_projected_prefill = (
            state.spk_emb is not None or state.prompt_latent is not None
        )

        req_input_ids_list = input_ids_list
        req_extra_key = f"ming_tts:{payload.request_id}"
        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=req_input_ids_list,
            sampling_params=sampling_params,
            eos_token_ids={int(tokenizer.special.end_of_audio)},
            vocab_size=vocab_size,
            extra_key=req_extra_key,
        )
        req.tokenizer = None
        req._input_embeds_are_projected = requires_projected_prefill

        input_ids = torch.tensor(req_input_ids_list, dtype=torch.long)
        data = MingTTSSGLangRequestData(
            input_ids=input_ids,
            max_new_tokens=int(state.max_decode_steps),
            temperature=0.0,
            output_ids=req.output_ids,
            req=req,
            state=state,
            input_embeds_are_projected=requires_projected_prefill,
            audio_patch_token_id=int(tokenizer.special.audio_patch),
            audio_eos_token_id=int(tokenizer.special.end_of_audio),
            engine_start_s=time.perf_counter(),
        )
        data.stage_payload = payload
        return data

    def result_adapter(data: MingTTSSGLangRequestData) -> StagePayload:
        request_id = data.stage_payload.request_id
        try:
            if not owns_acoustic_result:
                return data.stage_payload
            payload = data.stage_payload
            state = data.state
            generated = data.generated_latents
            if generated is None:
                generated = torch.empty(
                    (0, int(model.patch_size), int(model.latent_dim)),
                    dtype=torch.float32,
                )

            raw = data.finish_reason
            if raw is None and data.req is not None:
                finished_reason = getattr(data.req, "finished_reason", None)
                if finished_reason is not None and hasattr(finished_reason, "to_json"):
                    raw = finished_reason.to_json().get("type")
                elif finished_reason is not None:
                    raw = str(finished_reason)

            normalized = str(raw).lower() if raw is not None else None
            if data.stop_step is not None:
                finish_reason = "stop"
            elif normalized is not None:
                if "length" in normalized:
                    finish_reason = "length"
                elif "abort" in normalized:
                    finish_reason = "abort"
                elif "error" in normalized:
                    finish_reason = "error"
                else:
                    finish_reason = str(raw)
            elif int(generated.shape[0]) >= int(data.max_new_tokens):
                finish_reason = "length"
            else:
                finish_reason = "stop"

            state.generated_last_chunk = [
                bool(item) for item in data.generated_last_chunk
            ]
            state.stop_step = data.stop_step
            state.finish_reason = finish_reason
            state.prompt_tokens = len(data.input_ids)
            state.completion_tokens = int(generated.shape[0])
            state.engine_time_s = time.perf_counter() - data.engine_start_s
            state.generated_latents = generated

            return store_ming_tts_state(payload, state)
        finally:
            reset_request(request_id)

    return request_builder, result_adapter


__all__ = [
    "MingTTSSGLangRequestData",
    "make_ming_tts_scheduler_adapters",
]
