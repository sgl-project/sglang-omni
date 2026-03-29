# SPDX-License-Identifier: Apache-2.0
"""Stage executors for MiniCPM-V/MiniCPM-o pipelines.

This module provides factory functions for creating stage executors:
- Preprocessing (CPU): tokenization, image/audio processing
- Image Encoder (CUDA): SigLIP + Resampler
- Audio Encoder (CUDA): Whisper-based (MiniCPM-o only)
- Aggregate (CPU): merge encoder outputs
- LLM (CUDA): text generation with embedded images/audio
- Decode (CPU): token to text conversion
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoTokenizer

from sglang_omni.engines.ar.sglang_backend.server_args_builder import (
    build_sglang_server_args,
)
from sglang_omni.engines.omni import (
    create_ar_engine,
    create_sglang_ar_engine,
    create_single_pass_engine,
)
from sglang_omni.executors import EngineExecutor, PreprocessingExecutor
from sglang_omni.models.minicpm_v.components.image_encoder import MiniCPMVImageEncoder
from sglang_omni.models.minicpm_v.components.preprocessor import MiniCPMVPreprocessor
from sglang_omni.models.minicpm_v.io import LLMOutput, OmniEvent
from sglang_omni.models.minicpm_v.pipeline.engine_io import (
    apply_encoder_result,
    apply_llm_result,
    build_encoder_request,
    build_llm_request,
    build_sglang_llm_request,
)
from sglang_omni.models.minicpm_v.pipeline.merge import decode_events
from sglang_omni.models.minicpm_v.pipeline.next_stage import AUDIO_STAGE, IMAGE_STAGE, LLM_STAGE, VOCODER_STAGE
from sglang_omni.models.minicpm_v.pipeline.state_io import load_state, store_state
from sglang_omni.proto import StagePayload


def _event_to_dict(event: OmniEvent) -> dict[str, Any]:
    return {
        "type": event.type,
        "modality": event.modality,
        "payload": dict(event.payload),
        "is_final": bool(event.is_final),
    }


# ---------------------------------------------------------------------------
# Stage 1: Preprocessing (CPU)
# ---------------------------------------------------------------------------


def create_preprocessing_executor(model_path: str) -> PreprocessingExecutor:
    """Factory for the preprocessing stage.

    Creates a MiniCPM-V preprocessor that handles:
    - Chat template application
    - Image loading and slice processing
    - Tokenization with image placeholders
    """
    preprocessor = MiniCPMVPreprocessor(model_path=model_path)

    async def _preprocess(payload: StagePayload) -> StagePayload:
        return await preprocessor(payload)

    return PreprocessingExecutor(_preprocess)


# ---------------------------------------------------------------------------
# Stage 2: Aggregate (CPU)
# ---------------------------------------------------------------------------


def create_aggregate_executor() -> PreprocessingExecutor:
    """Factory for the aggregate stage.

    This is a pass-through executor - the actual merging is done by
    the input_handler's merge_fn (merge_for_llm).
    """

    def _identity(payload: StagePayload) -> StagePayload:
        return payload

    return PreprocessingExecutor(_identity)


# ---------------------------------------------------------------------------
# Stage 3: Image Encoder (CUDA)
# ---------------------------------------------------------------------------


def _create_encoder_executor(
    *,
    stage_name: str,
    model: torch.nn.Module,
    device: str,
    use_cache: bool = True,
    cache_size: int | None = 64,
) -> EngineExecutor:
    """Generic encoder executor factory."""

    def _request_builder(payload: StagePayload):
        state = load_state(payload)
        return build_encoder_request(state, stage_name=stage_name)

    def _result_builder(payload: StagePayload, result: Any) -> StagePayload:
        state = load_state(payload)
        apply_encoder_result(state, stage_name=stage_name, result=result)
        return store_state(payload, state)

    engine = create_single_pass_engine(
        model,
        device=device,
        use_cache=use_cache,
        cache_size=cache_size,
    )
    return EngineExecutor(
        engine=engine, request_builder=_request_builder, result_builder=_result_builder
    )


def create_image_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
) -> EngineExecutor:
    """Factory for the image encoder stage.

    Creates a MiniCPM-V image encoder (SigLIP + Resampler) wrapped
    in an EngineExecutor.
    """
    model = MiniCPMVImageEncoder(model_path=model_path, device=device, dtype=dtype)
    return _create_encoder_executor(stage_name=IMAGE_STAGE, model=model, device=device)


# ---------------------------------------------------------------------------
# Stage 3b: Audio Encoder (CUDA) - MiniCPM-o 2.6 only
# ---------------------------------------------------------------------------


def create_audio_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
) -> EngineExecutor:
    """Factory for the audio encoder stage (MiniCPM-o 2.6 only).

    Creates a MiniCPM-o audio encoder (Whisper-based) wrapped
    in an EngineExecutor.
    """
    from sglang_omni.models.minicpm_v.components.audio_encoder import (
        MiniCPMOAudioEncoder,
    )

    model = MiniCPMOAudioEncoder(model_path=model_path, device=device, dtype=dtype)
    return _create_encoder_executor(stage_name=AUDIO_STAGE, model=model, device=device)


# ---------------------------------------------------------------------------
# Stage 4: LLM (CUDA) - HuggingFace AR Engine
# ---------------------------------------------------------------------------


def create_llm_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
    max_seq_len: int = 8192,
) -> EngineExecutor:
    """Factory for the LLM stage using HuggingFace AR Engine.

    This is the Phase 1 implementation that uses HF generate() for
    text generation. Phase 2 will replace this with SGLang backend.
    """
    from transformers import AutoModelForCausalLM

    # Load the MiniCPM-V model for text generation
    # The model handles image embedding injection internally
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if dtype is None else getattr(torch, dtype),
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    step_counters: dict[str, int] = {}

    def _request_builder(payload: StagePayload):
        state = load_state(payload)
        step_counters.pop(payload.request_id, None)
        return build_llm_request(state, params=payload.request.params)

    def _result_builder(payload: StagePayload, result: Any) -> StagePayload:
        state = load_state(payload)
        apply_llm_result(state, stage_name=LLM_STAGE, result=result)
        step_counters.pop(payload.request_id, None)
        return store_state(payload, state)

    def _stream_builder(payload: StagePayload | None, item: Any) -> Any:
        if payload is None:
            return None
        request_id = payload.request_id
        step = step_counters.get(request_id, 0) + 1
        step_counters[request_id] = step

        try:
            token_id = int(item)
        except Exception:
            return {"token_id": item, "step": step}

        state = load_state(payload)
        llm_out: LLMOutput = {
            "output_ids": [token_id],
            "step": step,
            "is_final": False,
        }
        events = list(
            decode_events(
                llm_out=llm_out,
                state=state,
                tokenizer=tokenizer,
                eos_token_id=eos_token_id,
                step=step,
            )
        )
        store_state(payload, state)
        if eos_token_id is not None and token_id == eos_token_id and not events:
            events = [
                OmniEvent(type="text_final", modality="text", payload={}, is_final=True)
            ]
        text_delta = ""
        for event in events:
            if event.is_final:
                continue
            t = event.payload.get("text")
            if event.modality == "text" and t:
                text_delta += t

        result: dict[str, Any] = {
            "events": [_event_to_dict(event) for event in events],
            "token_id": token_id,
            "step": step,
            "stage": LLM_STAGE,
        }
        if text_delta:
            result["text"] = text_delta
        return result

    engine = create_ar_engine(
        model=model,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        device=device,
    )

    return EngineExecutor(
        engine=engine,
        request_builder=_request_builder,
        result_builder=_result_builder,
        stream_builder=_stream_builder,
    )


# ---------------------------------------------------------------------------
# Stage 4 (alt): LLM (CUDA) - SGLang Backend
# ---------------------------------------------------------------------------


def create_sglang_llm_executor(
    server_args: Any,
    model_path: str,
    *,
    gpu_id: int = 0,
) -> EngineExecutor:
    """Factory for the LLM stage using SGLang backend.

    This creates a SGLang-backed LLM executor for production use.
    Requires a custom SGLang model class for MiniCPM-V (Phase 2).
    """
    from sglang_omni.models.minicpm_v.components.common import load_llm_config
    from sglang_omni.models.minicpm_v.factory import _patch_minicpm_config_for_sglang

    # Patch config before SGLang loads it
    _patch_minicpm_config_for_sglang(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    vocab_size = getattr(tokenizer, "vocab_size", 32000)
    llm_config = load_llm_config(model_path)

    step_counters: dict[str, int] = {}

    def _request_builder(payload: StagePayload):
        state = load_state(payload)
        step_counters.pop(payload.request_id, None)
        data = build_sglang_llm_request(
            state,
            params=payload.request.params,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            request_id=payload.request_id,
            llm_config=llm_config,
        )
        return data

    def _result_builder(payload: StagePayload, result: Any) -> StagePayload:
        state = load_state(payload)
        apply_llm_result(state, stage_name=LLM_STAGE, result=result)
        step_counters.pop(payload.request_id, None)
        return store_state(payload, state)

    def _stream_builder(payload: StagePayload | None, item: Any) -> Any:
        if payload is None:
            return None
        request_id = payload.request_id
        step = step_counters.get(request_id, 0) + 1
        step_counters[request_id] = step

        try:
            token_id = int(item)
        except Exception:
            return {"token_id": item, "step": step}

        state = load_state(payload)
        llm_out: LLMOutput = {
            "output_ids": [token_id],
            "step": step,
            "is_final": False,
        }
        events = list(
            decode_events(
                llm_out=llm_out,
                state=state,
                tokenizer=tokenizer,
                eos_token_id=eos_token_id,
                step=step,
            )
        )
        store_state(payload, state)
        if eos_token_id is not None and token_id == eos_token_id and not events:
            events = [
                OmniEvent(type="text_final", modality="text", payload={}, is_final=True)
            ]

        text_to_add = ""
        for event in events:
            if event.modality == "text" and "text" in event.payload:
                if event.is_final:
                    text_to_add = event.payload["text"]
                    break
                else:
                    text_to_add += event.payload["text"]

        result: dict[str, Any] = {
            "events": [_event_to_dict(event) for event in events],
            "token_id": token_id,
            "step": step,
            "stage": LLM_STAGE,
        }
        if text_to_add:
            result["text"] = text_to_add
        return result

    engine = create_sglang_ar_engine(
        server_args=server_args,
        gpu_id=gpu_id,
    )

    return EngineExecutor(
        engine=engine,
        request_builder=_request_builder,
        result_builder=_result_builder,
        stream_builder=_stream_builder,
    )


def create_sglang_llm_executor_from_config(
    model_path: str,
    *,
    gpu_id: int = 0,
    llm_max_seq_len: int = 8192,
    server_args_overrides: dict[str, Any] | None = None,
) -> EngineExecutor:
    """Create a SGLang LLM executor from JSON-serializable config args.

    This keeps pipeline config args plain dict types while still constructing
    a typed ServerArgs object internally.
    """
    server_args = build_sglang_server_args(
        model_path, context_length=llm_max_seq_len, **(server_args_overrides or {})
    )
    return create_sglang_llm_executor(
        server_args=server_args,
        model_path=model_path,
        gpu_id=gpu_id,
    )


# ---------------------------------------------------------------------------
# Stage 5: Decode (CPU)
# ---------------------------------------------------------------------------


def create_decode_executor(model_path: str) -> PreprocessingExecutor:
    """Factory for the decode stage.

    Converts LLM output tokens to final text response.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def _decode(payload: StagePayload) -> StagePayload:
        state = load_state(payload)
        llm_out = state.llm_out or state.engine_outputs.get(LLM_STAGE)
        if not isinstance(llm_out, dict):
            llm_out = {
                "output_ids": [],
                "step": 0,
                "is_final": True,
                "extra_model_outputs": {},
            }

        step = int(llm_out.get("step") or len(llm_out.get("output_ids", [])))
        events = list(
            decode_events(
                llm_out=llm_out,
                state=state,
                tokenizer=tokenizer,
                eos_token_id=eos_token_id,
                step=step,
            )
        )
        event_dicts = [_event_to_dict(event) for event in events]

        result: dict[str, Any] = {"events": event_dicts}
        final_event = next(
            (
                event
                for event in reversed(events)
                if event.is_final or event.type in {"text_final", "final"}
            ),
            None,
        )
        if final_event is not None:
            result.update(final_event.payload)
            result.setdefault("modality", final_event.modality)

        if "text" not in result:
            output_ids = llm_out.get("output_ids")
            if (
                callable(getattr(tokenizer, "decode", None))
                and isinstance(output_ids, list)
                and output_ids
            ):
                result["text"] = tokenizer.decode(output_ids, skip_special_tokens=True)
                result.setdefault("modality", "text")

        payload.data = result
        return payload

    return PreprocessingExecutor(_decode)


# ---------------------------------------------------------------------------
# Stage 6: Vocoder (CUDA) - MiniCPM-o 2.6 audio output
# ---------------------------------------------------------------------------


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
) -> PreprocessingExecutor:
    """Factory for the vocoder stage (MiniCPM-o 2.6 only).

    Converts LLM-generated audio tokens to PCM waveform using
    CosyVoice flow-matching vocoder.
    """
    from sglang_omni.models.minicpm_v.vocoder.cosyvoice import CosyVoiceVocoder

    torch_dtype = None
    if dtype is not None:
        torch_dtype = getattr(torch, dtype, torch.float32)

    vocoder = CosyVoiceVocoder(model_path=model_path, device=device, dtype=torch_dtype)

    def _vocode(payload: StagePayload) -> StagePayload:
        state = load_state(payload)

        # Get LLM output
        llm_out = state.llm_out or state.engine_outputs.get(LLM_STAGE, {})
        if not isinstance(llm_out, dict):
            # No LLM output, return as-is
            return payload

        output_ids = llm_out.get("output_ids", [])
        if not output_ids:
            return payload

        # Extract and convert audio tokens
        try:
            result = vocoder.vocode_from_llm_output(output_ids)
        except Exception as e:
            # Log error but don't fail the pipeline
            import logging
            logging.getLogger(__name__).warning("Vocoder error: %s", e)
            return payload

        if result is None:
            # No audio tokens found in output
            return payload

        # Store audio output in payload
        audio_data = result.audio_samples.tolist()
        payload.data = payload.data if isinstance(payload.data, dict) else {}
        payload.data["audio_data"] = audio_data
        payload.data["sample_rate"] = result.sample_rate
        payload.data["audio_duration_s"] = result.duration_s
        payload.data["modality"] = "audio"

        # Store metadata
        if result.inference_time_s > 0:
            payload.data.setdefault("usage", {})
            payload.data["usage"]["vocoder_time_s"] = round(result.inference_time_s, 6)

        return payload

    return PreprocessingExecutor(_vocode)
