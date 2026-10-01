# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the framework-native dots.tts pipeline."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import torch

from sglang_omni.models.dots_tts.codec import (
    DotsReferenceEncoder,
    load_dots_audio_codec,
)
from sglang_omni.models.dots_tts.compat import import_dots_tts
from sglang_omni.models.dots_tts.payload_types import DotsTTSState
from sglang_omni.models.dots_tts.vocoder import DotsTTSStreamingVocoder
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_data_uri_from_reference
from sglang_omni.utils.checkpoint import resolve_checkpoint

_DEFAULT_CONTEXT_LENGTH = 2048


def configure_optimized_kernels() -> None:
    """Configure the process-global dots DiT compile hook.

    This temporary upstream monkey patch is restored once dots.tts supports
    shared-module FX for its one-shot prefill and recurrent decode modules.
    """
    import_dots_tts()
    from dots_tts.modules.backbone import dit_inference, inference_utils
    from sglang.srt.compilation.torch_compile_decoration import set_torch_compile_config

    set_torch_compile_config()
    inference_utils._COMPILE_OPTIONS["mode"] = (
        os.environ.get(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            "SGLANG_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs"
        )
    )
    os.environ.setdefault("DOTS_TTS_DELAYED_DIT_BACKEND", "sdpa")
    compile_module_forward = dit_inference.compile_module_forward

    def _compile_dit_step(module: torch.nn.Module):
        # ponytail: compile only the recurrent hot step; restore one-shot
        # prefill/first-patch compile after upstream supports shared-module FX.
        if isinstance(
            module,
            (dit_inference.FirstPatchStep, dit_inference.KvPrefillStep),
        ):
            return module
        else:
            pass
        return compile_module_forward(module)

    dit_inference.compile_module_forward = _compile_dit_step


def first_not_none(*values: Any, default: Any = None) -> Any:
    return next((value for value in values if value is not None), default)


def as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def inputs(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"text": value}
    else:
        pass
    return as_dict(value)


def reference_path(value: Any) -> str | None:
    if value is None:
        return None
    else:
        pass
    if isinstance(value, str):
        return value
    else:
        pass
    if isinstance(value, dict):
        path = (
            value.get("audio_path")
            or value.get("path")
            or audio_data_uri_from_reference(value)
        )
        return str(path) if path else None
    else:
        pass
    raise TypeError("dots.tts reference audio must be a path")


def preprocess_dots_tts_payload(
    payload: StagePayload,
    *,
    tokenizer: Any,
    model_config: Any,
    max_generate_length: int,
    max_sequence_length: int,
    num_steps: int = 4,
    audio_span_token_ids: tuple[int, int] | None = None,
    vocab_size: int | None = None,
) -> StagePayload:
    import_dots_tts()
    from dots_tts.data.pipelines.tokenizing import build_generation_schedule
    from dots_tts.data.pipelines.tts_pipeline import (
        DEFAULT_INSTRUCTION_TTS_TEMPLATE,
        DEFAULT_TEXT_TO_AUDIO_TEMPLATE,
        DEFAULT_TRAIN_TEMPLATE,
    )
    from dots_tts.utils.text import (
        attach_language_tag,
        detect,
        normalize_language_code,
        normalize_text,
    )
    from dots_tts.utils.tokenizer import (
        AUDIO_COMP_SPAN_TOKEN,
        AUDIO_GEN_SPAN_TOKEN,
        require_token_id,
    )

    request_inputs = inputs(payload.request.inputs)
    params = as_dict(payload.request.params)
    tts_params = as_dict(as_dict(payload.request.metadata).get("tts_params"))
    stage_params = as_dict(params.get("stage_params"))
    engine_params = as_dict(stage_params.get("latent_engine"))
    references = request_inputs.get("references")
    if isinstance(references, list) and len(references) > 1:
        raise ValueError("dots.tts accepts at most one reference audio")
    else:
        pass
    reference = (
        references[0]
        if isinstance(references, list)
        and references
        and isinstance(references[0], dict)
        else {}
    )

    text = str(
        first_not_none(
            request_inputs.get("input"), request_inputs.get("text"), default=""
        )
    )
    if not text.strip():
        raise ValueError("dots.tts requires non-empty input text")
    else:
        pass
    prompt_audio = first_not_none(
        reference_path(request_inputs.get("prompt_audio_path")),
        reference_path(request_inputs.get("prompt_audio")),
        reference_path(request_inputs.get("reference_audio")),
        reference_path(tts_params.get("ref_audio")),
        reference_path(reference),
    )
    prompt_text_value = first_not_none(
        request_inputs.get("prompt_text"),
        request_inputs.get("reference_text"),
        tts_params.get("ref_text"),
        reference.get("text"),
    )
    prompt_text = None if prompt_text_value is None else str(prompt_text_value)
    if prompt_text and not prompt_audio:
        raise ValueError("dots.tts prompt_text requires prompt audio")
    else:
        pass

    template_name = str(
        first_not_none(
            request_inputs.get("template_name"),
            tts_params.get("template_name"),
            params.get("template_name"),
            tts_params.get("task_type"),
            params.get("task_type"),
            default=(
                "instruction_tts"
                if first_not_none(
                    tts_params.get("instructions"), request_inputs.get("instructions")
                )
                else "tts"
            ),
        )
    )
    if template_name == "Base":
        template_name = "tts"
    else:
        pass
    templates = {
        "tts": DEFAULT_TRAIN_TEMPLATE,
        "instruction_tts": DEFAULT_INSTRUCTION_TTS_TEMPLATE,
        "text_to_audio": DEFAULT_TEXT_TO_AUDIO_TEMPLATE,
    }
    if template_name not in templates:
        raise ValueError(
            f"dots.tts base support accepts template_name in {sorted(templates)}"
        )
    else:
        pass

    normalize = bool(
        first_not_none(
            request_inputs.get("normalize_text"),
            tts_params.get("normalize_text"),
            params.get("normalize_text"),
            default=False,
        )
    )
    text = normalize_text(text.strip()) if normalize else text.strip()
    language_value = first_not_none(
        request_inputs.get("language"), tts_params.get("language")
    )
    language = None
    if language_value is not None:
        raw_language = str(language_value).strip()
        language = (
            normalize_language_code(detect(text))
            if raw_language.lower() in {"auto", "auto_detect"}
            else normalize_language_code(raw_language)
        )
        if raw_language and raw_language.lower() != "none" and language is None:
            raise ValueError(f"unsupported dots.tts language {raw_language!r}")
        else:
            pass
    else:
        pass
    normalized_prompt_text = ""
    if prompt_text and prompt_text.strip():
        normalized_prompt_text = prompt_text.strip() + "\n"
        if language is not None:
            normalized_prompt_text = attach_language_tag(
                normalized_prompt_text, language
            )
        else:
            pass
    elif language is not None:
        text = attach_language_tag(text, language)
    else:
        pass

    request_limit = int(
        first_not_none(
            request_inputs.get("max_generate_length"),
            tts_params.get("max_generate_length"),
            engine_params.get("max_generate_length"),
            params.get("max_generate_length"),
            default=max_generate_length,
        )
    )
    if request_limit <= 0 or request_limit > int(max_generate_length):
        raise ValueError(
            f"dots.tts max_generate_length must be in [1, {max_generate_length}]"
        )
    else:
        pass
    schedule_spec = build_generation_schedule(
        text=f"{normalized_prompt_text}{text}",
        tokenizer=tokenizer,
        template=templates[template_name],
        max_audio_tokens=request_limit,
    )
    schedule = torch.tensor([schedule_spec["schedule_ids"]], dtype=torch.long)
    if schedule.numel() > int(max_sequence_length):
        raise ValueError(
            f"dots.tts schedule exceeds context length {max_sequence_length}"
        )
    else:
        pass

    if audio_span_token_ids is None:
        audio_span_token_ids = (
            require_token_id(tokenizer, AUDIO_GEN_SPAN_TOKEN),
            require_token_id(tokenizer, AUDIO_COMP_SPAN_TOKEN),
        )
    else:
        pass
    if vocab_size is None:
        vocab_size = len(tokenizer)
    else:
        pass

    state = DotsTTSState(
        sample_rate=int(model_config.vocoder.sample_rate),
        prompt_audio_path=prompt_audio,
        use_prompt_prefill=bool(normalized_prompt_text),
        speaker_scale=float(
            first_not_none(
                request_inputs.get("speaker_scale"),
                tts_params.get("speaker_scale"),
                engine_params.get("speaker_scale"),
                params.get("speaker_scale"),
                default=1.5,
            )
        ),
        ode_method=str(
            first_not_none(
                request_inputs.get("ode_method"),
                tts_params.get("ode_method"),
                engine_params.get("ode_method"),
                params.get("ode_method"),
                default="euler",
            )
        ),
        num_steps=int(
            first_not_none(
                request_inputs.get("num_steps"),
                tts_params.get("num_steps"),
                engine_params.get("num_steps"),
                params.get("num_steps"),
                default=num_steps,
            )
        ),
        guidance_scale=float(
            first_not_none(
                request_inputs.get("guidance_scale"),
                tts_params.get("guidance_scale"),
                engine_params.get("guidance_scale"),
                params.get("guidance_scale"),
                default=1.2,
            )
        ),
        eos_threshold=float(
            first_not_none(
                request_inputs.get("eos_threshold"),
                tts_params.get("eos_threshold"),
                engine_params.get("eos_threshold"),
                params.get("eos_threshold"),
                default=0.8,
            )
        ),
        seed=(
            None
            if (
                seed := first_not_none(
                    request_inputs.get("seed"),
                    tts_params.get("seed"),
                    engine_params.get("seed"),
                    params.get("seed"),
                )
            )
            is None
            else int(seed)
        ),
        max_new_tokens=(
            None
            if params.get("max_new_tokens") is None
            else int(params["max_new_tokens"])
        ),
        stream=bool(params.get("stream", False)),
        generation_schedule=schedule,
        # note: copy per request; DotsTTSState owns a mutable list and the
        # cached tuple is shared across every request on this executor.
        audio_span_token_ids=list(audio_span_token_ids),
        latent_patch_size=int(model_config.patch_size),
        vocab_size=int(vocab_size),
    )
    if state.num_steps <= 0:
        raise ValueError("dots.tts num_steps must be positive")
    else:
        pass
    if not all(
        math.isfinite(value)
        for value in (state.speaker_scale, state.guidance_scale, state.eos_threshold)
    ):
        raise ValueError("dots.tts floating-point parameters must be finite")
    else:
        pass
    payload.data = state.to_dict()
    return payload


def load_model_metadata(model_path: str) -> tuple[str, Any, Any, int]:
    import_dots_tts()
    from dots_tts.models.dots_tts.config import ModelConfig
    from transformers import AutoTokenizer

    root = str(Path(resolve_checkpoint(model_path)).resolve())
    dots_config = ModelConfig.model_validate_json(
        (Path(root) / "config.json").read_text(encoding="utf-8")
    )
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    llm_config = json.loads((Path(root) / "llm_config.json").read_text())
    context_length = min(
        int(llm_config.get("max_position_embeddings", _DEFAULT_CONTEXT_LENGTH)),
        _DEFAULT_CONTEXT_LENGTH,
    )
    return root, dots_config, tokenizer, context_length


def create_preprocessing_executor(
    model_path: str,
    *,
    max_generate_length: int = 500,
    num_steps: int = 4,
    max_concurrency: int = 8,
) -> SimpleScheduler:
    _root, config, tokenizer, context_length = load_model_metadata(model_path)
    from dots_tts.utils.tokenizer import (
        AUDIO_COMP_SPAN_TOKEN,
        AUDIO_GEN_SPAN_TOKEN,
        require_token_id,
    )

    audio_span_token_ids = (
        require_token_id(tokenizer, AUDIO_GEN_SPAN_TOKEN),
        require_token_id(tokenizer, AUDIO_COMP_SPAN_TOKEN),
    )
    vocab_size = len(tokenizer)

    def _preprocess(payload: StagePayload) -> StagePayload:
        return preprocess_dots_tts_payload(
            payload,
            tokenizer=tokenizer,
            model_config=config,
            max_generate_length=max_generate_length,
            max_sequence_length=context_length,
            num_steps=num_steps,
            audio_span_token_ids=audio_span_token_ids,
            vocab_size=vocab_size,
        )

    return SimpleScheduler(_preprocess, max_concurrency=max_concurrency)


def create_reference_encode_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_concurrency: int = 8,
    max_batch_size: int = 1,
    max_batch_wait_ms: float = 4.0,
) -> SimpleScheduler:
    from sglang_omni.utils.device import resolve_concrete_device

    concrete_device = resolve_concrete_device(device, gpu_id)
    if concrete_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("dots.tts requires CUDA")
    else:
        pass
    codec = load_dots_audio_codec(model_path, device=str(concrete_device))
    encoder = DotsReferenceEncoder(
        codec,
        model_id=str(model_path),
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
    return SimpleScheduler(
        encoder.encode_payload,
        max_concurrency=max_concurrency,
        shutdown_callback=encoder.close,
    )


def create_sglang_latent_engine_executor(
    model_path: str,
    *,
    precision: str = "bfloat16",
    optimize: bool = True,
    max_generate_length: int = 500,
    num_steps: int = 4,
    device: str | None = None,
    gpu_id: int | None = None,
    server_args_overrides: dict[str, Any] | None = None,
) -> OmniScheduler:
    from sglang_omni.models.dots_tts.engine_builder import DotsTTSEngineBuilder

    if not torch.cuda.is_available():
        raise RuntimeError("dots.tts requires CUDA")
    else:
        pass
    return DotsTTSEngineBuilder(
        optimize=optimize,
        num_steps=num_steps,
        max_audio_patches=max_generate_length,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=precision,
        server_args_overrides=server_args_overrides,
    )


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    optimize: bool = True,
    enable_streaming_audio_vae_cuda_graph: bool = True,
    vocoder_merge_steps: int = 4,
    max_batch_size: int = 4,
    max_batch_wait_ms: int = 2,
    stream_slots: int = 16,
) -> DotsTTSStreamingVocoder:
    from sglang_omni.utils.device import resolve_concrete_device

    if not torch.cuda.is_available():
        raise RuntimeError("dots.tts requires CUDA")
    else:
        pass
    codec = load_dots_audio_codec(
        model_path, device=str(resolve_concrete_device(device, gpu_id))
    )
    vocoder = DotsTTSStreamingVocoder(
        codec,
        optimize=optimize,
        enable_streaming_audio_vae_cuda_graph=enable_streaming_audio_vae_cuda_graph,
        merge_steps=vocoder_merge_steps,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        stream_slots=stream_slots,
    )
    # note (guozhihao-224): allocate the slot pool at setup so OOM / shape
    # mismatch surface before readiness, not on the first live chunk.
    vocoder.ensure_slot_pool()
    backend = (
        f"{vocoder.cuda_graph_count} step CUDA graphs"
        if vocoder.cuda_graph_count
        else "eager step"
    )
    logging.getLogger(__name__).info(
        f"dots.tts vocoder backend: slot-pooled streaming, {backend} "
        f"(optimize={optimize}, merge_steps={vocoder.merge_steps}, "
        f"stream_slots={vocoder.stream_slots}, batch_size={max_batch_size}, "
        f"stream_batch_cap={vocoder.stream_chunk_batch_max}, wait_ms={max_batch_wait_ms})"
    )
    return vocoder


__all__ = [
    "create_preprocessing_executor",
    "create_reference_encode_executor",
    "create_sglang_latent_engine_executor",
    "create_vocoder_executor",
    "preprocess_dots_tts_payload",
]
