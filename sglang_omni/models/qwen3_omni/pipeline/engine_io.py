# SPDX-License-Identifier: Apache-2.0
"""Engine request/response helpers for Qwen3-Omni stages."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import xxhash

from sglang_omni.engines.omni.runtime import ARRequestData, EncoderRequestData
from sglang_omni.models.qwen3_omni.io import PipelineState, ThinkerOutput

if TYPE_CHECKING:
    from sglang_omni.engines.omni.runtime.sglang_ar import SGLangARRequestData


logger = logging.getLogger(__name__)

DEFAULT_THINKER_MAX_NEW_TOKENS = 2048
DEFAULT_THINKER_MAX_SEQ_LEN = 8192


def validate_prompt_seq_len(
    input_ids: torch.Tensor,
    *,
    max_seq_len: int | None,
    max_new_tokens: int = DEFAULT_THINKER_MAX_NEW_TOKENS,
    request_id: str | None = None,
) -> None:
    if max_seq_len is None:
        return
    prompt_len = int(input_ids.numel())
    if prompt_len >= max_seq_len:
        logger.info(
            f"rejecting request {request_id}: prompt {prompt_len} tokens "
            f">= max_seq_len {max_seq_len}"
        )
        raise ValueError(
            f"The input ({prompt_len} tokens) is longer than the model's "
            f"context length ({max_seq_len} tokens)."
        )
    total_tokens = prompt_len + int(max_new_tokens)
    if total_tokens >= max_seq_len:
        logger.info(
            f"rejecting request {request_id}: prompt {prompt_len} + "
            f"max_new_tokens {int(max_new_tokens)} = {total_tokens} tokens "
            f">= max_seq_len {max_seq_len}"
        )
        raise ValueError(
            f"Requested token count exceeds the model's maximum context length "
            f"of {max_seq_len} tokens. You requested a total of {total_tokens} "
            f"tokens: {prompt_len} tokens from the input messages and "
            f"{int(max_new_tokens)} tokens for the completion. Please reduce "
            f"the number of tokens in the input messages or the completion to "
            f"fit within the limit."
        )


def build_encoder_request(
    state: PipelineState, *, stage_name: str
) -> EncoderRequestData:
    inputs = state.encoder_inputs.get(stage_name)
    if not isinstance(inputs, dict) or not inputs:
        return EncoderRequestData(input_dict={"_skip": True, "_result": {}})
    if inputs.get("_skip"):
        skip_result = inputs.get("_result")
        return EncoderRequestData(
            input_dict=inputs,
            output_dict=skip_result if isinstance(skip_result, dict) else {},
        )
    cache_key = inputs.get("cache_key")
    return EncoderRequestData(
        input_dict=inputs,
        cache_key=str(cache_key) if cache_key is not None else None,
    )


def apply_encoder_result(
    state: PipelineState,
    *,
    stage_name: str,
    result: Any,
) -> None:
    if isinstance(result, EncoderRequestData):
        if result.output_dict is not None:
            encoder_out = result.output_dict
        elif result.embeddings is not None:
            encoder_out = result.embeddings
        else:
            encoder_out = {}
    else:
        encoder_out = result if isinstance(result, dict) else {"result": result}

    state.encoder_outs[stage_name] = encoder_out
    state.engine_outputs[stage_name] = encoder_out


def build_thinker_request(
    state: PipelineState,
    *,
    params: dict[str, Any],
    max_seq_len: int | None = None,
    request_id: str | None = None,
) -> ARRequestData:
    prompt = state.prompt
    if not isinstance(prompt, dict):
        raise TypeError("prompt missing for thinker request")

    input_ids = prompt.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("prompt.input_ids must be a torch.Tensor")
    max_new_tokens = params.get("max_new_tokens", DEFAULT_THINKER_MAX_NEW_TOKENS)
    validate_prompt_seq_len(
        input_ids,
        max_seq_len=max_seq_len,
        max_new_tokens=max_new_tokens,
        request_id=request_id,
    )

    attention_mask = prompt.get("attention_mask")
    thinker_inputs = state.thinker_inputs or {}

    model_inputs = dict(thinker_inputs.get("model_inputs", {}))
    if not model_inputs:
        model_inputs = {
            k: v
            for k, v in thinker_inputs.items()
            if k not in ("capture_model_output_keys", "media_cache_keys")
        }

    capture_keys = thinker_inputs.get("capture_model_output_keys", ())
    if "attention_mask" in model_inputs:
        model_inputs.pop("attention_mask", None)

    return ARRequestData(
        input_ids=input_ids.to(dtype=torch.long),
        attention_mask=(
            attention_mask if isinstance(attention_mask, torch.Tensor) else None
        ),
        model_inputs=model_inputs,
        capture_model_output_keys=tuple(capture_keys) if capture_keys else (),
        max_new_tokens=max_new_tokens,
        temperature=params.get("temperature", 0.0),
    )


def _compute_mrope_positions(
    input_ids: torch.Tensor,
    model_inputs: dict[str, Any],
    thinker_config: Any,
) -> torch.Tensor | None:
    """Compute M-RoPE positions for multimodal inputs."""
    from sglang.srt.layers.rotary_embedding import MRotaryEmbedding

    image_grid_thw = model_inputs.get("image_grid_thw")
    video_grid_thw = model_inputs.get("video_grid_thw")
    spatial_merge_size = thinker_config.vision_config.spatial_merge_size
    image_token_id = thinker_config.image_token_id
    video_token_id = thinker_config.video_token_id
    vision_start_token_id = thinker_config.vision_start_token_id
    tokens_per_second = thinker_config.vision_config.tokens_per_second
    audio_token_id = thinker_config.audio_token_id
    audio_start_token_id = thinker_config.audio_start_token_id
    position_id_per_seconds = thinker_config.position_id_per_seconds
    use_audio_in_video = model_inputs.get("use_audio_in_video", False)
    audio_feature_lengths = model_inputs.get("audio_feature_lengths")

    ids_2d = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids

    # Move all tensors to CPU — get_rope_index creates CPU tensors internally
    ids_2d = ids_2d.cpu()
    if isinstance(image_grid_thw, torch.Tensor):
        image_grid_thw = image_grid_thw.cpu()
    if isinstance(video_grid_thw, torch.Tensor):
        video_grid_thw = video_grid_thw.cpu()
    second_per_grid_ts = model_inputs.get("video_second_per_grid")
    if isinstance(second_per_grid_ts, torch.Tensor):
        second_per_grid_ts = second_per_grid_ts.cpu()
    if isinstance(audio_feature_lengths, torch.Tensor):
        audio_feature_lengths = audio_feature_lengths.cpu()

    kwargs: dict[str, Any] = {
        "audio_token_id": audio_token_id,
        "audio_start_token_id": audio_start_token_id,
        "position_id_per_seconds": position_id_per_seconds,
        "use_audio_in_video": use_audio_in_video,
        "audio_seqlens": audio_feature_lengths,
    }

    mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
        spatial_merge_size=spatial_merge_size,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_token_id=vision_start_token_id,
        model_type="qwen3_omni_moe",
        tokens_per_second=tokens_per_second,
        input_ids=ids_2d,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        **kwargs,
    )
    # mrope_positions: [3, 1, seq_len] -> [3, seq_len]
    return mrope_positions.squeeze(1), mrope_position_delta


def build_sglang_thinker_request(
    state: PipelineState,
    *,
    params: dict[str, Any],
    tokenizer: Any,
    vocab_size: int,
    max_seq_len: int | None = None,
    request_id: str | None = None,
    thinker_config: Any = None,
) -> "SGLangARRequestData":
    """Build SGLangARRequestData from pipeline state.

    Constructs a SGLang Req with normalized SamplingParams, then wraps it
    in SGLangARRequestData (which inherits ARRequestData).
    """
    from sglang.srt.managers.schedule_batch import MultimodalInputs, Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    from sglang_omni.engines.omni.runtime.sglang_ar import SGLangARRequestData

    prompt = state.prompt
    if not isinstance(prompt, dict):
        raise TypeError("prompt missing for thinker request")

    input_ids = prompt.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("prompt.input_ids must be a torch.Tensor")
    max_new_tokens = params.get("max_new_tokens", DEFAULT_THINKER_MAX_NEW_TOKENS)
    validate_prompt_seq_len(
        input_ids,
        max_seq_len=max_seq_len,
        max_new_tokens=max_new_tokens,
        request_id=request_id,
    )

    input_ids = input_ids.to(dtype=torch.long)

    attention_mask = prompt.get("attention_mask")
    thinker_inputs = state.thinker_inputs or {}

    model_inputs = dict(thinker_inputs.get("model_inputs", {}))
    if not model_inputs:
        model_inputs = {
            k: v
            for k, v in thinker_inputs.items()
            if k not in ("capture_model_output_keys", "media_cache_keys")
        }
    capture_keys = thinker_inputs.get("capture_model_output_keys", ())

    # Note (Yifei):
    # Compute pad_values from per-modality cache keys and replace placeholder
    # tokens in input_ids so that RadixCache naturally branches on different
    # media content while sharing common text prefixes (e.g. system prompt).
    original_input_ids = input_ids
    media_cache_keys = thinker_inputs.get("media_cache_keys", {})
    pad_values: dict[str, int] = {}
    if media_cache_keys and thinker_config is not None:
        token_id_map: dict[int, int] = {}
        for modality, orig_token_id in [
            ("image", thinker_config.image_token_id),
            ("video", thinker_config.video_token_id),
            ("audio", thinker_config.audio_token_id),
        ]:
            cache_key = media_cache_keys.get(modality)
            if cache_key is None:
                continue
            h = xxhash.xxh3_64(cache_key.encode()).intdigest()
            # Note (Yifei):
            # Unlike SGLang main (hash % 2^30 without offset),
            # offset by vocab_size to avoid collision with real token IDs.
            pad_val = vocab_size + h % (1 << 30)
            pad_values[modality] = pad_val
            token_id_map[orig_token_id] = pad_val

        if token_id_map:
            input_ids = input_ids.clone()
            for orig_id, pad_val in token_id_map.items():
                input_ids[input_ids == orig_id] = pad_val

        if pad_values:
            model_inputs["pad_values"] = pad_values

    input_ids_list = input_ids.tolist()
    if "attention_mask" in model_inputs:
        model_inputs.pop("attention_mask", None)

    temperature = params.get("temperature", 0.0)

    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    sampling_params.normalize(tokenizer)
    sampling_params.verify(vocab_size)

    # Build SGLang Req
    rid = request_id or "req-0"
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        vocab_size=vocab_size,
    )

    # Compute M-RoPE positions and attach multimodal_inputs to Req
    if thinker_config is not None and model_inputs:
        mrope_result = _compute_mrope_positions(
            original_input_ids.to(dtype=torch.long), model_inputs, thinker_config
        )
        if mrope_result is not None:
            mrope_positions, mrope_position_delta = mrope_result
            mm_inputs = MultimodalInputs(mm_items=[])
            mm_inputs.mrope_positions = mrope_positions
            mm_inputs.mrope_position_delta = mrope_position_delta
            req.multimodal_inputs = mm_inputs

    # Attach model_inputs to Req for image embedding merge in SGLangModelRunner.
    # Always initialize both attributes so downstream code can access directly.
    req.omni_model_inputs = model_inputs if model_inputs else None
    req._omni_consumed = None

    # Build SGLangARRequestData — output_ids points to req.output_ids
    data = SGLangARRequestData(
        input_ids=input_ids.to(dtype=torch.long),
        attention_mask=(
            attention_mask if isinstance(attention_mask, torch.Tensor) else None
        ),
        model_inputs=model_inputs,
        capture_model_output_keys=tuple(capture_keys) if capture_keys else (),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        output_ids=req.output_ids,
        req=req,
    )
    return data


def build_sglang_talker_request(
    thinker_hidden_states: torch.Tensor,
    *,
    tokenizer: Any,
    codec_vocab_size: int,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
    request_id: str | None = None,
    codec_bos_id: int = 2149,
    codec_eos_id: int | None = None,
    suppress_tokens: list[int] | None = None,
    thinker_layer_hidden: torch.Tensor | None = None,
    thinker_token_ids: list[int] | torch.Tensor | None = None,
    audio_token_id: int | None = None,
    image_token_id: int | None = None,
    video_token_id: int | None = None,
    talker_input_embeds: torch.Tensor | None = None,
    talker_input_ids: torch.Tensor | list[int] | None = None,
    input_embeds_are_projected: bool = False,
    trailing_text_hidden: list[torch.Tensor] | torch.Tensor | None = None,
    tts_pad_embed: torch.Tensor | None = None,
    thinker_chunks_done: bool = True,
    thinker_config: Any = None,
    talker_model_inputs: dict[str, Any] | None = None,
) -> "SGLangARRequestData":
    """Build SGLang AR request for the Talker from thinker hidden states.

    Stores thinker hidden states as Req.input_embeds so SGLang's pipeline
    passes them through ForwardBatch.input_embeds -> model.forward(input_embeds=...).
    Uses dummy input_ids of matching length for position tracking.

    Args:
        thinker_hidden_states: Embed layer hidden states [seq_len, hidden_size].
        thinker_layer_hidden: Optional layer-N hidden states for dual-layer mode.
        thinker_token_ids: Optional thinker output token ids aligned with hidden states.
    """
    from sglang.srt.managers.schedule_batch import MultimodalInputs, Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    from sglang_omni.engines.omni.runtime.sglang_ar import SGLangARRequestData

    if talker_input_embeds is not None:
        input_embeds = talker_input_embeds.float().cpu().tolist()
        if talker_input_ids is None:
            raise ValueError("talker_input_ids is required with talker_input_embeds")
        input_ids_tensor = torch.as_tensor(talker_input_ids, dtype=torch.long)
        input_ids_list = input_ids_tensor.tolist()
        seq_len = len(input_ids_list)
    else:
        # thinker_hidden_states: [seq_len, thinker_hidden_size]
        seq_len = thinker_hidden_states.shape[0]

        # Dummy input_ids — codec BOS token repeated for each position
        input_ids_list = [codec_bos_id] * seq_len
        input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)

        # Convert hidden states to list-of-lists for Req.input_embeds
        input_embeds = thinker_hidden_states.float().cpu().tolist()

    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop_token_ids=[int(codec_eos_id)] if codec_eos_id is not None else None,
        # SGLang currently materializes logit_bias against the scheduler vocab
        # width, which is the text tokenizer vocab rather than the codec vocab
        # for talker requests. Passing suppress_tokens here therefore produces a
        # shape mismatch in sampling. Keep the HF-aligned talker sampling knobs
        # that are safe in the codec path and leave special-token suppression to
        # a future codec-aware sampler integration.
        logit_bias=None,
    )
    sampling_params.normalize(tokenizer)
    sampling_params.verify(codec_vocab_size)

    rid = request_id or "talker-req-0"
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        input_embeds=input_embeds,
        eos_token_ids={int(codec_eos_id)} if codec_eos_id is not None else None,
        vocab_size=codec_vocab_size,
    )
    req.omni_model_inputs = None
    req._omni_consumed = None
    req._input_embeds_are_projected = bool(input_embeds_are_projected)
    req._codec_suppress_tokens = (
        tuple(int(token_id) for token_id in suppress_tokens)
        if suppress_tokens
        else None
    )
    if thinker_config is not None:
        mrope_positions, mrope_position_delta = _compute_mrope_positions(
            input_ids_tensor.to(dtype=torch.long),
            talker_model_inputs or {},
            thinker_config,
        )
        mm_inputs = MultimodalInputs(mm_items=[])
        mm_inputs.mrope_positions = mrope_positions
        mm_inputs.mrope_position_delta = mrope_position_delta
        req.multimodal_inputs = mm_inputs

    multimodal_mask: torch.Tensor | None = None
    if thinker_token_ids is not None:
        token_ids = torch.as_tensor(thinker_token_ids, dtype=torch.long)
        if token_ids.numel() == seq_len:
            mask = torch.zeros(seq_len, dtype=torch.bool)
            for token_id in (audio_token_id, image_token_id, video_token_id):
                if token_id is not None:
                    mask |= token_ids == int(token_id)
            multimodal_mask = mask

    if thinker_layer_hidden is not None:
        req.omni_model_inputs = {
            "talker_layer_hidden_states": thinker_layer_hidden,
            "talker_multimodal_mask": multimodal_mask,
        }

    data = SGLangARRequestData(
        input_ids=input_ids_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        output_ids=req.output_ids,
        req=req,
    )
    data.suppress_tokens = list(req._codec_suppress_tokens or [])
    data.talker_model_inputs = dict(talker_model_inputs or {})
    if thinker_layer_hidden is not None:
        data.extra_model_outputs["thinker_layer_hidden"] = thinker_layer_hidden
    if multimodal_mask is not None:
        data.extra_model_outputs["talker_multimodal_mask"] = multimodal_mask
    data.input_embeds_are_projected = bool(input_embeds_are_projected)
    data.thinker_chunks_done = bool(thinker_chunks_done)
    if trailing_text_hidden is not None:
        data.trailing_text_hidden = trailing_text_hidden
    if tts_pad_embed is not None:
        data.tts_pad_embed = tts_pad_embed
    return data


def apply_thinker_result(
    state: PipelineState,
    *,
    stage_name: str,
    result: Any,
) -> ThinkerOutput:
    if isinstance(result, ARRequestData):
        output_ids = list(result.output_ids)
        prompt_tokens = (
            int(result.input_ids.shape[0])
            if result.input_ids is not None and hasattr(result.input_ids, "shape")
            else 0
        )
        finish_reason = None
        req_finish_reason = getattr(
            getattr(result, "req", None), "finished_reason", None
        )
        if hasattr(req_finish_reason, "to_json"):
            finish_reason = req_finish_reason.to_json().get("type")
        thinker_out: ThinkerOutput = {
            "output_ids": output_ids,
            "step": len(output_ids),
            "is_final": True,
            "finish_reason": finish_reason,
            "extra_model_outputs": dict(result.extra_model_outputs),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(output_ids),
        }
    else:
        thinker_out = {
            "output_ids": [],
            "step": 0,
            "is_final": True,
            "finish_reason": None,
            "extra_model_outputs": {"result": result},
        }

    state.thinker_out = thinker_out
    state.engine_outputs[stage_name] = thinker_out
    return thinker_out
