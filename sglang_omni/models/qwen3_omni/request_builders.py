# SPDX-License-Identifier: Apache-2.0
"""Engine request/response helpers for Qwen3-Omni stages."""

from __future__ import annotations

import collections
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import xxhash

from sglang_omni.models.qwen3_omni.components.talker_prefill import TalkerPrefillBuilder
from sglang_omni.models.qwen3_omni.payload_types import PipelineState, ThinkerOutput
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData
from sglang_omni.scheduling.types import ARRequestData

IMAGE_STAGE = "image_encoder"
AUDIO_STAGE = "audio_encoder"


@dataclass(slots=True)
class EncoderRequestData:
    """Typed encoder request data for pre-thinker stages."""

    model_inputs: dict[str, Any]
    cache_key: str | None = None
    skip_result: dict[str, Any] | None = None


def build_encoder_request(
    state: PipelineState, *, stage_name: str
) -> EncoderRequestData:
    inputs = state.encoder_inputs.get(stage_name)
    if not isinstance(inputs, dict) or not inputs:
        return EncoderRequestData(model_inputs={}, skip_result={})
    if inputs.get("_skip"):
        skip_result = inputs.get("_result")
        return EncoderRequestData(
            model_inputs={},
            skip_result=skip_result if isinstance(skip_result, dict) else {},
        )
    cache_key = inputs.get("cache_key")
    model_inputs = {k: v for k, v in inputs.items() if k != "cache_key"}
    return EncoderRequestData(
        model_inputs=model_inputs,
        cache_key=str(cache_key) if cache_key is not None else None,
    )


def apply_encoder_result(
    state: PipelineState,
    *,
    stage_name: str,
    result: Any,
) -> None:
    if isinstance(result, EncoderRequestData):
        encoder_out = result.skip_result if result.skip_result is not None else {}
    else:
        encoder_out = result if isinstance(result, dict) else {"result": result}

    state.encoder_outs[stage_name] = encoder_out
    state.engine_outputs[stage_name] = encoder_out


def build_lightweight_mm_inputs(mm_inputs: dict[str, Any]) -> dict[str, Any]:
    mm_image = mm_inputs.get("image", {})
    mm_audio = mm_inputs.get("audio", {})
    mm_video = mm_inputs.get("video", {})
    return {
        "image": _select_present_fields(mm_image, ("image_grid_thw",)),
        "audio": _select_present_fields(
            mm_audio,
            ("feature_attention_mask", "audio_feature_lengths"),
        ),
        "video": _select_present_fields(
            mm_video,
            ("video_grid_thw", "video_second_per_grid", "use_audio_in_video"),
        ),
    }


def project_preprocessing_to_image_encoder(payload: StagePayload) -> StagePayload:
    return _project_preprocessing_to_encoder(payload, stage_name=IMAGE_STAGE)


def project_preprocessing_to_audio_encoder(payload: StagePayload) -> StagePayload:
    return _project_preprocessing_to_encoder(payload, stage_name=AUDIO_STAGE)


def project_preprocessing_to_mm_aggregate(payload: StagePayload) -> StagePayload:
    state = PipelineState.from_dict(payload.data)
    projected = PipelineState(
        prompt=dict(state.prompt) if isinstance(state.prompt, dict) else None,
        mm_inputs=build_lightweight_mm_inputs(state.mm_inputs),
        encoder_inputs=_project_encoder_input_metadata(state.encoder_inputs),
        stream_state=dict(state.stream_state),
    )
    return _payload_with_state(payload, projected)


def project_encoder_to_mm_aggregate(payload: StagePayload) -> StagePayload:
    state = PipelineState.from_dict(payload.data)
    stage_name = _single_encoder_stage_name(state)
    encoder_out = state.encoder_outs.get(stage_name, {})
    projected = PipelineState(encoder_outs={stage_name: encoder_out})
    return _payload_with_state(payload, projected)


def _project_preprocessing_to_encoder(
    payload: StagePayload,
    *,
    stage_name: str,
) -> StagePayload:
    state = PipelineState.from_dict(payload.data)
    projected = PipelineState(
        encoder_inputs=_select_encoder_inputs(
            state.encoder_inputs, stage_name=stage_name
        )
    )
    return _payload_with_state(payload, projected)


def _payload_with_state(payload: StagePayload, state: PipelineState) -> StagePayload:
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=state.to_dict(),
    )


def _select_encoder_inputs(
    encoder_inputs: dict[str, dict[str, Any]],
    *,
    stage_name: str,
) -> dict[str, dict[str, Any]]:
    stage_inputs = encoder_inputs.get(stage_name)
    if not isinstance(stage_inputs, dict):
        return {}
    return {stage_name: dict(stage_inputs)}


def _project_encoder_input_metadata(
    encoder_inputs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    projected: dict[str, dict[str, Any]] = {}
    for stage_name, stage_inputs in encoder_inputs.items():
        if not isinstance(stage_inputs, dict):
            continue
        stage_metadata: dict[str, Any] = {}
        cache_key = stage_inputs.get("cache_key")
        if cache_key is not None:
            stage_metadata["cache_key"] = cache_key
        if stage_inputs.get("_skip"):
            stage_metadata["_skip"] = True
        if stage_metadata:
            projected[stage_name] = stage_metadata
    return projected


def _select_present_fields(
    source: dict[str, Any],
    keys: tuple[str, ...],
) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key in keys:
        value = source.get(key)
        if value is not None:
            selected[key] = value
    return selected


def _single_encoder_stage_name(state: PipelineState) -> str:
    if len(state.encoder_outs) != 1:
        raise ValueError(
            f"Expected exactly one encoder output in payload, got {sorted(state.encoder_outs)}"
        )
    return next(iter(state.encoder_outs))


def build_thinker_request(
    state: PipelineState,
    *,
    params: dict[str, Any],
) -> ARRequestData:
    prompt = state.prompt
    input_ids = prompt["input_ids"]
    attention_mask = prompt.get("attention_mask")
    thinker_inputs = state.thinker_inputs or {}

    model_inputs = dict(thinker_inputs.get("model_inputs", {}))
    if not model_inputs:
        model_inputs = {
            k: v for k, v in thinker_inputs.items() if k != "capture_model_output_keys"
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
        max_new_tokens=params.get("max_new_tokens"),
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
    request_id: str | None = None,
    thinker_config: Any = None,
) -> "SGLangARRequestData":
    """Build SGLangARRequestData from pipeline state.

    Constructs a SGLang Req with normalized SamplingParams, then wraps it
    in SGLangARRequestData (which inherits ARRequestData).
    """
    from sglang.srt.managers.schedule_batch import MultimodalInputs, Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    # SGLangARRequestData already imported at module level

    prompt = state.prompt
    input_ids = prompt["input_ids"]
    original_input_ids = input_ids

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
            pad_val = vocab_size + h % (1 << 30)
            pad_values[modality] = pad_val
            token_id_map[orig_token_id] = pad_val
        if token_id_map:
            input_ids = input_ids.clone()
            for orig_id, pad_val in token_id_map.items():
                input_ids[input_ids == orig_id] = pad_val
        if pad_values:
            model_inputs["pad_values"] = pad_values
    if "attention_mask" in model_inputs:
        model_inputs.pop("attention_mask", None)
    input_ids_list = input_ids.to(dtype=torch.long).tolist()

    max_new_tokens = params.get("max_new_tokens", 2048)
    temperature = params.get("temperature", 0.0)
    top_p = params.get("top_p", 1.0)
    top_k = params.get("top_k", -1)
    min_p = params.get("min_p", 0.0)
    repetition_penalty = params.get("repetition_penalty", 1.0)
    stop = params.get("stop") or []
    stop_token_ids = params.get("stop_token_ids") or []
    seed = params.get("seed")

    # Build SGLang SamplingParams and normalize
    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        stop=stop,
        stop_token_ids=stop_token_ids,
        sampling_seed=seed,
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
    req.tokenizer = tokenizer

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

    req.omni_model_inputs = model_inputs if model_inputs else None
    req._omni_consumed = None
    req._codec_suppress_tokens = None

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
    pending_text_queue: (
        collections.deque[torch.Tensor] | list[torch.Tensor] | torch.Tensor | None
    ) = None,
    tts_pad_embed: torch.Tensor | None = None,
    thinker_chunks_done: bool = True,
    thinker_config: Any = None,
    talker_model_inputs: dict[str, Any] | None = None,
    sampling_seed: int | None = None,
) -> "SGLangARRequestData":
    """Build SGLang AR request for the Talker from thinker hidden states.

    Stores thinker hidden states as Req.input_embeds so SGLang's pipeline
    passes them through ForwardBatch.input_embeds -> model.forward(input_embeds=...).
    Uses dummy input_ids of matching length for position tracking, while the
    host-side request data keeps a FIFO queue of future text rows for decode.

    Args:
        thinker_hidden_states: Embed layer hidden states [seq_len, hidden_size].
        thinker_layer_hidden: Optional layer-N hidden states for dual-layer mode.
        thinker_token_ids: Optional thinker output token ids aligned with hidden states.
    """
    from sglang.srt.managers.schedule_batch import MultimodalInputs, Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    # SGLangARRequestData already imported at module level

    if talker_input_embeds is not None:
        input_embeds = talker_input_embeds.float().cpu().tolist()
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
        logit_bias=None,
        sampling_seed=sampling_seed,
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
    req.tokenizer = tokenizer
    req.omni_model_inputs = dict(talker_model_inputs or {})
    req._omni_consumed = None
    req._codec_suppress_tokens = (
        tuple(int(token_id) for token_id in suppress_tokens)
        if suppress_tokens
        else None
    )
    if thinker_config is not None and talker_model_inputs:
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
        req.omni_model_inputs["talker_layer_hidden_states"] = thinker_layer_hidden
        req.omni_model_inputs["talker_multimodal_mask"] = multimodal_mask
    elif req.omni_model_inputs:
        req.omni_model_inputs["talker_layer_hidden_states"] = None
        req.omni_model_inputs["talker_multimodal_mask"] = None
    else:
        req.omni_model_inputs = None

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
    if isinstance(pending_text_queue, torch.Tensor):
        pending_text_queue = [row.detach().cpu() for row in pending_text_queue]
    data.pending_text_queue = collections.deque(pending_text_queue or [])
    data.tts_pad_embed = tts_pad_embed
    return data


def apply_thinker_result(
    state: PipelineState,
    *,
    stage_name: str,
    result: Any,
) -> ThinkerOutput:
    output_ids = list(result.output_ids)
    thinker_out: ThinkerOutput = {
        "output_ids": output_ids,
        "step": len(output_ids),
        "is_final": True,
        "extra_model_outputs": dict(result.extra_model_outputs),
    }

    finish_reason = getattr(result, "finish_reason", None)
    if finish_reason is not None:
        thinker_out["finish_reason"] = finish_reason

    state.thinker_out = thinker_out
    state.engine_outputs[stage_name] = thinker_out
    return thinker_out


def make_thinker_stream_output_builder():
    def _normalize_chunk_hidden(hidden: torch.Tensor | None) -> torch.Tensor | None:
        if hidden is None:
            return None
        if hidden.ndim == 1:
            return hidden
        if hidden.ndim == 2:
            return hidden[0]
        return None

    def _split_dual_layer_hidden(
        hidden: dict[str | int, torch.Tensor] | torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if isinstance(hidden, torch.Tensor):
            return _normalize_chunk_hidden(hidden), None

        embed = hidden.get("embed")
        if embed is None and 0 in hidden:
            embed = hidden[0]
        if embed is None and "0" in hidden:
            embed = hidden["0"]

        layer_hidden = None
        for key, value in hidden.items():
            if key in ("embed", 0, "0"):
                continue
            if isinstance(value, torch.Tensor):
                layer_hidden = value
                break
        return _normalize_chunk_hidden(embed), _normalize_chunk_hidden(layer_hidden)

    def _build_stream_output(
        request_id: str, req_data: Any, req_output: Any
    ) -> list[OutgoingMessage]:
        req = getattr(req_data, "req", None)
        if req is not None and int(getattr(req, "is_chunked", 0) or 0) > 0:
            # Match the legacy thinker stream adapter: while chunked prefill is still
            # consuming prompt tokens, suppress hidden-state streaming to the talker.
            # Emitting chunks this early lets prompt-side states masquerade as the
            # first assistant token and can leak the user/ref-text prompt into TTS.
            return []
        if req_output.data is None:
            return []

        token_id = int(req_output.data)
        messages: list[OutgoingMessage] = []

        # Skip per-token decode emit when not streaming; talker_ar below stays
        # unconditional since talker generates audio either way.
        stage_payload = req_data.stage_payload
        is_streaming = bool(
            stage_payload is not None
            and (stage_payload.request.params or {}).get("stream", False)
        )
        if is_streaming:
            # Wrap int — relay_io.write_blob is tensor-only.
            messages.append(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data=torch.tensor([token_id], dtype=torch.long),
                    target="decode",
                    metadata={"token_id": token_id},
                )
            )

        # Speech mode: also stream hidden states to the talker for codec gen.
        extra = req_output.extra
        if isinstance(extra, dict) and "hidden_states" in extra:
            embed, layer_hidden = _split_dual_layer_hidden(extra["hidden_states"])
            if embed is not None:
                metadata = {"token_id": token_id}
                if layer_hidden is not None:
                    metadata["layer_hidden"] = layer_hidden
                messages.append(
                    OutgoingMessage(
                        request_id=request_id,
                        type="stream",
                        data=embed,
                        target="talker_ar",
                        metadata=metadata,
                    )
                )
            elif layer_hidden is not None:
                messages.append(
                    OutgoingMessage(
                        request_id=request_id,
                        type="stream",
                        data=layer_hidden,
                        target="talker_ar",
                        metadata={"token_id": token_id},
                    )
                )

        return messages

    return _build_stream_output


def make_thinker_scheduler_adapters(
    *,
    tokenizer: Any,
    vocab_size: int,
    thinker_config: Any = None,
    stage_name: str = "thinker",
):
    """Build model-specific StagePayload <-> scheduler adapters for thinker."""

    def request_builder(payload: StagePayload) -> SGLangARRequestData:
        state = PipelineState.from_dict(payload.data)
        params = payload.request.params or {}
        req_data = build_sglang_thinker_request(
            state,
            params=params,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            request_id=payload.request_id,
            thinker_config=thinker_config,
        )
        req_data.stage_payload = payload
        return req_data

    def result_adapter(data: SGLangARRequestData) -> StagePayload:
        payload = data.stage_payload
        state = PipelineState.from_dict(payload.data)
        apply_thinker_result(state, stage_name=stage_name, result=data)
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )

    return request_builder, result_adapter


def make_talker_scheduler_adapters(
    *,
    tokenizer: Any,
    codec_vocab_size: int,
    model: Any,
    model_path: str,
    thinker_config: Any,
    required_aux_hidden_key: int,
    codec_bos_id: int = 2149,
    codec_eos_id: int | None = None,
    codec_nothink_id: int = 2155,
    codec_think_bos_id: int = 2156,
    codec_think_eos_id: int = 2157,
    codec_pad_id: int = 2148,
    audio_token_id: int | None = None,
    image_token_id: int | None = None,
    video_token_id: int | None = None,
    tts_bos_token_id: int = 151672,
    tts_eos_token_id: int = 151673,
    tts_pad_token_id: int = 151671,
    im_start_token_id: int = 151644,
    im_end_token_id: int = 151645,
    system_token_id: int = 8948,
    user_token_id: int = 872,
    assistant_token_id: int = 77091,
    speaker_map: dict[str, int] | None = None,
):
    """Build model-specific StagePayload <-> scheduler adapters for talker."""
    prefill_builder = TalkerPrefillBuilder(
        model=model,
        model_path=model_path,
        audio_token_id=audio_token_id,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        tts_bos_token_id=tts_bos_token_id,
        tts_eos_token_id=tts_eos_token_id,
        tts_pad_token_id=tts_pad_token_id,
        im_start_token_id=im_start_token_id,
        im_end_token_id=im_end_token_id,
        system_token_id=system_token_id,
        user_token_id=user_token_id,
        assistant_token_id=assistant_token_id,
        codec_bos_id=codec_bos_id,
        codec_nothink_id=codec_nothink_id,
        codec_think_bos_id=codec_think_bos_id,
        codec_think_eos_id=codec_think_eos_id,
        codec_pad_id=codec_pad_id,
        speaker_map=speaker_map,
    )

    def _fallback_thinker_chunks_from_state(payload: StagePayload) -> list[Any]:
        state = PipelineState.from_dict(payload.data)
        thinker_out = state.thinker_out or state.engine_outputs.get("thinker")
        if not isinstance(thinker_out, dict):
            return []

        output_ids = thinker_out.get("output_ids")
        if not isinstance(output_ids, list) or not output_ids:
            return []

        token_ids = torch.tensor(
            [int(token_id) for token_id in output_ids],
            dtype=torch.long,
        )
        token_embeds = prefill_builder._load_prompt_token_embeddings(token_ids)
        return [
            SimpleNamespace(
                data=row.detach().cpu(),
                metadata={"token_id": int(token_id)},
            )
            for token_id, row in zip(token_ids.tolist(), token_embeds)
        ]

    def _resolve_talker_sampling_config(params: dict[str, Any]) -> dict[str, Any]:
        codec_eos_id = int(getattr(model.config, "codec_eos_token_id", -1))
        suppress_tokens = [
            token_id
            for token_id in range(max(codec_vocab_size - 1024, 0), codec_vocab_size)
            if token_id != codec_eos_id
        ]
        return {
            "max_new_tokens": int(params.get("talker_max_new_tokens", 4096)),
            "temperature": float(params.get("talker_temperature", 0.9)),
            "top_k": int(params.get("talker_top_k", 50)),
            "top_p": float(params.get("talker_top_p", 1.0)),
            "repetition_penalty": float(params.get("talker_repetition_penalty", 1.05)),
            "codec_eos_id": codec_eos_id if codec_eos_id >= 0 else None,
            "suppress_tokens": suppress_tokens,
            "seed": params.get("seed"),
        }

    def request_builder(payload: StagePayload) -> SGLangARRequestData:
        params = payload.request.params
        sampling_cfg = _resolve_talker_sampling_config(params)
        thinker_chunks = list(payload.prefetched_chunks)
        thinker_done = bool(payload.prefetched_stream_done)

        if not thinker_done:
            raise RuntimeError(
                "QwenTalkerScheduler called talker request_builder before thinker "
                "stream completion"
            )

        if not thinker_chunks:
            thinker_chunks = _fallback_thinker_chunks_from_state(payload)
        if not thinker_chunks:
            raise ValueError("talker request_builder requires thinker output tokens")

        prompt_prefill = prefill_builder.build_prompt_prefill(
            payload,
            thinker_chunks,
            thinker_done=True,
        )
        req_data = build_sglang_talker_request(
            thinker_hidden_states=torch.empty(0),
            tokenizer=tokenizer,
            codec_vocab_size=codec_vocab_size,
            max_new_tokens=sampling_cfg["max_new_tokens"],
            temperature=sampling_cfg["temperature"],
            top_k=sampling_cfg["top_k"],
            top_p=sampling_cfg["top_p"],
            repetition_penalty=sampling_cfg["repetition_penalty"],
            request_id=payload.request_id,
            codec_bos_id=codec_bos_id,
            codec_eos_id=sampling_cfg["codec_eos_id"],
            suppress_tokens=sampling_cfg["suppress_tokens"],
            audio_token_id=audio_token_id,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            talker_input_embeds=prompt_prefill["input_embeds"],
            talker_input_ids=prompt_prefill["input_ids"],
            input_embeds_are_projected=True,
            pending_text_queue=prompt_prefill["pending_text_queue"],
            tts_pad_embed=prompt_prefill["tts_pad_embed"],
            thinker_chunks_done=True,
            thinker_config=thinker_config,
            talker_model_inputs=prompt_prefill["prompt_model_inputs"],
            sampling_seed=sampling_cfg.get("seed"),
        )
        req_data.tts_eos_embed = prompt_prefill["tts_eos_embed"]
        req_data.stage_payload = payload
        return req_data

    def result_adapter(data: SGLangARRequestData) -> StagePayload:
        payload = data.stage_payload
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=payload.data,
        )

    return (
        request_builder,
        result_adapter,
        prefill_builder.append_text_chunk,
        prefill_builder.mark_thinker_done,
    )
