# SPDX-License-Identifier: Apache-2.0
"""Engine request/response helpers for Qwen3-Omni stages."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from sglang_omni.models.qwen3_omni.components.talker_prefill import TalkerPrefillBuilder
from sglang_omni.models.qwen3_omni.debug_dump import dump_precision_pt
from sglang_omni.models.qwen3_omni.payload_types import PipelineState, ThinkerOutput
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

logger = logging.getLogger(__name__)


def _tensor_head(tensor: torch.Tensor | None, n: int = 4) -> list[float]:
    if tensor is None:
        return []
    return tensor.detach().reshape(-1)[:n].float().cpu().tolist()


def _tensor_norm(tensor: torch.Tensor | None) -> float:
    if tensor is None:
        return 0.0
    return float(tensor.detach().float().norm().cpu().item())


# Lightweight request data types (previously in engines/omni/runtime/)
class EncoderRequestData:
    """Encoder request — just wraps a dict of inputs."""

    def __init__(self, input_dict: dict | None = None, **kwargs):
        self.input_dict = input_dict or kwargs

    def get(self, key, default=None):
        return self.input_dict.get(key, default)


class ARRequestData:
    """AR request data — base for SGLangARRequestData."""


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
    model_inputs = {k: v for k, v in inputs.items() if k != "cache_key"}
    return EncoderRequestData(
        input_dict=model_inputs,
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
    input_ids_list = input_ids.to(dtype=torch.long).tolist()

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
            input_ids.to(dtype=torch.long), model_inputs, thinker_config
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
    req._input_embeds_are_projected = bool(input_embeds_are_projected)
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
    data.feedback_embeds = None
    if thinker_layer_hidden is not None:
        data.extra_model_outputs["thinker_layer_hidden"] = thinker_layer_hidden
    if multimodal_mask is not None:
        data.extra_model_outputs["talker_multimodal_mask"] = multimodal_mask
    data.input_embeds_are_projected = bool(input_embeds_are_projected)
    data.thinker_chunks_done = bool(thinker_chunks_done)
    data.trailing_text_hidden = trailing_text_hidden
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

    def _build_stream_output(request_id: str, req_data: Any, req_output: Any):
        if req_output.data is None:
            if os.environ.get("QWEN_TALKER_TRACE") == "1":
                logger.info(
                    "QWEN_TRACE new thinker_chunk_skip req=%s token_id=%s data_is_none=%s",
                    request_id,
                    None if req_output.data is None else int(req_output.data),
                    req_output.data is None,
                )
            return None
        extra = req_output.extra
        if not isinstance(extra, dict) or "hidden_states" not in extra:
            return None

        embed, layer_hidden = _split_dual_layer_hidden(extra["hidden_states"])
        token_id = int(req_output.data)
        chunk_index = int(getattr(req_data, "_precision_thinker_chunk_index", 0))

        if embed is not None:
            metadata = {"token_id": token_id}
            if layer_hidden is not None:
                metadata["layer_hidden"] = layer_hidden
            dump_precision_pt(
                prefix="thinker_chunk",
                request_id=request_id,
                step=chunk_index,
                payload={
                    "request_id": request_id,
                    "chunk_index": chunk_index,
                    "token_id": token_id,
                    "embed": embed.detach().cpu(),
                    "layer_hidden": (
                        layer_hidden.detach().cpu()
                        if isinstance(layer_hidden, torch.Tensor)
                        else None
                    ),
                },
            )
            setattr(req_data, "_precision_thinker_chunk_index", chunk_index + 1)
            if os.environ.get("QWEN_TALKER_TRACE") == "1":
                logger.info(
                    "QWEN_TRACE new thinker_chunk req=%s token_id=%s embed_norm=%.4f embed_head=%s layer_norm=%.4f layer_head=%s",
                    request_id,
                    token_id,
                    _tensor_norm(embed),
                    _tensor_head(embed),
                    _tensor_norm(layer_hidden),
                    _tensor_head(layer_hidden),
                )
            return OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=embed,
                metadata=metadata,
            )

        if layer_hidden is not None:
            dump_precision_pt(
                prefix="thinker_chunk",
                request_id=request_id,
                step=chunk_index,
                payload={
                    "request_id": request_id,
                    "chunk_index": chunk_index,
                    "token_id": token_id,
                    "embed": None,
                    "layer_hidden": layer_hidden.detach().cpu(),
                },
            )
            setattr(req_data, "_precision_thinker_chunk_index", chunk_index + 1)
            if os.environ.get("QWEN_TALKER_TRACE") == "1":
                logger.info(
                    "QWEN_TRACE new thinker_chunk req=%s token_id=%s embed_norm=0.0000 embed_head=[] layer_norm=%.4f layer_head=%s",
                    request_id,
                    token_id,
                    _tensor_norm(layer_hidden),
                    _tensor_head(layer_hidden),
                )
            return OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=layer_hidden,
                metadata={"token_id": token_id},
            )

        return None

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
        }

    def request_builder(payload: StagePayload) -> SGLangARRequestData:
        params = payload.request.params
        sampling_cfg = _resolve_talker_sampling_config(params)
        thinker_chunks = list(payload.prefetched_chunks)
        thinker_done = bool(payload.prefetched_stream_done)

        if not thinker_chunks:
            raise ValueError(
                "talker request_builder requires thinker stream chunks; "
                "direct thinker_out fallback has been removed"
            )

        prompt_chunks = thinker_chunks[:1]
        buffered_chunks = thinker_chunks[1:]
        prompt_prefill = prefill_builder.build_prompt_prefill(
            payload,
            prompt_chunks,
            thinker_done=False,
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
            trailing_text_hidden=prompt_prefill["trailing_text_hidden"],
            tts_pad_embed=prompt_prefill["tts_pad_embed"],
            thinker_chunks_done=False,
            thinker_config=thinker_config,
            talker_model_inputs=prompt_prefill["prompt_model_inputs"],
        )
        req_data.tts_eos_embed = prompt_prefill["tts_eos_embed"]
        req_data.thinker_stream_chunks = list(prompt_chunks)
        for chunk in buffered_chunks:
            prefill_builder.append_trailing_chunk(req_data, chunk)
        if thinker_done:
            prefill_builder.mark_thinker_done(req_data)
        payload.prefetched_chunks = []
        payload.prefetched_stream_done = False
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
        prefill_builder.append_trailing_chunk,
        prefill_builder.mark_thinker_done,
    )
