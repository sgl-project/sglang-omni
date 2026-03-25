# SPDX-License-Identifier: Apache-2.0
"""Stage executors for Qwen3-Omni pipelines."""

from __future__ import annotations

import logging
import os
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
from sglang_omni.models.qwen3_omni.components.audio_encoder import Qwen3OmniAudioEncoder
from sglang_omni.models.qwen3_omni.components.image_encoder import Qwen3OmniImageEncoder
from sglang_omni.models.qwen3_omni.components.preprocessor import Qwen3OmniPreprocessor
from sglang_omni.models.qwen3_omni.components.talker_executor import (
    TalkerStreamingExecutor,
)
from sglang_omni.models.qwen3_omni.components.thinker import Qwen3OmniSplitThinker
from sglang_omni.models.qwen3_omni.io import OmniEvent, ThinkerOutput
from sglang_omni.models.qwen3_omni.pipeline.engine_io import (
    apply_encoder_result,
    apply_thinker_result,
    build_encoder_request,
    build_sglang_thinker_request,
    build_thinker_request,
)
from sglang_omni.models.qwen3_omni.pipeline.merge import decode_events
from sglang_omni.models.qwen3_omni.pipeline.next_stage import (
    AUDIO_STAGE,
    CODE2WAV_STAGE,
    IMAGE_STAGE,
    TALKER_AR_STAGE,
    THINKER_STAGE,
)
from sglang_omni.models.qwen3_omni.pipeline.state_io import load_state, store_state
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)


def _event_to_dict(event: OmniEvent) -> dict[str, Any]:
    return {
        "type": event.type,
        "modality": event.modality,
        "payload": dict(event.payload),
        "is_final": bool(event.is_final),
    }


def create_preprocessing_executor(model_path: str) -> PreprocessingExecutor:
    preprocessor = Qwen3OmniPreprocessor(model_path=model_path)

    async def _preprocess(payload: StagePayload) -> StagePayload:
        return await preprocessor(payload)

    return PreprocessingExecutor(_preprocess)


def create_aggregate_executor() -> PreprocessingExecutor:
    def _identity(payload: StagePayload) -> StagePayload:
        return payload

    return PreprocessingExecutor(_identity)


def _create_encoder_executor(
    *,
    stage_name: str,
    model: torch.nn.Module,
    device: str,
    use_cache: bool = True,
    cache_size: int | None = 64,
) -> EngineExecutor:
    def _request_builder(payload: StagePayload):
        state = load_state(payload)
        return build_encoder_request(state, stage_name=stage_name)

    def _result_builder(payload: StagePayload, result: Any) -> StagePayload:
        state = load_state(payload)
        apply_encoder_result(state, stage_name=stage_name, result=result)
        return store_state(payload, state)

    engine = create_single_pass_engine(model, device=device)
    return EngineExecutor(
        engine=engine, request_builder=_request_builder, result_builder=_result_builder
    )


def create_image_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
) -> EngineExecutor:
    model = Qwen3OmniImageEncoder(model_path=model_path, device=device, dtype=dtype)
    return _create_encoder_executor(stage_name=IMAGE_STAGE, model=model, device=device)


def create_audio_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
) -> EngineExecutor:
    model = Qwen3OmniAudioEncoder(model_path=model_path, device=device, dtype=dtype)
    return _create_encoder_executor(stage_name=AUDIO_STAGE, model=model, device=device)


def create_thinker_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
    max_seq_len: int = 8192,
) -> EngineExecutor:
    model = Qwen3OmniSplitThinker(model_path=model_path, device=device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    step_counters: dict[str, int] = {}

    def _request_builder(payload: StagePayload):
        state = load_state(payload)
        step_counters.pop(payload.request_id, None)
        return build_thinker_request(state, params=payload.request.params)

    def _result_builder(payload: StagePayload, result: Any) -> StagePayload:
        state = load_state(payload)
        apply_thinker_result(state, stage_name=THINKER_STAGE, result=result)
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
        thinker_out: ThinkerOutput = {
            "output_ids": [token_id],
            "step": step,
            "is_final": False,
        }
        events = list(
            decode_events(
                thinker_out=thinker_out,
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
            "stage": THINKER_STAGE,
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


def create_sglang_thinker_executor(
    server_args: Any,
    model_path: str,
    *,
    gpu_id: int = 0,
    stream_fn=None,
    speech_enabled: bool = False,
) -> EngineExecutor:
    """Create a thinker executor backed by SGLang's ModelWorker."""
    from sglang_omni.models.qwen3_omni.components.common import load_thinker_config

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    vocab_size = getattr(tokenizer, "vocab_size", 32000)
    thinker_config = load_thinker_config(model_path)

    step_counters: dict[str, int] = {}
    enqueue_fn_holder: dict[str, Any] = {"fn": stream_fn}

    def _enqueue_stream(request_id: str, hidden, metadata=None):
        fn = enqueue_fn_holder["fn"]
        if fn is not None:
            fn(request_id, hidden, TALKER_AR_STAGE, metadata=metadata)

    def _request_builder(payload: StagePayload):
        state = load_state(payload)
        step_counters.pop(payload.request_id, None)
        data = build_sglang_thinker_request(
            state,
            params=payload.request.params,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            request_id=payload.request_id,
            thinker_config=thinker_config,
        )
        return data

    def _result_builder(payload: StagePayload, result: Any) -> StagePayload:
        state = load_state(payload)
        apply_thinker_result(state, stage_name=THINKER_STAGE, result=result)
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
        thinker_out: ThinkerOutput = {
            "output_ids": [token_id],
            "step": step,
            "is_final": False,
        }
        events = list(
            decode_events(
                thinker_out=thinker_out,
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
                    # If a final text event is found, it contains the complete text for this step.
                    # This should override any accumulated delta.
                    text_to_add = event.payload["text"]
                    break  # No need to process further events for text accumulation
                else:
                    # Accumulate text from non-final delta events
                    text_to_add += event.payload["text"]

        result: dict[str, Any] = {
            "events": [_event_to_dict(event) for event in events],
            "token_id": token_id,
            "step": step,
            "stage": THINKER_STAGE,
        }
        if text_to_add:
            result["text"] = text_to_add
        return result

    stream_adapter = (
        make_thinker_stream_adapter(stream_fn=_enqueue_stream)
        if speech_enabled
        else None
    )

    # Dual-layer capture: embed (layer 0 input) + accept_hidden_layer (layer 24 input)
    # Layer 0 captures embed output; layer 24 captures output of transformer layer 23
    # (matching HF hidden_states[24] = accept_hidden_layer)
    capture_layers = [0, 24] if speech_enabled else None

    engine = create_sglang_ar_engine(
        server_args=server_args,
        gpu_id=gpu_id,
        stream_adapter=stream_adapter,
        capture_hidden=speech_enabled,
        capture_hidden_layers=capture_layers,
    )

    if speech_enabled:
        engine._set_graph_capture_hidden_mode()

    executor = EngineExecutor(
        engine=engine,
        request_builder=_request_builder,
        result_builder=_result_builder,
        stream_builder=_stream_builder,
    )

    def _set_stream_fn(fn) -> None:
        enqueue_fn_holder["fn"] = fn

    setattr(executor, "set_stream_fn", _set_stream_fn)
    return executor


def create_sglang_thinker_executor_from_config(
    model_path: str,
    *,
    gpu_id: int = 0,
    thinker_max_seq_len: int = 8192,
    server_args_overrides: dict[str, Any] | None = None,
    speech_enabled: bool = False,
) -> EngineExecutor:
    """Create a SGLang thinker executor from JSON-serializable config args.

    This keeps pipeline config args plain dict types while still constructing
    a typed ServerArgs object internally.
    """
    server_args = build_sglang_server_args(
        model_path, context_length=thinker_max_seq_len, **(server_args_overrides or {})
    )
    return create_sglang_thinker_executor(
        server_args=server_args,
        model_path=model_path,
        gpu_id=gpu_id,
        speech_enabled=speech_enabled,
    )


def make_thinker_stream_adapter(stream_fn=None):
    """Create a Thinker stream adapter with optional hidden state side-channel.

    Args:
        stream_fn: Optional callable(request_id, hidden_tensor, metadata=None).
            Called synchronously on each decode step that produces hidden states.

    When dual-layer hidden states are available (dict with "embed" and layer index),
    the embed tensor is sent as the main data and the layer hidden is sent in metadata.
    """

    def _split_dual_layer_hidden(
        hidden: dict[str | int, torch.Tensor] | torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if isinstance(hidden, torch.Tensor):
            return hidden, None
        if not isinstance(hidden, dict):
            return None, None

        embed = hidden.get("embed")
        if embed is None:
            embed = hidden.get(0)
        if embed is None:
            embed = hidden.get("0")

        layer_hidden = None
        for key, value in hidden.items():
            if key in ("embed", 0, "0"):
                continue
            if isinstance(value, torch.Tensor):
                layer_hidden = value
                break
        return embed, layer_hidden

    def _stream_adapter(request, output):
        if request.data.req.is_chunked > 0:
            return None
        if output.data is None:
            return None  # chunked prefill final round, no token yet

        token = output.data

        # Side-channel: send hidden states to stream target (sync)
        if stream_fn is not None and output.extra is not None:
            hidden = output.extra.get("hidden_states")
            if hidden is not None:
                embed, layer_hidden = _split_dual_layer_hidden(hidden)
                if embed is not None:
                    token_id = int(token) if token is not None else None
                    stream_fn(
                        request.request_id,
                        embed,
                        metadata={
                            "layer_hidden": layer_hidden,
                            "token_id": token_id,
                        },
                    )
                elif layer_hidden is not None:
                    token_id = int(token) if token is not None else None
                    stream_fn(
                        request.request_id,
                        layer_hidden,
                        metadata={"token_id": token_id},
                    )

        return int(token) if token is not None else None

    return _stream_adapter


def make_talker_ar_stream_adapter(executor=None):
    """Talker AR stream adapter: extract fused cp_codes/cp_embeds and route.

    The talker model runs the code predictor inline (fused). The output
    processor attaches cp_codes and cp_embeds to output.extra.  This adapter
    extracts them and calls executor._handle_fused_output.
    """

    def _stream_adapter(request, output):
        if request.data.req.is_chunked > 0:
            return None
        if output.data is None:
            return None
        token = output.data
        if executor is not None and output.extra is not None:
            cp_codes = output.extra.get("cp_codes")
            cp_embeds = output.extra.get("cp_embeds")
            if cp_codes is not None and cp_embeds is not None:
                executor._handle_fused_output(request.request_id, cp_codes, cp_embeds)
        return int(token) if token is not None else None

    return _stream_adapter


def build_talker_ar_request(payload: StagePayload) -> dict:
    """Build Talker AR engine input from prefetched thinker hidden state chunks.

    Supports dual-layer chunks (embed in tensor, layer_hidden in metadata)
    and single-tensor chunks (legacy).
    """
    chunks = getattr(payload, "prefetched_chunks", None) or []
    if chunks:
        thinker_hidden_states = torch.stack([c.data for c in chunks], dim=0)
        thinker_token_ids = [
            c.metadata["token_id"]
            for c in chunks
            if c.metadata is not None and c.metadata.get("token_id") is not None
        ]
        # Check for dual-layer data
        if chunks[0].metadata and "layer_hidden" in chunks[0].metadata:
            thinker_layer_hidden = torch.stack(
                [c.metadata["layer_hidden"] for c in chunks], dim=0
            )
        else:
            thinker_layer_hidden = None
    else:
        thinker_hidden_states = torch.empty(0)
        thinker_token_ids = []
        thinker_layer_hidden = None

    result = {
        "thinker_hidden_states": thinker_hidden_states,
        "request_id": payload.request_id,
    }
    if thinker_token_ids:
        result["thinker_token_ids"] = thinker_token_ids
    if thinker_layer_hidden is not None:
        result["thinker_layer_hidden"] = thinker_layer_hidden
    return result


def create_talker_ar_executor(
    server_args: Any,
    model_path: str,
    *,
    gpu_id: int = 0,
    stream_fn=None,
    speech_enabled: bool = True,
    weight_prefix: str | None = None,
    feedback_enabled: bool = False,
    feedback_mailbox=None,
) -> EngineExecutor:
    """Talker AR executor backed by SGLang AR engine.

    The code predictor runs inline (fused MTP+talker). Talker AR routes
    directly to code2wav via _enqueue_stream.
    """
    from transformers import AutoConfig

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    talker_cfg = getattr(hf_config, "talker_config", None)
    talker_text_cfg = getattr(talker_cfg, "text_config", None)
    codec_vocab_size = (
        (
            getattr(talker_text_cfg, "vocab_size", None)
            or getattr(talker_cfg, "vocab_size", None)
            or 3072
        )
        if talker_cfg
        else 3072
    )

    enqueue_fn_holder: dict[str, Any] = {"fn": stream_fn}

    def _enqueue_stream(request_id: str, chunk_data, metadata=None):
        fn = enqueue_fn_holder["fn"]
        if fn is not None:
            fn(request_id, chunk_data, CODE2WAV_STAGE, metadata=metadata)

    # Disable radix cache for talker (decode-only, never reuses prefixes)
    server_args.disable_radix_cache = True

    engine = create_sglang_ar_engine(
        server_args=server_args,
        gpu_id=gpu_id,
        stream_adapter=None,  # wired below after executor exists
        capture_hidden=False,
        capture_hidden_layers=None,
        model_arch_override="Qwen3OmniTalker",
        weight_prefix=weight_prefix,
        feedback_enabled=True,
        feedback_mailbox=feedback_mailbox,
    )

    executor = TalkerStreamingExecutor(
        engine=engine,
        model_path=model_path,
        tokenizer=tokenizer,
        codec_vocab_size=codec_vocab_size,
        tts_bos_token_id=getattr(hf_config, "tts_bos_token_id", 151672),
        tts_eos_token_id=getattr(hf_config, "tts_eos_token_id", 151673),
        tts_pad_token_id=getattr(hf_config, "tts_pad_token_id", 151671),
        im_start_token_id=getattr(hf_config, "im_start_token_id", 151644),
        im_end_token_id=getattr(hf_config, "im_end_token_id", 151645),
        system_token_id=getattr(hf_config, "system_token_id", 8948),
        user_token_id=getattr(hf_config, "user_token_id", 872),
        assistant_token_id=getattr(hf_config, "assistant_token_id", 77091),
        accept_hidden_layer=(
            getattr(talker_cfg, "accept_hidden_layer", 24) if talker_cfg else 24
        ),
        audio_token_id=getattr(
            getattr(hf_config, "thinker_config", None), "audio_token_id", None
        ),
        image_token_id=getattr(
            getattr(hf_config, "thinker_config", None), "image_token_id", None
        ),
        video_token_id=getattr(
            getattr(hf_config, "thinker_config", None), "video_token_id", None
        ),
        speaker_map=getattr(talker_cfg, "speaker_id", None),
        enqueue_fn_holder=enqueue_fn_holder,
        thinker_config=getattr(hf_config, "thinker_config", None),
    )

    # Wire fused stream adapter: extracts cp_codes/cp_embeds from output.extra
    stream_adapter = make_talker_ar_stream_adapter(executor=executor)
    engine.scheduler._stream_adapter = stream_adapter

    # Wire code2wav enqueue so _handle_fused_output can send codes downstream
    executor._code2wav_enqueue = _enqueue_stream

    # Setup inline code prediction (unified talker+MTP forward)
    # 1. Allocate buffers  2. Capture CUDA graphs with full forward
    if speech_enabled:
        talker_model = engine.model_runner._inner_model
        if hasattr(talker_model, "setup_code_predictor_decode"):
            default_bs = [1, 2, 4, 8, 12, 16]
            talker_cuda_graph_bs = server_args.cuda_graph_bs or default_bs
            max_running = server_args.max_running_requests
            talker_cuda_graph_bs = [b for b in talker_cuda_graph_bs if b <= max_running]
            if not talker_cuda_graph_bs:
                talker_cuda_graph_bs = [1]
            talker_max_bs = max(talker_cuda_graph_bs)
            talker_model.setup_code_predictor_decode(max_batch_size=talker_max_bs)
            if os.environ.get("SGLANG_OMNI_DISABLE_TALKER_CUDA_GRAPH") != "1":
                engine.enable_cuda_graph(
                    bs=talker_cuda_graph_bs,
                    capture_hidden=False,
                )
            else:
                logger.info(
                    "Talker CUDA graph disabled via SGLANG_OMNI_DISABLE_TALKER_CUDA_GRAPH=1"
                )

    return executor


def create_talker_ar_executor_from_config(
    model_path: str,
    *,
    gpu_id: int = 0,
    talker_max_seq_len: int = 4096,
    server_args_overrides: dict[str, Any] | None = None,
    speech_enabled: bool = True,
    stream_fn=None,
    weight_prefix: str = "talker.",
    feedback_enabled: bool = False,
    feedback_mailbox=None,
) -> EngineExecutor:
    """Create a Talker AR executor from config args."""
    # Use disable_cuda_graph=True so that graphs are captured after
    # setup_code_predictor_decode (deferred capture in create_talker_ar_executor)
    talker_overrides = {"disable_cuda_graph": True}
    if server_args_overrides:
        talker_overrides.update(server_args_overrides)
    server_args = build_sglang_server_args(
        model_path, context_length=talker_max_seq_len, **talker_overrides
    )
    return create_talker_ar_executor(
        server_args=server_args,
        model_path=model_path,
        gpu_id=gpu_id,
        stream_fn=stream_fn,
        speech_enabled=speech_enabled,
        weight_prefix=weight_prefix,
        feedback_enabled=feedback_enabled,
        feedback_mailbox=feedback_mailbox,
    )


def create_decode_executor(model_path: str) -> PreprocessingExecutor:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def _decode(payload: StagePayload) -> StagePayload:
        state = load_state(payload)
        thinker_out = state.thinker_out or state.engine_outputs.get(THINKER_STAGE)
        if not isinstance(thinker_out, dict):
            thinker_out = {
                "output_ids": [],
                "step": 0,
                "is_final": True,
                "extra_model_outputs": {},
            }

        step = int(thinker_out.get("step") or len(thinker_out.get("output_ids", [])))
        events = list(
            decode_events(
                thinker_out=thinker_out,
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
            output_ids = thinker_out.get("output_ids")
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
