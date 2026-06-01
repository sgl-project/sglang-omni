# SPDX-License-Identifier: Apache-2.0
"""Stage executors for Ming-Omni pipelines."""

from __future__ import annotations

from typing import Any

from sglang_omni.engines.ar.sglang_backend.server_args_builder import (
    apply_encoder_mem_reserve,
    build_sglang_server_args,
)
from sglang_omni.engines.omni import create_sglang_ar_engine, create_single_pass_engine
from sglang_omni.executors import EngineExecutor, PreprocessingExecutor
from sglang_omni.models.ming_omni.components.audio_encoder import MingAudioEncoder
from sglang_omni.models.ming_omni.components.common import (
    load_ming_config,
    load_ming_tokenizer,
)
from sglang_omni.models.ming_omni.components.preprocessor import MingPreprocessor
from sglang_omni.models.ming_omni.components.talker_executor import MingTalkerExecutor
from sglang_omni.models.ming_omni.io import OmniEvent, ThinkerOutput
from sglang_omni.models.ming_omni.pipeline.engine_io import (
    apply_encoder_result,
    apply_thinker_result,
    build_encoder_request,
    build_sglang_thinker_request,
)
from sglang_omni.models.ming_omni.pipeline.merge import decode_events
from sglang_omni.models.ming_omni.pipeline.next_stage import (
    AUDIO_STAGE,
    IMAGE_STAGE,
    THINKER_STAGE,
)
from sglang_omni.models.ming_omni.pipeline.state_io import load_state, store_state
from sglang_omni.proto import StagePayload
from sglang_omni.utils.misc import avail_gpu_mem


def _event_to_dict(event: OmniEvent) -> dict[str, Any]:
    return {
        "type": event.type,
        "modality": event.modality,
        "payload": dict(event.payload),
        "is_final": bool(event.is_final),
    }


def create_preprocessing_executor(model_path: str) -> PreprocessingExecutor:
    preprocessor = MingPreprocessor(model_path=model_path)

    async def _preprocess(payload: StagePayload) -> StagePayload:
        return await preprocessor(payload)

    return PreprocessingExecutor(_preprocess)


def create_aggregate_executor() -> PreprocessingExecutor:
    def _identity(payload: StagePayload) -> StagePayload:
        return payload

    return PreprocessingExecutor(_identity)


def create_audio_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
) -> EngineExecutor:
    model = MingAudioEncoder(model_path=model_path, device=device, dtype=dtype)

    def _request_builder(payload: StagePayload):
        state = load_state(payload)
        return build_encoder_request(state, stage_name=AUDIO_STAGE)

    def _result_builder(payload: StagePayload, result: Any) -> StagePayload:
        state = load_state(payload)
        apply_encoder_result(state, stage_name=AUDIO_STAGE, result=result)
        return store_state(payload, state)

    engine = create_single_pass_engine(
        model,
        device=device,
        use_cache=True,
        cache_size=64,
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
    """Create an image encoder executor for the Ming-Omni vision pipeline."""
    from sglang_omni.models.ming_omni.components.image_encoder import MingImageEncoder

    model = MingImageEncoder(model_path=model_path, device=device, dtype=dtype)

    def _request_builder(payload: StagePayload):
        state = load_state(payload)
        return build_encoder_request(state, stage_name=IMAGE_STAGE)

    def _result_builder(payload: StagePayload, result: Any) -> StagePayload:
        state = load_state(payload)
        apply_encoder_result(state, stage_name=IMAGE_STAGE, result=result)
        return store_state(payload, state)

    engine = create_single_pass_engine(
        model,
        device=device,
        use_cache=True,
        cache_size=64,
    )
    return EngineExecutor(
        engine=engine, request_builder=_request_builder, result_builder=_result_builder
    )


def create_sglang_thinker_executor(
    server_args: Any,
    model_path: str,
    *,
    gpu_id: int = 0,
) -> EngineExecutor:
    """Create a thinker executor backed by SGLang's ModelWorker."""
    tokenizer = load_ming_tokenizer(model_path)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    # Use config vocab_size (157184) instead of tokenizer.vocab_size (156891)
    # because Ming-Omni has special tokens beyond the tokenizer's base vocab.
    config = load_ming_config(model_path)
    llm_cfg = getattr(config, "llm_config", config)
    vocab_size = getattr(llm_cfg, "vocab_size", None) or getattr(
        tokenizer, "vocab_size", 32000
    )

    step_counters: dict[str, int] = {}

    def _request_builder(payload: StagePayload):
        state = load_state(payload)
        step_counters.pop(payload.request_id, None)
        data = build_sglang_thinker_request(
            state,
            params=payload.request.params,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            request_id=payload.request_id,
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
                    text_to_add = event.payload["text"]
                    break
                else:
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

    engine = create_sglang_ar_engine(
        server_args=server_args,
        gpu_id=gpu_id,
        model_arch_override="BailingMoeV2ForCausalLM",
    )

    return EngineExecutor(
        engine=engine,
        request_builder=_request_builder,
        result_builder=_result_builder,
        stream_builder=_stream_builder,
    )


_ming_config_registered = False

# Tokenizer fallback repo — same vocab as Ming-flash-omni-2.0
_TOKENIZER_FALLBACK_REPO = "inclusionAI/Ming-flash-omni-Preview"


def _ensure_ming_config_registered(model_path: str = "inclusionAI/Ming-flash-omni-2.0"):
    """Ensure ``AutoConfig`` can resolve Ming's ``bailingmm_moe_v2_lite`` model type."""
    global _ming_config_registered
    if _ming_config_registered:
        return
    _ming_config_registered = True

    import os
    import shutil

    from transformers import AutoConfig

    from sglang_omni.models.ming_omni.thinker import BailingMM2Config

    try:
        AutoConfig.register("bailingmm_moe_v2_lite", BailingMM2Config)
    except ValueError:
        pass

    # Patch HF cache with missing files
    try:
        from huggingface_hub import hf_hub_download, snapshot_download

        snapshot_dir = snapshot_download(model_path)

        # 1. Write configuration_bailingmm2.py shim
        shim_path = os.path.join(snapshot_dir, "configuration_bailingmm2.py")
        if not os.path.exists(shim_path):
            with open(shim_path, "w") as f:
                f.write(
                    "from sglang_omni.models.ming_omni.thinker "
                    "import BailingMM2Config\n"
                )

        # 2. Copy tokenizer files from fallback repo if missing
        for fname in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ):
            dst = os.path.join(snapshot_dir, fname)
            if not os.path.exists(dst):
                src = hf_hub_download(_TOKENIZER_FALLBACK_REPO, fname)
                shutil.copy2(src, dst)
    except Exception:
        pass  # Best-effort


def _resolve_local_model_path(model_path: str) -> str:
    """Resolve HF repo ID to local snapshot path (with patched files).

    If model_path is already a local directory, return as-is.
    Otherwise download the HF snapshot and return the local path.
    """
    import os

    if os.path.isdir(model_path):
        return model_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_path)
    except Exception:
        return model_path


def create_sglang_thinker_executor_from_config(
    model_path: str,
    *,
    gpu_id: int = 0,
    thinker_max_seq_len: int = 8192,
    server_args_overrides: dict[str, Any] | None = None,
) -> EngineExecutor:
    """Create a SGLang thinker executor from JSON-serializable config args."""
    import logging as _log

    _log.getLogger(__name__).info(
        f"create_sglang_thinker_executor_from_config: "
        f"server_args_overrides={server_args_overrides}"
    )
    _ensure_ming_config_registered(model_path)
    # Use local snapshot path so AutoConfig finds our patched files
    local_path = _resolve_local_model_path(model_path)

    overrides = dict(server_args_overrides or {})
    tp_size = overrides.get("tp_size", 1)

    if tp_size > 1:
        import json as _json

        config = load_ming_config(model_path)
        llm_cfg = getattr(config, "llm_config", config)
        model_override = _json.dumps(
            {
                "architectures": ["BailingMoeV2ForCausalLM"],
                "num_attention_heads": llm_cfg.num_attention_heads,
                "num_key_value_heads": llm_cfg.num_key_value_heads,
                "hidden_size": llm_cfg.hidden_size,
                "num_hidden_layers": llm_cfg.num_hidden_layers,
                "vocab_size": llm_cfg.vocab_size,
            }
        )
        overrides["json_model_override_args"] = model_override
        overrides.setdefault("base_gpu_id", gpu_id)

    pre_load_avail_mem = avail_gpu_mem(gpu_id)
    server_args = build_sglang_server_args(
        local_path,
        context_length=thinker_max_seq_len,
        **overrides,
    )
    if "mem_fraction_static" not in overrides:
        apply_encoder_mem_reserve(server_args, 0.05)
    pre_load_mem = (
        f", pre_load_avail_mem={pre_load_avail_mem:.2f} GB"
        if pre_load_avail_mem is not None
        else ""
    )
    _log.getLogger(__name__).info(
        f"ServerArgs: cpu_offload_gb={server_args.cpu_offload_gb}, "
        f"mem_fraction_static={server_args.mem_fraction_static}"
        f"{pre_load_mem}"
    )
    executor = create_sglang_thinker_executor(
        server_args=server_args,
        model_path=local_path,
        gpu_id=gpu_id,
    )
    post_load_avail_mem = avail_gpu_mem(gpu_id)
    post_load_mem = (
        f" post_load_avail_mem={post_load_avail_mem:.2f} GB"
        if post_load_avail_mem is not None
        else ""
    )
    _log.getLogger(__name__).info(
        f"Ming thinker SGLang executor initialized: gpu_id={gpu_id}{post_load_mem}"
    )
    return executor


def create_talker_executor(
    model_path: str,
    *,
    talker_model_path: str | None = None,
    device: str = "cuda",
    voice: str = "DB30",
) -> MingTalkerExecutor:
    """Create the Ming TTS talker executor.

    The talker is a self-contained MingOmniTalker with its own LLM + CFM + AudioVAE,
    wrapped as an Executor for the pipeline.
    """
    # Resolve HF repo ID to local snapshot path so that
    # Path(model_path) / "talker" yields a real filesystem path.
    local_path = _resolve_local_model_path(model_path)
    return MingTalkerExecutor(
        model_path=local_path,
        talker_model_path=talker_model_path,
        device=device,
        voice=voice,
    )


def create_decode_executor(model_path: str) -> PreprocessingExecutor:
    tokenizer = load_ming_tokenizer(model_path)
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
