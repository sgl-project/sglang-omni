# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the MOSS-TTS Delay pipeline."""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from typing import Any

import torch

from sglang_omni.models.moss_tts.codec import split_moss_audio_segments
from sglang_omni.models.moss_tts.payload_types import (
    MossTTSState,
    moss_tts_special_token_defaults,
)
from sglang_omni.models.moss_tts.request_builders import (
    cleanup_prepared_moss_tts_request,
    make_moss_tts_scheduler_adapters,
    preprocess_moss_tts_payload,
    set_moss_tts_preprocessing_context,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

_MOSS_TTS_INSTALL_HINT = (
    "MOSS-TTS support requires the upstream custom Transformers code. "
    "Launch with trust_remote_code=True and make sure the checkpoint can load "
    "OpenMOSS-Team/MOSS-Audio-Tokenizer."
)


def load_state(payload: StagePayload) -> MossTTSState:
    return MossTTSState.from_dict(payload.data)


def store_state(payload: StagePayload, state: MossTTSState) -> StagePayload:
    payload.data = state.to_dict()
    return payload


def _resolve_checkpoint(checkpoint: str) -> str:
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(checkpoint)


def _torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    return getattr(torch, dtype) if isinstance(dtype, str) else dtype


@contextmanager
def _moss_transformers_processor_compat():
    """Scope load-time Transformers API-drift patches to the MOSS processor load
    (renamed PreTrainedConfig, the audio-tokenizer modality mapping, and
    PreTrainedAudioTokenizerBase), restoring the globals afterwards so they don't
    leak into unrelated models."""
    import transformers.configuration_utils as configuration_utils
    from transformers import PreTrainedModel, processing_utils

    missing = object()
    undo: list[tuple[str, Any, str, Any]] = []

    def patch_attr(obj: Any, name: str, value: Any) -> None:
        undo.append(("attr", obj, name, getattr(obj, name, missing)))
        setattr(obj, name, value)

    def patch_item(mapping: dict, key: str, value: Any) -> None:
        undo.append(("item", mapping, key, mapping.get(key, missing)))
        mapping[key] = value

    try:
        if not hasattr(configuration_utils, "PreTrainedConfig"):
            patch_attr(
                configuration_utils,
                "PreTrainedConfig",
                configuration_utils.PretrainedConfig,
            )
        auto_mapping = getattr(processing_utils, "AUTO_TO_BASE_CLASS_MAPPING", None)
        if isinstance(auto_mapping, dict):
            if "AutoModel" not in auto_mapping:
                patch_item(auto_mapping, "AutoModel", "PreTrainedModel")
            if not hasattr(processing_utils, "MODALITY_TO_BASE_CLASS_MAPPING"):
                patch_attr(
                    processing_utils, "MODALITY_TO_BASE_CLASS_MAPPING", auto_mapping
                )
        if hasattr(processing_utils, "PreTrainedAudioTokenizerBase"):
            patch_attr(
                processing_utils, "PreTrainedAudioTokenizerBase", PreTrainedModel
            )
        yield
    finally:
        for kind, obj, key, old in reversed(undo):
            if kind == "attr":
                if old is missing:
                    if hasattr(obj, key):
                        delattr(obj, key)
                else:
                    setattr(obj, key, old)
            elif old is missing:
                obj.pop(key, None)
            else:
                obj[key] = old


def _load_moss_processor_class(checkpoint_dir: str) -> type:
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    processor_config_path = os.path.join(checkpoint_dir, "processor_config.json")
    with open(processor_config_path, encoding="utf-8") as f:
        processor_config = json.load(f)

    class_ref = (processor_config.get("auto_map") or {}).get("AutoProcessor")
    if not class_ref:
        raise RuntimeError("MOSS-TTS processor_config.json lacks AutoProcessor map")

    processor_cls = get_class_from_dynamic_module(class_ref, checkpoint_dir)
    if list(getattr(processor_cls, "attributes", [])) == [
        "feature_extractor",
        "tokenizer",
    ]:
        processor_cls.attributes = ["tokenizer"]
    return processor_cls


def _normalize_moss_processor_config(processor: Any) -> None:
    model_config = getattr(processor, "model_config", None)
    if model_config is None:
        return
    audio_vocab_size = int(getattr(model_config, "audio_vocab_size", 1024) or 1024)
    for attr, default in moss_tts_special_token_defaults(audio_vocab_size):
        if getattr(model_config, attr, None) is None:
            setattr(model_config, attr, default)


def _load_moss_processor(
    model_path: str,
    *,
    device: str = "cpu",
    dtype: str | torch.dtype = "float32",
) -> Any:
    checkpoint_dir = _resolve_checkpoint(model_path)
    logger.info("Loading MOSS-TTS processor from %s on %s", checkpoint_dir, device)
    try:
        with _moss_transformers_processor_compat():
            processor_cls = _load_moss_processor_class(checkpoint_dir)
            processor = processor_cls.from_pretrained(
                checkpoint_dir,
                trust_remote_code=True,
            )
    except Exception as exc:
        raise RuntimeError(_MOSS_TTS_INSTALL_HINT) from exc

    _normalize_moss_processor_config(processor)
    audio_tokenizer = getattr(processor, "audio_tokenizer", None)
    if audio_tokenizer is not None:
        if hasattr(audio_tokenizer, "eval"):
            audio_tokenizer.eval()
        if hasattr(audio_tokenizer, "to"):
            kwargs: dict[str, Any] = {"device": device}
            if device != "cpu":
                kwargs["dtype"] = _torch_dtype(dtype)
            audio_tokenizer.to(**kwargs)
    return processor


def _build_usage(state: MossTTSState) -> dict[str, Any] | None:
    if not (state.prompt_tokens or state.completion_tokens or state.engine_time_s):
        return None
    usage = {
        "prompt_tokens": int(state.prompt_tokens),
        "completion_tokens": int(state.completion_tokens),
        "total_tokens": int(state.prompt_tokens + state.completion_tokens),
    }
    if state.engine_time_s:
        usage["engine_time_s"] = round(float(state.engine_time_s), 6)
    return usage


def create_preprocessing_executor(
    model_path: str, *, max_concurrency: int = 8
) -> SimpleScheduler:
    processor = _load_moss_processor(model_path, device="cpu", dtype="float32")
    set_moss_tts_preprocessing_context(processor=processor)
    # Preprocessing is CPU-heavy: every request tokenizes text and encodes the
    # reference audio through the MOSS audio tokenizer. Serial execution
    # (max_concurrency=1) lets the codec encode dominate wall-clock and starves
    # the AR engine to batch size 1 (the dominant RTF cost). Run several in
    # parallel — threads release the GIL during the torch codec forward — so the
    # AR OmniScheduler receives a steady, batchable request stream. Mirrors the
    # fishaudio_s2_pro preprocessing stage, which encodes references the same way.
    return SimpleScheduler(
        preprocess_moss_tts_payload,
        abort_callback=cleanup_prepared_moss_tts_request,
        max_concurrency=max_concurrency,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, Any] | None = None,
) -> Any:
    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
    from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import (
        SGLangOutputProcessor,
        build_sglang_server_args,
    )

    checkpoint_dir = _resolve_checkpoint(model_path)
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    overrides: dict[str, Any] = {
        "dtype": dtype,
        "cuda_graph_bs": [1, 2, 4, 8, 16],
        "cuda_graph_max_bs": 16,
        "disable_cuda_graph": False,
        "disable_overlap_schedule": True,
        "enable_torch_compile": False,
        "max_prefill_tokens": 8192,
        "max_running_requests": 16,
        "sampling_backend": "pytorch",
        "torch_compile_max_bs": 16,
        "trust_remote_code": True,
    }
    if server_args_overrides:
        overrides.update(server_args_overrides)

    server_args = build_sglang_server_args(
        checkpoint_dir,
        context_length=8192,
        **overrides,
    )

    want_cuda_graph = not bool(getattr(server_args, "disable_cuda_graph", False))
    if want_cuda_graph:
        server_args.disable_cuda_graph = True

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure(
        server_args,
        gpu_id,
        model_arch_override="MossTTSDelaySGLangModel",
    )

    if want_cuda_graph:
        server_args.disable_cuda_graph = False

    model = model_worker.model_runner.model
    if want_cuda_graph:
        model_worker.model_runner.init_device_graphs()

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model,
    )
    request_builder, result_adapter = make_moss_tts_scheduler_adapters(model=model)

    return OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=MossTTSModelRunner(model_worker, output_proc),
        request_builder=request_builder,
        result_adapter=result_adapter,
        abort_callback=cleanup_prepared_moss_tts_request,
    )


def create_tts_engine_executor(*args, **kwargs) -> Any:
    return create_sglang_tts_engine_executor(*args, **kwargs)


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "float32",
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
) -> SimpleScheduler:
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    processor = _load_moss_processor(model_path, device=device, dtype=dtype)

    def _prepare_vocoder_item(
        payload: StagePayload,
    ) -> tuple[MossTTSState, torch.Tensor]:
        state = load_state(payload)
        if state.delayed_audio_codes is None:
            raise RuntimeError("MOSS-TTS vocoder requires delayed_audio_codes")
        delayed_codes = torch.as_tensor(state.delayed_audio_codes, dtype=torch.long)
        if delayed_codes.numel() == 0:
            raise RuntimeError("MOSS-TTS generated no delayed audio codes")
        return state, delayed_codes

    def _decode_audio(
        state: MossTTSState,
        delayed_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        delayed_codes = delayed_codes.to(device=device, dtype=torch.long)
        audio_pad_code = int(
            getattr(
                getattr(processor, "model_config", None),
                "audio_pad_code",
                1024,
            )
        )
        segments = split_moss_audio_segments(
            delayed_codes,
            audio_pad_code=audio_pad_code,
            assistant_start_length=int(state.assistant_start_length),
        )
        decoded = []
        for segment in segments:
            decoded.extend(processor.decode_audio_codes([segment]))
        if not decoded:
            raise RuntimeError("MOSS-TTS vocoder decoded no audio segments")
        waveforms = [
            torch.as_tensor(wav).detach().reshape(-1).to("cpu") for wav in decoded
        ]
        waveform = torch.cat(waveforms, dim=0)
        sample_rate = int(
            getattr(getattr(processor, "model_config", None), "sampling_rate", 0)
            or getattr(
                getattr(getattr(processor, "audio_tokenizer", None), "config", None),
                "sampling_rate",
                0,
            )
            or state.sample_rate
            or 24000
        )
        return waveform, sample_rate

    def _store_vocoder_result(
        payload: StagePayload,
        state: MossTTSState,
        wav: torch.Tensor,
        sample_rate: int,
    ) -> StagePayload:
        audio_payload = audio_waveform_payload(wav, source_hint="MOSS-TTS")
        state.delayed_audio_codes = None
        state.sample_rate = int(sample_rate)
        payload = store_state(payload, state)
        payload.data.update(audio_payload)
        payload.data["sample_rate"] = state.sample_rate
        payload.data["modality"] = "audio"
        usage = _build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload

    def _vocode(payload: StagePayload) -> StagePayload:
        state, delayed_codes = _prepare_vocoder_item(payload)
        wav, sample_rate = _decode_audio(state, delayed_codes)
        return _store_vocoder_result(payload, state, wav, sample_rate)

    def _vocode_batch(payloads: list[StagePayload]) -> list[StagePayload]:
        return [_vocode(payload) for payload in payloads]

    return SimpleScheduler(
        _vocode,
        batch_compute_fn=_vocode_batch,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
