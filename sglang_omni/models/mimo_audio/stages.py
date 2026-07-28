# SPDX-License-Identifier: Apache-2.0
"""Stage factories for MiMo-Audio input-to-text inference."""

from __future__ import annotations

import hashlib
from typing import Any

import torch
from huggingface_hub import snapshot_download
from sglang.srt.managers.mm_utils import init_mm_embedding_cache

from sglang_omni.models.mimo_audio.audio_tokenizer import MiMoAudioTokenizerEncoder
from sglang_omni.models.mimo_audio.preprocessor import (
    MiMoAudioPreprocessor,
    MiMoAudioTokenizerSettings,
)
from sglang_omni.models.mimo_audio.request_builders import (
    audio_source_from_payload,
    make_mimo_scheduler_adapters,
    validate_text_only_request,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


def _torch_dtype(value: str) -> torch.dtype:
    normalized = value.strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported MiMo dtype {value!r}")


def create_audio_tokenizer_executor(
    model_path: str,
    *,
    tokenizer_path: str,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
) -> SimpleScheduler:
    del model_path
    snapshot = snapshot_download(
        tokenizer_path,
        allow_patterns=["config.json", "model.safetensors"],
    )
    encoder = MiMoAudioTokenizerEncoder.from_checkpoint(
        snapshot,
        device=device,
        dtype=_torch_dtype(dtype),
    )
    preprocessor = MiMoAudioPreprocessor(
        MiMoAudioTokenizerSettings.from_config(encoder.config),
        device=device,
    )

    def _encode(payload: StagePayload) -> StagePayload:
        validate_text_only_request(payload)
        waveform = preprocessor.load_waveform(audio_source_from_payload(payload))
        duration_s = waveform.numel() / preprocessor.settings.sampling_rate
        log_mel = preprocessor.waveform_to_log_mel(waveform)
        codes = preprocessor.encode_log_mel(log_mel, encoder)
        payload.data = {
            "audio_codes": codes,
            "audio_duration_s": float(duration_s),
            "audio_fingerprint": hashlib.sha256(
                waveform.detach().cpu().numpy().tobytes()
            ).hexdigest(),
            "mel_shape": tuple(log_mel.shape),
            "code_shape": tuple(codes.shape),
            "modality": "audio_codes",
        }
        return payload

    return SimpleScheduler(_encode)


def create_thinker_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    max_running_requests: int = 1,
    max_new_tokens: int = 256,
    mem_fraction_static: float = 0.80,
    mm_embedding_cache_size_bytes: int = 0,
    server_args_overrides: dict[str, Any] | None = None,
):
    from transformers import AutoTokenizer

    from sglang_omni.model_runner.base import ModelRunner
    from sglang_omni.scheduling.bootstrap import (
        create_sglang_infrastructure_defer_cuda_graph,
    )
    from sglang_omni.scheduling.generation_batch_policy import (
        build_generation_batch_overrides,
        validate_generation_batch_policy,
    )
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import (
        SGLangOutputProcessor,
        build_sglang_server_args,
    )

    gpu_id = int(device.split(":")[-1]) if ":" in device else 0
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    overrides = build_generation_batch_overrides(
        max_running_requests=max_running_requests,
        server_args_overrides=server_args_overrides,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
        mem_fraction_static=mem_fraction_static,
        max_prefill_tokens=8192,
        chunked_prefill_size=8192,
        sampling_backend="pytorch",
        dtype=dtype,
    )
    server_args = build_sglang_server_args(
        model_path,
        context_length=8192,
        **overrides,
    )
    validate_generation_batch_policy(model_name="MiMo-Audio", server_args=server_args)
    (
        _,
        (
            model_worker,
            tree_cache,
            req_to_token_pool,
            token_to_kv_pool_allocator,
            prefill_mgr,
            decode_mgr,
            model_config,
        ),
    ) = create_sglang_infrastructure_defer_cuda_graph(
        server_args,
        gpu_id,
        model_arch_override="MiMoAudioModel",
    )
    init_mm_embedding_cache(mm_embedding_cache_size_bytes)
    output_processor = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model_worker.model_runner.model,
    )
    request_builder, result_adapter = make_mimo_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )
    return OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=ModelRunner(model_worker, output_processor),
        request_builder=request_builder,
        result_adapter=result_adapter,
    )


__all__ = ["create_audio_tokenizer_executor", "create_thinker_executor"]
