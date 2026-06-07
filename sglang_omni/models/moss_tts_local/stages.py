# SPDX-License-Identifier: Apache-2.0
"""Stage factories for MOSS-TTS Local.

Preprocessing is shared with Delay; the AR engine and vocoder are Local
(depth-transformer decode, time-synchronous codes with no de-delay).
"""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.models.moss_tts.request_builders import (
    cleanup_prepared_moss_tts_request,
)
from sglang_omni.models.moss_tts.stages import (
    _build_usage,
    _load_moss_processor,
    _resolve_checkpoint,
    load_state,
    store_state,
)
from sglang_omni.models.moss_tts_local.request_builders import (
    make_moss_tts_local_scheduler_adapters,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, Any] | None = None,
) -> Any:
    from sglang_omni.models.moss_tts_local.model_runner import MossTTSLocalModelRunner
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
        # Backbone under CUDA graph; the depth predictor runs eager in the runner.
        "disable_cuda_graph": False,
        "disable_overlap_schedule": True,
        "enable_torch_compile": False,
        "max_prefill_tokens": 8192,
        "max_running_requests": 16,
        "sampling_backend": "pytorch",
        "trust_remote_code": True,
    }
    if server_args_overrides:
        overrides.update(server_args_overrides)

    server_args = build_sglang_server_args(
        checkpoint_dir,
        context_length=8192,
        **overrides,
    )

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
        model_arch_override="MossTTSLocalSGLangModel",
    )

    model = model_worker.model_runner.model
    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model,
    )
    request_builder, result_adapter = make_moss_tts_local_scheduler_adapters(
        model=model
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
        model_runner=MossTTSLocalModelRunner(model_worker, output_proc),
        request_builder=request_builder,
        result_adapter=result_adapter,
        abort_callback=cleanup_prepared_moss_tts_request,
    )


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

    def _decode(payload: StagePayload) -> StagePayload:
        state = load_state(payload)
        if state.delayed_audio_codes is None:
            raise RuntimeError("MOSS-TTS Local vocoder requires audio codes")
        codes = torch.as_tensor(state.delayed_audio_codes, dtype=torch.long)
        if codes.numel() == 0:
            raise RuntimeError("MOSS-TTS Local generated no audio codes")
        # Time-synchronous: decode the (T, n_vq) code grid directly (no de-delay).
        decoded = processor.decode_audio_codes([codes.to(device)])
        if not decoded:
            raise RuntimeError("MOSS-TTS Local vocoder decoded no audio")
        waveforms = [
            torch.as_tensor(wav).detach().reshape(-1).to("cpu") for wav in decoded
        ]
        waveform = torch.cat(waveforms, dim=0)
        sample_rate = int(
            getattr(getattr(processor, "model_config", None), "sampling_rate", 0)
            or state.sample_rate
            or 24000
        )
        audio_payload = audio_waveform_payload(waveform, source_hint="MOSS-TTS-Local")
        state.delayed_audio_codes = None
        state.sample_rate = sample_rate
        payload = store_state(payload, state)
        payload.data.update(audio_payload)
        payload.data["sample_rate"] = sample_rate
        payload.data["modality"] = "audio"
        usage = _build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload

    def _decode_batch(payloads: list[StagePayload]) -> list[StagePayload]:
        return [_decode(p) for p in payloads]

    return SimpleScheduler(
        _decode,
        batch_compute_fn=_decode_batch,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
