# SPDX-License-Identifier: Apache-2.0
"""Native SGLang stage factory for MiniCPM-o 4.5 full duplex."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

_MODEL_ARCH = "MiniCPMO"

# These settings are correctness constraints for the first native integration,
# not tuning defaults.  The duplex state machines and hidden/token alignment
# are session-local, while SGLang's overlap scheduler and CUDA graph replay can
# move work across the unit boundary.  They stay disabled until those paths
# have dedicated state-slot and graph-replay coverage.
_REQUIRED_SERVER_ARGS: dict[str, Any] = {
    "trust_remote_code": True,
    "tp_size": 1,
    "pp_size": 1,
    "max_running_requests": 1,
    "enable_streaming_session": True,
    "enable_return_hidden_states": True,
    "disable_overlap_schedule": True,
    "disable_cuda_graph": True,
    # Non-final prefill chunks intentionally discard sampled tokens in
    # SGLang, while MiniCPM-o's sampler mutates cross-unit token/TTS state.
    # Keep each unit prefill atomic until that state machine has an explicit
    # chunk rollback contract.
    "chunked_prefill_size": -1,
    "sampling_backend": "pytorch",
}


@dataclass(frozen=True)
class _FactoryDependencies:
    build_server_args: Callable[..., Any]
    create_infrastructure: Callable[..., Any]
    perception_cls: type
    tts_runtime_cls: type
    model_runner_cls: type
    output_processor_cls: type
    scheduler_cls: type


def _load_factory_dependencies() -> _FactoryDependencies:
    """Import GPU/runtime dependencies only in the stage worker process."""

    from sglang_omni.models.minicpmo_4_5.model_runner import (
        MiniCPMO45ModelRunner,
        MiniCPMO45OutputProcessor,
    )
    from sglang_omni.models.minicpmo_4_5.perception import MiniCPMO45Perception
    from sglang_omni.models.minicpmo_4_5.scheduler import MiniCPMO45Scheduler
    from sglang_omni.models.minicpmo_4_5.tts_runtime import MiniCPMO45TTSRuntime
    from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni.scheduling.sglang_backend import build_sglang_server_args

    return _FactoryDependencies(
        build_server_args=build_sglang_server_args,
        create_infrastructure=create_sglang_infrastructure,
        perception_cls=MiniCPMO45Perception,
        tts_runtime_cls=MiniCPMO45TTSRuntime,
        model_runner_cls=MiniCPMO45ModelRunner,
        output_processor_cls=MiniCPMO45OutputProcessor,
        scheduler_cls=MiniCPMO45Scheduler,
    )


def _resolve_torch_dtype(dtype: str) -> Any:
    if dtype == "auto":
        return None
    import torch

    try:
        return {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    except KeyError as exc:
        raise ValueError(
            "MiniCPM-o dtype must be one of: auto, float16, bfloat16"
        ) from exc


def _native_server_overrides(
    *,
    dtype: str,
    tp_size: int,
    server_args_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    if int(tp_size) != 1:
        raise ValueError("MiniCPM-o 4.5 native duplex currently requires tp_size=1")
    if dtype not in {"auto", "float16", "bfloat16"}:
        raise ValueError("MiniCPM-o dtype must be one of: auto, float16, bfloat16")

    overrides = dict(server_args_overrides or {})
    forbidden_owner_keys = {
        "model_path",
        "context_length",
        "dtype",
        "revision",
    }
    misplaced = sorted(forbidden_owner_keys.intersection(overrides))
    if misplaced:
        raise ValueError(
            "MiniCPM-o server_args_overrides cannot own " + ", ".join(misplaced)
        )

    for key, required in _REQUIRED_SERVER_ARGS.items():
        if key in overrides and overrides[key] != required:
            raise ValueError(
                f"MiniCPM-o native duplex requires {key}={required!r}; "
                f"got {overrides[key]!r}"
            )
        overrides[key] = required
    overrides["dtype"] = dtype
    return overrides


def create_minicpmo_duplex_scheduler(
    model_path: str,
    *,
    revision: str | None = None,
    gpu_id: int | None = None,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    dtype: str = "bfloat16",
    context_length: int = 40960,
    ref_audio_path: str | None = None,
    prompt_wav_path: str | None = None,
    max_sessions: int = 1,
    max_pending_units: int = 4,
    max_pending_commands: int = 16,
    session_ttl_s: float = 300.0,
    duplex_sampling: dict[str, Any] | None = None,
    server_args_overrides: dict[str, Any] | None = None,
    total_gpu_memory_fraction: float | None = None,
) -> Any:
    """Build the complete duplex stack inside one native SGLang stage.

    There is no nested worker, JSONL protocol, or Demo RPC here.  The main LLM
    is loaded by ``ModelWorker`` and uses SGLang's paged KV/session cache.  The
    session-owned perception, TTS and token2wav components are ordinary Python
    objects in this same stage process.
    """

    if int(tp_rank) != 0:
        raise ValueError("MiniCPM-o 4.5 native duplex currently requires tp_rank=0")
    if int(max_sessions) != 1:
        raise ValueError(
            "MiniCPM-o 4.5 native duplex currently requires max_sessions=1"
        )
    if int(max_pending_units) < 1:
        raise ValueError("max_pending_units must be >= 1")
    if int(max_pending_commands) < int(max_pending_units):
        raise ValueError("max_pending_commands must be >= max_pending_units")
    if float(session_ttl_s) <= 0:
        raise ValueError("session_ttl_s must be positive")
    if int(context_length) < 2:
        raise ValueError("context_length must be >= 2")

    sampling = dict(duplex_sampling or {})
    overrides = _native_server_overrides(
        dtype=dtype,
        tp_size=tp_size,
        server_args_overrides=server_args_overrides,
    )
    deps = _load_factory_dependencies()
    resolved_gpu_id = 0 if gpu_id is None else int(gpu_id)
    device = f"cuda:{resolved_gpu_id}"

    # Load the sidecar speech decoder before ModelWorker profiles memory.  Its
    # resident weights then reduce SGLang's automatically selected KV budget
    # instead of becoming an unaccounted post-allocation surprise.
    tts_runtime = None
    try:
        if bool(sampling.get("generate_audio", True)):
            tts_runtime = deps.tts_runtime_cls.from_pretrained(
                model_path,
                revision=revision,
                device=device,
                dtype=_resolve_torch_dtype(dtype),
                enable_float16=dtype == "float16",
                temperature=float(sampling.get("tts_temperature", 0.8)),
                repetition_penalty=float(sampling.get("tts_repetition_penalty", 1.05)),
            )

        server_args = deps.build_server_args(
            model_path,
            context_length=int(context_length),
            revision=revision,
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
        ) = deps.create_infrastructure(
            server_args,
            resolved_gpu_id,
            tp_rank=0,
            nccl_port=nccl_port,
            model_arch_override=_MODEL_ARCH,
            total_gpu_memory_fraction=total_gpu_memory_fraction,
        )

        native_model = model_worker.model_runner.model
        perception = deps.perception_cls.from_pretrained(
            model_path,
            revision=revision,
            model=native_model,
            device=device,
            trust_remote_code=True,
        )
        output_processor = deps.output_processor_cls(
            capture_hidden=True,
            model=native_model,
        )
        model_runner = deps.model_runner_cls(model_worker, output_processor)
        tokenizer = perception.tokenizer
        model_runner.set_tokenizer(tokenizer)
        # The official duplex path uses the same voice reference for the LLM
        # system prompt and Token2wav.  Keep that behavior when the recipe omits
        # both paths, while still allowing either side to be overridden.
        effective_ref_audio_path = ref_audio_path
        if effective_ref_audio_path is None and tts_runtime is not None:
            effective_ref_audio_path = tts_runtime.default_prompt_wav_path

        scheduler = deps.scheduler_cls(
            tp_worker=model_worker,
            tree_cache=tree_cache,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            server_args=server_args,
            model_config=model_config,
            prefill_manager=prefill_mgr,
            decode_manager=decode_mgr,
            model_runner=model_runner,
            perception=perception,
            tokenizer=tokenizer,
            tts_runtime=tts_runtime,
            ref_audio_path=effective_ref_audio_path,
            prompt_wav_path=prompt_wav_path,
            duplex_sampling=sampling,
            max_sessions=max_sessions,
            max_pending_units=max_pending_units,
            max_pending_commands=max_pending_commands,
            session_ttl_s=session_ttl_s,
            enable_overlap=False,
            enable_async_decode=False,
        )
        logger.info(
            "Started native MiniCPM-o duplex stage on %s with SGLang paged KV",
            device,
        )
        return scheduler
    except BaseException:
        if tts_runtime is not None:
            try:
                tts_runtime.close()
            except Exception:
                logger.exception("Failed to close MiniCPM-o TTS runtime after startup")
        raise


__all__ = ["create_minicpmo_duplex_scheduler"]
