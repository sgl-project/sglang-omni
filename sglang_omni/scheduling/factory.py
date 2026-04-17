# SPDX-License-Identifier: Apache-2.0
"""Factory functions for creating OmniScheduler instances.

Creates the SGLang infrastructure (ModelWorker, PrefillManager, DecodeManager,
tree cache) and assembles it into an OmniScheduler with the appropriate
ModelRunner and request builder.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _create_sglang_infrastructure(
    server_args: Any,
    gpu_id: int,
    *,
    model_arch_override: str | None = None,
    weight_prefix: str | None = None,
    capture_hidden_layers: list[int] | None = None,
):
    """Create SGLang ModelWorker + memory pools + tree cache + prefill/decode managers.

    Returns (model_worker, tree_cache, req_to_token_pool, token_to_kv_pool_allocator,
             prefill_manager, decode_manager, model_config).
    """
    from sglang_omni.model_runner.model_worker import ModelWorker, ModelWorkerConfig
    from sglang_omni.scheduling.sglang_backend import (
        DecodeManager,
        PrefillManager,
        create_tree_cache,
    )

    model_worker = ModelWorker(
        config=ModelWorkerConfig(
            model_arch_override=model_arch_override,
            weight_prefix=weight_prefix,
        ),
        server_args=server_args,
        gpu_id=gpu_id,
    )

    if capture_hidden_layers:
        from sglang_omni.model_runner._hidden_capture import (
            install_hidden_capture_hooks,
        )

        model = model_worker.model_runner.model
        install_hidden_capture_hooks(model, capture_hidden_layers)

    req_to_token_pool, token_to_kv_pool_allocator = model_worker.get_memory_pool()

    tree_cache = create_tree_cache(
        server_args,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        server_args.page_size,
    )

    enable_overlap = not getattr(server_args, "disable_overlap_schedule", False)

    prefill_mgr = PrefillManager(
        page_size=server_args.page_size,
        chunked_prefill_size=server_args.chunked_prefill_size,
        max_prefill_tokens=server_args.max_prefill_tokens,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=model_worker.model_config,
        enable_overlap=enable_overlap,
    )

    decode_mgr = DecodeManager(
        server_args=server_args,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        on_retract=lambda req: prefill_mgr.add_one_request(req),
    )

    return (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_worker.model_config,
    )


# ---------------------------------------------------------------------------
# Scheduler factories
# ---------------------------------------------------------------------------


def create_thinker_scheduler(
    server_args: Any,
    gpu_id: int = 0,
    *,
    speech_enabled: bool = False,
) -> "OmniScheduler":
    """Create OmniScheduler for thinker (standard AR + optional hidden state streaming)."""
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
    from sglang_omni.models.qwen3_omni.request_builders import (
        make_thinker_scheduler_adapters,
        make_thinker_stream_output_builder,
    )
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import SGLangOutputProcessor

    capture_hidden_layers = [0, 24] if speech_enabled else None
    capture_hidden = speech_enabled
    want_cuda_graph = not bool(getattr(server_args, "disable_cuda_graph", False))
    if want_cuda_graph and capture_hidden:
        server_args.enable_return_hidden_states = True
        server_args.disable_cuda_graph = True

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = _create_sglang_infrastructure(
        server_args,
        gpu_id,
        capture_hidden_layers=capture_hidden_layers,
    )

    if want_cuda_graph:
        server_args.disable_cuda_graph = False
        model_worker.model_runner.init_device_graphs()

    output_proc = SGLangOutputProcessor(
        capture_hidden=capture_hidden,
        capture_hidden_layers=capture_hidden_layers,
        model=model_worker.model_runner.model if capture_hidden_layers else None,
    )

    model_runner = ThinkerModelRunner(model_worker, output_proc)

    tokenizer = get_tokenizer(
        model_config.model_path,
        trust_remote_code=True,
    )
    thinker_config = model_config.hf_config.thinker_config
    request_builder, result_adapter = make_thinker_scheduler_adapters(
        tokenizer=tokenizer,
        vocab_size=model_config.vocab_size,
        thinker_config=thinker_config,
    )
    stream_output_builder = make_thinker_stream_output_builder()

    scheduler = OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=model_runner,
        request_builder=request_builder,
        result_adapter=result_adapter,
        stream_output_builder=stream_output_builder,
    )

    return scheduler


def create_talker_scheduler(
    server_args: Any,
    gpu_id: int = 0,
    *,
    weight_prefix: str = "talker.",
    speech_enabled: bool = True,
    feedback_enabled: bool = True,
) -> "OmniScheduler":
    """Create OmniScheduler for talker (feedback AR + fused MTP)."""
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    from sglang_omni.models.qwen3_omni.request_builders import (
        make_talker_scheduler_adapters,
    )
    from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import SGLangOutputProcessor

    want_cuda_graph = not bool(getattr(server_args, "disable_cuda_graph", False))
    if feedback_enabled:
        server_args.disable_overlap_schedule = True
        if want_cuda_graph:
            server_args.disable_cuda_graph = True
        server_args.moe_runner_backend = "flashinfer_cutlass"
    server_args.disable_radix_cache = True
    server_args.chunked_prefill_size = 0

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = _create_sglang_infrastructure(
        server_args,
        gpu_id,
        model_arch_override="Qwen3OmniTalker",
        weight_prefix=weight_prefix,
    )
    if hasattr(model_worker.model_runner, "sampler"):
        model_worker.model_runner.model._sampler = model_worker.model_runner.sampler
    if want_cuda_graph:
        server_args.disable_cuda_graph = False
        model_worker.model_runner.init_device_graphs()

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model_worker.model_runner.model,
    )

    tokenizer = get_tokenizer(
        model_config.model_path,
        trust_remote_code=True,
    )
    root_config = model_config.hf_config
    thinker_config = root_config.thinker_config
    talker_config = root_config.talker_config
    codec_vocab_size = talker_config.text_config.vocab_size
    (
        request_builder,
        result_adapter,
        stream_chunk_handler,
        stream_done_handler,
    ) = make_talker_scheduler_adapters(
        tokenizer=tokenizer,
        codec_vocab_size=codec_vocab_size,
        model=model_worker.model_runner.model,
        model_path=model_config.model_path,
        thinker_config=thinker_config,
        required_aux_hidden_key=talker_config.accept_hidden_layer,
        codec_bos_id=talker_config.codec_bos_id,
        codec_eos_id=talker_config.codec_eos_token_id,
        codec_nothink_id=talker_config.codec_nothink_id,
        codec_think_bos_id=talker_config.codec_think_bos_id,
        codec_think_eos_id=talker_config.codec_think_eos_id,
        codec_pad_id=talker_config.codec_pad_id,
        audio_token_id=thinker_config.audio_token_id,
        image_token_id=thinker_config.image_token_id,
        video_token_id=thinker_config.video_token_id,
        tts_bos_token_id=root_config.tts_bos_token_id,
        tts_eos_token_id=root_config.tts_eos_token_id,
        tts_pad_token_id=root_config.tts_pad_token_id,
        im_start_token_id=root_config.im_start_token_id,
        im_end_token_id=root_config.im_end_token_id,
        system_token_id=root_config.system_token_id,
        user_token_id=root_config.user_token_id,
        assistant_token_id=root_config.assistant_token_id,
        speaker_map=talker_config.speaker_id,
    )

    scheduler = OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        request_builder=request_builder,
        result_adapter=result_adapter,
        stream_chunk_handler=stream_chunk_handler,
        stream_done_handler=stream_done_handler,
    )

    model_runner = QwenTalkerModelRunner(
        model_worker,
        output_proc,
        scheduler.outbox,
        feedback_enabled=feedback_enabled,
    )

    scheduler._model_runner = model_runner

    return scheduler
