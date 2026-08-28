# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni-specific scheduler construction."""

from __future__ import annotations

from typing import Any

from sglang_omni.vendor.sglang.server_args import override_server_args


def create_thinker_scheduler(
    server_args: Any,
    gpu_id: int = 0,
    *,
    speech_enabled: bool = False,
    tp_rank: int = 0,
    nccl_port: int | None = None,
    total_gpu_memory_fraction: float | None = None,
    enable_async_decode: bool = True,
    async_decode_min_batch_size: int = 2,
    prefill_coalesce_requests: int = 0,
    prefill_coalesce_wait_ms: float = 60.0,
    prefill_coalesce_when_idle: bool = False,
    scheduler_cls: type | None = None,
    scheduler_kwargs: dict[str, Any] | None = None,
):
    """Create the Qwen thinker scheduler."""
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
    from sglang_omni.models.qwen3_omni.request_builders import (
        make_thinker_scheduler_adapters,
        make_thinker_stream_output_builder,
        should_generate_audio_output,
    )
    from sglang_omni.models.qwen3_omni.thinker_model_runner import (
        Qwen3OmniThinkerModelRunner,
    )
    from sglang_omni.scheduling.bootstrap import (
        create_sglang_infrastructure,
        init_sglang_cuda_graphs,
    )
    from sglang_omni.scheduling.generation_batch_policy import (
        CudaGraphBackend,
        get_prefill_cuda_graph_backend,
    )
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import SGLangOutputProcessor
    from sglang_omni.utils import cuda_graph_batch_validator

    capture_hidden_layers = [0, 24] if speech_enabled else None
    capture_hidden = speech_enabled
    prefill_graph_backend = get_prefill_cuda_graph_backend(server_args)
    enable_prefill_input_embeds = prefill_graph_backend == CudaGraphBackend.BREAKABLE
    want_cuda_graph = not bool(server_args.disable_cuda_graph)
    defer_cuda_graph_capture = want_cuda_graph and capture_hidden
    if defer_cuda_graph_capture:
        saved_disable_cuda_graph = server_args.disable_cuda_graph
        saved_return_hidden_states = server_args.enable_return_hidden_states
        override_server_args(
            server_args,
            "sglang_omni.qwen3_omni.defer_cuda_graph_capture",
            enable_return_hidden_states=True,
            disable_cuda_graph=True,
        )

    try:
        infrastructure = create_sglang_infrastructure(
            server_args,
            gpu_id,
            tp_rank=tp_rank,
            nccl_port=nccl_port,
            model_arch_override="Qwen3OmniThinkerForCausalLM",
            capture_hidden_layers=capture_hidden_layers,
            total_gpu_memory_fraction=total_gpu_memory_fraction,
            defer_cuda_graph_capture=defer_cuda_graph_capture,
            enable_prefill_input_embeds=enable_prefill_input_embeds,
        )
    finally:
        if defer_cuda_graph_capture:
            override_server_args(
                server_args,
                "sglang_omni.qwen3_omni.restore_cuda_graph_capture",
                disable_cuda_graph=saved_disable_cuda_graph,
                enable_return_hidden_states=saved_return_hidden_states,
            )

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = infrastructure

    if defer_cuda_graph_capture:
        # Deferring capture must not also skip the omni wrapper: without it the
        # prefill embeds view is never applied, so the graph's input_embeds slot
        # exists only when the model config happens to be multimodal.
        init_sglang_cuda_graphs(model_worker)

    if prefill_graph_backend == CudaGraphBackend.BREAKABLE:
        cuda_graph_batch_validator.attest_prefill_cuda_graphs(
            model_worker.model_runner, server_args
        )

    def _should_generate_qwen_audio_output(request: Any) -> bool:
        return should_generate_audio_output(request.data.stage_payload)

    output_proc = SGLangOutputProcessor(
        capture_hidden=capture_hidden,
        capture_hidden_layers=capture_hidden_layers,
        model=model_worker.model_runner.model if capture_hidden_layers else None,
        should_emit_hidden=_should_generate_qwen_audio_output,
    )

    if speech_enabled and prefill_graph_backend != CudaGraphBackend.BREAKABLE:
        model_runner = ThinkerModelRunner(model_worker, output_proc)
    else:
        model_runner = Qwen3OmniThinkerModelRunner(model_worker, output_proc)

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

    scheduler_cls = scheduler_cls or OmniScheduler
    return scheduler_cls(
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
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
        prefill_coalesce_requests=prefill_coalesce_requests,
        prefill_coalesce_wait_ms=prefill_coalesce_wait_ms,
        prefill_coalesce_when_idle=prefill_coalesce_when_idle,
        **(scheduler_kwargs or {}),
    )


def create_talker_scheduler(
    server_args: Any,
    gpu_id: int = 0,
    *,
    weight_prefix: str = "talker.",
    speech_enabled: bool = True,
    feedback_enabled: bool = True,
    tp_rank: int = 0,
    nccl_port: int | None = None,
    total_gpu_memory_fraction: float | None = None,
    enable_partial_start: bool = False,
    partial_start_min_chunks: int = 5,
):
    """Create the Qwen talker scheduler."""
    del speech_enabled
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    from sglang_omni.models.qwen3_omni.request_builders import (
        make_talker_scheduler_adapters,
    )
    from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
    from sglang_omni.models.qwen3_omni.talker_scheduler import (
        QwenTalkerScheduler,
        configure_talker_server_args,
    )
    from sglang_omni.scheduling.bootstrap import (
        create_sglang_infrastructure,
        init_sglang_cuda_graphs,
    )
    from sglang_omni.scheduling.sglang_backend import SGLangOutputProcessor

    want_cuda_graph = configure_talker_server_args(
        server_args,
        feedback_enabled=feedback_enabled,
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
        tp_rank=tp_rank,
        nccl_port=nccl_port,
        model_arch_override="Qwen3OmniTalker",
        weight_prefix=weight_prefix,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        defer_cuda_graph_capture=want_cuda_graph,
    )
    # Note:(Chenchen Hong) align the talker vocab to the codec vocab: post1 sizes
    # the repetition-penalty orchestrator from model_config.vocab_size (the
    # thinker text vocab), which mismatches the talker's codec-vocab logits.
    _codec_vocab_size = model_config.hf_config.talker_config.text_config.vocab_size
    model_config.vocab_size = _codec_vocab_size
    _runner_cfg = model_worker.model_runner.model_config
    if _runner_cfg is not model_config:
        _runner_cfg.vocab_size = _codec_vocab_size
    model_worker.model_runner.model._sampler = model_worker.model_runner.sampler
    if want_cuda_graph:
        override_server_args(
            server_args,
            "sglang_omni.qwen3_omni.talker_restore_cuda_graph_capture",
            disable_cuda_graph=False,
        )
        # Equivalent to init_cuda_graphs() while the talker requests no prefill
        # embeds slot, but keeps both stages on one path so enabling talker
        # prefill graphs later cannot silently miss the embeds view.
        init_sglang_cuda_graphs(model_worker)

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

    scheduler = QwenTalkerScheduler(
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
        enable_partial_start=enable_partial_start,
        partial_start_min_chunks=partial_start_min_chunks,
        im_end_token_id=root_config.im_end_token_id,
    )

    model_runner = QwenTalkerModelRunner(
        model_worker,
        output_proc,
        scheduler.outbox,
        feedback_enabled=feedback_enabled,
    )
    scheduler.bind_model_runner(model_runner)
    return scheduler
