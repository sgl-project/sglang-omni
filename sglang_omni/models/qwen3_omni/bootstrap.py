# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni-specific scheduler construction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from sglang_omni.models.qwen3_omni.components.talker_prefill import TalkerPrefillBuilder
from sglang_omni.models.qwen3_omni.talker_session import QwenTalkerSessionAdapter
from sglang_omni.models.qwen3_omni.thinker_session import QwenThinkerSessionAdapter

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

    from sglang_omni.models.qwen3_omni.talker_scheduler import QwenTalkerScheduler
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
else:
    pass


def create_thinker_scheduler(
    server_args: "ServerArgs",
    gpu_id: int = 0,
    *,
    speech_enabled: bool = False,
    execution_mode: Literal["request", "session"] = "request",
    tp_rank: int = 0,
    nccl_port: int | None = None,
    total_gpu_memory_fraction: float | None = None,
    enable_async_decode: bool = True,
    async_decode_min_batch_size: int = 1,
    prefill_coalesce_requests: int = 0,
    prefill_coalesce_wait_ms: float = 60.0,
    prefill_coalesce_when_idle: bool = False,
    operator_selected_prefill_backend: bool = False,
    continuation_aware_admission: bool = True,
) -> "OmniScheduler[SGLangARRequestData]":
    """Create the Qwen thinker scheduler."""
    # Note (Junnan Li): The engine, runner and scheduler imports stay in the factory: this module is imported by the platform layer before a device exists.
    from sglang.srt.arg_groups.model_override_base import resolved_view
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
    from sglang_omni.models.qwen3_omni.request_builders import (
        make_thinker_scheduler_adapters,
        make_thinker_stream_output_builder,
    )
    from sglang_omni.models.qwen3_omni.thinker_model_runner import (
        Qwen3OmniThinkerModelRunner,
    )
    from sglang_omni.models.qwen3_omni.thinker_session_runner import (
        QwenSessionThinkerModelRunner,
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
    from sglang_omni.vendor.sglang.server_args import override_server_args

    prefill_graph_backend = get_prefill_cuda_graph_backend(server_args)
    enable_prefill_input_embeds = prefill_graph_backend == CudaGraphBackend.BREAKABLE

    # note (ratish): a session turn's prefill does not write its rows to the prompt hidden store,
    # so its talker keeps zeros at multimodal prompt rows.
    capture_prompt_hidden_states = speech_enabled and execution_mode == "request"
    # note (ratish): a batch capturing hidden states cannot replay graphs captured at NULL, so a
    # capturing thinker captures its graphs at FULL with the talker's layer, as MiniCPM-o's does.
    resolved_server_args = resolved_view(server_args)
    defer_cuda_graph_capture = (
        capture_prompt_hidden_states and not resolved_server_args.disable_cuda_graph
    )
    if defer_cuda_graph_capture:
        saved_return_hidden_states = resolved_server_args.enable_return_hidden_states
        saved_return_hidden_states_mode = resolved_server_args.return_hidden_states_mode
        override_server_args(
            server_args,
            "sglang_omni.qwen3_omni.capture_prompt_hidden_states",
            enable_return_hidden_states=True,
            return_hidden_states_mode="full",
        )
    else:
        pass

    try:
        infrastructure = create_sglang_infrastructure(
            server_args,
            gpu_id,
            tp_rank=tp_rank,
            nccl_port=nccl_port,
            model_arch_override="Qwen3OmniThinkerForCausalLM",
            total_gpu_memory_fraction=total_gpu_memory_fraction,
            enable_prefill_input_embeds=enable_prefill_input_embeds,
            defer_cuda_graph_capture=defer_cuda_graph_capture,
        )
        if capture_prompt_hidden_states:
            infrastructure[0].model_runner.model.set_talker_layers_to_capture()
        else:
            pass
        if defer_cuda_graph_capture:
            init_sglang_cuda_graphs(infrastructure[0])
        else:
            pass
    finally:
        if defer_cuda_graph_capture:
            override_server_args(
                server_args,
                "sglang_omni.qwen3_omni.restore_return_hidden_states",
                enable_return_hidden_states=saved_return_hidden_states,
                return_hidden_states_mode=saved_return_hidden_states_mode,
            )
        else:
            pass

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        model_config,
    ) = infrastructure

    if prefill_graph_backend == CudaGraphBackend.BREAKABLE:
        cuda_graph_batch_validator.attest_prefill_cuda_graphs(
            model_worker.model_runner,
            operator_selected=operator_selected_prefill_backend,
        )
    else:
        pass

    output_proc = SGLangOutputProcessor()

    if execution_mode == "session":
        model_runner = QwenSessionThinkerModelRunner(
            model_worker, output_proc, capture_prompt_hidden_states=False
        )
    elif speech_enabled and prefill_graph_backend != CudaGraphBackend.BREAKABLE:
        model_runner = ThinkerModelRunner(
            model_worker,
            output_proc,
            capture_prompt_hidden_states=capture_prompt_hidden_states,
        )
    else:
        model_runner = Qwen3OmniThinkerModelRunner(
            model_worker,
            output_proc,
            capture_prompt_hidden_states=capture_prompt_hidden_states,
        )

    tokenizer = get_tokenizer(
        model_config.model_path,
        trust_remote_code=True,
    )
    thinker_config = model_config.hf_config.thinker_config
    request_builder, result_adapter = make_thinker_scheduler_adapters(
        tokenizer=tokenizer,
        vocab_size=model_config.vocab_size,
        thinker_config=thinker_config,
        capture_prompt_hidden_states=capture_prompt_hidden_states,
    )
    # Note (Junnan Li): A session turn streams its tokens to the talker like any request; the session only owns the lifecycle.
    stream_output_builder = make_thinker_stream_output_builder(
        speech_enabled=speech_enabled,
        prompt_hidden_store=model_runner.prompt_hidden_store,
    )
    if execution_mode == "request":
        session_adapter = None
    else:
        session_adapter = QwenThinkerSessionAdapter(
            tokenizer, model_config.vocab_size, thinker_config
        )

    return OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        model_runner=model_runner,
        request_builder=request_builder,
        result_adapter=result_adapter,
        stream_output_builder=stream_output_builder,
        session_adapter=session_adapter,
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
        prefill_coalesce_requests=prefill_coalesce_requests,
        prefill_coalesce_wait_ms=prefill_coalesce_wait_ms,
        prefill_coalesce_when_idle=prefill_coalesce_when_idle,
        shutdown_callback=(
            model_runner.prompt_hidden_store.destroy
            if model_runner.prompt_hidden_store is not None
            else None
        ),
        continuation_aware_admission=continuation_aware_admission,
    )


def create_talker_scheduler(
    server_args: "ServerArgs",
    gpu_id: int = 0,
    *,
    weight_prefix: str = "talker.",
    speech_enabled: bool = True,
    execution_mode: Literal["request", "session"] = "request",
    feedback_enabled: bool = True,
    tp_rank: int = 0,
    nccl_port: int | None = None,
    total_gpu_memory_fraction: float | None = None,
    enable_partial_start: bool = False,
    partial_start_min_chunks: int = 5,
    enable_talker_start_topology: bool = False,
    code2wav_in_process: bool = False,
    operator_selected_prefill_backend: bool = False,
    codec_coalesce_frames: int = 0,
    codec_coalesce_first_frames: int = 0,
    codec_coalesce_early_frames: int = 0,
    continuation_aware_admission: bool = True,
) -> "QwenTalkerScheduler":
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
    from sglang_omni.scheduling.generation_batch_policy import (
        CudaGraphBackend,
        get_prefill_cuda_graph_backend,
    )
    from sglang_omni.scheduling.sglang_backend import SGLangOutputProcessor
    from sglang_omni.utils import cuda_graph_batch_validator

    want_cuda_graph = configure_talker_server_args(
        server_args,
        feedback_enabled=feedback_enabled,
    )
    prefill_graph_backend = get_prefill_cuda_graph_backend(server_args)

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
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
        enable_prefill_input_embeds=prefill_graph_backend == CudaGraphBackend.BREAKABLE,
    )
    # Note:(Chenchen Hong) align the talker vocab to the codec vocab: post1 sizes
    # the repetition-penalty orchestrator from model_config.vocab_size (the
    # thinker text vocab), which mismatches the talker's codec-vocab logits.
    _codec_vocab_size = model_config.hf_config.talker_config.text_config.vocab_size
    model_config.vocab_size = _codec_vocab_size
    _runner_cfg = model_worker.model_runner.model_config
    if _runner_cfg is not model_config:
        _runner_cfg.vocab_size = _codec_vocab_size
    else:
        pass
    model_worker.model_runner.model.sampler = model_worker.model_runner.sampler
    if want_cuda_graph:
        # note (ratish): capture after binding the sampler so decode graphs record it.
        init_sglang_cuda_graphs(model_worker)
        if prefill_graph_backend == CudaGraphBackend.BREAKABLE:
            cuda_graph_batch_validator.attest_prefill_cuda_graphs(
                model_worker.model_runner,
                operator_selected=operator_selected_prefill_backend,
            )
        else:
            pass
    else:
        pass

    output_proc = SGLangOutputProcessor()

    tokenizer = get_tokenizer(
        model_config.model_path,
        trust_remote_code=True,
    )
    root_config = model_config.hf_config
    thinker_config = root_config.thinker_config
    talker_config = root_config.talker_config
    codec_vocab_size = talker_config.text_config.vocab_size
    if execution_mode == "request":
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
        session_adapter = None
    else:
        prefill_builder = TalkerPrefillBuilder(
            model=model_worker.model_runner.model,
            model_path=model_config.model_path,
            codec_bos_id=talker_config.codec_bos_id,
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
        session_adapter = QwenTalkerSessionAdapter(
            prefill_builder,
            tokenizer,
            codec_vocab_size,
            talker_config.codec_eos_token_id,
            thinker_config,
        )
        request_builder = None
        result_adapter = None
        stream_chunk_handler = session_adapter.append_text_chunk
        stream_done_handler = session_adapter.mark_thinker_done

    scheduler = QwenTalkerScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        request_builder=request_builder,
        result_adapter=result_adapter,
        stream_chunk_handler=stream_chunk_handler,
        stream_done_handler=stream_done_handler,
        session_adapter=session_adapter,
        enable_partial_start=enable_partial_start,
        partial_start_min_chunks=partial_start_min_chunks,
        enable_talker_start_topology=enable_talker_start_topology,
        im_end_token_id=root_config.im_end_token_id,
        continuation_aware_admission=continuation_aware_admission,
    )

    if session_adapter is not None:
        session_adapter.outbox = scheduler.outbox
    else:
        pass
    model_runner = QwenTalkerModelRunner(
        model_worker,
        output_proc,
        scheduler.outbox,
        code2wav_in_process=code2wav_in_process,
        feedback_enabled=feedback_enabled,
        codec_coalesce_frames=codec_coalesce_frames,
        codec_coalesce_first_frames=codec_coalesce_first_frames,
        codec_coalesce_early_frames=codec_coalesce_early_frames,
    )
    scheduler.bind_model_runner(model_runner)
    return scheduler
