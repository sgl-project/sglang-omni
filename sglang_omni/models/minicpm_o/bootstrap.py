# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o-specific scheduler construction."""

from __future__ import annotations

from typing import Any


def create_talker_scheduler(
    server_args: Any,
    gpu_id: int = 0,
    *,
    tp_rank: int = 0,
    nccl_port: int | None = None,
    total_gpu_memory_fraction: float | None = None,
):
    """Create the MiniCPM-o talker scheduler.

    The talker runs the checkpoint's 20-layer TTS backbone as a native sglang
    AR stage: prefill consumes the projected condition embeddings, decode
    embeds the previous codec token natively (no feedback loop), so overlap
    scheduling and CUDA graphs both stay on. Radix caching is disabled because
    prompts are per-request embeddings, and chunked prefill is off because the
    condition must land in one prefill pass.
    """
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    from sglang_omni.models.minicpm_o.request_builders import (
        make_talker_scheduler_adapters,
    )
    from sglang_omni.models.minicpm_o.talker_model_runner import (
        MiniCPMOTalkerModelRunner,
    )
    from sglang_omni.scheduling.bootstrap import (
        create_sglang_infrastructure,
        init_sglang_cuda_graphs,
    )
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import SGLangOutputProcessor
    from sglang_omni.vendor.sglang.server_args import override_server_args

    want_cuda_graph = not bool(server_args.disable_cuda_graph)
    override_server_args(
        server_args,
        "sglang_omni.minicpm_o.talker",
        disable_radix_cache=True,
        chunked_prefill_size=0,
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
        model_arch_override="MiniCPMOTalkerForCausalLM",
        # The model's load_weights strips the ``tts.`` prefix itself.
        weight_prefix=None,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        defer_cuda_graph_capture=want_cuda_graph,
    )

    model = model_worker.model_runner.model
    # Align vocab bookkeeping to the codec vocab before graph capture: post1
    # sizes sampling-orchestrator buffers from model_config.vocab_size (the
    # thinker text vocab), which mismatches the talker's codec-vocab logits.
    codec_vocab_size = model.num_audio_tokens
    model_config.vocab_size = codec_vocab_size
    model._sampler = model_worker.model_runner.sampler
    if want_cuda_graph:
        override_server_args(
            server_args,
            "sglang_omni.minicpm_o.talker_restore_cuda_graph_capture",
            disable_cuda_graph=False,
        )
        init_sglang_cuda_graphs(model_worker)

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model,
    )
    model_runner = MiniCPMOTalkerModelRunner(model_worker, output_proc)

    tokenizer = get_tokenizer(model_config.model_path, trust_remote_code=True)
    request_builder, result_adapter = make_talker_scheduler_adapters(
        model=model,
        codec_vocab_size=codec_vocab_size,
        codec_eos_id=model.codec_eos_id,
        tts_bos_token_id=tokenizer.convert_tokens_to_ids("<|tts_bos|>"),
        tts_eos_token_id=tokenizer.convert_tokens_to_ids("<|tts_eos|>"),
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
        model_runner=model_runner,
        request_builder=request_builder,
        result_adapter=result_adapter,
    )


def create_thinker_scheduler(
    server_args: Any,
    gpu_id: int = 0,
    *,
    tp_rank: int = 0,
    nccl_port: int | None = None,
    total_gpu_memory_fraction: float | None = None,
    enable_async_decode: bool = True,
    async_decode_min_batch_size: int = 2,
    speech_enabled: bool = False,
):
    """Create the MiniCPM-o thinker scheduler.

    With ``speech_enabled`` the runner captures per-step last-layer hidden
    states (CaptureHiddenMode.LAST); the talker consumes them as the TTS
    condition alongside the generated token ids.
    """
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    from sglang_omni.models.minicpm_o.request_builders import (
        make_thinker_scheduler_adapters,
        make_thinker_stream_output_builder,
        should_generate_audio_output,
    )
    from sglang_omni.models.minicpm_o.thinker_model_runner import (
        MiniCPMOThinkerModelRunner,
    )
    from sglang_omni.scheduling.bootstrap import (
        create_sglang_infrastructure,
        init_sglang_cuda_graphs,
    )
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import SGLangOutputProcessor
    from sglang_omni.vendor.sglang.server_args import override_server_args

    # Hidden-state capture requires return_hidden_states in the runner; defer
    # cuda-graph capture past infrastructure creation so graphs are built with
    # the hidden-capture configuration (mirrors qwen3_omni bootstrap).
    want_cuda_graph = not bool(server_args.disable_cuda_graph)
    defer_cuda_graph_capture = want_cuda_graph and speech_enabled
    if defer_cuda_graph_capture:
        saved_disable_cuda_graph = server_args.disable_cuda_graph
        saved_return_hidden_states = server_args.enable_return_hidden_states
        override_server_args(
            server_args,
            "sglang_omni.minicpm_o.defer_cuda_graph_capture",
            enable_return_hidden_states=True,
            disable_cuda_graph=True,
        )

    try:
        infrastructure = create_sglang_infrastructure(
            server_args,
            gpu_id,
            tp_rank=tp_rank,
            nccl_port=nccl_port,
            model_arch_override="MiniCPMO",
            total_gpu_memory_fraction=total_gpu_memory_fraction,
            defer_cuda_graph_capture=defer_cuda_graph_capture,
        )
    finally:
        if defer_cuda_graph_capture:
            override_server_args(
                server_args,
                "sglang_omni.minicpm_o.restore_cuda_graph_capture",
                disable_cuda_graph=saved_disable_cuda_graph,
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
        # Graphs must capture with return_hidden_states still on: the runner
        # requests CaptureHiddenMode.FULL every decode step, and the graph's
        # can_run gate requires an exact hidden-mode match — a graph captured
        # without hidden capture would never replay.
        init_sglang_cuda_graphs(model_worker)
        override_server_args(
            server_args,
            "sglang_omni.minicpm_o.restore_return_hidden_states",
            enable_return_hidden_states=saved_return_hidden_states,
        )

    def _should_emit_hidden(request: Any) -> bool:
        return should_generate_audio_output(request.data.stage_payload)

    output_proc = SGLangOutputProcessor(
        capture_hidden=speech_enabled,
        capture_hidden_layers=None,
        model=None,
        should_emit_hidden=_should_emit_hidden if speech_enabled else None,
    )
    model_runner = MiniCPMOThinkerModelRunner(
        model_worker, output_proc, speech_enabled=speech_enabled
    )

    tokenizer = get_tokenizer(model_config.model_path, trust_remote_code=True)
    request_builder, result_adapter = make_thinker_scheduler_adapters(
        tokenizer=tokenizer,
        vocab_size=model_config.vocab_size,
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
        model_runner=model_runner,
        request_builder=request_builder,
        result_adapter=result_adapter,
        stream_output_builder=make_thinker_stream_output_builder(),
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
    )
