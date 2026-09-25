# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o-specific scheduler construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.types import SchedulerRequest
else:
    pass


def create_talker_scheduler(
    server_args: ServerArgs,
    gpu_id: int = 0,
    *,
    tp_rank: int = 0,
    nccl_port: int | None = None,
    total_gpu_memory_fraction: float | None = None,
    session_mode: bool = False,
) -> OmniScheduler:
    """Create a codec scheduler with per-request condition embeddings."""
    from sglang.srt.arg_groups.model_override_base import resolved_view
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    from sglang_omni.models.minicpm_o.talker_model_runner import (
        MiniCPMOTalkerModelRunner,
    )
    from sglang_omni.models.minicpm_o.talker_request import (
        make_talker_scheduler_adapters,
    )
    from sglang_omni.scheduling.bootstrap import (
        create_sglang_infrastructure,
        init_sglang_cuda_graphs,
    )
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend.output_processor import (
        SGLangOutputProcessor,
    )
    from sglang_omni.vendor.sglang.server_args import override_server_args

    want_cuda_graph = not bool(resolved_view(server_args).disable_cuda_graph)
    # note (MayDomine): condition embeddings require an uncached, unsplit prefill.
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
        model_config,
    ) = create_sglang_infrastructure(
        server_args,
        gpu_id,
        tp_rank=tp_rank,
        nccl_port=nccl_port,
        model_arch_override="MiniCPMOTalkerForCausalLM",
        # note (MayDomine): the model loader strips its own tts weight prefix.
        weight_prefix=None,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        defer_cuda_graph_capture=want_cuda_graph,
    )

    model = model_worker.model_runner.model
    # note (MayDomine): graph sampling buffers must use the codec vocabulary size.
    codec_vocab_size = model.num_audio_tokens
    model_config.vocab_size = codec_vocab_size
    model.sampler = model_worker.model_runner.sampler
    if want_cuda_graph:
        init_sglang_cuda_graphs(model_worker)
    else:
        pass

    output_proc = SGLangOutputProcessor()
    model_runner = MiniCPMOTalkerModelRunner(model_worker, output_proc)

    tokenizer = get_tokenizer(model_config.model_path, trust_remote_code=True)
    request_builder, result_adapter = make_talker_scheduler_adapters(
        model=model,
        codec_vocab_size=codec_vocab_size,
        codec_eos_id=model.codec_eos_id,
        tts_bos_token_id=tokenizer.convert_tokens_to_ids("<|tts_bos|>"),
        tts_eos_token_id=tokenizer.convert_tokens_to_ids("<|tts_eos|>"),
    )

    from sglang_omni.models.minicpm_o.talker_session import TalkerAdapter

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
        session_adapter=TalkerAdapter(model) if session_mode else None,
    )


def create_thinker_scheduler(
    server_args: ServerArgs,
    gpu_id: int = 0,
    *,
    tp_rank: int = 0,
    nccl_port: int | None = None,
    total_gpu_memory_fraction: float | None = None,
    enable_async_decode: bool = True,
    async_decode_min_batch_size: int = 2,
    speech_enabled: bool = False,
) -> OmniScheduler:
    """Create a thinker scheduler with optional hidden-state capture for speech."""
    from sglang.srt.arg_groups.model_override_base import resolved_view
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    from sglang_omni.models.minicpm_o.request_builders import (
        build_thinker_stream_output,
        make_thinker_scheduler_adapters,
    )
    from sglang_omni.models.minicpm_o.routing import should_generate_audio_output
    from sglang_omni.models.minicpm_o.thinker_model_runner import (
        MiniCPMOThinkerModelRunner,
    )
    from sglang_omni.scheduling.bootstrap import (
        create_sglang_infrastructure,
        init_sglang_cuda_graphs,
    )
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend.output_processor import (
        SGLangOutputProcessor,
    )
    from sglang_omni.vendor.sglang.server_args import override_server_args

    cfg = resolved_view(server_args)
    want_cuda_graph = not bool(cfg.disable_cuda_graph)
    defer_cuda_graph_capture = want_cuda_graph and speech_enabled
    if defer_cuda_graph_capture:
        saved_return_hidden_states = cfg.enable_return_hidden_states
        saved_return_hidden_states_mode = cfg.return_hidden_states_mode
        override_server_args(
            server_args,
            "sglang_omni.minicpm_o.defer_cuda_graph_capture",
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
            model_arch_override="MiniCPMO",
            total_gpu_memory_fraction=total_gpu_memory_fraction,
            defer_cuda_graph_capture=defer_cuda_graph_capture,
        )
        if defer_cuda_graph_capture:
            init_sglang_cuda_graphs(infrastructure[0])
        else:
            pass
    finally:
        if defer_cuda_graph_capture:
            override_server_args(
                server_args,
                "sglang_omni.minicpm_o.restore_return_hidden_states",
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

    def _should_emit_hidden(request: SchedulerRequest) -> bool:
        return should_generate_audio_output(request.data.stage_payload)

    output_proc = SGLangOutputProcessor(
        capture_hidden=speech_enabled,
        should_emit_hidden=_should_emit_hidden if speech_enabled else None,
    )
    model_runner = MiniCPMOThinkerModelRunner(model_worker, output_proc)

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
        model_runner=model_runner,
        request_builder=request_builder,
        result_adapter=result_adapter,
        stream_output_builder=build_thinker_stream_output,
        abort_callback=model_runner.reset_request,
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
    )
