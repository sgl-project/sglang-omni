# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o-specific scheduler construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    read_ragged_verify_mode,
)

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.minicpm_o.components.sglang_thinker import (
    SPECULATIVE_HIDDEN_ROWS,
)
from sglang_omni.scheduling.generation_batch_policy import (
    get_decode_cuda_graph_bs,
    get_decode_cuda_graph_max_bs,
)

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
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
    operator_selected_prefill_backend: bool = False,
) -> OmniScheduler[SGLangARRequestData]:
    """Create a codec scheduler with per-request condition embeddings."""
    from sglang.srt.arg_groups.model_override_base import resolved_view
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    from sglang_omni.models.minicpm_o.talker_model_runner import (
        MiniCPMOTalkerModelRunner,
    )
    from sglang_omni.models.minicpm_o.talker_request import (
        make_talker_scheduler_adapters,
    )
    from sglang_omni.models.minicpm_o.talker_session import TalkerAdapter
    from sglang_omni.scheduling.bootstrap import (
        create_sglang_infrastructure,
        init_sglang_cuda_graphs,
    )
    from sglang_omni.scheduling.generation_batch_policy import (
        CudaGraphBackend,
        get_prefill_cuda_graph_backend,
    )
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend.output_processor import (
        SGLangOutputProcessor,
    )
    from sglang_omni.utils import cuda_graph_batch_validator
    from sglang_omni.vendor.sglang.server_args import override_server_args

    want_cuda_graph = not bool(resolved_view(server_args).disable_cuda_graph)
    prefill_graph_backend = get_prefill_cuda_graph_backend(server_args)
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
        enable_prefill_input_embeds=prefill_graph_backend == CudaGraphBackend.BREAKABLE,
    )

    model = model_worker.model_runner.model
    # note (MayDomine): graph sampling buffers must use the codec vocabulary size.
    codec_vocab_size = model.num_audio_tokens
    model_config.vocab_size = codec_vocab_size
    model.sampler = model_worker.model_runner.sampler
    if want_cuda_graph:
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
    model_runner = MiniCPMOTalkerModelRunner(model_worker, output_proc)
    if session_mode:
        # note (0xtoward): a session's units replay host history, so it keeps the host sampler.
        pass
    else:
        model_runner.enable_device_sampling()

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
        model_runner=model_runner,
        request_builder=request_builder,
        result_adapter=result_adapter,
        session_adapter=TalkerAdapter(model) if session_mode else None,
        # note (0xtoward): the penalty and length floor read device state, never an unresolved token, so lookahead starts at one request.
        enable_async_decode=not session_mode,
        async_decode_min_batch_size=1,
    )


def create_thinker_scheduler(
    server_args: ServerArgs,
    gpu_id: int = 0,
    *,
    tp_rank: int = 0,
    nccl_port: int | None = None,
    total_gpu_memory_fraction: float | None = None,
    enable_async_decode: bool = True,
    async_decode_min_batch_size: int = 1,
    speech_enabled: bool = False,
) -> OmniScheduler[SGLangARRequestData]:
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
    speculative = cfg.speculative_algorithm is not None
    if (
        cfg.speculative_algorithm == "DSPARK"
        and read_ragged_verify_mode() != RaggedVerifyMode.STATIC
    ):
        raise ValueError(
            "MiniCPM-o thinker speculative decoding requires static verify"
        )
    else:
        pass
    if (
        speech_enabled
        and speculative
        and (
            cfg.speculative_algorithm not in ("EAGLE3", "DFLASH", "DSPARK")
            or cfg.speculative_eagle_topk not in (None, 1)
        )
    ):
        # note (0xtoward): the talker rows ride the drafts' layer-capture hooks, and a
        # linear verify block commits its leading rows.
        raise ValueError(
            "speech output needs an EAGLE3 chain (topk 1), DFLASH or DSPARK draft, got "
            f"{cfg.speculative_algorithm} topk={cfg.speculative_eagle_topk}"
        )
    else:
        pass
    # note (0xtoward): check the speech hidden capacity before capturing verify graphs.
    defer_cuda_graph_capture = want_cuda_graph and speech_enabled
    override_hidden_capture = defer_cuda_graph_capture and not speculative
    if override_hidden_capture:
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

    def configure_talker_capture(model_worker: ModelWorker) -> None:
        model_worker.model_runner.model.capture_hidden_for_talker = speech_enabled

    try:
        infrastructure = create_sglang_infrastructure(
            server_args,
            gpu_id,
            tp_rank=tp_rank,
            nccl_port=nccl_port,
            model_arch_override="MiniCPMO",
            total_gpu_memory_fraction=total_gpu_memory_fraction,
            defer_cuda_graph_capture=defer_cuda_graph_capture,
            supports_speculative_decoding=True,
            before_memory_pool=configure_talker_capture,
        )
        if speech_enabled and speculative:
            worker = infrastructure[0]
            verify_rows = (
                max(
                    worker.model_runner.max_running_requests,
                    get_decode_cuda_graph_max_bs(server_args) or 0,
                    max(get_decode_cuda_graph_bs(server_args) or [0]),
                )
                * worker.speculative_worker.speculative_num_draft_tokens
            )
            if verify_rows > SPECULATIVE_HIDDEN_ROWS:
                raise ValueError(
                    f"MiniCPM-o speech verify needs {verify_rows} hidden rows; "
                    f"capacity is {SPECULATIVE_HIDDEN_ROWS}. "
                    "Reduce max_running_requests or the draft block size."
                )
            else:
                pass
        else:
            pass
        if defer_cuda_graph_capture:
            init_sglang_cuda_graphs(infrastructure[0])
        else:
            pass
    finally:
        if override_hidden_capture:
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
    tokenizer = get_tokenizer(model_config.model_path, trust_remote_code=True)
    # note (ruinique): thinker chunks terminate on any of these tokens.
    eos_token_ids = [
        int(tokenizer.convert_tokens_to_ids(token))
        for token in ("<|tts_eos|>", "<|im_end|>", "</s>")
    ]
    model_runner = MiniCPMOThinkerModelRunner(model_worker, output_proc, eos_token_ids)

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
