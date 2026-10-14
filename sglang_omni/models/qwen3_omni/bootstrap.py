# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni-specific scheduler construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.qwen3_omni.request_builders import (
    DEFAULT_THINKER_MAX_NEW_TOKENS,
    should_generate_audio_output,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

    from sglang_omni.models.qwen3_omni.talker_scheduler import QwenTalkerScheduler
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
else:
    pass


def create_thinker_scheduler(
    server_args: "ServerArgs",
    gpu_id: int = 0,
    *,
    speech_enabled: bool = False,
    tp_rank: int = 0,
    nccl_port: int | None = None,
    total_gpu_memory_fraction: float | None = None,
    enable_async_decode: bool = True,
    async_decode_min_batch_size: int = 1,
    prefill_coalesce_requests: int = 0,
    prefill_coalesce_wait_ms: float = 60.0,
    prefill_coalesce_when_idle: bool = False,
    operator_selected_prefill_backend: bool = False,
    enable_early_first_token: bool = False,
) -> "OmniScheduler[SGLangARRequestData]":
    """Create the Qwen thinker scheduler."""
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
    from sglang_omni.models.qwen3_omni.request_builders import (
        make_thinker_scheduler_adapters,
        make_thinker_stream_output_builder,
    )
    from sglang_omni.models.qwen3_omni.thinker_model_runner import (
        Qwen3OmniThinkerModelRunner,
    )
    from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni.scheduling.generation_batch_policy import (
        CudaGraphBackend,
        get_prefill_cuda_graph_backend,
    )
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import SGLangOutputProcessor
    from sglang_omni.utils import cuda_graph_batch_validator

    prefill_graph_backend = get_prefill_cuda_graph_backend(server_args)
    enable_prefill_input_embeds = prefill_graph_backend == CudaGraphBackend.BREAKABLE

    algorithm = SpeculativeAlgorithm.from_string(
        resolved_view(server_args).speculative_algorithm
    )
    if enable_early_first_token and not algorithm.is_dflash():
        raise ValueError("Early first-token publication requires DFLASH")
    else:
        pass
    speculative_infrastructure = None
    if algorithm.is_dflash():
        speculative_infrastructure = (
            Qwen3OmniThinkerModelRunner.create_speculative_infrastructure(
                server_args,
                gpu_id,
                tp_rank=tp_rank,
                nccl_port=nccl_port,
                total_gpu_memory_fraction=total_gpu_memory_fraction,
                enable_prefill_input_embeds=enable_prefill_input_embeds,
            )
        )
        model_worker = speculative_infrastructure.model_worker
        tree_cache = speculative_infrastructure.tree_cache
        req_to_token_pool = speculative_infrastructure.req_to_token_pool
        token_to_kv_pool_allocator = (
            speculative_infrastructure.token_to_kv_pool_allocator
        )
        model_config = model_worker.model_config
        enable_async_decode = False
    elif not algorithm.is_none():
        raise ValueError("Qwen thinker supports only DFLASH speculative decoding")
    else:
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
            model_arch_override="Qwen3OmniThinkerForCausalLM",
            total_gpu_memory_fraction=total_gpu_memory_fraction,
            enable_prefill_input_embeds=enable_prefill_input_embeds,
        )

    if prefill_graph_backend == CudaGraphBackend.BREAKABLE:
        cuda_graph_batch_validator.attest_prefill_cuda_graphs(
            model_worker.model_runner,
            operator_selected=operator_selected_prefill_backend,
        )
    else:
        pass

    output_proc = SGLangOutputProcessor()

    if (
        speech_enabled
        and prefill_graph_backend != CudaGraphBackend.BREAKABLE
        and speculative_infrastructure is None
    ):
        model_runner = ThinkerModelRunner(model_worker, output_proc)
    else:
        model_runner = Qwen3OmniThinkerModelRunner(model_worker, output_proc)

    if speculative_infrastructure is not None:
        speculative_infrastructure.target_adapter.attach_runner(model_runner)
    else:
        pass

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
    if speculative_infrastructure is not None:
        build_thinker_request = request_builder
        draft_context_limit_tokens = (
            speculative_infrastructure.draft_context_limit_tokens
        )
        draft_block_tokens = speculative_infrastructure.draft_worker.block_size
        target_position_limit_tokens = (
            speculative_infrastructure.target_position_limit_tokens
        )

        def request_builder(payload: StagePayload) -> SGLangARRequestData:
            if should_generate_audio_output(payload):
                raise ValueError("Thinker DFlash currently supports text output only")
            else:
                pass
            pipeline_state = Qwen3OmniPipelineState.from_dict(payload.data)
            max_new_tokens = (payload.request.params or {}).get(
                "max_new_tokens", DEFAULT_THINKER_MAX_NEW_TOKENS
            )
            if type(max_new_tokens) is not int or max_new_tokens <= 0:
                raise ValueError(
                    "max_new_tokens must be a positive integer for Thinker DFlash"
                )
            else:
                pass
            prompt_tokens = pipeline_state.prompt["input_ids"].numel()
            # note (Codex): The final verify still evaluates a full draft block.
            required_tokens = prompt_tokens + max_new_tokens + draft_block_tokens - 1
            if required_tokens > draft_context_limit_tokens:
                raise ValueError(
                    "Requested token count exceeds the model's maximum context length: "
                    "Thinker DFlash draft context budget "
                    f"(prompt_tokens={prompt_tokens}, max_new_tokens={max_new_tokens}, "
                    f"draft_block_tokens={draft_block_tokens}, "
                    f"required_tokens={required_tokens}, "
                    f"draft_context_limit_tokens={draft_context_limit_tokens})"
                )
            else:
                request_data = build_thinker_request(payload)
                multimodal_inputs = request_data.req.multimodal_inputs
                if multimodal_inputs is not None:
                    assert multimodal_inputs.mrope_position_delta_repeated_cache is None
                    prompt_positions = multimodal_inputs.mrope_positions
                    position_delta = multimodal_inputs.mrope_position_delta
                    if (
                        prompt_positions.device.type != "cpu"
                        or position_delta.device.type != "cpu"
                        or prompt_positions.shape != (3, prompt_tokens)
                        or position_delta.numel() != 1
                    ):
                        raise RuntimeError(
                            "Thinker DFlash requires CPU prompt position metadata"
                        )
                    else:
                        pass
                    if (
                        not torch.isfinite(prompt_positions).all().item()
                        or not torch.isfinite(position_delta).all().item()
                    ):
                        raise ValueError(
                            "Requested token count exceeds the model's maximum context length: "
                            "Thinker DFlash requires finite multimodal positions"
                        )
                    else:
                        pass
                    # note (Codex): Decode truncates the delta before adding positions.
                    normalized_delta_tokens = int(position_delta.item())
                    first_generated_position = prompt_tokens + normalized_delta_tokens
                    last_verify_position = required_tokens - 1 + normalized_delta_tokens
                    if (
                        prompt_positions.min().item() < 0
                        or prompt_positions.max().item() >= target_position_limit_tokens
                        or first_generated_position < 0
                        or last_verify_position >= target_position_limit_tokens
                    ):
                        raise ValueError(
                            "Requested token count exceeds the model's maximum context length: "
                            "Thinker DFlash rotary position budget "
                            f"(first_generated_position={first_generated_position}, "
                            f"last_verify_position={last_verify_position}, "
                            f"target_position_limit_tokens={target_position_limit_tokens})"
                        )
                    else:
                        multimodal_inputs.mrope_position_delta = position_delta.to(
                            dtype=torch.int64
                        )
                else:
                    pass
                return request_data

    else:
        pass
    stream_output_builder = make_thinker_stream_output_builder(
        speech_enabled=speech_enabled
    )

    scheduler = OmniScheduler(
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
        draft_worker=(
            speculative_infrastructure.draft_worker
            if speculative_infrastructure is not None
            else None
        ),
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
        prefill_coalesce_requests=prefill_coalesce_requests,
        prefill_coalesce_wait_ms=prefill_coalesce_wait_ms,
        prefill_coalesce_when_idle=prefill_coalesce_when_idle,
    )
    if enable_early_first_token:
        assert speculative_infrastructure is not None
        scheduler.wait_for_prefill_publication_on_stop = True
        speculative_infrastructure.target_adapter.prefill_token_publisher = (
            scheduler.publish_prefill_first_token
        )
    else:
        pass
    return scheduler


def create_talker_scheduler(
    server_args: "ServerArgs",
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
    enable_talker_start_topology: bool = False,
    code2wav_in_process: bool = False,
    operator_selected_prefill_backend: bool = False,
    codec_coalesce_frames: int = 0,
    codec_coalesce_first_frames: int = 0,
    codec_coalesce_early_frames: int = 0,
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
        enable_partial_start=enable_partial_start,
        partial_start_min_chunks=partial_start_min_chunks,
        enable_talker_start_topology=enable_talker_start_topology,
        im_end_token_id=root_config.im_end_token_id,
    )

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
