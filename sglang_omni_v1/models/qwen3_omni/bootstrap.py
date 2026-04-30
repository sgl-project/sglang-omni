# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni-specific scheduler construction."""

from __future__ import annotations

from typing import Any


def create_thinker_scheduler(
    server_args: Any,
    gpu_id: int = 0,
    *,
    speech_enabled: bool = False,
    tp_rank: int = 0,
    nccl_port: int | None = None,
):
    """Create the Qwen thinker scheduler."""
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    from sglang_omni_v1.model_runner.thinker_model_runner import ThinkerModelRunner
    from sglang_omni_v1.models.qwen3_omni.request_builders import (
        make_thinker_scheduler_adapters,
        make_thinker_stream_output_builder,
    )
    from sglang_omni_v1.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni_v1.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni_v1.scheduling.sglang_backend import SGLangOutputProcessor

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
    ) = create_sglang_infrastructure(
        server_args,
        gpu_id,
        tp_rank=tp_rank,
        nccl_port=nccl_port,
        model_arch_override="Qwen3OmniThinkerForCausalLM",
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
        stream_output_builder=stream_output_builder,
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
):
    """Create the Qwen talker scheduler."""
    del speech_enabled
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    from sglang_omni_v1.models.qwen3_omni.request_builders import (
        make_talker_scheduler_adapters,
    )
    from sglang_omni_v1.models.qwen3_omni.talker_model_runner import (
        QwenTalkerModelRunner,
    )
    from sglang_omni_v1.models.qwen3_omni.talker_scheduler import (
        QwenTalkerScheduler,
        configure_talker_server_args,
    )
    from sglang_omni_v1.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni_v1.scheduling.sglang_backend import SGLangOutputProcessor

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
    )

    scheduler._model_runner = QwenTalkerModelRunner(
        model_worker,
        output_proc,
        scheduler.outbox,
        feedback_enabled=feedback_enabled,
    )
    return scheduler
