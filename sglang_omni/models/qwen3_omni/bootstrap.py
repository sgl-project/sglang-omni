# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni-specific scheduler construction."""

from __future__ import annotations

from typing import Any

# The external backend name ``create_external_model_worker`` records for the
# Apple Silicon (non-MLX) stages, whose forward the eager Torch runner owns.
_TORCH_MPS_BACKEND = "torch_mps"


def _uses_torch_mps_external_forward(model_worker: Any) -> bool:
    """Whether this worker's forward belongs to the eager Torch MPS thinker.

    ``create_external_model_worker`` is constructed with the external backend
    that owns the stage's forward; on Apple Silicon without MLX that is
    ``torch_mps``. Keying the runner off that declaration -- rather than off the
    host platform alone -- ties the selection to the worker actually built for
    it, so a worker that outsources its forward elsewhere is never handed the
    Torch MPS thinker.
    """
    from sglang_omni.model_runner.external_model_worker import uses_external_forward

    return (
        uses_external_forward(model_worker)
        and getattr(model_worker, "_external_backend_name", None) == _TORCH_MPS_BACKEND
    )


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
    operator_selected_prefill_backend: bool = False,
):
    """Create the Qwen thinker scheduler."""
    from sglang.srt.arg_groups.model_override_base import resolved_view
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    from sglang_omni.model_runner.external_model_worker import uses_external_forward
    from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
    from sglang_omni.models.qwen3_omni.apple_runtime import qwen3_omni_uses_mlx_backend
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

    cfg = resolved_view(server_args)
    capture_hidden_layers = [0, 24] if speech_enabled else None
    capture_hidden = speech_enabled
    prefill_graph_backend = get_prefill_cuda_graph_backend(server_args)
    enable_prefill_input_embeds = prefill_graph_backend == CudaGraphBackend.BREAKABLE
    want_cuda_graph = not bool(cfg.disable_cuda_graph)
    defer_cuda_graph_capture = want_cuda_graph and capture_hidden

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

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        model_config,
    ) = infrastructure

    if defer_cuda_graph_capture:
        # Deferring capture must not also skip the omni wrapper: without it the
        # prefill embeds view is never applied, so the graph's input_embeds slot
        # exists only when the model config happens to be multimodal.
        init_sglang_cuda_graphs(model_worker)

    if prefill_graph_backend == CudaGraphBackend.BREAKABLE:
        cuda_graph_batch_validator.attest_prefill_cuda_graphs(
            model_worker.model_runner,
            operator_selected=operator_selected_prefill_backend,
        )

    def _should_generate_qwen_audio_output(request: Any) -> bool:
        return should_generate_audio_output(request.data.stage_payload)

    # An external-forward worker holds SGLang's zero-weight stub model and gets
    # no hidden capture hooks, so the processor must read the hidden states its
    # own runner supplies instead of the stub's absent static capture. The same
    # holds for the native MLX worker: its runner populates
    # ``logits_output.hidden_states`` directly, which is then the sole source.
    uses_mlx = qwen3_omni_uses_mlx_backend()
    capture_from_model = (
        capture_hidden_layers
        and not uses_mlx
        and not uses_external_forward(model_worker)
    )

    output_proc = SGLangOutputProcessor(
        capture_hidden=capture_hidden,
        capture_hidden_layers=capture_hidden_layers,
        model=model_worker.model_runner.model if capture_from_model else None,
        should_emit_hidden=_should_generate_qwen_audio_output,
    )

    if uses_mlx:
        from sglang_omni.models.qwen3_omni.mlx.runner import (
            Qwen3OmniMlxSchedulerModelRunner,
        )

        model_runner = Qwen3OmniMlxSchedulerModelRunner(model_worker, output_proc)
    elif _uses_torch_mps_external_forward(model_worker):
        # Apple Silicon without MLX: the eager Torch MPS thinker owns every
        # forward, so the stub worker never runs one.
        from sglang_omni.models.qwen3_omni.torch_mps_runner import (
            build_qwen3_omni_torch_mps_thinker_runner,
        )

        talker_config = getattr(model_config.hf_config, "talker_config", None)
        model_runner = build_qwen3_omni_torch_mps_thinker_runner(
            tp_worker=model_worker,
            output_processor=output_proc,
            model_path=model_config.model_path,
            thinker_config=model_config.hf_config.thinker_config,
            capture_hidden_layers=capture_hidden_layers,
            accept_hidden_layer=getattr(talker_config, "accept_hidden_layer", None),
            dtype=getattr(model_config, "dtype", None),
        )
    elif speech_enabled and prefill_graph_backend != CudaGraphBackend.BREAKABLE:
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
        # A runner that owns per-request backend state (the native MLX thinker
        # and the eager Torch MPS thinker each own one KV cache per request)
        # releases it from the abort path.
        abort_callback=getattr(model_runner, "abort_request", None),
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
        prefill_coalesce_requests=prefill_coalesce_requests,
        prefill_coalesce_wait_ms=prefill_coalesce_wait_ms,
        prefill_coalesce_when_idle=prefill_coalesce_when_idle,
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

    from sglang_omni.model_runner.external_model_worker import uses_external_forward
    from sglang_omni.models.qwen3_omni.apple_runtime import qwen3_omni_uses_mlx_backend
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

    uses_mlx = qwen3_omni_uses_mlx_backend()
    if uses_mlx and enable_partial_start:
        # The Apple profile already disables partial talker start; the native
        # MLX talker refuses it explicitly rather than speculating on future
        # text rows it has no queue for.
        raise ValueError(
            "Apple Qwen3-Omni MLX talker does not support partial talker start"
        )

    want_cuda_graph = configure_talker_server_args(
        server_args,
        feedback_enabled=feedback_enabled,
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
        model_arch_override="Qwen3OmniTalker",
        weight_prefix=weight_prefix,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        defer_cuda_graph_capture=want_cuda_graph,
    )
    # Apple Silicon without MLX: the eager Torch MPS talker owns every forward,
    # so the stub worker never runs one.
    uses_torch_mps = not uses_mlx and _uses_torch_mps_external_forward(model_worker)
    if uses_torch_mps and enable_partial_start:
        raise ValueError(
            "Apple Qwen3-Omni Torch MPS talker does not support partial talker start"
        )
    # Note:(Chenchen Hong) align the talker vocab to the codec vocab: post1 sizes
    # the repetition-penalty orchestrator from model_config.vocab_size (the
    # thinker text vocab), which mismatches the talker's codec-vocab logits.
    _codec_vocab_size = model_config.hf_config.talker_config.text_config.vocab_size
    model_config.vocab_size = _codec_vocab_size
    _runner_cfg = model_worker.model_runner.model_config
    if _runner_cfg is not model_config:
        _runner_cfg.vocab_size = _codec_vocab_size
    if not uses_external_forward(model_worker):
        # An external-forward worker holds SGLang's zero-weight stub model and
        # no sampler; its own runner supplies greedy codec token ids instead.
        model_worker.model_runner.model._sampler = model_worker.model_runner.sampler
    if want_cuda_graph:
        # Equivalent to init_cuda_graphs() while the talker requests no prefill
        # embeds slot, but keeps both stages on one path so enabling talker
        # prefill graphs later cannot silently miss the embeds view.
        init_sglang_cuda_graphs(model_worker)

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        # Both Apple workers hold SGLang's zero-weight stub model. Nothing in
        # this processor reads ``model`` while ``capture_hidden`` is False, so the
        # stub is withheld rather than handed on.
        model=None if (uses_mlx or uses_torch_mps) else model_worker.model_runner.model,
    )

    from sglang_omni.models.qwen3_omni.components.talker_prefill import (
        TalkerPrefillBuilder,
    )

    prefill_model = model_worker.model_runner.model
    prefill_builder = None
    torch_mps_talker = None
    if uses_mlx:
        prefill_builder = getattr(model_worker, "mlx_talker_prefill_builder", None)
        if prefill_builder is None:
            raise RuntimeError(
                "Apple Qwen3-Omni MLX talker worker exposed no "
                "Qwen3OmniMlxTalkerPrefillBuilder; the talker weights did not load"
            )
    elif uses_torch_mps:
        import torch

        from sglang_omni.models.qwen3_omni.torch_mps import (
            TorchMpsTalkerPrefillShim,
            load_torch_mps_talker,
        )
        from sglang_omni.platforms import current_platform

        # The talker is loaded here rather than inside the runner factory
        # because the prompt builder is constructed first and must project with
        # the model's own weights.
        torch_mps_talker = load_torch_mps_talker(
            model_config.model_path,
            dtype=getattr(model_config, "dtype", None) or torch.bfloat16,
            device=current_platform.get_device(gpu_id),
        )
        prefill_model = TorchMpsTalkerPrefillShim.from_talker(torch_mps_talker)

    tokenizer = get_tokenizer(
        model_config.model_path,
        trust_remote_code=True,
    )
    root_config = model_config.hf_config
    thinker_config = root_config.thinker_config
    talker_config = root_config.talker_config
    codec_vocab_size = talker_config.text_config.vocab_size
    if prefill_builder is None:
        prefill_builder = TalkerPrefillBuilder(
            model=prefill_model,
            model_path=model_config.model_path,
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
            codec_bos_id=talker_config.codec_bos_id,
            codec_nothink_id=talker_config.codec_nothink_id,
            codec_think_bos_id=talker_config.codec_think_bos_id,
            codec_think_eos_id=talker_config.codec_think_eos_id,
            codec_pad_id=talker_config.codec_pad_id,
            speaker_map=talker_config.speaker_id,
        )
    (
        request_builder,
        result_adapter,
        stream_chunk_handler,
        stream_done_handler,
    ) = make_talker_scheduler_adapters(
        tokenizer=tokenizer,
        codec_vocab_size=codec_vocab_size,
        prefill_builder=prefill_builder,
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
        request_builder=request_builder,
        result_adapter=result_adapter,
        stream_chunk_handler=stream_chunk_handler,
        stream_done_handler=stream_done_handler,
        enable_partial_start=enable_partial_start,
        partial_start_min_chunks=partial_start_min_chunks,
        im_end_token_id=root_config.im_end_token_id,
    )

    if uses_mlx:
        model_runner = _build_talker_mlx_model_runner(
            model_worker,
            output_proc,
            scheduler.outbox,
            feedback_enabled=feedback_enabled,
        )
    elif uses_torch_mps:
        from sglang_omni.models.qwen3_omni.torch_mps_runner import (
            build_qwen3_omni_torch_mps_talker_runner,
        )

        model_runner = build_qwen3_omni_torch_mps_talker_runner(
            tp_worker=model_worker,
            output_processor=output_proc,
            outbox=scheduler.outbox,
            talker=torch_mps_talker,
            feedback_enabled=feedback_enabled,
        )
    else:
        model_runner = QwenTalkerModelRunner(
            model_worker,
            output_proc,
            scheduler.outbox,
            feedback_enabled=feedback_enabled,
        )
    scheduler.bind_model_runner(model_runner)
    return scheduler


def _build_talker_mlx_model_runner(
    model_worker: Any,
    output_processor: Any,
    outbox: Any,
    *,
    feedback_enabled: bool,
):
    """Build the native MLX talker model runner around the loaded weights."""

    from sglang_omni.models.qwen3_omni.mlx.runner import (
        build_qwen3_omni_talker_mlx_runner,
    )

    return build_qwen3_omni_talker_mlx_runner(
        tp_worker=model_worker,
        output_processor=output_processor,
        outbox=outbox,
        mlx_talker=getattr(model_worker, "mlx_talker", None),
        feedback_enabled=feedback_enabled,
    )
