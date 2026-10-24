# SPDX-License-Identifier: Apache-2.0
"""FireRedTTS3 SGLang engine builder (talker stage)."""

from __future__ import annotations

import logging
import time
from typing import Protocol

import torch
from sglang.srt.runtime_context import get_exec, get_schedule
from sglang.srt.server_args import ServerArgs
from transformers import AutoTokenizer

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.firered_tts3.compiled_dit import CompiledDiTExecutor
from sglang_omni.models.firered_tts3.flow_graph import (
    DEFAULT_CAPTURE_SIZES,
    FLOW_MODES,
    FlowGraphRunner,
    FlowMode,
    flow_t_span,
    prepare_flow_dit,
)
from sglang_omni.models.firered_tts3.hf_config import (
    FIREREDTTS3_MODEL_ARCH_OVERRIDE,
    register_fireredtts3_hf_config,
)
from sglang_omni.models.firered_tts3.model_runner import FireRedTTS3ModelRunner
from sglang_omni.models.firered_tts3.request_builders import (
    FireRedTTS3SGLangRequestData,
    apply_latent_result,
    build_sglang_firered_tts3_request,
    build_stream_output,
)
from sglang_omni.models.firered_tts3.sglang_model import FireRedTTS3SGLangModel
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.engine_factory import (
    GenerationDefaults,
    SchedulerExtras,
    TtsEngineBuilder,
)
from sglang_omni.scheduling.generation_batch_policy import get_decode_cuda_graph_max_bs
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor

logger = logging.getLogger(__name__)

CONTEXT_LENGTH = 8192
MIN_GRAPH_FEEDBACK_BATCH_SIZE = 256


class FireRedRequestBuilder(Protocol):
    def __call__(self, payload: StagePayload) -> FireRedTTS3SGLangRequestData: ...


class FireRedResultAdapter(Protocol):
    def __call__(self, request_data: FireRedTTS3SGLangRequestData) -> StagePayload: ...


class FireRedAbortCallback(Protocol):
    def __call__(self, request_id: str) -> None: ...


class FireRedTTS3EngineBuilder(TtsEngineBuilder[FireRedTTS3SGLangRequestData]):
    """Build talker resources through the shared engine hooks."""

    model_name: str = "fireredtts3"
    context_length: int = CONTEXT_LENGTH
    model_arch_override: str = FIREREDTTS3_MODEL_ARCH_OVERRIDE

    def __init__(
        self,
        *,
        n_timesteps: int,
        max_patches: int,
        max_running_requests: int,
        flow_cuda_graph: bool,
        flow_cuda_graph_capture_sizes: list[int] | None,
        flow_cuda_graph_max_graphs: int,
        flow_batching: bool,
        control_token_fastpath: bool,
        skip_min_patch_stop: bool,
        flow_mode: FlowMode,
        warmup_on_init: bool,
        enable_async_decode: bool,
        async_decode_min_batch_size: int,
    ) -> None:
        self.n_timesteps: int = int(n_timesteps)
        self.max_running_requests: int = int(max_running_requests)
        self.flow_cuda_graph: bool = bool(flow_cuda_graph)
        self.flow_cuda_graph_capture_sizes: list[int] | None = (
            sorted(
                {
                    int(batch_size)
                    for batch_size in flow_cuda_graph_capture_sizes
                    if int(batch_size) > 0
                }
            )
            or None
            if flow_cuda_graph_capture_sizes
            else None
        )
        self.flow_cuda_graph_max_graphs: int = int(flow_cuda_graph_max_graphs)
        self.flow_batching: bool = bool(flow_batching)
        self.control_token_fastpath: bool = bool(control_token_fastpath)
        self.skip_min_patch_stop: bool = bool(skip_min_patch_stop)
        if flow_mode not in FLOW_MODES:
            raise ValueError(
                f"FireRedTTS3 flow_mode {flow_mode!r} invalid; "
                f"expected one of {FLOW_MODES}"
            )
        else:
            pass
        self.flow_mode: FlowMode = flow_mode
        self.warmup_on_init: bool = bool(warmup_on_init)
        self.enable_async_decode: bool = bool(enable_async_decode)
        self.async_decode_min_batch_size: int = int(async_decode_min_batch_size)
        if min(self.n_timesteps, int(max_patches), self.max_running_requests) <= 0:
            raise ValueError("FireRedTTS3 batching limits must be positive")
        else:
            pass
        if self.flow_cuda_graph_max_graphs <= 0:
            raise ValueError("FireRedTTS3 flow_cuda_graph_max_graphs must be positive")
        else:
            pass
        if self.async_decode_min_batch_size <= 0:
            raise ValueError("FireRedTTS3 async_decode_min_batch_size must be positive")
        else:
            pass
        self.model_runner: FireRedTTS3ModelRunner | None = None
        self.flow_graphs: FlowGraphRunner | None = None
        self.flow_dit_stack: (
            tuple[torch.nn.Module | CompiledDiTExecutor, torch.nn.Module] | None
        ) = None

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        register_fireredtts3_hf_config()

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        return {
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            # note (kermit): Latent-conditioned requests cannot reuse radix prefixes.
            "disable_radix_cache": True,
            "enable_torch_compile": False,
            "max_running_requests": self.max_running_requests,
            "chunked_prefill_size": 0,
            "mem_fraction_static": 0.20,
            # note (kermit): FlashInfer rejects the seeded filtered text sampling used by Instruct.
            "sampling_backend": "pytorch",
            "dtype": dtype,
            "trust_remote_code": False,
        }

    def adjust_overrides(
        self, overrides: dict[str, int | float | str | bool | None]
    ) -> None:
        if int(overrides.get("tp_size", 1)) != 1:
            raise ValueError("FireRedTTS3 supports tp_size=1 only")
        else:
            pass
        requested = int(
            overrides.get("max_running_requests", self.max_running_requests)
        )
        if requested <= 0:
            raise ValueError("FireRedTTS3 max_running_requests must be positive")
        else:
            pass
        self.max_running_requests = requested
        overrides["disable_radix_cache"] = True
        overrides["chunked_prefill_size"] = 0
        if bool(overrides.get("enable_torch_compile", False)):
            raise ValueError(
                "FireRedTTS3 compiles its flow side path directly; SGLang "
                "backbone compile is disabled"
            )
        else:
            pass
        if not bool(overrides.get("disable_cuda_graph", True)):
            # note (kermit): Decode replay requires the captured and requested hidden modes to match.
            overrides["enable_return_hidden_states"] = True
        else:
            pass

    def setup_model(
        self,
        *,
        model_worker: ModelWorker,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: ServerArgs,
    ) -> None:
        model = model_worker.model_runner.model
        model.eval()
        logger.info(
            f"FireRedTTS3 talker dtypes: backbone_embed={next(model.backbone.parameters()).dtype} "
            f"qkv0={model.backbone.model.layers[0].self_attn.qkv_proj.weight.dtype} "
            f"patch_encoder={next(model.patch_encoder.parameters()).dtype}"
        )
        if bool(get_exec().graph.disable_cuda_graph):
            logger.info("FireRedTTS3 backbone decode: eager")
            return
        else:
            pass
        # note (kermit): Capture must retain the same feedback address even for padded graph buckets.
        max_running_requests = int(get_schedule().max_running_requests)
        model.enable_graph_feedback(
            max(
                max_running_requests,
                int(get_decode_cuda_graph_max_bs(server_args) or 0),
                MIN_GRAPH_FEEDBACK_BATCH_SIZE,
            )
        )
        logger.info(
            "FireRedTTS3 backbone decode: SGLang CUDA graph with model-owned "
            "feedback buffer"
        )

    def compile_model(
        self, model: FireRedTTS3SGLangModel, server_args: ServerArgs
    ) -> None:
        """Prepare the DiT stack before either graph family captures it."""
        if not self.flow_cuda_graph:
            logger.info(
                f"FireRedTTS3 flow_mode={self.flow_mode} ignored: the flow "
                "CUDA graph is disabled"
            )
            return
        else:
            pass
        started = time.perf_counter()
        model.weights_locked = True
        self.flow_dit_stack = prepare_flow_dit(model.dit, self.flow_mode)
        logger.info(
            f"FireRedTTS3 flow mode {self.flow_mode}: DiT stack prepared in "
            f"{time.perf_counter() - started:.1f}s (lazy compile on capture)"
        )

    def setup_model_resources(
        self,
        model: FireRedTTS3SGLangModel,
        server_args: ServerArgs,
        *,
        generation_cuda_graph_enabled: bool,
    ) -> None:
        """Capture Flow resources after the backbone graphs, then warm the side path."""
        if self.flow_cuda_graph:
            self.capture_flow_graphs(model)
        else:
            logger.info("FireRedTTS3 flow CUDA graph: disabled")
        if self.warmup_on_init:
            model.warmup_side_path()
            logger.info("FireRedTTS3 talker side-path warmup complete")
        else:
            pass

    def capture_flow_graphs(self, model: FireRedTTS3SGLangModel) -> None:
        """Pre-capture the flow-loop CUDA graphs for the hot batch buckets."""
        dit, dit_fallback = (
            self.flow_dit_stack
            if self.flow_dit_stack is not None
            else (model.dit, model.dit)
        )
        device = next(model.dit.parameters()).device
        if device.type != "cuda":
            logger.warning(
                "FireRedTTS3 flow CUDA graph needs a CUDA device; staying eager"
            )
            return
        else:
            pass
        conditioning_channels = int(model.config.dit_hidden_size) + int(
            model.config.spk_in_dim or 0
        )
        max_running_requests = int(get_schedule().max_running_requests)
        capture_batch_sizes = [
            batch_size
            for batch_size in (
                self.flow_cuda_graph_capture_sizes or DEFAULT_CAPTURE_SIZES
            )
            if batch_size <= max_running_requests
        ]
        runner = FlowGraphRunner(
            dit,
            flow_t_span(self.n_timesteps, device),
            n_timesteps=self.n_timesteps,
            patch_size=model.patch_size,
            history_length=model.history_length,
            max_graphs=max(self.flow_cuda_graph_max_graphs, len(capture_batch_sizes)),
            capture_sizes=capture_batch_sizes,
            mode=self.flow_mode,
            dit_fallback=dit_fallback,
        )
        reserved_before_bytes = torch.cuda.memory_reserved(device)
        started = time.perf_counter()
        for batch_size in capture_batch_sizes:
            history_latents = torch.zeros(
                batch_size, model.history_length, model.redae_dim, device=device
            )
            noise = torch.zeros(
                batch_size, model.patch_size, model.redae_dim, device=device
            )
            conditioning = torch.zeros(
                batch_size,
                model.history_length + model.patch_size,
                conditioning_channels,
                device=device,
            )
            guidance_scales = conditioning.new_full((batch_size, 1, 1), 2.0)
            if (
                runner.run(
                    history_latents,
                    noise,
                    conditioning,
                    guidance_scales,
                    self.n_timesteps,
                )
                is None
            ):
                logger.warning(
                    "FireRedTTS3 flow CUDA graph pre-capture failed; "
                    "the flow side path stays eager"
                )
                return
            else:
                pass
        self.flow_graphs = runner
        reserved_mib = (
            torch.cuda.memory_reserved(device) - reserved_before_bytes
        ) / 2**20
        logger.info(
            f"FireRedTTS3 flow CUDA graphs captured: sizes={capture_batch_sizes} "
            f"mode={self.flow_mode} reserved +{reserved_mib:.0f} MiB "
            f"in {time.perf_counter() - started:.1f}s"
        )

    def make_model_runner(
        self, model_worker: ModelWorker, output_processor: SGLangOutputProcessor
    ) -> FireRedTTS3ModelRunner:
        self.model_runner = FireRedTTS3ModelRunner(
            model_worker,
            output_processor,
            flow_graphs=self.flow_graphs,
            flow_batching=self.flow_batching,
            control_token_fastpath=self.control_token_fastpath,
            skip_min_patch_stop=self.skip_min_patch_stop,
        )
        return self.model_runner

    def make_adapters(
        self, model: FireRedTTS3SGLangModel
    ) -> tuple[FireRedRequestBuilder, FireRedResultAdapter]:
        text_tokenizer = (
            AutoTokenizer.from_pretrained(self.checkpoint_dir, local_files_only=True)
            if model.config.use_lm_head
            else None
        )
        raw_ratio = model.config.text_latent_interleave_ratio
        interleave_ratio = (
            (int(raw_ratio[0]), int(raw_ratio[1])) if raw_ratio is not None else None
        )

        def build_request(payload: StagePayload) -> FireRedTTS3SGLangRequestData:
            return build_sglang_firered_tts3_request(
                payload,
                patch_size=model.patch_size,
                vocab_size=int(model.config.vocab_size),
                interleave_ratio=interleave_ratio,
            )

        def apply_result(request_data: FireRedTTS3SGLangRequestData) -> StagePayload:
            return apply_latent_result(request_data, text_tokenizer=text_tokenizer)

        return build_request, apply_result

    def make_abort_callback(self) -> FireRedAbortCallback | None:
        assert self.model_runner is not None
        return self.model_runner.reset_request

    def extra_scheduler_kwargs(self) -> SchedulerExtras[FireRedTTS3SGLangRequestData]:
        return {
            "stream_output_builder": build_stream_output,
            "enable_async_decode": self.enable_async_decode,
            "async_decode_min_batch_size": self.async_decode_min_batch_size,
        }


__all__ = ["FireRedTTS3EngineBuilder"]
