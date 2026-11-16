# SPDX-License-Identifier: Apache-2.0
"""Fun-ASR SGLang engine builder."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from sglang.srt.managers.mm_utils import init_mm_embedding_cache
from sglang.srt.mem_cache.allocation_sizing import get_alloc_reserve_per_decode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.fun_asr import request_builders
from sglang_omni.models.fun_asr.configuration_fun_asr import FunAsrNanoFeatureExtractor
from sglang_omni.models.fun_asr.encoder_service import (
    FunASRPreLMEncoderService,
    build_cache_namespace,
)
from sglang_omni.models.fun_asr.tool_funcs.audio_lengths import (
    fun_asr_low_frame_rate_length,
)
from sglang_omni.platforms import current_platform
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.bootstrap import InfrastructureOptions
from sglang_omni.scheduling.engine_factory import (
    AsrEngineBuilder,
    GenerationDefaults,
    SchedulerExtras,
)
from sglang_omni.scheduling.generation_batch_policy import (
    CudaGraphBackend,
    build_default_prefill_cuda_graph_bs,
)
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.utils.gpu_compat import get_visible_gpu_sm_version

if TYPE_CHECKING:

    from sglang_omni.models.fun_asr.sglang_model import (
        FunAsrNanoForConditionalGeneration,
    )
    from sglang_omni.models.fun_asr.torch_mps_runner import FunASRTorchMpsModelRunner
else:
    pass

logger = logging.getLogger(__name__)


class FunASREngineBuilder(AsrEngineBuilder[request_builders.FunASRRequestData]):
    model_name = "Fun-ASR"
    model_arch_override = "FunAsrNanoForConditionalGeneration"
    supports_breakable_prefill_cuda_graph = True

    def __init__(
        self,
        *,
        max_running_requests: int,
        max_new_tokens: int,
        mem_fraction_static: float | None,
        mm_embedding_cache_size_bytes: int,
        enable_torch_compile: bool | None,
        enable_encoder_torch_compile: bool,
        enable_encoder_cuda_graph: bool,
        enable_async_decode: bool,
        async_decode_min_batch_size: int,
        prefill_coalesce_requests: int,
        prefill_coalesce_wait_ms: float,
        prefill_coalesce_when_idle: bool,
        prefill_coalesce_requires_pending_builds: bool,
        prefill_coalesce_after_builds_during_decode: bool,
        mm_attention_backend: str | None,
        enable_pre_lm_encoder: bool,
        pre_lm_cache_max_entries: int,
        pre_lm_cache_size_bytes: int,
        pre_lm_max_batch_size: int,
        pre_lm_max_batch_wait_ms: int,
        request_build_max_workers: int,
        request_build_max_pending: int | None,
        stream_emit_interval_s: float,
    ) -> None:
        self.max_running_requests = max_running_requests
        self.max_new_tokens = max_new_tokens
        self.mem_fraction_static = mem_fraction_static
        self.mm_embedding_cache_size_bytes = mm_embedding_cache_size_bytes
        self.enable_torch_compile = enable_torch_compile
        self.enable_encoder_torch_compile = enable_encoder_torch_compile
        self.enable_encoder_cuda_graph = enable_encoder_cuda_graph
        self.enable_async_decode = enable_async_decode
        self.async_decode_min_batch_size = async_decode_min_batch_size
        self.prefill_coalesce_requests = prefill_coalesce_requests
        self.prefill_coalesce_wait_ms = prefill_coalesce_wait_ms
        self.prefill_coalesce_when_idle = prefill_coalesce_when_idle
        self.prefill_coalesce_requires_pending_builds = (
            prefill_coalesce_requires_pending_builds
        )
        self.prefill_coalesce_after_builds_during_decode = (
            prefill_coalesce_after_builds_during_decode
        )
        self.mm_attention_backend = mm_attention_backend
        self.enable_pre_lm_encoder = enable_pre_lm_encoder
        self.pre_lm_cache_max_entries = pre_lm_cache_max_entries
        self.pre_lm_cache_size_bytes = pre_lm_cache_size_bytes
        self.pre_lm_max_batch_size = pre_lm_max_batch_size
        self.pre_lm_max_batch_wait_ms = pre_lm_max_batch_wait_ms
        self.request_build_max_workers = request_build_max_workers
        self.request_build_max_pending = request_build_max_pending
        self.stream_emit_interval_s = stream_emit_interval_s
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.feature_extractor: FunAsrNanoFeatureExtractor | None = None
        self.audio_encoder_service: FunASRPreLMEncoderService | None = None
        self.device: str | None = None
        self.torch_mps_model_runner: FunASRTorchMpsModelRunner | None = None
        self.speculative_worker_class: type[BaseSpecWorker] | None = None
        self.draft_checkpoint: TemporaryDirectory[str] | None = None
        self.speculative_reserve_tokens = 0
        self.engine_context_padding = 0
        self.context_length = 0

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_dir, trust_remote_code=True
        )
        self.feature_extractor = FunAsrNanoFeatureExtractor.from_pretrained(
            checkpoint_dir, trust_remote_code=True
        )
        encoder_token_count = int(
            fun_asr_low_frame_rate_length(self.feature_extractor.nb_max_frames)
        )
        prompt_overhead = request_builders.fun_asr_prompt_overhead_tokens(
            self.tokenizer
        )
        self.context_length = (
            encoder_token_count + self.max_new_tokens + prompt_overhead
        )

    def uses_torch_mps(self) -> bool:
        return self.device is not None and self.device.split(":")[0] == "mps"

    def adjust_overrides(self, overrides: dict[str, object]) -> None:
        if self.uses_torch_mps():
            overrides["enable_torch_compile"] = False
        else:
            pass

        draft_model_path = overrides.get("speculative_draft_model_path")
        speculative_algorithm = SpeculativeAlgorithm.from_string(
            overrides.get("speculative_algorithm")
        )
        if speculative_algorithm.is_none():
            if draft_model_path is not None:
                raise ValueError(
                    "speculative_draft_model_path needs a speculative_algorithm"
                )
            else:
                pass
        elif not speculative_algorithm.is_dflash():
            raise ValueError(
                f"{self.model_name} speculative decoding supports DFLASH, "
                f"not {speculative_algorithm.name}"
            )
        elif draft_model_path is None:
            raise ValueError(
                "speculative_algorithm=DFLASH needs speculative_draft_model_path"
            )
        elif not current_platform.is_cuda():
            raise ValueError(f"{self.model_name} DFLASH needs a CUDA GPU")
        else:
            # note (wilsonzheng0327): hosts without a draft never load its kernels.
            from sglang_omni.speculative.dflash_draft import (
                ReducedVocabularyDFlashWorker,
                materialize_draft_checkpoint,
            )

            # note (wilsonzheng0327): SGLang reads the draft config while resolving
            # the server args, so the flat copy must exist before that.
            self.draft_checkpoint = materialize_draft_checkpoint(
                str(draft_model_path), overrides.get("speculative_draft_model_revision")
            )
            overrides["speculative_draft_model_path"] = self.draft_checkpoint.name
            overrides["speculative_draft_model_revision"] = None
            self.speculative_worker_class = ReducedVocabularyDFlashWorker
            # note (wilsonzheng0327): async decode under a draft means speculative
            # overlap, which this runner has not opted into.
            self.enable_async_decode = False
            # note (wilsonzheng0327): on SGLang 0.5.23 the compiled path makes the draft half as fast as no draft at concurrency 1, while without a draft it still speeds up concurrency 1 by about 17%.
            overrides["enable_torch_compile"] = False
            dflash_block_size = overrides.get("speculative_dflash_block_size")
            num_draft_tokens = overrides.get("speculative_num_draft_tokens")
            # note (wilsonzheng0327): SGLang verifies speculative_dflash_block_size tokens per step, else speculative_num_draft_tokens, else the draft's block size.
            if dflash_block_size is not None:
                draft_width = int(dflash_block_size)
            elif num_draft_tokens is not None:
                draft_width = int(num_draft_tokens)
            else:
                with open(
                    os.path.join(self.draft_checkpoint.name, "config.json")
                ) as config_file:
                    draft_width = int(
                        json.load(config_file)["dflash_config"]["block_size"]
                    )
            # note (wilsonzheng0327): the context fits a 30 s clip and the full budget
            # exactly, and a draft request must stay two verify widths below the context.
            self.speculative_reserve_tokens = 2 * draft_width
            self.engine_context_padding = self.speculative_reserve_tokens + 1
            self.context_length += self.engine_context_padding

    def validate_before_infrastructure(self, server_args: ServerArgs) -> None:
        if self.uses_torch_mps():
            if not current_platform.is_mps():
                raise ValueError("Fun-ASR Torch/MPS requires the Apple Metal platform")
            else:
                pass
            if server_args.max_running_requests != 1:
                raise ValueError(
                    "Fun-ASR Apple currently requires max_running_requests=1"
                )
            else:
                pass
            if (
                not server_args.disable_radix_cache
                or server_args.chunked_prefill_size != -1
            ):
                raise ValueError(
                    "Fun-ASR Apple requires disabled radix cache and chunked prefill"
                )
            else:
                pass
            if server_args.quantization is not None:
                raise ValueError(
                    "Fun-ASR Apple currently requires unquantized HF weights"
                )
            else:
                pass
        else:
            pass
        super().validate_before_infrastructure(server_args)

    def infra_kwargs(self) -> InfrastructureOptions:
        if self.speculative_worker_class is None:
            return {}
        else:
            return {
                "supports_speculative_decoding": True,
                "speculative_worker_class": self.speculative_worker_class,
            }

    def validate_after_model_setup(
        self, model: FunAsrNanoForConditionalGeneration, server_args: ServerArgs
    ) -> None:
        del model, server_args
        # note (wilsonzheng0327): the reserve was sized before the server args resolved.
        if self.speculative_worker_class is None:
            pass
        elif get_alloc_reserve_per_decode() != self.speculative_reserve_tokens:
            raise RuntimeError(
                f"{self.model_name} sized its context for a "
                f"{self.speculative_reserve_tokens}-token speculative reserve, but "
                f"the scheduler reserves {get_alloc_reserve_per_decode()}"
            )
        else:
            pass

    def make_model_runner(
        self,
        model_worker: ModelWorker,
        output_proc: SGLangOutputProcessor,
    ) -> ModelRunner[request_builders.FunASRRequestData]:
        if self.uses_torch_mps():
            from .torch_mps_runner import FunASRTorchMpsModelRunner

            self.torch_mps_model_runner = FunASRTorchMpsModelRunner(
                model_worker, output_proc
            )
            return self.torch_mps_model_runner
        else:
            pass
        return super().make_model_runner(model_worker, output_proc)

    def setup_model(
        self,
        *,
        model_worker: ModelWorker,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: ServerArgs,
    ) -> None:
        if self.uses_torch_mps():
            from .torch_mps_runner import install_torch_mps_language_model

            install_torch_mps_language_model(
                model_worker.model_runner.model, checkpoint_dir
            )
        else:
            pass

    def make_abort_callback(self) -> Callable[[str], None] | None:
        return (
            self.torch_mps_model_runner.abort_request
            if self.torch_mps_model_runner is not None
            else None
        )

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        if self.uses_torch_mps():
            # Audio embeddings are inserted only at first prefill. Token-only
            # prefix reuse and split prefill cannot reconstruct that sidecar.
            return {
                "max_running_requests": 1,
                "disable_cuda_graph": True,
                "disable_overlap_schedule": True,
                "disable_radix_cache": True,
                "enable_torch_compile": False,
                "max_total_tokens": max(2048, self.context_length),
                "max_prefill_tokens": self.context_length,
                "chunked_prefill_size": -1,
                "attention_backend": "torch_native",
                "mm_attention_backend": "sdpa",
                "sampling_backend": "pytorch",
                "mem_fraction_static": self.mem_fraction_static,
                "dtype": dtype,
            }
        else:
            pass
        defaults: GenerationDefaults = {
            "max_running_requests": self.max_running_requests,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "enable_torch_compile": self.enable_torch_compile,
            "mem_fraction_static": self.mem_fraction_static,
            "max_prefill_tokens": 4096,
            "chunked_prefill_size": 4096,
            # Qualified capture budget; longer prefills run eager.
            "cuda_graph_backend_prefill": CudaGraphBackend.BREAKABLE,
            "cuda_graph_bs_prefill": build_default_prefill_cuda_graph_bs(256),
            "sampling_backend": "pytorch",
            "dtype": dtype,
        }
        if self.mm_attention_backend is not None:
            defaults["mm_attention_backend"] = self.mm_attention_backend
        else:
            sm_version = get_visible_gpu_sm_version(self.gpu_id)
            if sm_version is not None and sm_version >= 100:
                defaults["mm_attention_backend"] = "triton_attn"
            else:
                pass
        return defaults

    def setup_model_resources(
        self,
        model: FunAsrNanoForConditionalGeneration | None,
        server_args: ServerArgs,
        *,
        generation_cuda_graph_enabled: bool,
    ) -> None:
        if self.uses_torch_mps():
            return
        else:
            pass
        del generation_cuda_graph_enabled
        if self.enable_encoder_cuda_graph:
            # Capture needs the eager forwards; a dynamo-compiled callable
            # cannot be captured, so the graph takes precedence over encoder
            # compile.
            if self.enable_encoder_torch_compile:
                logger.warning(
                    "enable_encoder_cuda_graph supersedes "
                    "enable_encoder_torch_compile; the encoder runs from "
                    "captured CUDA graphs (eager capture), not dynamo"
                )
            else:
                pass
            from sglang_omni.models.fun_asr.encoder_cuda_graph import (
                FunASREncoderCudaGraphRunner,
            )

            device = next(model.audio_tower.parameters()).device
            graph_backend = current_platform.get_device_graph_backend(device)
            if graph_backend is None:
                logger.info(
                    f"Fun-ASR encoder CUDA graphs are unavailable on {device}; "
                    f"the encoder runs eager"
                )
            else:
                model.encoder_cuda_graph_runner = FunASREncoderCudaGraphRunner(
                    model.audio_tower,
                    model.multi_modal_projector,
                    graph_backend=graph_backend,
                    max_batch_size=self.pre_lm_max_batch_size,
                )
                logger.info(
                    f"Fun-ASR encoder CUDA graphs enabled (lazy capture per "
                    f"batch/length bucket, max_batch="
                    f"{self.pre_lm_max_batch_size})"
                )
        elif self.enable_encoder_torch_compile:
            from sglang_omni.models.fun_asr.stages import compile_fun_asr_audio_encoder

            compile_fun_asr_audio_encoder(
                model,
                warmup_inference_mode=self.enable_pre_lm_encoder,
            )
        else:
            pass
        init_mm_embedding_cache(self.mm_embedding_cache_size_bytes)

    def setup_runtime_resources(
        self,
        model: FunAsrNanoForConditionalGeneration | None,
        server_args: ServerArgs | None,
    ) -> None:
        if self.uses_torch_mps() or not self.enable_pre_lm_encoder:
            return
        else:
            pass
        self.audio_encoder_service = FunASRPreLMEncoderService(
            model,
            cache_namespace=build_cache_namespace(
                model,
                model_path=self.checkpoint_dir,
                feature_extractor=self.feature_extractor,
                mm_attention_backend=server_args.mm_attention_backend,
            ),
            cache_max_entries=self.pre_lm_cache_max_entries,
            cache_max_bytes=self.pre_lm_cache_size_bytes,
            max_batch_size=self.pre_lm_max_batch_size,
            max_batch_wait_ms=self.pre_lm_max_batch_wait_ms,
        )

    def make_adapters(self, model: object) -> tuple[
        Callable[[StagePayload], request_builders.FunASRRequestData],
        Callable[[request_builders.FunASRRequestData], StagePayload],
    ]:
        del model
        return request_builders.make_fun_asr_scheduler_adapters(
            tokenizer=self.tokenizer,
            feature_extractor=self.feature_extractor,
            max_new_tokens=self.max_new_tokens,
            # note (wilsonzheng0327): requests keep the limit they have without a
            # draft; only the engine holds the reserve.
            context_length=self.context_length - self.engine_context_padding,
            audio_encoder_service=self.audio_encoder_service,
            greedy_only=self.uses_torch_mps(),
        )

    def extra_scheduler_callbacks(self) -> dict[str, Callable[[], None] | None]:
        return {
            "shutdown_callback": (
                self.audio_encoder_service.close
                if self.audio_encoder_service is not None
                else None
            )
        }

    def cleanup_build_failure(self) -> None:
        if self.audio_encoder_service is not None:
            self.audio_encoder_service.close()
        else:
            pass

    def extra_scheduler_kwargs(
        self,
    ) -> SchedulerExtras[request_builders.FunASRRequestData]:
        return {
            "stream_output_builder": request_builders.make_fun_asr_stream_output_builder(
                tokenizer=self.tokenizer,
                min_emit_interval_s=self.stream_emit_interval_s,
            ),
            "enable_async_decode": (
                False if self.uses_torch_mps() else self.enable_async_decode
            ),
            "async_decode_min_batch_size": self.async_decode_min_batch_size,
            "prefill_coalesce_requests": (
                0 if self.uses_torch_mps() else self.prefill_coalesce_requests
            ),
            "prefill_coalesce_wait_ms": self.prefill_coalesce_wait_ms,
            "prefill_coalesce_when_idle": self.prefill_coalesce_when_idle,
            "prefill_coalesce_requires_pending_builds": (
                self.prefill_coalesce_requires_pending_builds
            ),
            "prefill_coalesce_after_builds_during_decode": (
                self.prefill_coalesce_after_builds_during_decode
            ),
            "request_build_max_workers": (
                1 if self.uses_torch_mps() else self.request_build_max_workers
            ),
            "request_build_max_pending": self.request_build_max_pending,
        }
