# SPDX-License-Identifier: Apache-2.0
"""ARK-ASR SGLang engine builder."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.managers.mm_utils import init_mm_embedding_cache
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    WhisperFeatureExtractor,
)

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.arkasr import request_builders
from sglang_omni.models.arkasr.encoder_service import (
    ArkasrPreLMEncoderService,
    build_cache_namespace,
)
from sglang_omni.models.arkasr.request_builders import ArkASRRequestData
from sglang_omni.platforms import current_platform
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.bootstrap import InfrastructureOptions
from sglang_omni.scheduling.engine_factory import (
    AsrEngineBuilder,
    GenerationDefaults,
    SchedulerExtras,
)
from sglang_omni.scheduling.generation_batch_policy import CudaGraphBackend
from sglang_omni.scheduling.types import DeferredAdmission
from sglang_omni.speculative.dflash_draft import (
    ReducedVocabularyDFlashWorker,
    materialize_draft_checkpoint,
)
from sglang_omni.utils.gpu_compat import get_visible_gpu_sm_version

if TYPE_CHECKING:

    from sglang_omni.model_runner.model_worker import ModelWorker
    from sglang_omni.models.arkasr.sglang_model import ArkasrForConditionalGeneration
    from sglang_omni.scheduling.sglang_backend import SGLangOutputProcessor
else:
    pass

logger = logging.getLogger(__name__)


class ArkasrModelRunner(ModelRunner[ArkASRRequestData]):
    """The shared AR runner, opted into speculative overlap.

    Audio embeddings enter only at prefill, so a decode round reads nothing but the tokens the previous round left on the device.
    """

    supports_speculative_async_decode: bool = True


class ArkasrEngineBuilder(AsrEngineBuilder[ArkASRRequestData]):
    model_name = "ARK-ASR"
    model_arch_override = "ArkasrForConditionalGeneration"
    supports_breakable_prefill_cuda_graph = True

    def __init__(
        self,
        *,
        max_running_requests: int,
        encoder_max_batch_size: int,
        max_new_tokens: int,
        enable_async_decode: bool | None,
        async_decode_min_batch_size: int,
        mem_fraction_static: float | None,
        mm_embedding_cache_size_bytes: int,
        enable_torch_compile: bool | None,
        mm_attention_backend: str | None,
        request_build_max_workers: int,
        request_build_max_pending: int | None,
        prefill_coalesce_requests: int,
        prefill_coalesce_wait_ms: float,
        prefill_coalesce_when_idle: bool,
        prefill_coalesce_requires_pending_builds: bool,
        enable_pre_lm_encoder: bool = True,
        pre_lm_cache_max_entries: int = 4096,
        pre_lm_cache_size_bytes: int = 2 * 1024**3,
        pre_lm_max_batch_size: int = 8,
        pre_lm_max_batch_wait_ms: int = 0,
        pre_lm_max_pending: int = 32,
        enable_encoder_cuda_graph: bool = False,
        stream_emit_interval_s: float = 0.05,
    ) -> None:
        if pre_lm_max_batch_size < 1:
            raise ValueError(
                f"pre_lm_max_batch_size must be >= 1, got {pre_lm_max_batch_size}"
            )
        else:
            pass
        if pre_lm_max_batch_wait_ms < 0:
            raise ValueError(
                f"pre_lm_max_batch_wait_ms must be >= 0, got {pre_lm_max_batch_wait_ms}"
            )
        else:
            pass
        if pre_lm_max_pending < 1:
            raise ValueError(
                f"pre_lm_max_pending must be >= 1, got {pre_lm_max_pending}"
            )
        else:
            pass
        self.max_running_requests = max_running_requests
        self.encoder_max_batch_size = encoder_max_batch_size
        self.max_new_tokens = max_new_tokens
        self.enable_async_decode = enable_async_decode
        self.async_decode_min_batch_size = async_decode_min_batch_size
        self.mem_fraction_static = mem_fraction_static
        self.mm_embedding_cache_size_bytes = mm_embedding_cache_size_bytes
        self.enable_torch_compile = enable_torch_compile
        self.mm_attention_backend = mm_attention_backend
        self.request_build_max_workers = request_build_max_workers
        self.request_build_max_pending = request_build_max_pending
        self.prefill_coalesce_requests = prefill_coalesce_requests
        self.prefill_coalesce_wait_ms = prefill_coalesce_wait_ms
        self.prefill_coalesce_when_idle = prefill_coalesce_when_idle
        self.prefill_coalesce_requires_pending_builds = (
            prefill_coalesce_requires_pending_builds
        )
        self.enable_pre_lm_encoder = enable_pre_lm_encoder
        self.pre_lm_cache_max_entries = pre_lm_cache_max_entries
        self.pre_lm_cache_size_bytes = pre_lm_cache_size_bytes
        self.pre_lm_max_batch_size = pre_lm_max_batch_size
        self.pre_lm_max_batch_wait_ms = pre_lm_max_batch_wait_ms
        self.pre_lm_max_pending = pre_lm_max_pending
        self.enable_encoder_cuda_graph = enable_encoder_cuda_graph
        self.stream_emit_interval_s = stream_emit_interval_s
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.feature_extractor: WhisperFeatureExtractor | None = None
        self.merge_factor = 4
        self.audio_token_id = 151663
        self.context_length = 0
        self.model_path: str | None = None
        self.audio_encoder_service: ArkasrPreLMEncoderService | None = None
        self.speculative_worker_class: type[BaseSpecWorker] | None = None
        self.draft_checkpoint: TemporaryDirectory[str] | None = None

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        self.model_path = checkpoint_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_dir, trust_remote_code=True
        )
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(checkpoint_dir)
        hf_config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
        self.merge_factor = int(getattr(hf_config, "merge_factor", 4))
        self.audio_token_id = int(getattr(hf_config, "audio_token_id", 151663))
        encoder_token_count = self.feature_extractor.nb_max_frames // 2
        self.context_length = encoder_token_count + self.max_new_tokens + 8

    def validate_before_infrastructure(self, server_args: ServerArgs) -> None:
        """Reject incompatible runtime settings before allocating engine resources."""
        cfg = resolved_view(server_args)
        # NPU pre-LM graphs need independent Host queues so a blocked stream does not stall submission on another stream. See the Host queue guidance in docs/other_platforms/installation_npu.md.
        if (
            current_platform.is_npu()
            and self.enable_pre_lm_encoder
            and (
                self.enable_encoder_cuda_graph
                or cfg.cuda_graph_config.decode.backend != CudaGraphBackend.DISABLED
                or cfg.cuda_graph_config.prefill.backend != CudaGraphBackend.DISABLED
            )
            and (
                os.environ.get("PER_STREAM_QUEUE") != "1"
                or os.environ.get("TASK_QUEUE_ENABLE", "1") != "1"
                or os.environ.get("ASCEND_LAUNCH_BLOCKING") == "1"
            )
        ):
            raise ValueError(
                "ARK-ASR NPU pre-LM graphs require PER_STREAM_QUEUE=1, "
                "TASK_QUEUE_ENABLE unset or 1, and ASCEND_LAUNCH_BLOCKING unset or 0 "
                "before starting the server"
            )
        else:
            pass
        super().validate_before_infrastructure(server_args)

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        defaults: GenerationDefaults = {
            "max_running_requests": self.max_running_requests,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "enable_torch_compile": self.enable_torch_compile,
            "mem_fraction_static": self.mem_fraction_static,
            "max_prefill_tokens": 4096,
            "chunked_prefill_size": 4096,
            "sampling_backend": "pytorch",
            "dtype": dtype,
            "cuda_graph_backend_prefill": CudaGraphBackend.BREAKABLE,
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

    def adjust_overrides(self, overrides: dict[str, object]) -> None:
        has_context_length_override = "context_length" in overrides
        if has_context_length_override:
            self.context_length = int(overrides.pop("context_length"))
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
            if self.enable_async_decode is None:
                self.enable_async_decode = True
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
            self.draft_checkpoint = materialize_draft_checkpoint(
                str(draft_model_path), overrides.get("speculative_draft_model_revision")
            )
            overrides["speculative_draft_model_path"] = self.draft_checkpoint.name
            overrides["speculative_draft_model_revision"] = None
            self.speculative_worker_class = ReducedVocabularyDFlashWorker
            # note (wilsonzheng0327): async decode under a draft means speculative overlap, so it stays off unless the operator asks for it.
            if self.enable_async_decode is None:
                self.enable_async_decode = False
            else:
                pass
            overrides["disable_overlap_schedule"] = not self.enable_async_decode
            # note (wilsonzheng0327): on SGLang 0.5.23 the compiled path makes each DFlash verify step several times slower (no gain at c1 despite 6.8 accepted tokens), while decode without a draft still gains from it.
            overrides["enable_torch_compile"] = False
            if not has_context_length_override:
                draft_config = json.loads(
                    (Path(self.draft_checkpoint.name) / "config.json").read_text()
                )
                draft_tokens = overrides.get("speculative_num_draft_tokens")
                if draft_tokens is None:
                    draft_tokens = draft_config["dflash_config"]["block_size"]
                else:
                    pass
                # note (Ruilin Gao): verification reserves two effective draft blocks.
                self.context_length += 2 * int(draft_tokens)
            else:
                pass

    def infra_kwargs(self) -> InfrastructureOptions:
        if self.speculative_worker_class is None:
            return {}
        else:
            # note (wilsonzheng0327): the worker is built for overlap or for the synchronous verify loop, so the choice must reach it at construction.
            return {
                "supports_speculative_decoding": True,
                "speculative_worker_class": self.speculative_worker_class,
                "enable_async_decode": bool(self.enable_async_decode),
            }

    def make_model_runner(
        self,
        model_worker: ModelWorker,
        output_proc: SGLangOutputProcessor,
    ) -> ArkasrModelRunner:
        return ArkasrModelRunner(model_worker, output_proc)

    def customize_server_args(self, server_args: ServerArgs) -> None:
        self.context_length = int(server_args.context_length)

    def setup_model_resources(
        self,
        model: ArkasrForConditionalGeneration,
        server_args: ServerArgs,
        *,
        generation_cuda_graph_enabled: bool,
    ) -> None:
        del generation_cuda_graph_enabled
        model.set_encoder_max_batch_size(self.encoder_max_batch_size)
        if self.enable_encoder_cuda_graph:
            from sglang_omni.models.arkasr.encoder_cuda_graph import (
                ArkasrEncoderCudaGraphRunner,
            )

            runner = ArkasrEncoderCudaGraphRunner(
                model.audio_encoder,
                max_batch_size=self.encoder_max_batch_size,
                max_mel_frames=self.feature_extractor.nb_max_frames,
            )
            runner.capture_working_set(self.feature_extractor.feature_size)
            model.encoder_cuda_graph_runner = runner
            logger.info(
                "ARK-ASR encoder CUDA graphs enabled (working-set precapture, max_batch=%d)",
                self.encoder_max_batch_size,
            )
        else:
            pass
        init_mm_embedding_cache(self.mm_embedding_cache_size_bytes)
        if self.enable_pre_lm_encoder:
            # note (guozhihao-224): constructed after generation CUDA graphs so the
            # encoder's dedicated stream never interleaves with graph capture.
            self.audio_encoder_service = ArkasrPreLMEncoderService(
                model,
                cache_namespace=build_cache_namespace(
                    model,
                    model_path=self.model_path or "",
                    feature_extractor=self.feature_extractor,
                    mm_attention_backend=getattr(
                        server_args, "mm_attention_backend", None
                    ),
                ),
                cache_max_entries=self.pre_lm_cache_max_entries,
                cache_max_bytes=self.pre_lm_cache_size_bytes,
                max_batch_size=self.pre_lm_max_batch_size,
                max_batch_wait_ms=self.pre_lm_max_batch_wait_ms,
                max_queue_size=self.pre_lm_max_pending,
            )
        else:
            pass

    def make_adapters(self, model: object) -> tuple[
        Callable[
            [StagePayload], ArkASRRequestData | DeferredAdmission[ArkASRRequestData]
        ],
        Callable[[ArkASRRequestData], StagePayload],
    ]:
        del model
        return request_builders.make_arkasr_scheduler_adapters(
            tokenizer=self.tokenizer,
            feature_extractor=self.feature_extractor,
            max_new_tokens=self.max_new_tokens,
            merge_factor=self.merge_factor,
            audio_token_id=self.audio_token_id,
            audio_encoder_service=self.audio_encoder_service,
        )

    def extra_scheduler_callbacks(self) -> dict[str, Callable[[], None]]:
        if self.audio_encoder_service is None:
            return {}
        else:
            pass
        return {"shutdown_callback": self.audio_encoder_service.close}

    def cleanup_build_failure(self) -> None:
        if self.audio_encoder_service is not None:
            self.audio_encoder_service.close()
            self.audio_encoder_service = None
        else:
            pass

    def extra_scheduler_kwargs(self) -> SchedulerExtras[ArkASRRequestData]:
        return {
            "stream_output_builder": request_builders.make_arkasr_stream_output_builder(
                tokenizer=self.tokenizer,
                min_emit_interval_s=self.stream_emit_interval_s,
            ),
            "enable_async_decode": self.enable_async_decode,
            "async_decode_min_batch_size": self.async_decode_min_batch_size,
            "request_build_max_workers": self.request_build_max_workers,
            "request_build_max_pending": self.request_build_max_pending,
            "prefill_coalesce_requests": self.prefill_coalesce_requests,
            "prefill_coalesce_wait_ms": self.prefill_coalesce_wait_ms,
            "prefill_coalesce_when_idle": self.prefill_coalesce_when_idle,
            "prefill_coalesce_requires_pending_builds": (
                self.prefill_coalesce_requires_pending_builds
            ),
        }
