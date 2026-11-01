# SPDX-License-Identifier: Apache-2.0
"""SGLang engine builder for Nemotron Transcribe."""

from __future__ import annotations

import atexit
import json
import logging
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from sglang.kernels.ops.mamba.triton_ops import (
    initialize_mamba_selective_state_update_backend,
)
from sglang.srt.managers.mm_utils import init_mm_embedding_cache
from sglang.srt.server_args import ServerArgs
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from sglang_omni.models.nemotron_transcribe.config import (
    DEFAULT_MAX_RUNNING_REQUESTS,
    EncoderCompileMode,
)
from sglang_omni.models.nemotron_transcribe.encoder_service import (
    NemotronTranscribePreLMEncoderService,
    build_cache_namespace,
)
from sglang_omni.models.nemotron_transcribe.hf_config import (
    MODEL_ARCH_OVERRIDE,
    NemotronTranscribeConfig,
    register_nemotron_transcribe_hf_config,
)
from sglang_omni.models.nemotron_transcribe.request_builders import (
    NemotronTranscribeRequestData,
    RequestBuilder,
    ResultAdapter,
    make_scheduler_adapters,
    make_stream_output_builder,
)
from sglang_omni.models.nemotron_transcribe.sglang_model import (
    NemotronTranscribeForConditionalGeneration,
)
from sglang_omni.models.nemotron_transcribe.speech_transformer import (
    SpeechTransformerEncoder,
)
from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.scheduling.engine_factory import (
    AsrEngineBuilder,
    GenerationDefaults,
    SchedulerExtras,
)
from sglang_omni.scheduling.generation_batch_policy import CudaGraphBackend

logger = logging.getLogger(__name__)


def create_checkpoint_shim(checkpoint_directory: Path) -> Path:
    checkpoint_directory = checkpoint_directory.resolve()
    shim_directory = Path(tempfile.mkdtemp(prefix="sglang-omni-nemotron-transcribe-"))
    atexit.register(shutil.rmtree, shim_directory, ignore_errors=True)
    for checkpoint_file in checkpoint_directory.iterdir():
        if checkpoint_file.name != "config.json":
            (shim_directory / checkpoint_file.name).symlink_to(checkpoint_file)
        else:
            pass
    return shim_directory


class NemotronTranscribeEngineBuilder(AsrEngineBuilder[NemotronTranscribeRequestData]):
    model_name = "Nemotron-Transcribe"
    model_arch_override = MODEL_ARCH_OVERRIDE
    supports_context_length_override = True

    def __init__(
        self,
        *,
        max_new_tokens: int,
        context_length: int,
        max_audio_seconds: float | None,
        mem_fraction_static: float | None,
        mm_embedding_cache_size_bytes: int,
        chunked_prefill_size: int,
        encoder_max_batch_size: int,
        enable_pre_lm_encoder: bool,
        pre_lm_cache_max_entries: int,
        pre_lm_cache_size_bytes: int,
        pre_lm_max_batch_wait_ms: int,
        request_build_max_workers: int,
        request_build_max_pending: int | None,
        stream_emit_interval_s: float,
        compile_perception: bool,
        perception_cuda_graphs: bool,
        perception_compile_mode: EncoderCompileMode,
    ) -> None:
        if perception_cuda_graphs and not compile_perception:
            raise ValueError("perception_cuda_graphs requires compile_perception=true")
        else:
            pass
        self.max_new_tokens = max_new_tokens
        self.context_length = context_length
        self.max_audio_seconds = max_audio_seconds
        self.mem_fraction_static = mem_fraction_static
        self.mm_embedding_cache_size_bytes = mm_embedding_cache_size_bytes
        self.chunked_prefill_size = chunked_prefill_size
        self.encoder_max_batch_size = encoder_max_batch_size
        self.enable_pre_lm_encoder = bool(enable_pre_lm_encoder)
        self.pre_lm_cache_max_entries = int(pre_lm_cache_max_entries)
        self.pre_lm_cache_size_bytes = int(pre_lm_cache_size_bytes)
        self.pre_lm_max_batch_wait_ms = int(pre_lm_max_batch_wait_ms)
        self.request_build_max_workers = request_build_max_workers
        self.request_build_max_pending = request_build_max_pending
        self.stream_emit_interval_s = stream_emit_interval_s
        self.compile_perception = compile_perception
        self.perception_cuda_graphs = perception_cuda_graphs
        self.perception_compile_mode = perception_compile_mode
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.hf_config: NemotronTranscribeConfig | None = None
        self.audio_encoder_service: NemotronTranscribePreLMEncoderService | None = None
        self.should_wait_for_encode_fn: Callable[[], bool] | None = None

    def resolve_checkpoint(self, model_path: str) -> str:
        register_nemotron_transcribe_hf_config()
        checkpoint_directory = Path(resolve_model_path(model_path))
        checkpoint_config = json.loads(
            (checkpoint_directory / "config.json").read_text()
        )
        if checkpoint_config.get("model_type") == "nemotron_transcribe":
            self.hf_config = NemotronTranscribeConfig.from_dict(checkpoint_config)
            return str(checkpoint_directory)
        else:
            self.hf_config = NemotronTranscribeConfig.from_nemo_dict(checkpoint_config)
            shim_directory = create_checkpoint_shim(checkpoint_directory)
            (shim_directory / "config.json").write_text(self.hf_config.to_json_string())
            return str(shim_directory)

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        return {
            "max_running_requests": DEFAULT_MAX_RUNNING_REQUESTS,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_torch_compile": False,
            "mem_fraction_static": self.mem_fraction_static,
            "max_prefill_tokens": self.chunked_prefill_size,
            "chunked_prefill_size": self.chunked_prefill_size,
            "sampling_backend": "pytorch",
            "dtype": dtype,
            "cuda_graph_backend_prefill": CudaGraphBackend.DISABLED,
        }

    def customize_server_args(self, server_args: ServerArgs) -> None:
        if server_args.enable_torch_compile:
            raise ValueError(
                "Nemotron Transcribe decoder compilation is unsupported; use "
                "--asr.factory.compile_perception true to compile the encoder"
            )
        else:
            pass
        self.context_length = int(server_args.context_length)
        initialize_mamba_selective_state_update_backend(server_args)

    def setup_model_resources(
        self,
        model: NemotronTranscribeForConditionalGeneration,
        server_args: ServerArgs,
        *,
        generation_cuda_graph_enabled: bool,
    ) -> None:
        """The builder interface invokes this after decoder graph capture."""
        model.encoder_max_batch_size = self.encoder_max_batch_size
        init_mm_embedding_cache(self.mm_embedding_cache_size_bytes)
        if self.compile_perception:
            device = next(model.perception.parameters()).device
            model.perception.to(device=device)
            for module in model.perception.modules():
                if isinstance(module, SpeechTransformerEncoder):
                    module.compile_layers(
                        self.perception_compile_mode,
                        cuda_graphs=self.perception_cuda_graphs,
                        dynamic=server_args.max_running_requests > 1,
                    )
                else:
                    pass
        else:
            pass

    def setup_runtime_resources(
        self,
        model: NemotronTranscribeForConditionalGeneration,
        server_args: ServerArgs,
    ) -> None:
        """Start perception encode after decoder graphs and encoder compile."""
        del server_args
        if not self.enable_pre_lm_encoder:
            return
        else:
            pass
        self.audio_encoder_service = NemotronTranscribePreLMEncoderService(
            model,
            cache_namespace=build_cache_namespace(
                model, model_path=self.checkpoint_dir
            ),
            cache_max_entries=self.pre_lm_cache_max_entries,
            cache_max_bytes=self.pre_lm_cache_size_bytes,
            max_batch_size=self.encoder_max_batch_size,
            max_batch_wait_ms=self.pre_lm_max_batch_wait_ms,
        )
        logger.info(
            "Nemotron Transcribe pre-LM encoder enabled (max_batch=%d, "
            "cache entries=%d, cache bytes=%d, batch wait=%dms)",
            self.encoder_max_batch_size,
            self.pre_lm_cache_max_entries,
            self.pre_lm_cache_size_bytes,
            self.pre_lm_max_batch_wait_ms,
        )

    def should_wait_for_encode(self) -> bool:
        if self.should_wait_for_encode_fn is None:
            return False
        else:
            return self.should_wait_for_encode_fn()

    def make_adapters(
        self,
        model: NemotronTranscribeForConditionalGeneration,
    ) -> tuple[RequestBuilder, ResultAdapter]:
        """The builder interface supplies the model; requests use its resolved config."""
        assert self.tokenizer is not None and self.hf_config is not None
        return make_scheduler_adapters(
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
            audio_token_estimator=self.hf_config.audio_token_estimator,
            audio_token_id=self.hf_config.audio_token_id,
            eos_token_id=self.hf_config.eos_token_id,
            max_audio_seconds=self.max_audio_seconds,
            vocab_size=self.hf_config.vocab_size,
            audio_encoder_service=self.audio_encoder_service,
            should_wait_for_encode=self.should_wait_for_encode,
        )

    def extra_scheduler_kwargs(self) -> SchedulerExtras[NemotronTranscribeRequestData]:
        assert self.tokenizer is not None and self.hf_config is not None
        return {
            "stream_output_builder": make_stream_output_builder(
                tokenizer=self.tokenizer,
                min_emit_interval_s=self.stream_emit_interval_s,
                eos_token_id=self.hf_config.eos_token_id,
            ),
            "request_build_max_workers": self.request_build_max_workers,
            "request_build_max_pending": self.request_build_max_pending,
        }

    def post_scheduler_setup(self, scheduler: object, model_runner: object) -> None:
        del model_runner
        self.should_wait_for_encode_fn = scheduler.request_build_queue_fits_workers

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
