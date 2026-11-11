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

from huggingface_hub import snapshot_download
from sglang.kernels.ops.mamba.triton_ops import (
    initialize_mamba_selective_state_update_backend,
)
from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.managers.mm_utils import init_mm_embedding_cache
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
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
    BACKBONE_ARCH,
    MODEL_ARCH_OVERRIDE,
    NemotronTranscribeConfig,
    register_nemotron_transcribe_hf_config,
)
from sglang_omni.models.nemotron_transcribe.mtp_model import NemotronTranscribeMTP
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
from sglang_omni.scheduling.bootstrap import InfrastructureOptions
from sglang_omni.scheduling.engine_factory import (
    AsrEngineBuilder,
    GenerationDefaults,
    SchedulerExtras,
)
from sglang_omni.scheduling.generation_batch_policy import CudaGraphBackend

logger = logging.getLogger(__name__)
PINNED_CHECKPOINT_REVISION = "c17667641642c4265ff2b7037dd101b7d456763"


def prepare_nemotron_speculative_options(
    model_path: str,
    server_args_overrides: dict[str, object] | None,
    *,
    enable_async_decode: bool = False,
) -> tuple[dict[str, object], bool]:
    """Validate this model's bundled NEXTN options without touching the checkpoint."""
    overrides = dict(server_args_overrides or {})
    algorithm_name = overrides.get("speculative_algorithm")
    if algorithm_name is not None and not isinstance(algorithm_name, str):
        raise ValueError("speculative_algorithm must be a string")
    algorithm = SpeculativeAlgorithm.from_string(algorithm_name)
    if algorithm.is_none():
        return overrides, False
    if enable_async_decode or overrides.get("enable_async_decode") is True:
        raise ValueError("Nemotron Transcribe does not support async decode")
    if "enable_async_decode" in overrides:
        overrides.pop("enable_async_decode")
    if algorithm != SpeculativeAlgorithm.EAGLE:
        raise ValueError("Nemotron Transcribe supports bundled NEXTN only")

    revision = overrides.get("revision")
    if revision not in (None, PINNED_CHECKPOINT_REVISION):
        raise ValueError(
            "Nemotron Transcribe NEXTN requires checkpoint revision "
            f"{PINNED_CHECKPOINT_REVISION}"
        )
    elif overrides.get("speculative_draft_model_path") not in (None, model_path):
        raise ValueError(
            "Nemotron Transcribe NEXTN draft must use the target checkpoint"
        )
    elif overrides.get("speculative_draft_model_revision") not in (
        None,
        PINNED_CHECKPOINT_REVISION,
    ):
        raise ValueError(
            "Nemotron Transcribe NEXTN draft must use the target checkpoint revision"
        )
    elif overrides.get("speculative_eagle_topk") not in (None, 1):
        raise ValueError("Nemotron Transcribe NEXTN requires speculative_eagle_topk=1")
    elif overrides.get("disable_overlap_schedule") is False:
        raise ValueError("Nemotron Transcribe NEXTN requires synchronous execution")
    elif overrides.get("enable_mixed_chunk") is True:
        raise ValueError("Nemotron Transcribe NEXTN does not support mixed-chunk mode")
    elif overrides.get("enable_multi_layer_eagle") is True:
        raise ValueError("Nemotron Transcribe NEXTN supports one logical MTP step")
    elif overrides.get("speculative_use_rejection_sampling") is True:
        raise ValueError("Nemotron Transcribe NEXTN supports greedy decoding only")
    for name in ("tp_size", "pp_size"):
        if overrides.get(name, 1) != 1:
            raise ValueError(f"Nemotron Transcribe NEXTN requires {name}=1")
    overrides.update(
        speculative_algorithm="NEXTN",
        speculative_draft_model_path=model_path,
        speculative_draft_model_revision=PINNED_CHECKPOINT_REVISION,
        speculative_eagle_topk=1,
        revision=PINNED_CHECKPOINT_REVISION,
        enable_mixed_chunk=False,
    )
    return overrides, True


def validate_nemotron_mtp_metadata(checkpoint_config: dict[str, object]) -> None:
    """Reject checkpoints whose bundled MTP layout differs from this adapter."""
    decoder_config = checkpoint_config.get("llm_config", checkpoint_config)
    mtp_config = checkpoint_config.get("mtp")
    if not isinstance(decoder_config, dict) or not isinstance(mtp_config, dict):
        raise ValueError(
            "Nemotron Transcribe NEXTN requires llm_config and mtp metadata"
        )
    elif decoder_config.get("num_nextn_predict_layers") != 1:
        raise ValueError("Nemotron Transcribe NEXTN requires one logical MTP step")
    elif mtp_config.get("enabled") is not True:
        raise ValueError("Nemotron Transcribe checkpoint does not enable bundled MTP")
    elif mtp_config.get("trained_depths") != 2:
        raise ValueError("Nemotron Transcribe NEXTN requires two physical MTP layers")
    elif mtp_config.get("use_repeated_layer") is not True:
        raise ValueError("Nemotron Transcribe NEXTN requires repeated MTP layers")
    elif mtp_config.get("hybrid_override_pattern") != "*E":
        raise ValueError("Nemotron Transcribe NEXTN requires MTP pattern '*E'")
    else:
        pass


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
        nextn_enabled: bool = False,
        checkpoint_revision: str | None = None,
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
        self.nextn_enabled = nextn_enabled
        self.checkpoint_revision = checkpoint_revision
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.hf_config: NemotronTranscribeConfig | None = None
        self.audio_encoder_service: NemotronTranscribePreLMEncoderService | None = None
        self.should_wait_for_encode_fn: Callable[[], bool] | None = None

    def resolve_checkpoint(self, model_path: str) -> str:
        register_nemotron_transcribe_hf_config()
        if self.nextn_enabled:
            if self.checkpoint_revision != PINNED_CHECKPOINT_REVISION:
                raise ValueError(
                    "Nemotron Transcribe NEXTN requires the pinned checkpoint revision"
                )
            else:
                pass
            if Path(model_path).exists():
                checkpoint_directory = Path(model_path)
            else:
                checkpoint_directory = Path(
                    snapshot_download(
                        model_path,
                        revision=self.checkpoint_revision,
                        local_files_only=False,
                    )
                )
        else:
            checkpoint_directory = Path(resolve_model_path(model_path))
        checkpoint_config = json.loads(
            (checkpoint_directory / "config.json").read_text()
        )
        if self.nextn_enabled:
            validate_nemotron_mtp_metadata(checkpoint_config)
        else:
            pass
        if checkpoint_config.get("model_type") == "nemotron_transcribe":
            self.hf_config = NemotronTranscribeConfig.from_dict(checkpoint_config)
            if self.nextn_enabled:
                mtp_config = checkpoint_config["mtp"]
                assert isinstance(mtp_config, dict)
                self.hf_config.architectures = [BACKBONE_ARCH]
                self.hf_config.mtp_hybrid_override_pattern = str(
                    mtp_config["hybrid_override_pattern"]
                )
                shim_directory = create_checkpoint_shim(checkpoint_directory)
                (shim_directory / "config.json").write_text(
                    self.hf_config.to_json_string()
                )
                return str(shim_directory)
            else:
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

    def adjust_overrides(self, overrides: dict[str, object]) -> None:
        if self.nextn_enabled:
            overrides["speculative_draft_model_path"] = self.checkpoint_dir
            overrides["speculative_draft_model_revision"] = self.checkpoint_revision
        else:
            pass

    def validate_before_infrastructure(self, server_args: ServerArgs) -> None:
        super().validate_before_infrastructure(server_args)
        if not self.nextn_enabled:
            return
        else:
            pass
        configuration = resolved_view(server_args)
        if SpeculativeAlgorithm.from_string(configuration.speculative_algorithm) != (
            SpeculativeAlgorithm.EAGLE
        ):
            raise ValueError("Nemotron Transcribe NEXTN was not enabled in ServerArgs")
        elif configuration.speculative_draft_model_path != self.checkpoint_dir:
            raise ValueError(
                "Nemotron Transcribe NEXTN draft must use the resolved "
                "target checkpoint"
            )
        elif configuration.revision != self.checkpoint_revision:
            raise ValueError("Nemotron Transcribe NEXTN target revision mismatch")
        elif configuration.speculative_draft_model_revision != self.checkpoint_revision:
            raise ValueError("Nemotron Transcribe NEXTN draft revision mismatch")
        elif configuration.speculative_eagle_topk != 1:
            raise ValueError(
                "Nemotron Transcribe NEXTN requires speculative_eagle_topk=1"
            )
        elif configuration.enable_mixed_chunk:
            raise ValueError(
                "Nemotron Transcribe NEXTN does not support mixed-chunk mode"
            )
        elif configuration.enable_multi_layer_eagle:
            raise ValueError("Nemotron Transcribe NEXTN supports one logical MTP step")
        elif not configuration.disable_overlap_schedule:
            raise ValueError("Nemotron Transcribe NEXTN requires synchronous execution")
        elif configuration.tp_size != 1 or configuration.pp_size != 1:
            raise ValueError("Nemotron Transcribe NEXTN requires tp_size=pp_size=1")
        else:
            pass

    def infra_kwargs(self) -> InfrastructureOptions:
        if self.nextn_enabled:
            ModelRegistry.models["NemotronHForCausalLMMTP"] = NemotronTranscribeMTP
            return {"supports_speculative_decoding": True}
        else:
            return {}

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
            greedy_only=self.nextn_enabled,
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
