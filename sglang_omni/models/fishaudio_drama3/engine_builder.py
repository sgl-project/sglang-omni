# SPDX-License-Identifier: Apache-2.0
"""Drama 3 SGLang engine builder: the AR stage that samples frames."""

from __future__ import annotations

import logging

import torch
from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.quantization import QUANTIZATION_METHODS
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.model_executor.cuda_graph_config import Backend as CudaGraphBackend
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.server_args import ServerArgs
from transformers import AutoConfig

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.request_id_prefill_cuda_graph_runner import (
    RequestIdPrefillCudaGraphRunner,
)
from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner
from sglang_omni.models.fishaudio_drama3 import CAPABILITIES
from sglang_omni.models.fishaudio_drama3.config import (
    DEFAULT_ASYNC_DECODE_MIN_BATCH_SIZE,
    DEFAULT_ENABLE_ASYNC_DECODE,
    DEFAULT_FIRST_FRAME_CUDA_GRAPH_MAX_REQUESTS,
    DEFAULT_FIRST_FRAME_CUDA_GRAPHS,
    DEFAULT_MAX_NEW_TOKENS_CAP,
    DEFAULT_MAX_PROMPT_ROWS,
    DEFAULT_STREAM_FOLLOWUP_STRIDE,
    DEFAULT_STREAM_FOLLOWUP_STRIDE_RAMP,
    DEFAULT_STREAM_STRIDE,
    DEFAULT_TOWER_FUSED_LAYERS,
    DEFAULT_TRITON_SAMPLING,
)
from sglang_omni.models.fishaudio_drama3.first_frame_graphs import (
    Drama3FirstFrameGraphs,
)
from sglang_omni.models.fishaudio_drama3.hf_config import FishOmniConfig
from sglang_omni.models.fishaudio_drama3.model_inputs import Drama3DecodeBuffers
from sglang_omni.models.fishaudio_drama3.model_runner import Drama3ModelRunner
from sglang_omni.models.fishaudio_drama3.quantization import (
    INT4_QUANTIZATION_METHOD,
    QUANTIZATION_METHOD,
    ExpertDenseInt4Config,
    ExpertFp8Config,
)
from sglang_omni.models.fishaudio_drama3.request_builders import (
    Drama3FrameStreamer,
    Drama3RequestBuilder,
    Drama3RequestData,
    Drama3ResultAdapter,
)
from sglang_omni.models.fishaudio_drama3.sampling_kernels import (
    PHILOX_SEED_MODULUS,
    CodebookSampler,
    triton_sampling_unsupported_reason,
)
from sglang_omni.models.fishaudio_drama3.sglang_model import Drama3ForCausalLM
from sglang_omni.models.fishaudio_drama3.tower_kernels import (
    FUSED_KERNELS,
    MAX_FUSED_ROWS,
)
from sglang_omni.models.fishaudio_drama3.tower_prefill import Drama3TowerPrefill
from sglang_omni.scheduling.engine_factory import (
    GenerationDefaults,
    SchedulerExtras,
    SGLangGenerationEngineBuilder,
)
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.utils.checkpoint import resolve_checkpoint

FISH_OMNI_ARCHITECTURE = "FishOmniModel"
# The tower re-plans this backend's metadata for its first frame.
ATTENTION_BACKEND = "triton"
PREFILL_CUDA_GRAPH_MAX_TOKENS = 2048
DECODE_STEPS_BETWEEN_PREFILLS = 4

logger = logging.getLogger(__name__)


class Drama3EngineBuilder(SGLangGenerationEngineBuilder[Drama3RequestData]):
    """Single-GPU engine replaying decode CUDA graphs; no radix, chunked prefill or
    overlap."""

    model_name = "Fish Audio Drama 3"
    supports_breakable_prefill_cuda_graph = (
        CAPABILITIES.supports_breakable_prefill_cuda_graph
    )

    def __init__(
        self,
        *,
        context_length: int,
        max_running_requests: int,
        stream_stride: int = DEFAULT_STREAM_STRIDE,
        stream_followup_stride_ramp: tuple[
            int, ...
        ] = DEFAULT_STREAM_FOLLOWUP_STRIDE_RAMP,
        stream_followup_stride: int = DEFAULT_STREAM_FOLLOWUP_STRIDE,
        max_prompt_rows: int = DEFAULT_MAX_PROMPT_ROWS,
        max_new_tokens_cap: int = DEFAULT_MAX_NEW_TOKENS_CAP,
        first_frame_cuda_graphs: bool = DEFAULT_FIRST_FRAME_CUDA_GRAPHS,
        first_frame_cuda_graph_max_requests: int = (
            DEFAULT_FIRST_FRAME_CUDA_GRAPH_MAX_REQUESTS
        ),
        triton_sampling: bool = DEFAULT_TRITON_SAMPLING,
        tower_fused_layers: bool = DEFAULT_TOWER_FUSED_LAYERS,
        enable_async_decode: bool = DEFAULT_ENABLE_ASYNC_DECODE,
        async_decode_min_batch_size: int = DEFAULT_ASYNC_DECODE_MIN_BATCH_SIZE,
    ) -> None:
        self.context_length = context_length
        self.max_running_requests = max_running_requests
        self.stream_stride = stream_stride
        self.stream_followup_stride_ramp = stream_followup_stride_ramp
        self.stream_followup_stride = stream_followup_stride
        self.max_prompt_rows = max_prompt_rows
        self.max_new_tokens_cap = max_new_tokens_cap
        self.first_frame_cuda_graphs = first_frame_cuda_graphs
        self.first_frame_cuda_graph_max_requests = first_frame_cuda_graph_max_requests
        self.triton_sampling = triton_sampling
        self.tower_fused_layers = tower_fused_layers
        self.enable_async_decode = enable_async_decode
        self.async_decode_min_batch_size = async_decode_min_batch_size
        self.frame_streamer: Drama3FrameStreamer | None = None
        self.sglang_runner: SGLModelRunner | None = None

    def resolve_checkpoint(self, model_path: str) -> str:
        return resolve_checkpoint(model_path)

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        """Register the classes the checkpoint names; the directory is the contract."""
        AutoConfig.register(FishOmniConfig.model_type, FishOmniConfig, exist_ok=True)
        ModelRegistry.models[FISH_OMNI_ARCHITECTURE] = Drama3ForCausalLM
        QUANTIZATION_METHODS[QUANTIZATION_METHOD] = ExpertFp8Config
        QUANTIZATION_METHODS[INT4_QUANTIZATION_METHOD] = ExpertDenseInt4Config

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        return {
            "max_running_requests": self.max_running_requests,
            "disable_cuda_graph": False,
            "cuda_graph_backend_prefill": CudaGraphBackend.BREAKABLE,
            "cuda_graph_max_bs_prefill": PREFILL_CUDA_GRAPH_MAX_TOKENS,
            "prefill_decode_interval": DECODE_STEPS_BETWEEN_PREFILLS,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_mixed_chunk": False,
            "chunked_prefill_size": -1,
            "attention_backend": ATTENTION_BACKEND,
            "enable_torch_compile": False,
            "trust_remote_code": False,
            "dtype": dtype,
        }

    def adjust_overrides(self, overrides: dict[str, object]) -> None:
        overrides.setdefault("page_size", 1)

    def validate_before_infrastructure(self, server_args: ServerArgs) -> None:
        attention_backends = {
            server_args.attention_backend,
            server_args.prefill_attention_backend or server_args.attention_backend,
            server_args.decode_attention_backend or server_args.attention_backend,
        }
        parallel_sizes = (server_args.tp_size, server_args.pp_size, server_args.dp_size)
        chunked_prefill_size = server_args.chunked_prefill_size or 0
        graph_config = resolved_view(server_args).cuda_graph_config
        max_position_embeddings = FishOmniConfig.from_pretrained(
            server_args.model_path
        ).text_config.max_position_embeddings
        if not server_args.disable_radix_cache:
            raise ValueError("Drama 3 needs disable_radix_cache=True")
        elif chunked_prefill_size > 0 or server_args.enable_mixed_chunk:
            raise ValueError(
                "Drama 3 needs chunked_prefill_size <= 0 and no mixed chunk: a "
                "prompt must reach the tower in one extend"
            )
        elif not server_args.disable_overlap_schedule:
            raise ValueError("Drama 3 runs without overlap scheduling")
        elif graph_config.prefill.backend not in (
            CudaGraphBackend.BREAKABLE,
            CudaGraphBackend.DISABLED,
        ):
            raise ValueError(
                f"Drama 3 prefill runs eager or as a breakable CUDA graph; got the "
                f"{graph_config.prefill.backend!r} prefill CUDA graph backend"
            )
        elif graph_config.decode.backend not in (
            CudaGraphBackend.FULL,
            CudaGraphBackend.DISABLED,
        ):
            raise ValueError(
                f"Drama 3 replays full decode CUDA graphs; got the "
                f"{graph_config.decode.backend!r} decode CUDA graph backend"
            )
        elif server_args.enable_torch_compile:
            raise ValueError("Drama 3 does not support torch.compile")
        elif attention_backends != {ATTENTION_BACKEND}:
            raise ValueError(
                f"Drama 3 needs the {ATTENTION_BACKEND} attention backend; got "
                f"{sorted(map(str, attention_backends))}"
            )
        elif server_args.page_size != 1 or server_args.kv_cache_dtype != "auto":
            raise ValueError(
                "Drama 3 keeps page_size=1 and the KV cache in the model dtype"
            )
        elif (
            parallel_sizes != (1, 1, 1) or server_args.speculative_algorithm is not None
        ):
            raise ValueError(
                f"Drama 3 runs on one rank without speculative decoding; got "
                f"tp/pp/dp={parallel_sizes} and "
                f"speculative_algorithm={server_args.speculative_algorithm}"
            )
        elif self.context_length > max_position_embeddings:
            raise ValueError(
                f"Drama 3 context_length ({self.context_length}) exceeds the model's "
                f"max_position_embeddings ({max_position_embeddings}); lower "
                f"--tts_engine.factory.context_length"
            )
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
        """Allocate the decode buffers before graph capture; the checkpoint and
        GPU arguments are the interface contract."""
        sglang_runner = model_worker.model_runner
        model = sglang_runner.model
        assert isinstance(model, Drama3ForCausalLM)
        # No decode batch or captured graph size exceeds the request slots.
        model.decode_buffers = Drama3DecodeBuffers.allocate(
            row_count=sglang_runner.req_to_token_pool.size,
            frame_width=1 + model.config.audio_decoder_config.num_codebooks,
            device=torch.device(device),
        )
        prefill_graph_config = resolved_view(server_args).cuda_graph_config.prefill
        largest_prefill_graph_token_count = (
            0
            if prefill_graph_config.backend == CudaGraphBackend.DISABLED
            else max(prefill_graph_config.bs or [0])
        )
        model.model.tower_prefill = Drama3TowerPrefill.allocate(
            audio_decoder=model.audio_decoder,
            tower_attention=model.tower_attention,
            row_count=max(
                server_args.max_prefill_tokens,
                self.context_length,
                largest_prefill_graph_token_count,
            ),
            frame_width=1 + model.config.audio_decoder_config.num_codebooks,
            device=torch.device(device),
        )
        audio_decoder = model.audio_decoder
        unsupported_reason = triton_sampling_unsupported_reason(
            torch.device(device),
            audio_decoder.codebook_size,
            audio_decoder.main_token_candidate_ids.shape[0],
        )
        if not self.triton_sampling:
            logger.info(
                "Drama 3 codebook sampling: PyTorch, "
                "tts_engine.factory.triton_sampling is false"
            )
        elif unsupported_reason is not None:
            logger.info(f"Drama 3 codebook sampling: PyTorch, {unsupported_reason}")
        else:
            audio_decoder.sampler = CodebookSampler(
                step=audio_decoder.sampling_step,
                seed=torch.cuda.initial_seed() % PHILOX_SEED_MODULUS,
            )
            logger.info("Drama 3 codebook sampling: Triton kernels")
        kv_pool = sglang_runner.token_to_kv_pool
        assert isinstance(kv_pool, HybridLinearKVPool)
        fused_layers_reason = audio_decoder.fused_layers_unsupported_reason(
            kv_pool=kv_pool,
            tower_layer_ids=[layer.layer_id for layer in model.tower_attention],
            context_length=self.context_length,
            device=torch.device(device),
        )
        if not self.tower_fused_layers:
            logger.info(
                "Drama 3 tower fused layers: off, "
                "tts_engine.factory.tower_fused_layers is false"
            )
        elif fused_layers_reason is not None:
            logger.info(f"Drama 3 tower fused layers: off, {fused_layers_reason}")
        else:
            audio_decoder.enable_fused_layers(torch.device(device))
            logger.info(
                f"Drama 3 tower fused layers: on for passes up to {MAX_FUSED_ROWS} "
                f"rows ({', '.join(FUSED_KERNELS)})"
            )
        sglang_runner.extra_prefill_graph_attention_layers = list(model.tower_attention)
        sglang_runner.prefill_graph_runner_class = RequestIdPrefillCudaGraphRunner
        self.sglang_runner = sglang_runner
        if resolved_view(server_args).disable_cuda_graph:
            logger.info(
                "Drama 3 first-frame CUDA graphs: off, CUDA graphs are disabled"
            )
        else:
            pass

    def post_cuda_graph_setup(
        self, model: Drama3ForCausalLM, server_args: ServerArgs
    ) -> None:
        sglang_runner = self.sglang_runner
        assert sglang_runner is not None
        attention_backend = sglang_runner.attn_backend
        decode_graph_runner = sglang_runner.decode_cuda_graph_runner
        decode_graph_config = resolved_view(server_args).cuda_graph_config.decode
        if not self.first_frame_cuda_graphs:
            logger.info(
                "Drama 3 first-frame CUDA graphs: off, "
                "tts_engine.factory.first_frame_cuda_graphs is false"
            )
        elif decode_graph_runner is None:
            logger.info(
                f"Drama 3 first-frame CUDA graphs: off, no decode CUDA graph "
                f"(decode backend {decode_graph_config.backend})"
            )
        else:
            assert isinstance(attention_backend, HybridLinearAttnBackend)
            max_request_count = min(
                self.first_frame_cuda_graph_max_requests, decode_graph_runner.max_bs
            )
            text_config = model.config.text_config
            first_frame_graphs = Drama3FirstFrameGraphs.capture(
                sample_first_frames=model.sample_first_frames,
                attention_backend=attention_backend,
                max_request_count=max_request_count,
                vocab_size=text_config.vocab_size,
                hidden_size=text_config.hidden_size,
                hidden_dtype=sglang_runner.model_config.dtype,
                frame_width=1 + model.config.audio_decoder_config.num_codebooks,
                device=model.lm_head.weight.device,
            )
            model.first_frame_graphs = first_frame_graphs
            logger.info(
                f"Drama 3 first-frame CUDA graphs: on for 1..{max_request_count} "
                f"requests (first_frame_cuda_graph_max_requests "
                f"{self.first_frame_cuda_graph_max_requests}, decode CUDA graph max "
                f"batch {decode_graph_runner.max_bs}), captured in "
                f"{first_frame_graphs.capture_seconds:.2f} s, static inputs "
                f"{first_frame_graphs.static_input_bytes / 2**20:.1f} MiB, reserved "
                f"memory +{first_frame_graphs.reserved_bytes / 2**20:.1f} MiB"
            )

    def make_model_runner(
        self, model_worker: ModelWorker, output_proc: SGLangOutputProcessor
    ) -> Drama3ModelRunner:
        return Drama3ModelRunner(model_worker, output_proc)

    def make_adapters(
        self, model: Drama3ForCausalLM
    ) -> tuple[Drama3RequestBuilder, Drama3ResultAdapter]:
        self.frame_streamer = Drama3FrameStreamer(
            eos_token_id=model.config.eos_token_id,
            stream_stride=self.stream_stride,
            stream_followup_stride_ramp=self.stream_followup_stride_ramp,
            stream_followup_stride=self.stream_followup_stride,
        )
        return (
            Drama3RequestBuilder(
                config=model.config,
                context_length=self.context_length,
                max_prompt_rows=self.max_prompt_rows,
                max_new_tokens_cap=self.max_new_tokens_cap,
            ),
            Drama3ResultAdapter(eos_token_id=model.config.eos_token_id),
        )

    def extra_scheduler_kwargs(self) -> SchedulerExtras[Drama3RequestData]:
        """The frame streamer make_adapters created, which build calls first."""
        assert self.frame_streamer is not None
        if self.enable_async_decode:
            logger.info(
                f"Drama 3 async decode: on, lookahead from batch size "
                f"{self.async_decode_min_batch_size}"
            )
        else:
            logger.info(
                "Drama 3 async decode: off, "
                "tts_engine.factory.enable_async_decode is false"
            )
        return {
            "stream_output_builder": self.frame_streamer,
            "enable_async_decode": self.enable_async_decode,
            "async_decode_min_batch_size": self.async_decode_min_batch_size,
            "async_decode_resolve_before_prefill": True,
        }
