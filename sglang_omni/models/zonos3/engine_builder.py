# SPDX-License-Identifier: Apache-2.0
"""Zonos3 SGLang engine builder: the AR stage that samples delayed frame rows."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile

from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.server_args import ServerArgs
from transformers import AutoConfig

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.zonos3 import CAPABILITIES
from sglang_omni.models.zonos3.gdn_prefill_graph import install_gdn_prefill_graph
from sglang_omni.models.zonos3.hf_config import (
    CHECKPOINT_FILE_NAME,
    LANGUAGE_MODEL_DIRECTORY,
    MODEL_TYPE,
    SGLANG_ARCHITECTURE,
    Zonos3Config,
    load_params,
    zonos3_config_from_params,
)
from sglang_omni.models.zonos3.model_runner import Zonos3ModelRunner
from sglang_omni.models.zonos3.request_builders import (
    Zonos3RequestBuilder,
    Zonos3RequestData,
    Zonos3ResultAdapter,
    Zonos3StreamOutputBuilder,
)
from sglang_omni.models.zonos3.sglang_model import Zonos3SGLangModel
from sglang_omni.platforms import current_platform
from sglang_omni.scheduling.engine_factory import (
    GenerationDefaults,
    SchedulerExtras,
    SGLangGenerationEngineBuilder,
)
from sglang_omni.scheduling.generation_batch_policy import CudaGraphBackend
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.vendor.sglang.server_args import override_server_args

ATTENTION_BACKEND = "triton"
PAGE_SIZE = 1
# note (zhaochenyang20): A 60 s voice reference is about 900 prompt rows at 15 frames per second; larger prefills run eagerly. SeedTTS EN prompts (sheared reference, three condition rows, three silence rows) are 69-178 rows, and at concurrency 16 every prefill batch, at most 1275 rows, replayed a captured bucket on one H100.
PREFILL_CUDA_GRAPH_MAX_TOKENS = 2048
# The file name the SGLang loader looks for next to config.json.
SHIM_CHECKPOINT_FILE_NAME = "pytorch_model.bin"


class Zonos3EngineBuilder(SGLangGenerationEngineBuilder[Zonos3RequestData]):
    """Single-GPU engine; no radix cache or chunked prefill; async decode overlaps the host and device steps."""

    model_name = "Zonos3"
    supports_breakable_prefill_cuda_graph = (
        CAPABILITIES.supports_breakable_prefill_cuda_graph
    )

    def __init__(
        self,
        *,
        context_length: int,
        max_running_requests: int,
        mem_fraction_static: float | None,
        cuda_graph: bool,
        async_decode: bool,
        vocoder_initial_chunk_frames: int,
        vocoder_steady_chunk_frames: int,
    ) -> None:
        self.context_length = context_length
        self.max_running_requests = max_running_requests
        self.mem_fraction_static = mem_fraction_static
        self.cuda_graph = cuda_graph
        self.async_decode = async_decode
        self.vocoder_initial_chunk_frames = vocoder_initial_chunk_frames
        self.vocoder_steady_chunk_frames = vocoder_steady_chunk_frames
        self.model: Zonos3SGLangModel | None = None

    def resolve_checkpoint(self, model_path: str) -> str:
        """A directory the SGLang loader reads: config.json plus the language model weights."""
        language_model_directory = os.path.join(
            resolve_checkpoint(model_path), LANGUAGE_MODEL_DIRECTORY
        )
        config = zonos3_config_from_params(load_params(language_model_directory))
        shim_directory = tempfile.mkdtemp(prefix="zonos3_sglang_")
        atexit.register(shutil.rmtree, shim_directory, ignore_errors=True)
        config.save_pretrained(shim_directory)
        os.symlink(
            os.path.join(
                os.path.abspath(language_model_directory), CHECKPOINT_FILE_NAME
            ),
            os.path.join(shim_directory, SHIM_CHECKPOINT_FILE_NAME),
        )
        return shim_directory

    def pre_infra_setup(self, checkpoint_directory: str) -> None:
        """Register the classes the shim config names and the gated delta net backend; the directory is the contract."""
        AutoConfig.register(MODEL_TYPE, Zonos3Config, exist_ok=True)
        ModelRegistry.models[SGLANG_ARCHITECTURE] = Zonos3SGLangModel
        install_gdn_prefill_graph()

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        return {
            "max_running_requests": self.max_running_requests,
            "disable_cuda_graph": not self.cuda_graph,
            # note (zhaochenyang20): an eager prefill launches every kernel of the residual mixing and the experts from Python while the running requests wait.
            "cuda_graph_backend_prefill": (
                CudaGraphBackend.BREAKABLE
                if self.cuda_graph
                else CudaGraphBackend.DISABLED
            ),
            "cuda_graph_max_bs_prefill": PREFILL_CUDA_GRAPH_MAX_TOKENS,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_mixed_chunk": False,
            "chunked_prefill_size": -1,
            "attention_backend": ATTENTION_BACKEND,
            "enable_torch_compile": False,
            "mem_fraction_static": self.mem_fraction_static,
            "sampling_backend": "pytorch",
            "trust_remote_code": False,
            "dtype": dtype,
        }

    def adjust_overrides(self, overrides: dict[str, object]) -> None:
        overrides.setdefault("page_size", PAGE_SIZE)

    def validate_before_infrastructure(self, server_args: ServerArgs) -> None:
        chunked_prefill_size = server_args.chunked_prefill_size or 0
        if not server_args.disable_radix_cache:
            raise ValueError("Zonos3 needs disable_radix_cache=True")
        elif chunked_prefill_size > 0 or server_args.enable_mixed_chunk:
            raise ValueError(
                "Zonos3 needs chunked_prefill_size <= 0 and no mixed chunk: the frame "
                "state has no rollback for a partial prompt"
            )
        elif not server_args.disable_overlap_schedule:
            raise ValueError("Zonos3 runs without overlap scheduling")
        elif server_args.page_size != PAGE_SIZE:
            raise ValueError(f"Zonos3 keeps page_size={PAGE_SIZE}")
        elif (server_args.tp_size, server_args.pp_size, server_args.dp_size) != (
            1,
            1,
            1,
        ):
            raise ValueError("Zonos3 runs on one rank")
        else:
            pass
        resolved_server_args = resolved_view(server_args)
        if resolved_server_args.max_total_tokens is None:
            decode_graph_config = resolved_server_args.cuda_graph_config.decode
            graph_batch_size = max(
                [
                    resolved_server_args.max_running_requests,
                    decode_graph_config.max_bs or 0,
                    *(decode_graph_config.bs or []),
                ]
            )
            # note (zhaochenyang20): Without the radix cache only running contexts and one padding slot per graph request hold KV.
            override_server_args(
                server_args,
                "zonos3-reachable-kv-capacity",
                max_total_tokens=(
                    resolved_server_args.max_running_requests * self.context_length
                    + graph_batch_size
                ),
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
        """Keep the model for the runner and adapters; the other arguments are the interface contract."""
        model = model_worker.model_runner.model
        assert isinstance(model, Zonos3SGLangModel)
        self.model = model

    def make_model_runner(
        self, model_worker: ModelWorker, output_processor: SGLangOutputProcessor
    ) -> Zonos3ModelRunner:
        assert self.model is not None
        return Zonos3ModelRunner(model_worker, output_processor, model=self.model)

    def make_adapters(
        self, model: Zonos3SGLangModel
    ) -> tuple[Zonos3RequestBuilder, Zonos3ResultAdapter]:
        return (
            Zonos3RequestBuilder(
                config=model.config, context_length=self.context_length
            ),
            Zonos3ResultAdapter(
                codebook_size=model.config.codebook_size,
                codebook_count=model.config.codebook_count,
            ),
        )

    def extra_scheduler_kwargs(self) -> SchedulerExtras[Zonos3RequestData]:
        assert self.model is not None
        return {
            "stream_output_builder": Zonos3StreamOutputBuilder(
                codebook_size=self.model.config.codebook_size,
                codebook_count=self.model.config.codebook_count,
                vocoder_initial_chunk_frames=self.vocoder_initial_chunk_frames,
                vocoder_steady_chunk_frames=self.vocoder_steady_chunk_frames,
            ),
            "enable_async_decode": self.async_decode,
            # note (zhaochenyang20): A synchronous prefill drained the pending decode and waited for its own first rows, about 6 ms of the engine loop per prefill and a fifth of it on SeedTTS EN at concurrency 16 on one H100; a lookahead prefill reads them one step later like a decode, which served 2.1% more requests per second non-streaming and 1.9% streaming (3 ABBA pairs each).
            "async_prefill": self.async_decode,
            # note (0xtoward): The engine's steps take a stream above the codec's and the reference encoder's, so its kernels go first when they contend for the GPU.
            "should_run_top_priority_stream": current_platform.is_cuda(),
        }
