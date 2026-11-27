# SPDX-License-Identifier: Apache-2.0
"""SGLang engine of the LFM3 AR stage (LFM3-Audio and LFM3-Omni)."""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from typing import Literal

from sglang.srt.models.registry import ModelRegistry
from sglang.srt.server_args import ServerArgs
from transformers import AutoConfig

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.lfm3_audio.depth_cuda_graph import Lfm3DepthCudaGraphs
from sglang_omni.models.lfm3_audio.frame_sampling import SpeechSampler
from sglang_omni.models.lfm3_audio.hf_config import (
    CHECKPOINT_FILE_NAME,
    MODEL_TYPE,
    SGLANG_ARCHITECTURE,
    Lfm3SGLangConfig,
    SpeechSamplingDefaults,
    lfm3_config_from_checkpoint,
    speech_defaults_from_checkpoint,
)
from sglang_omni.models.lfm3_audio.model_runner import Lfm3AudioModelRunner
from sglang_omni.models.lfm3_audio.request_builders import (
    Lfm3AudioRequestData,
    make_adapters,
)
from sglang_omni.models.lfm3_audio.sglang_model import Lfm3AudioSGLangModel
from sglang_omni.platforms import current_platform
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.engine_factory import (
    GenerationDefaults,
    SchedulerExtras,
    TtsEngineBuilder,
)
from sglang_omni.scheduling.generation_batch_policy import get_decode_cuda_graph_bs
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.utils.checkpoint import resolve_checkpoint

logger = logging.getLogger(__name__)

# Tokenizer files the runtime may read next to config.json.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
# note (ruilin): the sliding-window layers are validated on the triton backend only.
ATTENTION_BACKEND = "triton"


class Lfm3AudioEngineBuilder(TtsEngineBuilder[Lfm3AudioRequestData]):
    """Single GPU; decode runs in CUDA graphs, and each prompt, or a retracted request's replay, is prefilled eagerly in one extend."""

    model_name = "LFM3"
    context_length = 16384

    def __init__(
        self,
        *,
        speech_sampler: SpeechSampler,
        enable_async_decode: bool,
        prefill_coalesce_requests: int,
        prefill_coalesce_wait_ms: float,
    ) -> None:
        self.speech_sampler = speech_sampler
        # note (ruilin): launching a decode step before the previous one resolves needs the seeded sampler's device speech state.
        self.enable_async_decode = (
            enable_async_decode and speech_sampler is SpeechSampler.SEEDED
        )
        self.prefill_coalesce_requests = prefill_coalesce_requests
        self.prefill_coalesce_wait_ms = prefill_coalesce_wait_ms
        # note (ruilin): read from the checkpoint by resolve_checkpoint, which build() calls first.
        self.speech_defaults: SpeechSamplingDefaults | None = None

    def resolve_checkpoint(self, model_path: str) -> str:
        """A directory the SGLang loader reads: the backbone config plus the checkpoint's weights."""
        source_checkpoint_dir = resolve_checkpoint(model_path)
        self.speech_defaults = speech_defaults_from_checkpoint(source_checkpoint_dir)
        config = lfm3_config_from_checkpoint(source_checkpoint_dir)
        config.checkpoint_dir = os.path.abspath(source_checkpoint_dir)
        loader_dir = tempfile.mkdtemp(prefix="lfm3_sglang_")
        atexit.register(shutil.rmtree, loader_dir, ignore_errors=True)
        config.save_pretrained(loader_dir)
        for file_name in (CHECKPOINT_FILE_NAME, *TOKENIZER_FILES):
            source_path = os.path.join(source_checkpoint_dir, file_name)
            if os.path.exists(source_path):
                os.symlink(
                    os.path.abspath(source_path), os.path.join(loader_dir, file_name)
                )
            else:
                pass
        return loader_dir

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        del checkpoint_dir
        AutoConfig.register(MODEL_TYPE, Lfm3SGLangConfig, exist_ok=True)
        ModelRegistry.models[SGLANG_ARCHITECTURE] = Lfm3AudioSGLangModel

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        return {
            "max_running_requests": 64,
            "dtype": dtype,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "chunked_prefill_size": -1,
            "enable_mixed_chunk": False,
            "attention_backend": ATTENTION_BACKEND,
            "enable_torch_compile": False,
            "mem_fraction_static": 0.6,
            "sampling_backend": "pytorch",
            "trust_remote_code": True,
        }

    def get_model_buffer_bs(self, model: Lfm3AudioSGLangModel) -> int:
        # note (ruilin): the decode input buffers hold max_running_requests rows; a larger graph would read past them.
        return model.decode_input_embed_buffer.shape[0]

    def validate_before_infrastructure(self, server_args: ServerArgs) -> None:
        """Reject the settings the runner does not support, before any request reaches it."""
        attention_backends = {
            server_args.attention_backend,
            server_args.prefill_attention_backend or server_args.attention_backend,
            server_args.decode_attention_backend or server_args.attention_backend,
        }
        if not server_args.disable_radix_cache:
            raise ValueError(
                "LFM3 needs disable_radix_cache=True: its conv state is not cached"
            )
        elif (
            server_args.chunked_prefill_size or 0
        ) > 0 or server_args.enable_mixed_chunk:
            raise ValueError(
                "LFM3 needs chunked_prefill_size <= 0 and no mixed chunk: a prompt and its media "
                "embeddings are prefilled in one extend"
            )
        elif not server_args.disable_overlap_schedule:
            raise ValueError(
                "LFM3 runs without overlap scheduling: each step reads the last sampled token"
            )
        elif server_args.enable_torch_compile:
            raise ValueError("LFM3 does not support enable_torch_compile")
        elif (server_args.tp_size, server_args.pp_size, server_args.dp_size) != (
            1,
            1,
            1,
        ):
            raise ValueError(
                "LFM3 runs on one GPU: tp_size, pp_size and dp_size must be 1"
            )
        elif server_args.speculative_algorithm is not None:
            raise ValueError("LFM3 does not support speculative decoding")
        elif attention_backends != {ATTENTION_BACKEND}:
            raise ValueError(f"LFM3 runs on the {ATTENTION_BACKEND} attention backend")
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
        del model_worker, checkpoint_dir, device, gpu_id, server_args

    def post_cuda_graph_setup(
        self, model: Lfm3AudioSGLangModel, server_args: ServerArgs
    ) -> None:
        """Capture the depth graphs for the decode buckets; eager serving keeps the depth work eager."""
        batch_sizes = get_decode_cuda_graph_bs(server_args)
        graph_backend = current_platform.get_device_graph_backend(
            model.depth_decoder.mask_embedding.device
        )
        if not batch_sizes or graph_backend is None:
            return
        else:
            pass
        graphs = Lfm3DepthCudaGraphs(
            model.depth_decoder,
            graph_backend,
            hidden_size=model.config.hidden_size,
            schedule=model.tokens.maskgit_schedule,
            repetition_window=self.speech_defaults.repetition_window,
        )
        try:
            if self.speech_sampler is SpeechSampler.EXACT:
                graphs.capture_pass_graphs(batch_sizes)
            else:
                graphs.capture_frame_graphs(batch_sizes)
        except RuntimeError:
            logger.warning(
                "LFM3 depth CUDA graph capture failed; speech frames run eagerly",
                exc_info=True,
            )
            return
        model.depth_graphs = graphs

    def make_model_runner(
        self,
        model_worker: ModelWorker,
        output_proc: SGLangOutputProcessor,
    ) -> Lfm3AudioModelRunner:
        return Lfm3AudioModelRunner(
            model_worker,
            output_proc,
            speech_sampler=self.speech_sampler,
            repetition_window=self.speech_defaults.repetition_window,
        )

    def extra_scheduler_kwargs(self) -> SchedulerExtras[Lfm3AudioRequestData]:
        enable_async_decode = self.enable_async_decode and current_platform.is_cuda()
        logger.info(
            f"LFM3 async decode: {'on' if enable_async_decode else 'off'} "
            f"(speech sampler {self.speech_sampler.value})"
        )
        return {
            "enable_async_decode": enable_async_decode,
            # note (ruilin): a prefill reads the host speech state, so the pending step resolves first.
            "async_decode_resolve_before_prefill": True,
            "prefill_coalesce_requests": self.prefill_coalesce_requests,
            "prefill_coalesce_wait_ms": self.prefill_coalesce_wait_ms,
        }

    def make_adapters(self, model: Lfm3AudioSGLangModel) -> tuple[
        Callable[[StagePayload], Lfm3AudioRequestData],
        Callable[[Lfm3AudioRequestData], StagePayload],
    ]:
        return make_adapters(
            vocab_size=model.config.vocab_size,
            tokens=model.tokens,
            speech_defaults=self.speech_defaults,
            device=self.device,
        )


def create_generation_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    server_args_overrides: Mapping[str, object] | None = None,
    speech_sampler: Literal["exact", "seeded"],
    enable_async_decode: bool,
    prefill_coalesce_requests: int,
    prefill_coalesce_wait_ms: float,
) -> OmniScheduler[Lfm3AudioRequestData]:
    return Lfm3AudioEngineBuilder(
        speech_sampler=SpeechSampler(speech_sampler),
        enable_async_decode=enable_async_decode,
        prefill_coalesce_requests=prefill_coalesce_requests,
        prefill_coalesce_wait_ms=prefill_coalesce_wait_ms,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        server_args_overrides=server_args_overrides,
    )
