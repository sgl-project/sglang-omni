# SPDX-License-Identifier: Apache-2.0
"""SGLang engine of the LFM3 AR stage (LFM3-Audio and LFM3-Omni)."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping

from sglang.srt.models.registry import ModelRegistry
from sglang.srt.server_args import ServerArgs
from transformers import AutoConfig

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.lfm3_audio.hf_config import (
    CHECKPOINT_FILE_NAME,
    MODEL_TYPE,
    SGLANG_ARCHITECTURE,
    Lfm3SGLangConfig,
    lfm3_config_from_checkpoint,
    lfm3_tokens_from_checkpoint,
    speech_defaults_from_checkpoint,
)
from sglang_omni.models.lfm3_audio.model_runner import Lfm3AudioModelRunner
from sglang_omni.models.lfm3_audio.request_builders import (
    Lfm3AudioRequestData,
    make_adapters,
)
from sglang_omni.models.lfm3_audio.sglang_model import Lfm3AudioSGLangModel
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.engine_factory import GenerationDefaults, TtsEngineBuilder
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.utils.checkpoint import resolve_checkpoint

# Tokenizer files the runtime may read next to config.json.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
# note (ruilin): the sliding-window layers are validated on the triton backend only.
ATTENTION_BACKEND = "triton"


class Lfm3AudioEngineBuilder(TtsEngineBuilder[Lfm3AudioRequestData]):
    """Single GPU and eager; each prompt, or a retracted request's replay, is prefilled in one extend."""

    model_name = "LFM3"
    context_length = 16384

    def resolve_checkpoint(self, model_path: str) -> str:
        """A directory the SGLang loader reads: the backbone config plus the checkpoint's weights."""
        source_checkpoint_dir = resolve_checkpoint(model_path)
        self.source_checkpoint_dir = source_checkpoint_dir
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
            "max_running_requests": 8,
            "dtype": dtype,
            "disable_cuda_graph": True,
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
        elif not server_args.disable_cuda_graph or server_args.enable_torch_compile:
            raise ValueError(
                "LFM3 runs eager: disable_cuda_graph=True, enable_torch_compile=False"
            )
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

    def make_model_runner(
        self,
        model_worker: ModelWorker,
        output_proc: SGLangOutputProcessor,
    ) -> Lfm3AudioModelRunner:
        return Lfm3AudioModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Lfm3AudioSGLangModel) -> tuple[
        Callable[[StagePayload], Lfm3AudioRequestData],
        Callable[[Lfm3AudioRequestData], StagePayload],
    ]:
        return make_adapters(
            model=model,
            tokens=lfm3_tokens_from_checkpoint(self.source_checkpoint_dir),
            speech_defaults=speech_defaults_from_checkpoint(self.source_checkpoint_dir),
            device=self.device,
        )


def create_generation_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    server_args_overrides: Mapping[str, object] | None = None,
) -> OmniScheduler[Lfm3AudioRequestData]:
    return Lfm3AudioEngineBuilder().build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        server_args_overrides=server_args_overrides,
    )
