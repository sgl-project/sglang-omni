# SPDX-License-Identifier: Apache-2.0
"""SGLang engine of the LFM3-Audio AR stage."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from collections.abc import Callable

from sglang.srt.models.registry import ModelRegistry
from transformers import AutoConfig

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.lfm3_audio import request_builders
from sglang_omni.models.lfm3_audio.hf_config import (
    CHECKPOINT_FILE_NAME,
    MODEL_TYPE,
    SGLANG_ARCHITECTURE,
    Lfm3SGLangConfig,
    lfm3_config_from_checkpoint,
    lfm3_tokens_from_checkpoint,
)
from sglang_omni.models.lfm3_audio.model_runner import Lfm3AudioModelRunner
from sglang_omni.models.lfm3_audio.request_builders import Lfm3AudioRequestData
from sglang_omni.models.lfm3_audio.sglang_model import Lfm3AudioSGLangModel
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.engine_factory import GenerationDefaults, TtsEngineBuilder
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.utils.checkpoint import resolve_checkpoint

# Tokenizer files the runtime may read next to config.json.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
ATTENTION_BACKEND = "triton"


class Lfm3AudioEngineBuilder(TtsEngineBuilder[Lfm3AudioRequestData]):
    """Single GPU, eager decode, no radix cache or chunked prefill.

    The conv state has no rollback for a partial prompt and a prompt's audio
    embeddings are placed in one prefill, so chunked prefill stays off; the
    hybrid conv-state pool needs the radix cache off.
    """

    model_name = "LFM3-Audio"
    context_length = 8192

    def resolve_checkpoint(self, model_path: str) -> str:
        """A directory the SGLang loader reads: the backbone config plus the checkpoint's weights."""
        source = resolve_checkpoint(model_path)
        self.source_checkpoint_dir = source
        config = lfm3_config_from_checkpoint(source)
        config.checkpoint_dir = os.path.abspath(source)
        shim = tempfile.mkdtemp(prefix="lfm3_audio_sglang_")
        atexit.register(shutil.rmtree, shim, ignore_errors=True)
        config.save_pretrained(shim)
        for name in (CHECKPOINT_FILE_NAME, *TOKENIZER_FILES):
            path = os.path.join(source, name)
            if os.path.exists(path):
                os.symlink(os.path.abspath(path), os.path.join(shim, name))
            else:
                pass
        return shim

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

    def setup_model(
        self,
        *,
        model_worker: ModelWorker,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: object,
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
        return request_builders.make_adapters(
            model=model,
            tokens=lfm3_tokens_from_checkpoint(self.source_checkpoint_dir),
            device=self.device,
        )
