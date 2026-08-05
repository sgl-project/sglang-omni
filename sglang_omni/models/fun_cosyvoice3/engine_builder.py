# SPDX-License-Identifier: Apache-2.0
"""Fun-CosyVoice3 SGLang engine builder."""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any

import torch

from sglang_omni.models.fun_cosyvoice3 import request_builders
from sglang_omni.models.fun_cosyvoice3.utils import (
    CosyVoice3Tokenizer,
    SpeakerEncoder,
    SpeechTokenizerV3,
)
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder
from sglang_omni.utils.checkpoint import resolve_checkpoint as _resolve_checkpoint

logger = logging.getLogger(__name__)


class FunCosyVoice3EngineBuilder(TtsEngineBuilder):
    model_name = "Fun-CosyVoice3"
    context_length = 4096
    model_arch_override = "FunCosyVoice3SGLangModel"

    def __init__(self) -> None:
        super().__init__()
        self._checkpoint_root: str | None = None

    def _blanken_dir(self) -> str:
        assert self._checkpoint_root is not None, "checkpoint_root not set"
        return os.path.join(self._checkpoint_root, "CosyVoice-BlankEN")

    def resolve_checkpoint(self, model_path: str) -> str:
        resolved = _resolve_checkpoint(model_path)
        self._checkpoint_root = resolved
        # SGLang needs CosyVoice-BlankEN/ which has config.json (model_type: qwen2)
        return self._blanken_dir()

    def generation_defaults(
        self,
        *,
        dtype: str,
    ) -> dict[str, Any]:
        return {
            "max_running_requests": 32,
            "cuda_graph_max_bs": 32,
            "torch_compile_max_bs": 32,
            "dtype": dtype,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "enable_torch_compile": False,
            "mem_fraction_static": 0.85,
            "max_prefill_tokens": 4096,
            "sampling_backend": "pytorch",
            "trust_remote_code": True,
        }

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        del checkpoint_dir, gpu_id, server_args
        model = model_worker.model_runner.model
        root = self._checkpoint_root
        assert root is not None, "checkpoint_root not set"

        # SGLang's model_config reads vocab_size from HF config (151936 for
        # Qwen2 text). CosyVoice3 outputs speech tokens (vocab=6761). Patch
        # the model config so sampling penalty tensors match our logit dim.
        from sglang_omni.models.fun_cosyvoice3.sglang_model import TOTAL_VOCAB_SIZE
        model_worker.model_runner.model_config.vocab_size = TOTAL_VOCAB_SIZE

        # Load fine-tuned CosyVoice3 LLM weights from llm.pt (backbone +
        # speech_embedding + llm_decoder). SGLang already loaded the Qwen2
        # backbone from HF safetensors; this overwrites with the fine-tuned
        # speech-adapted weights.
        llm_pt_path = os.path.join(root, "llm.pt")
        logger.info("Loading CosyVoice3 fine-tuned weights from %s", llm_pt_path)
        state_dict = torch.load(llm_pt_path, map_location="cpu", weights_only=True)
        model.load_weights(list(state_dict.items()))
        logger.info("CosyVoice3 weights loaded")

        tokenizer_path = os.path.join(root, "CosyVoice-BlankEN")
        speech_tokenizer_path = os.path.join(root, "speech_tokenizer_v3.onnx")
        campplus_path = os.path.join(root, "campplus.onnx")

        tokenizer = CosyVoice3Tokenizer(tokenizer_path)
        speech_tokenizer = SpeechTokenizerV3(speech_tokenizer_path, device=device)
        speaker_encoder = SpeakerEncoder(campplus_path, device=device)

        request_builders.set_cosyvoice3_preprocessing_context(
            model=model,
            tokenizer=tokenizer,
            speech_tokenizer=speech_tokenizer,
            speaker_encoder=speaker_encoder,
        )

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        model_runner_mod = importlib.import_module(
            "sglang_omni.models.fun_cosyvoice3.model_runner"
        )
        return model_runner_mod.FunCosyVoice3ModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        return request_builders.make_cosyvoice3_scheduler_adapters(model=model)

    def make_abort_callback(self) -> Any | None:
        return request_builders.cleanup_prepared_cosyvoice3_request
