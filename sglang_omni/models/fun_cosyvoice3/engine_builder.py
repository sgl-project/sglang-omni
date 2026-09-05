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
from sglang_omni.platforms import current_platform
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder
from sglang_omni.utils.checkpoint import resolve_checkpoint as _resolve_checkpoint

logger = logging.getLogger(__name__)


class FunCosyVoice3EngineBuilder(TtsEngineBuilder):
    model_name = "Fun-CosyVoice3"
    context_length = 4096
    model_arch_override = "FunCosyVoice3SGLangModel"

    def __init__(
        self,
        *,
        mlx_model_path: str | None = None,
        mlx_model_revision: str | None = None,
    ) -> None:
        super().__init__()
        self._checkpoint_root: str | None = None
        self._mlx_model_path = mlx_model_path
        self._mlx_model_revision = mlx_model_revision
        self.device: str | None = None

    def _blanken_dir(self) -> str:
        assert self._checkpoint_root is not None, "checkpoint_root not set"
        return os.path.join(self._checkpoint_root, "CosyVoice-BlankEN")

    def resolve_checkpoint(self, model_path: str) -> str:
        resolved = _resolve_checkpoint(model_path)
        self._checkpoint_root = resolved
        # SGLang needs CosyVoice-BlankEN/ which has config.json (model_type: qwen2)
        return self._blanken_dir()

    def _uses_torch_mps(self) -> bool:
        from sglang.srt.utils.tensor_bridge import use_mlx

        return (
            not use_mlx()
            and self.device is not None
            and torch.device(self.device).type == "mps"
        )

    def generation_defaults(
        self,
        *,
        dtype: str,
    ) -> dict[str, Any]:
        from sglang.srt.utils.tensor_bridge import use_mlx

        if use_mlx():
            if not current_platform.is_mps():
                raise RuntimeError(
                    "Fun-CosyVoice3 MLX requires the Apple Metal platform"
                )
            return {
                "max_running_requests": 1,
                "disable_cuda_graph": True,
                "disable_overlap_schedule": True,
                "disable_radix_cache": True,
                "enable_torch_compile": False,
                "max_prefill_tokens": self.context_length,
                "max_total_tokens": self.context_length,
                "chunked_prefill_size": -1,
                "dtype": dtype,
                "sampling_backend": "pytorch",
                # Keep CosyVoice's top-p/top-k sampling in the native runner.
                "mlx_enable_sampling": True,
            }
        if self._uses_torch_mps():
            return {
                "max_running_requests": 1,
                "disable_cuda_graph": True,
                "disable_overlap_schedule": True,
                "disable_radix_cache": True,
                "enable_torch_compile": False,
                "max_prefill_tokens": self.context_length,
                "max_total_tokens": self.context_length,
                "chunked_prefill_size": -1,
                "dtype": dtype,
                "attention_backend": "torch_native",
                "sampling_backend": "pytorch",
            }
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
        from sglang.srt.utils.tensor_bridge import use_mlx

        del checkpoint_dir, gpu_id
        root = self._checkpoint_root
        assert root is not None, "checkpoint_root not set"
        from sglang_omni.models.fun_cosyvoice3.sglang_model import TOTAL_VOCAB_SIZE

        model_worker.model_runner.model_config.vocab_size = TOTAL_VOCAB_SIZE

        if use_mlx():
            model = None
        else:
            model = model_worker.model_runner.model
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
            use_mlx=use_mlx(),
            model_revision=root,
        )

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        from sglang.srt.utils.tensor_bridge import use_mlx

        if use_mlx():
            from sglang_omni.models.fun_cosyvoice3.model_runner import (
                FunCosyVoice3MlxSchedulerModelRunner,
            )

            # Use a CosyVoice collector on top of the shared MLX scheduler
            # bridge; the bridge still owns lazy cache ordering.
            return FunCosyVoice3MlxSchedulerModelRunner(model_worker, output_proc)
        model_runner_mod = importlib.import_module(
            "sglang_omni.models.fun_cosyvoice3.model_runner"
        )
        return model_runner_mod.FunCosyVoice3ModelRunner(model_worker, output_proc)

    def validate_before_infrastructure(self, server_args: Any) -> None:
        from sglang.srt.utils.tensor_bridge import use_mlx

        if not use_mlx():
            if self._uses_torch_mps() and server_args.max_running_requests != 1:
                raise ValueError(
                    "Fun-CosyVoice3 Torch MPS currently requires "
                    "max_running_requests=1"
                )
            return
        if server_args.max_running_requests != 1:
            raise ValueError(
                "Fun-CosyVoice3 MLX currently requires max_running_requests=1"
            )
        if not server_args.disable_radix_cache:
            raise ValueError("Fun-CosyVoice3 MLX requires disable_radix_cache=True")
        if server_args.chunked_prefill_size != -1:
            raise ValueError("Fun-CosyVoice3 MLX requires chunked_prefill_size=-1")
        if not server_args.disable_overlap_schedule:
            raise ValueError(
                "Fun-CosyVoice3 MLX requires disable_overlap_schedule=True"
            )
        if server_args.enable_priority_scheduling:
            raise ValueError("Fun-CosyVoice3 MLX does not support priority preemption")
        if not server_args.mlx_enable_sampling:
            raise ValueError("Fun-CosyVoice3 MLX requires mlx_enable_sampling=True")

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        return request_builders.make_cosyvoice3_scheduler_adapters(model=model)

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        from sglang.srt.utils.tensor_bridge import use_mlx

        if not use_mlx():
            return {}
        return {
            "enable_async_decode": True,
            "async_decode_min_batch_size": 1,
        }

    def infra_kwargs(self) -> dict[str, Any]:
        from sglang.srt.utils.tensor_bridge import use_mlx

        if not use_mlx():
            return {}
        # SGLang's bookkeeping stub reads the nested Qwen2 config, while the
        # native runner loads the official root or an explicitly supplied MLX
        # artifact. Keep that second path in Omni's typed worker config rather
        # than attaching a model-specific field to upstream ServerArgs.
        return {
            "mlx_model_path": self._mlx_model_path or self._checkpoint_root,
            "mlx_model_revision": self._mlx_model_revision,
        }

    def make_abort_callback(self) -> Any | None:
        return request_builders.cleanup_prepared_cosyvoice3_request
