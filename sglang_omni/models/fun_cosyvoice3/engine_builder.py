# SPDX-License-Identifier: Apache-2.0
"""Fun-CosyVoice3 SGLang engine builder."""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any

import torch

from sglang_omni.models.fun_cosyvoice3 import request_builders
from sglang_omni.models.fun_cosyvoice3.streaming import TOKEN_HOP_LEN
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
        token_hop_len: int = TOKEN_HOP_LEN,
        onnx_intra_op_threads: int = 16,
        mlx_model_path: str | None = None,
        mlx_model_revision: str | None = None,
    ) -> None:
        super().__init__()
        hop = int(token_hop_len)
        if hop <= 0:
            raise ValueError(f"token_hop_len must be positive, got {token_hop_len}")
        else:
            pass
        self.token_hop_len = hop
        self.checkpoint_root: str | None = None
        self.mlx_model_path = mlx_model_path
        self.mlx_model_revision = mlx_model_revision
        self.device: str | None = None

        # note (Dayuxiaoshui): both ONNX sessions get a pool of this size, so
        # cap it at the host core count instead of trusting the default of 16.
        self.onnx_intra_op_threads = max(
            1, min(int(onnx_intra_op_threads), os.cpu_count() or 1)
        )

    def blanken_dir(self) -> str:
        assert self.checkpoint_root is not None, "checkpoint_root not set"
        return os.path.join(self.checkpoint_root, "CosyVoice-BlankEN")

    def resolve_checkpoint(self, model_path: str) -> str:
        resolved = _resolve_checkpoint(model_path)
        self.checkpoint_root = resolved
        # SGLang needs CosyVoice-BlankEN/ which has config.json (model_type: qwen2)
        return self.blanken_dir()

    def generation_defaults(
        self,
        *,
        dtype: str,
    ) -> dict[str, Any]:
        from sglang.srt.hardware_backend.mlx.runtime import use_mlx

        if use_mlx():
            if not current_platform.is_mps():
                raise RuntimeError(
                    "Fun-CosyVoice3 MLX requires the Apple Metal platform"
                )
            else:
                pass
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
                "mlx_enable_sampling": True,
            }
        else:
            pass
        if self.uses_torch_mps():
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
        else:
            pass
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

    def before_memory_pool(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        from sglang.srt.hardware_backend.mlx.runtime import use_mlx

        # note(ratish): the fine-tuned weights and the ONNX sessions live for
        # the whole process, so they are built before sglang reads free memory
        # for the KV pool.
        del checkpoint_dir, gpu_id, server_args
        root = self.checkpoint_root
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
        speech_tokenizer = SpeechTokenizerV3(
            speech_tokenizer_path,
            device=device,
            intra_op_threads=self.onnx_intra_op_threads,
        )
        speaker_encoder = SpeakerEncoder(
            campplus_path,
            device=device,
            intra_op_threads=self.onnx_intra_op_threads,
        )

        request_builders.set_cosyvoice3_preprocessing_context(
            model=model,
            tokenizer=tokenizer,
            speech_tokenizer=speech_tokenizer,
            speaker_encoder=speaker_encoder,
            use_mlx=use_mlx(),
            model_revision=root,
        )

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        del model_worker, checkpoint_dir, device, gpu_id, server_args

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        from sglang.srt.hardware_backend.mlx.runtime import use_mlx

        if use_mlx():
            from sglang_omni.models.fun_cosyvoice3.model_runner import (
                FunCosyVoice3MlxSchedulerModelRunner,
            )

            return FunCosyVoice3MlxSchedulerModelRunner(
                model_worker,
                output_proc,
                token_hop_len=self.token_hop_len,
            )
        else:
            pass
        model_runner_mod = importlib.import_module(
            "sglang_omni.models.fun_cosyvoice3.model_runner"
        )
        return model_runner_mod.FunCosyVoice3ModelRunner(
            model_worker,
            output_proc,
            token_hop_len=self.token_hop_len,
        )

    def validate_before_infrastructure(self, server_args: Any) -> None:
        from sglang.srt.hardware_backend.mlx.runtime import use_mlx

        if not use_mlx():
            if self.uses_torch_mps() and server_args.max_running_requests != 1:
                raise ValueError(
                    "Fun-CosyVoice3 Torch MPS currently requires "
                    "max_running_requests=1"
                )
            else:
                pass
            return
        else:
            pass
        if server_args.max_running_requests != 1:
            raise ValueError(
                "Fun-CosyVoice3 MLX currently requires max_running_requests=1"
            )
        else:
            pass
        if not server_args.disable_radix_cache:
            raise ValueError("Fun-CosyVoice3 MLX requires disable_radix_cache=True")
        else:
            pass
        if server_args.chunked_prefill_size != -1:
            raise ValueError("Fun-CosyVoice3 MLX requires chunked_prefill_size=-1")
        else:
            pass
        if not server_args.disable_overlap_schedule:
            raise ValueError(
                "Fun-CosyVoice3 MLX requires disable_overlap_schedule=True"
            )
        else:
            pass
        if server_args.enable_priority_scheduling:
            raise ValueError("Fun-CosyVoice3 MLX does not support priority preemption")
        else:
            pass
        if not server_args.mlx_enable_sampling:
            raise ValueError("Fun-CosyVoice3 MLX requires mlx_enable_sampling=True")
        else:
            pass

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        return request_builders.make_cosyvoice3_scheduler_adapters(model=model)

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        from sglang.srt.hardware_backend.mlx.runtime import use_mlx

        if not use_mlx():
            return {}
        else:
            pass
        return {
            "enable_async_decode": True,
            "async_decode_min_batch_size": 1,
        }

    def infra_kwargs(self) -> dict[str, Any]:
        from sglang.srt.hardware_backend.mlx.runtime import use_mlx

        if not use_mlx():
            return {}
        else:
            pass
        # Note (yexiaodong): The stub reads nested Qwen2 config while the
        # native runner may load a separate artifact; keep that override in
        # Omni's typed worker config rather than upstream ServerArgs.
        return {
            "mlx_model_path": self.mlx_model_path or self.checkpoint_root,
            "mlx_model_revision": self.mlx_model_revision,
        }

    def make_abort_callback(self) -> Any | None:
        return request_builders.cleanup_prepared_cosyvoice3_request

    def post_scheduler_setup(self, scheduler: Any, model_runner: Any) -> None:
        model_runner.set_stream_outbox(scheduler.outbox)
