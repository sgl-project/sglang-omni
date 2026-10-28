# SPDX-License-Identifier: Apache-2.0
"""Fun-CosyVoice3 SGLang engine builder."""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch
from sglang.srt.arg_groups.model_override_base import (
    attention_backends_of,
    resolved_view,
)
from sglang.srt.model_executor.cuda_graph_config import Phase, with_phase
from sglang.srt.server_args import ServerArgs

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.fun_cosyvoice3 import CAPABILITIES, request_builders
from sglang_omni.models.fun_cosyvoice3.request_builders import (
    CosyVoice3SGLangRequestData,
)
from sglang_omni.models.fun_cosyvoice3.streaming import TOKEN_HOP_LEN
from sglang_omni.models.fun_cosyvoice3.utils import (
    MAX_REFERENCE_SAMPLES,
    CosyVoice3Tokenizer,
    SpeakerEncoder,
    SpeechTokenizerV3,
)
from sglang_omni.platforms import current_platform
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.engine_factory import (
    GenerationDefaults,
    SchedulerExtras,
    TtsEngineBuilder,
)
from sglang_omni.scheduling.generation_batch_policy import (
    FULL_PREFILL_ATTENTION_BACKENDS,
    CudaGraphBackend,
    nested_prefill_overrides,
    operator_selected_prefill_backend,
)
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.utils.checkpoint import resolve_checkpoint as _resolve_checkpoint
from sglang_omni.vendor.sglang.server_args import override_server_args

if TYPE_CHECKING:

    from sglang_omni.models.fun_cosyvoice3.model_runner import FunCosyVoice3ModelRunner
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
else:
    pass

logger = logging.getLogger(__name__)

_ENGINE_HEADROOM_BYTES = 0


def engine_headroom_bytes() -> int:
    """The memory the engine keeps free beside its static pools; 0 when this process
    runs no engine."""
    return _ENGINE_HEADROOM_BYTES


class FunCosyVoice3EngineBuilder(TtsEngineBuilder[CosyVoice3SGLangRequestData]):
    model_name = "Fun-CosyVoice3"
    context_length = 4096
    mem_fraction_static = 0.85
    model_arch_override = "FunCosyVoice3SGLangModel"
    supports_full_prefill_cuda_graph = CAPABILITIES.supports_full_prefill_cuda_graph

    def __init__(
        self,
        *,
        token_hop_len: int = TOKEN_HOP_LEN,
        onnx_intra_op_threads: int = 16,
    ) -> None:
        super().__init__()
        hop = int(token_hop_len)
        if hop <= 0:
            raise ValueError(f"token_hop_len must be positive, got {token_hop_len}")
        else:
            pass
        self.token_hop_len = hop
        self.checkpoint_root: str | None = None
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

    def uses_torch_mps(self) -> bool:
        return self.device is not None and torch.device(self.device).type == "mps"

    def generation_defaults(
        self,
        *,
        dtype: str,
    ) -> GenerationDefaults:
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
            "mem_fraction_static": self.mem_fraction_static,
            "max_prefill_tokens": 4096,
            "sampling_backend": (
                "flashinfer" if current_platform.is_cuda() else "pytorch"
            ),
            # note (ratish): run eagerly, a prefill launches about 320 kernels for 2 ms of
            # device work while every running row waits for it.
            "cuda_graph_backend_prefill": (
                CudaGraphBackend.FULL
                if current_platform.is_cuda()
                else CudaGraphBackend.DISABLED
            ),
            "trust_remote_code": True,
        }

    def customize_server_args(self, server_args: ServerArgs) -> None:
        resolved = resolved_view(server_args)
        graph_config = resolved.cuda_graph_config
        prefill = graph_config.prefill
        if (
            prefill.backend == CudaGraphBackend.FULL
            and attention_backends_of(resolved)[0]
            not in FULL_PREFILL_ATTENTION_BACKENDS
            and not operator_selected_prefill_backend(self.server_args_overrides)
        ):
            # note (ratish): the full prefill graph is this model's default only where
            # the attention backend captures it; elsewhere the prefill stays eager.
            override_server_args(
                server_args,
                "fun_cosyvoice3.prefill_cuda_graph_backend",
                cuda_graph_config=with_phase(
                    graph_config, Phase.PREFILL, backend=CudaGraphBackend.DISABLED
                ),
            )
        elif (
            prefill.backend == CudaGraphBackend.FULL
            and prefill.bs
            and prefill.bs[0] != 1
            and self.server_args_overrides.get("cuda_graph_bs_prefill") is None
            and "bs" not in nested_prefill_overrides(self.server_args_overrides)
        ):
            # note (ratish): a prompt the radix cache holds re-prefills only its last
            # token, which the ladder's first bucket of 4 pads past twice its size and
            # runs eagerly; every eager prefill measured at c16 and c64 was one.
            override_server_args(
                server_args,
                "fun_cosyvoice3.prefill_cuda_graph_shape",
                cuda_graph_config=with_phase(
                    graph_config, Phase.PREFILL, bs=[1, *prefill.bs]
                ),
            )
        else:
            pass

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        # Note (Jiaxin Deng): the fraction alone sized a 62 GB pool on an
        # H100 that the running requests can never fill; the vocoder shares
        # the GPU and was left about 10 GB of dynamic-shape headroom.
        if overrides.get("max_total_tokens") is None:
            overrides["max_total_tokens"] = (
                overrides["max_running_requests"] * self.context_length
            )
        else:
            pass

    def before_memory_pool(
        self,
        *,
        model_worker: ModelWorker,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: ServerArgs,
    ) -> None:
        # note(ratish): the fine-tuned weights and the ONNX sessions live for
        # the whole process, so they are built before sglang reads free memory
        # for the KV pool.
        del checkpoint_dir, gpu_id, server_args
        root = self.checkpoint_root
        assert root is not None, "checkpoint_root not set"
        from sglang_omni.models.fun_cosyvoice3.sglang_model import TOTAL_VOCAB_SIZE

        model_worker.model_runner.model_config.vocab_size = TOTAL_VOCAB_SIZE

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
        # note (ratish): ONNX Runtime keeps its arena outside torch at the peak it
        # reached, so one reference at the cap makes it resident before either pool
        # is sized from free memory.
        speech_tokenizer.extract_speech_token(
            np.zeros((1, MAX_REFERENCE_SAMPLES), dtype=np.float32)
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
            device=torch.device(device),
            model_revision=root,
        )
        global _ENGINE_HEADROOM_BYTES
        model_runner = model_worker.model_runner
        # note (ratish): the vocoder in this process sizes its Flow prefix pool after the
        # engine, so it leaves the share of pre-load memory the engine keeps free for its
        # activations; a declared budget bounds only the engine's KV pool.
        _ENGINE_HEADROOM_BYTES = int(
            model_runner.pre_model_load_memory
            * (1 - model_runner.mem_fraction_static)
            * 2**30
        )

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
    ) -> FunCosyVoice3ModelRunner:
        model_runner_mod = importlib.import_module(
            "sglang_omni.models.fun_cosyvoice3.model_runner"
        )
        return model_runner_mod.FunCosyVoice3ModelRunner(
            model_worker,
            output_proc,
            token_hop_len=self.token_hop_len,
        )

    def validate_before_infrastructure(self, server_args: ServerArgs | None) -> None:
        if self.uses_torch_mps() and server_args.max_running_requests != 1:
            raise ValueError(
                "Fun-CosyVoice3 Torch MPS currently requires max_running_requests=1"
            )
        else:
            pass

    def make_adapters(self, model: torch.nn.Module) -> tuple[
        Callable[[StagePayload], CosyVoice3SGLangRequestData],
        Callable[[CosyVoice3SGLangRequestData], StagePayload],
    ]:
        return request_builders.make_cosyvoice3_scheduler_adapters(model=model)

    def extra_scheduler_kwargs(self) -> SchedulerExtras[CosyVoice3SGLangRequestData]:
        # note (ratish): the vocoder shares this process and card on its own stream, and
        # at equal priority the AR's decode replays wait behind its kernels.
        return {"should_run_top_priority_stream": current_platform.is_cuda()}

    def make_abort_callback(self) -> Callable[[str], None]:
        return request_builders.cleanup_prepared_cosyvoice3_request

    def post_scheduler_setup(
        self,
        scheduler: OmniScheduler[CosyVoice3SGLangRequestData],
        model_runner: FunCosyVoice3ModelRunner,
    ) -> None:
        model_runner.set_stream_outbox(scheduler.outbox)
