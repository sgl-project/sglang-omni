# SPDX-License-Identifier: Apache-2.0
"""Ming-Omni-TTS SGLang engine builder."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from sglang_omni.models.ming_omni.tp_utils import validate_attention_tp_config
from sglang_omni.scheduling.engine_factory import GenerationDefaults, TtsEngineBuilder
from sglang_omni.scheduling.generation_batch_policy import get_decode_cuda_graph_bs

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker
    from sglang.srt.server_args import ServerArgs
    from transformers import PretrainedConfig

    from sglang_omni.model_runner.model_worker import ModelWorker
    from sglang_omni.models.ming_tts.engine_io import MingTTSSGLangRequestData
    from sglang_omni.models.ming_tts.model_runner import MingTTSModelRunner
    from sglang_omni.models.ming_tts.sglang_model import MingTTSSGLangModel
    from sglang_omni.models.ming_tts.tokenizer import MingTTSTokenizerBundle
    from sglang_omni.proto import StagePayload
    from sglang_omni.scheduling.bootstrap import InfrastructureOptions
    from sglang_omni.scheduling.message import OutgoingMessage
    from sglang_omni.scheduling.sglang_backend.output_processor import (
        SGLangOutputProcessor,
    )
else:
    pass


logger = logging.getLogger(__name__)


def is_truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    else:
        pass
    if isinstance(value, int):
        return value != 0
    else:
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    else:
        pass
    return False


class MingTtsEngineBuilder(TtsEngineBuilder["MingTTSSGLangRequestData"]):
    model_name = "Ming-Omni-TTS"
    context_length = 0

    def __init__(
        self,
        *,
        context_length: int | None = None,
        total_gpu_memory_fraction: float | None = None,
        tp_rank: int = 0,
        tp_size: int = 1,
        nccl_port: int | None = None,
    ) -> None:
        from sglang_omni.models.ming_tts.hf_config import MING_TTS_MODEL_ARCH_OVERRIDE

        tp_rank = int(tp_rank)
        tp_size = int(tp_size)
        if tp_size <= 0:
            raise ValueError(
                f"Ming-Omni-TTS tts_engine tp_size must be positive; got {tp_size}"
            )
        else:
            pass
        if tp_rank < 0 or tp_rank >= tp_size:
            raise ValueError(
                f"Ming-Omni-TTS tts_engine tp_rank={tp_rank} is out of range "
                f"for tp_size={tp_size}"
            )
        else:
            pass
        if tp_size > 1 and nccl_port is None:
            raise ValueError("Ming-Omni-TTS tts_engine TP requires nccl_port")
        else:
            pass

        self.model_arch_override = MING_TTS_MODEL_ARCH_OVERRIDE
        self.requested_context_length = context_length
        self.total_gpu_memory_fraction = total_gpu_memory_fraction
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.nccl_port = nccl_port
        self.config: PretrainedConfig | None = None
        self.tokenizer: MingTTSTokenizerBundle | None = None
        self.model_runner: MingTTSModelRunner | None = None

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        from sglang_omni.models.ming_tts import stages as ming_stages

        self.config = ming_stages.load_ming_tts_config(checkpoint_dir)
        if self.tp_size > 1:
            llm_config = self.config.llm_config
            hidden_size = int(llm_config.hidden_size)
            head_dim = int(llm_config.head_dim)
            num_heads = int(llm_config.num_attention_heads)
            num_kv_heads = int(llm_config.num_key_value_heads)
            if min(hidden_size, head_dim, num_heads, num_kv_heads) <= 0:
                raise ValueError(
                    "Ming-Omni-TTS TP requires positive hidden/head dimensions: "
                    f"hidden_size={hidden_size}, head_dim={head_dim}, "
                    f"num_attention_heads={num_heads}, "
                    f"num_key_value_heads={num_kv_heads}"
                )
            else:
                pass
            if head_dim * num_heads != hidden_size:
                raise ValueError(
                    "Ming-Omni-TTS TP requires head_dim * num_attention_heads "
                    f"to equal hidden_size ({head_dim} * {num_heads} != {hidden_size})"
                )
            else:
                pass
            if hidden_size % self.tp_size != 0:
                raise ValueError(
                    "Ming-Omni-TTS TP requires hidden_size divisible by tp_size: "
                    f"hidden_size={hidden_size}, tp_size={self.tp_size}"
                )
            else:
                pass
            validate_attention_tp_config(
                num_attention_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                tp_size=self.tp_size,
                context="Ming-Omni-TTS tts_engine",
            )
        else:
            pass
        context_length = int(self.requested_context_length or 0)
        if context_length <= 0:
            context_length = ming_stages.resolve_context_length(self.config)
        else:
            pass
        self.context_length = int(context_length)

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        return {
            "max_running_requests": 8,
            "dtype": dtype,
            "disable_cuda_graph": True,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_torch_compile": False,
            "max_prefill_tokens": min(int(self.context_length), 8192),
            "sampling_backend": "pytorch",
            "trust_remote_code": False,
        }

    def adjust_overrides(self, overrides: dict[str, object]) -> None:
        overrides.pop("context_length", None)
        overrides["tp_size"] = self.tp_size

        if not is_truthy(overrides["disable_overlap_schedule"]):
            raise ValueError(
                "Ming-Omni-TTS does not currently support SGLang overlap "
                "scheduling; set disable_overlap_schedule=true because the "
                "continuous acoustic feedback state has no overlap-safe lifecycle"
            )
        else:
            pass
        overrides["disable_overlap_schedule"] = True

        if not is_truthy(overrides["disable_radix_cache"]):
            raise ValueError(
                "Ming-Omni-TTS requires disable_radix_cache=true because "
                "prefix/radix cache is not currently supported"
            )
        else:
            pass
        overrides["disable_radix_cache"] = True

        if (
            "chunked_prefill_size" in overrides
            and int(overrides.get("chunked_prefill_size") or 0) != 0
        ):
            raise ValueError(
                "Ming-Omni-TTS requires chunked_prefill_size=0 because generated "
                "continuous state does not have chunk rollback semantics"
            )
        else:
            pass
        overrides["chunked_prefill_size"] = 0
        if is_truthy(overrides.get("enable_torch_compile", False)):
            raise ValueError("Ming-Omni-TTS torch.compile is not currently supported")
        else:
            pass

    def infra_kwargs(self) -> InfrastructureOptions:
        return {
            "tp_rank": self.tp_rank,
            "nccl_port": self.nccl_port,
            "total_gpu_memory_fraction": self.total_gpu_memory_fraction,
        }

    def setup_model(
        self,
        *,
        model_worker: ModelWorker | MlxTpModelWorker,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: ServerArgs,
    ) -> None:
        from sglang.srt.runtime_context import get_exec, get_memory

        from sglang_omni.models.ming_tts.tokenizer import load_ming_tts_tokenizer

        self.model_worker = model_worker
        model_worker.model_runner.model.eval()
        self.tokenizer = load_ming_tts_tokenizer(
            checkpoint_dir,
            llm_config=self.config.llm_config,
        )
        logger.info(
            "Ming AR SGLang startup: gpu_id=%s tp_rank=%s/%s "
            "total_gpu_memory_fraction=%s disable_cuda_graph=%s cuda_graph_bs=%s "
            "radix_cache=%s nccl_port=%s",
            gpu_id,
            self.tp_rank,
            self.tp_size,
            self.total_gpu_memory_fraction,
            bool(get_exec().graph.disable_cuda_graph),
            get_decode_cuda_graph_bs(server_args),
            not bool(get_memory().disable_radix_cache),
            self.nccl_port,
        )

    def get_model_buffer_bs(self, model: MingTTSSGLangModel) -> int | None:
        return int(model.decode_input_embedding.num_embeddings)

    def post_cuda_graph_setup(
        self, model: MingTTSSGLangModel | None, server_args: object
    ) -> None:
        del server_args
        # Note (yzxiao): Only the acoustic owner captures tail graphs because
        # follower ranks run the backbone graph without latent sampling.
        if self.tp_rank != 0:
            return
        else:
            pass
        model.init_tail_graphs(
            list(self.model_worker.model_runner.decode_cuda_graph_runner.capture_bs)
        )

    def make_model_runner(
        self,
        model_worker: ModelWorker | MlxTpModelWorker,
        output_proc: SGLangOutputProcessor | None,
    ) -> MingTTSModelRunner:
        from sglang_omni.models.ming_tts.model_runner import MingTTSModelRunner

        self.model_runner = MingTTSModelRunner(model_worker, output_proc)
        return self.model_runner

    def make_adapters(self, model: MingTTSSGLangModel | None) -> tuple[
        Callable[[StagePayload], MingTTSSGLangRequestData],
        Callable[[MingTTSSGLangRequestData], StagePayload],
    ]:
        from sglang_omni.models.ming_tts.engine_io import (
            make_ming_tts_scheduler_adapters,
        )

        return make_ming_tts_scheduler_adapters(
            model=model,
            tokenizer=self.tokenizer,
            reset_request=self.model_runner.reset_request,
            owns_acoustic_result=self.tp_rank == 0,
        )

    def extra_scheduler_kwargs(
        self,
    ) -> dict[
        str,
        Callable[[str, MingTTSSGLangRequestData, object], list[OutgoingMessage]],
    ]:
        if self.tp_rank != 0:
            return {}
        else:
            pass
        from sglang_omni.models.ming_tts.engine_io import build_ming_tts_stream_output

        return {"stream_output_builder": build_ming_tts_stream_output}

    def make_abort_callback(self) -> Callable[[str], None]:
        return self.model_runner.reset_request
