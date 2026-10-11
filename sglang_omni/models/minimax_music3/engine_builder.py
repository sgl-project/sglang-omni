# SPDX-License-Identifier: Apache-2.0
"""MiniMax Music 3 SGLang engine builder — OmniScheduler integration."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

from torch import Tensor
from torch_memory_saver import torch_memory_saver
from typing_extensions import Unpack

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.minimax_music3.serial_offload import get_coordinator
from sglang_omni.models.minimax_music3.weight_cache import checkpoint_identity
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.engine_factory import (
    GenerationDefaults,
    SchedulerExtras,
    TtsEngineBuilder,
)
from sglang_omni.scheduling.generation_batch_policy import build_default_cuda_graph_bs
from sglang_omni.scheduling.types import DeferredAdmission

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.models.qwen3 import Qwen3ForCausalLM
    from sglang.srt.server_args import ServerArgs

    from sglang_omni.models.minimax_music3.model_runner import MiniMaxMusic3ModelRunner
    from sglang_omni.models.minimax_music3.scheduler import MiniMaxMusic3Scheduler
    from sglang_omni.models.minimax_music3.sglang_request_builder import (
        MiniMaxMusic3SGLangRequestData,
    )
    from sglang_omni.scheduling.sglang_backend import SGLangOutputProcessor
else:
    pass

logger = logging.getLogger(__name__)

_AUDIO_WEIGHT_PREFIXES = ("model.audio_decoder.", "model.audio_extra_embedding.")


class MiniMaxMusic3SchedulerArguments(TypedDict):
    model_worker: ModelWorker | MlxTpModelWorker
    tree_cache: BasePrefixCache
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    server_args: ServerArgs
    model_config: ModelConfig
    model_runner: ModelRunner[MiniMaxMusic3SGLangRequestData]
    request_builder: (
        Callable[
            [StagePayload],
            MiniMaxMusic3SGLangRequestData
            | DeferredAdmission[MiniMaxMusic3SGLangRequestData],
        ]
        | None
    )
    result_adapter: Callable[[MiniMaxMusic3SGLangRequestData], StagePayload] | None


def rvq_graph_buckets(max_running_requests: int) -> list[int]:
    """Batch buckets to capture the RVQ depth pass at."""
    buckets, size = [], 2
    while size < 2 * max_running_requests:
        buckets.append(size)
        size *= 2
    buckets.append(2 * max_running_requests)
    return buckets


class MiniMaxMusic3EngineBuilder(TtsEngineBuilder["MiniMaxMusic3SGLangRequestData"]):
    model_name = "minimax_music3"
    context_length = 10240
    model_arch_override = "Qwen3ForCausalLM"

    def __init__(
        self,
        *,
        max_running_requests: int = 16,
        enable_serial_offload: bool = False,
        serial_offload_source: Literal["mmap", "ram"] = "mmap",
        serial_offload_cache_dir: str | None = None,
    ) -> None:
        self.max_running_requests = int(max_running_requests)
        if self.max_running_requests <= 0:
            raise ValueError("MiniMax Music 3 max_running_requests must be positive")
        else:
            pass
        self.enable_serial_offload = bool(enable_serial_offload)
        self.serial_offload_source = serial_offload_source
        self.serial_offload_cache_dir = serial_offload_cache_dir
        self.model_runner: MiniMaxMusic3ModelRunner | None = None
        self.checkpoint_root: str | None = None

    def resolve_checkpoint(self, model_path: str) -> str:
        from .checkpoint import resolve_checkpoint

        paths = resolve_checkpoint(model_path)
        self.checkpoint_root = str(paths.root)
        return str(paths.qwen_dir)

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        self.normalize_backbone_config(Path(checkpoint_dir) / "config.json")
        self.filter_audio_weights()

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        return {
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_torch_compile": False,
            "enable_deterministic_inference": True,
            "max_running_requests": (
                1 if self.enable_serial_offload else self.max_running_requests
            ),
            "chunked_prefill_size": 0,
            "mem_fraction_static": 0.50,
            "dtype": dtype,
            "trust_remote_code": False,
        }

    def adjust_overrides(self, overrides: dict[str, object]) -> None:
        if int(overrides.get("tp_size", 1)) != 1:
            raise ValueError("MiniMax Music 3 does not support TP")
        else:
            pass
        requested = int(
            overrides.get("max_running_requests", self.max_running_requests)
        )
        if requested <= 0:
            raise ValueError("MiniMax Music 3 max_running_requests must be positive")
        else:
            pass
        if self.enable_serial_offload:
            # Note (Akazaakane): One CFG pair must finish before AR weights leave the GPU.
            requested = 1
            overrides["enable_memory_saver"] = True
            overrides["enable_weights_cpu_backup"] = False
            overrides["enable_torch_compile"] = False
            overrides["disable_overlap_schedule"] = True
        else:
            pass
        self.max_running_requests = requested
        rows = 2 * requested
        overrides["max_running_requests"] = rows
        overrides["cuda_graph_max_bs"] = rows
        overrides["cuda_graph_bs"] = build_default_cuda_graph_bs(rows)
        overrides["disable_radix_cache"] = True
        overrides["chunked_prefill_size"] = 0
        if not bool(overrides.get("disable_cuda_graph", False)):
            overrides["enable_return_hidden_states"] = True
        else:
            pass

    def setup_model(
        self,
        *,
        model_worker: ModelWorker | MlxTpModelWorker,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: ServerArgs,
    ) -> None:
        del checkpoint_dir, device, gpu_id
        from sglang.srt.runtime_context import get_exec, get_schedule

        from sglang_omni.scheduling.generation_batch_policy import (
            get_decode_cuda_graph_max_bs,
        )

        from .sglang_model import attach_minimax_modules, enable_graph_feedback

        assert self.checkpoint_root is not None
        model = model_worker.model_runner.model
        if self.enable_serial_offload:
            with torch_memory_saver.region(tag="music3_audio", enable_cpu_backup=False):
                attach_minimax_modules(model, self.checkpoint_root)
        else:
            attach_minimax_modules(model, self.checkpoint_root)
        if not bool(get_exec().graph.disable_cuda_graph):
            enable_graph_feedback(
                model,
                max(
                    int(get_schedule().max_running_requests),
                    int(get_decode_cuda_graph_max_bs(server_args) or 0),
                ),
            )
        else:
            pass
        model.eval()

    def setup_model_resources(
        self,
        model: Qwen3ForCausalLM,
        server_args: ServerArgs,
        *,
        generation_cuda_graph_enabled: bool,
    ) -> None:
        del server_args
        if self.enable_serial_offload:
            assert self.checkpoint_root is not None
            paths = list(Path(self.checkpoint_root).rglob("*.safetensors"))
            paths.append(
                Path(self.checkpoint_root) / "qwen_7B" / "qwen_7B" / "config.json"
            )
            get_coordinator().register_ar(
                model,
                next(model.parameters()).device,
                source=self.serial_offload_source,
                cache_dir=self.serial_offload_cache_dir,
                checkpoint_contents=checkpoint_identity(paths),
            )
        else:
            pass
        from .sglang_model import enable_rvq_depth_cuda_graph

        if generation_cuda_graph_enabled:
            enable_rvq_depth_cuda_graph(
                model, rvq_graph_buckets(self.max_running_requests)
            )
        else:
            model.rvq_depth_graph = None

    def make_scheduler(
        self, **kwargs: Unpack[MiniMaxMusic3SchedulerArguments]
    ) -> MiniMaxMusic3Scheduler:
        from .scheduler import MiniMaxMusic3Scheduler

        return MiniMaxMusic3Scheduler(
            tp_worker=kwargs.pop("model_worker"),
            abort_callback=self.make_abort_callback(),
            request_finished_callback=self.make_request_finished_callback(),
            **kwargs,
            **self.extra_scheduler_kwargs(),
        )

    def make_model_runner(
        self,
        model_worker: ModelWorker | MlxTpModelWorker,
        output_proc: SGLangOutputProcessor,
    ) -> MiniMaxMusic3ModelRunner:
        from .model_runner import MiniMaxMusic3ModelRunner

        self.model_runner = MiniMaxMusic3ModelRunner(model_worker, output_proc)
        return self.model_runner

    def make_adapters(self, model: object) -> tuple[
        Callable[[StagePayload], MiniMaxMusic3SGLangRequestData],
        Callable[[MiniMaxMusic3SGLangRequestData], StagePayload],
    ]:
        del model
        from transformers import AutoTokenizer

        from .checkpoint import resolve_checkpoint
        from .prompt import validate_tokenizer_ids
        from .sglang_request_builder import (
            apply_minimax_result,
            build_sglang_minimax_request,
        )

        assert self.checkpoint_root is not None
        paths = resolve_checkpoint(self.checkpoint_root)
        tokenizer = AutoTokenizer.from_pretrained(
            str(paths.tokenizer_dir), trust_remote_code=False
        )
        validate_tokenizer_ids(tokenizer)

        def build_request(payload: StagePayload) -> MiniMaxMusic3SGLangRequestData:
            return build_sglang_minimax_request(payload, tokenizer)

        return build_request, apply_minimax_result

    def make_abort_callback(self) -> Callable[[str], None]:
        assert self.model_runner is not None
        return self.model_runner.reset_request

    def make_request_finished_callback(self) -> Callable[[str], None]:
        return get_coordinator().cancel_ar

    def extra_scheduler_kwargs(self) -> SchedulerExtras[MiniMaxMusic3SGLangRequestData]:
        from .sglang_request_builder import build_stream_output

        return {
            "stream_output_builder": build_stream_output,
            "enable_async_decode": False,
        }

    @staticmethod
    def normalize_backbone_config(config_path: Path) -> None:
        """Rewrite the backbone config so HuggingFace resolves a Qwen3 config."""
        config = json.loads(config_path.read_text())
        if config.get("model_type") == "qwen3":
            return
        else:
            pass
        backup_path = config_path.with_suffix(".json.bak")
        if not backup_path.exists():
            backup_path.write_text(config_path.read_text())
        else:
            pass
        config["model_type"] = "qwen3"
        config_path.unlink()
        config_path.write_text(json.dumps(config, indent=2))
        logger.info(
            f"MiniMax Music 3: rewrote {config_path} model_type to qwen3 (backup at {backup_path})"
        )

    @staticmethod
    def filter_audio_weights() -> None:
        from sglang.srt.models.qwen3 import Qwen3ForCausalLM

        if getattr(
            Qwen3ForCausalLM.load_weights, "_minimax_filtered", False
        ):  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            return
        else:
            pass
        load_weights = Qwen3ForCausalLM.load_weights

        def filtered_load_weights(
            self: Qwen3ForCausalLM, weights: Iterable[tuple[str, Tensor]]
        ) -> None:
            return load_weights(
                self,
                (
                    (name, tensor)
                    for name, tensor in weights
                    if not name.startswith(_AUDIO_WEIGHT_PREFIXES)
                ),
            )

        filtered_load_weights._minimax_filtered = True  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        Qwen3ForCausalLM.load_weights = filtered_load_weights
        logger.info("MiniMax Music 3: Qwen3 weight loader now skips audio-module keys")


__all__ = ["MiniMaxMusic3EngineBuilder"]
