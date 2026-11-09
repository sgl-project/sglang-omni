# SPDX-License-Identifier: Apache-2.0
"""SGLang engine builder for the YuE2 AR stage."""

from collections.abc import Callable
from pathlib import Path

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from sglang.srt.server_args import ServerArgs

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.engine_factory import (
    GenerationDefaults,
    SchedulerExtras,
    TtsEngineBuilder,
)
from sglang_omni.scheduling.generation_batch_policy import build_default_cuda_graph_bs
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.utils.checkpoint import resolve_checkpoint

from .layers import Yue2ModelConfig
from .model_runner import Yue2ModelRunner
from .request import Yue2Tokenizer
from .scheduler import Yue2Scheduler
from .sglang_model import register_yue2_config
from .sglang_request_builder import Yue2ARRequestData, Yue2RequestBuilder


class Yue2EngineBuilder(TtsEngineBuilder[Yue2ARRequestData]):
    model_name = "yue2"

    def __init__(self, *, max_running_requests: int) -> None:
        if max_running_requests <= 0:
            raise ValueError("YuE2 max_running_requests must be positive")
        else:
            self.max_running_requests = max_running_requests

    def resolve_checkpoint(self, model_path: str) -> str:
        directory = resolve_checkpoint(model_path)
        self.model_config = Yue2ModelConfig.model_validate_json(
            (Path(directory) / "config.json").read_text()
        )
        self.context_length = self.model_config.max_position_embeddings
        return directory

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        del checkpoint_dir
        register_yue2_config()

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        return {
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_torch_compile": False,
            "enable_deterministic_inference": True,
            "max_running_requests": self.max_running_requests,
            "chunked_prefill_size": 0,
            "mem_fraction_static": 0.5,
            "dtype": dtype,
            "trust_remote_code": False,
        }

    def adjust_overrides(self, overrides: dict[str, object]) -> None:
        if int(overrides.get("tp_size", 1)) != 1:
            raise ValueError("YuE2 does not support TP")
        else:
            pass
        requested = int(
            overrides.get("max_running_requests", self.max_running_requests)
        )
        if requested <= 0:
            raise ValueError("YuE2 max_running_requests must be positive")
        else:
            pass
        self.max_running_requests = requested
        # note (ruiling): every request may carry a CFG twin, so batches hold up to two rows each.
        rows = 2 * requested
        overrides["max_running_requests"] = rows
        overrides["cuda_graph_max_bs"] = rows
        overrides["cuda_graph_bs"] = build_default_cuda_graph_bs(rows)
        overrides["disable_radix_cache"] = True
        overrides["chunked_prefill_size"] = 0

    def setup_model(
        self,
        *,
        model_worker: ModelWorker,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: ServerArgs,
    ) -> None:
        del checkpoint_dir, device, gpu_id, server_args
        model_worker.model_runner.model.eval()

    def make_model_runner(
        self, model_worker: ModelWorker, output_proc: SGLangOutputProcessor
    ) -> Yue2ModelRunner:
        return Yue2ModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Qwen3ForCausalLM) -> tuple[
        Callable[[StagePayload], Yue2ARRequestData],
        Callable[[Yue2ARRequestData], StagePayload],
    ]:
        del model
        builder = Yue2RequestBuilder(
            Yue2Tokenizer(Path(self.checkpoint_dir) / "qwen.tiktoken"),
            context_tokens=self.model_config.max_position_embeddings,
            vocab_size=self.model_config.vocab_size,
        )
        return builder, builder.result

    def make_scheduler(
        self,
        *,
        model_worker: ModelWorker,
        tree_cache: BasePrefixCache,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        server_args: ServerArgs,
        model_config: ModelConfig,
        model_runner: ModelRunner[Yue2ARRequestData],
        request_builder: Callable[[StagePayload], Yue2ARRequestData] | None,
        result_adapter: Callable[[Yue2ARRequestData], StagePayload] | None,
        extra_scheduler_kwargs: SchedulerExtras[Yue2ARRequestData] | None = None,
    ) -> Yue2Scheduler:
        return Yue2Scheduler(
            tp_worker=model_worker,
            tree_cache=tree_cache,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            server_args=server_args,
            model_config=model_config,
            model_runner=model_runner,
            request_builder=request_builder,
            result_adapter=result_adapter,
            **self.extra_scheduler_kwargs(),
        )

    def extra_scheduler_kwargs(self) -> SchedulerExtras[Yue2ARRequestData]:
        return {"enable_async_decode": False}
