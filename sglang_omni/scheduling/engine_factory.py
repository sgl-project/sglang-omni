# SPDX-License-Identifier: Apache-2.0
"""Builders for SGLang-backed autoregressive engine stages."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from numbers import Integral
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from sglang.srt.arg_groups.model_override_base import resolved_view

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.generation_batch_policy import (
    CudaGraphBackend,
    build_generation_batch_overrides,
    get_prefill_cuda_graph_backend,
    operator_selected_prefill_backend,
    validate_generation_batch_policy,
)
from sglang_omni.scheduling.types import DeferredAdmission, RequestDataT
from sglang_omni.utils.checkpoint import resolve_checkpoint as _resolve_checkpoint

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.server_args import ServerArgs

    from sglang_omni.model_runner.base import ModelRunner
    from sglang_omni.model_runner.model_worker import ModelWorker
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend.output_processor import (
        SGLangOutputProcessor,
    )

logger = logging.getLogger(__name__)

ResultRequestT = TypeVar("ResultRequestT")


def _normalize_context_length(value: object, *, model_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(
            f"{model_name} context length must be a positive integer, got {value!r}"
        )
    context_length = int(value)
    if context_length <= 0:
        raise ValueError(
            f"{model_name} resolved an invalid context length: {context_length}"
        )
    return context_length


class SGLangGenerationEngineBuilder(ABC, Generic[RequestDataT]):
    """Build the model-neutral parts of a SGLang AR engine stage.

    Model-specific builders provide checkpoint preprocessing, model setup,
    request/result adapters, validation policy, and any stage-owned resources.
    Family-specific builders such as :class:`AsrEngineBuilder` and
    :class:`TtsEngineBuilder` define the lifecycle policy for each modality.
    """

    model_name: str
    context_length: int
    model_arch_override: str | None = None
    supports_context_length_override: ClassVar[bool] = False
    # Set True only by builders whose model has adopted the breakable prefill
    # CUDA graph contract; a deployment override cannot enable it otherwise.
    supports_breakable_prefill_cuda_graph: bool = False

    def build(
        self,
        model_path: str,
        *,
        device: str | None = None,
        gpu_id: int | None = None,
        dtype: str = "bfloat16",
        server_args_overrides: dict[str, Any] | None = None,
    ) -> "OmniScheduler[RequestDataT]":
        from sglang_omni.platforms import current_platform
        from sglang_omni.scheduling import bootstrap as scheduling_bootstrap
        from sglang_omni.scheduling import sglang_backend
        from sglang_omni.utils.device import resolve_concrete_device

        checkpoint_dir = self.resolve_checkpoint(model_path)
        concrete_device = resolve_concrete_device(device, gpu_id)
        device = str(concrete_device)
        gpu_id = concrete_device.index or 0
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.gpu_id = gpu_id
        self.dtype = dtype

        self.pre_infra_setup(checkpoint_dir)

        if current_platform.is_cpu():
            # A stage default asking for a graph would otherwise fail inside
            # capture rather than at configuration time.
            server_args_overrides = dict(server_args_overrides or {})
            server_args_overrides["disable_cuda_graph"] = True

        requested_context_length = (
            server_args_overrides.get("context_length")
            if server_args_overrides is not None
            else None
        )
        if (
            requested_context_length is not None
            and self.supports_context_length_override
        ):
            context_length = requested_context_length
        else:
            context_length = self.resolve_context_length(
                checkpoint_dir,
                server_args_overrides=server_args_overrides,
            )
        self.context_length = _normalize_context_length(
            context_length,
            model_name=self.model_name,
        )

        operator_selected = operator_selected_prefill_backend(server_args_overrides)
        overrides = build_generation_batch_overrides(
            server_args_overrides=server_args_overrides,
            **self.generation_defaults(dtype=dtype),
        )
        self.adjust_overrides(overrides)
        if "context_length" in overrides:
            if not self.supports_context_length_override:
                raise ValueError(
                    f"{self.model_name} does not support a context_length override"
                )
            overrides.pop("context_length")
        # Note (Jiaxin Deng): user fractions were rejected upstream; what remains
        # is a builder KV-tuned default, dropped so headroom derives cleanly.
        from sglang_omni.scheduling.stage_kv_budget import peek_stage_kv_cache_bytes

        if peek_stage_kv_cache_bytes() is not None:
            builder_default_fraction = overrides.pop("mem_fraction_static", None)
            if builder_default_fraction is not None:
                logger.info(
                    f"{self.model_name}: clearing builder default "
                    f"mem_fraction_static={builder_default_fraction} because the "
                    "stage declares engine.kv_cache_bytes"
                )
        sglang_backend.pin_resolved_device_type(overrides, concrete_device.type)

        server_args = sglang_backend.build_sglang_server_args(
            checkpoint_dir,
            context_length=self.context_length,
            **overrides,
        )
        self.customize_server_args(server_args)
        cfg = resolved_view(server_args)
        if (
            overrides.get("chunked_prefill_size") is None
            and cfg.cuda_graph_config.prefill.backend != CudaGraphBackend.DISABLED
        ):
            logger.info(
                f"{self.model_name}: chunked_prefill_size was unset, SGLang resolved "
                f"{cfg.chunked_prefill_size}, prefill CUDA graph cap "
                f"{cfg.cuda_graph_config.prefill.max_bs}"
            )
        self.validate_before_infrastructure(server_args)

        infra_kwargs = dict(self.infra_kwargs())
        if self.model_arch_override is not None:
            infra_kwargs.setdefault("model_arch_override", self.model_arch_override)

        def before_memory_pool(model_worker: ModelWorker | MlxTpModelWorker) -> None:
            self.before_memory_pool(
                model_worker=model_worker,
                checkpoint_dir=checkpoint_dir,
                device=device,
                gpu_id=gpu_id,
                server_args=server_args,
            )

        infra_kwargs["before_memory_pool"] = before_memory_pool
        prefill_graph_backend = get_prefill_cuda_graph_backend(server_args)
        if prefill_graph_backend == CudaGraphBackend.BREAKABLE:
            if not self.supports_breakable_prefill_cuda_graph:
                raise RuntimeError(
                    f"{self.model_name} has not adopted the breakable prefill "
                    "CUDA graph contract "
                    "(supports_breakable_prefill_cuda_graph=False); refusing "
                    "cuda_graph_backend_prefill='breakable'"
                )
            infra_kwargs.setdefault("enable_prefill_input_embeds", True)
        (
            want_cuda_graph,
            (
                model_worker,
                tree_cache,
                req_to_token_pool,
                token_to_kv_pool_allocator,
                model_config,
            ),
        ) = scheduling_bootstrap.create_sglang_infrastructure_defer_cuda_graph(
            server_args,
            gpu_id,
            **infra_kwargs,
        )
        model = model_worker.model_runner.model

        self.setup_model(
            model_worker=model_worker,
            checkpoint_dir=checkpoint_dir,
            device=device,
            gpu_id=gpu_id,
            server_args=server_args,
        )

        self.validate_after_model_setup(model, server_args)

        self.compile_model(model, server_args)

        if want_cuda_graph:
            scheduling_bootstrap.init_sglang_cuda_graphs(model_worker)
            self.post_cuda_graph_setup(model, server_args)
            if prefill_graph_backend != CudaGraphBackend.DISABLED:
                from sglang_omni.utils import cuda_graph_batch_validator

                cuda_graph_batch_validator.attest_prefill_cuda_graphs(
                    model_worker.model_runner,
                    operator_selected=operator_selected,
                )

        try:
            # Model-local encoder graphs and caches must be initialized after
            # SGLang's generation graphs to preserve the established order.
            self.setup_model_resources(
                model,
                server_args,
                generation_cuda_graph_enabled=want_cuda_graph,
            )

            output_proc = sglang_backend.SGLangOutputProcessor(
                capture_hidden=False,
                capture_hidden_layers=None,
                model=model,
            )
            self.setup_runtime_resources(model, server_args)
            scheduler, model_runner = self._build_runtime(
                model_worker=model_worker,
                model=model,
                output_proc=output_proc,
                tree_cache=tree_cache,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
                server_args=server_args,
                model_config=model_config,
            )
            self.post_scheduler_setup(scheduler, model_runner)
            return scheduler
        except Exception:
            self.cleanup_build_failure()
            raise

    def resolve_checkpoint(self, model_path: str) -> str:
        # The shared builder treats checkpoint resolution as a family policy.
        # Subclasses override this when they need a resolved local snapshot.
        return model_path

    @abstractmethod
    def generation_defaults(
        self,
        *,
        dtype: str,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        del checkpoint_dir

    def resolve_context_length(
        self,
        checkpoint_dir: str,
        *,
        server_args_overrides: Mapping[str, object] | None = None,
    ) -> int:
        del checkpoint_dir, server_args_overrides
        return self.context_length

    def validate_before_infrastructure(self, server_args: ServerArgs) -> None:
        del server_args

    def validate_after_model_setup(self, model: Any, server_args: ServerArgs) -> None:
        del model, server_args

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        del overrides

    def customize_server_args(self, server_args: ServerArgs) -> None:
        del server_args

    def infra_kwargs(self) -> Mapping[str, object]:
        return {}

    def before_memory_pool(
        self,
        *,
        model_worker: ModelWorker | MlxTpModelWorker,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: ServerArgs,
    ) -> None:
        """Attach what the stage keeps resident, before the KV pool is sized."""
        del model_worker, checkpoint_dir, device, gpu_id, server_args

    def setup_model(
        self,
        *,
        model_worker: ModelWorker | MlxTpModelWorker,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: ServerArgs,
    ) -> None:
        del model_worker, checkpoint_dir, device, gpu_id, server_args

    def get_model_buffer_bs(self, model: Any) -> int | None:
        del model
        return None

    def compile_model(self, model: Any, server_args: ServerArgs) -> None:
        del model, server_args

    def post_cuda_graph_setup(self, model: Any, server_args: ServerArgs) -> None:
        del model, server_args

    def setup_model_resources(
        self,
        model: Any,
        server_args: ServerArgs,
        *,
        generation_cuda_graph_enabled: bool,
    ) -> None:
        del model, server_args, generation_cuda_graph_enabled

    def setup_runtime_resources(self, model: Any, server_args: ServerArgs) -> None:
        del model, server_args

    @abstractmethod
    def make_model_runner(
        self,
        model_worker: ModelWorker | MlxTpModelWorker,
        output_proc: SGLangOutputProcessor,
    ) -> "ModelRunner[RequestDataT]":
        raise NotImplementedError

    @abstractmethod
    def make_adapters(
        self, model: Any
    ) -> tuple[Callable[[StagePayload], RequestDataT | DeferredAdmission] | None, Any]:
        raise NotImplementedError

    def _build_runtime(
        self,
        *,
        model_worker: ModelWorker | MlxTpModelWorker,
        model: Any,
        output_proc: SGLangOutputProcessor,
        tree_cache: BasePrefixCache,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        server_args: ServerArgs,
        model_config: ModelConfig,
    ) -> tuple["OmniScheduler[RequestDataT]", "ModelRunner[RequestDataT]"]:
        request_builder, result_adapter = self.make_adapters(model)
        scheduler_kwargs = self.extra_scheduler_kwargs()
        model_runner = self.make_model_runner(model_worker, output_proc)
        scheduler = self._make_scheduler(
            model_worker=model_worker,
            tree_cache=tree_cache,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            server_args=server_args,
            model_config=model_config,
            model_runner=model_runner,
            request_builder=request_builder,
            result_adapter=result_adapter,
            extra_scheduler_kwargs=scheduler_kwargs,
        )
        return scheduler, model_runner

    def make_abort_callback(self) -> Callable[[str], None] | None:
        return None

    def make_request_finished_callback(self) -> Callable[[str], None] | None:
        return None

    def extra_scheduler_callbacks(self) -> dict[str, Any]:
        return {}

    def cleanup_build_failure(self) -> None:
        pass

    def extra_scheduler_kwargs(self) -> Mapping[str, object]:
        return {}

    def _make_scheduler(
        self,
        *,
        model_worker: ModelWorker | MlxTpModelWorker,
        tree_cache: BasePrefixCache,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        server_args: ServerArgs,
        model_config: ModelConfig,
        model_runner: "ModelRunner[RequestDataT]",
        request_builder: (
            Callable[[StagePayload], RequestDataT | DeferredAdmission] | None
        ),
        result_adapter: Callable[[ResultRequestT], object] | None,
        extra_scheduler_kwargs: Mapping[str, object],
    ) -> "OmniScheduler[RequestDataT]":
        from sglang_omni.scheduling import omni_scheduler

        scheduler_kwargs = {
            "tp_worker": model_worker,
            "tree_cache": tree_cache,
            "req_to_token_pool": req_to_token_pool,
            "token_to_kv_pool_allocator": token_to_kv_pool_allocator,
            "server_args": server_args,
            "model_config": model_config,
            "model_runner": model_runner,
            "request_builder": request_builder,
            "result_adapter": result_adapter,
            "abort_callback": self.make_abort_callback(),
            "request_finished_callback": self.make_request_finished_callback(),
        }
        scheduler_kwargs.update(self.extra_scheduler_callbacks())
        scheduler_kwargs.update(extra_scheduler_kwargs)
        return omni_scheduler.OmniScheduler(**scheduler_kwargs)

    def post_scheduler_setup(
        self,
        scheduler: "OmniScheduler[RequestDataT]",
        model_runner: "ModelRunner[RequestDataT]",
    ) -> None:
        del scheduler, model_runner


class AsrEngineBuilder(SGLangGenerationEngineBuilder[RequestDataT]):
    """Shared lifecycle policy for SGLang-backed ASR stages."""

    def resolve_checkpoint(self, model_path: str) -> str:
        # ASR model loaders accept either a repo id or a local path and should
        # preserve the operator-provided value through server-args creation.
        return model_path

    def validate_before_infrastructure(self, server_args: ServerArgs) -> None:
        validate_generation_batch_policy(
            model_name=self.model_name,
            server_args=server_args,
        )

    def make_model_runner(
        self,
        model_worker: ModelWorker | MlxTpModelWorker,
        output_proc: SGLangOutputProcessor,
    ) -> "ModelRunner[RequestDataT]":
        from sglang_omni.model_runner.base import ModelRunner

        return ModelRunner(model_worker, output_proc)


class TtsEngineBuilder(SGLangGenerationEngineBuilder[RequestDataT]):
    """Compatibility builder preserving the historical TTS contract."""

    @abstractmethod
    def setup_model(
        self,
        *,
        model_worker: ModelWorker | MlxTpModelWorker,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: ServerArgs,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def make_model_runner(
        self,
        model_worker: ModelWorker | MlxTpModelWorker,
        output_proc: SGLangOutputProcessor,
    ) -> "ModelRunner[RequestDataT]":
        raise NotImplementedError

    def resolve_checkpoint(self, model_path: str) -> str:
        return _resolve_checkpoint(model_path)

    def validate_before_infrastructure(self, server_args: ServerArgs) -> None:
        del server_args

    def validate_after_model_setup(self, model: Any, server_args: ServerArgs) -> None:
        validate_generation_batch_policy(
            model_name=self.model_name,
            server_args=server_args,
            model_buffer_bs=self.get_model_buffer_bs(model),
        )

    def make_scheduler(
        self,
        *,
        model_worker: ModelWorker | MlxTpModelWorker,
        tree_cache: BasePrefixCache,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        server_args: ServerArgs,
        model_config: ModelConfig,
        model_runner: "ModelRunner[RequestDataT]",
        request_builder: (
            Callable[[StagePayload], RequestDataT | DeferredAdmission] | None
        ),
        result_adapter: Callable[[ResultRequestT], object] | None,
    ) -> "OmniScheduler[RequestDataT]":
        return self._make_scheduler(
            model_worker=model_worker,
            tree_cache=tree_cache,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            server_args=server_args,
            model_config=model_config,
            model_runner=model_runner,
            request_builder=request_builder,
            result_adapter=result_adapter,
            extra_scheduler_kwargs=self.extra_scheduler_kwargs(),
        )

    def _build_runtime(
        self,
        *,
        model_worker: ModelWorker | MlxTpModelWorker,
        model: Any,
        output_proc: SGLangOutputProcessor,
        tree_cache: BasePrefixCache,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        server_args: ServerArgs,
        model_config: ModelConfig,
    ) -> tuple["OmniScheduler[RequestDataT]", "ModelRunner[RequestDataT]"]:
        model_runner = self.make_model_runner(model_worker, output_proc)
        request_builder, result_adapter = self.make_adapters(model)
        scheduler = self.make_scheduler(
            model_worker=model_worker,
            tree_cache=tree_cache,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            server_args=server_args,
            model_config=model_config,
            model_runner=model_runner,
            request_builder=request_builder,
            result_adapter=result_adapter,
        )
        return scheduler, model_runner
