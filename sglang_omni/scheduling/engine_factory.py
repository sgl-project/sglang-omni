# SPDX-License-Identifier: Apache-2.0
"""Builders for SGLang-backed autoregressive engine stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from sglang_omni.scheduling.generation_batch_policy import (
    CudaGraphBackend,
    build_generation_batch_overrides,
    get_prefill_cuda_graph_backend,
    nested_prefill_overrides,
    validate_generation_batch_policy,
)
from sglang_omni.utils.checkpoint import resolve_checkpoint as _resolve_checkpoint


def _operator_selected_prefill_graph_backend(
    server_args_overrides: Mapping[str, Any] | None,
) -> bool:
    if not server_args_overrides:
        return False
    if "cuda_graph_backend_prefill" in server_args_overrides:
        return True
    return "backend" in nested_prefill_overrides(server_args_overrides)


class SGLangGenerationEngineBuilder(ABC):
    """Build the model-neutral parts of a SGLang AR engine stage.

    Model-specific builders provide checkpoint preprocessing, model setup,
    request/result adapters, validation policy, and any stage-owned resources.
    Family-specific builders such as :class:`AsrEngineBuilder` and
    :class:`TtsEngineBuilder` define the lifecycle policy for each modality.
    """

    model_name: str
    context_length: int
    model_arch_override: str | None = None
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
    ) -> Any:
        import torch

        from sglang_omni.scheduling import bootstrap as scheduling_bootstrap
        from sglang_omni.scheduling import sglang_backend
        from sglang_omni.utils.device import place_device_spec, resolve_device_spec

        checkpoint_dir = self.resolve_checkpoint(model_path)
        device = (
            resolve_device_spec(None, gpu_id)
            if device is None
            else place_device_spec(device, gpu_id)
        )
        gpu_id = torch.device(device).index or 0
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.gpu_id = gpu_id
        self.dtype = dtype

        self.pre_infra_setup(checkpoint_dir)

        operator_selected_prefill_backend = _operator_selected_prefill_graph_backend(
            server_args_overrides
        )
        overrides = build_generation_batch_overrides(
            server_args_overrides=server_args_overrides,
            **self.generation_defaults(dtype=dtype),
        )
        self.adjust_overrides(overrides)
        # Left unset, SGLang re-detects off a CUDA-first ladder that can contradict
        # placement. It owns the type, not the index.
        resolved_type = torch.device(device).type
        requested_type = overrides.get("device")
        if requested_type is not None and requested_type != resolved_type:
            raise ValueError(
                f"server_args_overrides set device={requested_type!r}, but this stage "
                f"resolved to {device!r}. Omni owns placement, so drop the override or "
                f"set device={resolved_type!r}."
            )
        overrides["device"] = resolved_type

        server_args = sglang_backend.build_sglang_server_args(
            checkpoint_dir,
            context_length=self.context_length,
            **overrides,
        )
        self.customize_server_args(server_args)
        self.validate_before_infrastructure(server_args)

        infra_kwargs = dict(self.infra_kwargs())
        if self.model_arch_override is not None:
            infra_kwargs.setdefault("model_arch_override", self.model_arch_override)
        prefill_graph_backend = get_prefill_cuda_graph_backend(server_args)
        if (
            prefill_graph_backend != CudaGraphBackend.DISABLED
            and not operator_selected_prefill_backend
        ):
            # SGLang treats every non-default source as operator-locked. A
            # model-qualified stage default should survive compatibility
            # resolution, but must remain eligible for the late free-memory
            # safety gate immediately before graph capture.
            server_args._cuda_graph_config_locked.discard(("prefill", "backend"))
        if prefill_graph_backend == CudaGraphBackend.BREAKABLE:
            if not self.supports_breakable_prefill_cuda_graph:
                raise RuntimeError(
                    f"{self.model_name} has not adopted the breakable prefill "
                    "CUDA graph contract "
                    "(supports_breakable_prefill_cuda_graph=False); refusing "
                    "cuda_graph_backend_prefill='breakable'"
                )
            infra_kwargs.setdefault("enable_prefill_input_embeds", True)
        want_cuda_graph, (
            model_worker,
            tree_cache,
            req_to_token_pool,
            token_to_kv_pool_allocator,
            prefill_mgr,
            decode_mgr,
            model_config,
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
                    model_worker.model_runner, server_args
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
                prefill_manager=prefill_mgr,
                decode_manager=decode_mgr,
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

    def validate_before_infrastructure(self, server_args: Any) -> None:
        del server_args

    def validate_after_model_setup(self, model: Any, server_args: Any) -> None:
        del model, server_args

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        del overrides

    def customize_server_args(self, server_args: Any) -> None:
        del server_args

    def infra_kwargs(self) -> dict[str, Any]:
        return {}

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

    def get_model_buffer_bs(self, model: Any) -> int | None:
        del model
        return None

    def compile_model(self, model: Any, server_args: Any) -> None:
        del model, server_args

    def post_cuda_graph_setup(self, model: Any, server_args: Any) -> None:
        del model, server_args

    def setup_model_resources(
        self,
        model: Any,
        server_args: Any,
        *,
        generation_cuda_graph_enabled: bool,
    ) -> None:
        del model, server_args, generation_cuda_graph_enabled

    def setup_runtime_resources(self, model: Any, server_args: Any) -> None:
        del model, server_args

    @abstractmethod
    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        raise NotImplementedError

    def _build_runtime(
        self,
        *,
        model_worker: Any,
        model: Any,
        output_proc: Any,
        tree_cache: Any,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        server_args: Any,
        model_config: Any,
        prefill_manager: Any,
        decode_manager: Any,
    ) -> tuple[Any, Any]:
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
            prefill_manager=prefill_manager,
            decode_manager=decode_manager,
            model_runner=model_runner,
            request_builder=request_builder,
            result_adapter=result_adapter,
            extra_scheduler_kwargs=scheduler_kwargs,
        )
        return scheduler, model_runner

    def make_abort_callback(self) -> Any | None:
        return None

    def make_request_finished_callback(self) -> Any | None:
        return None

    def extra_scheduler_callbacks(self) -> dict[str, Any]:
        return {}

    def cleanup_build_failure(self) -> None:
        pass

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {}

    def _make_scheduler(
        self,
        *,
        model_worker: Any,
        tree_cache: Any,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        server_args: Any,
        model_config: Any,
        prefill_manager: Any,
        decode_manager: Any,
        model_runner: Any,
        request_builder: Any,
        result_adapter: Any,
        extra_scheduler_kwargs: dict[str, Any],
    ) -> Any:
        from sglang_omni.scheduling import omni_scheduler

        scheduler_kwargs = {
            "tp_worker": model_worker,
            "tree_cache": tree_cache,
            "req_to_token_pool": req_to_token_pool,
            "token_to_kv_pool_allocator": token_to_kv_pool_allocator,
            "server_args": server_args,
            "model_config": model_config,
            "prefill_manager": prefill_manager,
            "decode_manager": decode_manager,
            "model_runner": model_runner,
            "request_builder": request_builder,
            "result_adapter": result_adapter,
            "abort_callback": self.make_abort_callback(),
            "request_finished_callback": self.make_request_finished_callback(),
        }
        scheduler_kwargs.update(self.extra_scheduler_callbacks())
        scheduler_kwargs.update(extra_scheduler_kwargs)
        return omni_scheduler.OmniScheduler(**scheduler_kwargs)

    def post_scheduler_setup(self, scheduler: Any, model_runner: Any) -> None:
        del scheduler, model_runner


class AsrEngineBuilder(SGLangGenerationEngineBuilder):
    """Shared lifecycle policy for SGLang-backed ASR stages."""

    def resolve_checkpoint(self, model_path: str) -> str:
        # ASR model loaders accept either a repo id or a local path and should
        # preserve the operator-provided value through server-args creation.
        return model_path

    def validate_before_infrastructure(self, server_args: Any) -> None:
        validate_generation_batch_policy(
            model_name=self.model_name,
            server_args=server_args,
        )

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        from sglang_omni.model_runner.base import ModelRunner

        return ModelRunner(model_worker, output_proc)


class TtsEngineBuilder(SGLangGenerationEngineBuilder):
    """Compatibility builder preserving the historical TTS contract."""

    @abstractmethod
    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        raise NotImplementedError

    def resolve_checkpoint(self, model_path: str) -> str:
        return _resolve_checkpoint(model_path)

    def validate_before_infrastructure(self, server_args: Any) -> None:
        del server_args

    def validate_after_model_setup(self, model: Any, server_args: Any) -> None:
        validate_generation_batch_policy(
            model_name=self.model_name,
            server_args=server_args,
            model_buffer_bs=self.get_model_buffer_bs(model),
        )

    def make_scheduler(
        self,
        *,
        model_worker: Any,
        tree_cache: Any,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        server_args: Any,
        model_config: Any,
        prefill_manager: Any,
        decode_manager: Any,
        model_runner: Any,
        request_builder: Any,
        result_adapter: Any,
    ) -> Any:
        from sglang_omni.scheduling import omni_scheduler

        return omni_scheduler.OmniScheduler(
            tp_worker=model_worker,
            tree_cache=tree_cache,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            server_args=server_args,
            model_config=model_config,
            prefill_manager=prefill_manager,
            decode_manager=decode_manager,
            model_runner=model_runner,
            request_builder=request_builder,
            result_adapter=result_adapter,
            abort_callback=self.make_abort_callback(),
            request_finished_callback=self.make_request_finished_callback(),
            **self.extra_scheduler_kwargs(),
        )

    def _build_runtime(
        self,
        *,
        model_worker: Any,
        model: Any,
        output_proc: Any,
        tree_cache: Any,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        server_args: Any,
        model_config: Any,
        prefill_manager: Any,
        decode_manager: Any,
    ) -> tuple[Any, Any]:
        model_runner = self.make_model_runner(model_worker, output_proc)
        request_builder, result_adapter = self.make_adapters(model)
        scheduler = self.make_scheduler(
            model_worker=model_worker,
            tree_cache=tree_cache,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            server_args=server_args,
            model_config=model_config,
            prefill_manager=prefill_manager,
            decode_manager=decode_manager,
            model_runner=model_runner,
            request_builder=request_builder,
            result_adapter=result_adapter,
        )
        return scheduler, model_runner
