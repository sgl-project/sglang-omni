# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS Local SGLang engine builder."""

from __future__ import annotations

import importlib
from typing import Any

from sglang_omni.models.moss_tts_local import request_builders
from sglang_omni.models.moss_tts_local import stages as moss_local_stages
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder


class MossTtsLocalEngineBuilder(TtsEngineBuilder):
    model_name = "MOSS-TTS Local"
    context_length = 8192
    model_arch_override = "MossTTSLocalSGLangModel"

    def __init__(
        self,
        *,
        enable_async_decode: bool,
        async_decode_min_batch_size: int,
        total_gpu_memory_fraction: float | None,
        codec_mem_reserve: float,
    ) -> None:
        self.enable_async_decode = enable_async_decode
        self.async_decode_min_batch_size = async_decode_min_batch_size
        self.total_gpu_memory_fraction = total_gpu_memory_fraction
        self.codec_mem_reserve = codec_mem_reserve
        self.memory_budget = moss_local_stages._ArMemoryBudget(
            effective_total_gpu_memory_fraction=None,
            applied_codec_mem_reserve=0.0,
        )
        self.profile_total_gpu_memory_fraction: float | None = None
        self.model: Any | None = None

    def resolve_checkpoint(self, model_path: str) -> str:
        return moss_local_stages.resolve_moss_checkpoint(model_path)

    def generation_defaults(
        self,
        *,
        dtype: str,
    ) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "max_running_requests": 16,
            "dtype": dtype,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "enable_torch_compile": False,
            "max_prefill_tokens": 8192,
            "sampling_backend": "pytorch",
            "trust_remote_code": True,
        }
        if self.total_gpu_memory_fraction is None:
            defaults["mem_fraction_static"] = (
                0.6 if moss_local_stages.torch.cuda.device_count() > 1 else 0.5
            )
        return defaults

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        self.memory_budget = moss_local_stages._apply_colocated_ar_memory_budget(
            overrides,
            total_gpu_memory_fraction=self.total_gpu_memory_fraction,
            codec_mem_reserve=self.codec_mem_reserve,
        )
        self.profile_total_gpu_memory_fraction = (
            self.memory_budget.effective_total_gpu_memory_fraction
        )
        if self.profile_total_gpu_memory_fraction is None:
            return

        from sglang_omni.utils.gpu_memory import get_process_gpu_memory_bytes

        if get_process_gpu_memory_bytes(self.gpu_id) is None:
            moss_local_stages.logger.warning(
                f"MOSS-TTS Local colocated process memory accounting is unavailable; "
                f"falling back to upstream SGLang free-memory profiling. "
                f"effective_total_gpu_memory_fraction="
                f"{self.profile_total_gpu_memory_fraction}"
            )
            self.profile_total_gpu_memory_fraction = None

    def customize_server_args(self, server_args: Any) -> None:
        moss_local_stages.logger.info(
            f"MOSS-TTS Local SGLang startup: gpu_id={self.gpu_id} "
            f"total_gpu_memory_fraction={self.total_gpu_memory_fraction} "
            f"effective_total_gpu_memory_fraction="
            f"{self.memory_budget.effective_total_gpu_memory_fraction} "
            f"codec_mem_reserve={self.memory_budget.applied_codec_mem_reserve:.3f} "
            f"mem_fraction_static={server_args.mem_fraction_static} "
            f"profile_total_gpu_memory_fraction="
            f"{self.profile_total_gpu_memory_fraction}"
        )

    def infra_kwargs(self) -> dict[str, Any]:
        return {
            "total_gpu_memory_fraction": self.profile_total_gpu_memory_fraction,
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
        del checkpoint_dir, device, gpu_id, server_args
        self.model = model_worker.model_runner.model

    def post_cuda_graph_setup(self, model: Any, server_args: Any) -> None:
        # note (luojiaxuan): Also graph the per-frame local-transformer decode
        # (1 + n_vq micro-steps and 13 seeded sampling passes per frame):
        # eager it is kernel-launch-bound at ~22 ms/frame independent of batch
        # size.
        model.init_frame_decode_graphs(list(server_args.cuda_graph_bs))

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        model_runner_mod = importlib.import_module(
            "sglang_omni.models.moss_tts_local.model_runner"
        )

        return model_runner_mod.MossTTSLocalModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        return request_builders.make_moss_tts_local_scheduler_adapters(model=model)

    def make_abort_callback(self) -> Any | None:
        assert self.model is not None
        model = self.model

        def abort_request(request_id: str) -> None:
            request_builders.cleanup_prepared_moss_tts_local_request(request_id)
            model.reset_request(request_id)

        return abort_request

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {
            "enable_async_decode": self.enable_async_decode,
            "async_decode_min_batch_size": self.async_decode_min_batch_size,
        }

    def post_scheduler_setup(self, scheduler: Any, model_runner: Any) -> None:
        model_runner.set_stream_outbox(scheduler.outbox)
