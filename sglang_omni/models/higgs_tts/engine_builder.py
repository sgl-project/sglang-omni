# SPDX-License-Identifier: Apache-2.0
"""Higgs TTS SGLang engine builder."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from typing import Any

from sglang_omni.models.higgs_tts import request_builders
from sglang_omni.models.higgs_tts import utils as higgs_utils
from sglang_omni.models.higgs_tts.vocoder_scheduler import (
    DEFAULT_HIGGS_INITIAL_CHUNK_FRAMES,
    DEFAULT_HIGGS_STREAM_FOLLOWUP_STRIDE,
    DEFAULT_HIGGS_STREAM_STRIDE,
)
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder
from sglang_omni.scheduling.generation_batch_policy import (
    CudaGraphBackend,
    build_default_prefill_cuda_graph_bs,
)
from sglang_omni.vendor.sglang.server_args import override_server_args

logger = logging.getLogger(__name__)


class HiggsTtsEngineBuilder(TtsEngineBuilder):
    model_name = "Higgs TTS"
    context_length = 4096
    supports_breakable_prefill_cuda_graph = True

    def _uses_mps(self) -> bool:
        # Placement is resolved by build() before generation defaults are read.
        return str(getattr(self, "device", "")).split(":")[0] == "mps"

    @staticmethod
    def _uses_mlx() -> bool:
        from sglang.srt.hardware_backend.mlx.runtime import use_mlx

        return use_mlx()

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        if self._uses_mlx():
            self.model_arch_override = "HiggsTTSModel"

    def _mps_requirements(self) -> dict[str, Any]:
        return {
            "max_running_requests": 1,
            "disable_cuda_graph": True,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_torch_compile": False,
            "chunked_prefill_size": -1,
            "attention_backend": "torch_native",
            "sampling_backend": "pytorch",
        }

    def __init__(
        self,
        *,
        max_new_tokens: int | None,
        max_running_requests: int,
        cuda_graph_max_bs: int,
        enable_async_decode: bool,
        async_decode_min_batch_size: int,
        stream_stride: int = DEFAULT_HIGGS_STREAM_STRIDE,
        stream_followup_stride: int = DEFAULT_HIGGS_STREAM_FOLLOWUP_STRIDE,
        initial_chunk_frames: int = DEFAULT_HIGGS_INITIAL_CHUNK_FRAMES,
        prefill_coalesce_requests: int = 0,
        prefill_coalesce_wait_ms: float = 60.0,
        total_gpu_memory_fraction: float | None = None,
    ) -> None:
        if total_gpu_memory_fraction is not None and not (
            0.0 < total_gpu_memory_fraction < 1.0
        ):
            raise ValueError(
                "Higgs tts_engine total_gpu_memory_fraction must be in (0, 1): "
                "it drives sglang mem_fraction_static, which requires < 1"
            )
        self.max_new_tokens = max_new_tokens
        self.max_running_requests = max_running_requests
        self.cuda_graph_max_bs = cuda_graph_max_bs
        self.enable_async_decode = enable_async_decode
        self.async_decode_min_batch_size = async_decode_min_batch_size
        self.stream_stride = stream_stride
        self.stream_followup_stride = stream_followup_stride
        self.initial_chunk_frames = initial_chunk_frames
        self.prefill_coalesce_requests = prefill_coalesce_requests
        self.prefill_coalesce_wait_ms = prefill_coalesce_wait_ms
        self.total_gpu_memory_fraction = total_gpu_memory_fraction
        self.model: Any | None = None

    def generation_defaults(
        self,
        *,
        dtype: str,
    ) -> dict[str, Any]:
        del dtype
        if self._uses_mlx() and not self._uses_mps():
            raise ValueError("Higgs MLX requires an MPS stage device")
        if self._uses_mps():
            # Reference embeddings and delayed sampler state have no split-
            # prefill rollback contract. Start with whole prompts and no reuse.
            return {
                **self._mps_requirements(),
                "cuda_graph_max_bs": 1,
                "cuda_graph_backend_prefill": CudaGraphBackend.DISABLED,
                "max_prefill_tokens": self.context_length,
                # The HF runner owns its actual KV cache; bound the shared
                # scheduler pool instead of reserving most unified memory twice.
                "max_total_tokens": self.context_length,
                "mem_fraction_static": (
                    self.total_gpu_memory_fraction
                    if self.total_gpu_memory_fraction is not None
                    else 0.85
                ),
                "dtype": "bfloat16",
            }
        # note (luojiaxuan): Radix cache is namespaced per ref-audio via
        # Req.extra_key (set in build_sglang_higgs_request); shared -100
        # placeholder prefixes from different ref audios can't cross-contaminate
        # the KV tree.
        return {
            "max_running_requests": self.max_running_requests,
            "cuda_graph_max_bs": self.cuda_graph_max_bs,
            "disable_cuda_graph": False,
            "mem_fraction_static": (
                self.total_gpu_memory_fraction
                if self.total_gpu_memory_fraction is not None
                else 0.85
            ),
            "chunked_prefill_size": 8192,
            # Qualified capture budget; longer prefills run eager.
            "cuda_graph_backend_prefill": CudaGraphBackend.BREAKABLE,
            "cuda_graph_bs_prefill": build_default_prefill_cuda_graph_bs(512),
            "dtype": "bfloat16",
        }

    def _validate_mlx_dtype(self, dtype: str) -> None:
        # The MLX worker currently loads both language and audio weights in BF16.
        # Reject overrides instead of silently ignoring the requested precision.
        if self._uses_mlx() and dtype not in ("bfloat16", "bf16"):
            raise ValueError("Higgs MLX requires dtype='bfloat16'")

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        if self._uses_mps():
            self._validate_mlx_dtype(overrides.get("dtype", "bfloat16"))
            if self._uses_mlx() and overrides.get("mlx_enable_sampling", False):
                raise ValueError(
                    "Higgs MLX uses its own sampler; mlx_enable_sampling must be False"
                )
            if overrides.get("quantization") is not None:
                raise ValueError("Higgs Apple requires unquantized weights")
            for key, expected_value in self._mps_requirements().items():
                if overrides.get(key, expected_value) != expected_value:
                    raise ValueError(
                        f"Higgs Torch MPS requires {key}={expected_value!r}"
                    )
            for key in ("prefill_attention_backend", "decode_attention_backend"):
                if overrides.get(key) not in (None, "torch_native"):
                    raise ValueError(f"Higgs Torch MPS requires {key}='torch_native'")
            if (
                overrides.get("max_prefill_tokens", self.context_length)
                < self.context_length
            ):
                raise ValueError(
                    "Higgs Torch MPS max_prefill_tokens must cover context_length"
                )
            for phase in ("decode", "prefill"):
                if (
                    overrides.get(
                        f"cuda_graph_backend_{phase}", CudaGraphBackend.DISABLED
                    )
                    != CudaGraphBackend.DISABLED
                ):
                    raise ValueError(
                        f"Higgs Torch MPS requires disabled {phase} CUDA graphs"
                    )
            config = overrides.get("cuda_graph_config", {})
            if hasattr(config, "to_dict"):
                config = config.to_dict()
            if isinstance(config, Mapping):
                for phase in ("decode", "prefill"):
                    settings = config.get(phase, {})
                    if (
                        settings.get("backend", CudaGraphBackend.DISABLED)
                        != CudaGraphBackend.DISABLED
                    ):
                        raise ValueError(
                            f"Higgs Torch MPS requires disabled {phase} CUDA graphs"
                        )
        # Note: (Jiaxin Deng) an explicit mem_fraction_static override (e.g.
        # --tts_engine.engine.mem_fraction_static) wins, but never silently.
        expected = self.total_gpu_memory_fraction
        if expected is None:
            return
        actual = overrides.get("mem_fraction_static")
        if actual is not None and abs(actual - expected) <= 1e-9:
            return
        logger.warning(
            "Higgs tts_engine mem_fraction_static=%s overrides the "
            "placement-validated total_gpu_memory_fraction=%s",
            actual,
            expected,
        )

    def validate_before_infrastructure(self, server_args: Any) -> None:
        if not self._uses_mps():
            return
        from sglang.srt.arg_groups.model_override_base import resolved_view

        cfg = resolved_view(server_args)
        self._validate_mlx_dtype(getattr(cfg, "dtype", "bfloat16"))
        # Validate the resolved record too: nested graph settings take precedence
        # over legacy flags in SGLang's typed configuration.
        for key, expected in self._mps_requirements().items():
            if getattr(cfg, key) != expected:
                raise ValueError(f"Higgs Torch MPS requires {key}={expected!r}")
        for phase in ("decode", "prefill"):
            if (
                getattr(cfg.cuda_graph_config, phase).backend
                != CudaGraphBackend.DISABLED
            ):
                raise ValueError(
                    f"Higgs Torch MPS requires disabled {phase} CUDA graphs"
                )

    def customize_server_args(self, server_args: Any) -> None:
        override_server_args(
            server_args,
            "sglang_omni.higgs_tts.disable_overlap_schedule",
            disable_overlap_schedule=True,
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
        del checkpoint_dir, device, gpu_id, server_args
        self.model = model_worker.model_runner.model
        if self._uses_mps():
            from sglang_omni.models.higgs_tts.torch_mps_runner import (
                install_torch_mps_language_model,
            )

            if not self._uses_mlx():
                install_torch_mps_language_model(self.model, self.checkpoint_dir)
        higgs_utils.truncate_rope_to_bf16(self.model)

    def get_model_buffer_bs(self, model: Any) -> int | None:
        return model.sampler_pool_max_running_requests

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        if self._uses_mps():
            from sglang_omni.models.higgs_tts.torch_mps_runner import (
                HiggsTorchMpsModelRunner,
            )

            runner_cls = HiggsTorchMpsModelRunner
            if self._uses_mlx():
                from sglang_omni.models.higgs_tts.mlx.scheduler_runner import (
                    HiggsMlxModelRunner,
                )

                runner_cls = HiggsMlxModelRunner
            self._mps_runner = runner_cls(model_worker, output_proc)
            return self._mps_runner
        model_runner_mod = importlib.import_module(
            "sglang_omni.models.higgs_tts.model_runner"
        )

        return model_runner_mod.HiggsTTSModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        del model
        return request_builders.make_higgs_scheduler_adapters(
            max_new_tokens_cap=self.max_new_tokens,
            stream_stride=self.stream_stride,
            stream_followup_stride=self.stream_followup_stride,
            initial_chunk_frames=self.initial_chunk_frames,
        )

    def make_abort_callback(self) -> Any | None:
        assert self.model is not None
        if self._uses_mps():
            return self._mps_runner.reset_request
        return self.model.reset_request

    def make_request_finished_callback(self) -> Any | None:
        assert self.model is not None
        if self._uses_mps():
            return self._mps_runner.reset_request
        return self.model.reset_request

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {
            "enable_async_decode": (
                False if self._uses_mps() else self.enable_async_decode
            ),
            "async_decode_min_batch_size": self.async_decode_min_batch_size,
            "prefill_coalesce_requests": (
                0 if self._uses_mps() else self.prefill_coalesce_requests
            ),
            "prefill_coalesce_wait_ms": self.prefill_coalesce_wait_ms,
        }

    def post_scheduler_setup(self, scheduler: Any, model_runner: Any) -> None:
        model_runner.set_stream_outbox(scheduler.outbox)
