# SPDX-License-Identifier: Apache-2.0
"""Build the native CUDA stage for Haru realtime sessions."""

from __future__ import annotations

import atexit
import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from haru.config import HaruConfig
from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.server_args import ServerArgs
from torch import nn

from sglang_omni.admission import REQUEST_TO_TOKEN_SLOTS_RESERVED_FOR_RETAINED_KV
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.haru.config import (
    ARCHITECTURE,
    INPUT_FRAME_SAMPLES,
    SAMPLE_RATE_HZ,
)
from sglang_omni.models.haru.runner import HaruModelRunner, HaruSessionAdapter
from sglang_omni.models.haru.sglang_model import HaruSGLangModel
from sglang_omni.scheduling.engine_factory import GenerationDefaults, TtsEngineBuilder
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData

DEFAULT_CONTEXT_LENGTH = 32_768


class HaruEngineBuilder(TtsEngineBuilder[SGLangARRequestData]):
    model_name = "Haru"
    model_arch_override = ARCHITECTURE
    supports_context_length_override = True
    context_length = DEFAULT_CONTEXT_LENGTH

    def __init__(
        self,
        *,
        max_sessions: int,
        codec_backend: Literal["torch", "triton", "auto"],
    ) -> None:
        self.max_sessions: int = max_sessions
        self.codec_backend: Literal["torch", "triton", "auto"] = codec_backend
        self.source_directory: Path
        self.runner: HaruModelRunner

    def resolve_checkpoint(self, model_path: str) -> str:
        self.source_directory = Path(super().resolve_checkpoint(model_path)).resolve()
        haru_config = json.loads((self.source_directory / "config.json").read_text())
        audio_config = HaruConfig.from_dict(haru_config)
        if (
            audio_config.sample_rate != SAMPLE_RATE_HZ
            or audio_config.frame_samples != INPUT_FRAME_SAMPLES
        ):
            raise ValueError(
                f"Haru realtime requires {INPUT_FRAME_SAMPLES} samples per frame "
                f"at {SAMPLE_RATE_HZ} Hz, but the checkpoint declares "
                f"{audio_config.frame_samples} samples at {audio_config.sample_rate} Hz"
            )
        else:
            pass
        # note (BeiningWu): The SGLang config loader selects the hybrid cache from the Qwen3.5 text config class.
        backbone_config = {
            **haru_config["lm"],
            "model_type": "qwen3_5_text",
            "architectures": [ARCHITECTURE],
            "dtype": haru_config["dtype"],
            "haru_config": haru_config,
        }
        shim = Path(tempfile.mkdtemp(prefix="sglang-omni-haru-"))
        atexit.register(shutil.rmtree, shim, ignore_errors=True)
        for filename in ("model.safetensors", "tokenizer.json"):
            checkpoint_file = self.source_directory / filename
            if not checkpoint_file.is_file():
                raise FileNotFoundError(f"Haru checkpoint is missing {checkpoint_file}")
            else:
                (shim / filename).symlink_to(checkpoint_file)
        (shim / "config.json").write_text(json.dumps(backbone_config))
        return str(shim)

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        ModelRegistry.models[ARCHITECTURE] = HaruSGLangModel

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        return {
            "chunked_prefill_size": -1,
            "disable_cuda_graph": True,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "dtype": dtype,
            "enable_streaming_session": True,
            "enable_torch_compile": False,
            "max_running_requests": (
                self.max_sessions + REQUEST_TO_TOKEN_SLOTS_RESERVED_FOR_RETAINED_KV
            ),
            "max_total_tokens": self.max_sessions * self.context_length,
            "sampling_backend": "pytorch",
            "skip_tokenizer_init": True,
            "trust_remote_code": False,
        }

    def validate_before_infrastructure(self, server_args: ServerArgs) -> None:
        config = resolved_view(server_args)
        if config.device != "cuda" or config.tp_size != 1 or config.pp_size != 1:
            raise ValueError("Haru realtime requires a single CUDA device")
        elif not config.enable_streaming_session or not config.disable_radix_cache:
            raise ValueError(
                "Haru requires streaming sessions with radix cache disabled"
            )
        elif not config.disable_cuda_graph or not config.disable_overlap_schedule:
            raise ValueError("Haru requires eager, synchronous inference")
        elif config.max_running_requests < (
            self.max_sessions + REQUEST_TO_TOKEN_SLOTS_RESERVED_FOR_RETAINED_KV
        ):
            raise ValueError(
                "Haru needs a free request slot alongside its retained sessions"
            )
        else:
            pass

    def setup_model(
        self,
        *,
        model_worker: ModelWorker,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: ServerArgs,
    ) -> None:
        model = model_worker.model_runner.model
        assert isinstance(model, HaruSGLangModel)
        assert isinstance(
            model_worker.model_runner.req_to_token_pool, HybridReqToTokenPool
        )

    def make_adapters(self, model: nn.Module) -> tuple[None, None]:
        return None, None

    def make_model_runner(
        self,
        model_worker: ModelWorker,
        output_proc: SGLangOutputProcessor,
    ) -> HaruModelRunner:
        self.runner = HaruModelRunner(model_worker, output_proc)
        return self.runner

    def extra_scheduler_kwargs(self) -> dict[str, HaruSessionAdapter]:
        return {
            "session_adapter": HaruSessionAdapter(
                self.runner,
                model_path=str(self.source_directory),
                context_length=self.context_length,
                codec_backend=self.codec_backend,
            )
        }


def create_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    max_sessions: int = 1,
    codec_backend: Literal["torch", "triton", "auto"] = "auto",
    server_args_overrides: Mapping[str, object] | None = None,
    total_gpu_memory_fraction: float | None = None,
) -> OmniScheduler[SGLangARRequestData]:
    builder = HaruEngineBuilder(max_sessions=max_sessions, codec_backend=codec_backend)
    return builder.build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
    )
