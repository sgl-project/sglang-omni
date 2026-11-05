# SPDX-License-Identifier: Apache-2.0
"""BF16 decode-graph policy and adapters on the shared engine builder."""

from __future__ import annotations

from typing import Protocol

import torch
from sglang.srt.arg_groups.model_override_base import (
    attention_backends_of,
    resolved_view,
)
from sglang.srt.server_args import ServerArgs

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.moss_speech.hf_config import (
    KV_RESERVE_TOKENS,
    MAX_CONTEXT_TOKENS,
    ensure_moss_speech_config_registered,
)
from sglang_omni.models.moss_speech.model_runner import MossSpeechModelRunner
from sglang_omni.models.moss_speech.request_builders import (
    MossSpeechSGLangRequestData,
    build_result,
    build_sglang_moss_request,
)
from sglang_omni.models.moss_speech.sglang_model import MossSpeechSGLangModel
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.engine_factory import (
    GenerationDefaults,
    SGLangGenerationEngineBuilder,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor


class RequestBuilder(Protocol):
    def __call__(self, payload: StagePayload) -> MossSpeechSGLangRequestData: ...


class ResultAdapter(Protocol):
    def __call__(self, request: MossSpeechSGLangRequestData) -> StagePayload: ...


class MossSpeechEngineBuilder(
    SGLangGenerationEngineBuilder[MossSpeechSGLangRequestData]
):
    model_name = "MOSS-Speech AR"
    context_length = MAX_CONTEXT_TOKENS
    model_arch_override = "MossSpeechForCausalLM"

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        ensure_moss_speech_config_registered()

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        max_running_requests = self.server_args_overrides.get("max_running_requests")
        if (
            isinstance(max_running_requests, bool)
            or not isinstance(max_running_requests, int)
            or max_running_requests <= 0
        ):
            raise ValueError(
                "MOSS-Speech requires a positive max_running_requests in the stage engine configuration"
            )
        else:
            return {
                "max_running_requests": max_running_requests,
                "attention_backend": "flashinfer",
                "dtype": dtype,
                "cuda_graph_max_bs": max_running_requests,
                "enable_torch_compile": False,
                "disable_radix_cache": True,
                "tp_size": 1,
                "chunked_prefill_size": -1,
                "trust_remote_code": False,
            }

    def adjust_overrides(self, overrides: dict[str, object]) -> None:
        for name, value in self.generation_defaults(dtype="bfloat16").items():
            if name in ("attention_backend", "cuda_graph_max_bs"):
                continue
            elif name in overrides and overrides[name] != value:
                raise ValueError(
                    f"MOSS-Speech inference rejects {name}={overrides[name]!r}"
                )
            else:
                overrides[name] = value
        if overrides.get("quantization") is not None:
            raise ValueError("MOSS-Speech inference does not support quantization")
        else:
            pass
        if overrides.get("kv_cache_dtype", "auto") not in ("auto", "bfloat16", "bf16"):
            raise ValueError("MOSS-Speech requires BF16 KV storage")
        else:
            pass
        if overrides.get("dtype") not in ("bfloat16", "bf16"):
            raise ValueError("MOSS-Speech requires BF16 model weights")
        else:
            pass

    def validate_before_infrastructure(self, server_args: ServerArgs) -> None:
        configuration = resolved_view(server_args)
        problems = []
        if configuration.enable_torch_compile:
            problems.append("torch.compile is not supported")
        else:
            pass
        if (
            not configuration.disable_radix_cache
            or configuration.chunked_prefill_size != -1
        ):
            problems.append("prefix caching and chunked prefill must be disabled")
        else:
            pass
        if configuration.tp_size != 1:
            problems.append("tensor parallel size must be one")
        else:
            pass
        if configuration.dtype not in ("bfloat16", "bf16"):
            problems.append("model dtype must be BF16")
        else:
            pass
        attention_backends = attention_backends_of(configuration)
        if any(
            backend not in ("torch_native", "flashinfer")
            for backend in attention_backends
        ):
            problems.append("attention must use flashinfer or torch_native")
        elif (
            not configuration.disable_cuda_graph
            and attention_backends[1] != "flashinfer"
        ):
            problems.append("decode CUDA Graph requires flashinfer attention")
        else:
            pass
        if problems:
            raise ValueError(f"{self.model_name}: {', '.join(problems)}")
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
        assert isinstance(model, MossSpeechSGLangModel)
        configuration = resolved_view(server_args)
        if (
            configuration.max_running_requests
            > model.decode_input_embedding.num_embeddings
        ):
            raise ValueError(
                "max_running_requests exceeds the MOSS-Speech decode embedding capacity"
            )
        else:
            pass
        model.decode_input_embedding = model.decode_input_embedding.to(
            torch.device(device)
        )

    def make_model_runner(
        self, model_worker: ModelWorker, output_proc: SGLangOutputProcessor
    ) -> MossSpeechModelRunner:
        return MossSpeechModelRunner(model_worker, output_proc)

    def post_scheduler_setup(
        self,
        scheduler: OmniScheduler[MossSpeechSGLangRequestData],
        model_runner: MossSpeechModelRunner,
    ) -> None:
        assert scheduler.model_runner is model_runner
        context_tokens = min(
            scheduler.model_config.context_len,
            scheduler.max_total_num_tokens - KV_RESERVE_TOKENS,
        )
        # Upstream clipping subtracts one from this sentinel; physical KV guards remain reserved.
        scheduler.max_req_len = context_tokens + 1
        scheduler.max_req_input_len = context_tokens

    def make_adapters(
        self, model: torch.nn.Module
    ) -> tuple[RequestBuilder, ResultAdapter]:
        assert isinstance(model, MossSpeechSGLangModel)
        return build_sglang_moss_request, build_result
