# SPDX-License-Identifier: Apache-2.0
"""Starts the SGLang engine for PersonaPlex from a Llama-shaped shim checkpoint."""

from __future__ import annotations

import atexit
import json
import shutil
import tempfile
from pathlib import Path
from typing import Protocol

import torch
from sglang.srt.server_args import ServerArgs

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.personaplex.architecture import MOSHI_WEIGHTS_NAME
from sglang_omni.models.personaplex.hf_config import (
    DEFAULT_CONTEXT_LENGTH,
    PERSONAPLEX_ARCH,
    build_backbone_config,
)
from sglang_omni.models.personaplex.model_runner import PersonaPlexModelRunner
from sglang_omni.models.personaplex.request_builders import (
    apply_lm_result,
    build_lm_request,
    lm_stream_output_builder,
)
from sglang_omni.models.personaplex.sglang_model import PersonaPlexForCausalLM
from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.engine_factory import (
    GenerationDefaults,
    SchedulerExtras,
    TtsEngineBuilder,
)
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


class LMRequestBuilder(Protocol):
    def __call__(self, payload: StagePayload) -> SGLangARRequestData: ...


class LMResultBuilder(Protocol):
    def __call__(self, data: SGLangARRequestData) -> StagePayload: ...


def shim_checkpoint_dir(source: Path, *, context_length: int) -> Path:
    """A directory SGLang can load: the LM weights and a Llama config.

    Only model.safetensors is linked. The Mimi weights sit next to it in
    the checkpoint and SGLang would otherwise sweep them into the LM.
    """
    source = source.resolve()
    weights = source / MOSHI_WEIGHTS_NAME
    if not weights.is_file():
        raise FileNotFoundError(f"PersonaPlex LM weights missing: {weights}")
    else:
        pass
    shim = Path(tempfile.mkdtemp(prefix="sglang-omni-personaplex-"))
    atexit.register(shutil.rmtree, shim, ignore_errors=True)
    (shim / MOSHI_WEIGHTS_NAME).symlink_to(weights)
    (shim / "config.json").write_text(
        json.dumps(build_backbone_config(context_length=context_length), indent=2)
    )
    return shim


class PersonaPlexEngineBuilder(TtsEngineBuilder[SGLangARRequestData]):
    model_name = "personaplex"
    context_length = DEFAULT_CONTEXT_LENGTH
    supports_context_length_override = True

    def __init__(
        self, *, max_running_requests: int = 1, context_length: int | None = None
    ) -> None:
        self.max_running_requests = max_running_requests
        self.model_arch_override = PERSONAPLEX_ARCH
        if context_length is not None:
            self.context_length = int(context_length)
        else:
            pass

    def resolve_checkpoint(self, model_path: str) -> str:
        source = Path(resolve_model_path(model_path))
        return str(shim_checkpoint_dir(source, context_length=self.context_length))

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        return {
            "disable_cuda_graph": True,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_torch_compile": False,
            "max_running_requests": self.max_running_requests,
            "chunked_prefill_size": -1,
            "dtype": dtype,
            "trust_remote_code": False,
            # Note (wilsonzheng0327): Seeded top-k sampling needs the torch sampler.
            "sampling_backend": "pytorch",
        }

    def setup_model(
        self,
        *,
        model_worker: ModelWorker,
        checkpoint_dir: str,
        device: str | torch.device,
        gpu_id: int,
        server_args: ServerArgs,
    ) -> None:
        """Nothing beyond SGLang's own load: the model owns its buffers."""

    def make_model_runner(
        self,
        model_worker: ModelWorker,
        output_proc: SGLangOutputProcessor,
    ) -> PersonaPlexModelRunner:
        return PersonaPlexModelRunner(model_worker, output_proc)

    def make_adapters(self, model: PersonaPlexForCausalLM) -> tuple[
        LMRequestBuilder,
        LMResultBuilder,
    ]:
        vocab_size = int(model.config.vocab_size)

        def build(payload: StagePayload) -> SGLangARRequestData:
            return build_lm_request(
                payload, vocab_size=vocab_size, context_length=self.context_length
            )

        return build, apply_lm_result

    def extra_scheduler_kwargs(self) -> SchedulerExtras[SGLangARRequestData]:
        return {"stream_output_builder": lm_stream_output_builder}
