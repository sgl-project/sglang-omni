# SPDX-License-Identifier: Apache-2.0
"""Starts the SGLang engine for PersonaPlex from a Llama-shaped shim checkpoint."""

from __future__ import annotations

import atexit
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Protocol

import torch
from sentencepiece import SentencePieceProcessor
from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.server_args import ServerArgs

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.personaplex.architecture import MOSHI_WEIGHTS_NAME, TEXT_CARD
from sglang_omni.models.personaplex.config import DEFAULT_LM_MAX_RUNNING_REQUESTS
from sglang_omni.models.personaplex.hf_config import (
    DEFAULT_CONTEXT_LENGTH,
    PERSONAPLEX_ARCH,
    build_backbone_config,
)
from sglang_omni.models.personaplex.lm_session import PersonaPlexSessionAdapter
from sglang_omni.models.personaplex.model_runner import PersonaPlexModelRunner
from sglang_omni.models.personaplex.request_builders import (
    apply_lm_result,
    build_lm_request,
    lm_stream_output_builder,
)
from sglang_omni.models.personaplex.sglang_model import PersonaPlexForCausalLM
from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.platforms import current_platform
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.engine_factory import (
    GenerationDefaults,
    SchedulerExtras,
    TtsEngineBuilder,
)
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.vendor.sglang.server_args import override_server_args

logger = logging.getLogger(__name__)


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
        self,
        *,
        enable_async_decode: bool,
        max_running_requests: int = DEFAULT_LM_MAX_RUNNING_REQUESTS,
        context_length: int | None = None,
    ) -> None:
        self.max_running_requests = max_running_requests
        self.enable_async_decode = enable_async_decode
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
            # note (wilsonzheng0327): The decode step reads its row from fusion_buffer
            # and leaves its hidden state in hidden_out, so SGLang can replay it.
            # Other platforms keep the eager path they were validated on.
            "disable_cuda_graph": not current_platform.is_cuda(),
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

    def customize_server_args(self, server_args: ServerArgs) -> None:
        config = resolved_view(server_args)
        if (
            current_platform.is_cuda()
            and config.device == "cuda"
            and config.page_size == 1
            and config.disable_radix_cache
        ):
            pass
        else:
            override_server_args(
                server_args,
                "sglang_omni.personaplex.window_kv",
                disable_hybrid_swa_memory=True,
            )
            logger.info(
                f"PersonaPlex uses the full KV pool on device={config.device}, "
                f"platform={current_platform.device_type} "
                f"with page_size={config.page_size}, "
                f"disable_radix_cache={config.disable_radix_cache}; "
                "window KV requires CUDA, page_size=1 and disabled radix caching"
            )

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
        return {
            "stream_output_builder": lm_stream_output_builder,
            "async_decode_resolve_before_prefill": True,
            "enable_async_decode": self.enable_async_decode
            and current_platform.is_cuda(),
        }


class PersonaPlexSessionEngineBuilder(PersonaPlexEngineBuilder):
    """The LM of full-duplex sessions: one SGLang streaming session each."""

    def __init__(
        self,
        *,
        tokenizer: SentencePieceProcessor,
        enable_async_decode: bool,
        context_length: int | None = None,
    ) -> None:
        super().__init__(
            enable_async_decode=enable_async_decode, context_length=context_length
        )
        self.tokenizer = tokenizer

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        return {
            **super().generation_defaults(dtype=dtype),
            "enable_streaming_session": True,
        }

    def extra_scheduler_kwargs(self) -> dict[str, PersonaPlexSessionAdapter]:
        # Note (wilsonzheng0327): The shim config's vocabulary is TEXT_CARD.
        return {
            "session_adapter": PersonaPlexSessionAdapter(
                vocab_size=TEXT_CARD,
                context_length=self.context_length,
                tokenizer=self.tokenizer,
            )
        }
