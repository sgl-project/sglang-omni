# SPDX-License-Identifier: Apache-2.0
"""Build the native Audio8 TTS engine and adapt its stage payloads."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Protocol

import torch
from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.hf_transformers_utils import attach_additional_stop_token_ids
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.audio8_tts.hf_config import ArkttsConfig
from sglang_omni.models.audio8_tts.model_runner import (
    Audio8ModelRunner,
    Audio8RequestData,
)
from sglang_omni.models.audio8_tts.payload_types import (
    Audio8SamplingOptions,
    Audio8State,
)
from sglang_omni.models.audio8_tts.sglang_model import Audio8SGLangModel
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.engine_factory import GenerationDefaults, TtsEngineBuilder
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.sglang_backend.request_data import validate_prompt_token_ids

DEFAULT_MAX_RUNNING_REQUESTS = 16


class Audio8RequestBuilder(Protocol):
    def __call__(self, payload: StagePayload, /) -> Audio8RequestData: ...


class Audio8ResultAdapter(Protocol):
    def __call__(self, request_data: Audio8RequestData, /) -> StagePayload: ...


class Audio8EngineBuilder(TtsEngineBuilder[Audio8RequestData]):
    """Pair the dense Audio8 checkpoint with SGLang's paged KV cache."""

    model_name: str = "Audio8 TTS"
    model_arch_override: str = Audio8SGLangModel.__name__
    checkpoint_config: ArkttsConfig
    tokenizer: PreTrainedTokenizerBase
    model_device: torch.device

    def __init__(self, max_new_tokens: int) -> None:
        if max_new_tokens < 1:
            raise ValueError("Audio8 max_new_tokens must be positive")
        else:
            self.max_new_tokens_cap: int = max_new_tokens

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        self.checkpoint_config = ArkttsConfig.from_pretrained(checkpoint_dir)
        AutoConfig.register(ArkttsConfig.model_type, ArkttsConfig, exist_ok=True)
        ModelRegistry.models[self.model_arch_override] = Audio8SGLangModel
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_dir, trust_remote_code=False
        )
        attach_additional_stop_token_ids(self.tokenizer)

    def resolve_context_length(
        self,
        checkpoint_dir: str,
        *,
        server_args_overrides: Mapping[str, object] | None = None,
    ) -> int:
        return self.checkpoint_config.max_position_embeddings

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        # note (Beining): reference codec states are embedded by the runner, so radix reuse is disabled.
        return {
            "dtype": dtype,
            "max_running_requests": DEFAULT_MAX_RUNNING_REQUESTS,
            "max_total_tokens": DEFAULT_MAX_RUNNING_REQUESTS * self.context_length,
            "disable_cuda_graph": True,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_mixed_chunk": False,
            "enable_torch_compile": False,
            "trust_remote_code": False,
            "chunked_prefill_size": -1,
            "mem_fraction_static": 0.55,
        }

    def validate_before_infrastructure(self, server_args: ServerArgs) -> None:
        resolved_args = resolved_view(server_args)
        if resolved_args.tp_size != 1 or resolved_args.pp_size != 1:
            raise ValueError("Audio8 requires tensor and pipeline parallel sizes of 1")
        elif resolved_args.trust_remote_code:
            raise ValueError("Audio8 uses the registered native checkpoint loader")
        elif (
            not resolved_args.disable_cuda_graph
            or not resolved_args.disable_overlap_schedule
            or not resolved_args.disable_radix_cache
            or resolved_args.enable_mixed_chunk
            or resolved_args.enable_torch_compile
        ):
            raise ValueError(
                "Audio8 requires eager, non-overlapped decode and a disabled radix cache"
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
        self.model_device = torch.device(device)

    def make_model_runner(
        self, model_worker: ModelWorker, output_proc: SGLangOutputProcessor
    ) -> Audio8ModelRunner:
        output_proc.capture_hidden = True
        return Audio8ModelRunner(model_worker, output_proc)

    def make_adapters(
        self, model: torch.nn.Module
    ) -> tuple[Audio8RequestBuilder, Audio8ResultAdapter]:
        assert isinstance(model, Audio8SGLangModel)
        return self.build_request, make_vocoder_payload

    def build_request(self, payload: StagePayload) -> Audio8RequestData:
        config = self.checkpoint_config
        prompt_codes = Audio8State.from_dict(payload.data).prompt_codes
        assert isinstance(prompt_codes, torch.Tensor)
        assert prompt_codes.ndim == 2 and prompt_codes.dtype == torch.long
        assert (
            0 < prompt_codes.shape[0]
            and prompt_codes.shape[1] == config.num_codebooks + 1
        )
        prompt_token_ids = prompt_codes[:, 0]
        validate_prompt_token_ids(prompt_token_ids, config.vocab_size)

        sampling_options = Audio8SamplingOptions.model_validate(payload.request.params)
        max_new_tokens = min(sampling_options.max_new_tokens, self.max_new_tokens_cap)
        random_generator = (
            torch.Generator(device=self.model_device).manual_seed(sampling_options.seed)
            if sampling_options.seed is not None
            else None
        )
        sampling_params = SamplingParams(
            max_new_tokens=max_new_tokens,
            stop_token_ids=[config.eos_token_id],
        )
        sampling_params.normalize(self.tokenizer)
        sampling_params.verify(config.vocab_size)
        request = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=prompt_token_ids.tolist(),
            sampling_params=sampling_params,
            eos_token_ids={config.eos_token_id},
            vocab_size=config.vocab_size,
        )
        request.tokenizer = self.tokenizer
        return Audio8RequestData(
            req=request,
            input_ids=prompt_token_ids,
            prompt_codes=prompt_codes.to(self.model_device),
            temperature=sampling_options.temperature,
            top_p=sampling_options.top_p,
            top_k=sampling_options.top_k,
            max_new_tokens=max_new_tokens,
            enforce_request_limits=True,
            semantic_history_token_ids=deque(maxlen=config.ras_window_size),
            random_generator=random_generator,
            stage_payload=payload,
        )


def make_vocoder_payload(request_data: Audio8RequestData) -> StagePayload:
    """Hand the generated codec frames to the waveform decoder."""
    payload = request_data.stage_payload
    assert payload is not None
    if not request_data.generated_codec_frames:
        raise ValueError("Audio8 generated no codec frames")
    else:
        audio_codes = torch.stack(request_data.generated_codec_frames, dim=1)
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=Audio8State(
            audio_codes=audio_codes,
            prompt_tokens=len(request_data.prompt_codes),
            completion_tokens=audio_codes.shape[1],
            finish_reason=request_data.finish_reason or "stop",
        ).to_dict(),
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    max_new_tokens: int,
    device: str | None = None,
    gpu_id: int | None = None,
    server_args_overrides: Mapping[str, object] | None = None,
) -> OmniScheduler[Audio8RequestData]:
    return Audio8EngineBuilder(max_new_tokens).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        server_args_overrides=server_args_overrides,
    )
