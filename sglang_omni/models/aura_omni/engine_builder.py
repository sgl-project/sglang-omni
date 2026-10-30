# SPDX-License-Identifier: Apache-2.0
"""SGLang engine builder for the AURA stage.

AURA is a stock Qwen3-VL checkpoint, so the stage runs SGLang's own model and
multimodal processor; only the request and result adapters are AURA's.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from array import array
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar, TypedDict

import torch
from sglang.srt.environ import envs
from sglang.srt.managers.mm_utils import init_mm_embedding_cache
from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
from sglang.srt.managers.schedule_batch import MultimodalInputs, Req
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from transformers import (
    AutoProcessor,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.aura_omni.request_builders import (
    DECODE_STAGE,
    AuraAnswer,
    AuraTurn,
    render_aura_prompt,
)
from sglang_omni.platforms import current_platform
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.engine_factory import (
    GenerationDefaults,
    SchedulerExtras,
    SGLangGenerationEngineBuilder,
)
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import RequestOutput
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.cpu import effective_cpu_count

logger = logging.getLogger(__name__)

SILENT_TOKEN = "<|silent|>"
SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "repetition_penalty")
# About a minute of video: the 6.5 s, 30 fps test clip is 3,456 video tokens at
# the processor defaults.
AURA_CONTEXT_LENGTH = 32768
AURA_MAX_RUNNING_REQUESTS = 8
# Measured on a 6.5 s 4K clip (Arc Pro B60 host): one thread preprocesses it in
# 4.4 s, eight in 2.4 s, and more threads gain nothing.
AURA_PREPROCESSING_INTRAOP_THREADS = 8


@dataclass
class AuraRequestData(SGLangARRequestData):
    prompt_token_count: int = 0


class SamplingDefaults(TypedDict):
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float


def generation_default(value: float | int | None, unset: float | int) -> float | int:
    """A generation-config field, or SGLang's neutral value when the checkpoint omits it."""
    if value is None:
        return unset
    else:
        return value


def validate_silent_token(
    tokenizer: PreTrainedTokenizerBase, checkpoint_dir: str
) -> None:
    """Refuse a checkpoint whose tokenizer cannot mark a silent turn the way AURA does."""
    token_ids = tokenizer(SILENT_TOKEN, add_special_tokens=False).input_ids
    if len(token_ids) != 1 or tokenizer.decode(token_ids, skip_special_tokens=True):
        raise ValueError(
            f"AURA Omni serves AURA checkpoints: {checkpoint_dir} has no special "
            f"{SILENT_TOKEN} token, so a silent turn would reach Qwen3-TTS as text"
        )
    else:
        pass


def make_aura_scheduler_adapters(
    *,
    tokenizer: PreTrainedTokenizerBase,
    mm_processor: BaseMultimodalProcessor,
    model: Qwen3VLForConditionalGeneration,
    device: torch.device,
    sampling_defaults: SamplingDefaults,
    stop_token_ids: list[int],
    max_new_tokens: int,
    context_length: int,
) -> tuple[
    Callable[[StagePayload], AuraRequestData],
    Callable[[AuraRequestData], StagePayload],
]:
    vocab_size = len(tokenizer)

    def request_builder(payload: StagePayload) -> AuraRequestData:
        # note (siju): SGLang's processor sends media to the thread's current
        # device, and request-build threads start on the first visible card.
        current_platform.set_device(device)
        turn: AuraTurn = payload.data
        prompt = render_aura_prompt(turn)
        if turn["images"] or turn["videos"]:
            if turn["video_sampling"]:
                # A video request bypasses the processor's cache, so a per-request
                # copy changes this request's frame budget and no other's.
                request_processor = copy.copy(mm_processor)
                request_processor.video_config = {
                    **mm_processor.video_config,
                    **turn["video_sampling"],
                }
            else:
                request_processor = mm_processor
            processed = asyncio.run(
                request_processor.process_mm_data_async(
                    image_data=list(turn["images"]),
                    input_text=prompt,
                    request_obj=SimpleNamespace(
                        rid=payload.request_id,
                        video_data=list(turn["videos"]) or None,
                        audio_data=None,
                    ),
                )
            )
            mm_inputs = MultimodalInputs.from_processor_output(processed)
            input_ids = array("q", processed.input_ids)
            if mm_inputs.padded_input_ids is not None and len(
                mm_inputs.padded_input_ids
            ) == len(input_ids):
                input_ids = array("q", mm_inputs.padded_input_ids)
            else:
                input_ids = model.pad_input_ids(input_ids, mm_inputs)
        else:
            mm_inputs = None
            input_ids = array(
                "q", tokenizer(prompt, add_special_tokens=False).input_ids
            )

        params = payload.request.params
        request_max_new_tokens = (
            int(params["max_new_tokens"])
            if params.get("max_new_tokens") is not None
            else max_new_tokens
        )
        if len(input_ids) + request_max_new_tokens > context_length - 1:
            raise ValueError(
                f"AURA Omni request needs {len(input_ids)} prompt tokens + "
                f"{request_max_new_tokens} max_new_tokens, more than the "
                f"{context_length - 1} the aura stage's context holds; send a "
                "shorter clip or raise aura.engine.context_length"
            )
        else:
            pass
        sampling = {
            field: params[field] if field in params else sampling_defaults[field]
            for field in SAMPLING_FIELDS
        }
        sampling_params = SamplingParams(
            max_new_tokens=request_max_new_tokens,
            temperature=float(sampling["temperature"]),
            top_p=float(sampling["top_p"]),
            top_k=int(sampling["top_k"]),
            repetition_penalty=float(sampling["repetition_penalty"]),
            stop_token_ids=stop_token_ids,
        )
        sampling_params.normalize(tokenizer=None)
        req = Req(
            rid=payload.request_id,
            origin_input_text=prompt,
            origin_input_ids=input_ids,
            sampling_params=sampling_params,
            vocab_size=vocab_size,
        )
        req.multimodal_inputs = mm_inputs
        logger.info(
            f"AURA request {payload.request_id}: {len(turn['images'])} images, "
            f"{len(turn['videos'])} videos, {len(turn['transcript'])} transcript "
            f"chars, {len(input_ids)} prompt tokens"
        )
        return AuraRequestData(
            input_ids=torch.tensor(input_ids.tolist(), dtype=torch.long),
            req=req,
            max_new_tokens=request_max_new_tokens,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            repetition_penalty=sampling_params.repetition_penalty,
            stage_payload=payload,
            prompt_token_count=len(input_ids),
        )

    def result_adapter(data: AuraRequestData) -> StagePayload:
        payload = data.stage_payload
        output_ids = list(data.output_ids)
        text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        answer: AuraAnswer = {
            "text": text,
            # note (siju): AURA answers silence with its own special token, so
            # nothing speakable is left once special tokens are removed.
            "is_silent": not text,
            "output_ids": output_ids,
            "finish_reason": data.finish_reason,
            "prompt_tokens": data.prompt_token_count,
        }
        return StagePayload(
            request_id=payload.request_id, request=payload.request, data=answer
        )

    return request_builder, result_adapter


def build_aura_stream_output(
    request_id: str, request_data: AuraRequestData, request_output: RequestOutput
) -> Iterable[OutgoingMessage]:
    # A middle chunk of a chunked prefill samples a token the request discards.
    if request_data.req.inflight_middle_chunks > 0 or request_output.data is None:
        return []
    elif not request_data.stage_payload.request.params.get("stream", False):
        return []
    else:
        token_id = int(request_output.data)
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=torch.tensor([token_id], dtype=torch.long),
                target=DECODE_STAGE,
                metadata={"token_id": token_id},
            )
        ]


class AuraEngineBuilder(SGLangGenerationEngineBuilder[AuraRequestData]):
    model_name = "AURA"
    context_length = AURA_CONTEXT_LENGTH
    supports_context_length_override: ClassVar[bool] = True

    def __init__(self, *, max_new_tokens: int, request_build_max_workers: int) -> None:
        self.max_new_tokens = max_new_tokens
        self.request_build_max_workers = request_build_max_workers
        self.processor: ProcessorMixin | None = None
        self.generation_config: GenerationConfig | None = None
        self.mm_processor: BaseMultimodalProcessor | None = None

    def resolve_checkpoint(self, model_path: str) -> str:
        return resolve_checkpoint(model_path)

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        # note (siju): the checkpoint's auto_map names remote config and
        # processor modules, one of which it does not ship; the stock
        # Transformers Qwen3-VL classes load it as-is.
        self.processor = AutoProcessor.from_pretrained(checkpoint_dir)
        validate_silent_token(self.processor.tokenizer, checkpoint_dir)
        self.generation_config = GenerationConfig.from_pretrained(checkpoint_dir)

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        return {
            "max_running_requests": AURA_MAX_RUNNING_REQUESTS,
            "dtype": dtype,
            "trust_remote_code": False,
            "disable_overlap_schedule": True,
            "enable_torch_compile": False,
            "sampling_backend": "pytorch",
        }

    def adjust_overrides(self, overrides: dict[str, object]) -> None:
        if not current_platform.enable_thinker_decode_graph():
            overrides.setdefault("disable_decode_cuda_graph", True)
        else:
            pass

    def setup_model_resources(
        self,
        model: Qwen3VLForConditionalGeneration,
        server_args: ServerArgs,
        *,
        generation_cuda_graph_enabled: bool,
    ) -> None:
        # SGLang leaves one intra-op thread after loading weights, and the video
        # frames this stage's request builders resize are CPU tensors.
        torch.set_num_threads(
            min(AURA_PREPROCESSING_INTRAOP_THREADS, effective_cpu_count())
        )
        init_mm_embedding_cache(envs.SGLANG_VLM_CACHE_SIZE_MB.get() * 1024 * 1024)
        import_processors("sglang.srt.multimodal.processors")
        self.mm_processor = get_mm_processor(
            model.config, server_args, self.processor, "default", skip_mm_pool=True
        )

    def make_model_runner(
        self, model_worker: ModelWorker, output_proc: SGLangOutputProcessor
    ) -> ModelRunner[AuraRequestData]:
        return ModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Qwen3VLForConditionalGeneration) -> tuple[
        Callable[[StagePayload], AuraRequestData],
        Callable[[AuraRequestData], StagePayload],
    ]:
        tokenizer = self.processor.tokenizer
        generation_config = self.generation_config
        eos_token_ids = generation_config.eos_token_id
        stop_token_ids = sorted(
            set(eos_token_ids) if isinstance(eos_token_ids, list) else {eos_token_ids}
        )
        return make_aura_scheduler_adapters(
            tokenizer=tokenizer,
            mm_processor=self.mm_processor,
            model=model,
            device=torch.device(self.device),
            sampling_defaults={
                "temperature": float(
                    generation_default(generation_config.temperature, 1.0)
                ),
                "top_p": float(generation_default(generation_config.top_p, 1.0)),
                "top_k": int(generation_default(generation_config.top_k, -1)),
                "repetition_penalty": float(
                    generation_default(generation_config.repetition_penalty, 1.0)
                ),
            },
            stop_token_ids=stop_token_ids,
            max_new_tokens=self.max_new_tokens,
            context_length=self.context_length,
        )

    def extra_scheduler_kwargs(self) -> SchedulerExtras[AuraRequestData]:
        return {
            "stream_output_builder": build_aura_stream_output,
            "request_build_max_workers": self.request_build_max_workers,
        }
