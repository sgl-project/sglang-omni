# SPDX-License-Identifier: Apache-2.0
"""Breeze StagePayload / SGLang adapters and request-owned generation state."""

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

from .sampling import SamplingConfig

CFG_SUFFIX = "::breeze-uncond"


@dataclass
class BreezeGenerationState:
    sampling: SamplingConfig
    generator: torch.Generator
    started: float = field(default_factory=time.perf_counter)
    history: list[int] = field(default_factory=list)
    codes: list[torch.Tensor] = field(default_factory=list)
    feedback: torch.Tensor | None = None
    pending_chunk: torch.Tensor | None = None


@dataclass
class BreezeRequestData(SGLangARRequestData):
    enforce_request_limits: bool = True
    generation: BreezeGenerationState | None = None
    cfg_uncond: "BreezeRequestData | None" = None
    is_cfg_uncond: bool = False


def build_request(payload: StagePayload, model) -> BreezeRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    if payload.request_id.endswith(CFG_SUFFIX):
        raise ValueError(f"Breeze request IDs must not end in {CFG_SUFFIX!r}")
    prepared = payload.data
    settings = SamplingConfig(**prepared["sampling"])
    device = model.lm_head.weight.device
    dtype = model.model.norm.weight.dtype
    generation = BreezeGenerationState(
        sampling=settings,
        generator=torch.Generator(device=device).manual_seed(settings.seed),
    )
    eos_id = model.config.audio_vocab_size

    def branch(suffix: str, embeds: torch.Tensor) -> BreezeRequestData:
        embeds = embeds.to(device=device, dtype=dtype)
        ids = [0] * len(embeds)
        sampling = SamplingParams(
            max_new_tokens=settings.max_new_tokens,
            temperature=0,
            stop_token_ids=[eos_id],
        )
        sampling.normalize(None)
        sampling.verify(model.config.vocab_size)
        req = Req(
            rid=payload.request_id + suffix,
            origin_input_text="",
            origin_input_ids=ids,
            sampling_params=sampling,
            eos_token_ids={eos_id},
            vocab_size=model.config.vocab_size,
        )
        req.tokenizer = None
        req._input_embeds_are_projected = True
        return BreezeRequestData(
            req=req,
            stage_payload=payload,
            input_ids=torch.tensor(ids, dtype=torch.long),
            output_ids=req.output_ids,
            max_new_tokens=settings.max_new_tokens,
            input_embeds_are_projected=True,
            prefill_input_embeds=embeds,
            generation=generation,
            is_cfg_uncond=bool(suffix),
        )

    cond = branch("", prepared["prompt_embeds"])
    cond.cfg_uncond = branch(CFG_SUFFIX, prepared["negative_embeds"])
    # RequestData now owns the tensors, not the relay payload. No process-global
    # prepared-request cache is needed, including on late preprocessing abort.
    payload.data = {}
    return cond


def stream_output(request_id: str, data: BreezeRequestData, req_output: Any):
    del req_output
    if data.is_cfg_uncond:
        return
    generation = data.generation
    chunk = generation.pending_chunk
    generation.pending_chunk = None
    if not (data.stage_payload.request.params or {}).get("stream", False):
        return
    if chunk is not None:
        yield OutgoingMessage(
            request_id=request_id,
            type="stream",
            data=chunk.unsqueeze(0),
            target="vocoder",
            metadata={
                "stream": True,
                "modality": "audio_codes",
                "num_quantizers": chunk.numel(),
            },
        )


def apply_result(data: BreezeRequestData) -> StagePayload:
    generation = data.generation
    if not generation.codes:
        raise ValueError("Breeze-TTS-2 generated no audio frames")
    # Matches the Qwen streaming-vocoder wire contract because Breeze bundles
    # that codec. Do not prepend reference codes: upstream decode starts fresh.
    data.stage_payload.data = {
        "audio_codes": torch.stack(generation.codes).cpu(),
        "sample_rate": 24000,
        "ref_code_len": 0,
        "prompt_tokens": len(data.req.origin_input_ids),
        "completion_tokens": len(generation.codes),
        "engine_time_s": time.perf_counter() - generation.started,
    }
    return data.stage_payload
