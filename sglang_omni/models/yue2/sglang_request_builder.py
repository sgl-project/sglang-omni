# SPDX-License-Identifier: Apache-2.0
"""YuE2 request adapters between StagePayload and SGLang requests."""

from dataclasses import dataclass, field
from typing import Literal

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.proto.request import StagePayload
from sglang_omni.sampling.seed import derive_sampling_seed
from sglang_omni.scheduling.pipeline_state import store_state
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData

from .payload_types import Yue2State
from .request import (
    ABC_END,
    MUSIC_END,
    SamplingConfig,
    Yue2Request,
    Yue2Tokenizer,
    build_request,
)

SCORE_SUFFIX = "-abc"
GUIDANCE_SUFFIX = "-cfg"
SAMPLING_NAMESPACE = "yue2-ar"


def score_request_id(request_id: str) -> str:
    return f"{request_id}{SCORE_SUFFIX}"


def guidance_request_id(request_id: str) -> str:
    return f"{request_id}{GUIDANCE_SUFFIX}"


def public_request_id(request_id: str) -> str:
    """The client-visible id behind an internal score or guidance request id."""
    for suffix in (SCORE_SUFFIX, GUIDANCE_SUFFIX):
        if request_id.endswith(suffix):
            return request_id.removesuffix(suffix)
        else:
            pass
    return request_id


@dataclass
class Yue2ARRequestData(SGLangARRequestData):
    request: Yue2Request | None = None
    phase: Literal["abc", "semantic"] = "semantic"
    sampling: SamplingConfig | None = None
    legacy_off: bool = False
    prefix: list[int] = field(default_factory=list)
    abc_truncated: bool = False
    cfg_scale: float = 1.0
    cfg_uncond: "Yue2ARRequestData | None" = None
    is_cfg_uncond: bool = False
    sampling_seed: int = 0
    generated_tokens: int = 0
    recent_tokens: torch.Tensor | None = None


class Yue2RequestBuilder:
    """Build one SGLang request per generation phase; the score phase comes first."""

    def __init__(
        self, tokenizer: Yue2Tokenizer, context_tokens: int, vocab_size: int
    ) -> None:
        self.tokenizer = tokenizer
        self.context_tokens = context_tokens
        self.vocab_size = vocab_size

    def __call__(self, payload: StagePayload) -> Yue2ARRequestData:
        if payload.request_id.endswith((SCORE_SUFFIX, GUIDANCE_SUFFIX)):
            raise ValueError(
                f"YuE2 request: ids ending in {SCORE_SUFFIX} or {GUIDANCE_SUFFIX} "
                "are reserved for internal requests"
            )
        else:
            pass
        try:
            request = build_request(payload)
        except ValueError as error:
            raise ValueError(f"YuE2 request: {error}") from error
        options = request.options
        if options.cot == "off":
            return self.semantic(payload, request, [], abc_truncated=False)
        elif options.abc is not None:
            return self.semantic(
                payload,
                request,
                self.tokenizer.encode(options.abc),
                abc_truncated=False,
            )
        else:
            sampling = request.planning_sampling
            prefix = self.tokenizer.prefix(request, None)
            return Yue2ARRequestData(
                request=request,
                phase="abc",
                sampling=sampling,
                prefix=prefix,
                sampling_seed=derive_sampling_seed(
                    SAMPLING_NAMESPACE, request.seed, "abc"
                ),
                stage_payload=payload,
                **self.sglang_fields(
                    score_request_id(payload.request_id), prefix, sampling, ABC_END
                ),
            )

    def semantic(
        self,
        payload: StagePayload,
        request: Yue2Request,
        abc_tokens: list[int],
        *,
        abc_truncated: bool,
    ) -> Yue2ARRequestData:
        sampling = request.semantic_sampling
        prefix = self.tokenizer.prefix(request, abc_tokens)
        seed = derive_sampling_seed(SAMPLING_NAMESPACE, request.seed, "semantic")
        legacy_off = request.options.cot == "off"
        if request.guidance != 1:
            negative = self.tokenizer.negative_prefix(request, abc_tokens)
            unconditional = Yue2ARRequestData(
                request=request,
                sampling=sampling,
                legacy_off=legacy_off,
                prefix=negative,
                is_cfg_uncond=True,
                stage_payload=payload,
                **self.sglang_fields(
                    guidance_request_id(payload.request_id),
                    negative,
                    sampling,
                    MUSIC_END,
                ),
            )
        else:
            unconditional = None
        return Yue2ARRequestData(
            request=request,
            sampling=sampling,
            legacy_off=legacy_off,
            prefix=prefix,
            abc_truncated=abc_truncated,
            cfg_scale=request.guidance,
            cfg_uncond=unconditional,
            sampling_seed=seed,
            stage_payload=payload,
            **self.sglang_fields(payload.request_id, prefix, sampling, MUSIC_END),
        )

    def sglang_fields(
        self, request_id: str, prefix: list[int], sampling: SamplingConfig, end: int
    ) -> dict[str, Req | list[int] | torch.Tensor | int]:
        if len(prefix) + sampling.max_tokens >= self.context_tokens:
            raise ValueError(
                "YuE2 request: prefix and token budget exceed the model context"
            )
        else:
            pass
        parameters = SamplingParams(
            max_new_tokens=sampling.max_tokens,
            temperature=1.0,
            stop_token_ids=[end],
        )
        parameters.normalize(None)
        parameters.verify(self.vocab_size)
        req = Req(
            rid=request_id,
            origin_input_text="",
            origin_input_ids=list(prefix),
            sampling_params=parameters,
            eos_token_ids={end},
            vocab_size=self.vocab_size,
        )
        req.tokenizer = None
        return {
            "req": req,
            "output_ids": req.output_ids,
            "input_ids": torch.tensor([prefix], dtype=torch.long),
            "max_new_tokens": sampling.max_tokens,
        }

    def result(self, data: Yue2ARRequestData) -> StagePayload:
        assert data.request is not None and data.stage_payload is not None
        tokens = list(data.req.output_ids)
        finished = bool(tokens) and tokens[-1] == MUSIC_END
        semantic_tokens = tokens[:-1] if finished else tokens
        state = Yue2State(
            semantic_prefix=data.prefix,
            semantic_tokens=semantic_tokens,
            seed=data.request.seed,
            ode_steps=data.request.options.ode_steps,
            abc_truncated=data.abc_truncated,
            semantic_truncated=not finished,
            prompt_tokens=len(data.prefix),
            completion_tokens=len(semantic_tokens),
            finish_reason="length" if data.abc_truncated or not finished else "stop",
        )
        return store_state(data.stage_payload, state)
