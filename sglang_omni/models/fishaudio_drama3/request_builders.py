# SPDX-License-Identifier: Apache-2.0
"""Drama 3 generation requests: prompt frame rows in, sampled frames out."""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass, field

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.fishaudio_drama3.hf_config import FishOmniConfig
from sglang_omni.models.fishaudio_drama3.stream_windows import (
    frames_in_complete_windows,
)
from sglang_omni.models.fishaudio_drama3.text_config import checked_int
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import RequestOutput

# The tower prefill pairs every row with the next one, so a prompt needs two.
MINIMUM_PROMPT_ROWS = 2
# Leads every error a caller's request causes; the serving layer maps it to 400.
REQUEST_ERROR_PREFIX = "Drama 3 request:"
# Batch sampling info stores top_k as int32.
MAXIMUM_TOP_K = torch.iinfo(torch.int32).max
VOCODER_STAGE = "vocoder"


@dataclass(frozen=True, kw_only=True)
class Drama3GenerationInput:
    """One request: prompt frame rows [rows, 1 + num_codebooks] and sampling knobs.

    max_new_tokens None lets the engine fill the remaining context, up to its cap.
    """

    prompt_rows: torch.Tensor
    max_new_tokens: int | None
    temperature: float
    top_k: int
    top_p: float


@dataclass(frozen=True, kw_only=True)
class Drama3GenerationResult:
    """Sampled frames [frames, 1 + num_codebooks], without the end-of-speech frame."""

    frames: torch.Tensor
    prompt_row_count: int
    finish_reason: str | None


@dataclass
class Drama3RequestData(SGLangARRequestData):
    """frame_rows holds the prompt rows and then every sampled frame, eos included."""

    frame_rows: list[list[int]] = field(default_factory=list)
    prompt_row_count: int = 0
    streamed_frame_count: int = 0


class Drama3RequestBuilder:
    """Validates an Drama3GenerationInput payload and builds its SGLang request."""

    def __init__(
        self,
        *,
        config: FishOmniConfig,
        context_length: int,
        max_prompt_rows: int,
        max_new_tokens_cap: int,
    ) -> None:
        self.vocab_size = config.text_config.vocab_size
        self.codebook_size = config.audio_decoder_config.vocab_size
        self.frame_width = 1 + config.audio_decoder_config.num_codebooks
        self.eos_token_id = config.eos_token_id
        self.context_length = context_length
        self.max_prompt_rows = checked_int(
            max_prompt_rows, "max_prompt_rows", MINIMUM_PROMPT_ROWS
        )
        self.max_new_tokens_cap = checked_int(
            max_new_tokens_cap, "max_new_tokens_cap", 1
        )

    def __call__(self, payload: StagePayload) -> Drama3RequestData:
        generation_input = payload.data
        if not isinstance(generation_input, Drama3GenerationInput):
            raise TypeError(
                f"Drama 3 requests carry Drama3GenerationInput; got "
                f"{type(generation_input).__name__}"
            )
        else:
            pass
        prompt_rows = generation_input.prompt_rows
        if (
            prompt_rows.dtype != torch.int64
            or prompt_rows.dim() != 2
            or prompt_rows.shape[1] != self.frame_width
        ):
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} prompt_rows must be int64 "
                f"[rows, {self.frame_width}]; got {prompt_rows.dtype} "
                f"{tuple(prompt_rows.shape)}"
            )
        else:
            pass
        prompt_row_count = prompt_rows.shape[0]
        remaining_rows = self.context_length - prompt_row_count
        requested_max_new_tokens = generation_input.max_new_tokens
        if prompt_row_count < MINIMUM_PROMPT_ROWS:
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} a prompt needs at least "
                f"{MINIMUM_PROMPT_ROWS} rows; got {prompt_row_count}"
            )
        elif prompt_row_count > self.max_prompt_rows:
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} a prompt of {prompt_row_count} rows exceeds "
                f"the maximum allowed length of {self.max_prompt_rows} rows"
            )
        elif not bool(
            ((prompt_rows[:, 0] >= 0) & (prompt_rows[:, 0] < self.vocab_size)).all()
        ):
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} prompt main tokens must lie in "
                f"[0, {self.vocab_size})"
            )
        elif not bool(
            (
                (prompt_rows[:, 1:] >= 0) & (prompt_rows[:, 1:] < self.codebook_size)
            ).all()
        ):
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} prompt codes must lie in "
                f"[0, {self.codebook_size})"
            )
        elif requested_max_new_tokens is None and remaining_rows < 1:
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} a prompt of {prompt_row_count} rows leaves "
                f"no room for new frames in the context length {self.context_length}"
            )
        elif requested_max_new_tokens is not None and requested_max_new_tokens < 1:
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} max_new_tokens must be >= 1; got "
                f"{requested_max_new_tokens}"
            )
        elif (
            requested_max_new_tokens is not None
            and requested_max_new_tokens > remaining_rows
        ):
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} {prompt_row_count} prompt rows plus "
                f"max_new_tokens={requested_max_new_tokens} exceed the context "
                f"length {self.context_length}"
            )
        elif (
            requested_max_new_tokens is not None
            and requested_max_new_tokens > self.max_new_tokens_cap
        ):
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} max_new_tokens must be at most "
                f"{self.max_new_tokens_cap}; got {requested_max_new_tokens}"
            )
        elif not (
            math.isfinite(generation_input.temperature)
            and generation_input.temperature >= 0.0
        ):
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} temperature must be finite and >= 0; got "
                f"{generation_input.temperature}"
            )
        elif not 1 <= generation_input.top_k <= MAXIMUM_TOP_K:
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} top_k must be in [1, {MAXIMUM_TOP_K}]; got "
                f"{generation_input.top_k}"
            )
        elif not 0.0 < generation_input.top_p <= 1.0:
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} top_p must be in (0, 1]; got "
                f"{generation_input.top_p}"
            )
        else:
            pass
        max_new_tokens = (
            min(self.max_new_tokens_cap, remaining_rows)
            if requested_max_new_tokens is None
            else requested_max_new_tokens
        )
        sampling_params = SamplingParams(
            max_new_tokens=max_new_tokens,
            temperature=generation_input.temperature,
            top_p=generation_input.top_p,
            top_k=generation_input.top_k,
            stop_token_ids=[self.eos_token_id],
        )
        sampling_params.normalize(None)
        sampling_params.verify(self.vocab_size)
        frame_rows = prompt_rows.tolist()
        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=array("q", [frame_row[0] for frame_row in frame_rows]),
            sampling_params=sampling_params,
            vocab_size=self.vocab_size,
            eos_token_ids={self.eos_token_id},
        )
        return Drama3RequestData(
            req=req,
            input_ids=prompt_rows[:, 0].clone(),
            stage_payload=payload,
            max_new_tokens=max_new_tokens,
            temperature=generation_input.temperature,
            top_k=generation_input.top_k,
            top_p=generation_input.top_p,
            frame_rows=frame_rows,
            prompt_row_count=len(frame_rows),
        )


class Drama3ResultAdapter:
    """Puts a finished request's sampled frames into its stage payload."""

    def __init__(self, *, eos_token_id: int) -> None:
        self.eos_token_id = eos_token_id

    def __call__(self, request_data: Drama3RequestData) -> StagePayload:
        payload = request_data.stage_payload
        assert payload is not None
        sampled_rows = request_data.frame_rows[request_data.prompt_row_count :]
        if sampled_rows and sampled_rows[-1][0] == self.eos_token_id:
            sampled_rows = sampled_rows[:-1]
        else:
            pass
        frame_width = len(request_data.frame_rows[0])
        payload.data = Drama3GenerationResult(
            frames=torch.tensor(sampled_rows, dtype=torch.int64, device="cpu").reshape(
                len(sampled_rows), frame_width
            ),
            prompt_row_count=request_data.prompt_row_count,
            finish_reason=request_data.finish_reason,
        )
        return payload


class Drama3FrameStreamer:
    def __init__(
        self,
        *,
        eos_token_id: int,
        stream_stride: int,
        stream_followup_stride_ramp: tuple[int, ...],
        stream_followup_stride: int,
    ) -> None:
        self.eos_token_id = eos_token_id
        self.stream_stride = stream_stride
        self.stream_followup_stride_ramp = stream_followup_stride_ramp
        self.stream_followup_stride = stream_followup_stride

    def __call__(
        self,
        request_id: str,
        request_data: Drama3RequestData,
        request_output: RequestOutput,
        /,
    ) -> list[OutgoingMessage]:
        """The scheduler hands over request_output by contract; frames come from
        the request's history."""
        payload = request_data.stage_payload
        assert payload is not None
        if not payload.request.params.get("stream"):
            return []
        elif request_data.frame_rows[-1][0] == self.eos_token_id:
            return []
        else:
            pass
        sampled_frame_count = (
            len(request_data.frame_rows) - request_data.prompt_row_count
        )
        if sampled_frame_count <= self.stream_stride:
            window_end = sampled_frame_count
        else:
            window_end = frames_in_complete_windows(
                sampled_frame_count,
                stream_stride=self.stream_stride,
                stream_followup_stride_ramp=self.stream_followup_stride_ramp,
                stream_followup_stride=self.stream_followup_stride,
            )
        if window_end <= request_data.streamed_frame_count:
            return []
        else:
            window_start_row = (
                request_data.prompt_row_count + request_data.streamed_frame_count
            )
            window_end_row = request_data.prompt_row_count + window_end
            window_rows = request_data.frame_rows[window_start_row:window_end_row]
            request_data.streamed_frame_count = window_end
            return [
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data=torch.tensor(window_rows, dtype=torch.int64, device="cpu").T,
                    target=VOCODER_STAGE,
                    metadata={"modality": "audio_codes"},
                )
            ]
