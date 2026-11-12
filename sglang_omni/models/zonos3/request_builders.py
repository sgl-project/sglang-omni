# SPDX-License-Identifier: Apache-2.0
"""Zonos3 generation requests: prompt frame rows in, undelayed codes out."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.zonos3.components.codec import (
    BOUNDARY_HOLDBACK_FRAMES,
    SAMPLE_RATE_HZ,
)
from sglang_omni.models.zonos3.hf_config import Zonos3Config
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.streaming_vocoder import resolve_initial_codec_chunk_frames
from sglang_omni.scheduling.types import RequestOutput

# The scheduler only sees one token per frame: continue, or stop after the last row.
CONTINUE_TOKEN_ID = 0
STOP_TOKEN_ID = 1
FRAME_TOKEN_VOCAB_SIZE = 2
REQUEST_ERROR_PREFIX = "Zonos3 request:"
# The repetition penalty reaches back this many generated rows on the leading codebooks.
REPETITION_WINDOW_ROWS = 96
REPETITION_CODEBOOKS = 6
# Sampling seeds are int64 Philox keys below 2 ** SEED_BITS.
SEED_BITS = 63


@dataclass(frozen=True, kw_only=True)
class Zonos3GenerationInput:
    """Prompt frame rows [rows, frame_width], one all-pad row per condition table, and the sampling knobs.

    The prompt may end with codec-silence rows that open the audio stream; generated row g carries codebook c of spoken frame g - c.
    """

    prompt_rows: torch.Tensor
    condition_row_index: int
    condition_buckets: tuple[int, ...]
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    seed: int | None


@dataclass(frozen=True, kw_only=True)
class Zonos3GenerationResult:
    """Undelayed codes [codebooks, frames] ready for the codec decoder."""

    codes: torch.Tensor
    prompt_row_count: int
    finish_reason: str | None


def result_metadata(
    result: Zonos3GenerationResult,
) -> dict[str, str | int | dict[str, int]]:
    """Response fields besides the audio: sample rate, usage and finish reason."""
    frame_count = result.codes.shape[1]
    response: dict[str, str | int | dict[str, int]] = {
        "modality": "audio",
        "sample_rate": SAMPLE_RATE_HZ,
        "usage": {
            "prompt_tokens": result.prompt_row_count,
            "completion_tokens": frame_count,
            "total_tokens": result.prompt_row_count + frame_count,
        },
    }
    if result.finish_reason is not None:
        response["finish_reason"] = result.finish_reason
    else:
        pass
    return response


@dataclass
class Zonos3RequestData(SGLangARRequestData):
    """generated_rows holds every sampled delayed row [codebooks] on the CPU; end_of_audio_row is the first spoken frame that ended."""

    generation_input: Zonos3GenerationInput | None = None
    generated_rows: list[torch.Tensor] = field(default_factory=list)
    end_of_audio_row: int | None = None
    streamed_frame_count: int = 0
    next_stream_frame_count: int | None = None


class Zonos3RequestBuilder:
    """Validates a Zonos3GenerationInput payload and builds its SGLang request."""

    def __init__(self, *, config: Zonos3Config, context_length: int) -> None:
        self.audio_vocab_size = config.audio_vocab_size
        self.context_length = context_length

    def __call__(self, payload: StagePayload) -> Zonos3RequestData:
        generation_input = payload.data
        assert isinstance(generation_input, Zonos3GenerationInput)
        prompt_row_count = generation_input.prompt_rows.shape[0]
        if generation_input.max_new_tokens < 1:
            raise ValueError(f"{REQUEST_ERROR_PREFIX} max_new_tokens must be >= 1")
        elif prompt_row_count + generation_input.max_new_tokens > self.context_length:
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} {prompt_row_count} prompt rows plus "
                f"max_new_tokens={generation_input.max_new_tokens} exceed the context "
                f"length {self.context_length}"
            )
        elif generation_input.temperature < 0.0:
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} temperature must be >= 0 (0 decodes greedily)"
            )
        elif not 1 <= generation_input.top_k <= self.audio_vocab_size:
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} top_k must be in [1, {self.audio_vocab_size}]"
            )
        elif not 0.0 < generation_input.top_p <= 1.0:
            raise ValueError(f"{REQUEST_ERROR_PREFIX} top_p must be in (0, 1]")
        elif generation_input.repetition_penalty < 1.0:
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} repetition_penalty must be >= 1 (1 disables it)"
            )
        elif generation_input.seed is not None and not (
            0 <= generation_input.seed < 2**SEED_BITS
        ):
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} seed must be in [0, 2**{SEED_BITS})"
            )
        else:
            pass
        sampling_params = SamplingParams(
            max_new_tokens=generation_input.max_new_tokens,
            stop_token_ids=[STOP_TOKEN_ID],
        )
        sampling_params.normalize(None)
        sampling_params.verify(FRAME_TOKEN_VOCAB_SIZE)
        sglang_request = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=[CONTINUE_TOKEN_ID] * prompt_row_count,
            sampling_params=sampling_params,
            vocab_size=FRAME_TOKEN_VOCAB_SIZE,
            eos_token_ids={STOP_TOKEN_ID},
        )
        return Zonos3RequestData(
            req=sglang_request,
            input_ids=torch.full((prompt_row_count,), CONTINUE_TOKEN_ID),
            stage_payload=payload,
            max_new_tokens=generation_input.max_new_tokens,
            temperature=generation_input.temperature,
            generation_input=generation_input,
        )


def complete_frame_count(
    row_count: int, end_of_audio_row: int | None, codebook_count: int
) -> int:
    """Frames whose every codebook has been generated, excluding end-of-audio and later frames."""
    frame_count = max(row_count - codebook_count + 1, 0)
    if end_of_audio_row is None:
        return frame_count
    else:
        return min(end_of_audio_row, frame_count)


def undelay_rows(
    generated_rows: torch.Tensor, end_of_audio_row: int | None, codebook_size: int
) -> torch.Tensor:
    """Delayed rows [rows, codebooks] to codes [codebooks, frames]; codebook c lags c rows."""
    codebook_count = generated_rows.shape[1]
    frame_count = complete_frame_count(
        generated_rows.shape[0], end_of_audio_row, codebook_count
    )
    codes = torch.stack(
        [
            generated_rows[
                codebook_index : codebook_index + frame_count, codebook_index
            ]
            for codebook_index in range(codebook_count)
        ]
    )
    return codes.clamp(max=codebook_size - 1)


class Zonos3ResultAdapter:
    """Puts a finished request's undelayed codes into its stage payload."""

    def __init__(self, *, codebook_size: int, codebook_count: int) -> None:
        self.codebook_size = codebook_size
        self.codebook_count = codebook_count

    def __call__(self, request_data: Zonos3RequestData) -> StagePayload:
        payload = request_data.stage_payload
        generation_input = request_data.generation_input
        assert payload is not None and generation_input is not None
        if request_data.generated_rows:
            generated_rows = torch.stack(request_data.generated_rows)
        else:
            generated_rows = torch.empty((0, self.codebook_count), dtype=torch.int64)
        payload.data = Zonos3GenerationResult(
            codes=undelay_rows(
                generated_rows, request_data.end_of_audio_row, self.codebook_size
            ),
            prompt_row_count=generation_input.prompt_rows.shape[0],
            finish_reason=request_data.finish_reason,
        )
        return payload


class Zonos3StreamOutputBuilder:
    """Sends a streaming request's complete frames when they complete the vocoder's next chunk, and the rest at finish.

    Sends land on the frame counts where the vocoder's chunks become ready, so it decodes no later than with a send per frame.
    """

    def __init__(
        self,
        *,
        codebook_size: int,
        codebook_count: int,
        vocoder_initial_chunk_frames: int,
        vocoder_steady_chunk_frames: int,
    ) -> None:
        self.codebook_size = codebook_size
        self.codebook_count = codebook_count
        self.vocoder_initial_chunk_frames = vocoder_initial_chunk_frames
        self.vocoder_steady_chunk_frames = vocoder_steady_chunk_frames

    def __call__(
        self,
        request_id: str,
        request_data: Zonos3RequestData,
        request_output: RequestOutput,
    ) -> list[OutgoingMessage]:
        """request_output is the step's frame token; the step's row is already in request_data."""
        return self.stream_messages(request_id, request_data, is_final=False)

    def flush(
        self, request_id: str, request_data: Zonos3RequestData
    ) -> list[OutgoingMessage]:
        return self.stream_messages(request_id, request_data, is_final=True)

    def stream_messages(
        self, request_id: str, request_data: Zonos3RequestData, *, is_final: bool
    ) -> list[OutgoingMessage]:
        payload = request_data.stage_payload
        assert payload is not None
        request_params = payload.request.params
        if not request_params.get("stream", False):
            return []
        elif request_data.next_stream_frame_count is None:
            # The vocoder latches its first chunk from the same request params.
            request_data.next_stream_frame_count = (
                resolve_initial_codec_chunk_frames(
                    request_params,
                    steady_chunk_frames=self.vocoder_steady_chunk_frames,
                    default_frames=self.vocoder_initial_chunk_frames,
                )
                or self.vocoder_steady_chunk_frames
            ) + BOUNDARY_HOLDBACK_FRAMES
        else:
            pass
        streamed_frame_count = request_data.streamed_frame_count
        frame_count = complete_frame_count(
            len(request_data.generated_rows),
            request_data.end_of_audio_row,
            self.codebook_count,
        )
        if frame_count <= streamed_frame_count:
            return []
        elif frame_count < request_data.next_stream_frame_count and not is_final:
            return []
        else:
            pass
        while request_data.next_stream_frame_count <= frame_count:
            request_data.next_stream_frame_count += self.vocoder_steady_chunk_frames
        # Rows from the first new frame through the last codebook of the last complete frame.
        new_rows = request_data.generated_rows[
            streamed_frame_count : frame_count + self.codebook_count - 1
        ]
        codes = undelay_rows(torch.stack(new_rows), None, self.codebook_size)
        request_data.streamed_frame_count = frame_count
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=codes,
                metadata={
                    "initial_codec_chunk_frames": request_params.get(
                        "initial_codec_chunk_frames"
                    ),
                    "modality": "audio_codes",
                    "stream": True,
                },
            )
        ]
