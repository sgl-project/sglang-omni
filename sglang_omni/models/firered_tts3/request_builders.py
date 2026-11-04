# SPDX-License-Identifier: Apache-2.0
"""FireRedTTS3 request construction, retained history, and stream output."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.firered_tts3.payload_types import (
    ENGINE_TOKEN_SLACK,
    TEXT_PHASE_TASKS,
    FireRedTTS3State,
    load_firered_tts3_state,
    store_firered_tts3_state,
)
from sglang_omni.models.firered_tts3.profiling import ProfilingProbe
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import RequestOutput

TextDecodeMode = Literal["text", "finalize", "sosp", "audio_zero", "audio"]
DualStreamStage = Literal["text", "patch"]
TEXT_MODE_TEXT: TextDecodeMode = "text"
TEXT_MODE_FINALIZE: TextDecodeMode = "finalize"
TEXT_MODE_SOSP: TextDecodeMode = "sosp"
TEXT_MODE_AUDIO_ZERO: TextDecodeMode = "audio_zero"
TEXT_MODE_AUDIO: TextDecodeMode = "audio"
DS_STAGE_TEXT: DualStreamStage = "text"
DS_STAGE_PATCH: DualStreamStage = "patch"
# note (kermit): Latent-only steps still need an in-vocabulary token on the engine rail.
CONTROL_TOKEN_ID = 0
# note (kermit): Early small chunks reduce time to first audio.
STREAM_CHUNK_PATCH_RAMP = (1, 2, 4)


@dataclass(frozen=True, kw_only=True)
class FireRedFlowResume:
    """Generator state retained while request flow buffers are released."""

    rng_state: torch.Tensor | None


@dataclass(kw_only=True, eq=False)
class FlowState:
    """Ephemeral flow state rebuilt from retained history after retraction."""

    generator: torch.Generator
    prompt_patch_count: int
    speaker_conditioning: torch.Tensor | None
    backbone_conditioning: torch.Tensor
    latent_history: torch.Tensor
    patch_count: int


@dataclass(kw_only=True)
class FireRedTTS3SGLangRequestData(SGLangARRequestData):
    """Retained request history and the current ephemeral flow state."""

    req: Req = field()
    stage_payload: StagePayload = field()
    state: FireRedTTS3State = field(default_factory=FireRedTTS3State)
    prompt_patch_count: int = 0
    prefill_len: int = 0
    flow_state: FlowState | FireRedFlowResume | None = None
    latent_patches: list[torch.Tensor] = field(default_factory=list)
    probe: ProfilingProbe | None = None
    engine_start_s: float = 0.0
    streamed_patches: int = 0
    stream_chunk_index: int = 0
    text_ar_mode: TextDecodeMode = TEXT_MODE_AUDIO
    text_fed: int = 0
    text_token_ids: list[int] = field(default_factory=list)
    eot_sampled: bool = False
    dual_stream: bool = False
    interleave_ratio: tuple[int, int] | None = None
    ds_stage: DualStreamStage = DS_STAGE_TEXT
    ds_index: int = 0
    text_pos: int = 0
    stop_pending: bool = False


def build_sglang_firered_tts3_request(
    payload: StagePayload,
    *,
    patch_size: int,
    vocab_size: int,
    interleave_ratio: tuple[int, int] | None = None,
) -> FireRedTTS3SGLangRequestData:
    """Build token inputs, sampling parameters, and retained request state."""
    state = load_firered_tts3_state(payload)
    if not state.input_ids:
        raise RuntimeError("FireRedTTS3 preprocessing did not produce input_ids")
    else:
        pass
    prompt_frame_count = (
        0 if state.prompt_latents is None else int(state.prompt_latents.shape[1])
    )
    if prompt_frame_count % patch_size:
        raise ValueError(
            f"FireRedTTS3 prompt latents ({prompt_frame_count} frames) are not aligned to the patch size ({patch_size})"
        )
    else:
        pass
    prompt_patch_count = prompt_frame_count // patch_size
    if state.variant in ("instruct", "speaker"):
        origin_input_ids = list(state.input_ids)
        prefill_token_count = len(origin_input_ids)
    else:
        prefill_token_count = 1 + len(state.input_ids) + prompt_patch_count
        origin_input_ids = [CONTROL_TOKEN_ID] * prefill_token_count
    if state.dual_stream:
        if interleave_ratio is None:
            raise ValueError(
                "FireRedTTS3 dual-stream requests require a checkpoint with text_latent_interleave_ratio"
            )
        else:
            pass
        if not state.text_stream_ids or state.text_pad_id < 0:
            raise ValueError(
                "FireRedTTS3 dual-stream requests require text_stream_ids and text_pad_id from preprocessing"
            )
        else:
            pass
    else:
        pass
    max_engine_tokens = (
        state.max_engine_tokens or state.max_patches + ENGINE_TOKEN_SLACK
    )
    sampling_params = SamplingParams(
        max_new_tokens=max_engine_tokens,
        temperature=(
            state.text_temperature if state.text_temperature is not None else 0.0
        ),
        stop_token_ids=[],
    )
    if state.text_top_p is not None:
        sampling_params.top_p = state.text_top_p
    else:
        pass
    if state.text_top_k is not None:
        sampling_params.top_k = state.text_top_k
    else:
        pass
    if state.seed is not None and sampling_params.temperature > 0:
        sampling_params.sampling_seed = int(state.seed)
    else:
        pass
    sampling_params.normalize(None)
    sampling_params.verify(int(vocab_size))
    request = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=origin_input_ids,
        sampling_params=sampling_params,
        eos_token_ids=set(),
        vocab_size=int(vocab_size),
    )
    request.tokenizer = None
    request._input_embeds_are_projected = (
        True  # noqa: SLF001  # leading-underscore: SGLang request attribute
    )
    request._codec_suppress_tokens = (
        None  # noqa: SLF001  # leading-underscore: SGLang request attribute
    )
    return FireRedTTS3SGLangRequestData(
        state=state,
        stage_payload=payload,
        req=request,
        output_ids=request.output_ids,
        prompt_patch_count=prompt_patch_count,
        prefill_len=prefill_token_count,
        max_new_tokens=max_engine_tokens,
        input_embeds_are_projected=True,
        engine_start_s=time.perf_counter(),
        text_ar_mode="text" if state.task_type in TEXT_PHASE_TASKS else "audio",
        dual_stream=state.dual_stream,
        interleave_ratio=interleave_ratio if state.dual_stream else None,
    )


def apply_latent_result(
    request_data: FireRedTTS3SGLangRequestData,
    text_tokenizer: PreTrainedTokenizerBase | None = None,
) -> StagePayload:
    """Store generated latents and text with request usage."""
    state = request_data.state
    if not request_data.latent_patches:
        raise RuntimeError("FireRedTTS3 generated no latent patches")
    else:
        pass
    state.generated_latents = torch.cat(request_data.latent_patches, dim=1)
    state.prompt_tokens = request_data.prefill_len
    state.completion_tokens = len(request_data.latent_patches)
    state.engine_time_s = time.perf_counter() - request_data.engine_start_s
    state.finish_reason = request_data.finish_reason
    if request_data.text_token_ids and text_tokenizer is not None:
        state.gen_text = text_tokenizer.decode(request_data.text_token_ids)
    else:
        pass
    return store_firered_tts3_state(request_data.stage_payload, state)


def emit_latent_chunk(
    request_id: str, request_data: FireRedTTS3SGLangRequestData, patch_count: int
) -> OutgoingMessage:
    """Publish generated patches with their producer readiness event."""
    patches = request_data.latent_patches[
        request_data.streamed_patches : request_data.streamed_patches + patch_count
    ]
    latent_chunk = patches[0] if len(patches) == 1 else torch.cat(patches, dim=1)
    request_data.streamed_patches += len(patches)
    request_data.stream_chunk_index += 1
    metadata: dict[str, str | bool | torch.cuda.Event] = {
        "modality": "audio_latents",
        "stream": True,
    }
    if latent_chunk.device.type == "cuda":
        codes_ready_event = torch.cuda.Event()
        codes_ready_event.record()
        metadata["codes_ready_event"] = codes_ready_event
    else:
        pass
    return OutgoingMessage(
        request_id=request_id,
        type="stream",
        target="vocoder",
        data=latent_chunk,
        metadata=metadata,
    )


def build_stream_output(
    request_id: str,
    request_data: FireRedTTS3SGLangRequestData,
    req_output: RequestOutput,
) -> list[OutgoingMessage]:
    """Emit one dual-stream patch or the next 1/2/4-patch chunk."""
    # note (kermit): req_output is required by the shared stream-builder callback.
    if not request_data.state.stream:
        return []
    else:
        pass
    if request_data.dual_stream:
        pending_patch_count = (
            len(request_data.latent_patches) - request_data.streamed_patches
        )
        if pending_patch_count < 1:
            return []
        else:
            pass
        return [emit_latent_chunk(request_id, request_data, 1)]
    else:
        pass
    chunk_patch_count = STREAM_CHUNK_PATCH_RAMP[
        min(request_data.stream_chunk_index, len(STREAM_CHUNK_PATCH_RAMP) - 1)
    ]
    pending_patch_count = (
        len(request_data.latent_patches) - request_data.streamed_patches
    )
    if pending_patch_count < chunk_patch_count:
        return []
    else:
        pass
    return [emit_latent_chunk(request_id, request_data, chunk_patch_count)]


def flush_stream_output(
    request_id: str, request_data: FireRedTTS3SGLangRequestData
) -> list[OutgoingMessage]:
    """Drain the patches held back by the streaming chunk ramp."""
    if not request_data.state.stream:
        return []
    else:
        pass
    pending_patch_count = (
        len(request_data.latent_patches) - request_data.streamed_patches
    )
    if pending_patch_count <= 0:
        return []
    else:
        pass
    return [emit_latent_chunk(request_id, request_data, pending_patch_count)]


build_stream_output.flush = flush_stream_output  # type: ignore[attr-defined]
__all__ = [
    "CONTROL_TOKEN_ID",
    "STREAM_CHUNK_PATCH_RAMP",
    "DS_STAGE_PATCH",
    "DS_STAGE_TEXT",
    "TEXT_MODE_TEXT",
    "TEXT_MODE_FINALIZE",
    "TEXT_MODE_SOSP",
    "TEXT_MODE_AUDIO_ZERO",
    "TEXT_MODE_AUDIO",
    "TextDecodeMode",
    "DualStreamStage",
    "FlowState",
    "FireRedFlowResume",
    "FireRedTTS3SGLangRequestData",
    "apply_latent_result",
    "build_sglang_firered_tts3_request",
    "build_stream_output",
]
