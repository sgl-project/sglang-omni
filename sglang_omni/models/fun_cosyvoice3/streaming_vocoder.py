# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduler for Fun-CosyVoice3."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
from sglang_omni.models.fun_cosyvoice3.streaming import (
    PRE_LOOKAHEAD_LEN,
    SAMPLE_RATE,
    TOKEN_HOP_LEN,
    as_flow_embedding,
    as_flow_prompt_feat,
    as_flow_prompt_token,
    next_stream_hop_len,
    prompt_token_pad,
    stream_hop_len,
    tokens_needed_for_causal_chunk,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload


@dataclass
class _CosyVoice3StreamState:
    tokens: list[int] = field(default_factory=list)
    token_offset: int = 0
    hop_len: int = TOKEN_HOP_LEN
    prompt_pad: int = 0
    prompt_token: torch.Tensor | None = None
    prompt_feat: torch.Tensor | None = None
    embedding: torch.Tensor | None = None
    prompts_latched: bool = False
    hift_mel: torch.Tensor | None = None
    speech_offset: int = 0


class FunCosyVoice3StreamingVocoderScheduler(
    StreamingVocoderBase[_CosyVoice3StreamState, None]
):
    """Decode CosyVoice3 speech tokens incrementally through Flow + HiFT."""

    def __init__(
        self,
        vocoder: Any,
        *,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 2,
        sample_rate: int = SAMPLE_RATE,
        request_cost_fn: Callable[[Any], int] | None = None,
        max_batch_cost: int | None = None,
    ) -> None:
        self._vocoder = vocoder
        super().__init__(
            self._vocode_payload,
            batch_compute_fn=self._vocode_payloads,
            sample_rate=int(sample_rate),
            stream_source_hint="Fun-CosyVoice3",
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            request_cost_fn=request_cost_fn,
            max_batch_cost=max_batch_cost,
        )

    async def _vocode_payload(self, payload: StagePayload) -> StagePayload:
        results = await self._vocoder.decode_payloads([payload])
        return results[0]

    async def _vocode_payloads(
        self, payloads: list[StagePayload]
    ) -> list[StagePayload]:
        return await self._vocoder.decode_payloads(payloads)

    def create_stream_state(self, request_id: str) -> _CosyVoice3StreamState:
        del request_id
        return _CosyVoice3StreamState()

    def latch_stream_contract(
        self,
        request_id: str,
        state: _CosyVoice3StreamState,
        source: StagePayload | Mapping[str, Any],
        *,
        origin: str,
    ) -> None:
        if origin == "payload":
            payload = source
            if not isinstance(payload, StagePayload):
                raise TypeError(
                    f"Fun-CosyVoice3 streaming payload for {request_id!r} must "
                    f"be a StagePayload, got {type(payload).__name__}"
                )
            pipeline_state = FunCosyVoice3State.from_dict(payload.data)
            self._latch_prompts(
                request_id,
                state,
                prompt_token=pipeline_state.flow_prompt_speech_token,
                prompt_feat=pipeline_state.flow_prompt_speech_feat,
                embedding=pipeline_state.flow_embedding,
            )
            return
        metadata: Mapping[str, Any] = source
        if any(
            key in metadata
            for key in (
                "flow_prompt_speech_token",
                "flow_prompt_speech_feat",
                "flow_embedding",
            )
        ):
            self._latch_prompts(
                request_id,
                state,
                prompt_token=metadata.get("flow_prompt_speech_token"),
                prompt_feat=metadata.get("flow_prompt_speech_feat"),
                embedding=metadata.get("flow_embedding"),
            )

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        super().on_streaming_new_request(request_id, payload)
        # note (guozhihao-224): payload can arrive after buffered chunks; drain
        # once prompts latch. Skip if stream_done already parked and base
        # new_request finalized.
        if request_id in self._pending_done:
            return
        state = self._stream_states.get(request_id)
        if state is None:
            return
        for message in self._decode_and_emit(request_id, state):
            self.outbox.put(message)

    def _latch_prompts(
        self,
        request_id: str,
        state: _CosyVoice3StreamState,
        *,
        prompt_token: Any,
        prompt_feat: Any,
        embedding: Any,
    ) -> None:
        token = as_flow_prompt_token(prompt_token)
        feat = as_flow_prompt_feat(prompt_feat)
        spk = as_flow_embedding(embedding)
        if state.prompts_latched:
            # note (guozhihao-224): latch is shape-stable; payload and first
            # chunk metadata must carry the same prompt tensors.
            if (
                tuple(token.shape) != tuple(state.prompt_token.shape)
                or tuple(feat.shape) != tuple(state.prompt_feat.shape)
                or tuple(spk.shape) != tuple(state.embedding.shape)
            ):
                raise ValueError(
                    f"Fun-CosyVoice3 stream prompt tensors changed for {request_id!r}"
                )
            return
        state.prompt_token = token
        state.prompt_feat = feat
        state.embedding = spk
        state.prompt_pad = prompt_token_pad(int(token.shape[1]))
        state.prompts_latched = True

    def validate_chunk(
        self,
        request_id: str,
        state: _CosyVoice3StreamState,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        del request_id, state
        chunk = codes.to(dtype=torch.long)
        if chunk.ndim == 2 and chunk.shape[-1] == 1:
            chunk = chunk.reshape(-1)
        if chunk.ndim != 1:
            raise ValueError(
                f"Fun-CosyVoice3 stream chunk must be 1-D speech tokens, "
                f"got {tuple(chunk.shape)}"
            )
        return chunk.contiguous()

    def ingest(
        self,
        request_id: str,
        state: _CosyVoice3StreamState,
        codes: torch.Tensor,
    ) -> None:
        del request_id
        state.tokens.extend(int(token) for token in codes.tolist())

    def should_decode(self, state: _CosyVoice3StreamState, *, is_final: bool) -> bool:
        del is_final
        return self._ready_for_causal_chunk(state)

    def _ready_for_causal_chunk(self, state: _CosyVoice3StreamState) -> bool:
        if not state.prompts_latched:
            return False
        needed = tokens_needed_for_causal_chunk(
            state.token_offset,
            hop_len=state.hop_len,
            prompt_pad=state.prompt_pad,
        )
        return len(state.tokens) >= needed

    def decode_delta(
        self,
        request_id: str,
        state: _CosyVoice3StreamState,
        *,
        is_final: bool,
    ) -> torch.Tensor | None:
        del request_id
        pieces: list[torch.Tensor] = []
        while self._ready_for_causal_chunk(state):
            hop = stream_hop_len(
                state.token_offset,
                hop_len=state.hop_len,
                prompt_pad=state.prompt_pad,
            )
            token_end = state.token_offset + hop + PRE_LOOKAHEAD_LEN
            delta = self._run_flow_hift(
                state,
                token_end=token_end,
                token_offset=state.token_offset,
                streaming=True,
                finalize=False,
            )
            state.token_offset += hop
            state.hop_len = next_stream_hop_len(state.hop_len)
            if delta is not None and delta.numel() > 0:
                pieces.append(delta)
        if is_final:
            if not state.tokens:
                return None if not pieces else torch.cat(pieces, dim=-1)
            delta = self._run_flow_hift(
                state,
                token_end=len(state.tokens),
                token_offset=state.token_offset,
                streaming=False,
                finalize=True,
            )
            if delta is not None and delta.numel() > 0:
                pieces.append(delta)
        if not pieces:
            return None
        return torch.cat(pieces, dim=-1)

    def _run_flow_hift(
        self,
        state: _CosyVoice3StreamState,
        *,
        token_end: int,
        token_offset: int,
        streaming: bool,
        finalize: bool,
    ) -> torch.Tensor | None:
        if state.prompt_token is None or state.prompt_feat is None:
            raise RuntimeError(
                "Fun-CosyVoice3 streaming vocoder decoded before prompt "
                "conditioning was latched"
            )
        if state.embedding is None:
            raise RuntimeError(
                "Fun-CosyVoice3 streaming vocoder decoded before speaker "
                "embedding was latched"
            )
        generated = state.tokens[:token_end]
        if not generated:
            raise RuntimeError(
                "Fun-CosyVoice3 streaming vocoder has no speech tokens to decode"
            )
        token = torch.tensor(generated, dtype=torch.int32).unsqueeze(0)
        wav, hift_mel, speech_offset = self._vocoder.token2wav_chunk(
            token=token,
            prompt_token=state.prompt_token,
            prompt_feat=state.prompt_feat,
            embedding=state.embedding,
            token_offset=token_offset,
            streaming=streaming,
            finalize=finalize,
            hift_mel=state.hift_mel,
            speech_offset=state.speech_offset,
        )
        state.hift_mel = hift_mel
        state.speech_offset = speech_offset
        return wav

    def fallback_full_decode(
        self,
        request_id: str,
        payload: StagePayload,
        state: _CosyVoice3StreamState,
    ) -> torch.Tensor | None:
        del request_id, state
        pipeline_state = FunCosyVoice3State.from_dict(payload.data)
        if pipeline_state.audio_codes is None:
            codes = torch.zeros(0, dtype=torch.long)
        else:
            codes = torch.as_tensor(
                pipeline_state.audio_codes, dtype=torch.long
            ).reshape(-1)
        if codes.numel() == 0:
            raise RuntimeError(
                "Fun-CosyVoice3 generation produced no usable speech tokens"
            )
        prompt_token = as_flow_prompt_token(pipeline_state.flow_prompt_speech_token)
        prompt_feat = as_flow_prompt_feat(pipeline_state.flow_prompt_speech_feat)
        embedding = as_flow_embedding(pipeline_state.flow_embedding)
        return self._vocoder.token2wav(
            token=codes.unsqueeze(0),
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding,
        )

    def final_result_data(
        self,
        request_id: str,
        payload: StagePayload,
        state: _CosyVoice3StreamState,
    ) -> dict[str, Any]:
        del request_id, state
        final_data: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": self._sample_rate,
        }
        pipeline_state = FunCosyVoice3State.from_dict(payload.data)
        usage = build_usage(pipeline_state)
        if usage is not None:
            final_data["usage"] = usage
        return final_data

    def stream_payload(self, request_id: str, waveform: torch.Tensor) -> dict[str, Any]:
        del request_id
        return audio_waveform_payload(
            waveform,
            sample_rate=self._sample_rate,
            modality="audio",
            source_hint="Fun-CosyVoice3",
        )

    def release_stream_resources(
        self, request_id: str, state: _CosyVoice3StreamState
    ) -> None:
        del request_id
        state.tokens.clear()
        state.hift_mel = None
        state.prompt_token = None
        state.prompt_feat = None
        state.embedding = None
