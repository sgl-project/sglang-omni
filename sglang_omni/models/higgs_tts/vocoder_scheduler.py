# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduler for Higgs TTS."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.models.higgs_tts import streaming_vocoder
from sglang_omni.models.higgs_tts.audio_codec import HiggsAudioCodec
from sglang_omni.models.higgs_tts.codebook_layout import reverse_delay_pattern
from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.models.tts_runtime import build_tts_usage, require_batch_result_count
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload


class HiggsStreamingVocoderScheduler(StreamingSimpleScheduler):
    """Decode Higgs codec rows incrementally, with batched final decode."""

    def __init__(
        self,
        codec: HiggsAudioCodec,
        *,
        stream_stride: int = 75,
        stream_followup_stride: int = 75,
        stream_overlap_tokens: int = 8,
        stream_holdback_tokens: int = 4,
        max_batch_size: int = 4,
        max_batch_wait_ms: int = 2,
    ) -> None:
        if stream_stride <= 0 or stream_followup_stride <= 0:
            raise ValueError("stream_stride and stream_followup_stride must be > 0")
        if stream_overlap_tokens < 0:
            raise ValueError("stream_overlap_tokens must be >= 0")
        if stream_holdback_tokens < 0:
            raise ValueError("stream_holdback_tokens must be >= 0")

        self._codec = codec
        self._stream_stride = int(stream_stride)
        self._stream_followup_stride = int(stream_followup_stride)
        self._stream_overlap_tokens = int(stream_overlap_tokens)
        self._stream_holdback_tokens = int(stream_holdback_tokens)
        self._sample_rate = HiggsAudioCodec.SAMPLE_RATE
        self._stream_states: dict[str, streaming_vocoder.HiggsStreamState] = {}
        self._samples_per_frame = self._resolve_samples_per_frame(codec)
        self._stream_config = streaming_vocoder.HiggsStreamConfig(
            stream_stride=self._stream_stride,
            stream_followup_stride=self._stream_followup_stride,
            stream_overlap_tokens=self._stream_overlap_tokens,
            stream_holdback_tokens=self._stream_holdback_tokens,
            samples_per_frame=self._samples_per_frame,
            sample_rate=self._sample_rate,
        )

        super().__init__(
            self._vocode_payload,
            batch_compute_fn=self._vocode_payloads,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        params = payload.request.params
        if not isinstance(params, dict):
            raise TypeError(
                f"Higgs request params must be a dict, got {type(params).__name__}"
            )
        return bool(params.get("stream", False))

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        stream_state = self._stream_states.setdefault(
            request_id, streaming_vocoder.HiggsStreamState()
        )
        if not isinstance(payload.data, dict):
            raise TypeError(
                f"Higgs streaming payload for {request_id!r} must be a dict, "
                f"got {type(payload.data).__name__}"
            )
        missing = [
            key for key in ("num_codebooks", "codebook_size") if key not in payload.data
        ]
        if not missing:
            streaming_vocoder.latch_higgs_stream_contract(
                request_id,
                stream_state,
                num_codebooks=payload.data["num_codebooks"],
                codebook_size=payload.data["codebook_size"],
                source="payload",
            )
            streaming_vocoder.latch_initial_codec_chunk_frames_from_mapping(
                payload.request_id,
                stream_state,
                (
                    payload.request.params
                    if isinstance(payload.request.params, dict)
                    else None
                ),
                config=self._stream_config,
            )
            return
        if (
            stream_state.num_codebooks is not None
            and stream_state.codebook_size is not None
        ):
            streaming_vocoder.latch_higgs_stream_contract(
                request_id,
                stream_state,
                num_codebooks=payload.data.get(
                    "num_codebooks", stream_state.num_codebooks
                ),
                codebook_size=payload.data.get(
                    "codebook_size", stream_state.codebook_size
                ),
                source="payload",
            )
            streaming_vocoder.latch_initial_codec_chunk_frames_from_mapping(
                payload.request_id,
                stream_state,
                (
                    payload.request.params
                    if isinstance(payload.request.params, dict)
                    else None
                ),
                config=self._stream_config,
            )
            return
        raise RuntimeError(
            f"Higgs streaming payload for {request_id!r} is missing fields: "
            f"{', '.join(missing)}"
        )

    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        state = self._stream_states.setdefault(
            request_id, streaming_vocoder.HiggsStreamState()
        )
        streaming_vocoder.latch_higgs_stream_metadata(
            request_id,
            state,
            item.metadata,
            config=self._stream_config,
        )

        row = item.data
        if not isinstance(row, torch.Tensor):
            raise TypeError(
                f"Higgs stream chunk for {request_id!r} must carry a torch.Tensor, "
                f"got {type(row).__name__}"
            )
        row = row.to(dtype=torch.long)
        if row.ndim != 1:
            raise ValueError(
                f"Higgs stream chunk must be 1-D [N], got {tuple(row.shape)}"
            )

        num_codebooks = streaming_vocoder.require_higgs_stream_contract(
            state, request_id
        )[0]
        if int(row.shape[0]) != num_codebooks:
            raise ValueError(
                f"Higgs stream chunk has {int(row.shape[0])} codebooks, "
                f"expected {num_codebooks}"
            )
        state.delayed_rows.append(row)

        output = streaming_vocoder.build_higgs_stream_delta(
            state,
            config=self._stream_config,
            decode_delayed_rows=self._decode_delayed_rows,
            is_final=False,
        )
        if output is None:
            return []
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=output,
                metadata={"modality": "audio"},
            )
        ]

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        payload = self._stream_payloads[request_id]
        state = self._stream_states.setdefault(
            request_id, streaming_vocoder.HiggsStreamState()
        )
        output = streaming_vocoder.build_higgs_stream_delta(
            state,
            config=self._stream_config,
            decode_delayed_rows=self._decode_delayed_rows,
            is_final=True,
        )
        if output is None and not state.has_emitted:
            output = self._audio_payload_from_stage_payload(payload)

        messages: list[OutgoingMessage] = []
        if output is not None:
            messages.append(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data=output,
                    metadata={"modality": "audio"},
                )
            )

        final_data: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": self._sample_rate,
        }
        usage = self._build_usage(HiggsTtsState.from_dict(payload.data))
        if usage is not None:
            final_data["usage"] = usage
        messages.append(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=StagePayload(
                    request_id=payload.request_id,
                    request=payload.request,
                    data=final_data,
                ),
            )
        )
        return messages

    def clear_stream_state(self, request_id: str) -> None:
        self._stream_states.pop(request_id, None)

    def _audio_payload_from_stage_payload(
        self, payload: StagePayload
    ) -> dict[str, Any] | None:
        state = HiggsTtsState.from_dict(payload.data)
        audio = self._decode_state_to_audio(state)
        if audio is None:
            return None
        return audio_waveform_payload(
            audio,
            sample_rate=self._sample_rate,
            modality="audio",
            source_hint="Higgs TTS streaming",
        )

    def _vocode_payload(self, payload: StagePayload) -> StagePayload:
        return self._vocode_payloads([payload])[0]

    def _vocode_payloads(self, payloads: list[StagePayload]) -> list[StagePayload]:
        items = [self._prepare_vocoder_item(payload) for payload in payloads]
        valid = [(i, codes) for i, (_, codes) in enumerate(items) if codes is not None]
        waveforms: list[torch.Tensor | None] = [None] * len(items)
        if valid:
            indices, codes_list = zip(*valid)
            wavs = self._codec.decode_batch(list(codes_list))
            require_batch_result_count(
                owner="Higgs vocoder decode_batch",
                result_label="audios",
                actual=len(wavs),
                expected=len(valid),
            )
            for idx, wav in zip(indices, wavs):
                waveforms[idx] = wav
        return [
            self._store_vocoder_result(payload, state, wav)
            for payload, (state, _), wav in zip(payloads, items, waveforms)
        ]

    def _prepare_vocoder_item(
        self,
        payload: StagePayload,
    ) -> tuple[HiggsTtsState, torch.Tensor | None]:
        state = HiggsTtsState.from_dict(payload.data)
        delayed_rows = state.output_codes_delayed
        if not delayed_rows:
            return state, None
        delayed_LN = torch.tensor(delayed_rows, dtype=torch.long)
        if delayed_LN.shape[0] < state.num_codebooks:
            return state, None
        codes_TN = reverse_delay_pattern(delayed_LN)
        codec_vocab = int(state.codebook_size) - 2
        return state, torch.where(
            codes_TN >= codec_vocab, torch.zeros_like(codes_TN), codes_TN
        )

    def _store_vocoder_result(
        self,
        payload: StagePayload,
        state: HiggsTtsState,
        waveform: torch.Tensor | None,
    ) -> StagePayload:
        data = audio_waveform_payload(
            waveform if waveform is not None else [],
            sample_rate=self._sample_rate,
            modality="audio",
            source_hint="Higgs TTS vocoder",
        )
        usage = self._build_usage(state)
        if usage is not None:
            data["usage"] = usage
        payload.data = data
        return payload

    def _decode_state_to_audio(self, state: HiggsTtsState) -> torch.Tensor | None:
        delayed_rows = state.output_codes_delayed
        if not delayed_rows:
            return None
        rows = [torch.tensor(row, dtype=torch.long) for row in delayed_rows]
        if len(rows) < int(state.num_codebooks):
            return None
        return self._decode_delayed_rows(
            rows,
            num_codebooks=int(state.num_codebooks),
            codebook_size=int(state.codebook_size),
        )

    def _decode_delayed_rows(
        self,
        rows: list[torch.Tensor],
        *,
        num_codebooks: int,
        codebook_size: int,
    ) -> torch.Tensor:
        if len(rows) < int(num_codebooks):
            raise ValueError(
                f"Higgs delayed rows must include at least {num_codebooks} rows, "
                f"got {len(rows)}"
            )
        delayed_LN = torch.stack(rows, dim=0).to(torch.long)
        codes_TN = reverse_delay_pattern(delayed_LN)
        codec_vocab = int(codebook_size) - 2
        codes_TN = torch.where(
            codes_TN >= codec_vocab, torch.zeros_like(codes_TN), codes_TN
        )
        return self._codec.decode(codes_TN).detach().to(torch.float32)

    @staticmethod
    def _build_usage(state: HiggsTtsState) -> dict[str, Any] | None:
        return build_tts_usage(
            prompt_tokens=state.prompt_tokens,
            completion_tokens=state.completion_tokens,
            engine_time_s=state.engine_time_s,
        )

    @staticmethod
    def _resolve_samples_per_frame(codec: HiggsAudioCodec) -> int | None:
        hop_length = getattr(getattr(codec, "model", None), "config", None)
        hop_length = getattr(hop_length, "hop_length", None)
        if hop_length is None:
            return None
        hop_length_i = int(hop_length)
        return hop_length_i if hop_length_i > 0 else None


__all__ = ["HiggsStreamingVocoderScheduler"]
