# SPDX-License-Identifier: Apache-2.0
"""StagePayload <-> SGLang request adapters for Voxtral realtime ASR.

Offline/batched ASR uses the Mistral transcription path: the tokenizer inserts
audio placeholder tokens into the prompt, and the model adds computed audio
embeddings to those placeholder positions.

Streaming token feedback (realtime chunk-by-chunk) is not yet wired; it will
need a custom scheduler/model-runner that carries per-request audio buffer and
encoder KV-cache state across decode steps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from mistral_common.protocol.transcription.request import (
    StreamingMode,
    TranscriptionRequest,
)
from mistral_common.tokens.tokenizers.audio import Audio
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
)
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData
from sglang_omni.utils.audio import audio_fingerprint, audio_fingerprint_int, load_audio

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000


@dataclass
class VoxtralASRRequestData(SGLangARRequestData):
    prompt_token_ids: list[int] | None = None
    output_ids: list[int] | None = None
    audio_duration_s: float = 0.0
    language: str | None = None
    audio_offsets: list[tuple[int, int]] | None = None


def _audio_source_from_payload(payload: StagePayload) -> Any:
    inputs = payload.request.inputs
    if isinstance(inputs, dict):
        for key in ("audio_bytes", "bytes", "file"):
            value = inputs.get(key)
            if value is not None:
                return value
        for key in ("audio_path", "path", "url"):
            value = inputs.get(key)
            if value is not None:
                return value
    return inputs


def _load_audio(source: Any) -> np.ndarray:
    return load_audio(
        source,
        source_name="Voxtral-ASR",
        target_sample_rate=_SAMPLE_RATE,
    )


def _import_mistral_common() -> tuple[Any, Any]:
    try:
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Voxtral ASR requires the `mistral_common` package. "
            "Install it with: pip install 'mistral_common[audio]>=1.11.0'"
        ) from exc
    return MistralTokenizer


def make_voxtral_asr_scheduler_adapters(
    *,
    tokenizer: Any,
    max_new_tokens: int,
) -> tuple[
    Callable[[StagePayload], VoxtralASRRequestData],
    Callable[[VoxtralASRRequestData], StagePayload],
]:
    """Build request/result adapters for a Voxtral ASR stage."""

    instruct = tokenizer.instruct_tokenizer
    audio_encoder = instruct.audio_encoder
    audio_config = audio_encoder.audio_config
    audio_token_id = int(audio_encoder.audio_token)
    inner = instruct.tokenizer
    eos_token_id = int(inner.eos_id)
    vocab_size = int(inner.n_words)

    def request_builder(payload: StagePayload) -> VoxtralASRRequestData:
        params = payload.request.params or {}
        metadata = payload.request.metadata or {}
        asr_params = (
            metadata.get("asr_params", {}) if isinstance(metadata, dict) else {}
        )

        audio = _load_audio(_audio_source_from_payload(payload))
        audio_duration_s = float(len(audio) / _SAMPLE_RATE)
        fingerprint = audio_fingerprint(audio)

        language = params.get("language") or asr_params.get("language")

        # Build the offline transcription prompt with audio placeholder tokens.
        audio_obj = Audio(audio, _SAMPLE_RATE, format="wav")
        transcription_request = TranscriptionRequest(
            model="voxtral",
            audio=audio_obj.to_base64(audio_obj.format),
            language=language,
            streaming=StreamingMode.OFFLINE,
        )
        tokenized = instruct.encode_transcription(transcription_request)
        prompt_token_ids = list(tokenized.tokens)

        # Use the padded waveform returned by mistral_common: it embeds the
        # raw audio between 32 left-pad tokens (2.56s of silence aligning with
        # the prompt's left-pad streaming tokens) and right padding, which is
        # exactly what vLLM feeds the encoder (tokenized.audios[0].audio_array).
        padded_audio = np.asarray(tokenized.audios[0].audio_array, dtype=np.float32)

        # Compute expected number of audio tokens from the padded waveform.
        num_audio_tokens = int(audio_config.num_audio_tokens(len(padded_audio)))

        # Voxtral realtime has no audio placeholder tokens in the prompt; the
        # audio embeddings are fused into the text embeddings across the whole
        # prompt span (mirroring vLLM's realtime offline path).
        if len(prompt_token_ids) == 0:
            raise ValueError("Empty prompt token ids were produced.")
        audio_offsets = [(0, len(prompt_token_ids) - 1)]

        audio_tensor = torch.from_numpy(padded_audio).to(dtype=torch.float32)
        audio_item = MultimodalDataItem(
            modality=Modality.AUDIO,
            hash=audio_fingerprint_int(fingerprint),
            feature=audio_tensor,
            offsets=audio_offsets,
        )

        # vLLM realtime offline bounds generation by the audio length so that
        # every decode position still has a matching audio frame:
        #   max_tokens = num_audio_tokens - len(prompt) - 1
        audio_bounded_max = max(num_audio_tokens - len(prompt_token_ids) - 1, 1)
        effective_max_new = min(
            int(params.get("max_new_tokens", max_new_tokens)), audio_bounded_max
        )

        sampling_params = SamplingParams(
            max_new_tokens=effective_max_new,
            temperature=float(params.get("temperature", 0.0)),
        )
        sampling_params.normalize(None)
        sampling_params.verify(vocab_size)

        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=prompt_token_ids,
            sampling_params=sampling_params,
            eos_token_ids={eos_token_id},
            vocab_size=vocab_size,
        )
        req.tokenizer = None
        req.multimodal_inputs = MultimodalInputs([audio_item])

        data = VoxtralASRRequestData(
            prompt_token_ids=prompt_token_ids,
            output_ids=req.output_ids,
            audio_duration_s=audio_duration_s,
            language=language,
            audio_offsets=audio_offsets,
            input_ids=torch.tensor(prompt_token_ids, dtype=torch.long),
            max_new_tokens=effective_max_new,
            req=req,
        )
        data.stage_payload = payload
        return data

    def result_adapter(data: VoxtralASRRequestData) -> StagePayload:
        from mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy

        payload = data.stage_payload
        output_ids = list(data.output_ids or [])
        logger.info(
            "Voxtral ASR output_ids len=%d first=%s last=%s",
            len(output_ids),
            output_ids[:20],
            output_ids[-20:] if output_ids else [],
        )
        text = tokenizer.decode(
            output_ids, special_token_policy=SpecialTokenPolicy.IGNORE
        )
        logger.info("Voxtral ASR decoded text: %r", text)
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                "text": text,
                "language": data.language,
                "duration_s": data.audio_duration_s,
                "usage": {"output_tokens": len(output_ids)},
                "modality": "text",
            },
        )

    return request_builder, result_adapter
