# SPDX-License-Identifier: Apache-2.0
"""Stage factory for Parakeet TDT ASR."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import ParakeetForTDT, ParakeetProcessor

from sglang_omni.preprocessing.transcription import prepare_audio
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.device import resolve_concrete_device

SAMPLE_RATE = 16000
SOURCE_NAME = "Parakeet TDT"
BAD_REQUEST_PREFIX = "Parakeet TDT request:"
# note (kzhou003): Bark 3.0 was evaluated with 0.25 s of silence before the speech.
LEADING_SILENCE = np.zeros(SAMPLE_RATE // 4, dtype=np.float32)
UNSUPPORTED_PARAMETERS = ("prompt", "max_new_tokens", "repetition_penalty")
TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def validate_request_parameters(parameters: dict[str, object]) -> None:
    """Reject knobs that greedy transducer decoding cannot honor."""
    unsupported = [name for name in UNSUPPORTED_PARAMETERS if name in parameters]
    if parameters.get("task", "transcribe") != "transcribe":
        raise ValueError(f"{BAD_REQUEST_PREFIX} only transcription is supported")
    elif unsupported:
        raise ValueError(f"{BAD_REQUEST_PREFIX} {unsupported[0]} is not supported")
    elif parameters.get("temperature", 0.0) != 0.0:
        raise ValueError(
            f"{BAD_REQUEST_PREFIX} decoding is greedy, temperature must be 0"
        )
    else:
        pass


class ParakeetTDTTranscriber:
    def __init__(
        self,
        model_path: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        max_audio_seconds: float,
    ) -> None:
        self.processor: ParakeetProcessor = ParakeetProcessor.from_pretrained(
            model_path
        )
        self.model: ParakeetForTDT = (
            ParakeetForTDT.from_pretrained(model_path, dtype=dtype).to(device).eval()
        )
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype
        self.max_audio_seconds: float = max_audio_seconds

    def prepare_waveform(self, payload: StagePayload) -> NDArray[np.float32]:
        validate_request_parameters(payload.request.params)
        audio = prepare_audio(
            payload,
            source_name=SOURCE_NAME,
            target_sample_rate=SAMPLE_RATE,
            max_duration_s=self.max_audio_seconds,
        )
        return np.concatenate([LEADING_SILENCE, audio.waveform])

    @torch.inference_mode()
    def transcribe_batch(
        self, payloads: list[StagePayload]
    ) -> list[StagePayload | ValueError]:
        """Fail only the requests whose audio or parameters are invalid."""
        results: list[StagePayload | ValueError] = []
        waveforms: list[NDArray[np.float32]] = []
        for payload in payloads:
            try:
                waveforms.append(self.prepare_waveform(payload))
                results.append(payload)
            except ValueError as error:
                results.append(error)
        if not waveforms:
            return results
        else:
            pass
        # note (kzhou003): each row stops at its own encoder length and is then
        # padded with pad tokens, so batching does not change any transcript.
        features = self.processor(
            waveforms, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).to(self.device, self.dtype)
        output = self.model.generate(**features)
        texts = iter(
            self.processor.batch_decode(output.sequences, skip_special_tokens=True)
        )
        for result in results:
            if isinstance(result, StagePayload):
                result.data = {"text": next(texts).strip()}
            else:
                pass
        return results

    def transcribe(self, payload: StagePayload) -> StagePayload:
        result = self.transcribe_batch([payload])[0]
        if isinstance(result, ValueError):
            raise result
        else:
            return result


def create_parakeet_tdt_executor(
    model_path: str,
    *,
    dtype: str,
    max_batch_size: int,
    max_batch_wait_ms: float,
    max_audio_seconds: float,
    device: str | None = None,
    gpu_id: int | None = None,
) -> SimpleScheduler:
    if dtype not in TORCH_DTYPES:
        raise ValueError(f"Parakeet TDT dtype must be one of {sorted(TORCH_DTYPES)}")
    else:
        pass
    transcriber = ParakeetTDTTranscriber(
        model_path,
        device=resolve_concrete_device(device, gpu_id),
        dtype=TORCH_DTYPES[dtype],
        max_audio_seconds=max_audio_seconds,
    )
    return SimpleScheduler(
        transcriber.transcribe,
        batch_compute_fn=transcriber.transcribe_batch,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )


__all__ = ["ParakeetTDTTranscriber", "create_parakeet_tdt_executor"]
