# SPDX-License-Identifier: Apache-2.0
"""A forced-alignment worker for the `forced` method of the cascade's alignment node: word ends of known text in its audio.

Each request is aligned in its `language`, a language name such as `English`
or `Chinese`, or in the worker's `--language` when it names none.
"""

import argparse
import base64
import binascii
import logging
import threading

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from qwen_asr import Qwen3ForcedAligner

from sglang_omni.models.streaming_cascade.alignment import (
    MAX_ALIGNMENT_AUDIO_BYTES,
    AlignmentBoundary,
    AlignmentResult,
)
from sglang_omni.models.streaming_cascade.numerals import spoken_text

logger = logging.getLogger("alignment_worker")


class AlignmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1, max_length=16000)
    audio: str = Field(max_length=MAX_ALIGNMENT_AUDIO_BYTES * 4 // 3 + 4)
    sample_rate_hz: int = Field(gt=0, le=192000)
    audio_format: str
    language: str | None = Field(default=None, min_length=1, max_length=64)


def create_app(model: str, device: str, default_language: str) -> FastAPI:
    torch.set_num_threads(4)
    aligner = Qwen3ForcedAligner.from_pretrained(
        model, device_map=device, dtype=torch.bfloat16, attn_implementation="eager"
    )
    aligner.align(
        audio=(np.zeros(16000, dtype=np.float32), 16000),
        text="hello",
        language="English",
    )
    inference_lock = threading.Lock()
    application = FastAPI()

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ready"}

    @application.post("/align", response_model=AlignmentResult)
    def align(request: AlignmentRequest) -> AlignmentResult:
        try:
            pcm16 = base64.b64decode(request.audio, validate=True)
        except binascii.Error as exc:
            raise HTTPException(422, "Invalid base64 audio") from exc
        if (
            request.audio_format != "pcm16"
            or not pcm16
            or len(pcm16) % 2
            or len(pcm16) > MAX_ALIGNMENT_AUDIO_BYTES
        ):
            logger.warning(
                "rejected %s audio of %d bytes", request.audio_format, len(pcm16)
            )
            raise HTTPException(422, "Expected bounded, complete mono PCM16")
        else:
            pass
        samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768
        if request.language is None:
            language = default_language
        else:
            language = request.language
        text, origin = spoken_text(request.text, language)
        with inference_lock:
            result = aligner.align(
                audio=(samples, request.sample_rate_hz),
                text=text,
                language=language,
            )[0]
        boundaries = []
        cursor = 0
        for token in result:
            position = text.find(token.text, cursor)
            if position < 0:
                logger.warning(
                    "aligner token %r not found in %r after %d",
                    token.text,
                    text,
                    cursor,
                )
                raise HTTPException(
                    422,
                    "Aligner normalized text requires an explicit original-text mapping",
                )
            else:
                pass
            cursor = position + len(token.text)
            # Note (Junnan Li): The aligner works in frames of its own, so a token may end a few milliseconds past the audio; that is the audio's end.
            sample_end = min(
                round(token.end_time * request.sample_rate_hz), len(samples)
            )
            if sample_end < 0:
                logger.warning(
                    "aligner token %r ends before the audio starts", token.text
                )
                raise HTTPException(422, "Alignment precedes source audio")
            else:
                pass
            boundaries.append(
                AlignmentBoundary(
                    text_end_chars=origin[cursor - 1] + 1, audio_end_samples=sample_end
                )
            )
        return AlignmentResult(boundaries=boundaries)

    return application


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--language",
        default="Chinese",
        help="the language of a request that names none",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    arguments = parser.parse_args()
    uvicorn.run(
        create_app(arguments.model, arguments.device, arguments.language),
        host=arguments.host,
        port=arguments.port,
    )


if __name__ == "__main__":
    main()
