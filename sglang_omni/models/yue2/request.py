# SPDX-License-Identifier: Apache-2.0
"""YuE2 request validation and checkpoint-native text prompts."""

import base64
import unicodedata
from pathlib import Path
from typing import Literal, Protocol

import tiktoken
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sglang_omni.proto.request import StagePayload

TEXT_END = 151643
ABC_START, ABC_END = 151847, 151848
MUSIC_START, MUSIC_END = 151851, 151852
CODEC_OFFSET, CODEC_SIZE = 151853, 32768
INSTRUCTIONS = {
    "off": "Generate music with codec tokens from the given conditions.",
    "melody": "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.",
    "full": "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.",
}


class CancellationCheck(Protocol):
    def __call__(self) -> bool: ...


def check_cancelled(cancelled: CancellationCheck) -> None:
    if cancelled():
        raise InterruptedError("YuE2 request cancelled")
    else:
        pass


class SamplingConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)
    temperature: float = Field(default=1.0, ge=0, le=5)
    top_p: float = Field(default=0.95, gt=0, le=1)
    top_k: int = Field(default=100, ge=1)
    repetition_penalty: float = Field(default=1.2, gt=0)
    penalty_window: int = Field(default=50, ge=1, le=100)
    min_tokens: int = Field(default=200, ge=0)
    max_tokens: int = Field(default=9000, ge=1)

    @model_validator(mode="after")
    def validate_budget(self) -> "SamplingConfig":
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens must not exceed max_tokens")
        else:
            return self


class Yue2Options(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)
    cot: Literal["full", "melody", "off"] = "full"
    abc: str | None = None
    cfg_scale: float | None = Field(default=None, ge=0, le=20)
    abc_max_tokens: int = Field(default=4096, ge=1)
    ode_steps: int = Field(default=32, ge=1, le=200)

    @model_validator(mode="after")
    def validate_score(self) -> "Yue2Options":
        if self.abc is not None and (not self.abc.strip() or self.cot == "off"):
            raise ValueError(
                "External ABC requires nonempty text and cot=full or melody"
            )
        else:
            return self


class Yue2Request(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    lyrics: str
    style: str
    seed: int = Field(default=831001, ge=0, lt=2**63)
    options: Yue2Options
    semantic_sampling: SamplingConfig

    @property
    def planning_sampling(self) -> SamplingConfig:
        return SamplingConfig(
            temperature=0.7,
            top_p=0.9,
            top_k=30,
            repetition_penalty=1.005,
            penalty_window=100,
            min_tokens=min(32, self.options.abc_max_tokens),
            max_tokens=self.options.abc_max_tokens,
        )

    @property
    def guidance(self) -> float:
        if self.options.cfg_scale is not None:
            return self.options.cfg_scale
        else:
            return 1.01 if self.options.cot == "off" else 1.0


def build_request(payload: StagePayload) -> Yue2Request:
    request = payload.request
    speech = request.metadata.get("tts_params")
    if not isinstance(speech, dict):
        raise ValueError("YuE2 expects a /v1/audio/speech request")
    else:
        pass
    unsupported = {
        name
        for name in (
            "language",
            "ref_audio",
            "ref_text",
            "task_type",
            "x_vector_only_mode",
            "token_count",
            "duration_tokens",
            "initial_codec_chunk_frames",
            "suppress_bootstrap_silence",
            "stream_codec_output",
        )
        if speech.get(name) is not None
    }
    if unsupported:
        raise ValueError(f"YuE2 does not support: {', '.join(sorted(unsupported))}")
    elif speech.get("voice", "default") not in (None, "", "default"):
        raise ValueError("YuE2 does not support voice selection")
    elif speech.get("speed", 1.0) != 1.0:
        raise ValueError("YuE2 requires speed=1.0")
    elif request.params.get("stream", False):
        raise ValueError("YuE2 requires stream=false")
    elif speech.get("instructions") is None:
        raise ValueError("YuE2 requires instructions (the musical style)")
    elif any(
        request.params.get(name)
        for name in (
            "epsilon_cutoff",
            "min_p",
            "stop",
            "stop_token_ids",
            "stage_sampling",
        )
    ):
        raise ValueError(
            "YuE2 does not support custom stopping, epsilon_cutoff, min_p or stage_sampling"
        )
    else:
        pass
    stages = request.params.get("stage_params") or {}
    if not isinstance(stages, dict) or set(stages) - {"synthesis"}:
        raise ValueError("YuE2 stage_params must target synthesis")
    else:
        options = Yue2Options.model_validate(stages.get("synthesis", {}))
    sampling_values = {
        name: request.params[name]
        for name in ("temperature", "top_p", "top_k", "repetition_penalty")
        if request.params.get(name) is not None
    }
    token_budget = request.params.get("max_new_tokens")
    if token_budget is not None:
        sampling_values["max_tokens"] = token_budget
        if isinstance(token_budget, int) and not isinstance(token_budget, bool):
            sampling_values["min_tokens"] = min(200, token_budget)
        else:
            pass
    else:
        pass
    values = {
        "lyrics": request.inputs,
        "style": speech["instructions"],
        "options": options,
        "semantic_sampling": SamplingConfig.model_validate(sampling_values),
    }
    if speech.get("seed") is not None:
        values["seed"] = speech["seed"]
    else:
        pass
    return Yue2Request.model_validate(values)


class Yue2Tokenizer:
    def __init__(self, path: Path) -> None:
        ranks = {
            base64.b64decode(token): int(rank)
            for token, rank in (
                line.split() for line in path.read_bytes().splitlines() if line
            )
        }
        if len(ranks) != TEXT_END or set(ranks.values()) != set(range(TEXT_END)):
            raise ValueError("YuE2 requires the checkpoint's qwen.tiktoken vocabulary")
        else:
            pass
        self.encoding = tiktoken.Encoding(
            "YuE2",
            mergeable_ranks=ranks,
            special_tokens={},
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        )

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode_ordinary(unicodedata.normalize("NFC", text))

    def prefix(self, request: Yue2Request, abc_tokens: list[int] | None) -> list[int]:
        prompt = f"{INSTRUCTIONS[request.options.cot]}\n[Tags]\n{request.style}\n[Lyrics]\n{request.lyrics}\n"
        prefix = [TEXT_END, *self.encode(prompt), ABC_START]
        if request.options.cot == "off":
            return [*prefix, ABC_END, MUSIC_START]
        elif abc_tokens is not None:
            return [*prefix, *abc_tokens, ABC_END, MUSIC_START]
        else:
            return prefix

    def negative_prefix(self, request: Yue2Request, abc_tokens: list[int]) -> list[int]:
        prefix = [TEXT_END, *self.encode(INSTRUCTIONS[request.options.cot])]
        if request.options.cot == "off":
            return [*prefix, MUSIC_START]
        else:
            return [*prefix, ABC_START, *abc_tokens, ABC_END, MUSIC_START]
