# SPDX-License-Identifier: Apache-2.0
"""Repeat seeded Qwen3-TTS requests under the legacy limit and the server default, keeping the evidence."""

import argparse
import asyncio
import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import httpx
import soundfile as sf
from pydantic import BaseModel, Field, TypeAdapter

from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.eval.benchmark_tts_seedtts import DEFAULT_MAX_NEW_TOKENS
from sglang_omni.serve.protocol import (
    CreateSpeechRequest,
    SpeechReference,
    SpeechStreamOutcome,
)

SEEDTTS_SPLIT_LANGUAGES = {"en": "English", "zh": "Chinese"}
PCM_SAMPLE_RATE_HZ = 24000
PCM_SAMPLE_BYTES = 2
PROBE_REQUEST_TIMEOUT_SECONDS = 300.0
MAX_PROBE_RESPONSE_BYTES = 32 * 1024 * 1024


class GenerationProbeCase(BaseModel):
    id: str
    input: str = Field(min_length=1)
    language: str
    instructions: str | None = None
    references: list[SpeechReference] | None = None


# note (luojiaxuan): Ordinary short lines in both languages, for checking that the default budget leaves normal speech whole.
PROBE_CASES = (
    GenerationProbeCase(
        id="short_en", input="Hello! How can I help you?", language="English"
    ),
    GenerationProbeCase(
        id="short_zh", input="你好，有什么可以帮你的吗？", language="Chinese"
    ),
    GenerationProbeCase(
        id="numbers_en",
        input="Your verification code is 102938. The total is $12.50.",
        language="English",
    ),
    GenerationProbeCase(
        id="mixed_zh",
        input="今天下午三点开会，请带上显卡测试结果。",
        language="Chinese",
    ),
)
# note (luojiaxuan): SeedTTS EN prompts that never emitted codec EOS on 0.6B Base in #1179.
SEEDTTS_RUNAWAY_SAMPLE_IDS = (
    "common_voice_en_17879420-common_voice_en_17879418",
    "common_voice_en_25572984-common_voice_en_25572982",
    "common_voice_en_581024-common_voice_en_581027",
    "common_voice_en_652024-common_voice_en_652023",
    "common_voice_en_20918607-common_voice_en_20918611",
    "common_voice_en_25014861-common_voice_en_25014863",
    "common_voice_en_27099623-common_voice_en_27099624",
    "common_voice_en_27099623-common_voice_en_27099625",
    "common_voice_en_96260-common_voice_en_96263",
)


class GenerationProbeResult(BaseModel):
    case_id: str
    request: CreateSpeechRequest
    policy: Literal["baseline", "text"]
    request_id: str | None = None
    worker_id: str | None = None
    finish_reason: str | None = None
    completion_frames: int | None = None
    audio_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    chunk_arrival_seconds: list[float] = Field(default_factory=list)
    chunk_bytes: list[int] = Field(default_factory=list)
    audio_sha256: str | None = None
    errors: list[str] = Field(default_factory=list)


async def synthesize_probe(
    client: httpx.AsyncClient,
    base_url: str,
    result: GenerationProbeResult,
    audio_path: Path,
) -> None:
    audio_bytes = bytearray()
    started_seconds = time.perf_counter()
    sample_rate_hz = PCM_SAMPLE_RATE_HZ
    outcome_id: str | None = None
    try:
        async with client.stream(
            "POST",
            f"{base_url.rstrip('/')}/audio/speech",
            json=result.request.model_dump(exclude_none=True),
        ) as response:
            response.raise_for_status()
            result.request_id = response.headers.get("X-Request-Id")
            result.worker_id = response.headers.get("X-SGLang-Omni-Worker")
            if result.request.stream:
                sample_rate_header = response.headers.get("X-Sample-Rate", "")
                sample_rate_hz = (
                    int(sample_rate_header) if sample_rate_header.isdigit() else 0
                )
                if sample_rate_hz != PCM_SAMPLE_RATE_HZ:
                    result.errors.append(
                        f"Expected {PCM_SAMPLE_RATE_HZ} Hz, got "
                        f"{sample_rate_header or 'missing'}"
                    )
                else:
                    pass
            async for chunk in response.aiter_bytes():
                result.chunk_arrival_seconds.append(
                    time.perf_counter() - started_seconds
                )
                result.chunk_bytes.append(len(chunk))
                audio_bytes.extend(chunk)
                if len(audio_bytes) > MAX_PROBE_RESPONSE_BYTES:
                    raise ValueError("Audio response exceeded the probe byte limit")
                else:
                    pass
            if result.request.stream:
                outcome_id = response.headers["X-SGLang-Omni-Speech-Id"]
            else:
                result.finish_reason = response.headers.get("X-Finish-Reason")
                result.completion_frames = int(response.headers["X-Completion-Tokens"])
        if result.request.stream and outcome_id:
            outcome_response = await client.get(
                f"{base_url.rstrip('/')}/audio/speech/{quote(outcome_id, safe='')}",
                headers=(
                    {"x-sglang-omni-route-worker": result.worker_id}
                    if result.worker_id
                    else {}
                ),
            )
            outcome_response.raise_for_status()
            outcome = SpeechStreamOutcome.model_validate_json(outcome_response.content)
            if outcome.request_id != outcome_id:
                raise ValueError("Stream outcome request id does not match")
            else:
                result.finish_reason = outcome.finish_reason
                result.completion_frames = (
                    outcome.usage.completion_tokens
                    if outcome.usage is not None
                    else None
                )
        elif result.request.stream:
            result.errors.append("Missing speech stream outcome id")
        else:
            pass
    except (httpx.HTTPError, ValueError, KeyError) as error:
        result.errors.append(f"{type(error).__name__}: {error}")
    finally:
        result.elapsed_seconds = time.perf_counter() - started_seconds
        if result.request.stream:
            audio_path.with_suffix(".pcm").write_bytes(audio_bytes)
            if len(audio_bytes) % PCM_SAMPLE_BYTES:
                result.errors.append("Stream ended with an incomplete PCM sample")
            elif audio_bytes and sample_rate_hz > 0:
                samples, sample_rate_hz = sf.read(
                    io.BytesIO(audio_bytes),
                    format="RAW",
                    subtype="PCM_16",
                    endian="LITTLE",
                    channels=1,
                    samplerate=sample_rate_hz,
                )
                sf.write(audio_path, samples, sample_rate_hz, subtype="PCM_16")
                result.audio_seconds = len(samples) / sample_rate_hz
            else:
                pass
        else:
            audio_path.write_bytes(audio_bytes)
            if audio_bytes:
                try:
                    samples, sample_rate_hz = sf.read(io.BytesIO(audio_bytes))
                    result.audio_seconds = len(samples) / sample_rate_hz
                except (sf.LibsndfileError, ValueError) as error:
                    result.errors.append(f"Invalid audio: {error}")
            else:
                pass
        result.audio_sha256 = hashlib.sha256(audio_bytes).hexdigest()

    if result.audio_seconds <= 0:
        result.errors.append("No decodable audio")
    else:
        pass
    # note (ratish): only a limit the request sets is known here; the server default is per task, the text budget on Base and the checkpoint's limit on CustomVoice and VoiceDesign.
    if result.completion_frames is None:
        result.errors.append("Missing completion frame count")
    elif (
        result.request.max_new_tokens is not None
        and result.completion_frames > result.request.max_new_tokens
    ):
        result.errors.append(
            f"Generation exceeded the request's {result.request.max_new_tokens}-frame limit"
        )
    else:
        pass
    if result.finish_reason not in ("stop", "length"):
        result.errors.append(f"Unexpected finish reason: {result.finish_reason}")
    else:
        pass


async def run_generation_probe(
    client: httpx.AsyncClient,
    base_url: str,
    request_template: CreateSpeechRequest,
    cases: list[GenerationProbeCase],
    seeds: range,
    stream: bool,
    output_dir: Path,
    *,
    include_baseline: bool = True,
    keep_passing_audio: bool = True,
    request_timeout_seconds: float = PROBE_REQUEST_TIMEOUT_SECONDS,
) -> list[GenerationProbeResult]:
    """Compare the same inputs and seeds; save partial audio even when a request fails, and without keep_passing_audio keep only the audio of pairs with an error."""
    if not cases or not seeds or request_timeout_seconds <= 0:
        raise ValueError("Cases, seeds, and a positive request timeout are required")
    elif (
        request_template.stage_params is not None
        and request_template.stage_params.get("tts_engine", {}).get("max_new_tokens")
        is not None
    ):
        raise ValueError(
            "Remove stage_params.tts_engine.max_new_tokens to exercise both budget policies"
        )
    else:
        pass
    output_dir.mkdir(parents=True, exist_ok=False)
    results: list[GenerationProbeResult] = []
    policies: tuple[Literal["baseline", "text"], ...] = (
        ("baseline", "text") if include_baseline else ("text",)
    )
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as report:
        for case in cases:
            for seed in seeds:
                baseline: GenerationProbeResult | None = None
                pair_start = len(results)
                for policy in policies:
                    request = request_template.model_copy(
                        update={
                            "input": case.input,
                            "language": case.language,
                            "instructions": case.instructions
                            or request_template.instructions,
                            "references": case.references
                            or request_template.references,
                            "seed": seed,
                            "stream": stream,
                            "stream_format": "audio",
                            "response_format": "pcm" if stream else "wav",
                            "speed": 1.0,
                            "max_new_tokens": (
                                DEFAULT_MAX_NEW_TOKENS if policy == "baseline" else None
                            ),
                        }
                    )
                    result = GenerationProbeResult(
                        case_id=case.id,
                        request=request,
                        policy=policy,
                    )
                    try:
                        await asyncio.wait_for(
                            synthesize_probe(
                                client,
                                base_url,
                                result,
                                output_dir / f"{len(results):05d}.wav",
                            ),
                            timeout=request_timeout_seconds,
                        )
                    except asyncio.TimeoutError:
                        result.errors.append(
                            "Speech request exceeded its total time limit"
                        )
                    if policy == "baseline":
                        baseline = result
                    else:
                        pass
                    # note(Alistairshi): Seeded pairs can diverge with prefix-cache or batch changes.
                    if (
                        policy == "text"
                        and baseline is not None
                        and baseline.finish_reason == "stop"
                        and result.finish_reason == "length"
                        and baseline.completion_frames is not None
                        and result.completion_frames is not None
                        and result.completion_frames < baseline.completion_frames
                    ):
                        result.errors.append(
                            "Text budget is shorter than a naturally completed baseline"
                        )
                    else:
                        pass
                    results.append(result)
                    report.write(result.model_dump_json() + "\n")
                    report.flush()
                if not keep_passing_audio and not any(
                    result.errors for result in results[pair_start:]
                ):
                    for index in range(pair_start, len(results)):
                        audio_path = output_dir / f"{index:05d}.wav"
                        audio_path.unlink()
                        if stream:
                            audio_path.with_suffix(".pcm").unlink()
                        else:
                            pass
                else:
                    pass
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", required=True, help="Speech service URL including /v1"
    )
    parser.add_argument(
        "--request",
        type=Path,
        required=True,
        help="Speech request JSON with model and voice/reference settings",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        help="JSON list of probe cases that replaces the built-in short lines",
    )
    parser.add_argument(
        "--seedtts-samples",
        action="store_true",
        help=(
            "Probe the SeedTTS prompts that never emitted codec EOS on 0.6B Base "
            "(#1179) instead. Each case keeps its own reference clip, staged on "
            "this server."
        ),
    )
    parser.add_argument("--seedtts-split", choices=("en", "zh"), default="en")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument(
        "--timeout-seconds", type=float, default=PROBE_REQUEST_TIMEOUT_SECONDS
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--server-revision", required=True)
    parser.add_argument("--model-revision", required=True)
    arguments = parser.parse_args()
    if arguments.repeats <= 0 or arguments.timeout_seconds <= 0:
        parser.error("repeats and timeout-seconds must be positive")
    else:
        pass
    request = CreateSpeechRequest.model_validate_json(arguments.request.read_text())
    if arguments.seedtts_samples:
        cases = seedtts_probe_cases(
            list(SEEDTTS_RUNAWAY_SAMPLE_IDS), arguments.seedtts_split
        )
    elif arguments.cases is not None:
        cases = TypeAdapter(list[GenerationProbeCase]).validate_json(
            arguments.cases.read_text()
        )
    else:
        cases = list(PROBE_CASES)
    api_key = os.environ.get(arguments.api_key_env)

    async def run() -> list[GenerationProbeResult]:
        async with httpx.AsyncClient(
            timeout=arguments.timeout_seconds,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        ) as client:
            return await run_generation_probe(
                client,
                arguments.base_url,
                request,
                cases,
                range(arguments.seed, arguments.seed + arguments.repeats),
                arguments.stream,
                arguments.output_dir,
                include_baseline=not arguments.skip_baseline,
                request_timeout_seconds=arguments.timeout_seconds,
            )

    results = asyncio.run(run())
    summary = {
        "server_revision": arguments.server_revision,
        "model_revision": arguments.model_revision,
        "requests": len(results),
        "failed": sum(bool(result.errors) for result in results),
        "length_stops": {
            policy: sum(
                result.policy == policy and result.finish_reason == "length"
                for result in results
            )
            for policy in ("baseline", "text")
        },
        "quality_validation": "Requires transcript comparison and listening to saved audio",
    }
    (arguments.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    raise SystemExit(1 if summary["failed"] else 0)


def seedtts_probe_cases(
    sample_ids: list[str], split: Literal["en", "zh"]
) -> list[GenerationProbeCase]:
    """Build cases with each SeedTTS sample's reference clip and transcript."""
    samples = {
        sample.sample_id: sample
        for sample in load_seedtts_samples(SEEDTTS_DATASET_ID, split=split)
    }
    missing = [sample_id for sample_id in sample_ids if sample_id not in samples]
    if not sample_ids or missing:
        raise ValueError(f"SeedTTS {split} has no samples {missing or sample_ids}")
    else:
        pass
    return [
        GenerationProbeCase(
            id=sample_id,
            input=samples[sample_id].target_text,
            language=SEEDTTS_SPLIT_LANGUAGES[split],
            references=[
                SpeechReference(
                    audio_path=samples[sample_id].ref_audio,
                    text=samples[sample_id].ref_text,
                )
            ],
        )
        for sample_id in sample_ids
    ]


if __name__ == "__main__":
    main()
