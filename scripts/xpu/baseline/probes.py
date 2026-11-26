# SPDX-License-Identifier: Apache-2.0
"""One request per probe kind and what is scored from the answer."""

from __future__ import annotations

import io
import math
import re
import time
import wave
from typing import Protocol

import requests

from scripts.xpu.baseline.models import (
    ARITHMETIC_PROMPT,
    MODEL_BY_KEY,
    MUSIC_PROMPT,
    REFERENCE_AUDIO,
    REFERENCE_TEXT,
    SPOKEN_TEXT,
    BaselineModel,
    ProbeKind,
    Sample,
)


class Probe(Protocol):
    def __call__(self, base_url: str, model: BaselineModel) -> Sample: ...


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", text).strip().lower())


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level edit distance over reference length; jiwer is not in the XPU install."""
    want = normalize_text(reference).split()
    got = normalize_text(hypothesis).split()
    if not want:
        return 0.0 if not got else 1.0
    else:
        previous = list(range(len(got) + 1))
    for i, want_word in enumerate(want, start=1):
        current = [i]
        for j, got_word in enumerate(got, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (want_word != got_word),
                )
            )
        previous = current
    return previous[-1] / len(want)


def audio_shape(raw: bytes) -> dict[str, float]:
    """Frame count, rate, length and loudness of a WAV body."""
    with wave.open(io.BytesIO(raw)) as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        width = handle.getsampwidth()
        pcm = handle.readframes(frames)

    if frames == 0 or rate == 0:
        raise AssertionError("audio holds no frames")
    else:
        scale = float(1 << (8 * width - 1))
    samples = memoryview(pcm).cast({1: "b", 2: "h", 4: "i"}[width])
    peak = max(abs(sample) for sample in samples) / scale
    rms = math.sqrt(sum((sample / scale) ** 2 for sample in samples) / len(samples))
    if peak <= 0.01:
        raise AssertionError(
            f"audio peaks at {peak:.4f} of full scale, which is silence"
        )
    else:
        pass
    return {
        "frames": float(frames),
        "sample_rate": float(rate),
        "audio_seconds": frames / rate,
        "peak": peak,
        "rms": rms,
    }


def transcribe(base_url: str, model_path: str, audio: bytes, filename: str) -> str:
    response = requests.post(
        f"{base_url}/v1/audio/transcriptions",
        files={"file": (filename, audio, "audio/wav")},
        data={"model": model_path, "temperature": "0"},
        timeout=300,
    )
    response.raise_for_status()
    return response.json().get("text", "")


def probe_transcription(base_url: str, model: BaselineModel) -> Sample:
    audio = REFERENCE_AUDIO.read_bytes()
    started = time.perf_counter()
    transcript = transcribe(base_url, model.model_path, audio, REFERENCE_AUDIO.name)
    elapsed = time.perf_counter() - started
    if not transcript.strip():
        raise AssertionError("empty transcript")
    else:
        reference_seconds = audio_shape(audio)["audio_seconds"]
    return {
        "exact": {"transcript": normalize_text(transcript)},
        "atmost": {"word_error_rate": word_error_rate(REFERENCE_TEXT, transcript)},
        "speed": {
            "latency_s": elapsed,
            "audio_seconds_per_wall_second": reference_seconds / elapsed,
        },
    }


def synthesize(
    base_url: str, model: BaselineModel, text: str, timeout_seconds: int
) -> tuple[bytes, float]:
    request_body = {
        "model": model.model_path,
        "input": text,
        "response_format": "wav",
        **({} if model.seedless else {"seed": 0}),
        **model.payload,
    }
    started = time.perf_counter()
    response = requests.post(
        f"{base_url}/v1/audio/speech", json=request_body, timeout=timeout_seconds
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    return response.content, elapsed


def probe_speech(base_url: str, model: BaselineModel, judge_url: str) -> Sample:
    audio, elapsed = synthesize(base_url, model, SPOKEN_TEXT, timeout_seconds=600)
    shape = audio_shape(audio)
    result: Sample = {
        "exact": {"sample_rate": shape["sample_rate"]},
        "atmost": {},
        # Seeded tokens repeat exactly, but a kernel change moves the vocoder's floats, so loudness gets a band.
        "close": {"peak": shape["peak"], "rms": shape["rms"]},
        "speed": {
            "latency_s": elapsed,
            "audio_seconds_per_wall_second": shape["audio_seconds"] / elapsed,
        },
    }
    if model.seedless:
        # Unseeded, the model picks its own length, so length and everything derived from it are only observed.
        result["observed"] = {
            "frames": shape["frames"],
            "peak": result["close"].pop("peak"),
            "rms": result["close"].pop("rms"),
            "audio_seconds_per_wall_second": result["speed"].pop(
                "audio_seconds_per_wall_second"
            ),
        }
    else:
        result["exact"]["frames"] = shape["frames"]
    if judge_url:
        heard = transcribe(
            judge_url, MODEL_BY_KEY[model.judge].model_path, audio, "tts.wav"
        )
        result["atmost"]["round_trip_wer"] = word_error_rate(SPOKEN_TEXT, heard)
        result["heard"] = normalize_text(heard)
    else:
        pass
    return result


def probe_music(base_url: str, model: BaselineModel) -> Sample:
    audio, elapsed = synthesize(base_url, model, MUSIC_PROMPT, timeout_seconds=900)
    shape = audio_shape(audio)
    return {
        "exact": {"sample_rate": shape["sample_rate"]},
        "close": {"audio_seconds": shape["audio_seconds"], "rms": shape["rms"]},
        "speed": {"latency_s": elapsed},
    }


def probe_chat(base_url: str, model: BaselineModel) -> Sample:
    request_body = {
        "model": model.model_path,
        "messages": [{"role": "user", "content": ARITHMETIC_PROMPT}],
        "max_tokens": 32,
        "temperature": 0,
        "seed": 0,
    }
    started = time.perf_counter()
    response = requests.post(
        f"{base_url}/v1/chat/completions", json=request_body, timeout=600
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    completion = response.json()
    content = completion["choices"][0]["message"]["content"]
    if not content:
        raise AssertionError("no assistant content")
    else:
        completion_tokens = completion.get("usage", {}).get("completion_tokens", 0)
    speed = {"latency_s": elapsed}
    if completion_tokens:
        speed["decode_tokens_per_second"] = completion_tokens / elapsed
    else:
        pass
    return {"exact": {"content": content.strip()}, "speed": speed}


PROBES: dict[ProbeKind, Probe] = {
    "transcription": probe_transcription,
    "music": probe_music,
    "chat": probe_chat,
}


def take_sample(model: BaselineModel, base_url: str, judge_url: str) -> Sample:
    """One request; speech also takes the judge that transcribes it back."""
    if model.probe == "speech":
        return probe_speech(base_url, model, judge_url)
    else:
        return PROBES[model.probe](base_url, model)
