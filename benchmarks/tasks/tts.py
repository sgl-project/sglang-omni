# SPDX-License-Identifier: Apache-2.0
"""TTS task utilities: voice-clone API clients, ASR evaluation, and HTTP send functions.

Replaces tasks/tts_speed.py and tasks/voice_clone.py.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import csv
import functools
import io
import json
import logging
import os
import string
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import aiohttp
import soundfile as sf
import torch
import transformers
from jiwer import process_words
from tqdm import tqdm

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import SendFn
from benchmarks.benchmarker.utils import (
    SSE_DATA_PREFIX,
    SSE_DONE_MARKER,
    WAV_HEADER_SIZE,
    get_wav_duration,
    parse_sse_event,
    process_sse_line,
    save_json_results,
)
from benchmarks.dataset.seedtts import SampleInput
from benchmarks.metrics.performance import build_speed_results
from benchmarks.metrics.wer import (
    calculate_asr_speed_metrics,
    calculate_wer_metrics,
    print_asr_speed_summary,
    print_wer_summary,
)

logger = logging.getLogger(__name__)

TEXT_PREVIEW_LENGTH = 60


# ---------------------------------------------------------------------------
# ASR / WER utilities
# ---------------------------------------------------------------------------


@dataclass
class SampleOutput:
    sample_id: str = ""
    target_text: str = ""
    whisper_text: str = ""
    ref_norm: str = ""
    hyp_norm: str = ""
    wer: float = 0.0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0
    audio_duration_s: float = 0.0
    latency_s: float = 0.0
    asr_latency_s: float = 0.0
    is_success: bool = False
    error: str = ""


@functools.lru_cache(maxsize=1)
def _get_en_normalizer():
    """Lazy-load the English text normalizer.

    Tries whisper_normalizer (standalone pip package) first, then openai-whisper,
    then the transformers built-in normalizer.

    note (Chenyang): The three fallbacks exist because our deployments don't always
    have whisper_normalizer installed, whisper's own normalizer lives under a
    different path depending on the release, and on minimal CI images we rely on
    the transformers copy bundled with the library.  Keeping all three paths lets
    the WER numbers stay stable across environments (the official seed-tts-eval
    reference uses whisper_normalizer, so we prefer it when available).
    """
    try:
        from whisper_normalizer.english import EnglishTextNormalizer

        normalizer = EnglishTextNormalizer()
        logger.info("Using whisper_normalizer.english.EnglishTextNormalizer")
        return normalizer
    except ImportError:
        logger.debug("whisper_normalizer.english.EnglishTextNormalizer failed")

    try:
        from whisper.normalizers import EnglishTextNormalizer

        normalizer = EnglishTextNormalizer()
        logger.info("Using whisper.normalizers.EnglishTextNormalizer")
        return normalizer
    except ImportError:
        logger.debug("whisper.normalizers.EnglishTextNormalizer failed")

    try:
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

        json_path = (
            Path(transformers.__file__).parent / "models" / "whisper" / "english.json"
        )
        with open(json_path) as f:
            english_spelling_mapping = json.load(f)

        normalizer = EnglishTextNormalizer(english_spelling_mapping)
        logger.info(
            "Using transformers.models.whisper.english_normalizer.EnglishTextNormalizer"
        )
        return normalizer
    except (ImportError, FileNotFoundError) as exc:
        logger.debug(f"transformers EnglishTextNormalizer failed: {exc}")

    logger.warning(
        "EnglishTextNormalizer not found; falling back to punctuation-strip normalizer."
    )
    return None


def normalize_text(text: str, lang: str) -> str:
    if lang == "zh":
        from zhon.hanzi import punctuation as zh_punct

        all_punct = zh_punct + string.punctuation
        for ch in all_punct:
            if ch == "'":
                continue
            text = text.replace(ch, "")
        text = text.replace(" ", "").replace("\u3000", "").strip()
        text = " ".join(list(text))
        return text

    normalizer = _get_en_normalizer()
    if normalizer is not None:
        return normalizer(text)

    for ch in string.punctuation:
        if ch == "'":
            continue
        text = text.replace(ch, "")
    text = text.replace("  ", " ").strip().lower()
    return text


def load_asr_model(lang: str, device: str, generation_mode: str | None = None):
    """Load ASR model for voice clone WER evaluation."""
    mode_suffix = f" for {generation_mode} generation" if generation_mode else ""
    if lang == "en":
        from transformers import pipeline

        logger.info(
            f"Loading Whisper-large-v3 on {device}{mode_suffix} "
            "(pipeline, chunk_length_s=30)..."
        )
        t0 = time.perf_counter()
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
            chunk_length_s=30,
            device=device,
        )
        logger.info(f"Whisper loaded in {time.perf_counter() - t0:.1f}s{mode_suffix}")
        return {"type": "whisper", "pipeline": pipe}
    elif lang == "zh":
        from funasr import AutoModel

        logger.info(f"Loading FunASR paraformer-zh{mode_suffix}...")
        t0 = time.perf_counter()
        model = AutoModel(model="paraformer-zh")
        logger.info(f"FunASR loaded in {time.perf_counter() - t0:.1f}s{mode_suffix}")
        return {"type": "funasr", "model": model}
    else:
        raise ValueError(f"Unsupported language: {lang}")


def transcribe(asr: dict, wav_path: str, lang: str, device: str) -> str:
    if asr["type"] == "whisper":
        # pipeline internally chunks at chunk_length_s=30 with stride overlap,
        # so outputs longer than 30 s are transcribed end-to-end rather than
        # silently truncated to the first 30 s.
        result = asr["pipeline"](
            wav_path,
            generate_kwargs={"language": "english", "task": "transcribe"},
        )
        return result["text"]
    elif asr["type"] == "funasr":
        import zhconv

        res = asr["model"].generate(input=wav_path, batch_size_s=300)
        transcription = res[0]["text"]
        return zhconv.convert(transcription, "zh-cn")
    else:
        raise ValueError(f"Unknown ASR type: {asr['type']}")


def transcribe_and_compute_wer(
    output: SampleOutput,
    wav_path: str,
    asr: dict,
    lang: str,
    device: str,
) -> SampleOutput:
    """Transcribe audio and compute per-sample WER metrics."""
    try:
        hyp_text = transcribe(asr, wav_path, lang, device)
    except Exception as exc:
        output.error = f"Transcription failed: {exc}"
        logger.error(f"[{output.sample_id}] {output.error}")
        return output

    output.whisper_text = hyp_text
    output.ref_norm = normalize_text(output.target_text, lang)
    output.hyp_norm = normalize_text(hyp_text, lang)

    if not output.ref_norm:
        output.error = "Empty reference after normalization"
        return output

    measures = process_words(output.ref_norm, output.hyp_norm)
    output.wer = measures.wer
    output.substitutions = measures.substitutions
    output.deletions = measures.deletions
    output.insertions = measures.insertions
    output.hits = measures.hits
    output.is_success = True
    return output


def compute_text_audio_consistency(
    request_results: list[RequestResult],
    lang: str,
    asr_device: str,
) -> dict:
    """WER between each request's text output (ref) and ASR-transcribed audio (hyp)."""
    asr = load_asr_model(lang, asr_device)

    outputs: list[SampleOutput] = []
    for result in request_results:
        ref_text = " ".join(result.text.split())
        out = SampleOutput(
            sample_id=result.request_id,
            target_text=ref_text,
            latency_s=result.latency_s,
            audio_duration_s=result.audio_duration_s,
        )
        if not result.is_success or not result.wav_path:
            out.error = result.error or "No audio in response"
            outputs.append(out)
            continue
        outputs.append(
            transcribe_and_compute_wer(out, result.wav_path, asr, lang, asr_device)
        )

    per_sample = [
        {
            "id": o.sample_id,
            "is_success": o.is_success,
            "wer": o.wer if o.is_success else None,
            "ref_text": o.target_text[:100],
            "hyp_text": o.whisper_text[:100],
            "ref_norm": o.ref_norm,
            "hyp_norm": o.hyp_norm,
            "audio_duration_s": o.audio_duration_s,
            "error": o.error,
        }
        for o in outputs
    ]
    return {"summary": calculate_wer_metrics(outputs, lang), "per_sample": per_sample}


def save_wer_results(
    outputs: list[SampleOutput], metrics: dict, config: dict, output_dir: str
) -> None:
    json_results = {
        "summary": metrics,
        "config": config,
        "per_sample": [
            {
                "id": o.sample_id,
                "target_text": o.target_text,
                "whisper_text": o.whisper_text,
                "ref_norm": o.ref_norm,
                "hyp_norm": o.hyp_norm,
                "wer": round(o.wer, 6) if o.is_success else None,
                "substitutions": o.substitutions if o.is_success else None,
                "deletions": o.deletions if o.is_success else None,
                "insertions": o.insertions if o.is_success else None,
                "hits": o.hits if o.is_success else None,
                "audio_duration_s": round(o.audio_duration_s, 4),
                "latency_s": round(o.latency_s, 4),
                "is_success": o.is_success,
                "error": o.error or None,
            }
            for o in outputs
        ],
    }
    save_json_results(json_results, output_dir, "wer_results.json")

    csv_path = os.path.join(output_dir, "wer_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "target_text",
                "whisper_text",
                "wer",
                "substitutions",
                "deletions",
                "insertions",
                "hits",
                "audio_duration_s",
                "latency_s",
                "is_success",
                "error",
            ]
        )
        for o in outputs:
            writer.writerow(
                [
                    o.sample_id,
                    o.target_text,
                    o.whisper_text,
                    f"{o.wer:.6f}" if o.is_success else "",
                    o.substitutions if o.is_success else "",
                    o.deletions if o.is_success else "",
                    o.insertions if o.is_success else "",
                    o.hits if o.is_success else "",
                    f"{o.audio_duration_s:.4f}",
                    f"{o.latency_s:.4f}",
                    o.is_success,
                    o.error or "",
                ]
            )


# ---------------------------------------------------------------------------
# Shared transcribe pipeline (seed-tts-eval style)
# ---------------------------------------------------------------------------


class ServerEndpointConfig(Protocol):
    """Subset used by :func:`build_base_url` to resolve a server endpoint."""

    base_url: str | None
    host: str
    port: int


class SeedttsTranscribeConfig(Protocol):
    """Subset of config fields the shared transcribe pipeline reads.

    Kept narrow on purpose: ``run_seedtts_transcribe`` does not touch any
    server fields, so callers whose configs lack ``host``/``port`` can still
    satisfy this Protocol.
    """

    model: str
    output_dir: str
    lang: str
    device: str


def build_base_url(config: ServerEndpointConfig) -> str:
    """Resolve the server base URL from an explicit override or host/port."""
    return config.base_url or f"http://{config.host}:{config.port}"


def _transcribe_one_entry(
    entry: dict,
    asr: dict,
    lang: str,
    device: str,
) -> SampleOutput:
    """Transcribe a single ``generated.json`` entry and compute its WER."""
    output = SampleOutput(
        sample_id=entry["sample_id"],
        target_text=entry["target_text"],
    )
    if not entry.get("is_success", False):
        output.error = f"Generation failed: {entry.get('error', 'unknown')}"
        return output

    output.latency_s = entry.get("latency_s", 0.0)
    output.audio_duration_s = entry.get("audio_duration_s", 0.0)
    asr_t0 = time.perf_counter()
    output = transcribe_and_compute_wer(output, entry["wav_path"], asr, lang, device)
    output.asr_latency_s = time.perf_counter() - asr_t0
    return output


def _log_transcribe_result(
    *,
    idx: int,
    total: int,
    entry: dict,
    output: SampleOutput,
    log_per_sample: bool,
) -> None:
    if output.is_success:
        if log_per_sample:
            logger.info(
                f"[{idx + 1}/{total}] "
                f"WER={output.wer:.3f}  "
                f"asr={output.asr_latency_s:.3f}s  "
                f"ref={output.ref_norm[:50]}  "
                f"hyp={output.hyp_norm[:50]}",
            )
        return

    # Only warn for post-generation transcription failures; generation
    # failures are surfaced at speed-benchmark time and already logged.
    if entry.get("is_success", False):
        logger.warning(
            f"[{idx + 1}/{total}] Transcription failed: "
            f"{entry['sample_id']} -- {output.error}",
        )


def run_seedtts_transcribe(
    config: SeedttsTranscribeConfig,
    *,
    wer_config: dict,
    generation_mode: str | None = None,
    log_per_sample: bool = False,
) -> dict:
    """Transcribe saved audio, compute WER + ASR-speed metrics, and persist them.

    Shared pipeline used by both Qwen3-Omni and S2-Pro seed-tts-eval benchmarks.
    The caller-specific ``wer_config`` dict is embedded in ``wer_results.json``
    to preserve backward-compatible fields.

    Returns a dict with keys:
        - ``wer_summary``: corpus-level WER metrics (see :func:`calculate_wer_metrics`)
        - ``asr_speed``:   ASR transcription latency/throughput metrics
        - ``per_sample``:  list[SampleOutput] with per-sample details
    """
    if "cuda" in config.device:
        torch.cuda.set_device(config.device)
        logger.info(f"Set ASR CUDA device to {config.device}")

    generated_path = os.path.join(config.output_dir, "generated.json")
    with open(generated_path) as f:
        generated: list[dict] = json.load(f)
    logger.info(f"Loaded {len(generated)} entries from {generated_path}")

    asr = load_asr_model(config.lang, config.device, generation_mode)

    tqdm_desc = (
        f"Transcribing ({config.lang})" if not generation_mode else "WER transcribe"
    )
    outputs: list[SampleOutput] = []
    for idx, entry in enumerate(tqdm(generated, desc=tqdm_desc)):
        output = _transcribe_one_entry(entry, asr, config.lang, config.device)
        outputs.append(output)
        _log_transcribe_result(
            idx=idx,
            total=len(generated),
            entry=entry,
            output=output,
            log_per_sample=log_per_sample,
        )

    wer_metrics = calculate_wer_metrics(outputs, config.lang)
    asr_metrics = calculate_asr_speed_metrics(outputs)

    print_asr_speed_summary(asr_metrics, config.model)
    print_wer_summary(wer_metrics, config.model, generation_mode)

    save_wer_results(outputs, wer_metrics, wer_config, config.output_dir)
    save_json_results(asr_metrics, config.output_dir, "asr_speed_results.json")

    return {
        "wer_summary": wer_metrics,
        "asr_speed": asr_metrics,
        "per_sample": outputs,
    }


# ---------------------------------------------------------------------------
# Voice-clone API clients
# ---------------------------------------------------------------------------


class VoiceCloneTTS:
    """Voice cloning via /v1/audio/speech (OAI TTS API format)."""

    async def generate_speech(
        self,
        session: aiohttp.ClientSession,
        api_url: str,
        model_name: str,
        sample: SampleInput,
        max_new_tokens: int = 2048,
        temperature: float = 0.8,
        seed: int | None = None,
    ) -> tuple[bytes, float]:
        payload: dict = {
            "model": model_name,
            "input": sample.target_text,
            "ref_audio": sample.ref_audio,
            "ref_text": sample.ref_text,
            "response_format": "wav",
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if seed is not None:
            payload["seed"] = seed

        t0 = time.perf_counter()
        async with session.post(api_url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {error_text}")
            wav_bytes = await response.read()
        latency = time.perf_counter() - t0

        if len(wav_bytes) <= WAV_HEADER_SIZE:
            raise ValueError(
                f"Empty or invalid audio response ({len(wav_bytes)} bytes)"
            )
        return wav_bytes, latency

    async def generate_speech_streaming(
        self,
        session: aiohttp.ClientSession,
        api_url: str,
        model_name: str,
        sample: SampleInput,
        max_new_tokens: int = 2048,
        temperature: float = 0.8,
        seed: int | None = None,
    ) -> tuple[bytes, float]:
        """Generate speech via streaming SSE, concatenate audio chunks into WAV."""
        payload: dict = {
            "model": model_name,
            "input": sample.target_text,
            "ref_audio": sample.ref_audio,
            "ref_text": sample.ref_text,
            "response_format": "wav",
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if seed is not None:
            payload["seed"] = seed

        t0 = time.perf_counter()
        pcm_chunks: list[bytes] = []
        sample_rate = None
        num_channels = None
        sample_width = None

        async with session.post(api_url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {error_text}")

            buffer = bytearray()
            async for chunk in response.content.iter_any():
                buffer.extend(chunk)
                while b"\n" in buffer:
                    idx = buffer.index(b"\n")
                    raw_line = bytes(buffer[:idx])
                    del buffer[: idx + 1]
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith(SSE_DATA_PREFIX) or line == SSE_DONE_MARKER:
                        continue
                    try:
                        event = json.loads(line[len(SSE_DATA_PREFIX) :])
                    except json.JSONDecodeError:
                        continue
                    audio = event.get("audio")
                    if not isinstance(audio, dict) or not audio.get("data"):
                        continue
                    chunk_bytes = base64.b64decode(audio["data"])
                    if len(chunk_bytes) <= WAV_HEADER_SIZE:
                        continue
                    try:
                        with io.BytesIO(chunk_bytes) as buf:
                            with wave.open(buf, "rb") as wf:
                                sr = wf.getframerate()
                                ch = wf.getnchannels()
                                sw = wf.getsampwidth()
                                pcm = wf.readframes(wf.getnframes())
                        if sample_rate is None:
                            sample_rate, num_channels, sample_width = sr, ch, sw
                        pcm_chunks.append(pcm)
                    except Exception:
                        continue

            if buffer.strip():
                line = bytes(buffer).decode("utf-8", errors="replace").strip()
                if line.startswith(SSE_DATA_PREFIX) and line != SSE_DONE_MARKER:
                    try:
                        event = json.loads(line[len(SSE_DATA_PREFIX) :])
                        audio = event.get("audio")
                        if isinstance(audio, dict) and audio.get("data"):
                            chunk_bytes = base64.b64decode(audio["data"])
                            if len(chunk_bytes) > WAV_HEADER_SIZE:
                                with io.BytesIO(chunk_bytes) as buf:
                                    with wave.open(buf, "rb") as wf:
                                        pcm = wf.readframes(wf.getnframes())
                                        if sample_rate is None:
                                            sample_rate = wf.getframerate()
                                            num_channels = wf.getnchannels()
                                            sample_width = wf.getsampwidth()
                                pcm_chunks.append(pcm)
                    except (json.JSONDecodeError, Exception) as exc:
                        logger.debug(
                            "Failed to parse trailing SSE audio chunk: %s", exc
                        )

        latency = time.perf_counter() - t0

        if not pcm_chunks or sample_rate is None:
            raise ValueError("No audio chunks received from streaming response")

        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(pcm_chunks))
        return wav_buf.getvalue(), latency

    async def evaluate_sample(
        self,
        session: aiohttp.ClientSession,
        api_url: str,
        model_name: str,
        asr: dict,
        sample: SampleInput,
        lang: str,
        device: str,
        audio_dir: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.8,
        seed: int | None = None,
        stream: bool = False,
    ) -> SampleOutput:
        output = SampleOutput(
            sample_id=sample.sample_id,
            target_text=sample.target_text,
        )
        wav_path = os.path.join(audio_dir, f"{sample.sample_id}.wav")

        try:
            gen_fn = self.generate_speech_streaming if stream else self.generate_speech
            wav_bytes, latency = await gen_fn(
                session, api_url, model_name, sample, max_new_tokens, temperature, seed
            )
            with open(wav_path, "wb") as f:
                f.write(wav_bytes)
            output.latency_s = round(latency, 4)
            output.audio_duration_s = round(sf.info(wav_path).duration, 4)
        except Exception as exc:
            output.error = f"Generation failed: {exc}"
            logger.error(f"[{sample.sample_id}] {output.error}")
            return output

        return transcribe_and_compute_wer(output, wav_path, asr, lang, device)


class VoiceCloneOmni:
    """Voice cloning via /v1/chat/completions (Omni API format).

    Shared by Qwen3 Omni and future Omni models.
    """

    THINKER_MAX_NEW_TOKENS = 256

    async def generate_speech(
        self,
        session: aiohttp.ClientSession,
        api_url: str,
        model_name: str,
        sample: SampleInput,
        lang: str,
        speaker: str = "Ethan",
        max_tokens: int | None = None,
        temperature: float = 0.7,
        voice_clone: bool = False,
    ) -> tuple[bytes, float, dict]:
        if max_tokens is None:
            max_tokens = self.THINKER_MAX_NEW_TOKENS

        if voice_clone:
            if lang == "en":
                prompt_text = (
                    f'Listen to the audio above. The speaker is reading: "{sample.ref_text}". '
                    f"Now please read the following text out loud in the same voice and style: "
                    f"{sample.target_text}"
                )
            else:
                prompt_text = (
                    f'听上面的音频，说话人正在朗读："{sample.ref_text}"。'
                    f"现在请用同样的声音和风格朗读以下文本：{sample.target_text}"
                )
        else:
            if lang == "en":
                prompt_text = (
                    f"Please read the following text out loud in English: "
                    f"{sample.target_text}"
                )
            else:
                prompt_text = f"请用中文朗读以下文本: {sample.target_text}"

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "modalities": ["text", "audio"],
            "audio": {"format": "wav"},
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if voice_clone:
            payload["audios"] = [sample.ref_audio]

        t0 = time.perf_counter()
        async with session.post(api_url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {error_text}")
            resp_json = await response.json()
        latency = time.perf_counter() - t0

        choices = resp_json.get("choices", [])
        if not choices:
            raise ValueError("No choices in response")

        message = choices[0].get("message", {})
        audio_obj = message.get("audio")
        if audio_obj is None:
            raise ValueError(
                f"No audio in response for sample '{sample.sample_id}'. "
                f"Text response: {message.get('content', 'N/A')[:100]}"
            )

        audio_b64 = audio_obj.get("data")
        if not audio_b64:
            raise ValueError("Empty audio data in response")

        wav_bytes = base64.b64decode(audio_b64)
        usage = resp_json.get("usage", {})
        return wav_bytes, latency, usage

    async def evaluate_sample(
        self,
        session: aiohttp.ClientSession,
        api_url: str,
        model_name: str,
        asr: dict,
        sample: SampleInput,
        lang: str,
        asr_device: str,
        audio_dir: str,
        speaker: str = "Ethan",
        max_tokens: int | None = None,
        voice_clone: bool = False,
    ) -> SampleOutput:
        output = SampleOutput(
            sample_id=sample.sample_id,
            target_text=sample.target_text,
        )
        wav_path = os.path.join(audio_dir, f"{sample.sample_id}.wav")

        try:
            wav_bytes, latency, _usage = await self.generate_speech(
                session,
                api_url,
                model_name,
                sample,
                lang,
                speaker,
                max_tokens,
                voice_clone=voice_clone,
            )
            with open(wav_path, "wb") as f:
                f.write(wav_bytes)
            output.latency_s = round(latency, 4)
            output.audio_duration_s = round(sf.info(wav_path).duration, 4)
        except Exception as exc:
            output.error = f"Generation failed: {exc}"
            logger.error(f"[{sample.sample_id}] {output.error}")
            return output

        return transcribe_and_compute_wer(output, wav_path, asr, lang, asr_device)


# ---------------------------------------------------------------------------
# TTS HTTP send layer  (/v1/audio/speech)
# ---------------------------------------------------------------------------


def _build_tts_payload(
    sample: SampleInput,
    model_name: str,
    *,
    stream: bool = False,
    no_ref_audio: bool = False,
    voice: str | None = None,
    **gen_kwargs,
) -> dict:
    payload: dict = {
        "model": model_name,
        "input": sample.target_text,
        "response_format": "wav",
    }
    if not no_ref_audio:
        payload["ref_audio"] = sample.ref_audio
        payload["ref_text"] = sample.ref_text
    if voice is not None:
        payload["voice"] = voice
    for key, value in gen_kwargs.items():
        if value is not None:
            payload[key] = value
    if stream:
        payload["stream"] = True
    return payload


def _parse_response_headers(result: RequestResult, headers: dict) -> None:
    prompt_tok = headers.get("X-Prompt-Tokens")
    comp_tok = headers.get("X-Completion-Tokens")
    eng_time = headers.get("X-Engine-Time")
    if prompt_tok is not None:
        result.prompt_tokens = int(prompt_tok)
    if comp_tok is not None:
        result.completion_tokens = int(comp_tok)
    if eng_time is not None:
        result.engine_time_s = float(eng_time)
    if result.completion_tokens > 0 and result.engine_time_s > 0:
        result.tok_per_s = result.completion_tokens / result.engine_time_s


async def _handle_streaming_response(
    response: aiohttp.ClientResponse,
    result: RequestResult,
    start_time: float,
    save_audio_dir: str | None,
) -> None:
    total_audio_duration = 0.0
    usage_data: dict | None = None
    buffer = bytearray()
    pcm_chunks: list[bytes] = []
    stream_format: tuple[int, int, int] | None = None
    async for chunk in response.content.iter_any():
        buffer.extend(chunk)
        while b"\n" in buffer:
            idx = buffer.index(b"\n")
            raw_line = bytes(buffer[:idx])
            del buffer[: idx + 1]
            line = raw_line.decode("utf-8", errors="replace").strip()
            total_audio_duration, usage_data = process_sse_line(
                line, total_audio_duration, usage_data
            )
            stream_format = _collect_streaming_audio(line, pcm_chunks, stream_format)
    if buffer.strip():
        line = bytes(buffer).decode("utf-8", errors="replace").strip()
        total_audio_duration, usage_data = process_sse_line(
            line, total_audio_duration, usage_data
        )
        stream_format = _collect_streaming_audio(line, pcm_chunks, stream_format)
    result.audio_duration_s = total_audio_duration
    if total_audio_duration > 0:
        elapsed = time.perf_counter() - start_time
        result.rtf = elapsed / total_audio_duration
    result.is_success = total_audio_duration > 0
    if save_audio_dir and pcm_chunks and stream_format is not None:
        audio_path = os.path.join(save_audio_dir, f"{result.request_id}.wav")
        with open(audio_path, "wb") as fh:
            fh.write(_build_streaming_wav_bytes(pcm_chunks, stream_format))
        result.wav_path = audio_path
    if usage_data:
        prompt_tok = usage_data.get("prompt_tokens")
        comp_tok = usage_data.get("completion_tokens")
        eng_time = usage_data.get("engine_time_s")
        if prompt_tok is not None:
            result.prompt_tokens = int(prompt_tok)
        if comp_tok is not None:
            result.completion_tokens = int(comp_tok)
        if eng_time is not None:
            result.engine_time_s = float(eng_time)
        if result.completion_tokens > 0 and result.engine_time_s > 0:
            result.tok_per_s = result.completion_tokens / result.engine_time_s


async def _handle_non_streaming_response(
    response: aiohttp.ClientResponse,
    result: RequestResult,
    start_time: float,
    save_audio_dir: str | None,
) -> None:
    audio_bytes = await response.read()
    result.audio_duration_s = get_wav_duration(audio_bytes)
    elapsed = time.perf_counter() - start_time
    if result.audio_duration_s > 0:
        result.is_success = True
        result.rtf = elapsed / result.audio_duration_s
    else:
        result.error = f"Empty or invalid audio response ({len(audio_bytes)} bytes)"
        return
    _parse_response_headers(result, response.headers)
    if save_audio_dir and audio_bytes:
        audio_path = os.path.join(save_audio_dir, f"{result.request_id}.wav")
        with open(audio_path, "wb") as fh:
            fh.write(audio_bytes)
        result.wav_path = audio_path


def _collect_streaming_audio(
    line: str,
    pcm_chunks: list[bytes],
    stream_format: tuple[int, int, int] | None,
) -> tuple[int, int, int] | None:
    event = parse_sse_event(line)
    if event is None:
        return stream_format

    audio = event.get("audio")
    if not isinstance(audio, dict) or not audio.get("data"):
        return stream_format

    try:
        chunk_bytes = base64.b64decode(audio["data"])
        if len(chunk_bytes) <= WAV_HEADER_SIZE:
            return stream_format
        with io.BytesIO(chunk_bytes) as buf:
            with wave.open(buf, "rb") as wf:
                pcm_chunks.append(wf.readframes(wf.getnframes()))
                if stream_format is not None:
                    return stream_format
                return (wf.getframerate(), wf.getnchannels(), wf.getsampwidth())
    except (binascii.Error, wave.Error, EOFError) as exc:
        logger.debug(f"Skipping malformed streaming audio chunk: {exc}")
        return stream_format


def _build_streaming_wav_bytes(
    pcm_chunks: list[bytes],
    stream_format: tuple[int, int, int],
) -> bytes:
    sample_rate, num_channels, sample_width = stream_format
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setframerate(sample_rate)
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width)
        wf.writeframes(b"".join(pcm_chunks))
    return wav_buffer.getvalue()


def make_tts_send_fn(
    model_name: str,
    api_url: str,
    *,
    stream: bool = False,
    no_ref_audio: bool = False,
    voice: str | None = None,
    save_audio_dir: str | None = None,
    **gen_kwargs,
) -> SendFn:
    """Return a *send_fn(session, sample) -> RequestResult* for the runner."""

    async def send_fn(
        session: aiohttp.ClientSession, sample: SampleInput
    ) -> RequestResult:
        result = RequestResult(
            request_id=sample.sample_id,
            text=sample.target_text[:TEXT_PREVIEW_LENGTH],
        )
        payload = _build_tts_payload(
            sample,
            model_name,
            stream=stream,
            no_ref_audio=no_ref_audio,
            voice=voice,
            **gen_kwargs,
        )
        start_time = time.perf_counter()
        try:
            async with session.post(api_url, json=payload) as response:
                if response.status != 200:
                    result.error = f"HTTP {response.status}: {await response.text()}"
                elif stream:
                    await _handle_streaming_response(
                        response, result, start_time, save_audio_dir
                    )
                else:
                    await _handle_non_streaming_response(
                        response, result, start_time, save_audio_dir
                    )
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            result.error = str(exc)
        finally:
            result.latency_s = time.perf_counter() - start_time
        return result

    return send_fn


def save_generated_audio_metadata(
    outputs: list[RequestResult],
    samples: list[SampleInput],
    output_dir: str,
) -> None:
    sample_by_id = {sample.sample_id: sample for sample in samples}
    generated = [
        _request_result_to_generated_entry(output, sample_by_id[output.request_id])
        for output in outputs
    ]
    metadata_path = os.path.join(output_dir, "generated.json")
    with open(metadata_path, "w") as fh:
        json.dump(generated, fh, indent=2, ensure_ascii=False)
    logger.info(f"Generated audio metadata saved to {metadata_path}")


def _request_result_to_generated_entry(
    output: RequestResult,
    sample: SampleInput,
) -> dict:
    entry: dict = {
        "sample_id": output.request_id,
        "target_text": sample.target_text,
        "wav_path": output.wav_path,
        "is_success": output.is_success,
        "latency_s": round(output.latency_s, 4),
        "audio_duration_s": round(output.audio_duration_s, 4),
    }
    if output.error:
        entry["error"] = output.error
    return entry


def save_speed_results(
    outputs: list[RequestResult],
    metrics: dict,
    config: dict,
    output_dir: str,
) -> None:
    json_results = build_speed_results(outputs, metrics, config)
    save_json_results(json_results, output_dir, "speed_results.json")

    csv_path = os.path.join(output_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "text",
                "latency_s",
                "audio_duration_s",
                "rtf",
                "prompt_tokens",
                "completion_tokens",
                "tok_per_s",
                "is_success",
                "error",
            ]
        )
        for o in outputs:
            writer.writerow(
                [
                    o.request_id,
                    o.text,
                    f"{o.latency_s:.4f}",
                    f"{o.audio_duration_s:.4f}",
                    f"{o.rtf:.4f}" if o.rtf < float("inf") else "",
                    o.prompt_tokens or "",
                    o.completion_tokens or "",
                    f"{o.tok_per_s:.1f}" if o.tok_per_s > 0 else "",
                    o.is_success,
                    o.error or "",
                ]
            )
    logger.info(f"Results saved to {output_dir}")
