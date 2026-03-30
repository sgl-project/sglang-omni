# SPDX-License-Identifier: Apache-2.0
"""Voice Cloning WER case -- TTS API (/v1/audio/speech) and Omni API
(/v1/chat/completions) unified implementation."""

from __future__ import annotations

import base64
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

import aiohttp
import numpy as np
import scipy.signal
import soundfile as sf
import torch
import transformers
from jiwer import process_words

from benchmarks.benchmarker.utils import (
    SSE_DATA_PREFIX,
    SSE_DONE_MARKER,
    WAV_HEADER_SIZE,
)
from benchmarks.dataset.seedtts import SampleInput

logger = logging.getLogger(__name__)

SUMMARY_LABEL_WIDTH = 30
SUMMARY_LINE_WIDTH = 60


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
    is_success: bool = False
    error: str = ""


@functools.lru_cache(maxsize=1)
def _get_en_normalizer():
    """Lazy-load the English text normalizer.

    Tries whisper_normalizer (standalone pip package) first, then openai-whisper,
    then the transformers built-in normalizer.

    # Note (chenyang):

    For Chinese and English wer evaluation, the normalizer is critical.
    For human understanding, "1" is equal to "一" and "one". But without
    normalizer, these identical words will be treated as different words.

    Also, we prefer to import all the dependencies at the top of the file.
    But the normalizer is rather complex, so we use lazy import here.

    # TODO (chenyang):

    Refactor the normalizer load function and ensure the best normalizer is used.
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
        logger.debug("transformers EnglishTextNormalizer failed: %s", exc)

    logger.warning(
        "EnglishTextNormalizer not found in whisper_normalizer, whisper, "
        "or transformers; falling back to simple punctuation-strip normalizer."
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


def load_asr_model(lang: str, device: str):
    """Load ASR model for voice clone WER evaluation."""
    if lang == "en":
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        logger.info("Loading Whisper-large-v3 on %s...", device)
        t0 = time.perf_counter()
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3"
        ).to(device)
        logger.info("Whisper loaded in %.1fs", time.perf_counter() - t0)
        return {"type": "whisper", "processor": processor, "model": model}
    elif lang == "zh":
        from funasr import AutoModel

        logger.info("Loading FunASR paraformer-zh...")
        t0 = time.perf_counter()
        model = AutoModel(model="paraformer-zh")
        logger.info("FunASR loaded in %.1fs", time.perf_counter() - t0)
        return {"type": "funasr", "model": model}
    else:
        raise ValueError(f"Unsupported language: {lang}")


def transcribe(asr: dict, wav_path: str, lang: str, device: str) -> str:
    if asr["type"] == "whisper":
        processor = asr["processor"]
        model = asr["model"]
        wav, sr = sf.read(wav_path)
        if sr != 16000:
            wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
        input_features = processor(
            wav, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="english", task="transcribe"
        )
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features, forced_decoder_ids=forced_decoder_ids
            )
        return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    elif asr["type"] == "funasr":
        import zhconv

        res = asr["model"].generate(input=wav_path, batch_size_s=300)
        transcription = res[0]["text"]
        return zhconv.convert(transcription, "zh-cn")
    else:
        raise ValueError(f"Unknown ASR type: {asr['type']}")


def _transcribe_and_compute_wer(
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
        logger.error("[%s] %s", output.sample_id, output.error)
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

            # process any remaining bytes in buffer
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
                            f"Failed to parse or decode trailing SSE audio chunk: {exc}"
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
        wav_bytes = wav_buf.getvalue()

        return wav_bytes, latency

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
                session,
                api_url,
                model_name,
                sample,
                max_new_tokens,
                temperature,
                seed,
            )
            with open(wav_path, "wb") as f:
                f.write(wav_bytes)
            output.latency_s = round(latency, 4)
            wav_info = sf.info(wav_path)
            output.audio_duration_s = round(wav_info.duration, 4)
        except Exception as exc:
            output.error = f"Generation failed: {exc}"
            logger.error("[%s] %s", sample.sample_id, output.error)
            return output

        return _transcribe_and_compute_wer(output, wav_path, asr, lang, device)


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
    ) -> tuple[bytes, float]:
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
        return wav_bytes, latency

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
            wav_bytes, latency = await self.generate_speech(
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
            wav_info = sf.info(wav_path)
            output.audio_duration_s = round(wav_info.duration, 4)
        except Exception as exc:
            output.error = f"Generation failed: {exc}"
            logger.error("[%s] %s", sample.sample_id, output.error)
            return output

        return _transcribe_and_compute_wer(output, wav_path, asr, lang, asr_device)


def calculate_wer_metrics(outputs: list[SampleOutput], lang: str) -> dict:
    """Compute corpus-level WER metrics from per-sample outputs."""
    successes = [o for o in outputs if o.is_success]
    if not successes:
        return {
            "lang": lang,
            "total_samples": len(outputs),
            "evaluated": 0,
            "skipped": len(outputs),
            "wer_corpus": 0.0,
            "wer_per_sample_mean": 0.0,
            "wer_per_sample_median": 0.0,
            "wer_per_sample_std": 0.0,
            "wer_per_sample_p95": 0.0,
            "wer_below_50_corpus": 0.0,
            "n_above_50_pct_wer": 0,
            "pct_above_50_pct_wer": 0.0,
            "latency_mean_s": 0.0,
            "audio_duration_mean_s": 0.0,
        }

    total_errors = sum(o.substitutions + o.deletions + o.insertions for o in successes)
    total_ref_words = sum(o.substitutions + o.deletions + o.hits for o in successes)
    corpus_wer = total_errors / total_ref_words if total_ref_words > 0 else 0.0

    wer_arr = np.array([o.wer for o in successes])
    latencies = [o.latency_s for o in successes]
    audio_durations = [o.audio_duration_s for o in successes if o.audio_duration_s > 0]

    n_above_50 = int(np.sum(wer_arr > 0.5))

    ok_samples = [o for o in successes if o.wer <= 0.5]
    if ok_samples:
        ok_errors = sum(
            o.substitutions + o.deletions + o.insertions for o in ok_samples
        )
        ok_ref = sum(o.substitutions + o.deletions + o.hits for o in ok_samples)
        wer_below_50_micro = ok_errors / ok_ref if ok_ref > 0 else 0.0
    else:
        wer_below_50_micro = 0.0

    return {
        "lang": lang,
        "total_samples": len(outputs),
        "evaluated": len(successes),
        "skipped": len(outputs) - len(successes),
        "wer_corpus": float(corpus_wer),
        "wer_per_sample_mean": float(np.mean(wer_arr)),
        "wer_per_sample_median": float(np.median(wer_arr)),
        "wer_per_sample_std": float(np.std(wer_arr)),
        "wer_per_sample_p95": float(np.percentile(wer_arr, 95)),
        "wer_below_50_corpus": float(wer_below_50_micro),
        "n_above_50_pct_wer": n_above_50,
        "pct_above_50_pct_wer": (n_above_50 / len(successes) * 100 if successes else 0),
        "latency_mean_s": float(np.mean(latencies)),
        "audio_duration_mean_s": (
            float(np.mean(audio_durations)) if audio_durations else 0
        ),
    }


def print_wer_summary(metrics: dict, model_name: str) -> None:
    lw = SUMMARY_LABEL_WIDTH
    w = SUMMARY_LINE_WIDTH
    print(f"\n{'=' * w}")
    print(f"{'TTS WER Benchmark Result':^{w}}")
    print(f"{'=' * w}")
    print(f"  {'Model:':<{lw}} {model_name}")
    print(f"  {'Language:':<{lw}} {metrics.get('lang', 'N/A')}")
    print(
        f"  {'Evaluated / Total:':<{lw}} "
        f"{metrics.get('evaluated', 0)}/{metrics.get('total_samples', 0)}"
    )
    print(f"  {'Skipped:':<{lw}} {metrics.get('skipped', 0)}")
    print(f"{'-' * w}")
    print(
        f"  {'WER (corpus, micro-avg):':<{lw}} "
        f"{metrics.get('wer_corpus', 0):.4f} "
        f"({metrics.get('wer_corpus', 0) * 100:.2f}%)"
    )
    print(f"{'-' * w}")
    print(
        f"  {'WER per-sample mean:':<{lw}} "
        f"{metrics.get('wer_per_sample_mean', 0):.4f} "
        f"({metrics.get('wer_per_sample_mean', 0) * 100:.2f}%)"
    )
    print(
        f"  {'WER per-sample median:':<{lw}} "
        f"{metrics.get('wer_per_sample_median', 0):.4f}"
    )
    print(
        f"  {'WER per-sample std:':<{lw}} "
        f"{metrics.get('wer_per_sample_std', 0):.4f}"
    )
    print(
        f"  {'WER per-sample p95:':<{lw}} "
        f"{metrics.get('wer_per_sample_p95', 0):.4f}"
    )
    print(
        f"  {'WER corpus (excl >50%):':<{lw}} "
        f"{metrics.get('wer_below_50_corpus', 0):.4f} "
        f"({metrics.get('wer_below_50_corpus', 0) * 100:.2f}%)"
    )
    print(
        f"  {'>50% WER samples:':<{lw}} "
        f"{metrics.get('n_above_50_pct_wer', 0)} "
        f"({metrics.get('pct_above_50_pct_wer', 0):.1f}%)"
    )
    print(f"{'-' * w}")
    print(f"  {'Latency mean (s):':<{lw}} {metrics.get('latency_mean_s', 'N/A')}")
    print(
        f"  {'Audio duration mean (s):':<{lw}} "
        f"{metrics.get('audio_duration_mean_s', 'N/A')}"
    )
    print(f"{'=' * w}\n")


def save_wer_results(
    outputs: list[SampleOutput], metrics: dict, config: dict, output_dir: str
) -> None:
    os.makedirs(output_dir, exist_ok=True)

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
    json_path = os.path.join(output_dir, "wer_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

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
