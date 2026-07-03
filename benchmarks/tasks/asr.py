# SPDX-License-Identifier: Apache-2.0
"""Shared ASR task layer: transcription, WER scoring, and ASR speed assembly.

Owns the ASR/WER primitives shared by the standalone ASR benchmark
(benchmarks/eval/benchmark_qwen3_asr_concurrency.py), the Qwen3-ASR CI gate
(tests/test_model/test_qwen3_asr_ci.py), the TTS WER stage
(benchmarks.tasks.tts.run_seedtts_transcribe), and the talker WER paths.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import io
import logging
import os
import string
import time
import wave

import aiohttp
import requests
import torch
from jiwer import process_words

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig, SendFn
from benchmarks.benchmarker.utils import get_wav_duration
from benchmarks.dataset.seedtts import SampleInput
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.metrics.wer import (
    SampleOutput,
    calculate_asr_speed_metrics,
    calculate_wer_metrics,
)

logger = logging.getLogger(__name__)

OMNI_WHISPER_MODEL_PATH = "openai/whisper-large-v3"
OMNI_WHISPER_REQUEST_TIMEOUT_S = 300
# note (aaron): the Whisper encoder accepts at most ~30 s per request
# (nb_max_frames=3000). The transformers pipeline uses chunk_length_s=30 and
# long talker audio mirrors that.
OMNI_WHISPER_CHUNK_LENGTH_S = 30
OMNI_WHISPER_CHUNK_STRIDE_S = 25
OMNI_WHISPER_SAMPLE_RATE = 16000

QWEN3_ASR_MODEL_PATH = "Qwen/Qwen3-ASR-1.7B"
QWEN3_ASR_REQUEST_TIMEOUT_S = 300
# note (aaron): ASR transcription fan-out for WER, not TTS generation concurrency.
DEFAULT_ASR_TRANSCRIBE_CONCURRENCY = 32
# note (aaron): warmup requests sent before the timed window, per unit of concurrency.
ASR_WARMUP_MULTIPLIER = 2


@functools.lru_cache(maxsize=1)
def _get_en_normalizer():
    """Lazy-load the required English WER normalizer from openai-whisper."""
    try:
        from whisper.normalizers import EnglishTextNormalizer
    except ImportError as exc:
        raise RuntimeError(
            "English WER requires openai-whisper "
            "(whisper.normalizers.EnglishTextNormalizer). "
            "Install pinned deps with uv pip install -e ."
        ) from exc

    return EnglishTextNormalizer()


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
    return normalizer(text)


def _load_wav_mono_16k(wav_path: str) -> torch.Tensor:
    import torchaudio

    audio, sample_rate = torchaudio.load(wav_path)
    if audio.ndim == 2 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = audio.squeeze(0).to(torch.float32)
    if sample_rate != OMNI_WHISPER_SAMPLE_RATE:
        audio = torchaudio.functional.resample(
            audio, sample_rate, OMNI_WHISPER_SAMPLE_RATE
        )
    return audio


def _wav_bytes_from_mono_16k(audio: torch.Tensor) -> bytes:
    pcm16 = (audio.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).cpu()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(OMNI_WHISPER_SAMPLE_RATE)
        wav_file.writeframes(pcm16.numpy().tobytes())
    return buffer.getvalue()


def _post_omni_whisper_transcription(
    asr: dict,
    audio_bytes: bytes,
    filename: str,
    lang: str,
) -> str:
    response = requests.post(
        f"http://127.0.0.1:{asr['router_port']}/v1/audio/transcriptions",
        data={
            "model": asr["model_path"],
            "language": "en" if lang == "en" else lang,
            "temperature": "0",
        },
        files={
            "file": (
                filename,
                audio_bytes,
                "audio/wav",
            )
        },
        timeout=OMNI_WHISPER_REQUEST_TIMEOUT_S,
        proxies={"http": None, "https": None},
    )
    response.raise_for_status()
    return str(response.json()["text"])


def _transcribe_omni_whisper(asr: dict, wav_path: str, lang: str) -> str:
    audio = _load_wav_mono_16k(wav_path)
    duration_s = float(audio.shape[0]) / OMNI_WHISPER_SAMPLE_RATE
    if duration_s <= OMNI_WHISPER_CHUNK_LENGTH_S:
        with open(wav_path, "rb") as audio_file:
            return _post_omni_whisper_transcription(
                asr,
                audio_file.read(),
                os.path.basename(wav_path),
                lang,
            )

    chunk_samples = OMNI_WHISPER_CHUNK_LENGTH_S * OMNI_WHISPER_SAMPLE_RATE
    stride_samples = OMNI_WHISPER_CHUNK_STRIDE_S * OMNI_WHISPER_SAMPLE_RATE
    chunk_texts: list[str] = []
    for start in range(0, int(audio.shape[0]), stride_samples):
        chunk = audio[start : start + chunk_samples]
        if chunk.numel() == 0:
            break
        chunk_bytes = _wav_bytes_from_mono_16k(chunk)
        chunk_texts.append(
            _post_omni_whisper_transcription(
                asr,
                chunk_bytes,
                f"{os.path.basename(wav_path)}.chunk{start // stride_samples}.wav",
                lang,
            ).strip()
        )
        if start + chunk_samples >= audio.shape[0]:
            break
    return " ".join(text for text in chunk_texts if text)


def load_omni_whisper_asr(
    router_port: int,
    model_path: str = OMNI_WHISPER_MODEL_PATH,
) -> dict:
    """Return an ASR handle that transcribes via SGLang Omni Whisper router."""
    return {
        "type": "omni_whisper",
        "router_port": router_port,
        "model_path": model_path,
    }


def load_qwen3_asr(
    router_port: int,
    model_path: str = QWEN3_ASR_MODEL_PATH,
) -> dict:
    """Return an ASR handle that transcribes via a Qwen3-ASR sglang-omni router."""
    return {
        "type": "qwen3_asr",
        "router_port": router_port,
        "model_path": model_path,
    }


def _is_whisper_asr_model(model_path: str) -> bool:
    return "whisper" in model_path.lower()


def load_router_asr(
    router_port: int,
    model_path: str = QWEN3_ASR_MODEL_PATH,
) -> dict:
    """Return an ASR handle backed by a running SGLang Omni ASR server."""
    if _is_whisper_asr_model(model_path):
        return load_omni_whisper_asr(router_port, model_path=model_path)
    return load_qwen3_asr(router_port, model_path=model_path)


def _transcribe_qwen3_asr(asr: dict, wav_path: str, lang: str) -> str:
    """Transcribe one wav via the Qwen3-ASR server's /v1/audio/transcriptions.

    Note: do not send temperature=0 because Qwen3-ASR degenerates under pure
    greedy (the server bumps it to 0.01). The language field selects the
    forced prefix. max_new_tokens comes from the Qwen3 ASR pipeline config.
    """
    with open(wav_path, "rb") as audio_file:
        response = requests.post(
            f"http://127.0.0.1:{asr['router_port']}/v1/audio/transcriptions",
            data={
                "model": asr["model_path"],
                "language": "en" if lang == "en" else lang,
                "response_format": "json",
            },
            files={"file": (os.path.basename(wav_path), audio_file, "audio/wav")},
            timeout=QWEN3_ASR_REQUEST_TIMEOUT_S,
            proxies={"http": None, "https": None},
        )
    response.raise_for_status()
    return str(response.json()["text"])


def _resolve_asr_backend(
    lang: str,
    asr_device: str,
    *,
    asr_router_port: int | None = None,
    asr_model_path: str = QWEN3_ASR_MODEL_PATH,
    generation_mode: str | None = None,
) -> dict:
    if asr_router_port is not None:
        return load_router_asr(asr_router_port, model_path=asr_model_path)
    return load_asr_model(lang, asr_device, generation_mode)


def load_asr_model(lang: str, device: str, generation_mode: str | None = None):
    """Legacy local ASR entry point.

    WER now runs through SGLang Omni's OpenAI-compatible transcription endpoint.
    Start an ASR server with sglang_omni.cli serve and pass its port instead
    of loading local ASR backends in-process.
    """
    mode_suffix = f" for {generation_mode} generation" if generation_mode else ""
    del device
    if lang not in {"en", "zh"}:
        raise ValueError(f"Unsupported language: {lang}")
    raise ValueError(
        "WER transcription requires a running SGLang Omni ASR server"
        f"{mode_suffix}. Start Qwen3-ASR (default "
        f"{QWEN3_ASR_MODEL_PATH}) or {OMNI_WHISPER_MODEL_PATH} and pass "
        "asr_router_port."
    )


def transcribe(asr: dict, wav_path: str, lang: str, device: str) -> str:
    if asr["type"] == "qwen3_asr":
        return _transcribe_qwen3_asr(asr, wav_path, lang)
    if asr["type"] == "omni_whisper":
        return _transcribe_omni_whisper(asr, wav_path, lang)
    raise ValueError(f"Unknown ASR type: {asr['type']}")


def apply_wer(output: SampleOutput, hyp_text: str, lang: str) -> SampleOutput:
    """Fill output with normalized texts and per-sample WER fields."""
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
    return apply_wer(output, hyp_text, lang)


def make_asr_send_fn(
    model_name: str,
    api_url: str,
    lang: str = "en",
) -> SendFn:
    """Return a send_fn(session, sample) -> RequestResult that transcribes one
    SeedTTS reference clip via the Omni /v1/audio/transcriptions endpoint.

    Note: do not send temperature=0 because Qwen3-ASR degenerates under pure
    greedy (the server bumps it to 0.01). The language field selects the
    forced prefix. max_new_tokens comes from the Qwen3 ASR pipeline config.
    """

    async def send_fn(
        session: aiohttp.ClientSession, sample: SampleInput
    ) -> RequestResult:
        result = RequestResult(request_id=sample.sample_id)
        try:
            with open(sample.ref_audio, "rb") as audio_file:
                audio_bytes = audio_file.read()
        except OSError as exc:
            result.error = str(exc)
            return result
        result.audio_duration_s = get_wav_duration(audio_bytes)

        form = aiohttp.FormData()
        form.add_field("model", model_name)
        form.add_field("language", "en" if lang == "en" else lang)
        form.add_field("response_format", "json")
        form.add_field(
            "file",
            audio_bytes,
            filename=os.path.basename(sample.ref_audio),
            content_type="audio/wav",
        )

        start_time = time.perf_counter()
        try:
            async with session.post(api_url, data=form) as response:
                if response.status != 200:
                    result.error = f"HTTP {response.status}: {await response.text()}"
                else:
                    payload = await response.json()
                    result.text = str(payload.get("text", ""))
                    result.is_success = True
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            result.error = str(exc)
        finally:
            result.latency_s = time.perf_counter() - start_time
        if result.is_success and result.audio_duration_s > 0:
            result.rtf = result.latency_s / result.audio_duration_s
        return result

    return send_fn


async def run_asr_transcription(
    samples: list[SampleInput],
    *,
    host: str = "127.0.0.1",
    port: int,
    model_path: str = QWEN3_ASR_MODEL_PATH,
    lang: str = "en",
    concurrency: int = DEFAULT_ASR_TRANSCRIBE_CONCURRENCY,
    warmup: int = 0,
    request_timeout_s: int = QWEN3_ASR_REQUEST_TIMEOUT_S,
    disable_tqdm: bool = True,
) -> tuple[list[RequestResult], float]:
    """Transcribe samples against a running ASR router at one concurrency.

    Returns (outputs, wall_clock_s) via the shared BenchmarkRunner.
    """
    api_url = f"http://{host}:{port}/v1/audio/transcriptions"
    send_fn = make_asr_send_fn(model_path, api_url, lang=lang)
    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=concurrency,
            warmup=warmup,
            disable_tqdm=disable_tqdm,
            timeout_s=request_timeout_s,
        )
    )
    outputs = await runner.run(samples, send_fn)
    return outputs, runner.wall_clock_s


def build_asr_eval_results(
    samples: list[SampleInput],
    outputs: list[RequestResult],
    wall_clock_s: float,
    lang: str,
    *,
    model_path: str = QWEN3_ASR_MODEL_PATH,
    concurrency: int = DEFAULT_ASR_TRANSCRIBE_CONCURRENCY,
) -> dict:
    """Score transcriptions and assemble WER + speed metrics.

    Returns {"summary": wer, "speed": speed, "per_sample": [...]} with the
    exact summary.* and speed.* keys the Qwen3-ASR gate writes and the
    tune-ci-thresholds config reads. WER/speed reuse benchmarks.metrics.
    """
    result_by_id = {result.request_id: result for result in outputs}
    sample_outputs: list[SampleOutput] = []
    per_sample: list[dict] = []
    for sample in samples:
        result = result_by_id.get(sample.sample_id)
        output = SampleOutput(
            sample_id=sample.sample_id,
            target_text=sample.ref_text,
        )
        if result is None or not result.is_success:
            output.error = (result.error if result else "") or "No transcription"
        else:
            output.latency_s = result.latency_s
            output.asr_latency_s = result.latency_s
            output.audio_duration_s = result.audio_duration_s
            output = apply_wer(output, result.text, lang)
        sample_outputs.append(output)
        per_sample.append(
            {
                "id": output.sample_id,
                "is_success": output.is_success,
                "wer": output.wer if output.is_success else None,
                "ref_text": output.target_text,
                "hyp_text": output.whisper_text,
                "ref_norm": output.ref_norm,
                "hyp_norm": output.hyp_norm,
                "audio_duration_s": output.audio_duration_s,
                "latency_s": output.latency_s,
                "error": output.error,
            }
        )

    wer_summary = calculate_wer_metrics(sample_outputs, lang)
    # note (Yue Yin): gate + tune-ci-thresholds read summary.corpus_wer
    wer_summary["corpus_wer"] = wer_summary["wer_corpus"]

    asr_speed = calculate_asr_speed_metrics(sample_outputs, wall_time_s=wall_clock_s)
    # note (Yue Yin): compute_speed_metrics supplies rtf_p95 (the asr metrics omit it)
    perf = compute_speed_metrics(outputs, wall_clock_s=wall_clock_s)
    speed = {
        **asr_speed,
        "asr_model": model_path,
        "asr_concurrency": concurrency,
        "asr_rtf_p95": perf.get("rtf_p95"),
        # note (Yue Yin): plain calibration keys read by tune-ci-thresholds + gate
        "throughput_samples_per_s": asr_speed["asr_throughput_samples_per_s"],
        "latency_mean_s": asr_speed["asr_latency_mean_s"],
        "latency_median_s": asr_speed["asr_latency_median_s"],
        "latency_p95_s": asr_speed["asr_latency_p95_s"],
        "latency_p99_s": asr_speed["asr_latency_p99_s"],
        "rtf_mean": asr_speed["asr_rtf_mean"],
        "rtf_median": asr_speed["asr_rtf_median"],
        "rtf_p95": perf.get("rtf_p95"),
    }
    return {"summary": wer_summary, "speed": speed, "per_sample": per_sample}


def compute_text_audio_consistency(
    request_results: list[RequestResult],
    lang: str,
    asr_device: str,
    *,
    asr_router_port: int | None = None,
    asr_model_path: str = QWEN3_ASR_MODEL_PATH,
    asr_concurrency: int = DEFAULT_ASR_TRANSCRIBE_CONCURRENCY,
) -> dict:
    """WER between each request's text output (ref) and ASR-transcribed audio (hyp)."""
    asr = _resolve_asr_backend(
        lang,
        asr_device,
        asr_router_port=asr_router_port,
        asr_model_path=asr_model_path,
    )

    outputs_by_idx: list[SampleOutput | None] = [None] * len(request_results)
    pending: list[tuple[int, RequestResult, SampleOutput]] = []
    for idx, result in enumerate(request_results):
        ref_text = " ".join(result.text.split())
        out = SampleOutput(
            sample_id=result.request_id,
            target_text=ref_text,
            latency_s=result.latency_s,
            audio_duration_s=result.audio_duration_s,
        )
        if not result.is_success or not result.wav_path:
            out.error = result.error or "No audio in response"
            outputs_by_idx[idx] = out
            continue
        pending.append((idx, result, out))

    def _transcribe_pending(
        result: RequestResult,
        output: SampleOutput,
    ) -> SampleOutput:
        asr_t0 = time.perf_counter()
        output = transcribe_and_compute_wer(
            output,
            result.wav_path,
            asr,
            lang,
            asr_device,
        )
        output.asr_latency_s = time.perf_counter() - asr_t0
        return output

    asr_concurrency = max(1, int(asr_concurrency))
    if asr_concurrency == 1:
        for idx, result, output in pending:
            outputs_by_idx[idx] = _transcribe_pending(result, output)
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=asr_concurrency,
        ) as executor:
            future_to_idx = {
                executor.submit(_transcribe_pending, result, output): idx
                for idx, result, output in pending
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                outputs_by_idx[future_to_idx[future]] = future.result()

    outputs = [output for output in outputs_by_idx if output is not None]

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


def compute_text_audio_consistency_from_records(
    per_sample: list[dict],
    lang: str,
    asr_device: str,
    *,
    audio_dir: str | None = None,
    sample_id_key: str = "sample_id",
    text_key: str = "raw_response",
    asr_router_port: int | None = None,
    asr_model_path: str = QWEN3_ASR_MODEL_PATH,
    asr_concurrency: int = DEFAULT_ASR_TRANSCRIBE_CONCURRENCY,
) -> dict:
    """Compute WER from saved eval records after the inference server is stopped."""
    request_results: list[RequestResult] = []
    for record in per_sample:
        sample_id = record.get(sample_id_key)
        wav_path = record.get("wav_path") or ""
        if not wav_path and audio_dir and sample_id:
            wav_path = os.path.join(audio_dir, f"{sample_id}.wav")
        request_results.append(
            RequestResult(
                request_id=str(sample_id or ""),
                text=str(record.get(text_key) or ""),
                is_success=bool(record.get("is_success")),
                latency_s=float(record.get("latency_s") or 0),
                audio_duration_s=float(record.get("audio_duration_s") or 0),
                wav_path=wav_path,
                error=str(record.get("error") or ""),
            )
        )
    return compute_text_audio_consistency(
        request_results,
        lang,
        asr_device,
        asr_router_port=asr_router_port,
        asr_model_path=asr_model_path,
        asr_concurrency=asr_concurrency,
    )
