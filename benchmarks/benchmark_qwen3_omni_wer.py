#!/usr/bin/env python3
"""
Qwen3-Omni seed-tts-eval WER benchmark (self-contained).

Replicates the WER evaluation from Section 5.2 of the Qwen3-Omni paper
(arxiv 2509.17765) using the seed-tts-eval dataset.

Published results (Table 13):
  test-zh: WER 1.07%
  test-en: WER 1.39%

Usage:
    # Full EN benchmark (auto-launches server):
    python benchmarks/benchmark_qwen3_omni_wer.py \
        --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --testset /tmp/seed-tts-eval/seedtts_testset/en

    # Full ZH benchmark:
    python benchmarks/benchmark_qwen3_omni_wer.py \
        --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --testset /tmp/seed-tts-eval/seedtts_testset/zh

    # Quick test with 10 samples:
    python benchmarks/benchmark_qwen3_omni_wer.py \
        --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --testset /tmp/seed-tts-eval/seedtts_testset/en \
        --max-samples 10

    # Connect to already-running server:
    python benchmarks/benchmark_qwen3_omni_wer.py \
        --api-base http://localhost:8000 \
        --testset /tmp/seed-tts-eval/seedtts_testset/en
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import jiwer
import numpy as np
import requests
import torch
import torchaudio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# seed-tts-eval dataset
# ---------------------------------------------------------------------------


def parse_meta_lst(testset_dir: str) -> list[dict]:
    """Parse seed-tts-eval meta.lst file.

    Format: id|ref_text|ref_audio_path|target_text
    """
    meta_path = Path(testset_dir) / "meta.lst"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.lst not found at {meta_path}")

    samples = []
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) != 4:
            logger.warning("Skipping malformed line: %s", line[:80])
            continue
        sample_id, ref_text, ref_audio_rel, target_text = parts
        ref_audio = str(Path(testset_dir) / ref_audio_rel)
        if not Path(ref_audio).is_file():
            logger.warning("Missing ref audio %s, skipping", ref_audio)
            continue
        samples.append(
            {
                "id": sample_id,
                "ref_text": ref_text,
                "ref_audio": ref_audio,
                "target_text": target_text,
            }
        )
    return samples


def detect_language(testset_dir: str) -> str:
    """Detect language from testset directory name."""
    name = Path(testset_dir).name.lower()
    if "zh" in name:
        return "zh"
    return "en"


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

STARTUP_TIMEOUT = 900
REQUEST_TIMEOUT = 600


def _resolve_config_path() -> str:
    candidates = [
        Path(__file__).resolve().parent.parent
        / "examples"
        / "configs"
        / "qwen3_omni_speech.yaml",
        Path("examples/configs/qwen3_omni_speech.yaml"),
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    raise FileNotFoundError(
        "Cannot find qwen3_omni_speech.yaml. "
        "Pass --config explicitly or run from the repo root."
    )


def start_server(
    model_path: str,
    port: int,
    config_path: str | None = None,
    log_level: str = "info",
) -> subprocess.Popen:
    if config_path is None:
        config_path = _resolve_config_path()

    cmd = [
        sys.executable,
        "-m",
        "sglang_omni.cli.cli",
        "serve",
        "--model-path",
        model_path,
        "--config",
        config_path,
        "--port",
        str(port),
        "--log-level",
        log_level,
    ]
    logger.info("Starting server: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )

    api_base = f"http://localhost:{port}"
    healthy = False
    t_start = time.monotonic()

    for _ in range(STARTUP_TIMEOUT):
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(
                f"Server exited with code {proc.returncode}.\n{out[:2000]}"
            )
        try:
            resp = requests.get(f"{api_base}/health", timeout=2)
            if resp.status_code == 200 and "healthy" in resp.text:
                healthy = True
                break
        except requests.ConnectionError:
            pass
        time.sleep(1)

    startup_s = time.monotonic() - t_start
    if not healthy:
        stop_server(proc)
        raise RuntimeError(f"Server did not become healthy within {STARTUP_TIMEOUT}s")

    logger.info("Server healthy in %.1fs at %s", startup_s, api_base)
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=10)
    logger.info("Server stopped.")


# ---------------------------------------------------------------------------
# Text normalization (matches seed-tts-eval convention)
# ---------------------------------------------------------------------------


def normalize_text(text: str, language: str = "en") -> str:
    text = text.strip()
    if language == "zh":
        # For Chinese: remove all non-Chinese and non-alphanumeric characters,
        # collapse whitespace
        text = re.sub(r"[^\u4e00-\u9fff\w]", "", text)
        return text
    # English
    text = text.lower()
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Whisper ASR
# ---------------------------------------------------------------------------


def load_whisper(model_name: str, device: str):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    logger.info("Loading Whisper model %s on %s ...", model_name, device)
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device).eval()
    return processor, model


@torch.no_grad()
def transcribe(
    processor,
    model,
    audio: torch.Tensor,
    sample_rate: int,
    language: str = "en",
) -> str:
    if sample_rate != 16000:
        audio = torchaudio.functional.resample(audio, sample_rate, 16000)
    if audio.ndim > 1:
        audio = audio.squeeze(0)

    inputs = processor(audio.cpu().numpy(), sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(model.device)
    gen_kwargs = {"max_new_tokens": 444, "task": "transcribe"}
    if language == "zh":
        gen_kwargs["language"] = "chinese"
    else:
        gen_kwargs["language"] = "english"
    predicted_ids = model.generate(input_features, **gen_kwargs)
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text


# ---------------------------------------------------------------------------
# Audio decoding from base64 WAV chunks
# ---------------------------------------------------------------------------


def decode_audio_chunks(
    audio_b64_parts: list[str],
) -> tuple[torch.Tensor, int] | None:
    if not audio_b64_parts:
        return None

    combined_b64 = "".join(audio_b64_parts)
    try:
        audio_bytes = base64.b64decode(combined_b64)
    except Exception:
        return None

    if len(audio_bytes) < 44:
        return None

    # Valid WAV
    if audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        buf = io.BytesIO(audio_bytes)
        try:
            audio, sr = torchaudio.load(buf)
            return audio, sr
        except Exception:
            pass

    # Fallback: raw PCM16 at 24kHz
    try:
        n_samples = len(audio_bytes) // 2
        samples = struct.unpack(f"<{n_samples}h", audio_bytes[: n_samples * 2])
        audio = torch.tensor(samples, dtype=torch.float32) / 32768.0
        return audio.unsqueeze(0), 24000
    except Exception:
        return None


def assert_not_silence(audio: torch.Tensor) -> bool:
    if audio.ndim > 1:
        audio = audio.squeeze(0)
    return len(torch.unique(audio)) > 1


# ---------------------------------------------------------------------------
# HTTP client — voice cloning request
# ---------------------------------------------------------------------------


def generate_speech(
    api_base: str,
    target_text: str,
    language: str = "en",
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> tuple[str, list[str], float]:
    """Send target text for TTS, return (text_output, audio_b64_parts, latency).

    Asks the model to read the target text aloud. WER is measured between
    the target text and Whisper transcription of the generated audio.
    """
    if language == "zh":
        prompt = f"请朗读以下文本：{target_text}"
    else:
        prompt = f"Please read the following text out loud: {target_text}"

    payload = {
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["text", "audio"],
        "audio": {"format": "wav"},
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    text_parts: list[str] = []
    audio_b64_parts: list[str] = []
    t0 = time.perf_counter()

    with requests.post(
        f"{api_base}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=REQUEST_TIMEOUT,
    ) as resp:
        if resp.status_code != 200:
            raise RuntimeError(f"Server returned {resp.status_code}: {resp.text[:500]}")
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[len("data: ") :]
            if data_str.strip() == "[DONE]":
                break
            chunk = json.loads(data_str)
            delta = chunk["choices"][0].get("delta", {})
            text = delta.get("content", "")
            if text:
                text_parts.append(text)
            audio = delta.get("audio")
            if audio and audio.get("data"):
                audio_b64_parts.append(audio["data"])

    latency = time.perf_counter() - t0
    return "".join(text_parts), audio_b64_parts, latency


# ---------------------------------------------------------------------------
# Evaluate one sample
# ---------------------------------------------------------------------------


def evaluate_sample(
    api_base: str,
    sample: dict,
    whisper_processor,
    whisper_model,
    language: str,
    max_tokens: int,
    temperature: float,
    audio_dir: str | None = None,
) -> dict | None:
    sample_id = sample["id"]
    target_text = sample["target_text"]
    ref_audio = sample["ref_audio"]

    try:
        text_output, audio_b64_parts, latency = generate_speech(
            api_base,
            target_text,
            language=language,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as e:
        logger.error("Request failed for %s: %s", sample_id, e)
        return None

    result = decode_audio_chunks(audio_b64_parts)
    if result is None:
        logger.warning("[%s] No valid audio output", sample_id)
        return None

    audio_tensor, sample_rate = result
    audio_dur = audio_tensor.shape[-1] / sample_rate

    if not assert_not_silence(audio_tensor):
        logger.warning("[%s] All-silence audio detected", sample_id)
        return None

    if audio_dir is not None:
        out_path = str(Path(audio_dir) / f"{sample_id}.wav")
        torchaudio.save(out_path, audio_tensor, sample_rate)

    # Transcribe generated audio
    hyp_text = transcribe(
        whisper_processor, whisper_model, audio_tensor, sample_rate, language
    )

    # WER: reference = target_text (what should have been spoken)
    ref_norm = normalize_text(target_text, language)
    hyp_norm = normalize_text(hyp_text, language)

    if not ref_norm:
        return None

    measures = jiwer.process_words(ref_norm, hyp_norm)
    return {
        "id": sample_id,
        "target_text": target_text,
        "ref_audio": ref_audio,
        "text_output": text_output,
        "whisper_text": hyp_text,
        "ref_norm": ref_norm,
        "hyp_norm": hyp_norm,
        "wer": measures.wer,
        "substitutions": measures.substitutions,
        "deletions": measures.deletions,
        "insertions": measures.insertions,
        "audio_duration_s": round(audio_dur, 4),
        "latency_s": round(latency, 4),
    }


# ---------------------------------------------------------------------------
# Print and save results
# ---------------------------------------------------------------------------


def print_summary(summary: dict, api_base: str, language: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Qwen3-Omni seed-tts-eval WER Benchmark ({language.upper()})")
    print(f"{'='*60}")
    print(f"  Server:          {api_base}")
    print(f"  Samples:         {summary['evaluated']}/{summary['total_samples']}")
    print(
        f"  WER mean:        {summary['wer_mean']:.4f} ({summary['wer_mean']*100:.2f}%)"
    )
    print(f"  WER median:      {summary['wer_median']:.4f}")
    print(f"  WER std:         {summary['wer_std']:.4f}")
    print(f"  WER p95:         {summary['wer_p95']:.4f}")
    n_above_50 = summary["n_above_50_pct_wer"]
    print(f"  >50% WER:        {n_above_50} ({summary['pct_above_50_pct_wer']:.1f}%)")
    print(f"  Latency mean:    {summary['latency_mean_s']:.2f}s")
    print(f"  Audio dur mean:  {summary['audio_duration_mean_s']:.2f}s")
    if language == "zh":
        print(f"  Published WER:   1.07%")
    else:
        print(f"  Published WER:   1.39%")
    print(f"{'='*60}")


def save_results(
    output_dir: str, config: dict, summary: dict, results: list[dict]
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {"config": config, "summary": summary, "per_sample": results}
    out_path = out_dir / "wer.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logger.info("Results saved to %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args):
    api_base = f"http://localhost:{args.port}"
    server_proc = None

    if args.api_base:
        api_base = args.api_base
    elif not args.model_path:
        logger.error(
            "Provide --model-path to launch a server, or --api-base to connect."
        )
        return

    # Parse dataset
    samples = parse_meta_lst(args.testset)
    language = args.language or detect_language(args.testset)
    logger.info(
        "Loaded %d samples from %s (language=%s)", len(samples), args.testset, language
    )

    if args.max_samples and len(samples) > args.max_samples:
        samples = samples[: args.max_samples]
        logger.info("Truncated to %d samples", len(samples))

    # Launch server if needed
    if not args.api_base:
        server_proc = start_server(
            args.model_path, args.port, args.config, args.log_level
        )

    try:
        _run_benchmark(args, api_base, samples, language)
    finally:
        if server_proc is not None:
            logger.info("Shutting down server...")
            stop_server(server_proc)


def _run_benchmark(args, api_base: str, samples: list[dict], language: str):
    # Verify server
    try:
        health = requests.get(f"{api_base}/health", timeout=10)
        if health.status_code != 200:
            logger.error("Server not healthy at %s", api_base)
            return
    except requests.ConnectionError:
        logger.error("Cannot connect to server at %s", api_base)
        return

    whisper_device = args.whisper_device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    whisper_processor, whisper_model = load_whisper(args.whisper_model, whisper_device)

    # Warmup
    if args.warmup > 0 and samples:
        logger.info("Warmup (%d requests)...", args.warmup)
        for i in range(min(args.warmup, len(samples))):
            s = samples[i]
            try:
                generate_speech(
                    api_base,
                    s["target_text"],
                    language=language,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                logger.info("  warmup %d/%d done", i + 1, args.warmup)
            except Exception as e:
                logger.warning("  warmup %d failed: %s", i + 1, e)

    audio_dir = None
    if args.save_audio:
        audio_dir = str(Path(args.output_dir) / "audio")
        Path(audio_dir).mkdir(parents=True, exist_ok=True)

    # Evaluate
    results = []
    for i, sample in enumerate(samples):
        r = evaluate_sample(
            api_base,
            sample,
            whisper_processor,
            whisper_model,
            language,
            args.max_tokens,
            args.temperature,
            audio_dir,
        )
        if r is None:
            logger.warning("[%d/%d] SKIPPED: %s", i + 1, len(samples), sample["id"])
            continue
        results.append(r)
        logger.info(
            "[%d/%d] WER=%.3f  target=%.40s  whisper=%.40s",
            i + 1,
            len(samples),
            r["wer"],
            r["ref_norm"],
            r["hyp_norm"],
        )

    if not results:
        logger.error("No successful evaluations.")
        return

    # Compute summary
    wers = [r["wer"] for r in results]
    n_above_50 = sum(1 for w in wers if w > 0.5)
    latencies = [r["latency_s"] for r in results]
    audio_durs = [r["audio_duration_s"] for r in results]

    summary = {
        "language": language,
        "total_samples": len(samples),
        "evaluated": len(results),
        "skipped": len(samples) - len(results),
        "wer_mean": float(np.mean(wers)),
        "wer_median": float(np.median(wers)),
        "wer_std": float(np.std(wers)),
        "wer_p95": float(np.percentile(wers, 95)),
        "n_above_50_pct_wer": n_above_50,
        "pct_above_50_pct_wer": n_above_50 / len(results) * 100,
        "latency_mean_s": float(np.mean(latencies)),
        "latency_median_s": float(np.median(latencies)),
        "latency_p95_s": float(np.percentile(latencies, 95)),
        "audio_duration_mean_s": float(np.mean(audio_durs)),
    }

    print_summary(summary, api_base, language)

    config = {
        "model_path": args.model_path,
        "api_base": api_base,
        "testset": args.testset,
        "language": language,
        "whisper_model": args.whisper_model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "max_samples": args.max_samples,
        "warmup": args.warmup,
    }
    save_results(args.output_dir, config, summary, results)


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-Omni seed-tts-eval WER benchmark",
    )

    # Server launch options
    server = parser.add_argument_group("server (auto-launch)")
    server.add_argument(
        "--model-path",
        default=None,
        help="HF model ID or local path (launches server automatically)",
    )
    server.add_argument(
        "--config",
        default=None,
        help="Pipeline config YAML (default: examples/configs/qwen3_omni_speech.yaml)",
    )
    server.add_argument("--port", type=int, default=18900)
    server.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
    )

    # External server mode
    parser.add_argument(
        "--api-base",
        default=None,
        help="Connect to an already-running server instead of launching one",
    )

    # Dataset
    parser.add_argument(
        "--testset",
        required=True,
        help="Path to seed-tts-eval testset directory (e.g. seedtts_testset/en)",
    )
    parser.add_argument(
        "--language",
        default=None,
        choices=["en", "zh"],
        help="Language (auto-detected from testset path if not set)",
    )

    # Benchmark options
    bench = parser.add_argument_group("benchmark")
    bench.add_argument(
        "--whisper-model",
        default="openai/whisper-large-v3",
        help="HF Whisper model ID",
    )
    bench.add_argument(
        "--whisper-device",
        default=None,
        help="Device for Whisper (default: cuda if available)",
    )
    bench.add_argument("--output-dir", default="results/qwen3_omni_seed_tts_eval")
    bench.add_argument("--max-samples", type=int, default=None)
    bench.add_argument("--max-tokens", type=int, default=2048)
    bench.add_argument("--temperature", type=float, default=0.7)
    bench.add_argument("--warmup", type=int, default=1)
    bench.add_argument("--save-audio", action="store_true")

    args = parser.parse_args()

    if not args.api_base and not args.model_path:
        parser.error(
            "Either --model-path (to launch server) or --api-base is required."
        )

    run(args)


if __name__ == "__main__":
    main()
