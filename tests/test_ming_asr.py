#!/usr/bin/env python3
"""Test Ming-Omni ASR (audio understanding) functionality and performance.

Prerequisites:
    Server must be running:
    CUDA_VISIBLE_DEVICES=0,1 python examples/run_ming_omni_server.py \
        --model-path inclusionAI/Ming-flash-omni-2.0 --port 8000 --tp-size 2

Usage:
    # Basic test (uses TTS-generated audio)
    python tests/test_ming_asr.py --url http://localhost:8000

    # With custom audio file
    python tests/test_ming_asr.py --url http://localhost:8000 --audio /path/to/audio.wav

    # Generate test audio first, then test ASR
    python tests/test_ming_asr.py --url http://localhost:8000 --generate-audio

    # Performance benchmark
    python tests/test_ming_asr.py --url http://localhost:8000 --benchmark --num-runs 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test cases: (audio_path, prompt, expected_keywords)
# ---------------------------------------------------------------------------
ASR_PROMPTS = {
    "transcribe": "Please transcribe this audio.",
    "transcribe_zh": "请逐字转写这段音频。",
    "understand": "What language is spoken in this audio? What is the speaker talking about?",
    "summarize": "Summarize the content of this audio in one sentence.",
}


def generate_test_audio(model_path: str, device: str, output_path: str) -> str:
    """Generate a test WAV using the talker for ASR testing."""
    logger.info("Generating test audio with TTS...")
    try:
        import torch
        import torchaudio

        from sglang_omni.models.ming_omni.talker import (
            MingOmniTalker,
            MingOmniTalkerConfig,
            SpkembExtractor,
        )
        from sglang_omni.models.ming_omni.talker.audio_vae.modeling_audio_vae import (
            AudioVAE,
        )
        from sglang_omni.models.weight_loader import load_weights_by_prefix

        local_path = model_path
        if not os.path.isdir(model_path):
            from huggingface_hub import snapshot_download

            local_path = snapshot_download(model_path)
        talker_path = os.path.join(local_path, "talker")

        config = MingOmniTalkerConfig.from_pretrained_dir(talker_path)
        talker = MingOmniTalker(config)
        talker.eval()
        weights = load_weights_by_prefix(talker_path, prefix="")
        talker.load_weights(weights.items())
        talker.to(device=device, dtype=torch.bfloat16)

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(os.path.join(talker_path, "llm"))
        talker.set_tokenizer(tokenizer)
        try:
            talker.set_spkemb_extractor(
                SpkembExtractor(os.path.join(talker_path, "campplus.onnx"))
            )
        except Exception:
            pass
        try:
            from talker_tn.talker_tn import TalkerTN

            talker.set_normalizer(TalkerTN())
        except ImportError:
            pass
        talker.initial_graph()

        vae_path = os.path.join(talker_path, "vae")
        vae = AudioVAE.from_pretrained(vae_path, dtype=torch.bfloat16)
        vae.to(device).eval()

        text = "Hello, this is a test of the Ming Omni speech recognition system."
        all_wavs = []
        with torch.no_grad():
            for tts_speech, _, _, _ in talker.omni_audio_generation(
                tts_text=text,
                voice_name=None,
                audio_detokenizer=vae,
                stream=False,
            ):
                if tts_speech is not None:
                    all_wavs.append(tts_speech)

        if all_wavs:
            waveform = torch.cat(all_wavs, dim=-1)
            sr = getattr(vae.config, "sample_rate", 44100)
            wav_tensor = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform
            torchaudio.save(output_path, wav_tensor.cpu().float(), sr)
            logger.info(
                "Generated %s (%.2fs, %dHz)", output_path, waveform.shape[-1] / sr, sr
            )
            return text
        else:
            logger.error("TTS generated no audio")
            sys.exit(1)
    except Exception as e:
        logger.error("Failed to generate audio: %s", e)
        raise


def send_asr_request(
    url: str,
    audio_path: str,
    prompt: str,
    max_tokens: int = 256,
) -> dict:
    """Send ASR request to server, return {text, latency_s, success, error}."""
    payload = {
        "model": "ming-omni",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": audio_path}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "audios": [audio_path],
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        latency = time.perf_counter() - t0

        if resp.status_code != 200:
            return {
                "text": "",
                "latency_s": latency,
                "success": False,
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
            }

        body = resp.json()
        if "detail" in body:
            return {
                "text": "",
                "latency_s": latency,
                "success": False,
                "error": str(body["detail"])[:200],
            }

        text = body["choices"][0]["message"]["content"]
        return {"text": text, "latency_s": latency, "success": True, "error": ""}

    except Exception as e:
        return {
            "text": "",
            "latency_s": time.perf_counter() - t0,
            "success": False,
            "error": str(e)[:200],
        }


def run_functional_tests(url: str, audio_path: str, expected_text: str | None):
    """Run ASR functional tests."""
    w = 80
    print(f"\n{'=' * w}")
    print(f"{'MING-OMNI ASR — FUNCTIONAL TESTS':^{w}}")
    print(f"{'=' * w}")
    print(f"  Audio: {audio_path}")
    if expected_text:
        print(f"  Expected content: {expected_text}")
    print()

    results = []
    for label, prompt in ASR_PROMPTS.items():
        logger.info("Testing '%s'...", label)
        r = send_asr_request(url, audio_path, prompt)

        status = "PASS" if r["success"] else "FAIL"
        print(f"  [{status}] {label} ({r['latency_s']:.2f}s)")
        if r["success"]:
            print(f"         Prompt: {prompt}")
            print(f"         Output: {r['text'][:200]}")
        else:
            print(f"         Error: {r['error']}")
        print()

        results.append({"label": label, "prompt": prompt, **r})

    passed = sum(1 for r in results if r["success"])
    print(f"  Result: {passed}/{len(results)} tests passed")
    print(f"{'=' * w}")
    return results


def run_benchmark(url: str, audio_path: str, num_runs: int):
    """Run ASR performance benchmark."""
    w = 80
    print(f"\n{'=' * w}")
    print(f"{'MING-OMNI ASR — PERFORMANCE BENCHMARK':^{w}}")
    print(f"{'=' * w}")
    print(f"  Audio: {audio_path}")
    print(f"  Runs: {num_runs}")
    print()

    # Get audio duration
    audio_duration_s = None
    try:
        import torchaudio

        waveform, sr = torchaudio.load(audio_path)
        audio_duration_s = waveform.shape[-1] / sr
        print(f"  Audio duration: {audio_duration_s:.2f}s")
    except Exception:
        pass

    prompt = "Please transcribe this audio."

    # Warmup
    logger.info("Warmup...")
    send_asr_request(url, audio_path, prompt, max_tokens=32)

    # Benchmark
    latencies = []
    for i in range(num_runs):
        r = send_asr_request(url, audio_path, prompt, max_tokens=128)
        if r["success"]:
            latencies.append(r["latency_s"])
            logger.info("  Run %d/%d: %.3fs", i + 1, num_runs, r["latency_s"])
        else:
            logger.warning("  Run %d/%d: FAILED - %s", i + 1, num_runs, r["error"])

    if latencies:
        arr = np.array(latencies)
        print(f"\n  {'Metric':<30} {'Value':>10}")
        print(f"  {'-' * 42}")
        print(f"  {'Latency Mean':<30} {arr.mean():>10.3f} s")
        print(f"  {'Latency Median':<30} {np.median(arr):>10.3f} s")
        print(f"  {'Latency P95':<30} {np.percentile(arr, 95):>10.3f} s")
        print(f"  {'Latency Std':<30} {arr.std():>10.3f} s")
        if audio_duration_s:
            rtf = arr.mean() / audio_duration_s
            print(f"  {'Audio Duration':<30} {audio_duration_s:>10.2f} s")
            print(f"  {'RTF (lower=faster)':<30} {rtf:>10.3f}")
            print(f"  {'Real-time factor':<30} {f'{1/rtf:.1f}x real-time':>10}")
    else:
        print("  No successful runs!")

    print(f"{'=' * w}")
    return latencies


def main():
    parser = argparse.ArgumentParser(description="Test Ming-Omni ASR")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--audio", default=None, help="Path to test audio file")
    parser.add_argument(
        "--generate-audio",
        action="store_true",
        help="Generate test audio using TTS before testing",
    )
    parser.add_argument("--model-path", default="inclusionAI/Ming-flash-omni-2.0")
    parser.add_argument("--device", default="cuda:0", help="Device for TTS generation")
    parser.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmark"
    )
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--output", default="/tmp/asr_test_results.json")
    args = parser.parse_args()

    # Determine audio path
    audio_path = args.audio
    expected_text = None

    if audio_path is None:
        # Try default locations
        defaults = ["/tmp/test_ming_omni_talker.wav", "/tmp/test_asr.wav"]
        for p in defaults:
            if os.path.exists(p):
                audio_path = p
                expected_text = "Hello, this is a test of the Ming Omni Talker model."
                break

    if audio_path is None and args.generate_audio:
        audio_path = "/tmp/test_asr.wav"
        expected_text = generate_test_audio(args.model_path, args.device, audio_path)

    if audio_path is None:
        logger.error("No audio file found. Use --audio <path> or --generate-audio")
        sys.exit(1)

    if not os.path.exists(audio_path):
        logger.error("Audio file not found: %s", audio_path)
        sys.exit(1)

    # Health check
    try:
        resp = requests.get(f"{args.url}/v1/models", timeout=5)
        resp.raise_for_status()
        logger.info("Server OK: %s", args.url)
    except Exception as e:
        logger.error("Cannot reach server: %s", e)
        sys.exit(1)

    # Run tests
    all_results = {}

    func_results = run_functional_tests(args.url, audio_path, expected_text)
    all_results["functional"] = func_results

    if args.benchmark:
        latencies = run_benchmark(args.url, audio_path, args.num_runs)
        all_results["benchmark"] = {
            "latencies": latencies,
            "num_runs": args.num_runs,
            "audio_path": audio_path,
        }

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
