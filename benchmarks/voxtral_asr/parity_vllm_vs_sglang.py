#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Numerical parity check: vLLM vs sglang-omni for Voxtral realtime ASR.

Usage:
    # Terminal 1: start vllm server
    python -m vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 \
      --tokenizer-mode mistral --config-format mistral --load-format mistral \
      --compilation-config '{"cudagraph_mode":"PIECEWISE"}' --port 8000

    # Terminal 2: start sglang-omni server
    python -m sglang_omni.launcher --config examples/configs/voxtral_asr.yaml \
      --port 8001

    # Terminal 3: run parity check
    python benchmarks/voxtral_asr/parity_vllm_vs_sglang.py \
      --audio /path/to/audio.wav \
      --vllm-url http://127.0.0.1:8000/v1/audio/transcriptions \
      --sglang-url http://127.0.0.1:8001/v1/audio/transcriptions
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests


def transcribe_vllm(url: str, audio_path: Path, language: str | None = None) -> dict:
    # vLLM serves OpenAI-compatible /v1/audio/transcriptions (multipart upload).
    files = {"file": (audio_path.name, audio_path.read_bytes(), "audio/wav")}
    data = {"model": "mistralai/Voxtral-Mini-4B-Realtime-2602"}
    if language:
        data["language"] = language
    resp = requests.post(url, files=files, data=data, timeout=120)
    resp.raise_for_status()
    return resp.json()


def transcribe_sglang(url: str, audio_path: Path, language: str | None = None) -> dict:
    files = {"file": (audio_path.name, audio_path.read_bytes(), "audio/wav")}
    data = {"model": "voxtral"}
    if language:
        data["language"] = language
    resp = requests.post(url, files=files, data=data, timeout=120)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument(
        "--vllm-url",
        default="http://127.0.0.1:8000/v1/audio/transcriptions",
    )
    parser.add_argument(
        "--sglang-url",
        default="http://127.0.0.1:8001/v1/audio/transcriptions",
    )
    parser.add_argument("--language", default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    vllm_result = transcribe_vllm(args.vllm_url, args.audio, args.language)
    vllm_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    sglang_result = transcribe_sglang(args.sglang_url, args.audio, args.language)
    sglang_time = time.perf_counter() - t0

    vllm_text = vllm_result.get("text", "")
    sglang_text = sglang_result.get("text", "")

    print("=" * 60)
    print(f"Audio: {args.audio}")
    print(f"vLLM  ({vllm_time:.2f}s): {vllm_text!r}")
    print(f"SGLang ({sglang_time:.2f}s): {sglang_text!r}")
    print(f"Match: {vllm_text == sglang_text}")
    print("=" * 60)

    out = {
        "vllm": {"text": vllm_text, "latency_s": vllm_time},
        "sglang": {"text": sglang_text, "latency_s": sglang_time},
        "match": vllm_text == sglang_text,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
