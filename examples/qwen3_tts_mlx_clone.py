#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS Base voice cloning on Apple Silicon via MLX.

Needs a Base checkpoint (CustomVoice and VoiceDesign ship no speech-tokenizer
encoder, so they cannot encode reference audio):

    huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir q3tts-base

    python examples/qwen3_tts_mlx_clone.py \\
        --model q3tts-base \\
        --ref-audio reference.wav \\
        --ref-text "exact transcript of reference.wav" \\
        --text "Text to speak in that voice." \\
        --out cloned.wav

``--ref-text`` must transcribe the *whole* reference clip. A mismatched or
partial transcript is a conditioning error, not a tolerance: it degrades output
badly in the official implementation too.
"""

from __future__ import annotations

import argparse
import time

from sglang_omni.models.qwen3_tts.mlx.generate import (
    CloneRequest,
    Qwen3TTSMlxGenerator,
    frames_to_seconds,
    write_wav,
)
from sglang_omni.models.qwen3_tts.mlx.sampling import SamplingParams


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to a Base checkpoint")
    parser.add_argument("--ref-audio", required=True, help="Reference audio file")
    parser.add_argument("--ref-text", required=True, help="Transcript of --ref-audio")
    parser.add_argument("--text", required=True, help="Text to synthesise")
    parser.add_argument("--out", default="cloned.wav")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--max-frames", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    started = time.perf_counter()
    generator = Qwen3TTSMlxGenerator.from_pretrained(args.model)
    print(f"loaded {args.model} in {time.perf_counter() - started:.1f}s")
    if not generator.speech_tokenizer.has_encoder:
        parser.error(
            f"{args.model} has no speech-tokenizer encoder, so it cannot clone a "
            "voice; use a *-Base checkpoint"
        )

    request = CloneRequest(
        text=args.text,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        language=args.language,
        max_frames=args.max_frames,
        semantic=SamplingParams(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        ),
        subtalker=SamplingParams(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        ),
        seed=args.seed,
    )

    started = time.perf_counter()
    audio = generator.clone(request)
    elapsed = time.perf_counter() - started
    seconds = write_wav(args.out, audio)
    print(
        f"wrote {args.out}: {seconds:.2f}s of audio in {elapsed:.1f}s "
        f"(RTF {elapsed / max(seconds, 1e-6):.2f}, "
        f"{frames_to_seconds(1) * 1000:.0f}ms per codec frame)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
