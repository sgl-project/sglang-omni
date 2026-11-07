#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""YuE2 standalone inference: lyrics + style -> 48 kHz stereo wav.

Loads the YuE2 AR-NAR model + VAE directly (no serving stack) and renders one
song, printing the end-to-end latency.

    python examples/run_yue2_generate.py \
        --model-path /raid/yiakwy/YuE2-3B --vae-dir /raid/yiakwy/YuE2-Vae \
        --style "soft pop, female vocal" --lyrics "..." \
        --semantic-max-tokens 1024 --ode-steps 8 --out yue2.wav

Play the result with: aplay yue2.wav  (or ffplay yue2.wav)
"""

from __future__ import annotations

import argparse
import time

import soundfile as sf

from sglang_omni.models.yue2.payload_types import Yue2State
from sglang_omni.models.yue2.synth import Yue2Synthesizer

LYRICS = (
    "In the city lights we find our way,\n"
    "chasing shadows of a brighter day.\n"
    "Hold on tight, we are not alone,\n"
    "every heartbeat leads us home."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/raid/yiakwy/YuE2-3B")
    parser.add_argument("--vae-dir", default=None)
    parser.add_argument("--style", default="soft pop, female vocal")
    parser.add_argument("--lyrics", default=LYRICS)
    parser.add_argument("--cot", default="full", choices=["off", "melody", "full"])
    parser.add_argument("--seed", type=int, default=831001)
    parser.add_argument("--semantic-max-tokens", type=int, default=9000)
    parser.add_argument("--abc-max-tokens", type=int, default=4096)
    parser.add_argument("--ode-steps", type=int, default=32)
    parser.add_argument("--vae-core-frames", type=int, default=1024)
    parser.add_argument("--out", default="yue2.wav")
    args = parser.parse_args()

    synthesizer = Yue2Synthesizer(args.model_path, args.vae_dir)
    state = Yue2State(
        style=args.style,
        lyrics=args.lyrics,
        cot=args.cot,
        seed=args.seed,
        abc_max_tokens=args.abc_max_tokens,
        semantic_max_tokens=args.semantic_max_tokens,
        ode_steps=args.ode_steps,
        vae_core_frames=args.vae_core_frames,
    )
    started = time.perf_counter()
    waveform, _seconds = synthesizer.synthesize(state)
    elapsed = time.perf_counter() - started
    sf.write(args.out, waveform.cpu().numpy().T, 48000)
    duration = waveform.shape[-1] / 48000
    print(f"wrote {args.out}: duration={duration:.1f}s latency={elapsed:.2f}s "
          f"rtf={elapsed / max(duration, 1e-6):.3f}")


if __name__ == "__main__":
    main()
