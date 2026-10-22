# SPDX-License-Identifier: Apache-2.0
"""Run the Python Voxtral reference and print generated code frames as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from threading import Event

from sglang_omni_mlx.voxtral_tts.config import read_voxtral_config
from sglang_omni_mlx.voxtral_tts.model import load_voxtral_model
from sglang_omni_mlx.voxtral_tts.synthesis import VoxtralSynthesizer
from sglang_omni_mlx.voxtral_tts.tokenizer import (
    VoxtralSpeechTokenizer,
    load_voice_embeddings,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    arguments = parser.parse_args()
    config = read_voxtral_config(arguments.model_path)
    model = load_voxtral_model(config)
    tokenizer = VoxtralSpeechTokenizer(arguments.model_path)
    voice_embeddings = load_voice_embeddings(config)
    prompt_token_ids = tokenizer.encode(arguments.text, arguments.voice)
    synthesizer = VoxtralSynthesizer(
        model,
        config.audio.audio_token_id,
        arguments.max_new_tokens,
    )
    result = synthesizer.generate_codes(
        prompt_token_ids,
        voice_embeddings[arguments.voice],
        Event(),
        arguments.seed,
    )
    print(
        json.dumps(
            {
                "frames": result.audio_codes.tolist(),
                "finish_reason": result.finish_reason,
            }
        )
    )


if __name__ == "__main__":
    main()
