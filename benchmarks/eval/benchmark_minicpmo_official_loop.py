# SPDX-License-Identifier: Apache-2.0
"""Run the baseline trial inputs through the checkpoint's own MiniCPMODuplex loop, text only, in a transformers==4.51.0 environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import soundfile
import torch
from transformers import AutoModel

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# note (Jisen Li): The loop front-pads the first chunk to its 1,035 ms window, as the session does, so both cut units at the same input times.
CHUNK_MS = 1000
SYSTEM_PROMPT = "Streaming Omni Conversation."
REFERENCE_AUDIO = Path("assets") / "HT_ref_audio.wav"
CONFIGURATION_FILE = "configuration.json"
# note (Jisen Li): This runs in its own transformers 4.51 environment, so it reads the inputs directory without importing the trial modules.
TRIALS_FILE = "trials.json"
PCM16_SCALE = 32768
# note (Jisen Li): These equal both the loop's defaults and the session's, and are passed so the logs record them.
MAX_SPEAK_TOKENS_PER_CHUNK = 20
TEMPERATURE = 0.7
TOP_P = 0.8
LISTEN_PROB_SCALE = 1.0
TEXT_REPETITION_PENALTY = 1.05
TEXT_REPETITION_WINDOW_SIZE = 512

ConfigurationName = Literal["official", "matched"]


@dataclass(frozen=True, kw_only=True)
class LoopConfiguration:
    top_k: int
    force_listen_count: int


# note (Jisen Li): matched takes the session defaults, so a gap from the session is not a sampling difference.
CONFIGURATIONS: dict[ConfigurationName, LoopConfiguration] = {
    "official": LoopConfiguration(top_k=100, force_listen_count=0),
    "matched": LoopConfiguration(top_k=20, force_listen_count=3),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", type=Path, required=True, help="Checkpoint directory"
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        required=True,
        help="Directory written by benchmark_minicpmo_baselines inputs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for <trial>_r<round>.jsonl logs of one configuration",
    )
    parser.add_argument(
        "--configuration", choices=sorted(CONFIGURATIONS), required=True
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--first-round", type=int, default=1)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configuration_record = {
        "configuration": arguments.configuration,
        **asdict(CONFIGURATIONS[arguments.configuration]),
        "max_speak_tokens_per_chunk": MAX_SPEAK_TOKENS_PER_CHUNK,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "listen_prob_scale": LISTEN_PROB_SCALE,
        "text_repetition_penalty": TEXT_REPETITION_PENALTY,
        "text_repetition_window_size": TEXT_REPETITION_WINDOW_SIZE,
        "chunk_ms": CHUNK_MS,
        "model_path": str(arguments.model_path),
        "trials_sha256": hashlib.sha256(
            (arguments.inputs / TRIALS_FILE).read_bytes()
        ).hexdigest(),
    }
    configuration_path = arguments.output / CONFIGURATION_FILE
    if (
        configuration_path.exists()
        and json.loads(configuration_path.read_text()) != configuration_record
    ):
        raise ValueError(
            f"{arguments.output} holds logs of another configuration or other inputs"
        )
    else:
        arguments.output.mkdir(parents=True, exist_ok=True)
        configuration_path.write_text(json.dumps(configuration_record, indent=2) + "\n")
    configuration = CONFIGURATIONS[arguments.configuration]
    model = (
        AutoModel.from_pretrained(
            arguments.model_path,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )
        .eval()
        .cuda()
    )
    duplex = model.as_duplex(
        generate_audio=False, force_listen_count=configuration.force_listen_count
    )
    reference_audio, _ = librosa.load(
        arguments.model_path / REFERENCE_AUDIO, sr=SAMPLE_RATE, mono=True
    )
    chunk_samples = CHUNK_MS * SAMPLE_RATE // 1000
    for round_index in range(
        arguments.first_round, arguments.first_round + arguments.rounds
    ):
        for trial_entry in json.loads((arguments.inputs / TRIALS_FILE).read_text()):
            input_path = arguments.inputs / f"{trial_entry['name']}.wav"
            log_path = arguments.output / f"{trial_entry['name']}_r{round_index}.jsonl"
            input_samples, _ = soundfile.read(str(input_path), dtype="int16")
            if (
                hashlib.sha256(input_samples.tobytes()).hexdigest()
                != trial_entry["sha256"]
            ):
                raise ValueError(f"{input_path.name} no longer matches {TRIALS_FILE}")
            elif log_path.exists():
                logger.info(f"Keeping {log_path.name}")
            else:
                torch.manual_seed(round_index)
                waveform = input_samples.astype(np.float32) / PCM16_SCALE
                duplex.prepare(
                    prefix_system_prompt=SYSTEM_PROMPT, ref_audio=reference_audio
                )
                unit_results = []
                started_s = time.perf_counter()
                for unit_index, start_sample in enumerate(
                    range(0, len(waveform), chunk_samples)
                ):
                    prefill_result = duplex.streaming_prefill(
                        audio_waveform=waveform[
                            start_sample : start_sample + chunk_samples
                        ]
                    )
                    if not prefill_result["success"]:
                        raise RuntimeError(
                            f"{input_path.name} unit {unit_index}: {prefill_result['reason']}"
                        )
                    else:
                        pass
                    generate_result = duplex.streaming_generate(
                        max_new_speak_tokens_per_chunk=MAX_SPEAK_TOKENS_PER_CHUNK,
                        decode_mode="sampling",
                        temperature=TEMPERATURE,
                        top_k=configuration.top_k,
                        top_p=TOP_P,
                        listen_prob_scale=LISTEN_PROB_SCALE,
                        text_repetition_penalty=TEXT_REPETITION_PENALTY,
                        text_repetition_window_size=TEXT_REPETITION_WINDOW_SIZE,
                    )
                    unit_results.append(
                        {
                            "unit_index": unit_index,
                            "is_listen": bool(generate_result["is_listen"]),
                            "text": generate_result["text"],
                            "end_of_turn": bool(generate_result["end_of_turn"]),
                        }
                    )
                log_path.write_text(
                    "".join(
                        json.dumps(unit_result) + "\n" for unit_result in unit_results
                    )
                )
                logger.info(
                    f"Ran {log_path.name} in {time.perf_counter() - started_s:.1f} s"
                )


if __name__ == "__main__":
    main()
else:
    pass
