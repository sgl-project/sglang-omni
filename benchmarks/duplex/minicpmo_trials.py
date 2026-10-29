# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o baseline trials: their layouts, and their input WAVs composed from spoken clips."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import soundfile
from pydantic import BaseModel, ConfigDict, TypeAdapter

from benchmarks.duplex.client import SAMPLE_RATE

TRIALS_FILE = "trials.json"
CLIPS_DIR = Path(__file__).parent / "assets" / "minicpmo_baseline_clips"
PCM16_MIN = -32768
PCM16_MAX = 32767
NOISE_SEED = 0
STOP_TRIAL_S = 30.0
COUNT_QUESTION_S = 1.0
# note (Jisen Li): The counting reply's audio starts when unit 3 ends, so a stop in unit 7 lands about 3 s into what is heard.
STOP_UNIT_START_S = 7.0
STOP_OFFSETS_MS = (0, 250, 500, 750)
GREETING_TRIAL_S = 32.0
COLD_GREETINGS_S = (2.0, 10.0, 16.0)
COLD_QUESTION_S = 23.0
QUIET_QUESTION_S = 10.0
ONGOING_QUESTION_S = 1.0
ONGOING_GREETINGS_S = (12.0, 18.0, 24.0)
SOFT_GAIN = 0.1
SOFTER_GAIN = 0.03
NOISE_DBFS = -50.0

EventRole = Literal["prompt", "stop", "greeting", "question"]


@dataclass(frozen=True, kw_only=True)
class Placement:
    clip_name: str
    start_s: float
    role: EventRole
    gain: float = 1.0


@dataclass(frozen=True, kw_only=True)
class Trial:
    name: str
    duration_s: float
    placements: tuple[Placement, ...]
    noise_dbfs: float | None = None


class SpokenEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: EventRole
    start_s: float
    end_s: float


class TrialInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    duration_s: float
    sha256: str
    events: list[SpokenEvent]


TRIAL_INPUTS = TypeAdapter(list[TrialInput])


def greeting_trial(
    name: str,
    *,
    greeting: str,
    greeting_gain: float,
    greetings_s: tuple[float, ...],
    question_s: float,
    noise_dbfs: float | None = None,
) -> Trial:
    greetings = tuple(
        Placement(
            clip_name=greeting, start_s=start_s, role="greeting", gain=greeting_gain
        )
        for start_s in greetings_s
    )
    question = Placement(
        clip_name="capital_question", start_s=question_s, role="question"
    )
    return Trial(
        name=name,
        duration_s=GREETING_TRIAL_S,
        placements=tuple(
            sorted((*greetings, question), key=lambda placement: placement.start_s)
        ),
        noise_dbfs=noise_dbfs,
    )


def quiet_question_trial(name: str, *, gain: float, noise_dbfs: float | None) -> Trial:
    return Trial(
        name=name,
        duration_s=GREETING_TRIAL_S,
        placements=(
            Placement(
                clip_name="capital_question",
                start_s=QUIET_QUESTION_S,
                role="question",
                gain=gain,
            ),
        ),
        noise_dbfs=noise_dbfs,
    )


COUNT_QUESTION = Placement(
    clip_name="count_question", start_s=COUNT_QUESTION_S, role="prompt"
)
TRIALS = (
    Trial(name="control", duration_s=STOP_TRIAL_S, placements=(COUNT_QUESTION,)),
    *(
        Trial(
            name=f"{stop_length}_o{offset_ms:03d}",
            duration_s=STOP_TRIAL_S,
            placements=(
                COUNT_QUESTION,
                Placement(
                    clip_name=f"stop_{stop_length}",
                    start_s=STOP_UNIT_START_S + offset_ms / 1000,
                    role="stop",
                ),
            ),
        )
        for offset_ms in STOP_OFFSETS_MS
        for stop_length in ("short", "long")
    ),
    *(
        greeting_trial(
            f"cold_{volume}_{greeting}",
            greeting=greeting,
            greeting_gain=gain,
            greetings_s=COLD_GREETINGS_S,
            question_s=COLD_QUESTION_S,
        )
        for volume, gain in (("soft", SOFT_GAIN), ("normal", 1.0))
        for greeting in ("hi", "hello")
    ),
    quiet_question_trial("cold_quiet_question", gain=SOFT_GAIN, noise_dbfs=None),
    *(
        greeting_trial(
            f"ongoing_{volume}_hi",
            greeting="hi",
            greeting_gain=gain,
            greetings_s=ONGOING_GREETINGS_S,
            question_s=ONGOING_QUESTION_S,
        )
        for volume, gain in (("soft", SOFT_GAIN), ("normal", 1.0))
    ),
    *(
        greeting_trial(
            f"noisy_cold_{volume}_hi",
            greeting="hi",
            greeting_gain=gain,
            greetings_s=COLD_GREETINGS_S,
            question_s=COLD_QUESTION_S,
            noise_dbfs=NOISE_DBFS,
        )
        for volume, gain in (
            ("softer", SOFTER_GAIN),
            ("soft", SOFT_GAIN),
            ("normal", 1.0),
        )
    ),
    quiet_question_trial(
        "noisy_cold_quiet_question", gain=SOFTER_GAIN, noise_dbfs=NOISE_DBFS
    ),
)
CLIPS = tuple(
    dict.fromkeys(
        placement.clip_name for trial in TRIALS for placement in trial.placements
    )
)


def compose_trial(
    trial: Trial, clip_samples_by_name: dict[str, npt.NDArray[np.int16]]
) -> npt.NDArray[np.int16]:
    """PCM16 input: seeded white noise, then every clip added at its start with its gain."""
    sample_count = round(trial.duration_s * SAMPLE_RATE)
    if trial.noise_dbfs is not None:
        noise_generator = random.Random(NOISE_SEED)
        noise_sigma = (PCM16_MAX + 1) * 10 ** (trial.noise_dbfs / 20)
        samples = np.clip(
            np.trunc(
                [noise_generator.gauss(0, noise_sigma) for _ in range(sample_count)]
            ),
            PCM16_MIN,
            PCM16_MAX,
        )
    else:
        samples = np.zeros(sample_count)
    for placement in trial.placements:
        clip_samples = clip_samples_by_name[placement.clip_name]
        start_sample = round(placement.start_s * SAMPLE_RATE)
        end_sample = start_sample + len(clip_samples)
        if end_sample > sample_count:
            raise ValueError(
                f"{placement.clip_name} at {placement.start_s} s runs past the end of {trial.name}"
            )
        else:
            samples[start_sample:end_sample] = np.clip(
                np.trunc(
                    samples[start_sample:end_sample] + clip_samples * placement.gain
                ),
                PCM16_MIN,
                PCM16_MAX,
            )
    return samples.astype(np.int16)


def write_inputs(clips_dir: Path, inputs_dir: Path) -> list[TrialInput]:
    """Write every trial's input WAV and the trial list that record and report read."""
    clip_samples_by_name: dict[str, npt.NDArray[np.int16]] = {}
    for clip_name in CLIPS:
        clip_path = clips_dir / f"{clip_name}.wav"
        clip_format = soundfile.info(str(clip_path))
        if (clip_format.samplerate, clip_format.channels, clip_format.subtype) != (
            SAMPLE_RATE,
            1,
            "PCM_16",
        ):
            raise ValueError(f"{clip_path} must be PCM16 mono at {SAMPLE_RATE} Hz")
        else:
            clip_samples, _ = soundfile.read(str(clip_path), dtype="int16")
            clip_samples_by_name[clip_name] = clip_samples
    trial_inputs: list[TrialInput] = []
    trial_samples: list[npt.NDArray[np.int16]] = []
    for trial in TRIALS:
        events = [
            SpokenEvent(
                role=placement.role,
                start_s=placement.start_s,
                end_s=placement.start_s
                + len(clip_samples_by_name[placement.clip_name]) / SAMPLE_RATE,
            )
            for placement in trial.placements
        ]
        for event, next_event in zip(events, events[1:]):
            if event.end_s > next_event.start_s:
                raise ValueError(
                    f"In {trial.name}, the clip ending at {event.end_s:.2f} s overlaps the one at {next_event.start_s} s"
                )
            else:
                pass
        samples = compose_trial(trial, clip_samples_by_name)
        trial_samples.append(samples)
        trial_inputs.append(
            TrialInput(
                name=trial.name,
                duration_s=trial.duration_s,
                sha256=hashlib.sha256(samples.tobytes()).hexdigest(),
                events=events,
            )
        )
    inputs_dir.mkdir(parents=True, exist_ok=False)
    for trial_input, samples in zip(trial_inputs, trial_samples):
        soundfile.write(
            str(inputs_dir / f"{trial_input.name}.wav"), samples, SAMPLE_RATE, "PCM_16"
        )
    (inputs_dir / TRIALS_FILE).write_bytes(
        TRIAL_INPUTS.dump_json(trial_inputs, indent=2)
    )
    return trial_inputs


def read_trial_inputs(inputs_dir: Path) -> list[TrialInput]:
    return TRIAL_INPUTS.validate_json((inputs_dir / TRIALS_FILE).read_bytes())
