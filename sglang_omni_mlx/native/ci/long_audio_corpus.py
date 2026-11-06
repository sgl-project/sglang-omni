# SPDX-License-Identifier: Apache-2.0
"""Build reproducible long recordings without cutting reference transcripts."""

from __future__ import annotations

import json
import random
import wave
from dataclasses import dataclass
from pathlib import Path

from sglang_omni_mlx.native.ci.provision import sha256

SAMPLE_RATE = 16000
SAMPLE_BYTES = 2


@dataclass(frozen=True, kw_only=True)
class Clip:
    clip_id: str
    path: Path
    reference: str
    frames: int


@dataclass(frozen=True, kw_only=True)
class Recording:
    requested_seconds: float
    audio_seconds: float
    seed: int
    clip_ids: list[str]
    clip_frames: list[int]
    reference: str
    wav_sha256: str


def load_clips(corpus_directory: Path, language: str) -> list[Clip]:
    clips: list[Clip] = []
    for line in (corpus_directory / "manifest.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record["lang"] == language and record["stratum"] in (
            f"{language}_short",
            f"{language}_mid",
        ):
            path = corpus_directory / "clips" / f"{record['id']}.wav"
            with wave.open(str(path), "rb") as reader:
                if (
                    reader.getframerate(),
                    reader.getnchannels(),
                    reader.getsampwidth(),
                ) != (SAMPLE_RATE, 1, SAMPLE_BYTES):
                    raise ValueError(f"Expected mono 16 kHz PCM16: {path}")
                else:
                    frames = reader.getnframes()
            clips.append(
                Clip(
                    clip_id=record["id"],
                    path=path,
                    reference=record["reference"],
                    frames=frames,
                )
            )
        else:
            pass
    if len(clips) < 2:
        raise ValueError(f"Need at least two {language} clips")
    else:
        return clips


def build_recording(
    clips: list[Clip], seconds: float, seed: int, destination: Path
) -> Recording:
    """End at a complete clip, at or just beyond the requested duration."""
    generator = random.Random(seed)
    selected: list[Clip] = []
    frame_count = 0
    with wave.open(str(destination), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(SAMPLE_BYTES)
        writer.setframerate(SAMPLE_RATE)
        while frame_count < seconds * SAMPLE_RATE:
            cycle = generator.sample(clips, len(clips))
            if selected and cycle[0].clip_id == selected[-1].clip_id:
                cycle[0], cycle[1] = cycle[1], cycle[0]
            else:
                pass
            for clip in cycle:
                with wave.open(str(clip.path), "rb") as reader:
                    writer.writeframesraw(reader.readframes(reader.getnframes()))
                selected.append(clip)
                frame_count += clip.frames
                if frame_count >= seconds * SAMPLE_RATE:
                    break
                else:
                    pass
    return Recording(
        requested_seconds=seconds,
        audio_seconds=frame_count / SAMPLE_RATE,
        seed=seed,
        clip_ids=[clip.clip_id for clip in selected],
        clip_frames=[clip.frames for clip in selected],
        reference=" ".join(clip.reference for clip in selected),
        wav_sha256=sha256(destination),
    )


def chunk_frame_counts(clip_frames: list[int], chunk_seconds: float) -> list[int]:
    """Whole clips per request, at most chunk_seconds unless one clip alone is longer."""
    counts = [0]
    for frames in clip_frames:
        if counts[-1] > 0 and counts[-1] + frames > chunk_seconds * SAMPLE_RATE:
            counts.append(frames)
        else:
            counts[-1] += frames
    return counts
