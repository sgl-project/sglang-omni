# SPDX-License-Identifier: Apache-2.0
"""Bounded audio retention for persistence after a benchmark cohort finishes."""

import wave
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(kw_only=True)
class TtsAudioArtifact:
    chunks: tuple[bytes, ...]
    pcm_format: tuple[int, int, int] | None

    @property
    def size_bytes(self) -> int:
        return sum(len(chunk) for chunk in self.chunks)


@dataclass(kw_only=True)
class TtsAudioArtifacts:
    output_directory: Path
    maximum_bytes: int
    pending: dict[str, TtsAudioArtifact] = field(default_factory=dict, init=False)
    retained_bytes: int = field(default=0, init=False)
    peak_bytes: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.maximum_bytes <= 0:
            raise ValueError("audio_artifact_max_bytes must be positive")
        else:
            pass

    def stage(
        self,
        request_id: str,
        chunks: tuple[bytes, ...],
        pcm_format: tuple[int, int, int] | None = None,
    ) -> str:
        artifact = TtsAudioArtifact(chunks=chunks, pcm_format=pcm_format)
        previous = self.pending.get(request_id)
        retained_bytes = (
            self.retained_bytes
            - (previous.size_bytes if previous is not None else 0)
            + artifact.size_bytes
        )
        if retained_bytes > self.maximum_bytes:
            raise BufferError(
                f"Audio artifacts require {retained_bytes} bytes, exceeding "
                f"--audio-artifact-max-bytes={self.maximum_bytes}. "
                "Increase the limit or reduce the number of samples."
            )
        else:
            self.pending[request_id] = artifact
            self.retained_bytes = retained_bytes
            self.peak_bytes = max(self.peak_bytes, retained_bytes)
        return str(self.output_directory / f"{request_id}.wav")

    def flush(self) -> None:
        """Persist the cohort before publishing benchmark results."""
        for request_id in list(self.pending):
            artifact = self.pending[request_id]
            audio_path = self.output_directory / f"{request_id}.wav"
            if artifact.pcm_format is None:
                with audio_path.open("wb") as audio_file:
                    for chunk in artifact.chunks:
                        if audio_file.write(chunk) != len(chunk):
                            raise OSError(f"Incomplete audio write: {audio_path}")
                        else:
                            pass
            else:
                sample_rate, channels, sample_width = artifact.pcm_format
                with wave.open(str(audio_path), "wb") as audio_file:
                    audio_file.setframerate(sample_rate)
                    audio_file.setnchannels(channels)
                    audio_file.setsampwidth(sample_width)
                    audio_file.setnframes(
                        artifact.size_bytes // (channels * sample_width)
                    )
                    for chunk in artifact.chunks:
                        audio_file.writeframesraw(chunk)
            self.retained_bytes -= artifact.size_bytes
            self.pending.pop(request_id)

    def clear(self) -> None:
        self.pending.clear()
        self.retained_bytes = 0
