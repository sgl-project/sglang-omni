# SPDX-License-Identifier: Apache-2.0
"""Speaker-attributed transcripts of the streaming cascade on multi-party clips.

Streams each Movies800Time clip with two or more speakers and a pause the VAD
cuts at through one Session of the cascade's vad and speaker_transcription nodes, joins the final transcripts'
segments into one session transcript and scores it with cpCER, CER without
speakers and timestamp DER. Each context_s is one run; the whole clip sent to
the service in one request is the reference run.

Start a MOSS-Transcribe-Diarize service first, for example the CUDA server:

    sgl-omni serve --model-path OpenMOSS-Team/MOSS-Transcribe-Diarize --port 8002

or the native MLX server on Apple Silicon:

    moss_transcribe_diarize_server --model-path DIR --model-name OpenMOSS-Team/MOSS-Transcribe-Diarize --port 8002

Then:

    python -m benchmarks.eval.benchmark_cascade_speaker_transcription \
        --base-url http://127.0.0.1:8002/v1 --context-s 20 0 \
        --output-dir results/cascade_speaker_transcription
"""

import argparse
import asyncio
import io
import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Final

import httpx
import soundfile

from benchmarks.metrics.transcribe_diarize_metrics import (
    DiarizationRow,
    compute_diarization_metrics,
)
from benchmarks.tasks.transcribe_diarize import (
    MOVIES800_REPO_ID,
    Movies800Sample,
    load_movies800_samples,
)
from sglang_omni.models.streaming_cascade.base import SpeakerSegment
from sglang_omni.models.streaming_cascade.config import StreamingCascadePipelineConfig
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.proto import OmniRequest
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import (
    AUDIO,
    PCM16,
    PCM16_BYTES_PER_SAMPLE,
    SESSION_CONTROL,
    SESSION_INPUT_SAMPLE_RATE_HZ,
    SPEECH_STARTED,
    SPEECH_STOPPED,
    TRANSCRIPT_FORMAT,
    control_type,
)

REFERENCE_SEGMENT: Final = re.compile(
    r"\[(\d+(?:\.\d+)?)\]\[S0*(\d+)\](.*?)\[(\d+(?:\.\d+)?)\]", re.DOTALL
)
# note (fweilun): a pause the cascade's VAD cuts at, 800 ms of silence, with room for where it places the speech end.
PAUSE_S: Final = 1.0
SESSION_END_TIMEOUT_S: Final = 300.0
INPUT_BYTES_PER_MS: Final = (
    PCM16_BYTES_PER_SAMPLE * SESSION_INPUT_SAMPLE_RATE_HZ // 1000
)
METRICS: Final = ("cp_cer", "cer_no_spk", "delta_cer", "speaker_timestamp_der")


@dataclass(frozen=True, kw_only=True)
class SessionTranscript:
    markup: str
    utterances: int


@dataclass(frozen=True, kw_only=True)
class RunReport:
    scores: dict[str, float | int | None]
    utterances: int
    predictions: list[str]


def is_multi_party(reference: str) -> bool:
    """Whether the reference has two or more speakers and a pause the cascade cuts at, so its utterances must agree on who is who."""
    segments = [
        (float(start), speaker, float(end))
        for start, speaker, _, end in REFERENCE_SEGMENT.findall(reference)
    ]
    has_pause = any(
        later[0] - earlier[2] >= PAUSE_S
        for earlier, later in zip(segments, segments[1:])
    )
    return len({speaker for _, speaker, _ in segments}) >= 2 and has_pause


def read_pcm16(path: str) -> bytes:
    samples, sample_rate_hz = soundfile.read(path, dtype="int16")
    if samples.ndim != 1 or sample_rate_hz != SESSION_INPUT_SAMPLE_RATE_HZ:
        raise ValueError(f"{path} must be mono 16 kHz audio")
    else:
        return samples.astype("<i2").tobytes()


def segment_markup(segments: list[SpeakerSegment]) -> str:
    return "".join(
        f"[{segment.start_ms / 1000:.2f}][{segment.speaker}]"
        f"{segment.text}[{segment.end_ms / 1000:.2f}]"
        for segment in segments
    )


def cascade_config(
    base_url: str, model: str, context_s: float
) -> StreamingCascadePipelineConfig:
    return StreamingCascadePipelineConfig(
        model_path="speaker-transcription-benchmark",
        session_nodes={
            "vad": {"stages": ["vad"], "inputs": ["input"]},
            "asr": {
                "kind": "speaker_transcription",
                "inputs": ["input", "vad:session"],
                "base_url": base_url,
                "model": model,
                "context_s": context_s,
            },
        },
        session_publish={"vad": ["session"], "asr": ["transcript"]},
    )


async def stream_clip(
    runner: MultiProcessPipelineRunner,
    config: StreamingCascadePipelineConfig,
    pcm: bytes,
) -> SessionTranscript:
    """The session transcript of one clip streamed as fast as the Session accepts it."""
    coordinator = runner.coordinator
    identity = await coordinator.open_session(
        OmniRequest(None),
        limits=replace(
            config.session_limits(), operation_timeout_s=SESSION_END_TIMEOUT_S
        ),
    )
    outputs = coordinator.session_outputs(identity)
    acknowledged: asyncio.Queue[int] = asyncio.Queue()
    segments: list[SpeakerSegment] = []
    counts = {SPEECH_STARTED: 0, SPEECH_STOPPED: 0, "finals": 0}
    is_input_done = asyncio.Event()
    is_session_done = asyncio.Event()

    def settle() -> None:
        # note (fweilun): the recognizer answers each utterance the VAD closed, and the first opens with the audio before its speech.started.
        if (
            is_input_done.is_set()
            and counts[SPEECH_STOPPED] == counts[SPEECH_STARTED]
            and counts["finals"] >= max(counts[SPEECH_STARTED], 1)
        ):
            is_session_done.set()
        else:
            pass

    async def receive() -> None:
        async for chunk in outputs:
            if chunk.kind == "input_done":
                acknowledged.put_nowait(chunk.input_seq)
            elif chunk.format == TRANSCRIPT_FORMAT and chunk.payload["is_final"]:
                counts["finals"] += 1
                segments.extend(
                    SpeakerSegment(**segment) for segment in chunk.payload["segments"]
                )
            elif chunk.format == SESSION_CONTROL and control_type(chunk) in counts:
                counts[control_type(chunk)] += 1
            else:
                pass
            settle()

    async def send() -> None:
        unit_bytes = config.unit_ms * INPUT_BYTES_PER_MS
        offsets = range(0, len(pcm), unit_bytes)
        for sequence, offset in enumerate(offsets):
            audio = pcm[offset : offset + unit_bytes]
            await coordinator.append_session(
                identity,
                TimedChunk(
                    AUDIO,
                    offset / INPUT_BYTES_PER_MS,
                    len(audio) / INPUT_BYTES_PER_MS,
                    sequence,
                    audio,
                    format=PCM16,
                ),
            )
            await acknowledged.get()
        await coordinator.append_session(
            identity,
            TimedChunk(
                AUDIO,
                len(pcm) / INPUT_BYTES_PER_MS,
                0,
                len(offsets),
                b"",
                format=PCM16,
                eos=True,
            ),
        )
        await acknowledged.get()
        is_input_done.set()
        settle()

    receiver = asyncio.create_task(receive())
    finished = asyncio.create_task(is_session_done.wait())
    try:
        await send()
        await asyncio.wait(
            {receiver, finished},
            timeout=SESSION_END_TIMEOUT_S,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if receiver.done():
            receiver.result()
            raise RuntimeError("the session ended before its last transcript")
        elif not finished.done():
            raise TimeoutError("the last transcript did not arrive")
        else:
            pass
    finally:
        for task in (receiver, finished):
            task.cancel()
        await asyncio.gather(receiver, finished, return_exceptions=True)
        await outputs.aclose()
        await coordinator.close_session(identity)
    return SessionTranscript(
        markup=segment_markup(segments), utterances=counts["finals"]
    )


def whole_clip_markup(
    client: httpx.Client, base_url: str, model: str, path: str
) -> str:
    samples, sample_rate_hz = soundfile.read(path, dtype="int16")
    wav = io.BytesIO()
    soundfile.write(wav, samples, sample_rate_hz, format="WAV", subtype="PCM_16")
    response = client.post(
        f"{base_url.rstrip('/')}/audio/transcriptions",
        data={"model": model, "response_format": "json"},
        files={"file": ("audio.wav", wav.getvalue(), "audio/wav")},
    )
    response.raise_for_status()
    return str(response.json()["text"])


def scores(
    samples: list[Movies800Sample], predictions: list[str]
) -> dict[str, float | int | None]:
    result = compute_diarization_metrics(
        [
            DiarizationRow(
                sample.sample_id, sample.audio_path, sample.expected_text, prediction
            )
            for sample, prediction in zip(samples, predictions, strict=True)
        ]
    )
    return {name: result.metrics_percent[name] for name in (*METRICS, "count")}


async def run(arguments: argparse.Namespace) -> tuple[list[str], dict[str, RunReport]]:
    samples = [
        sample
        for sample in load_movies800_samples(
            arguments.repo_id,
            "validation",
            "audio",
            "transcription",
            expected_sample_count=None,
        )
        if is_multi_party(sample.expected_text)
    ][: arguments.max_samples]
    reports: dict[str, RunReport] = {}
    with httpx.Client(timeout=SESSION_END_TIMEOUT_S) as client:
        whole = [
            whole_clip_markup(
                client, arguments.base_url, arguments.model, sample.audio_path
            )
            for sample in samples
        ]
    reports["whole_clip"] = RunReport(
        scores=scores(samples, whole), utterances=len(samples), predictions=whole
    )
    for context_s in arguments.context_s:
        config = cascade_config(arguments.base_url, arguments.model, context_s)
        runner = MultiProcessPipelineRunner(config)
        await runner.start()
        try:
            transcripts = [
                await stream_clip(runner, config, read_pcm16(sample.audio_path))
                for sample in samples
            ]
        finally:
            await runner.stop()
        predictions = [transcript.markup for transcript in transcripts]
        reports[f"context_s={context_s:g}"] = RunReport(
            scores=scores(samples, predictions),
            utterances=sum(transcript.utterances for transcript in transcripts),
            predictions=predictions,
        )
    return [sample.sample_id for sample in samples], reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="The MOSS-Transcribe-Diarize service, including /v1",
    )
    parser.add_argument("--model", default="OpenMOSS-Team/MOSS-Transcribe-Diarize")
    parser.add_argument("--context-s", type=float, nargs="+", default=[20.0])
    parser.add_argument("--repo-id", default=MOVIES800_REPO_ID)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    sample_ids, reports = asyncio.run(run(arguments))
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    (arguments.output_dir / "report.json").write_text(
        json.dumps(
            {
                "samples": sample_ids,
                **{name: asdict(report) for name, report in reports.items()},
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    columns = (*METRICS, "count")
    print(f"{'run':<16}" + "".join(f"{name:>24}" for name in columns))
    for name, report in reports.items():
        print(
            f"{name:<16}"
            + "".join(
                (
                    f"{report.scores[column]:>24.2f}"
                    if isinstance(report.scores[column], float)
                    else f"{report.scores[column]!s:>24}"
                )
                for column in columns
            )
        )


if __name__ == "__main__":
    main()
