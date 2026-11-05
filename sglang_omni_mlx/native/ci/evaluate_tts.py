# SPDX-License-Identifier: Apache-2.0
"""Run SeedTTS generation and ASR scoring with native MLX servers."""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import shutil
import socket
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from huggingface_hub import snapshot_download

from benchmarks.benchmarker.utils import STARTUP_TIMEOUT, stop_server, wait_healthy
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.eval.benchmark_tts_seedtts import (
    TtsSeedttsBenchmarkConfig,
    run_tts_seedtts_benchmark,
    run_tts_seedtts_transcribe,
)
from benchmarks.metrics.performance import print_speed_summary

logger = logging.getLogger(__name__)
ASR_SAMPLE_RATE = 16000
REFERENCE_SAMPLE_RATE = 16000


def prepare_reference_samples(config: TtsSeedttsBenchmarkConfig) -> list[SampleInput]:
    if config.sample_offset < 0:
        raise ValueError("--sample-offset must be nonnegative")
    else:
        pass
    limit = (
        config.sample_offset + config.max_samples
        if config.max_samples is not None
        else None
    )
    samples = load_seedtts_samples(config.meta, limit, split=config.lang)[
        config.sample_offset :
    ]
    prepared = []
    for sample in samples:
        audio_samples, sample_rate = sf.read(
            sample.ref_audio, dtype="float32", always_2d=True
        )
        waveform = torch.from_numpy(audio_samples).mean(dim=1)
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, REFERENCE_SAMPLE_RATE
        )
        with io.BytesIO() as buffer:
            sf.write(
                buffer,
                waveform.numpy(),
                REFERENCE_SAMPLE_RATE,
                format="WAV",
                subtype="PCM_16",
            )
            reference_url = "data:audio/wav;base64," + base64.b64encode(
                buffer.getvalue()
            ).decode("ascii")
        prepared.append(replace(sample, ref_audio=reference_url))
    return prepared


def prepare_asr_audio(config: TtsSeedttsBenchmarkConfig) -> TtsSeedttsBenchmarkConfig:
    # note (altale): native ASR requires 16 kHz mono; use copies to preserve TTS audio.
    output_directory = Path(config.output_dir)
    asr_directory = output_directory / "asr"
    asr_directory.mkdir(parents=True, exist_ok=True)
    generated = json.loads((output_directory / "generated.json").read_text())
    for index, entry in enumerate(generated):
        if not entry["is_success"]:
            continue
        else:
            pass
        audio_path = Path(entry["wav_path"])
        if not audio_path.is_file():
            continue
        else:
            pass
        audio_samples, sample_rate = sf.read(
            audio_path, dtype="float32", always_2d=True
        )
        destination = asr_directory / f"{index}.wav"
        if sample_rate == ASR_SAMPLE_RATE and audio_samples.shape[1] == 1:
            shutil.copyfile(audio_path, destination)
        else:
            waveform = torch.from_numpy(audio_samples).mean(dim=1)
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, ASR_SAMPLE_RATE
            )
            pcm16 = (waveform.clamp(-1, 1) * 32767).to(torch.int16).numpy()
            sf.write(destination, pcm16, ASR_SAMPLE_RATE, subtype="PCM_16")
        entry["wav_path"] = str(destination.resolve())
    (asr_directory / "generated.json").write_text(
        json.dumps(generated, indent=2, ensure_ascii=False)
    )
    speed_results = output_directory / "speed_results.json"
    if speed_results.is_file():
        shutil.copyfile(speed_results, asr_directory / speed_results.name)
    else:
        pass
    return replace(config, output_dir=str(asr_directory))


@contextmanager
def managed_native_server(
    *,
    binary: Path,
    model_path: str,
    revision: str | None,
    port: int,
    host: str,
    log_file: Path,
    timeout_seconds: int,
) -> Iterator[None]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_check:
        # note (altale): match CivetWeb so TTS connections in TIME_WAIT do not block ASR.
        port_check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port_check.bind((host, port))
    if Path(model_path).is_dir():
        model_directory = model_path
    else:
        model_directory = snapshot_download(model_path, revision=revision)
    command = [
        str(binary),
        "--model-path", model_directory,
        "--model-name", model_path,
        "--host", host,
        "--port", str(port),
    ]  # fmt: skip
    logger.info(f"Starting native server: {' '.join(command)}")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w") as server_log:
        process = subprocess.Popen(
            command,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_healthy(process, port, log_file, timeout=timeout_seconds)
            yield
        finally:
            if process.poll() is None:
                stop_server(process)
            else:
                pass


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-bin", type=Path, required=True)
    parser.add_argument("--model", default="inclusionAI/Ming-omni-tts-16.8B-A3B")
    parser.add_argument("--asr-model-path", default="mlx-community/Qwen3-ASR-1.7B-8bit")
    parser.add_argument(
        "--model-revision", help="Hub revision; ignored for local directories."
    )
    parser.add_argument(
        "--asr-model-revision", help="Hub revision; ignored for local directories."
    )
    parser.add_argument("--meta", default="zhaochenyang20/seed-tts-eval-arrow")
    parser.add_argument("--lang", choices=("en", "zh"), default="en")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument(
        "--voice-clone",
        action="store_true",
        help="Condition generation on the SeedTTS reference audio and transcript.",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--server-timeout", type=int, default=STARTUP_TIMEOUT)
    parser.add_argument("--output-dir", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--generate-only", action="store_true")
    mode.add_argument("--transcribe-only", action="store_true")
    arguments = parser.parse_args()
    if not 1 <= arguments.max_new_tokens <= 256:
        parser.error("--max-new-tokens must be in [1,256]")
    elif arguments.warmup < 0:
        parser.error("--warmup must be nonnegative")
    else:
        pass
    binaries: list[str] = []
    if not arguments.transcribe_only:
        binaries.append("ming_tts_server")
    else:
        pass
    if not arguments.generate_only:
        binaries.append("qwen3_asr_server")
    else:
        pass
    for binary in binaries:
        if not os.access(arguments.runtime_bin / binary, os.X_OK):
            parser.error(
                f"Native executable not found: {arguments.runtime_bin / binary}"
            )
        else:
            pass
    config = TtsSeedttsBenchmarkConfig(
        model=arguments.model,
        asr_model_path=arguments.asr_model_path,
        meta=arguments.meta,
        lang=arguments.lang,
        port=arguments.port,
        max_samples=arguments.max_samples,
        sample_offset=arguments.sample_offset,
        max_new_tokens=arguments.max_new_tokens,
        warmup=arguments.warmup,
        stream=arguments.stream,
        response_format="pcm" if arguments.stream else "wav",
        output_dir=arguments.output_dir,
        voice_clone=arguments.voice_clone,
        concurrency=1,
        asr_concurrency=1,
        collect_stream_outcomes=False,
    )
    log_directory = Path(config.output_dir) / "server_logs"
    if not arguments.transcribe_only:
        samples = prepare_reference_samples(config) if config.voice_clone else None
        with managed_native_server(
            binary=arguments.runtime_bin / "ming_tts_server",
            model_path=config.model,
            revision=arguments.model_revision,
            port=config.port,
            host=config.host,
            log_file=log_directory / "tts_server.log",
            timeout_seconds=arguments.server_timeout,
        ):
            results = asyncio.run(run_tts_seedtts_benchmark(config, samples=samples))
            print_speed_summary(
                results["summary"], config.model, concurrency=config.concurrency
            )
    else:
        pass
    if not arguments.generate_only:
        asr_config = prepare_asr_audio(config)
        with managed_native_server(
            binary=arguments.runtime_bin / "qwen3_asr_server",
            model_path=config.asr_model_path,
            revision=arguments.asr_model_revision,
            port=config.port,
            host=config.host,
            log_file=log_directory / "asr_server.log",
            timeout_seconds=arguments.server_timeout,
        ):
            run_tts_seedtts_transcribe(asr_config, asr_router_port=config.port)
    else:
        pass


if __name__ == "__main__":
    main()
