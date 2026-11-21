#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record and re-check per-model accuracy and speed for the models Intel XPU serves.

A run serves each model with the recipe from docs/other_platforms/installation_xpu.md, sends a fixed request set, and compares the result against a committed baseline, so "is this branch slower or less correct than the last known-good tree" has an answer that does not depend on anyone remembering numbers.

Four comparison classes, one per kind of comparison the field deserves:

    exact   text output and sample rate. Greedy decode with the radix cache off makes a transcript or a chat answer reproducible, so these must match character for character.
    atmost  round-trip word error rate, held under an absolute ceiling.
    close   loudness. Reproducible under a seed, but a kernel change moves the vocoder's floats, so it gets a band rather than equality.
    speed   latency and throughput, with a directional relative floor.

A fifth group, observed, is recorded and never gated. It exists for the two models whose server refuses a seed (Ming-Omni-TTS, ZONOS2): they choose their own utterance length, which moved 2x between repeats, so length, loudness and the throughput derived from length cannot gate anything. Those models gate on sample rate, latency, and the round-trip WER ceiling, which tolerates a varying utterance while still catching one that stops being intelligible.

Reproducibility is asserted, not assumed: an exact field that disagrees across repeats fails the run rather than being written to the baseline. That check is what showed Qwen3-TTS emitting 92160 to 134400 frames for one input until the request carried a seed, and what proves the seed works now that it does.

Speed is the median of repeated samples after two discarded warmups, at concurrency 1. Concurrency 1 is deliberate: Qwen3-TTS throughput at concurrency 4 has been seen to swing by more than 1.5x between runs of one tree on this hardware, which is too noisy to gate on.

A speed field is allowed the larger of a floor and three times the spread it showed when it was recorded, because one tolerance for every model is wrong in both directions at once: in one B60 sweep Qwen3-ASR latency wobbled 0.4 percent and Whisper 8.9 percent, so a shared 10 percent would let a 5 percent ASR regression through while still flaring on Whisper.

That spread is the trimmed one, with the best and worst sample dropped, while the full range is reported beside it. A single stalled sample is a real event worth seeing and a bad description of a model's dispersion: nine Nemotron 3.5 ASR samples sat inside nine percent except one, and sizing on the full range would have handed that entry a 201 percent tolerance -- wide enough to miss a doubling.

With --launches above 1 the speed figure is the slowest launch rather than the median of them. These models turn out to be modal, not noisy, and the slow mode is the only anchor a gate can hold without firing on a clean run.

Baselines are per environment. The file holds one entry per host, card model, cards and torch/sglang version, and a check against an environment the file has never seen stops rather than comparing numbers that were never comparable.

Usage:
    xpu_model_baseline.py                  # check every model against the baseline
    xpu_model_baseline.py --list
    xpu_model_baseline.py --models qwen3-asr,qwen3-tts --record
    xpu_model_baseline.py --models qwen3-asr,qwen3-tts
    xpu_model_baseline.py --models qwen3-tts --calibrate --repeats 9
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.benchmarker.utils import (  # noqa: E402  (needs REPO_ROOT on sys.path)
    start_server_from_cmd,
    stop_server,
)

DATA_DIR = REPO_ROOT / "tests" / "data"
REFERENCE_AUDIO = DATA_DIR / "query_to_cars.wav"
REFERENCE_TEXT = "How many cars are there in the picture?"
SPOKEN_TEXT = "The quick brown fox jumps over the lazy dog."
ARITHMETIC_PROMPT = "Answer with the number only: 17 plus 25?"
MUSIC_PROMPT = "A short calm piano melody."

DEFAULT_BASELINE = REPO_ROOT / "scripts" / "xpu" / "xpu_model_baseline.json"

# Engine knobs that make a stage reproducible, applied to every engine stage a model declares.
#
# SGLang's own enable_deterministic_inference is deliberately absent. With no explicit
# attention_backend it rewrites the backend to fa3 on anything that is not SM100/SM120,
# which is CUDA-only and kills the server on XPU; naming a backend to avoid that would pin
# a kernel choice and change the very speed this script measures. Whisper cannot use it at
# all, because XPU pins torch_native cross-attention and that is not in SGLang's
# deterministic backend list.
DETERMINISM_ENGINE_ARGS = {
    "disable_radix_cache": "true",
    "random_seed": "0",
}


@dataclass(frozen=True)
class XpuModel:
    """One servable model: how to launch it, how to ask it, how to score it."""

    key: str
    name: str
    model_path: str
    probe: str
    cards: int = 1
    serve_args: tuple[str, ...] = ()
    # Why this model cannot be measured here. Non-empty means skip, not fail.
    unservable: str = ""
    # Variants entry the serve args select. Stage derivation has to follow it: --text-only
    # drops the talker, and aiming a determinism override at a stage the chosen pipeline
    # does not have is rejected at engine build.
    config_variant: str = ""
    # ZONOS2 rejects a seed outright: its sampling uses the shared device RNG. Without one
    # the generated length is not reproducible, so the frame count cannot be held exact.
    seedless: bool = False
    payload: dict[str, Any] = field(default_factory=dict)
    # Extra process environment the guide's recipe calls for.
    env: dict[str, str] = field(default_factory=dict)
    # Another model key, served alongside, that transcribes this model's audio back to text.
    judge: str = ""
    startup_seconds: int = 900


MODELS: tuple[XpuModel, ...] = (
    XpuModel(
        key="qwen3-asr",
        name="Qwen3-ASR",
        model_path="Qwen/Qwen3-ASR-1.7B",
        probe="transcription",
        # The metadata glue graph is CUDA-only and leaves ASR without a transcript on main.
        env={"SGLANG_ENABLE_METADATA_GLUE_GRAPH": "0"},
    ),
    XpuModel(
        key="whisper",
        name="Whisper large-v3",
        model_path="openai/whisper-large-v3",
        probe="transcription",
    ),
    XpuModel(
        key="fun-asr",
        name="Fun-ASR-Nano",
        model_path="FunAudioLLM/Fun-ASR-Nano-2512-hf",
        probe="transcription",
    ),
    XpuModel(
        key="qwen3-tts",
        name="Qwen3-TTS Base",
        model_path="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        probe="speech",
        payload={
            "voice": "default",
            "ref_audio": str(REFERENCE_AUDIO),
            "ref_text": REFERENCE_TEXT,
        },
        judge="qwen3-asr",
    ),
    XpuModel(
        key="qwen3-tts-cv",
        name="Qwen3-TTS CustomVoice",
        model_path="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        probe="speech",
        serve_args=("--config", "examples/configs/qwen3_tts_1_7b_customvoice.yaml"),
        payload={"voice": "Ryan", "task_type": "CustomVoice", "language": "English"},
        judge="qwen3-asr",
    ),
    XpuModel(
        key="cosyvoice3",
        name="Fun-CosyVoice3",
        model_path="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        probe="speech",
        unservable="needs the fun-cosyvoice3 extra; module cosyvoice is absent",
        payload={
            "voice": "default",
            "ref_audio": str(REFERENCE_AUDIO),
            "ref_text": REFERENCE_TEXT,
        },
        judge="qwen3-asr",
    ),
    XpuModel(
        key="zonos2",
        name="ZONOS2",
        model_path="Zyphra/zonos2",
        probe="speech",
        seedless=True,
        unservable="needs the dac codec module, which this install does not have",
        payload={"voice": "default"},
        judge="qwen3-asr",
    ),
    XpuModel(
        key="dots-tts",
        name="dots.tts SOAR",
        model_path="dots-studio/dots.tts-soar",
        probe="speech",
        serve_args=("--config", "examples/configs/dots_tts_soar.yaml"),
        payload={
            "voice": "default",
            "references": [
                {"audio_path": str(REFERENCE_AUDIO), "text": REFERENCE_TEXT}
            ],
        },
        judge="qwen3-asr",
    ),
    XpuModel(
        key="minimax-music3",
        name="MiniMax Music 3",
        model_path="MiniMaxAI/MiniMax-Music3",
        probe="music",
        cards=2,
        # The AR stage turns on deterministic inference and names no backend, and
        # SGLang 0.5.23's deterministic fallback picks fa3 below SM100 without checking
        # the device, so the stage asserts "requires SM>=80 and SM<=90" on XPU.
        # Fixed upstream in sgl-project/sglang#46623; drop this once omni pins a
        # SGLang that carries it.
        serve_args=(
            "--mem-fraction-static",
            "0.7",
            "--minimax_music3_ar.engine.attention_backend",
            "triton",
        ),
        # Music takes a style prompt, not a voice, and an explicit token cap: the default
        # length makes the run unbounded and not comparable between sweeps.
        payload={
            "instructions": (
                "An energetic arena rock anthem with distorted electric guitars, punchy live "
                "drums and a soaring male vocal at 130 BPM, wide stereo image"
            ),
            "max_new_tokens": 750,
        },
    ),
    XpuModel(
        key="ming-tts",
        name="Ming-Omni-TTS",
        model_path="inclusionAI/Ming-omni-tts-16.8B-A3B",
        probe="speech",
        cards=2,
        serve_args=(
            "--config",
            "examples/configs/ming_omni_tts.yaml",
            "--tts_engine.tp_size",
            "2",
            "--tts_engine.gpu",
            "[0, 1]",
            "--reference_encode.gpu",
            "0",
            "--audio_decode.gpu",
            "0",
            "--tts_engine.gpu_memory_fraction",
            "0.78",
            "--tts_engine.engine.mem_fraction_static",
            "0.78",
        ),
        payload={"voice": "default"},
        # FlowLoss sampling is unseeded; the request builder rejects a seed outright
        seedless=True,
        # Unseeded, so the WER ceiling is the only accuracy gate it can carry.
        judge="qwen3-asr",
        startup_seconds=1800,
    ),
    XpuModel(
        key="qwen3-omni-text",
        name="Qwen3-Omni thinker (text only)",
        model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        probe="chat",
        cards=8,
        config_variant="text",
        serve_args=(
            # The guide's recipe no longer starts: an XPU-only reserve added in sglang
            # 0.5.20 walks the auto fraction down to 0.302, and the loader then refuses
            # with "Loaded weights leave no GPU memory for the KV cache ... raise above
            # 0.304". 0.435 is what the auto path resolved to before that reserve.
            "--thinker.engine.mem_fraction_static",
            "0.435",
            # The XPU platform hands the vision encoder sgl-kernel's joint RoPE
            # unconditionally, and that kernel only compiles for rope_dim in
            # [64, 80, 96, 128, 256, 512]. This encoder's head dim is 1152/16 = 72, so
            # the server's own warmup request dies with "Unsupported rope_dim for XPU
            # RoPE: 72" once all eight ranks have loaded. --text-only still builds the
            # encoder, so the only way to a text baseline is to not send that request.
            # Drop this once the platform checks the dim before claiming the kernel.
            "--skip-server-warmup",
            "--text-only",
            "--thinker.process",
            "thinker",
            "--thinker.tp_size",
            "8",
            "--thinker.gpu",
            "[0, 1, 2, 3, 4, 5, 6, 7]",
        ),
        startup_seconds=2400,
    ),
    XpuModel(
        key="moss-transcribe-diarize",
        name="MOSS-Transcribe-Diarize",
        model_path="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        probe="transcription",
        # Its transcript carries diarization tokens ("145s01 how many cars ... picture441"),
        # so the word error rate reads around 0.25 with every word correct. The exact
        # transcript is the real gate here; the rate only has to stop rising.
        serve_args=(
            "--asr.engine.max_running_requests",
            "16",
            "--asr.engine.cuda_graph_max_bs",
            "16",
            "--asr.engine.enable_torch_compile",
            "false",
            "--mem-fraction-static",
            "0.70",
        ),
    ),
    XpuModel(
        key="nemotron35-asr",
        name="Nemotron 3.5 ASR",
        model_path="nvidia/nemotron-3.5-asr-streaming-0.6b",
        probe="transcription",
    ),
    XpuModel(
        key="arkasr",
        name="ARK-ASR",
        model_path="AutoArk-AI/ARK-ASR-3B",
        probe="transcription",
    ),
    XpuModel(
        key="higgs",
        name="Higgs Audio v3 TTS",
        model_path="bosonai/higgs-audio-v3-tts-4b",
        probe="speech",
        serve_args=("--allowed-local-media-path", "docs/_static/audio"),
        payload={"voice": "default"},
        judge="qwen3-asr",
        # 8.7 GB of weights: a cold pull does not finish inside the default window.
        startup_seconds=2400,
    ),
    XpuModel(
        key="auk",
        name="AuK",
        model_path="tencent/AuK",
        probe="speech",
        # Without reference audio AuK requires a generation length.
        payload={
            "voice": "default",
            "instructions": "A warm, relaxed voice.",
            "stage_params": {"auk_engine": {"gen_seconds": 4}},
        },
        judge="qwen3-asr",
    ),
    XpuModel(
        key="auk-flash",
        name="AuK-Flash",
        model_path="tencent/AuK-Flash",
        probe="speech",
        payload={
            "voice": "default",
            "instructions": "A warm, relaxed voice.",
            "stage_params": {"auk_engine": {"gen_seconds": 4}},
        },
        judge="qwen3-asr",
    ),
    XpuModel(
        key="qwen3-tts-vd",
        name="Qwen3-TTS VoiceDesign",
        model_path="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        probe="speech",
        serve_args=("--config", "examples/configs/qwen3_tts_1_7b_voicedesign.yaml"),
        payload={
            "voice": "default",
            "task_type": "VoiceDesign",
            "instructions": "A warm, natural young adult voice.",
        },
        judge="qwen3-asr",
    ),
    XpuModel(
        key="dots-tts-base",
        name="dots.tts Base",
        model_path="dots-studio/dots.tts-base",
        probe="speech",
        # Base shares SOAR's config; the solver settings live there, not on the CLI.
        serve_args=("--config", "examples/configs/dots_tts_soar.yaml"),
        payload={
            "voice": "default",
            "references": [
                {"audio_path": str(REFERENCE_AUDIO), "text": REFERENCE_TEXT}
            ],
        },
        judge="qwen3-asr",
    ),
    XpuModel(
        key="dots-tts-mf",
        name="dots.tts MF",
        # The config pins a revision, so the path carries it: passing the bare repo id
        # would override the pin and measure a checkpoint nobody validated.
        model_path="dots-studio/dots.tts-mf@c28105adc8228143392b4e346994ff613ee48a06",
        probe="speech",
        # MF is the one dots checkpoint with continuous batching; the config ships 16
        # slots and the guide validated 4 on a 24 GB B60.
        serve_args=(
            "--config",
            "examples/configs/dots_tts.yaml",
            "--latent_engine.engine.max_running_requests",
            "4",
            "--latent_engine.engine.cuda_graph_max_bs",
            "4",
        ),
        payload={
            "voice": "default",
            "references": [
                {"audio_path": str(REFERENCE_AUDIO), "text": REFERENCE_TEXT}
            ],
        },
        judge="qwen3-asr",
    ),
    XpuModel(
        key="ming-flash-omni",
        name="Ming-flash-omni-2.0",
        model_path="inclusionAI/Ming-flash-omni-2.0",
        probe="chat",
        config_variant="text",
        # 194 GiB of bf16 thinker weights on a 24 GB card: TP=1 plus host offload.
        # Offload re-uploads weights inside a captured region, so the decode graph is off.
        serve_args=(
            "--model-name",
            "ming-omni",
            "--text-only",
            "--thinker.tp_size",
            "1",
            "--thinker.gpu",
            "0",
            "--audio_encoder.gpu",
            "0",
            "--image_encoder.gpu",
            "0",
            "--mem-fraction-static",
            "0.30",
            "--thinker.engine.cpu_offload_gb",
            "190",
            "--thinker.engine.cuda_graph_backend_decode",
            "disabled",
        ),
        startup_seconds=1800,
    ),
    XpuModel(
        key="minicpm-o",
        name="MiniCPM-o 4.5",
        model_path="openbmb/MiniCPM-o-4_5",
        probe="chat",
        config_variant="text",
        # The auto thinker fraction leaves the KV cache no room for its ~15 GB of weights.
        serve_args=("--text-only", "--thinker.engine.mem_fraction_static", "0.85"),
        startup_seconds=1800,
    ),
    XpuModel(
        key="personaplex",
        name="PersonaPlex",
        model_path="nvidia/personaplex-7b-v1",
        probe="speech",
        unservable="runs through examples/run_personaplex.py, not an HTTP server",
    ),
    XpuModel(
        key="nemotron-voicechat",
        name="Nemotron VoiceChat",
        model_path="nvidia/NVIDIA-NemotronLabs-VoiceChat-11B",
        probe="speech",
        unservable="runs through examples/run_nemotron_voicechat.py, not an HTTP server",
    ),
    XpuModel(
        key="audar-tts",
        name="Audar-TTS V1 Turbo",
        model_path="AudarAI/Audar-TTS-V1-Turbo",
        probe="speech",
        unservable="needs the audar-tts extra and a SYCL build of llama.cpp, not an SGLang engine",
    ),
)

MODEL_BY_KEY = {model.key: model for model in MODELS}
SINGLE_CARD_KEYS = tuple(model.key for model in MODELS if model.cards == 1)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level edit distance over reference length.

    Written out rather than taken from jiwer: jiwer lives in the eval extra, and an XPU install does not have it.
    """
    want = normalize_text(reference).split()
    got = normalize_text(hypothesis).split()
    if not want:
        return 0.0 if not got else 1.0

    previous = list(range(len(got) + 1))
    for i, want_word in enumerate(want, start=1):
        current = [i]
        for j, got_word in enumerate(got, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (want_word != got_word),
                )
            )
        previous = current
    return previous[-1] / len(want)


def environment_fingerprint(host_label: str = "") -> dict[str, str]:
    """Identify the machine a baseline belongs to, so numbers are never compared across hosts.

    Deliberately not the card pool: that is a run option, and keying on it forked the environment whenever a subset was measured with --card-pool. Which cards a model actually ran on is recorded with the model instead, and a later run on different cards says so.

    host_label replaces the hostname for a run inside a container, where the hostname names the container rather than the machine. A cluster pod gets a fresh generated name per job, so without it every run records a brand new environment and nothing is ever compared. Pass the machine's stable identity, not the container's.
    """
    import torch

    device = "unknown"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.xpu.get_device_properties(0).name

    try:
        import sglang

        sglang_version = sglang.__version__
    except Exception:
        sglang_version = "unknown"

    return {
        "host": host_label or socket.gethostname(),
        "device": device,
        "torch": torch.__version__,
        "sglang": sglang_version,
    }


def fingerprint_key(fingerprint: dict[str, str]) -> str:
    blob = json.dumps(fingerprint, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def recorded_from_commit() -> str:
    """The tree a baseline came from. Informational: a new commit must not invalidate it."""
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def normalize_text(text: str) -> str:
    """Collapse whitespace, case and punctuation so a comparison is not defeated by formatting."""
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", text).strip().lower())


def audio_shape(raw: bytes) -> dict[str, float]:
    """Frame count, rate and loudness of a WAV payload.

    All of it is stable across runs once the request carries a seed; the caller decides which comparison class each field belongs in.
    """
    with wave.open(io.BytesIO(raw)) as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        width = handle.getsampwidth()
        payload = handle.readframes(frames)

    if frames == 0 or rate == 0:
        raise AssertionError("audio holds no frames")

    scale = float(1 << (8 * width - 1))
    samples = memoryview(payload).cast({1: "b", 2: "h", 4: "i"}[width])
    peak = max(abs(sample) for sample in samples) / scale
    rms = math.sqrt(sum((sample / scale) ** 2 for sample in samples) / len(samples))
    if peak <= 0.01:
        raise AssertionError(
            f"audio peaks at {peak:.4f} of full scale, which is silence"
        )

    return {
        "frames": float(frames),
        "sample_rate": float(rate),
        "audio_seconds": frames / rate,
        "peak": peak,
        "rms": rms,
    }


def transcribe(base_url: str, model_path: str, audio: bytes, name: str) -> str:
    response = requests.post(
        f"{base_url}/v1/audio/transcriptions",
        files={"file": (name, audio, "audio/wav")},
        data={"model": model_path, "temperature": "0"},
        timeout=300,
    )
    response.raise_for_status()
    return response.json().get("text", "")


def probe_transcription(base_url: str, model: XpuModel, judge_url: str) -> dict:
    del judge_url
    audio = REFERENCE_AUDIO.read_bytes()
    started = time.perf_counter()
    transcript = transcribe(base_url, model.model_path, audio, REFERENCE_AUDIO.name)
    elapsed = time.perf_counter() - started
    if not transcript.strip():
        raise AssertionError("empty transcript")

    reference_seconds = audio_shape(audio)["audio_seconds"]
    return {
        "exact": {"transcript": normalize_text(transcript)},
        "atmost": {"word_error_rate": word_error_rate(REFERENCE_TEXT, transcript)},
        "speed": {
            "latency_s": elapsed,
            "audio_seconds_per_wall_second": reference_seconds / elapsed,
        },
    }


def synthesize(
    base_url: str, model: XpuModel, text: str, timeout: int
) -> tuple[bytes, float]:
    body = {
        "model": model.model_path,
        "input": text,
        "response_format": "wav",
        **({} if model.seedless else {"seed": 0}),
        **model.payload,
    }
    started = time.perf_counter()
    response = requests.post(f"{base_url}/v1/audio/speech", json=body, timeout=timeout)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    return response.content, elapsed


def probe_speech(base_url: str, model: XpuModel, judge_url: str) -> dict:
    audio, elapsed = synthesize(base_url, model, SPOKEN_TEXT, timeout=600)
    shape = audio_shape(audio)
    result: dict[str, Any] = {
        # Seeded, Qwen3-TTS repeated frames and loudness exactly across six reps, so the
        # frame count is held exact. Loudness still gets a band: the token stream is
        # reproducible but a kernel change moves the vocoder's floats.
        "exact": {"sample_rate": shape["sample_rate"]},
        "atmost": {},
        "close": {"peak": shape["peak"], "rms": shape["rms"]},
        "speed": {
            "latency_s": elapsed,
            "audio_seconds_per_wall_second": shape["audio_seconds"] / elapsed,
        },
    }
    # Seeded, frames repeat exactly and belong with the text.
    #
    # Unseeded, the model chooses its own length: Ming-Omni-TTS read the same sentence
    # 2x longer between repeats (frames spread 112%, throughput 137%). No band holds
    # that, and leaving it in a compared group would size this entry's whole speed floor
    # from it -- including latency, which is steady to 20%. So length is recorded for
    # the record and not gated, throughput (audio seconds per wall second, which is
    # length divided by time) goes with it, and the accuracy gate becomes the round-trip
    # WER below: a ceiling tolerates a varying utterance while still catching a model
    # that has stopped being intelligible.
    if model.seedless:
        result["observed"] = {
            "frames": shape["frames"],
            "peak": result["close"].pop("peak"),
            "rms": result["close"].pop("rms"),
            "audio_seconds_per_wall_second": result["speed"].pop(
                "audio_seconds_per_wall_second"
            ),
        }
    else:
        result["exact"]["frames"] = shape["frames"]
    if judge_url:
        heard = transcribe(
            judge_url, MODEL_BY_KEY[model.judge].model_path, audio, "tts.wav"
        )
        result["atmost"]["round_trip_wer"] = word_error_rate(SPOKEN_TEXT, heard)
        result["heard"] = normalize_text(heard)
    return result


def probe_music(base_url: str, model: XpuModel, judge_url: str) -> dict:
    del judge_url
    audio, elapsed = synthesize(base_url, model, MUSIC_PROMPT, timeout=900)
    shape = audio_shape(audio)
    return {
        "exact": {"sample_rate": shape["sample_rate"]},
        "close": {"audio_seconds": shape["audio_seconds"], "rms": shape["rms"]},
        "speed": {"latency_s": elapsed},
    }


def probe_chat(base_url: str, model: XpuModel, judge_url: str) -> dict:
    del judge_url
    body = {
        "model": model.model_path,
        "messages": [{"role": "user", "content": ARITHMETIC_PROMPT}],
        "max_tokens": 32,
        "temperature": 0,
        "seed": 0,
    }
    started = time.perf_counter()
    response = requests.post(f"{base_url}/v1/chat/completions", json=body, timeout=600)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    if not content:
        raise AssertionError("no assistant content")
    completion_tokens = payload.get("usage", {}).get("completion_tokens", 0)
    speed = {"latency_s": elapsed}
    if completion_tokens:
        speed["decode_tokens_per_second"] = completion_tokens / elapsed
    return {"exact": {"content": normalize_text(content)}, "speed": speed}


PROBES: dict[str, Callable[[str, XpuModel, str], dict]] = {
    "transcription": probe_transcription,
    "speech": probe_speech,
    "music": probe_music,
    "chat": probe_chat,
}

COMPARED_GROUPS = ("exact", "atmost", "close", "speed")


ENGINE_STAGE_CACHE: dict[str, tuple[str, ...]] = {}


def engine_stages(model_path: str, variant: str = "") -> tuple[str, ...]:
    """Stage names that own an SGLang engine, read from the model's own pipeline config.

    Derived rather than listed per model: a stage-scoped override aimed at a stage the
    pipeline does not have is rejected only when the engine is built, minutes into a load.
    """
    cache_key = f"{model_path}@{variant}"
    if cache_key in ENGINE_STAGE_CACHE:
        return ENGINE_STAGE_CACHE[cache_key]
    try:
        import importlib

        from sglang_omni.config.manager import resolve_config_cls_for_model_path

        config_cls = resolve_config_cls_for_model_path(model_path)
        if variant:
            variants = getattr(
                importlib.import_module(config_cls.__module__), "Variants", {}
            )
            config_cls = variants.get(variant, config_cls)
        config = config_cls(model_path=model_path)
        stages = tuple(
            stage.name for stage in config.stages if getattr(stage, "engine", None)
        )
    except Exception as exc:
        print(f"  could not derive engine stages ({exc}); serving without them")
        stages = ()
    ENGINE_STAGE_CACHE[cache_key] = stages
    return stages


def serve_command(model: XpuModel, port: int, extra: tuple[str, ...] = ()) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model-path",
        model.model_path,
        *model.serve_args,
        *extra,
    ]
    for stage in engine_stages(model.model_path, model.config_variant):
        for key, value in DETERMINISM_ENGINE_ARGS.items():
            command += [f"--{stage}.engine.{key}", value]
    return command


def launch(model: XpuModel, cards: str, extra: tuple[str, ...] = ()) -> tuple[Any, str]:
    port = free_port()
    env = dict(os.environ, ZE_AFFINITY_MASK=cards, **model.env)
    env.setdefault("SGLANG_OMNI_STARTUP_TIMEOUT", str(model.startup_seconds))
    # The server runs with proxy variables stripped, so a Hub lookup there waits out a
    # connect timeout instead of reading the cache: AuK resolves a second model by repo
    # id and sat 30 minutes in one. --prefetch fetches in this process, before launch.
    env.setdefault("HF_HUB_OFFLINE", "1")
    command = serve_command(model, port, extra)
    print(f"  serving {model.key} on card(s) {cards}, port {port}", flush=True)
    proc = start_server_from_cmd(
        command,
        None,
        port,
        timeout=model.startup_seconds,
        env=env,
        strip_proxy=True,
        health_path="/v1/models",
        health_body_contains=None,
    )
    return proc, f"http://127.0.0.1:{port}"


def existing_ancestor(path: str) -> str:
    """The nearest ancestor of path that exists.

    A cache directory's leaves are created on first use, so sizing the filesystem it will
    live on has to ask about a directory that is already there.
    """
    while path and not os.path.isdir(path):
        path = os.path.dirname(path)
    return path or "/"


def hub_cache_dir() -> str:
    """The hub cache directory this host is using, which need not exist yet."""
    cache = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    return os.path.join(existing_ancestor(cache), "hub")


def needed_checkpoints(model: XpuModel) -> list[str]:
    """The model paths one measurement has to have locally, including its judge."""
    paths = [model.model_path]
    if model.judge:
        paths.append(MODEL_BY_KEY[model.judge].model_path)
    else:
        pass
    return paths


def prune_hub_cache(model: XpuModel) -> str:
    """Delete every cached revision this measurement does not need, and say what that freed.

    Opt-in, because on some hosts the hub cache holds other people's work and deleting
    what this run happens not to need is destructive. Through the hub's own API rather
    than by removing directories: contents live in one content-addressed store that the
    per-repo directories only link into, so removing a directory frees almost nothing.
    """
    from huggingface_hub import scan_cache_dir

    cache = hub_cache_dir()
    if not os.path.isdir(cache):
        return "no hub cache to prune yet"
    else:
        pass
    keep = {path.partition("@")[0] for path in needed_checkpoints(model)}
    before = shutil.disk_usage("/").free / 2**30
    info = scan_cache_dir(cache)
    doomed = [
        revision.commit_hash
        for repo in info.repos
        if repo.repo_id not in keep
        for revision in repo.revisions
    ]
    if doomed:
        info.delete_revisions(*doomed).execute()
    else:
        pass
    free_now = shutil.disk_usage("/").free / 2**30
    return (
        f"pruned {len(doomed)} revision(s), freed {free_now - before:.1f} GiB, "
        f"{free_now:.1f} GiB free"
    )


def prefetch_checkpoints(model: XpuModel) -> str:
    """Fetch this measurement's weights before the server starts.

    Opt-in, for hosts whose server has no outbound network: the launcher strips proxy
    variables from the server environment, so a checkpoint that is not already local never
    arrives and the run fails as a startup timeout that names neither cause nor fix.
    """
    from huggingface_hub import snapshot_download

    for path in needed_checkpoints(model):
        repo, _, revision = path.partition("@")
        snapshot_download(repo, revision=revision or None, max_workers=8)
    free_now = shutil.disk_usage("/").free / 2**30
    return f"fetched {len(needed_checkpoints(model))} checkpoint(s), {free_now:.1f} GiB free"


def download_blocked(model_path: str, max_checkpoint_gib: float) -> str:
    """Why this checkpoint will not be fetched here, or an empty string if it will.

    Ming-flash-omni-2.0 is 194 GiB and was half cached: the run spent 30 minutes
    downloading, filled the volume, and then reported "service not ready within 1800s",
    which names neither the cause nor the fix. A cold checkpoint needs room, so say so
    before spending the half hour.
    """
    repo, _, revision = model_path.partition("@")
    blobs = os.path.join(hub_cache_dir(), f"models--{repo.replace('/', '--')}", "blobs")
    # What this model has resident, which is not the size of its cache directory: blobs
    # are content-addressed in a store shared with every other repo and linked in from
    # here, so a complete checkpoint's own directory holds nothing but links. A link
    # that resolves is a finished blob and counts its length; a plain file is a transfer
    # in progress and counts the disk it holds, since a part-downloaded blob is sparse.
    local_gib = 0.0
    interrupted = False
    for root, _, names in os.walk(blobs, followlinks=False):
        for name in names:
            path = os.path.join(root, name)
            if name.endswith(".incomplete"):
                interrupted = True
            else:
                pass
            if os.path.islink(path):
                target = os.path.realpath(path)
                if os.path.exists(target):
                    local_gib += os.path.getsize(target) / 2**30
                else:
                    continue
            else:
                local_gib += os.stat(path).st_blocks * 512 / 2**30
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, revision=revision or None, files_metadata=True)
        repo_gib = sum(item.size or 0 for item in info.siblings) / 2**30
    except Exception as exc:
        # Offline, or gated. Not knowing the size is not grounds to skip a model: fall
        # through and let the launch report whatever actually goes wrong.
        print(f"  could not size {repo} before fetching it ({exc})")
        return ""
    usage = os.statvfs(existing_ancestor(hub_cache_dir()))
    free_gib = usage.f_bavail * usage.f_frsize / 2**30
    # The repo total is an upper bound on what a load needs, not an estimate of it:
    # Whisper publishes the same weights as safetensors, bin and ONNX, and the loader
    # fetches one of the three. So the repo size only decides the cold case, and a
    # checkpoint that already has weights resident is never called too big -- a false
    # skip silently stops testing a model that works.
    #
    # An interrupted transfer is the case worth catching, and it is the one that can be
    # measured: the blobs it left behind say how much is still outstanding.
    if repo_gib > max_checkpoint_gib:
        # The checkpoint's size, not what is left to fetch: Ming-flash-omni was 196 of
        # 222 GiB cached, so a cap on the outstanding fetch let it through to a 30
        # minute offloaded load on every sweep. Skipped, not failed, so the run goes
        # on to the rest and still exits 0.
        return (
            f"the checkpoint is about {repo_gib:.0f} GiB, over the "
            f"{max_checkpoint_gib:.0f} GiB --max-checkpoint-gib allows"
        )
    elif interrupted:
        missing_gib = max(repo_gib - local_gib, 0.0)
        if free_gib >= missing_gib * 1.1:
            return ""
        else:
            return (
                f"a previous fetch was interrupted and needs about {missing_gib:.0f} GiB "
                f"more; the checkpoint volume has {free_gib:.1f} GiB free"
            )
    elif local_gib < 1.0 and free_gib < repo_gib * 1.1:
        return (
            f"not cached: the repo is about {repo_gib:.0f} GiB and the checkpoint "
            f"volume has {free_gib:.1f} GiB free"
        )
    else:
        return ""


class CardsBusy(RuntimeError):
    """Raised instead of measuring when something else already holds the card.

    Its own type so a check can tell "could not measure" from "measured and it regressed". Contention reported as a regression is how a gate learns to be ignored: a neighbour on the card can make an unchanged model read several times slower.
    """


def cards_in_use(cards: str) -> dict[str, float]:
    """Megabytes already allocated on each card, device-wide so a neighbour's work shows.

    A measurement taken while someone else holds the card is not a slow run, it is a different machine, and alternating A/B runs on quiet cards are what tell the two apart.
    """
    used: dict[str, float] = {}
    try:
        import torch

        for card in cards.split(","):
            free, total = torch.xpu.mem_get_info(int(card))
            used[card] = round((total - free) / 1024**2, 1)
    except Exception as exc:
        print(f"    cannot read card memory ({exc}); proceeding without the idle check")
    return used


def allocate_cards(pool: list[int], count: int, taken: set[int]) -> str:
    """Lowest-numbered unused cards from the pool, as a ZE_AFFINITY_MASK.

    Deterministic so a model lands on the same cards every run: on the reference host card
    0's PCIe link trains at 5 GT/s where cards 1-7 run at 32, and a baseline that wandered
    onto it would be comparing two different machines.
    """
    free = [card for card in pool if card not in taken]
    if len(free) < count:
        raise AssertionError(
            f"need {count} card(s), pool {pool} has {len(free)} free ({sorted(taken)} in use)"
        )
    chosen = free[:count]
    taken.update(chosen)
    return ",".join(str(card) for card in chosen)


def measure_launches(
    model: XpuModel,
    pool: list[int],
    repeats: int,
    warmups: int,
    extras: dict[str, tuple[str, ...]],
    launches: int,
    idle_mib: float,
    idle_wait_s: int,
) -> dict:
    """Measure the model across whole server restarts, not just repeats inside one.

    Restarting is the variance a gate has to survive and repeats cannot see it. Measured on one B60, three launches of Qwen3-ASR gave medians 0.17559, 0.16885 and 0.17466 -- 3.9 percent apart -- while each launch reported about 1 percent internally. With more than one launch the recorded spread becomes the launch-to-launch figure, which is what the tolerance should be derived from.
    """
    runs = []
    for index in range(launches):
        if launches > 1:
            print(f"  launch {index + 1}/{launches}", flush=True)
        runs.append(
            measure(model, pool, repeats, warmups, extras, idle_mib, idle_wait_s)
        )
    if launches == 1:
        return runs[0]
    folded = fold_samples(runs)
    folded["speed"] = pessimistic_speed(runs)
    # The worst within-run dispersion any launch showed, not the first launch's: this is
    # what the tolerance is derived from once the recorded value is the slowest launch,
    # so it should be the least favourable one observed rather than whichever ran first.
    #
    # Each launch's trimmed figure, for the same reason the single-launch path trims: one
    # of these three launches stalled once, and its full range of 77 percent would have
    # set a 230 percent tolerance -- worse than the double-count this replaced.
    worst_within: dict[str, float] = {}
    for run in runs:
        per_run = run.get("trimmed_spread") or run.get("spread", {})
        for name, value in per_run.items():
            worst_within[name] = max(worst_within.get(name, 0.0), value)
    folded["within_run_spread"] = worst_within
    folded["launches"] = launches
    return folded


def pessimistic_speed(runs: list[dict]) -> dict[str, float]:
    """The slowest speed each field reached across the launches, not the median of them.

    These models are modal rather than noisy: on a B60, Whisper's three launches gave medians of 0.356, 0.387 and 0.430 s, each internally tight to about 1 percent. A baseline at the median of such a set false-fires whenever the slow mode comes back. Anchored at the slowest, a fast mode can only ever read as an improvement, which the run reports, and a real regression still has to clear the slowest mode plus the tolerance.
    """
    worst: dict[str, float] = {}
    for run in runs:
        for name, value in run.get("speed", {}).items():
            if name not in worst:
                worst[name] = value
            elif "latency" in name:
                worst[name] = max(worst[name], value)
            else:
                worst[name] = min(worst[name], value)
    return worst


def measure(
    model: XpuModel,
    pool: list[int],
    repeats: int,
    warmups: int,
    extras: dict[str, tuple[str, ...]],
    idle_mib: float = 1024.0,
    idle_wait_s: int = 0,
) -> dict:
    """Serve the model, and its judge when it has one, then sample it warmups+repeats times."""
    servers = []
    taken: set[int] = set()
    cards = allocate_cards(pool, model.cards, taken)
    busy_before = cards_in_use(cards)
    deadline = time.time() + idle_wait_s
    while True:
        noisy = {card: mib for card, mib in busy_before.items() if mib > idle_mib}
        if not noisy:
            break
        else:
            pass
        if time.time() >= deadline:
            raise CardsBusy(
                f"card(s) not idle before launch: {noisy} MiB in use, over the "
                f"{idle_mib:g} MiB bar; another process on the card makes this a "
                "different machine"
            )
        else:
            pass
        print(f"  waiting for {noisy} MiB to clear", flush=True)
        time.sleep(30)
        busy_before = cards_in_use(cards)
    try:
        judge_url = ""
        if model.judge:
            judge = MODEL_BY_KEY[model.judge]
            judge_proc, judge_url = launch(
                judge,
                allocate_cards(pool, judge.cards, taken),
                extras.get(judge.key, ()),
            )
            servers.append(judge_proc)
        proc, base_url = launch(model, cards, extras.get(model.key, ()))
        servers.append(proc)

        samples = []
        for index in range(warmups + repeats):
            sample = PROBES[model.probe](base_url, model, judge_url)
            label = (
                f"warmup {index + 1}"
                if index < warmups
                else f"rep {index - warmups + 1}"
            )
            extra = (
                f" wer={sample['atmost']['round_trip_wer']:.4f}"
                if sample.get("atmost", {}).get("round_trip_wer") is not None
                else ""
            )
            print(f"    {label}: {sample['speed']}{extra}", flush=True)
            samples.append(sample)
    finally:
        for proc in reversed(servers):
            stop_server(proc)

    folded = fold_samples(samples[warmups:])
    folded["cards"] = cards
    folded["cards_used_mib"] = {"before": busy_before, "after": cards_in_use(cards)}
    return folded


def fold_samples(samples: list[dict]) -> dict:
    """Collapse repeats: exact fields must agree, numeric fields take the median."""
    folded: dict[str, Any] = {group: {} for group in COMPARED_GROUPS}
    folded["spread"] = {}
    for group in COMPARED_GROUPS:
        for key in samples[0].get(group, {}):
            values = [sample[group][key] for sample in samples]
            if group == "exact":
                unique = set(values)
                if len(unique) != 1:
                    raise AssertionError(
                        f"{key} is not reproducible across repeats: {sorted(unique)}"
                    )
                folded[group][key] = values[0]
                continue
            # A ceiling takes the worst sample, everything else the median. Recording the
            # median of a ceiling metric understates it: Ming-Omni-TTS, which cannot be
            # seeded, scored WER 0.0, 0.222, 0.0 in one run, and a 0.0 median plus the
            # 0.05 band would have failed the next healthy run that babbled once.
            folded[group][key] = (
                max(values) if group == "atmost" else statistics.median(values)
            )
            if len(values) > 1 and folded[group][key]:
                span = (max(values) - min(values)) / folded[group][key]
                folded["spread"][f"{group}.{key}"] = round(span, 4)
                trimmed = trimmed_span(values, folded[group][key])
                if group == "speed" and trimmed is not None:
                    folded.setdefault("trimmed_spread", {})[f"{group}.{key}"] = trimmed
                else:
                    pass
                if group == "speed":
                    drift = warming_drift(values)
                    if drift:
                        early, late = drift
                        print(
                            f"    still warming on {key}: the run's two halves sit apart "
                            f"({early:.6g} then {late:.6g}); raise --warmups before trusting the spread"
                        )
                    else:
                        pass
                else:
                    pass
    # Measured but deliberately not gated, so it has to be carried here explicitly:
    # the loop above only walks the compared groups, which would drop it.
    if samples[0].get("observed"):
        folded["observed"] = {}
        folded["observed_span"] = {}
        for key in samples[0]["observed"]:
            values = [sample["observed"][key] for sample in samples]
            middle = statistics.median(values)
            folded["observed"][key] = middle
            folded["observed_span"][key] = round(
                (max(values) - min(values)) / (middle or 1), 4
            )
    return folded


def trimmed_span(values: list[float], middle: float) -> float | None:
    """Spread with the best and worst sample dropped, or None when there are too few to drop any.

    What a gate should be sized from. Nemotron 3.5 ASR ran nine samples on a B60 at 0.1399, 0.1393, 0.1399, 0.2333, 0.1636, 0.1476, 0.1431, 0.1446, 0.1372: eight of them inside nine percent, and one stall. Sizing on the full range calls that model 67 percent noisy and, at three times headroom, hands it a 201 percent tolerance that would miss a doubling. The stall is real and still shows in the reported spread -- it is just not the typical dispersion, and the typical dispersion is what a tolerance describes.
    """
    if len(values) < 5:
        return None
    else:
        inner = sorted(values)[1:-1]
        return round((max(inner) - min(inner)) / middle, 4)


def warming_drift(values: list[float]) -> tuple[float, float] | None:
    """The two half-medians when the run is still settling, else None.

    Comparing halves rather than checking monotonicity: a settling server ties or ticks the wrong way once, and either defeats a per-sample test while the drift is still there. Whisper on a B60 ran 0.4451, 0.4472, 0.4421, 0.4339, 0.4105 -- down 7.8 percent overall, with rep 2 breaking the slope.

    Halves are compared by median, not mean, because one stalled sample drags a mean into looking like drift: Nemotron 3.5 ASR tripped this with a single 0.2333 among eight samples near 0.14, and the warning then quoted the first and last sample, which were 2 percent apart and illustrated nothing.
    """
    if len(values) < 4:
        return None
    else:
        middle = statistics.median(values)
        half = len(values) // 2
        early = statistics.median(values[:half])
        late = statistics.median(values[-half:])
        if middle and abs(late - early) / abs(middle) > 0.03:
            return (early, late)
        else:
            return None


def compare(
    key: str, baseline: dict, observed: dict, tolerances: dict
) -> tuple[list[str], list[str]]:
    """Return regressions and improvements. A regression fails the run; an improvement does not.

    An improvement is reported rather than swallowed, because a baseline nobody re-records drifts into a floor so low it stops catching anything.
    """
    failures: list[str] = []
    gains: list[str] = []

    recorded_cards = baseline.get("cards")
    observed_cards = observed.get("cards")
    if recorded_cards and observed_cards and recorded_cards != observed_cards:
        gains.append(
            f"{key}: recorded on card(s) {recorded_cards}, measured on {observed_cards}; "
            "speed is not comparable across cards on a host with uneven PCIe links"
        )

    for name, want in baseline.get("exact", {}).items():
        got = observed.get("exact", {}).get(name)
        if got != want:
            failures.append(
                f"{key}: {name} changed\n    baseline {want!r}\n    observed {got!r}"
            )

    for name, want in baseline.get("atmost", {}).items():
        got = observed.get("atmost", {}).get(name)
        ceiling = want + tolerances["absolute"]
        if got is None:
            failures.append(f"{key}: {name} missing from this run")
        elif got > ceiling:
            failures.append(
                f"{key}: {name} rose to {got:.4f} (baseline {want:.4f}, ceiling {ceiling:.4f})"
            )
        elif got < want:
            gains.append(f"{key}: {name} fell to {got:.4f} from {want:.4f}")

    for name, want in baseline.get("close", {}).items():
        got = observed.get("close", {}).get(name)
        if got is None:
            failures.append(f"{key}: {name} missing from this run")
        elif want and abs(got - want) / want > tolerances["close"]:
            failures.append(
                f"{key}: {name} moved {(got - want) / want:+.2%} "
                f"(baseline {want:.6g}, observed {got:.6g}, allowed +-{tolerances['close']:.1%})"
            )

    for name, want in baseline.get("speed", {}).items():
        got = observed.get("speed", {}).get(name)
        if got is None:
            failures.append(f"{key}: {name} missing from this run")
            continue
        allowed = speed_tolerance_for(baseline, name, tolerances)
        # Latency must not grow; a rate must not shrink.
        regression = (got - want) / want if "latency" in name else (want - got) / want
        if regression > allowed:
            failures.append(
                f"{key}: {name} regressed {regression:.2%} "
                f"(baseline {want:.6g}, observed {got:.6g}, allowed {allowed:.1%})"
            )
        elif -regression > allowed:
            gains.append(
                f"{key}: {name} improved {-regression:.2%} "
                f"(baseline {want:.6g}, observed {got:.6g})"
            )
    return failures, gains


def speed_tolerance_for(baseline: dict, name: str, tolerances: dict) -> float:
    """How far this field may move, derived from how much it wobbled when it was recorded.

    One tolerance for every model is wrong in both directions at once. Measured on a B60 in one sweep, Qwen3-ASR latency wobbled 0.4 percent and Whisper 8.9 percent: a shared 10 percent floor lets a 5 percent ASR regression through and still flares on Whisper's noise. The floor stays as the lower bound for a model quiet enough that three times its spread is tiny.
    """
    # A multi-launch entry already holds its slowest launch, so its launch-to-launch
    # spread is in the recorded value and must not be charged again as tolerance: that
    # double-count gave Nemotron 3.5 ASR a 31.9% allowance on an anchor picked precisely
    # so it would not need one. Such an entry derives from the worst within-run
    # dispersion instead, which is the noise left once the mode is pinned.
    #
    # Otherwise the trimmed spread, which describes the dispersion a tolerance is for,
    # where the full range describes the worst sample seen and is reported beside it.
    # Falls back to the full range for entries recorded before trimming, and to the floor
    # for a field with no spread at all.
    if baseline.get("launches", 1) > 1:
        spread = baseline.get("within_run_spread", {}).get(f"speed.{name}")
    else:
        spread = baseline.get("trimmed_spread", {}).get(f"speed.{name}")
    if spread is None:
        spread = baseline.get("spread", {}).get(f"speed.{name}")
    else:
        pass
    if spread is None:
        return tolerances["speed"]
    else:
        return max(tolerances["speed"], spread * tolerances["speed_headroom"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models",
        default="all",
        help="comma-separated keys, or 'all'",
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--record",
        action="store_true",
        help="write this environment's entry instead of checking",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="report observed spread and exit without checking",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="scored samples per model, taken after the warmups",
    )
    # One was not enough: a lone trailing-warm sample set the whole spread, and on a B60
    # a second warmup took Qwen3-ASR's spread from 3.2 to 0.42 percent.
    parser.add_argument(
        "--launches",
        type=int,
        default=1,
        help="whole server restarts per model; above 1, speed is anchored at the slowest launch and exact fields must agree across restarts",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=2,
        help="samples taken and discarded before scoring begins",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="re-measure models already recorded for this environment instead of resuming",
    )
    parser.add_argument(
        "--card-pool",
        default="0,1,2,3,4,5,6,7",
        help="cards this run may claim; each model takes what its topology needs",
    )
    parser.add_argument(
        "--close-tolerance",
        type=float,
        default=None,
        help="relative band for loudness, which moves when kernels change even under a seed",
    )
    parser.add_argument(
        "--absolute-tolerance",
        type=float,
        default=None,
        help="absolute headroom for a rate such as WER",
    )
    # A floor, not the usual gate: the per-field value is three times the recorded spread
    # once that is larger. Default covers relaunch variance that a single launch cannot
    # measure -- B60 moved 3.9 percent between restarts of one model. Lower it only when recording with --launches 3 or more, where the spread
    # already carries that movement.
    parser.add_argument(
        "--speed-tolerance",
        type=float,
        default=None,
        help="lower bound on the speed regression allowed, before per-field spread is considered",
    )
    parser.add_argument(
        "--speed-headroom",
        type=float,
        default=None,
        help="multiple of a field's recorded spread to allow; the larger of this and the floor wins",
    )
    # A driver bug on one host must not change what another host measures, so the
    # workaround lives in the invocation and is recorded in that host's entry.
    parser.add_argument(
        "--serve-extra",
        action="append",
        default=[],
        metavar="KEY=ARGS",
        help="extra serve args for one model on this host, e.g. qwen3-tts=--vocoder.factory.async_decode false",
    )
    parser.add_argument(
        "--idle-mib",
        type=float,
        default=1024.0,
        help="refuse to measure a card already holding more than this; a neighbour's work reads as a regression otherwise",
    )
    parser.add_argument(
        "--idle-wait-s",
        type=int,
        default=0,
        help="seconds to wait for a busy card before skipping the model",
    )
    parser.add_argument(
        "--prune-hf-cache",
        action="store_true",
        help="delete cached checkpoints this measurement does not need; destructive, and "
        "on some hosts the cache holds other users' work",
    )
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="fetch weights before launching, for hosts whose server has no network",
    )
    parser.add_argument(
        "--host-label",
        default="",
        help="identify the machine as this instead of by hostname, for runs inside a "
        "container whose hostname names the container and not the machine",
    )
    parser.add_argument(
        "--max-checkpoint-gib",
        type=float,
        default=100.0,
        help="skip a model whose checkpoint is larger than this, cached or not; it keeps its row and the run still exits 0 (Ming-flash-omni-2.0 is 222 GiB, every other row is under 70)",
    )
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        # Read from the baseline rather than a flag on the entry: a hand-kept "verified"
        # field drifts the moment a model records, and this one had, claiming two of
        # nineteen. The file is the record of what has actually served somewhere.
        recorded_on: dict[str, list[str]] = {}
        if args.baseline.exists():
            for env in json.loads(args.baseline.read_text())["environments"].values():
                for key in env.get("models", {}):
                    recorded_on.setdefault(key, []).append(env["fingerprint"]["host"])
        else:
            pass
        print(f"{'key':22s} {'cards':>5s} {'judge':10s} {'recorded on':22s} model")
        for model in MODELS:
            where = (
                "skip"
                if model.unservable
                else ",".join(sorted(recorded_on.get(model.key, []))) or "-"
            )
            print(
                f"{model.key:22s} {model.cards:5d} {model.judge or '-':10s} {where:22s} {model.model_path}"
            )
        return 0

    keys = (
        [model.key for model in MODELS]
        if args.models == "all"
        else args.models.split(",")
    )
    unknown = [key for key in keys if key not in MODEL_BY_KEY]
    if unknown:
        parser.error(f"unknown model keys: {unknown}; see --list")

    extras: dict[str, tuple[str, ...]] = {}
    for item in args.serve_extra:
        key, _, argument_text = item.partition("=")
        if key not in MODEL_BY_KEY or not argument_text:
            parser.error(f"--serve-extra wants <model key>=<args>, got {item!r}")
        extras[key] = tuple(argument_text.split())

    pool = [int(c) for c in args.card_pool.split(",") if c.strip()]
    fingerprint = environment_fingerprint(args.host_label)
    env_key = fingerprint_key(fingerprint)
    print(f"environment {env_key}: {json.dumps(fingerprint)}")

    stored = {"environments": {}}
    if args.baseline.exists():
        stored = json.loads(args.baseline.read_text())
    entry = stored.get("environments", {}).get(env_key, {})

    if not args.record and not args.calibrate and not entry:
        known = stored.get("environments", {})
        print(
            f"no baseline for environment {env_key} in {args.baseline}.\n"
            f"  known environments: {[e.get('fingerprint', {}).get('host') for e in known.values()] or 'none'}\n"
            "  run with --record to add this one.",
            file=sys.stderr,
        )
        return 2

    # An explicit flag wins, then whatever this environment recorded, then the built-in.
    # The recorded value used to be decorative: a check took the CLI default and ignored the
    # floor the host had measured for itself, so one host's gate was applied to another and
    # its models false-fired.
    BUILTIN_TOLERANCES = {
        "close": 0.03,
        "absolute": 0.05,
        "speed": 0.10,
        "speed_headroom": 3.0,
    }
    FLAG_FOR = {
        "close": args.close_tolerance,
        "absolute": args.absolute_tolerance,
        "speed": args.speed_tolerance,
        "speed_headroom": args.speed_headroom,
    }
    recorded_tolerances = entry.get("tolerances", {})
    tolerances = {}
    for name, builtin in BUILTIN_TOLERANCES.items():
        if FLAG_FOR[name] is not None:
            tolerances[name], source = FLAG_FOR[name], "the flag"
        elif name in recorded_tolerances:
            tolerances[name], source = recorded_tolerances[name], "this entry"
        else:
            tolerances[name], source = builtin, "the built-in default"
        if name == "speed":
            print(f"speed floor {tolerances[name]:.1%}, from {source}")

    def flush(partial: dict[str, Any]) -> None:
        """Persist what is measured so far, so an interrupted sweep resumes instead of restarting."""
        on_disk = {"environments": {}}
        if args.baseline.exists():
            on_disk = json.loads(args.baseline.read_text())
        else:
            pass
        previous = on_disk.get("environments", {}).get(env_key, {})
        # Provenance belongs to the model it was measured with, so a model measured in this
        # run carries what this run passed for it and nothing else. Merging globally meant a
        # re-record without the workaround could not clear one, and the file went on
        # claiming a flag the recorded number had not been measured with.
        serve_extra = {
            key: value
            for key, value in previous.get("serve_extra", {}).items()
            if key not in partial
        }
        serve_extra.update(
            {key: list(value) for key, value in extras.items() if key in partial}
        )
        on_disk.setdefault("environments", {})[env_key] = {
            "fingerprint": fingerprint,
            "recorded_from_commit": recorded_from_commit(),
            "tolerances": tolerances,
            "serve_extra": serve_extra,
            "models": {**previous.get("models", {}), **partial},
        }
        args.baseline.write_text(json.dumps(on_disk, indent=2, sort_keys=True) + "\n")

    results: dict[str, Any] = {}
    failures: list[str] = []
    gains: list[str] = []
    skipped: list[str] = []
    done = entry.get("models", {}) if args.record and not args.fresh else {}
    for index, key in enumerate(keys, start=1):
        model = MODEL_BY_KEY[key]
        print(f"\n== [{index}/{len(keys)}] {model.name} ({key}) ==", flush=True)
        if model.unservable:
            print(f"  SKIPPED: {model.unservable}", flush=True)
            skipped.append(f"{key}: {model.unservable}")
            continue
        if key in done:
            print(
                "  already recorded for this environment; --fresh to redo", flush=True
            )
            skipped.append(f"{key}: already recorded")
            continue
        # A topology the pool cannot host is absent, not broken: a four-card host skips the
        # TP=8 entry rather than reporting it as a failure.
        if model.cards > len(pool):
            reason = f"needs {model.cards} cards, pool has {len(pool)}"
            print(f"  SKIPPED: {reason}", flush=True)
            skipped.append(f"{key}: {reason}")
            continue
        if args.prune_hf_cache:
            print(f"  {prune_hub_cache(model)}", flush=True)
        else:
            pass
        shortfall = download_blocked(model.model_path, args.max_checkpoint_gib)
        if shortfall:
            print(f"  SKIPPED: {shortfall}", flush=True)
            skipped.append(f"{key}: {shortfall}")
            continue
        if args.prefetch:
            print(f"  {prefetch_checkpoints(model)}", flush=True)
        else:
            pass
        try:
            observed = measure_launches(
                model,
                pool,
                args.repeats,
                args.warmups,
                extras,
                args.launches,
                args.idle_mib,
                args.idle_wait_s,
            )
        except CardsBusy as exc:
            print(f"  SKIPPED: {exc}", flush=True)
            skipped.append(f"{key}: {exc}")
            continue
        except Exception as exc:
            failures.append(f"{key}: did not produce a measurement: {exc}")
            print(f"  FAILED: {exc}", flush=True)
            print(
                f"  running tally: {len(results)} ok, {len(failures)} failed, {len(skipped)} skipped",
                flush=True,
            )
            continue
        results[key] = observed
        if args.record:
            flush(results)
        print(f"  median  {json.dumps(observed['speed'])}")
        print(
            f"  accuracy {json.dumps(observed['exact'])} {json.dumps(observed.get('atmost', {}))}"
        )
        print(f"  spread  {json.dumps(observed['spread'])}")
        if not args.record and not args.calibrate:
            if key not in entry.get("models", {}):
                failures.append(
                    f"{key}: absent from this environment's baseline; re-record to add it"
                )
            else:
                model_failures, model_gains = compare(
                    key, entry["models"][key], observed, tolerances
                )
                failures += model_failures
                gains += model_gains
                for message in model_gains:
                    print(f"  IMPROVED {message.split(': ', 1)[1]}", flush=True)
        print(
            f"  running tally: {len(results)} ok, {len(failures)} failed, {len(skipped)} skipped",
            flush=True,
        )

    if args.record:
        if results:
            flush(results)
            print(
                f"\nwrote {len(results)} model(s) for environment {env_key} to {args.baseline}"
            )
        else:
            print("\nnothing measured; leaving the baseline untouched")

    if args.calibrate:
        print("\nobserved spread per field, as a fraction of the median:")
        for key, observed in results.items():
            for name, span in sorted(observed["spread"].items()):
                print(f"  {key:22s} {name:46s} {span:.2%}")
        print(
            "\nSet --speed-tolerance above the largest speed spread, or the gate will be flaky."
        )
        return 0

    print()
    # While recording there is nothing to regress against yet: a failure here is a model
    # that would not serve or answer, which is a different thing to report.
    label = "FAILED     " if args.record else "REGRESSION "
    for message in failures:
        print(f"{label} {message}")
    for message in gains:
        print(f"IMPROVED    {message}")
    if gains:
        print(
            "  these beat the baseline: re-record so the floor rises with the code, "
            "or the gate stops catching anything."
        )
    for message in skipped:
        print(f"SKIPPED     {message}")
    print(
        f"{len(results)} measured, {len(skipped)} skipped, {len(failures)} "
        f"{'failure(s)' if args.record else 'regression(s)'}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
