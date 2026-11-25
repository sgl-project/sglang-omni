#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record and re-check per-model accuracy and speed for the models Intel XPU serves.

Run from the repository root: python -m scripts.xpu.xpu_model_baseline [--list | --record | --calibrate] [--models a,b].
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
import re
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, TypedDict

import requests
import sglang
import torch
from huggingface_hub import HfApi, scan_cache_dir, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

from benchmarks.benchmarker.utils import start_server_from_cmd, stop_server
from sglang_omni.config.manager import resolve_config_cls_for_model_path
from tests.test_ci.xpu_model.model_table import XPU_MODELS, XpuModel

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "tests" / "data"
REFERENCE_AUDIO = DATA_DIR / "query_to_cars.wav"
REFERENCE_TEXT = "How many cars are there in the picture?"
SPOKEN_TEXT = "The quick brown fox jumps over the lazy dog."
ARITHMETIC_PROMPT = "Answer with the number only: 17 plus 25?"
MUSIC_PROMPT = "A short calm piano melody."

DEFAULT_BASELINE = REPO_ROOT / "scripts" / "xpu" / "xpu_model_baseline.json"

# SGLang's own enable_deterministic_inference is absent: with no backend named it picks fa3, which is CUDA-only.
DETERMINISM_ENGINE_ARGS = {
    "disable_radix_cache": "true",
    "random_seed": "0",
}

COMPARED_GROUPS = ("exact", "atmost", "close", "speed")

BUILTIN_TOLERANCES = {
    "close": 0.03,
    "absolute": 0.05,
    "speed": 0.10,
    "speed_headroom": 3.0,
}

ProbeKind = Literal["transcription", "speech", "music", "chat"]


class Reference(TypedDict):
    audio_path: str
    text: str


class RequestFields(TypedDict, total=False):
    voice: str
    ref_audio: str
    ref_text: str
    task_type: str
    language: str
    instructions: str
    references: list[Reference]
    stage_params: dict[str, dict[str, int]]
    max_new_tokens: int


class Sample(TypedDict, total=False):
    exact: dict[str, str | float]
    atmost: dict[str, float]
    close: dict[str, float]
    speed: dict[str, float]
    observed: dict[str, float]
    heard: str


class CardMemory(TypedDict):
    before: dict[str, float]
    after: dict[str, float]


class Record(TypedDict, total=False):
    exact: dict[str, str | float]
    atmost: dict[str, float]
    close: dict[str, float]
    speed: dict[str, float]
    spread: dict[str, float]
    trimmed_spread: dict[str, float]
    observed: dict[str, float]
    observed_span: dict[str, float]
    cards: str
    judge_cards: str
    cards_used_mib: CardMemory
    launches: int
    within_run_spread: dict[str, float]


class Tolerances(TypedDict):
    close: float
    absolute: float
    speed: float
    speed_headroom: float


class Environment(TypedDict, total=False):
    fingerprint: dict[str, str]
    recorded_from_commit: str
    tolerances: Tolerances
    serve_extra: dict[str, list[str]]
    models: dict[str, Record]


class BaselineFile(TypedDict):
    environments: dict[str, Environment]


@dataclass(frozen=True, kw_only=True)
class BaselineModel:
    """One lane row as the baseline measures it: the row carries the recipe, this adds the request and scoring."""

    row: XpuModel
    name: str
    probe: ProbeKind
    payload: RequestFields = field(default_factory=RequestFields)
    # Another model key, served alongside, that transcribes this model's audio back to text.
    judge: str = ""
    # The server refuses a seed, so the generated length is observed rather than gated.
    seedless: bool = False
    # The key the committed baseline records this row under, when it is not the row's own.
    recorded_as: str = ""

    @property
    def key(self) -> str:
        return self.recorded_as or self.row.key

    @property
    def model_path(self) -> str:
        return self.row.checkpoint_spec

    @property
    def cards(self) -> int:
        return self.row.cards

    @property
    def serve_args(self) -> tuple[str, ...]:
        return self.row.serve_args

    @property
    def env(self) -> dict[str, str]:
        return dict(self.row.env)

    @property
    def startup_seconds(self) -> int:
        return self.row.startup_seconds

    @property
    def companions(self) -> tuple[str, ...]:
        return self.row.companions

    @property
    def unservable(self) -> str:
        return self.row.serving

    @property
    def config_variant(self) -> str:
        return self.row.variant or ""


LANE_ROWS = {row.key: row for row in XPU_MODELS}
CLONED_VOICE = RequestFields(
    voice="default", ref_audio=str(REFERENCE_AUDIO), ref_text=REFERENCE_TEXT
)
REFERENCE_LIST = RequestFields(
    voice="default",
    references=[Reference(audio_path=str(REFERENCE_AUDIO), text=REFERENCE_TEXT)],
)
AUK_REQUEST = RequestFields(
    voice="default",
    instructions="A warm, relaxed voice.",
    stage_params={"auk_engine": {"gen_seconds": 4}},
)

MODELS: tuple[BaselineModel, ...] = (
    BaselineModel(row=LANE_ROWS["qwen3-asr"], name="Qwen3-ASR", probe="transcription"),
    BaselineModel(
        row=LANE_ROWS["whisper"], name="Whisper large-v3", probe="transcription"
    ),
    BaselineModel(row=LANE_ROWS["fun-asr"], name="Fun-ASR-Nano", probe="transcription"),
    BaselineModel(
        row=LANE_ROWS["qwen3-tts"],
        name="Qwen3-TTS Base",
        probe="speech",
        payload=CLONED_VOICE,
        judge="qwen3-asr",
    ),
    BaselineModel(
        row=LANE_ROWS["qwen3-tts-cv"],
        name="Qwen3-TTS CustomVoice",
        probe="speech",
        payload=RequestFields(
            voice="Ryan", task_type="CustomVoice", language="English"
        ),
        judge="qwen3-asr",
    ),
    BaselineModel(
        row=LANE_ROWS["cosyvoice3"],
        name="Fun-CosyVoice3",
        probe="speech",
        payload=CLONED_VOICE,
        judge="qwen3-asr",
    ),
    BaselineModel(
        row=LANE_ROWS["zonos2"],
        name="ZONOS2",
        probe="speech",
        payload=RequestFields(voice="default"),
        judge="qwen3-asr",
        seedless=True,
    ),
    BaselineModel(
        row=LANE_ROWS["dots-tts"],
        name="dots.tts SOAR",
        probe="speech",
        payload=REFERENCE_LIST,
        judge="qwen3-asr",
    ),
    BaselineModel(
        row=LANE_ROWS["minimax-music3"],
        name="MiniMax Music 3",
        probe="music",
        # An explicit token cap: the default length makes the run unbounded and not comparable.
        payload=RequestFields(
            instructions=(
                "An energetic arena rock anthem with distorted electric guitars, punchy live "
                "drums and a soaring male vocal at 130 BPM, wide stereo image"
            ),
            max_new_tokens=750,
        ),
    ),
    BaselineModel(
        row=LANE_ROWS["ming-tts"],
        name="Ming-Omni-TTS",
        probe="speech",
        payload=RequestFields(voice="default"),
        judge="qwen3-asr",
        seedless=True,
    ),
    BaselineModel(
        row=LANE_ROWS["omni-text"],
        name="Qwen3-Omni thinker (text only)",
        probe="chat",
        recorded_as="qwen3-omni-text",
    ),
    BaselineModel(
        row=LANE_ROWS["moss-transcribe-diarize"],
        name="MOSS-Transcribe-Diarize",
        probe="transcription",
    ),
    BaselineModel(
        row=LANE_ROWS["nemotron35-asr"], name="Nemotron 3.5 ASR", probe="transcription"
    ),
    BaselineModel(row=LANE_ROWS["arkasr"], name="ARK-ASR", probe="transcription"),
    BaselineModel(
        row=LANE_ROWS["higgs"],
        name="Higgs Audio v3 TTS",
        probe="speech",
        payload=RequestFields(voice="default"),
        judge="qwen3-asr",
    ),
    BaselineModel(
        row=LANE_ROWS["auk"],
        name="AuK",
        probe="speech",
        payload=AUK_REQUEST,
        judge="qwen3-asr",
    ),
    BaselineModel(
        row=LANE_ROWS["auk-flash"],
        name="AuK-Flash",
        probe="speech",
        payload=AUK_REQUEST,
        judge="qwen3-asr",
    ),
    BaselineModel(
        row=LANE_ROWS["qwen3-tts-vd"],
        name="Qwen3-TTS VoiceDesign",
        probe="speech",
        payload=RequestFields(
            voice="default",
            task_type="VoiceDesign",
            instructions="A warm, natural young adult voice.",
        ),
        judge="qwen3-asr",
    ),
    BaselineModel(
        row=LANE_ROWS["dots-tts-base"],
        name="dots.tts Base",
        probe="speech",
        payload=REFERENCE_LIST,
        judge="qwen3-asr",
    ),
    BaselineModel(
        row=LANE_ROWS["dots-tts-mf"],
        name="dots.tts MF",
        probe="speech",
        payload=REFERENCE_LIST,
        judge="qwen3-asr",
    ),
    BaselineModel(
        row=LANE_ROWS["ming-flash-omni"], name="Ming-flash-omni-2.0", probe="chat"
    ),
    BaselineModel(row=LANE_ROWS["minicpm-o"], name="MiniCPM-o 4.5", probe="chat"),
    BaselineModel(row=LANE_ROWS["personaplex"], name="PersonaPlex", probe="speech"),
    BaselineModel(
        row=LANE_ROWS["nemotron-voicechat"], name="Nemotron VoiceChat", probe="speech"
    ),
    BaselineModel(
        row=LANE_ROWS["audar-tts"], name="Audar-TTS V1 Turbo", probe="speech"
    ),
)
MODEL_BY_KEY = {model.key: model for model in MODELS}


class Probe(Protocol):
    def __call__(self, base_url: str, model: BaselineModel) -> Sample: ...


class CardsBusy(RuntimeError):
    """Raised instead of measuring when another process holds a card, so it is not read as a regression."""


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level edit distance over reference length; jiwer is not in the XPU install."""
    want = normalize_text(reference).split()
    got = normalize_text(hypothesis).split()
    if not want:
        return 0.0 if not got else 1.0
    else:
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


def installed_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def environment_fingerprint(host_label: str = "") -> dict[str, str]:
    """The machine, card and software a baseline belongs to; host_label names the machine from inside a container."""
    device = (
        torch.xpu.get_device_properties(0).name
        if torch.xpu.is_available()
        else "unknown"
    )
    # oneCCL decides whether a TP>1 decode graph replays its all-reduces, so it is part of the environment.
    return {
        "host": host_label or socket.gethostname(),
        "device": device,
        "torch": torch.__version__,
        "sglang": sglang.__version__,
        "oneccl": installed_version("oneccl"),
    }


def fingerprint_key(fingerprint: dict[str, str]) -> str:
    blob = json.dumps(fingerprint, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def recorded_from_commit() -> str:
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
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", text).strip().lower())


def audio_shape(raw: bytes) -> dict[str, float]:
    """Frame count, rate, length and loudness of a WAV body."""
    with wave.open(io.BytesIO(raw)) as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        width = handle.getsampwidth()
        pcm = handle.readframes(frames)

    if frames == 0 or rate == 0:
        raise AssertionError("audio holds no frames")
    else:
        scale = float(1 << (8 * width - 1))
    samples = memoryview(pcm).cast({1: "b", 2: "h", 4: "i"}[width])
    peak = max(abs(sample) for sample in samples) / scale
    rms = math.sqrt(sum((sample / scale) ** 2 for sample in samples) / len(samples))
    if peak <= 0.01:
        raise AssertionError(
            f"audio peaks at {peak:.4f} of full scale, which is silence"
        )
    else:
        pass
    return {
        "frames": float(frames),
        "sample_rate": float(rate),
        "audio_seconds": frames / rate,
        "peak": peak,
        "rms": rms,
    }


def transcribe(base_url: str, model_path: str, audio: bytes, filename: str) -> str:
    response = requests.post(
        f"{base_url}/v1/audio/transcriptions",
        files={"file": (filename, audio, "audio/wav")},
        data={"model": model_path, "temperature": "0"},
        timeout=300,
    )
    response.raise_for_status()
    return response.json().get("text", "")


def probe_transcription(base_url: str, model: BaselineModel) -> Sample:
    audio = REFERENCE_AUDIO.read_bytes()
    started = time.perf_counter()
    transcript = transcribe(base_url, model.model_path, audio, REFERENCE_AUDIO.name)
    elapsed = time.perf_counter() - started
    if not transcript.strip():
        raise AssertionError("empty transcript")
    else:
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
    base_url: str, model: BaselineModel, text: str, timeout_seconds: int
) -> tuple[bytes, float]:
    request_body = {
        "model": model.model_path,
        "input": text,
        "response_format": "wav",
        **({} if model.seedless else {"seed": 0}),
        **model.payload,
    }
    started = time.perf_counter()
    response = requests.post(
        f"{base_url}/v1/audio/speech", json=request_body, timeout=timeout_seconds
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    return response.content, elapsed


def probe_speech(base_url: str, model: BaselineModel, judge_url: str) -> Sample:
    audio, elapsed = synthesize(base_url, model, SPOKEN_TEXT, timeout_seconds=600)
    shape = audio_shape(audio)
    result: Sample = {
        "exact": {"sample_rate": shape["sample_rate"]},
        "atmost": {},
        # Seeded tokens repeat exactly, but a kernel change moves the vocoder's floats, so loudness gets a band.
        "close": {"peak": shape["peak"], "rms": shape["rms"]},
        "speed": {
            "latency_s": elapsed,
            "audio_seconds_per_wall_second": shape["audio_seconds"] / elapsed,
        },
    }
    if model.seedless:
        # Unseeded, the model picks its own length, so length and everything derived from it are only observed.
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
    else:
        pass
    return result


def probe_music(base_url: str, model: BaselineModel) -> Sample:
    audio, elapsed = synthesize(base_url, model, MUSIC_PROMPT, timeout_seconds=900)
    shape = audio_shape(audio)
    return {
        "exact": {"sample_rate": shape["sample_rate"]},
        "close": {"audio_seconds": shape["audio_seconds"], "rms": shape["rms"]},
        "speed": {"latency_s": elapsed},
    }


def probe_chat(base_url: str, model: BaselineModel) -> Sample:
    request_body = {
        "model": model.model_path,
        "messages": [{"role": "user", "content": ARITHMETIC_PROMPT}],
        "max_tokens": 32,
        "temperature": 0,
        "seed": 0,
    }
    started = time.perf_counter()
    response = requests.post(
        f"{base_url}/v1/chat/completions", json=request_body, timeout=600
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    completion = response.json()
    content = completion["choices"][0]["message"]["content"]
    if not content:
        raise AssertionError("no assistant content")
    else:
        completion_tokens = completion.get("usage", {}).get("completion_tokens", 0)
    speed = {"latency_s": elapsed}
    if completion_tokens:
        speed["decode_tokens_per_second"] = completion_tokens / elapsed
    else:
        pass
    return {"exact": {"content": normalize_text(content)}, "speed": speed}


PROBES: dict[ProbeKind, Probe] = {
    "transcription": probe_transcription,
    "music": probe_music,
    "chat": probe_chat,
}

ENGINE_STAGE_CACHE: dict[str, tuple[str, ...]] = {}


def engine_stages(model_path: str, variant: str = "") -> tuple[str, ...]:
    """Stage names that own an SGLang engine, read from the model's pipeline config so an override names a real stage."""
    cache_key = f"{model_path}@{variant}"
    if cache_key in ENGINE_STAGE_CACHE:
        return ENGINE_STAGE_CACHE[cache_key]
    else:
        pass
    try:
        config_cls = resolve_config_cls_for_model_path(model_path)
        if variant:
            variants = getattr(
                importlib.import_module(config_cls.__module__), "Variants", {}
            )
            config_cls = variants.get(variant, config_cls)
        else:
            pass
        config = config_cls(model_path=model_path)
        stages = tuple(
            stage.name for stage in config.stages if getattr(stage, "engine", None)
        )
    except Exception as exc:
        print(f"  could not derive engine stages ({exc}); serving without them")
        stages = ()
    ENGINE_STAGE_CACHE[cache_key] = stages
    return stages


def serve_command(
    model: BaselineModel, port: int, extra: tuple[str, ...] = ()
) -> list[str]:
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
        for option, value in DETERMINISM_ENGINE_ARGS.items():
            command += [f"--{stage}.engine.{option}", value]
    return command


def launch(
    model: BaselineModel, cards: str, extra: tuple[str, ...] = ()
) -> tuple[subprocess.Popen[bytes], str]:
    port = free_port()
    env = dict(os.environ, ZE_AFFINITY_MASK=cards, **model.env)
    env.setdefault("SGLANG_OMNI_STARTUP_TIMEOUT", str(model.startup_seconds))
    # The server strips proxy variables, so a Hub lookup there waits out a connect timeout instead of reading the cache.
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
    """The nearest existing ancestor of path, for sizing a filesystem a cache will be created on."""
    while path and not os.path.isdir(path):
        path = os.path.dirname(path)
    return path or "/"


def cache_volume_free_gib() -> float:
    return shutil.disk_usage(existing_ancestor(HF_HUB_CACHE)).free / 2**30


def needed_checkpoints(model: BaselineModel) -> list[str]:
    """The model paths one measurement needs locally, its judge's and companions included."""
    paths = [model.model_path, *model.companions]
    if model.judge:
        judge = MODEL_BY_KEY[model.judge]
        paths += [judge.model_path, *judge.companions]
    else:
        pass
    return paths


def prune_hub_cache(model: BaselineModel) -> str:
    """Delete every cached revision this measurement does not need; opt-in, since the cache may be shared."""
    if not os.path.isdir(HF_HUB_CACHE):
        return "no hub cache to prune yet"
    else:
        keep = {path.partition("@")[0] for path in needed_checkpoints(model)}
    before = cache_volume_free_gib()
    cache_info = scan_cache_dir(HF_HUB_CACHE)
    doomed = [
        revision.commit_hash
        for repo in cache_info.repos
        if repo.repo_id not in keep
        for revision in repo.revisions
    ]
    if doomed:
        cache_info.delete_revisions(*doomed).execute()
    else:
        pass
    free_now = cache_volume_free_gib()
    return (
        f"pruned {len(doomed)} revision(s), freed {free_now - before:.1f} GiB, "
        f"{free_now:.1f} GiB free"
    )


def prefetch_checkpoints(model: BaselineModel) -> str:
    """Fetch this measurement's weights before the server starts, for a server with no network."""
    for path in needed_checkpoints(model):
        repo, _, revision = path.partition("@")
        snapshot_download(repo, revision=revision or None, max_workers=8)
    free_now = cache_volume_free_gib()
    return f"fetched {len(needed_checkpoints(model))} checkpoint(s), {free_now:.1f} GiB free"


def download_blocked(model_path: str, max_checkpoint_gib: float) -> str:
    """Why this checkpoint will not be fetched here, or an empty string if it will."""
    repo, _, revision = model_path.partition("@")
    blobs = os.path.join(HF_HUB_CACHE, f"models--{repo.replace('/', '--')}", "blobs")
    # A resolving link is a finished blob; a plain file is a transfer in progress and counts the disk it holds.
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
        repo_info = HfApi().model_info(
            repo, revision=revision or None, files_metadata=True
        )
        repo_gib = sum(item.size or 0 for item in repo_info.siblings) / 2**30
    except Exception as exc:
        print(f"  could not size {repo} before fetching it ({exc})")
        return ""
    usage = os.statvfs(existing_ancestor(HF_HUB_CACHE))
    free_gib = usage.f_bavail * usage.f_frsize / 2**30
    # The repo total is an upper bound (Whisper publishes its weights three ways), so it only decides the cold case.
    if repo_gib > max_checkpoint_gib:
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


def visible_cards() -> list[int]:
    """Physical ids of the cards this process may use, in torch's order: an inherited mask applies ascending."""
    mask = os.environ.get("ZE_AFFINITY_MASK", "").strip()
    if mask:
        return sorted({int(part) for part in mask.split(",") if part.strip()})
    else:
        return list(range(torch.xpu.device_count()))


def cards_in_use(cards: str) -> dict[str, float]:
    """MiB already allocated on each physical card, device-wide so a neighbour's work shows."""
    visible = visible_cards()
    index = {card: visible.index(int(card)) for card in cards.split(",")}
    used: dict[str, float] = {}
    for card, device in index.items():
        free, total = torch.xpu.mem_get_info(device)
        used[card] = round((total - free) / 1024**2, 1)
    return used


def allocate_cards(pool: list[int], count: int, taken: set[int]) -> str:
    """The lowest unused cards of the pool as a ZE_AFFINITY_MASK, so a model lands on the same cards every run."""
    free = [card for card in pool if card not in taken]
    if len(free) < count:
        raise AssertionError(
            f"need {count} card(s), pool {pool} has {len(free)} free ({sorted(taken)} in use)"
        )
    else:
        chosen = free[:count]
    taken.update(chosen)
    return ",".join(str(card) for card in chosen)


def measure_launches(
    model: BaselineModel,
    pool: list[int],
    repeats: int,
    warmups: int,
    extras: dict[str, tuple[str, ...]],
    launches: int,
    idle_mib: float,
    idle_wait_s: int,
) -> Record:
    """Measure across whole server restarts, the variance a gate has to survive and repeats cannot see."""
    runs = []
    for index in range(launches):
        if launches > 1:
            print(f"  launch {index + 1}/{launches}", flush=True)
        else:
            pass
        runs.append(
            measure(model, pool, repeats, warmups, extras, idle_mib, idle_wait_s)
        )
    if launches == 1:
        return runs[0]
    else:
        folded = fold_samples(runs)
    folded["speed"] = pessimistic_speed(runs)
    for field_name in ("cards", "judge_cards"):
        layouts = sorted({run[field_name] for run in runs if run.get(field_name)})
        if layouts:
            folded[field_name] = " | ".join(layouts)
        else:
            pass
    # The worst trimmed within-run dispersion, which sizes the tolerance once the value is the slowest launch.
    worst_within: dict[str, float] = {}
    for run in runs:
        per_run = run.get("trimmed_spread") or run.get("spread", {})
        for name, value in per_run.items():
            worst_within[name] = max(worst_within.get(name, 0.0), value)
    folded["within_run_spread"] = worst_within
    folded["launches"] = launches
    return folded


def pessimistic_speed(runs: list[Record]) -> dict[str, float]:
    """The slowest value each speed field reached across launches: these models are modal, not noisy."""
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


def settled_after_run(
    cards: str, idle_mib: float, wait_s: int = 60
) -> dict[str, float]:
    """Card memory once this run's own servers have released it; what remains belongs to another process."""
    deadline = time.time() + wait_s
    used = cards_in_use(cards)
    while any(mib > idle_mib for mib in used.values()) and time.time() < deadline:
        time.sleep(5)
        used = cards_in_use(cards)
    return used


def measure(
    model: BaselineModel,
    pool: list[int],
    repeats: int,
    warmups: int,
    extras: dict[str, tuple[str, ...]],
    idle_mib: float = 1024.0,
    idle_wait_s: int = 0,
) -> Record:
    """Serve the model, and its judge when it has one, then sample it warmups+repeats times."""
    servers = []
    taken: set[int] = set()
    cards = allocate_cards(pool, model.cards, taken)
    judge = MODEL_BY_KEY[model.judge] if model.judge else None
    judge_cards = allocate_cards(pool, judge.cards, taken) if judge else ""
    layout = ",".join(part for part in (cards, judge_cards) if part)
    busy_before = cards_in_use(layout)
    deadline = time.time() + idle_wait_s
    while True:
        noisy = {card: mib for card, mib in busy_before.items() if mib > idle_mib}
        if not noisy:
            break
        elif time.time() >= deadline:
            raise CardsBusy(
                f"card(s) not idle before launch: {noisy} MiB in use, over the "
                f"{idle_mib:g} MiB bar; another process on the card makes this a "
                "different machine"
            )
        else:
            print(f"  waiting for {noisy} MiB to clear", flush=True)
        time.sleep(30)
        busy_before = cards_in_use(layout)
    try:
        judge_url = ""
        if judge:
            judge_proc, judge_url = launch(
                judge, judge_cards, extras.get(judge.key, ())
            )
            servers.append(judge_proc)
        else:
            pass
        proc, base_url = launch(model, cards, extras.get(model.key, ()))
        servers.append(proc)

        samples = []
        for index in range(warmups + repeats):
            if model.probe == "speech":
                sample = probe_speech(base_url, model, judge_url)
            else:
                sample = PROBES[model.probe](base_url, model)
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

    busy_after = settled_after_run(layout, idle_mib)
    noisy_after = {card: mib for card, mib in busy_after.items() if mib > idle_mib}
    if noisy_after:
        raise CardsBusy(
            f"card(s) still in use after this run's servers exited: {noisy_after} MiB, "
            f"over the {idle_mib:g} MiB bar; another process arrived during the run, so "
            "its numbers are not kept"
        )
    else:
        folded = fold_samples(samples[warmups:])
    folded["cards"] = cards
    if judge_cards:
        folded["judge_cards"] = judge_cards
    else:
        pass
    folded["cards_used_mib"] = {"before": busy_before, "after": busy_after}
    return folded


def fold_samples(samples: list[Sample] | list[Record]) -> Record:
    """Collapse repeats: exact fields must agree, a ceiling takes the worst sample, the rest the median."""
    folded: Record = {group: {} for group in COMPARED_GROUPS}
    folded["spread"] = {}
    for group in COMPARED_GROUPS:
        for name in samples[0].get(group, {}):
            values = [sample[group][name] for sample in samples]
            if group == "exact":
                unique = set(values)
                if len(unique) != 1:
                    raise AssertionError(
                        f"{name} is not reproducible across repeats: {sorted(unique)}"
                    )
                else:
                    folded[group][name] = values[0]
                continue
            else:
                folded[group][name] = (
                    max(values) if group == "atmost" else statistics.median(values)
                )
            if len(values) > 1 and folded[group][name]:
                span = (max(values) - min(values)) / folded[group][name]
                folded["spread"][f"{group}.{name}"] = round(span, 4)
                trimmed = trimmed_span(values, folded[group][name])
                if group == "speed" and trimmed is not None:
                    folded.setdefault("trimmed_spread", {})[f"{group}.{name}"] = trimmed
                else:
                    pass
                drift = warming_drift(values) if group == "speed" else None
                if drift:
                    early, late = drift
                    print(
                        f"    still warming on {name}: the run's two halves sit apart "
                        f"({early:.6g} then {late:.6g}); raise --warmups before trusting the spread"
                    )
                else:
                    pass
            else:
                pass
    if samples[0].get("observed"):
        folded["observed"] = {}
        folded["observed_span"] = {}
        for name in samples[0]["observed"]:
            values = [sample["observed"][name] for sample in samples]
            middle = statistics.median(values)
            folded["observed"][name] = middle
            folded["observed_span"][name] = round(
                (max(values) - min(values)) / (middle or 1), 4
            )
    else:
        pass
    return folded


def trimmed_span(values: list[float], middle: float) -> float | None:
    """Spread with the best and worst sample dropped, which is what a gate is sized from; None below five samples."""
    if len(values) < 5:
        return None
    else:
        inner = sorted(values)[1:-1]
        return round((max(inner) - min(inner)) / middle, 4)


def warming_drift(values: list[float]) -> tuple[float, float] | None:
    """The two half-medians when they sit more than 3 percent apart, which means the run is still settling."""
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
    key: str, baseline: Record, observed: Record, tolerances: Tolerances
) -> tuple[list[str], list[str], list[str]]:
    """Return regressions, improvements and notes; only a regression fails the run."""
    failures: list[str] = []
    gains: list[str] = []
    notes: list[str] = []

    recorded_cards = baseline.get("cards")
    observed_cards = observed.get("cards")
    if not recorded_cards:
        notes.append(
            f"{key}: recorded without its card assignment, so a change of card cannot be "
            "detected; re-record this entry"
        )
    elif observed_cards and recorded_cards != observed_cards:
        notes.append(
            f"{key}: recorded on card(s) {recorded_cards}, measured on {observed_cards}; "
            "speed is not comparable across cards on a host with uneven PCIe links"
        )
    else:
        pass

    for name, want in baseline.get("exact", {}).items():
        got = observed.get("exact", {}).get(name)
        if got != want:
            failures.append(
                f"{key}: {name} changed\n    baseline {want!r}\n    observed {got!r}"
            )
        else:
            pass

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
        else:
            pass

    for name, want in baseline.get("close", {}).items():
        got = observed.get("close", {}).get(name)
        if got is None:
            failures.append(f"{key}: {name} missing from this run")
        elif want and abs(got - want) / want > tolerances["close"]:
            failures.append(
                f"{key}: {name} moved {(got - want) / want:+.2%} "
                f"(baseline {want:.6g}, observed {got:.6g}, allowed +-{tolerances['close']:.1%})"
            )
        else:
            pass

    for name, want in baseline.get("speed", {}).items():
        got = observed.get("speed", {}).get(name)
        if got is None:
            failures.append(f"{key}: {name} missing from this run")
            continue
        else:
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
        else:
            pass
    return failures, gains, notes


def speed_tolerance_for(baseline: Record, name: str, tolerances: Tolerances) -> float:
    """The larger of the floor and headroom times the spread the field showed when recorded."""
    # A multi-launch entry already holds its slowest launch, so its launch-to-launch spread is not charged twice.
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


def parse_arguments() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models", default="all", help="comma-separated keys, or 'all'"
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
        default="",
        help="cards this run may claim, by physical id; each model takes what its topology "
        "needs (default: every card this process can see, within any inherited "
        "ZE_AFFINITY_MASK)",
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
    parser.add_argument(
        "--serve-extra",
        action="append",
        default=[],
        metavar="KEY=ARGS",
        help="extra serve args for one model on this host, recorded with its entry, e.g. qwen3-tts=--vocoder.factory.async_decode false",
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
        help="skip a model whose checkpoint is larger than this, cached or not; it keeps its row and the run still exits 0",
    )
    parser.add_argument("--list", action="store_true")
    return parser, parser.parse_args()


def main() -> int:
    parser, args = parse_arguments()

    if args.list:
        recorded_on: dict[str, list[str]] = {}
        if args.baseline.exists():
            listed: BaselineFile = json.loads(args.baseline.read_text())
            for environment in listed["environments"].values():
                for key in environment.get("models", {}):
                    recorded_on.setdefault(key, []).append(
                        environment["fingerprint"]["host"]
                    )
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
    else:
        pass

    keys = (
        [model.key for model in MODELS]
        if args.models == "all"
        else args.models.split(",")
    )
    unknown = [key for key in keys if key not in MODEL_BY_KEY]
    if unknown:
        parser.error(f"unknown model keys: {unknown}; see --list")
    else:
        pass

    extras: dict[str, tuple[str, ...]] = {}
    for item in args.serve_extra:
        key, _, argument_text = item.partition("=")
        if key not in MODEL_BY_KEY or not argument_text:
            parser.error(f"--serve-extra wants <model key>=<args>, got {item!r}")
        else:
            extras[key] = tuple(shlex.split(argument_text))

    mask = os.environ.get("ZE_AFFINITY_MASK", "").strip()
    try:
        visible = visible_cards()
    except ValueError:
        parser.error(f"ZE_AFFINITY_MASK={mask!r} names sub-devices; give whole cards")
    pool = (
        [int(card) for card in args.card_pool.split(",") if card.strip()]
        if args.card_pool
        else visible
    )
    outside = sorted(set(pool) - set(visible))
    if outside:
        parser.error(
            f"--card-pool {args.card_pool} names card(s) {outside} this process cannot use; "
            f"it can use {visible}" + (f" (ZE_AFFINITY_MASK={mask})" if mask else "")
        )
    else:
        pass
    print(f"card pool {pool}", flush=True)
    fingerprint = environment_fingerprint(args.host_label)
    env_key = fingerprint_key(fingerprint)
    print(f"environment {env_key}: {json.dumps(fingerprint)}")

    stored: BaselineFile = {"environments": {}}
    if args.baseline.exists():
        stored = json.loads(args.baseline.read_text())
    else:
        pass
    entry = stored["environments"].get(env_key, {})

    if not args.record and not args.calibrate and not entry:
        known = [
            environment.get("fingerprint", {})
            for environment in stored["environments"].values()
        ]
        print(
            f"no baseline for environment {env_key} in {args.baseline}.\n"
            f"  known environments: {known or 'none'}\n"
            "  run with --record to add this one.",
            file=sys.stderr,
        )
        return 2
    else:
        pass

    # A check measures the recipe its entry was recorded with; a flag given here still wins.
    if not args.record:
        restored = {
            key: tuple(value)
            for key, value in entry.get("serve_extra", {}).items()
            if key not in extras
        }
        for key, value in sorted(restored.items()):
            print(f"serve flags for {key}, from this entry: {shlex.join(value)}")
        extras = {**restored, **extras}
    else:
        pass

    # An explicit flag wins, then what this environment recorded, then the built-in.
    flag_for = {
        "close": args.close_tolerance,
        "absolute": args.absolute_tolerance,
        "speed": args.speed_tolerance,
        "speed_headroom": args.speed_headroom,
    }
    recorded_tolerances = entry.get("tolerances", {})
    tolerances: Tolerances = dict(BUILTIN_TOLERANCES)
    for name in BUILTIN_TOLERANCES:
        if flag_for[name] is not None:
            tolerances[name], source = flag_for[name], "the flag"
        elif name in recorded_tolerances:
            tolerances[name], source = recorded_tolerances[name], "this entry"
        else:
            source = "the built-in default"
        if name == "speed":
            print(f"speed floor {tolerances[name]:.1%}, from {source}")
        else:
            pass

    def flush(partial: dict[str, Record]) -> None:
        """Persist what is measured so far, so an interrupted sweep resumes instead of restarting."""
        on_disk: BaselineFile = {"environments": {}}
        if args.baseline.exists():
            on_disk = json.loads(args.baseline.read_text())
        else:
            pass
        previous = on_disk["environments"].get(env_key, {})
        # Provenance belongs to the model it was measured with: a re-record can clear a workaround.
        serve_extra = {
            key: value
            for key, value in previous.get("serve_extra", {}).items()
            if key not in partial
        }
        serve_extra.update(
            {key: list(value) for key, value in extras.items() if key in partial}
        )
        on_disk["environments"][env_key] = {
            "fingerprint": fingerprint,
            "recorded_from_commit": recorded_from_commit(),
            "tolerances": tolerances,
            "serve_extra": serve_extra,
            "models": {**previous.get("models", {}), **partial},
        }
        args.baseline.write_text(json.dumps(on_disk, indent=2, sort_keys=True) + "\n")

    results: dict[str, Record] = {}
    failures: list[str] = []
    gains: list[str] = []
    notes: list[str] = []
    skipped: list[str] = []
    done = entry.get("models", {}) if args.record and not args.fresh else {}
    for index, key in enumerate(keys, start=1):
        model = MODEL_BY_KEY[key]
        print(f"\n== [{index}/{len(keys)}] {model.name} ({key}) ==", flush=True)
        judge_cards = MODEL_BY_KEY[model.judge].cards if model.judge else 0
        if model.unservable:
            print(f"  SKIPPED: {model.unservable}", flush=True)
            skipped.append(f"{key}: {model.unservable}")
            continue
        elif key in done:
            print(
                "  already recorded for this environment; --fresh to redo", flush=True
            )
            skipped.append(f"{key}: already recorded")
            continue
        elif model.cards + judge_cards > len(pool):
            reason = (
                f"needs {model.cards} card(s)"
                + (f" and {judge_cards} for its judge" if judge_cards else "")
                + f", pool has {len(pool)}"
            )
            print(f"  SKIPPED: {reason}", flush=True)
            skipped.append(f"{key}: {reason}")
            continue
        else:
            pass
        if args.prune_hf_cache:
            print(f"  {prune_hub_cache(model)}", flush=True)
        else:
            pass
        shortfall = download_blocked(model.model_path, args.max_checkpoint_gib)
        if shortfall:
            print(f"  SKIPPED: {shortfall}", flush=True)
            skipped.append(f"{key}: {shortfall}")
            continue
        elif args.prefetch:
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
        else:
            pass
        print(f"  median  {json.dumps(observed['speed'])}")
        print(
            f"  accuracy {json.dumps(observed['exact'])} {json.dumps(observed.get('atmost', {}))}"
        )
        print(f"  spread  {json.dumps(observed['spread'])}")
        if args.record or args.calibrate:
            pass
        elif key not in entry.get("models", {}):
            failures.append(
                f"{key}: absent from this environment's baseline; re-record to add it"
            )
        else:
            model_failures, model_gains, model_notes = compare(
                key, entry["models"][key], observed, tolerances
            )
            failures += model_failures
            gains += model_gains
            notes += model_notes
            for message in model_gains:
                print(f"  IMPROVED {message.split(': ', 1)[1]}", flush=True)
            for message in model_notes:
                print(f"  NOTE {message.split(': ', 1)[1]}", flush=True)
        print(
            f"  running tally: {len(results)} ok, {len(failures)} failed, {len(skipped)} skipped",
            flush=True,
        )

    if args.record and results:
        flush(results)
        print(
            f"\nwrote {len(results)} model(s) for environment {env_key} to {args.baseline}"
        )
    elif args.record:
        print("\nnothing measured; leaving the baseline untouched")
    else:
        pass

    if args.calibrate:
        print("\nobserved spread per field, as a fraction of the median:")
        for key, observed in results.items():
            for name, span in sorted(observed["spread"].items()):
                print(f"  {key:22s} {name:46s} {span:.2%}")
        print(
            "\nSet --speed-tolerance above the largest speed spread, or the gate will be flaky."
        )
        for message in failures:
            print(f"FAILED      {message}")
        return 1 if failures else 0
    else:
        pass

    print()
    # While recording there is nothing to regress against: a failure is a model that would not serve or answer.
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
    else:
        pass
    for message in notes:
        print(f"NOTE        {message}")
    for message in skipped:
        print(f"SKIPPED     {message}")
    print(
        f"{len(results)} measured, {len(skipped)} skipped, {len(failures)} "
        f"{'failure(s)' if args.record else 'regression(s)'}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
