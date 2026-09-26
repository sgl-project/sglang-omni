# SPDX-License-Identifier: Apache-2.0
"""Opt-in greedy parity of the PersonaPlex port against the reference offline script.

Set-up and the expected numbers are under Tests in docs/cookbook/personaplex.md.
"""

from __future__ import annotations

import asyncio
import base64
import itertools
import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import soundfile
import torch

from sglang_omni.models.personaplex.architecture import (
    MIMI_WEIGHTS_GLOB,
    MOSHI_WEIGHTS_NAME,
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
)
from sglang_omni.models.personaplex.prompts import (
    DEFAULT_TEXT_PROMPT,
    TEXT_TOKENIZER_NAME,
    resolve_voice_path,
)
from tests.test_model.personaplex_repro import (
    ReferenceInputs,
    checkpoint_digests,
    ensure_reference_run,
    file_digest,
    pinned_checkpoint,
    reference_checkout,
)

pytestmark = pytest.mark.accelerator

DEFAULT_CHECKPOINT = "nvidia/personaplex-7b-v1"
# The reference README's seed; irrelevant under --greedy, kept so the command matches.
REFERENCE_SEED = 42424242
# The reference maps BOS and EOS through the tokenizer, so they arrive as <s> and </s>.
REFERENCE_TEXT_MARKERS = frozenset({"EPAD", "BOS", "EOS", "PAD", "<s>", "</s>"})
DEFAULT_ATOL = 1e-4  # a few int16 steps, for rounding between the two codec paths


@dataclass(frozen=True, kw_only=True)
class ParityCase:
    input_wav: str
    voice: str
    text_prompt_file: str | None
    min_identical_frames: int


# Minimums are the lowest observed against several reference runs (100-114
# frames on the assistant recording, 109 on the service one).
CASES = {
    "assistant": ParityCase(
        input_wav="input_assistant.wav",
        voice="NATF2",
        text_prompt_file=None,
        min_identical_frames=100,
    ),
    "service": ParityCase(
        input_wav="input_service.wav",
        voice="NATM1",
        text_prompt_file="prompt_service.txt",
        min_identical_frames=100,
    ),
}


@dataclass(kw_only=True)
class Reply:
    text: str
    audio: np.ndarray  # float32 mono at 24 kHz


@dataclass(kw_only=True)
class FrameParity:
    total_frames: int
    identical_frames: int
    max_diff_before_divergence: float


def read_wav(path: Path) -> np.ndarray:
    data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path} is {rate} Hz, expected {SAMPLE_RATE}")
    return np.ascontiguousarray(data[:, 0])


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def reference_text(pieces: list[str], frames: int | None = None) -> str:
    """Reply text from the reference's per-frame token pieces, markers dropped."""
    if frames is not None:
        pieces = pieces[:frames]
    return normalize_text("".join(p for p in pieces if p not in REFERENCE_TEXT_MARKERS))


def compare_frames(port: np.ndarray, reference: np.ndarray, atol: float) -> FrameParity:
    if port.shape != reference.shape:
        raise ValueError(
            f"Audio sample counts differ: {port.shape} versus {reference.shape}"
        )
    if not port.size:
        raise ValueError("Cannot compare empty audio")
    if not np.isfinite(port).all() or not np.isfinite(reference).all():
        raise ValueError("Audio contains non-finite samples")
    per_frame = np.maximum.reduceat(
        np.abs(port - reference), np.arange(0, port.size, SAMPLES_PER_FRAME)
    )
    frames = len(per_frame)
    identical = per_frame <= atol
    first_divergence = frames if identical.all() else int(np.argmin(identical))
    return FrameParity(
        total_frames=frames,
        identical_frames=first_divergence,
        max_diff_before_divergence=(
            float(per_frame[:first_divergence].max()) if first_divergence else 0.0
        ),
    )


def text_prompt_for(case: ParityCase, assets: Path) -> str | None:
    return (
        None
        if case.text_prompt_file is None
        else (assets / case.text_prompt_file).read_text().strip()
    )


@pytest.fixture(scope="module")
def assets_dir() -> Path:
    return reference_checkout() / "assets" / "test"


@pytest.fixture(scope="module")
def checkpoint(assets_dir: Path) -> Path:
    return pinned_checkpoint("PERSONAPLEX_PARITY_CHECKPOINT")


@pytest.fixture(scope="module")
def reference_outputs(
    assets_dir: Path, checkpoint: Path, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Validate or generate reference outputs before the port starts on the GPU."""
    python = os.environ.get("PERSONAPLEX_REFERENCE_PYTHON")
    configured = os.environ.get("PERSONAPLEX_REFERENCE_DIR")
    root = (
        Path(configured).expanduser().resolve()
        if configured
        else tmp_path_factory.mktemp("personaplex-reference")
    )
    source = assets_dir.parents[1]
    weights = checkpoint_digests(checkpoint)
    (mimi_weight,) = checkpoint.glob(MIMI_WEIGHTS_GLOB)
    outputs = {}
    for name, case in CASES.items():
        output = root / name
        voice = resolve_voice_path(checkpoint, case.voice).resolve()
        prompt = text_prompt_for(case, assets_dir) or DEFAULT_TEXT_PROMPT
        inputs = ReferenceInputs(
            revision=os.environ["PERSONAPLEX_REFERENCE_REVISION"],
            checkpoint=weights,
            files={
                "input_wav": file_digest(assets_dir / case.input_wav),
                "voice": file_digest(voice),
            },
            settings={"text_prompt": prompt, "greedy": 1, "seed": REFERENCE_SEED},
        )
        command = (
            [
                python,
                "-m",
                "moshi.offline",
                "--hf-repo",
                os.environ.get("PERSONAPLEX_REFERENCE_REPO", DEFAULT_CHECKPOINT),
                "--moshi-weight",
                str(checkpoint / MOSHI_WEIGHTS_NAME),
                "--mimi-weight",
                str(mimi_weight),
                "--tokenizer",
                str(checkpoint / TEXT_TOKENIZER_NAME),
                "--voice-prompt-dir",
                str(voice.parent),
                "--voice-prompt",
                voice.name,
                "--text-prompt",
                prompt,
                "--input-wav",
                str(assets_dir / case.input_wav),
                "--greedy",
                "--seed",
                str(REFERENCE_SEED),
                "--output-wav",
                str(output / "output.wav"),
                "--output-text",
                str(output / "output.json"),
            ]
            if python
            else None
        )
        ensure_reference_run(
            command=command,
            source=source,
            inputs=inputs,
            manifest=output / "manifest.json",
            artifacts=(output / "output.wav", output / "output.json"),
        )
        outputs[name] = (
            read_wav(output / "output.wav"),
            json.loads((output / "output.json").read_text()),
        )
    return outputs


@pytest.fixture(scope="module")
def generate(reference_outputs, checkpoint: Path):
    if not torch.cuda.is_available():
        pytest.skip("PersonaPlex parity requires CUDA")
    from sglang_omni.client import Client, GenerateRequest, SamplingParams
    from sglang_omni.config.manager import ConfigManager
    from sglang_omni.models.personaplex.config import PersonaPlexPipelineConfig
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
    from sglang_omni.proto import EXPLICIT_GENERATION_PARAMS_KEY

    config = PersonaPlexPipelineConfig(model_path=str(checkpoint))
    overrides = shlex.split(os.environ.get("PERSONAPLEX_PARITY_STAGE_ARGS", ""))
    if overrides:
        manager = ConfigManager(config)
        config = manager.merge_config(manager.parse_extra_args(overrides))
    runner = MultiProcessPipelineRunner(config)
    loop = asyncio.new_event_loop()
    loop.run_until_complete(
        runner.start(
            timeout=float(os.environ.get("SGLANG_OMNI_STARTUP_TIMEOUT", "900"))
        )
    )
    client = Client(runner.coordinator)
    request_ids = itertools.count(1)

    def generate_reply(
        audio_path: Path,
        *,
        voice: str,
        text_prompt: str | None,
        greedy: bool,
        seed: int | None = None,
    ) -> Reply:
        extra = {"voice": voice}
        if text_prompt is not None:
            extra["text_prompt"] = text_prompt
        if greedy:
            extra["audio_temperature"] = 0.0
        if seed is not None:
            extra["seed"] = seed
        request = GenerateRequest(
            model=config.name,
            prompt={"audio_path": str(audio_path)},
            sampling=SamplingParams(temperature=0.0) if greedy else SamplingParams(),
            extra_params=extra,
            metadata={
                EXPLICIT_GENERATION_PARAMS_KEY: ["temperature"] if greedy else []
            },
            output_modalities=["text", "audio"],
            stream=False,
        )
        result = loop.run_until_complete(
            client.completion(
                request,
                request_id=f"parity-{next(request_ids)}",
                audio_format="pcm",
            )
        )
        blob = result.audio.data if result.audio else b""
        pcm = base64.b64decode(blob) if isinstance(blob, str) else blob
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        return Reply(text=result.text or "", audio=audio)

    try:
        yield generate_reply
    finally:
        loop.run_until_complete(runner.stop())
        loop.close()


@pytest.fixture(scope="module")
def port_greedy(generate, assets_dir: Path) -> dict[str, Reply]:
    return {
        name: generate(
            assets_dir / case.input_wav,
            voice=case.voice,
            text_prompt=text_prompt_for(case, assets_dir),
            greedy=True,
        )
        for name, case in CASES.items()
    }


@pytest.mark.parametrize("name", list(CASES))
def test_greedy_matches_reference(
    name, port_greedy, reference_outputs, assets_dir: Path
):
    case = CASES[name]
    reply = port_greedy[name]
    ref_audio, ref_pieces = reference_outputs[name]
    atol = float(os.environ.get("PERSONAPLEX_PARITY_ATOL", DEFAULT_ATOL))

    expected_samples = read_wav(assets_dir / case.input_wav).size
    assert reply.audio.size == ref_audio.size == expected_samples, (
        f"{name}: expected {expected_samples} samples, got "
        f"port={reply.audio.size}, reference={ref_audio.size}"
    )
    expected_frames = (expected_samples + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME
    assert len(ref_pieces) == expected_frames
    parity = compare_frames(reply.audio, ref_audio, atol)
    ref_text_prefix = reference_text(ref_pieces, parity.identical_frames)
    port_text = normalize_text(reply.text)

    diverged = parity.identical_frames < parity.total_frames
    print(
        f"\n[{name}] audio identical for the first {parity.identical_frames} of "
        f"{parity.total_frames} frames ({len(ref_pieces)} reference text frames), "
        + (
            f"first divergence at frame {parity.identical_frames}"
            if diverged
            else "no divergence"
        )
        + f", max diff before divergence {parity.max_diff_before_divergence:.2e}"
    )
    print(f"[{name}] reference text up to divergence: {ref_text_prefix!r}")
    print(f"[{name}] reference text, full: {reference_text(ref_pieces)!r}")
    print(f"[{name}] port text: {port_text!r}")

    assert parity.identical_frames >= case.min_identical_frames, (
        f"{name}: only {parity.identical_frames} leading frames identical, "
        f"expected at least {case.min_identical_frames}"
    )
    assert (
        port_text.startswith(ref_text_prefix)
        if diverged
        else port_text == ref_text_prefix
    ), (
        f"{name}: text differs before the audio divergence at frame "
        f"{parity.identical_frames}"
    )


def test_port_is_deterministic(generate, port_greedy, assets_dir: Path):
    """The port's greedy output is identical across reruns, unlike the reference's."""
    case = CASES["assistant"]
    rerun = generate(
        assets_dir / case.input_wav,
        voice=case.voice,
        text_prompt=None,
        greedy=True,
    )
    first = port_greedy["assistant"]
    assert rerun.text == first.text
    assert np.array_equal(rerun.audio, first.audio)


def test_seed_reproducibility(generate, assets_dir: Path):
    """Same seed, same audio and text under sampling; a different seed differs."""
    case = CASES["service"]
    path = assets_dir / case.input_wav
    text_prompt = text_prompt_for(case, assets_dir)
    seeded = generate(
        path, voice=case.voice, text_prompt=text_prompt, greedy=False, seed=1234
    )
    again = generate(
        path, voice=case.voice, text_prompt=text_prompt, greedy=False, seed=1234
    )
    other = generate(
        path, voice=case.voice, text_prompt=text_prompt, greedy=False, seed=1235
    )

    assert again.text == seeded.text
    assert np.array_equal(again.audio, seeded.audio)
    assert not np.array_equal(other.audio, seeded.audio)
