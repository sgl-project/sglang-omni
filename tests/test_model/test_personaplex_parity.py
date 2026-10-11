# SPDX-License-Identifier: Apache-2.0
"""Opt-in greedy prefix and repeatability checks against NVIDIA/personaplex.

Needs a separate reference environment; set it up as "Reference parity" in
docs/cookbook/personaplex.md describes. Skips unless PERSONAPLEX_REFERENCE_SOURCE
and PERSONAPLEX_REFERENCE_PYTHON are set.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import soundfile
import torch

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateRequest, SamplingParams
from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.personaplex.architecture import (
    MIMI_WEIGHTS_GLOB,
    MOSHI_WEIGHTS_NAME,
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
)
from sglang_omni.models.personaplex.config import PersonaPlexPipelineConfig
from sglang_omni.models.personaplex.prompts import (
    DEFAULT_TEXT_PROMPT,
    TEXT_TOKENIZER_NAME,
    resolve_voice_path,
)
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.proto.request import EXPLICIT_GENERATION_PARAMS_KEY
from sglang_omni.utils.checkpoint import resolve_checkpoint

pytestmark = pytest.mark.accelerator

DEFAULT_CHECKPOINT = "nvidia/personaplex-7b-v1"
# The reference README's seed; irrelevant under --greedy, kept so the command matches.
REFERENCE_SEED = 42424242
# The reference maps BOS and EOS through the tokenizer, so they arrive as <s> and </s>.
REFERENCE_TEXT_MARKERS = frozenset({"EPAD", "BOS", "EOS", "PAD", "<s>", "</s>"})
ATOL = 1e-4  # a few int16 steps, for rounding between the two codec paths


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


@dataclass(frozen=True, kw_only=True)
class ParityInputs:
    assets: Path
    checkpoint: Path
    reference_python: str
    reference_repo: str
    stage_args: list[str]


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
    else:
        pass
    return np.ascontiguousarray(data[:, 0])


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def reference_text(pieces: list[str], frames: int | None = None) -> str:
    """Reply text from the reference's per-frame token pieces, markers dropped."""
    if frames is not None:
        pieces = pieces[:frames]
    else:
        pass
    return normalize_text("".join(p for p in pieces if p not in REFERENCE_TEXT_MARKERS))


def compare_frames(port: np.ndarray, reference: np.ndarray, atol: float) -> FrameParity:
    if port.shape != reference.shape:
        raise ValueError(
            f"Audio sample counts differ: {port.shape} versus {reference.shape}"
        )
    else:
        pass
    if not port.size:
        raise ValueError("Cannot compare empty audio")
    else:
        pass
    if not np.isfinite(port).all() or not np.isfinite(reference).all():
        raise ValueError("Audio contains non-finite samples")
    else:
        pass
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
def parity_inputs() -> ParityInputs:
    source = os.environ.get("PERSONAPLEX_REFERENCE_SOURCE")
    python = os.environ.get("PERSONAPLEX_REFERENCE_PYTHON")
    if not source or not python:
        pytest.skip(
            "Set PERSONAPLEX_REFERENCE_SOURCE and PERSONAPLEX_REFERENCE_PYTHON "
            "for reference parity"
        )
    else:
        pass
    if not torch.cuda.is_available():
        pytest.skip("PersonaPlex reference parity requires CUDA")
    else:
        pass
    checkpoint = os.environ.get("PERSONAPLEX_PARITY_CHECKPOINT", DEFAULT_CHECKPOINT)
    return ParityInputs(
        assets=Path(source).expanduser().resolve() / "assets" / "test",
        checkpoint=Path(resolve_checkpoint(checkpoint)),
        reference_python=python,
        reference_repo=os.environ.get("PERSONAPLEX_REFERENCE_REPO", DEFAULT_CHECKPOINT),
        stage_args=shlex.split(os.environ.get("PERSONAPLEX_STAGE_ARGS", "")),
    )


@pytest.fixture(scope="module")
def references(
    parity_inputs: ParityInputs, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Reference outputs, generated before the port takes the GPU."""
    assets = parity_inputs.assets
    checkpoint = parity_inputs.checkpoint
    source = assets.parents[1]
    root = tmp_path_factory.mktemp("personaplex_reference")
    print(f"\nreference outputs and logs: {root}")
    (mimi_weight,) = checkpoint.glob(MIMI_WEIGHTS_GLOB)
    outputs = {}
    for name, case in CASES.items():
        output = root / name
        voice = resolve_voice_path(checkpoint, case.voice).resolve()
        prompt = text_prompt_for(case, assets) or DEFAULT_TEXT_PROMPT
        command = [
            parity_inputs.reference_python,
            "-m",
            "moshi.offline",
            "--hf-repo",
            parity_inputs.reference_repo,
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
            str(assets / case.input_wav),
            "--greedy",
            "--seed",
            str(REFERENCE_SEED),
            "--output-wav",
            str(output / "output.wav"),
            "--output-text",
            str(output / "output.json"),
        ]
        output.mkdir(parents=True, exist_ok=True)
        with (output / "reference.log").open("w") as log:
            subprocess.run(
                command,
                cwd=source,
                env=dict(os.environ, PYTHONPATH=str(source / "moshi")),
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        outputs[name] = (
            read_wav(output / "output.wav"),
            json.loads((output / "output.json").read_text()),
        )
    return outputs


async def generate_replies(parity_inputs: ParityInputs) -> dict[str, Reply]:
    """Every reply the tests compare, from one pipeline started once."""
    assets = parity_inputs.assets
    config = PersonaPlexPipelineConfig(model_path=str(parity_inputs.checkpoint))
    if parity_inputs.stage_args:
        manager = ConfigManager(config)
        config = manager.merge_config(
            manager.parse_extra_args(parity_inputs.stage_args)
        )
    else:
        pass
    runner = MultiProcessPipelineRunner(config)
    await runner.start(
        timeout=float(os.environ.get("SGLANG_OMNI_STARTUP_TIMEOUT", "900"))
    )
    try:
        client = Client(runner.coordinator)
        request_number = 0

        async def generate_reply(
            name: str, *, greedy: bool, seed: int | None = None
        ) -> Reply:
            nonlocal request_number
            request_number += 1
            case = CASES[name]
            extra = {"voice": case.voice}
            prompt = text_prompt_for(case, assets)
            if prompt is not None:
                extra["text_prompt"] = prompt
            else:
                pass
            if greedy:
                extra["audio_temperature"] = 0.0
            else:
                pass
            if seed is not None:
                extra["seed"] = seed
            else:
                pass
            request = GenerateRequest(
                model=config.name,
                prompt={"audio_path": str(assets / case.input_wav)},
                sampling=(
                    SamplingParams(temperature=0.0) if greedy else SamplingParams()
                ),
                extra_params=extra,
                metadata={
                    EXPLICIT_GENERATION_PARAMS_KEY: ["temperature"] if greedy else []
                },
                output_modalities=["text", "audio"],
                stream=False,
            )
            result = await client.completion(
                request, request_id=f"parity-{request_number}", audio_format="pcm"
            )
            assert result.audio is not None, "PersonaPlex returned no audio"
            blob = result.audio.data
            pcm = base64.b64decode(blob) if isinstance(blob, str) else blob
            return Reply(
                text=result.text or "",
                audio=np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0,
            )

        replies = {name: await generate_reply(name, greedy=True) for name in CASES}
        replies["assistant_repeat"] = await generate_reply("assistant", greedy=True)
        replies["seed_1234"] = await generate_reply("service", greedy=False, seed=1234)
        replies["seed_1234_repeat"] = await generate_reply(
            "service", greedy=False, seed=1234
        )
        replies["seed_1235"] = await generate_reply("service", greedy=False, seed=1235)
        return replies
    finally:
        await runner.stop()


@pytest.fixture(scope="module")
def replies(parity_inputs: ParityInputs) -> dict[str, Reply]:
    return asyncio.run(generate_replies(parity_inputs))


@pytest.mark.parametrize("name", list(CASES))
def test_greedy_prefix_matches_reference(
    name: str,
    parity_inputs: ParityInputs,
    references: dict[str, tuple[np.ndarray, list[str]]],
    replies: dict[str, Reply],
) -> None:
    """The leading frames and their text match the reference; not full-output parity."""
    case = CASES[name]
    reply = replies[name]
    ref_audio, ref_pieces = references[name]

    expected_samples = read_wav(parity_inputs.assets / case.input_wav).size
    assert reply.audio.size == ref_audio.size == expected_samples, (
        f"{name}: expected {expected_samples} samples, got "
        f"port={reply.audio.size}, reference={ref_audio.size}"
    )
    expected_frames = (expected_samples + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME
    assert (
        len(ref_pieces) == expected_frames
    ), f"{name}: reference wrote {len(ref_pieces)} text frames, expected {expected_frames}"
    parity = compare_frames(reply.audio, ref_audio, ATOL)
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


def test_greedy_reply_repeats(replies: dict[str, Reply]) -> None:
    first, repeat = replies["assistant"], replies["assistant_repeat"]
    assert first.text == repeat.text and np.array_equal(
        first.audio, repeat.audio
    ), "greedy replies differ between two runs"


def test_same_seed_repeats(replies: dict[str, Reply]) -> None:
    seeded, repeat = replies["seed_1234"], replies["seed_1234_repeat"]
    assert seeded.text == repeat.text and np.array_equal(
        seeded.audio, repeat.audio
    ), "same-seed replies differ"


def test_different_seed_differs(replies: dict[str, Reply]) -> None:
    assert not np.array_equal(
        replies["seed_1235"].audio, replies["seed_1234"].audio
    ), "different seeds produced identical audio"
