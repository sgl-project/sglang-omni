# SPDX-License-Identifier: Apache-2.0
"""Greedy prefix comparison and repeatability checks; see personaplex.md.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
from sglang_omni.utils.logging import configure_dependency_loggers

DEFAULT_CHECKPOINT = "nvidia/personaplex-7b-v1"
# The reference README's seed; irrelevant under --greedy, kept so the command matches.
REFERENCE_SEED = 42424242
# The reference maps BOS and EOS through the tokenizer, so they arrive as <s> and </s>.
REFERENCE_TEXT_MARKERS = frozenset({"EPAD", "BOS", "EOS", "PAD", "<s>", "</s>"})
DEFAULT_ATOL = 1e-4  # a few int16 steps, for rounding between the two codec paths


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)
    else:
        pass


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


def reference_outputs(
    assets_dir: Path, checkpoint: Path, *, python: str, root: Path, repo: str
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Generate reference outputs before allocating the port on the GPU."""
    source = assets_dir.parents[1]
    (mimi_weight,) = checkpoint.glob(MIMI_WEIGHTS_GLOB)
    outputs = {}
    for name, case in CASES.items():
        output = root / name
        voice = resolve_voice_path(checkpoint, case.voice).resolve()
        prompt = text_prompt_for(case, assets_dir) or DEFAULT_TEXT_PROMPT
        command = [
            python,
            "-m",
            "moshi.offline",
            "--hf-repo",
            repo,
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


def compare_greedy(
    name: str,
    reply: Reply,
    reference: tuple[np.ndarray, list[str]],
    assets_dir: Path,
    *,
    atol: float,
) -> None:
    case = CASES[name]
    ref_audio, ref_pieces = reference

    expected_samples = read_wav(assets_dir / case.input_wav).size
    check(
        reply.audio.size == ref_audio.size == expected_samples,
        f"{name}: expected {expected_samples} samples, got "
        f"port={reply.audio.size}, reference={ref_audio.size}",
    )
    expected_frames = (expected_samples + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME
    check(
        len(ref_pieces) == expected_frames,
        f"{name}: reference wrote {len(ref_pieces)} text frames, expected {expected_frames}",
    )
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

    check(
        parity.identical_frames >= case.min_identical_frames,
        f"{name}: only {parity.identical_frames} leading frames identical, "
        f"expected at least {case.min_identical_frames}",
    )
    check(
        (
            port_text.startswith(ref_text_prefix)
            if diverged
            else port_text == ref_text_prefix
        ),
        f"{name}: text differs before the audio divergence at frame "
        f"{parity.identical_frames}",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PersonaPlex greedy comparison.")
    p.add_argument(
        "--reference-source",
        required=True,
        help="clean NVIDIA/personaplex checkout; its assets/test recordings are used",
    )
    p.add_argument(
        "--reference-python",
        required=True,
        help="interpreter of the reference environment (its torch pin differs)",
    )
    p.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="PersonaPlex checkpoint: local directory or resolvable model id",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="directory that receives the reference outputs and logs",
    )
    p.add_argument(
        "--reference-repo",
        default=DEFAULT_CHECKPOINT,
        help="repo the reference CLI uses for its config lookup only",
    )
    p.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    p.add_argument(
        "--stage-args",
        default="",
        help="pipeline overrides, e.g. '--lm.engine.mem_fraction_static 0.5'",
    )
    return p.parse_args()


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    configure_dependency_loggers()
    args = parse_args()
    assets = Path(args.reference_source).expanduser().resolve() / "assets" / "test"
    checkpoint = Path(resolve_checkpoint(args.checkpoint))
    if not torch.cuda.is_available():
        raise RuntimeError("PersonaPlex parity requires CUDA")
    else:
        pass
    references = reference_outputs(
        assets,
        checkpoint,
        python=args.reference_python,
        root=Path(args.output_dir).expanduser().resolve(),
        repo=args.reference_repo,
    )
    config = PersonaPlexPipelineConfig(model_path=str(checkpoint))
    overrides = shlex.split(args.stage_args)
    if overrides:
        manager = ConfigManager(config)
        config = manager.merge_config(manager.parse_extra_args(overrides))
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
            check(result.audio is not None, "PersonaPlex returned no audio")
            blob = result.audio.data
            pcm = base64.b64decode(blob) if isinstance(blob, str) else blob
            return Reply(
                text=result.text or "",
                audio=np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0,
            )

        replies = {}
        for name in CASES:
            replies[name] = await generate_reply(name, greedy=True)
            compare_greedy(
                name, replies[name], references[name], assets, atol=args.atol
            )
        repeated = await generate_reply("assistant", greedy=True)
        check(
            repeated.text == replies["assistant"].text
            and np.array_equal(repeated.audio, replies["assistant"].audio),
            "greedy replies differ between two runs",
        )
        seeded = await generate_reply("service", greedy=False, seed=1234)
        repeated = await generate_reply("service", greedy=False, seed=1234)
        other = await generate_reply("service", greedy=False, seed=1235)
        check(
            repeated.text == seeded.text
            and np.array_equal(repeated.audio, seeded.audio),
            "same-seed replies differ",
        )
        check(
            not np.array_equal(other.audio, seeded.audio),
            "different seeds produced identical audio",
        )
        print(
            "Greedy prefix checks and request reproducibility checks passed; this is not full-output parity."
        )
    finally:
        await runner.stop()


if __name__ == "__main__":
    asyncio.run(main())
