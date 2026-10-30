# SPDX-License-Identifier: Apache-2.0
"""Chatterbox-Turbo golden check for check_golden.py: seeded speech requests through
chatterbox_synthesize, whose text ids, cloned voice tokens, speech tokens and 16-bit
audio a chip with golden outputs must give exactly. Another chip's GPU arithmetic may
sample other tokens, so there only the text ids, which the tokenizer alone sets, must
match.

A golden file of kind chatterbox names its cases (text, seed, max_new_tokens, and a
reference recording under the repository root to clone, or none for the built-in
voice). A case with voiced_token_count skips T3 and voices that many fixed speech
tokens, so the flow and vocoder run at the server's token cap. Each chip's outputs
keep the speech tokens one second (25 tokens) per string, so a divergence reads as the
second it starts at, and the audio's SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TOKENS_PER_SECOND = 25
SPEECH_TOKEN_COUNT = 6561


def joined(values: list[int]) -> str:
    return " ".join(str(value) for value in values)


def synthesize(runtime_bin: Path, model_directory: Path, case: dict) -> dict:
    reference = case.get("reference_audio")
    reference_flags = (
        ["--reference-audio", str(REPOSITORY_ROOT / reference)] if reference else []
    )
    voiced_token_count = case.get("voiced_token_count")
    # Fixed tokens spread over the vocabulary.
    voiced_flags = (
        [
            "--speech-tokens",
            joined(
                [
                    (index * 1103 + 7) % SPEECH_TOKEN_COUNT
                    for index in range(voiced_token_count)
                ]
            ),
        ]
        if voiced_token_count
        else []
    )
    with tempfile.TemporaryDirectory(prefix="chatterbox-golden-") as directory:
        tokens_path = Path(directory) / "tokens.json"
        audio_path = Path(directory) / "speech.wav"
        completed = subprocess.run(
            [
                str(runtime_bin / "chatterbox_synthesize"),
                "--model-path",
                str(model_directory),
                "--text",
                case["text"],
                "--seed",
                str(case["seed"]),
                "--max-new-tokens",
                str(case["max_new_tokens"]),
                "--tokens-output",
                str(tokens_path),
                "--output",
                str(audio_path),
            ]
            + reference_flags
            + voiced_flags,  # fmt: skip
            check=True,
            capture_output=True,
            text=True,
        )
        tokens = json.loads(tokens_path.read_text())
        audio_sha256 = hashlib.sha256(audio_path.read_bytes()).hexdigest()
    speech_tokens = tokens["speech_tokens"]
    return {
        "text_ids": joined(tokens["text_ids"]),
        "voice_tokens": joined(tokens["voice_tokens"]),
        "finish_reason": json.loads(completed.stdout.splitlines()[-1])["finish_reason"],
        "speech_tokens": [
            joined(speech_tokens[start : start + TOKENS_PER_SECOND])
            for start in range(0, len(speech_tokens), TOKENS_PER_SECOND)
        ],
        "audio_sha256": audio_sha256,
    }


def synthesize_cases(
    runtime_bin: Path, model_directory: Path, cases: dict[str, dict]
) -> dict[str, dict]:
    return {
        case_id: synthesize(runtime_bin, model_directory, case)
        for case_id, case in cases.items()
    }


def difference(expected: dict, actual: dict | None) -> str | None:
    """Where actual first departs from expected, or None when they match."""
    if actual is None:
        return "was not synthesized"
    elif actual["text_ids"] != expected["text_ids"]:
        return "has other text ids"
    elif actual["voice_tokens"] != expected["voice_tokens"]:
        return "has other cloned voice tokens"
    else:
        pass
    for second, (want, got) in enumerate(
        zip(expected["speech_tokens"], actual["speech_tokens"])
    ):
        if want != got:
            return f"differs from second {second}"
        else:
            pass
    if len(actual["speech_tokens"]) != len(expected["speech_tokens"]):
        return (
            f"has {len(actual['speech_tokens'])} seconds of tokens, "
            f"not {len(expected['speech_tokens'])}"
        )
    elif actual["finish_reason"] != expected["finish_reason"]:
        return (
            f"finished with {actual['finish_reason']}, not {expected['finish_reason']}"
        )
    elif actual["audio_sha256"] != expected["audio_sha256"]:
        return "has other audio from the same tokens"
    else:
        return None


def check(
    golden: dict, device: str, results: dict[str, dict]
) -> tuple[list[str], list[str]]:
    """Returns (report lines, failures)."""
    devices = golden["devices"]
    if not devices:
        return [f"### {golden['model']} on {device}"], [
            "the golden file has no outputs"
        ]
    elif device in devices:
        reference_device = device
    else:
        reference_device = next(iter(devices))
    reference = devices[reference_device]["clips"]
    differences = {
        case_id: difference(outputs, results.get(case_id))
        for case_id, outputs in reference.items()
    }
    identical = sum(found is None for found in differences.values())
    missing = [
        f"`{case_id}` has no golden output"
        for case_id in golden["cases"]
        if case_id not in reference
    ]
    if not reference:
        failures = ["the golden file has no cases"]
    elif device in devices:
        failures = missing + [
            f"`{case_id}` {found}"
            for case_id, found in differences.items()
            if found is not None
        ]
    else:
        failures = missing + [
            f"`{case_id}` has other text ids than {reference_device}'s"
            for case_id, outputs in reference.items()
            if results.get(case_id, {}).get("text_ids") != outputs["text_ids"]
        ]
    mode = (
        "exact"
        if device in devices
        else "no golden outputs for this chip; text ids exact"
    )
    lines = [
        f"### {golden['model']} on {device}",
        "",
        f"Golden cases identical to {reference_device}: "
        f"{identical}/{len(reference)} ({mode})",
    ]
    return lines, failures
