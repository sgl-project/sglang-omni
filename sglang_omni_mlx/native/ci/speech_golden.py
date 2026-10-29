# SPDX-License-Identifier: Apache-2.0
"""The check every speech golden shares (voxtral_golden.py,
fishaudio_drama3_golden.py): a chip with golden outputs must give each seeded case's
prompt ids, audio codes, finish reason and, when the golden holds one, decoded-audio
hash exactly; another chip, whose GPU arithmetic may sample other codes, only the
prompt ids. Each chip's outputs keep one string of codes per audio frame, so a
divergence reads as the frame it starts at.
"""

from __future__ import annotations

from typing import NotRequired, Required, TypedDict


class SpeechReference(TypedDict, total=False):
    """A voice to clone: its transcript, and a WAV in this directory or one
    string of codes per codebook."""

    text: Required[str]
    audio: str
    codes: list[str]


class SpeechCase(TypedDict, total=False):
    """A seeded request; voice for Voxtral, sampling and a reference for Drama 3."""

    text: Required[str]
    seed: Required[int]
    max_new_tokens: Required[int]
    voice: str
    temperature: float
    top_p: float
    top_k: int
    reference: SpeechReference


class SpeechOutputs(TypedDict):
    prompt_ids: str
    finish_reason: str
    codes: list[str]
    pcm_sha256: NotRequired[str]


class DeviceOutputs(TypedDict):
    metrics: dict[str, float]
    clips: dict[str, SpeechOutputs]


class SpeechGolden(TypedDict):
    model: str
    kind: str
    tool: NotRequired[str]
    cases: dict[str, SpeechCase]
    devices: dict[str, DeviceOutputs]


def joined(values: list[int]) -> str:
    return " ".join(str(value) for value in values)


def difference(expected: SpeechOutputs, actual: SpeechOutputs | None) -> str | None:
    """Where actual first departs from expected, or None when they match."""
    if actual is None:
        return "was not synthesized"
    elif actual["prompt_ids"] != expected["prompt_ids"]:
        return "has other prompt ids"
    else:
        pass
    for frame, (want, got) in enumerate(zip(expected["codes"], actual["codes"])):
        if want != got:
            return f"differs from frame {frame}"
        else:
            pass
    if len(actual["codes"]) != len(expected["codes"]):
        return f"has frame count {len(actual['codes'])}, not {len(expected['codes'])}"
    elif actual["finish_reason"] != expected["finish_reason"]:
        return (
            f"finished with {actual['finish_reason']}, not {expected['finish_reason']}"
        )
    elif (
        "pcm_sha256" in expected and actual.get("pcm_sha256") != expected["pcm_sha256"]
    ):
        return "decodes to other audio"
    else:
        return None


def check(
    golden: SpeechGolden, device: str, results: dict[str, SpeechOutputs]
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
    # Every chip checks only cases with a reference, so a case without one fails.
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
            f"`{case_id}` has other prompt ids than {reference_device}'s"
            for case_id, outputs in reference.items()
            if results.get(case_id, {}).get("prompt_ids") != outputs["prompt_ids"]
        ]
    mode = (
        "exact"
        if device in devices
        else "no golden outputs for this chip; prompt ids exact"
    )
    lines = [
        f"### {golden['model']} on {device}",
        "",
        f"Golden cases identical to {reference_device}: "
        f"{identical}/{len(reference)} ({mode})",
    ]
    return lines, failures
