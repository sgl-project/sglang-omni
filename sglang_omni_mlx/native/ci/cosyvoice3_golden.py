# SPDX-License-Identifier: Apache-2.0
"""Fun-CosyVoice3 golden check for check_golden.py: seeded speech requests through
cosyvoice3_synthesize, whose text ids and speech tokens a chip with golden outputs
must give exactly; speech_golden.check compares them. Another chip's GPU
arithmetic may sample other tokens, so there only the text ids, which the
tokenizer alone sets, must match.

A golden file of kind cosyvoice3 names its cases (text, optional instructions,
seed and max_new_tokens; a cloning case also names a corpus clip as reference_clip,
with its transcript as reference_text unless it is cross-lingual); each chip's
outputs keep one string per speech token, so a divergence reads as the token it
starts at. A cloning case's first code is the reference's speech tokens, which
the GPU computes, so only a chip with golden outputs checks them.

A vocode case (vocode_tokens_from, and optionally reference_clip) vocodes the
speech tokens a sampling case gave on the chip with golden outputs, so every chip
runs the flow and HiFT on the same tokens; its mel's mean per band and per frame
must stay within MEL_TOLERANCE of that chip's, and on that chip its 16-bit PCM
must hash the same.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import speech_golden

# Note (Jiaxin Deng): the two MLX wheel builds (macOS 15 and 26 kernels) on one
# Apple M5 give vocode means at most 0.021 apart in log-mel, a stand-in for
# another chip's kernels.
MEL_TOLERANCE = 0.05


def reference_device(golden: dict, device: str) -> str | None:
    devices = golden["devices"]
    return device if device in devices else next(iter(devices), None)


def golden_tokens(golden: dict, device: str, case_id: str) -> list[int] | None:
    """The speech tokens a case gave on the chip whose outputs are compared."""
    reference = reference_device(golden, device)
    if reference is None:
        return None
    else:
        pass
    codes = golden["devices"][reference]["clips"][case_id]["codes"]
    return [int(code) for code in codes if not code.startswith("reference ")]


def synthesize(
    runtime_bin: Path, data_root: Path, model: str, case: dict, tokens: list[int] | None
) -> dict:
    flags = ["--instructions", case["instructions"]] if "instructions" in case else []
    if "reference_clip" in case:
        clip = data_root / "corpus" / "v1" / "clips" / f"{case['reference_clip']}.wav"
        flags += ["--reference-audio", str(clip)]
    else:
        pass
    if "reference_text" in case:
        flags += ["--reference-text", case["reference_text"]]
    else:
        pass
    if "max_new_tokens" in case:
        flags += ["--max-new-tokens", str(case["max_new_tokens"])]
    else:
        pass
    with tempfile.TemporaryDirectory(prefix="cosyvoice3-golden-") as directory:
        tokens_path = Path(directory) / "tokens.json"
        mel_path = Path(directory) / "mel.npy"
        wav_path = Path(directory) / "speech.wav"
        if tokens is not None:
            given_path = Path(directory) / "given.json"
            given_path.write_text(json.dumps({"speech_tokens": tokens}))
            flags += ["--tokens-input", str(given_path), "--mel-output", str(mel_path)]
            flags += ["--output", str(wav_path)]
        else:
            pass
        completed = subprocess.run(
            [
                str(runtime_bin / "cosyvoice3_synthesize"),
                "--model-path",
                str(data_root / "models" / model.replace("/", "_")),
                "--text",
                case["text"],
                "--seed",
                str(case.get("seed", 0)),
                "--tokens-output",
                str(tokens_path),
                *flags,
            ],  # fmt: skip
            check=True,
            capture_output=True,
            text=True,
        )
        outputs = json.loads(tokens_path.read_text())
        mel = np.load(mel_path).astype(np.float32)[0] if tokens is not None else None
        pcm_sha256 = (
            hashlib.sha256(wav_path.read_bytes()).hexdigest()
            if tokens is not None
            else None
        )
    reference = outputs["reference"]
    reference_codes = (
        ["reference " + " ".join(str(value) for value in reference["speech_tokens"])]
        if reference
        else []
    )
    result = {
        "prompt_ids": " ".join(str(value) for value in outputs["text_ids"]),
        "finish_reason": json.loads(completed.stdout.splitlines()[-1])["finish_reason"],
        "codes": reference_codes
        + ([] if tokens is not None else [str(t) for t in outputs["speech_tokens"]]),
    }
    if mel is not None:
        # mel is [80 bands, frames].
        result["mel_band_means"] = [round(float(v), 4) for v in mel.mean(axis=1)]
        result["mel_frame_means"] = [round(float(v), 4) for v in mel.mean(axis=0)]
        result["pcm_sha256"] = pcm_sha256
    else:
        pass
    return result


def synthesize_cases(
    runtime_bin: Path, data_root: Path, golden: dict, device: str
) -> dict[str, dict]:
    results = {}
    for case_id, case in golden["cases"].items():
        source = case.get("vocode_tokens_from")
        tokens = (
            None
            if source is None
            else golden_tokens(golden, device, source)
            or [int(code) for code in results[source]["codes"] if " " not in code]
        )
        results[case_id] = synthesize(
            runtime_bin, data_root, golden["model"], case, tokens
        )
    return results


def check(
    golden: dict, device: str, results: dict[str, dict]
) -> tuple[list[str], list[str]]:
    """speech_golden.check, plus every chip's vocode cases within tolerance."""
    lines, failures = speech_golden.check(golden, device, results)
    reference = reference_device(golden, device)
    for case_id, case in golden["cases"].items():
        if "vocode_tokens_from" not in case or reference is None:
            continue
        else:
            pass
        expected = golden["devices"][reference]["clips"].get(case_id)
        actual = results.get(case_id)
        if expected is None or actual is None:
            continue
        else:
            pass
        for field in ("mel_band_means", "mel_frame_means"):
            want, got = np.array(expected[field]), np.array(actual[field])
            if not np.isfinite(want).all() or not np.isfinite(got).all():
                failures.append(f"`{case_id}` {field} contain nonfinite values")
                continue
            else:
                pass
            gap = float(np.abs(want - got).max()) if want.shape == got.shape else None
            if gap is None or gap > MEL_TOLERANCE:
                failures.append(
                    f"`{case_id}` {field} differ from {reference}'s by "
                    f"{'a shape' if gap is None else round(gap, 4)}"
                )
            else:
                lines.append(f"`{case_id}` {field} within {gap:.4f} of {reference}")
    return lines, failures
