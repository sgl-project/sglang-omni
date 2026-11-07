"""Contract between the transcription endpoint and ASR request builders.

The endpoint emits a fixed set of params keys. Each ASR model must either
consume a key or declare here why it does not. Two silent failures motivated
this: qwen3_asr dropped ``prompt`` entirely (#1807), and fun_asr listened for
a ``hotwords`` key that nothing emits (#1874). In both cases the endpoint
returned 200 and the caller's vocabulary did nothing.

The checks read source text rather than importing model code, so they run on
any platform and add no coupling to model internals.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3] / "sglang_omni"
_SERVE = _ROOT / "serve" / "speech_to_text.py"
_MODELS = _ROOT / "models"

# Keys build_speech_to_text_generate_request can place into params.
ENDPOINT_KEYS = {"task", "language", "prompt", "detect_language", "segment_timestamps"}

ASR_MODELS = ("qwen3_asr", "whisper_asr", "fun_asr", "arkasr", "moss_transcribe_diarize")

# A model that does not consume an endpoint key states its reason here.
UNSUPPORTED: dict[str, dict[str, str]] = {
    "qwen3_asr": {
        "task": "single-task transcription model",
        "detect_language": "auto-detection is built into the prompt template when language is omitted",
        "segment_timestamps": "no segment-timestamp capability; the endpoint gates on the adapter before sending",
    },
    "whisper_asr": {},
    "fun_asr": {
        "task": "single-task transcription model",
        "detect_language": "language handling is explicit via the language param",
        "segment_timestamps": "no segment-timestamp capability",
    },
    "arkasr": {
        "task": "single-task transcription model",
        "detect_language": "no detection flag; language param only",
        "segment_timestamps": "no segment-timestamp capability",
        "prompt": "no biasing mechanism in this integration today",
    },
    "moss_transcribe_diarize": {
        "task": "single-task transcription model",
        "detect_language": "no detection flag; language param only",
        "segment_timestamps": "diarization output carries its own structure",
    },
}

# Keys a model should consume but does not yet; each entry names the fix.
# The staleness check below fails once the fix lands, so the entry is removed
# rather than rotting.
KNOWN_GAPS: dict[tuple[str, str], str] = {
    ("qwen3_asr", "prompt"): "wired in #1807",
}

_KEY_PATTERN = re.compile(r'params(?:\.get\(|\[)\s*"([a-z_]+)"')


def _consumed_keys(model: str) -> set[str]:
    source = (_MODELS / model / "request_builders.py").read_text()
    return set(_KEY_PATTERN.findall(source))


def test_endpoint_key_declaration_matches_the_serve_source() -> None:
    source = _SERVE.read_text()
    # Keys assigned one at a time, and keys in the params dict literal itself.
    emitted = set(re.findall(r'params\["([a-z_]+)"\]\s*=', source))
    for literal in re.findall(r"params(?::[^=\n]+)?=\s*\{([^}]*)\}", source):
        emitted.update(re.findall(r'"([a-z_]+)":', literal))
    assert emitted == ENDPOINT_KEYS, (
        "speech_to_text.py emits a different key set than this contract "
        f"declares; update ENDPOINT_KEYS and the per-model tables. New: "
        f"{sorted(emitted - ENDPOINT_KEYS)}, gone: {sorted(ENDPOINT_KEYS - emitted)}"
    )


def test_every_asr_model_accounts_for_every_endpoint_key() -> None:
    problems = []
    for model in ASR_MODELS:
        consumed = _consumed_keys(model)
        for key in sorted(ENDPOINT_KEYS):
            if key in consumed:
                continue
            if key in UNSUPPORTED.get(model, {}):
                continue
            if (model, key) in KNOWN_GAPS:
                continue
            problems.append(f"{model} neither consumes nor declares {key!r}")
    assert not problems, (
        "The endpoint sends these keys and the model silently ignores them "
        "(the two known instances of this returned 200 and did nothing): "
        + "; ".join(problems)
    )


def test_known_gaps_are_still_gaps() -> None:
    stale = [
        f"{model}:{key} ({fix})"
        for (model, key), fix in KNOWN_GAPS.items()
        if key in _consumed_keys(model)
    ]
    assert not stale, (
        "These fixes have landed; move the entries out of KNOWN_GAPS: "
        + ", ".join(stale)
    )
