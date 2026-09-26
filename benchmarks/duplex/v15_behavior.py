# SPDX-License-Identifier: Apache-2.0
"""Build, judge and aggregate paired-transcript behavior evidence for duplex v1.5 runs."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Annotated, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from benchmarks.duplex.v15_scoring import (
    CATEGORIES,
    DURATION_TOLERANCE_S,
    Category,
    canonical_hash,
    check_category,
    check_interval,
    index_records,
)

BEHAVIOR_RUBRIC_VERSION = "fdb-v15-behavior-rubric-v2"
BEHAVIOR_LABELS = ("C_RESPOND", "C_RESUME", "C_UNCERTAIN_HANDLING", "C_UNKNOWN")
TRANSCRIPT_ROLES = ("clean_input", "noisy_input", "clean_output", "noisy_output")
ALIGNED_SOURCES = ("provided_aligned", "asr_aligned")

BEHAVIOR_RUBRIC = f"""You grade one full-duplex spoken-dialogue sample (rubric {BEHAVIOR_RUBRIC_VERSION}).
The model heard the same user speech twice: clean_input without an overlapping event and
noisy_input with an overlapping event in the given window (category names the event type).
clean_output and noisy_output are complete word-timestamped transcripts of the model's replies.
First identify, in noisy_output, the first NEW semantic segment that begins at or after the
event onset (event.start_s). A new semantic segment starts a new utterance, idea or reply; it
may follow even a very short pause. Words that continue a sentence or idea already begun
before the onset are NOT a new segment, even if they are spoken after the onset.
Then label how that segment handled the event, using clean_output as the no-event reference:
- C_RESPOND: the new segment addresses the content of the overlapping speech.
- C_RESUME: the new segment continues or restarts the original reply without addressing it.
- C_UNCERTAIN_HANDLING: the new segment asks for clarification or signals it did not follow.
- C_UNKNOWN: no new segment, silence, or none of the above applies.
Return only a JSON object with exactly three keys: "label" (one of the four labels),
"evidence" (a short quote or paraphrase from noisy_output supporting the label) and
"first_new_segment" ({"text", "start_s", "end_s"} copied from noisy_output words, or null
when there is no new segment; only C_UNKNOWN may use null)."""

Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(min_length=1)]


class WordChunk(BaseModel):
    text: str
    timestamp: tuple[Seconds, Seconds]


class AlignedTranscript(BaseModel):
    text: str
    chunks: list[WordChunk]


class TranscriptEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transcript: AlignedTranscript
    timestamp_source: Literal["provided_aligned", "asr_aligned", "native_generated"]
    duration_s: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    source_sha256: Sha256


def build_behavior_input(
    *,
    sample_id: str,
    category: Category,
    event_start_s: float,
    event_end_s: float,
    transcripts: dict[str, JsonValue | None],
    metadata: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Build the hashed judge payload, or an unscorable record naming the gap."""
    check_category(category)
    unscorable = {"status": "unscorable", "sample_id": sample_id, "category": category}
    evidence = {}
    for role in TRANSCRIPT_ROLES:
        raw = transcripts.get(role)
        if raw is None:
            return {**unscorable, "reason": f"missing_transcript:{role}"}
        try:
            item = TranscriptEvidence.model_validate(raw)
        except ValidationError as error:
            return {
                **unscorable,
                "reason": f"malformed_transcript:{role}",
                "detail": str(error),
            }
        if item.timestamp_source not in ALIGNED_SOURCES:
            return {**unscorable, "reason": f"timestamps_not_asr_aligned:{role}"}
        starts = [chunk.timestamp[0] for chunk in item.transcript.chunks]
        if (
            any(
                start > end
                for start, end in (c.timestamp for c in item.transcript.chunks)
            )
            or starts != sorted(starts)
            or any(
                c.timestamp[1] > item.duration_s + DURATION_TOLERANCE_S
                for c in item.transcript.chunks
            )
        ):
            return {**unscorable, "reason": f"inconsistent_word_times:{role}"}
        evidence[role] = item
    check_interval(
        event_start_s, event_end_s, evidence["noisy_input"].duration_s, "event"
    )
    payload = {
        "rubric_version": BEHAVIOR_RUBRIC_VERSION,
        "sample_id": sample_id,
        "category": category,
        "event": {"start_s": event_start_s, "end_s": event_end_s},
        "metadata": metadata,
        "transcripts": {
            role: {
                "text": item.transcript.text,
                "words": [
                    {"text": c.text, "start_s": c.timestamp[0], "end_s": c.timestamp[1]}
                    for c in item.transcript.chunks
                ],
                "timestamp_source": item.timestamp_source,
                "duration_s": item.duration_s,
                "source_sha256": item.source_sha256,
            }
            for role, item in evidence.items()
        },
    }
    return {
        "status": "ready",
        "sample_id": sample_id,
        "category": category,
        "rubric_version": BEHAVIOR_RUBRIC_VERSION,
        "input_hash": canonical_hash(payload),
        "payload": payload,
    }


def label_error(label: object, evidence: object) -> str | None:
    if label not in BEHAVIOR_LABELS:
        return "invalid_label"
    elif not isinstance(evidence, str) or not evidence.strip():
        return "missing_evidence"
    else:
        return None


class NewSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: Identifier
    start_s: Seconds
    end_s: Seconds


def new_segment_error(
    segment: object, label: str, payload: dict[str, JsonValue]
) -> str | None:
    """Check a judge-chosen segment is null only for C_UNKNOWN and cites noisy words."""
    if segment is None:
        return None if label == "C_UNKNOWN" else "missing_first_new_segment"
    try:
        checked = NewSegment.model_validate(segment)
    except ValidationError:
        return "invalid_first_new_segment"
    words = payload["transcripts"]["noisy_output"]["words"]
    tokens = ["".join(word["text"].split()) for word in words]
    slices = [
        "".join(tokens[first : last + 1])
        for first, start_word in enumerate(words)
        if start_word["start_s"] == checked.start_s
        for last in range(first, len(words))
        if words[last]["end_s"] == checked.end_s
    ]
    # Note (wenyao): Only whitespace may differ, e.g. ASR tokens "yes" "," vs "yes,".
    cited = "".join(checked.text.split())
    if checked.start_s < payload["event"]["start_s"]:
        return "pre_onset_first_new_segment"
    elif not slices:
        return "segment_bounds_not_word_bounds"
    elif not cited or cited not in slices:
        return "segment_text_mismatch"
    else:
        return None


def call_openai_compatible_judge(
    behavior_input: dict[str, JsonValue],
    *,
    base_url: str,
    model: str,
    api_key_env: str,
    timeout_s: float,
    client: httpx.Client,
) -> dict[str, JsonValue]:
    """Ask one caller-chosen chat-completions judge for a strict JSON label."""
    assert behavior_input["status"] == "ready", behavior_input
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"judge API key environment variable {api_key_env} is unset")
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": BEHAVIOR_RUBRIC},
            {
                "role": "user",
                "content": json.dumps(behavior_input["payload"], sort_keys=True),
            },
        ],
    }
    response = client.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout_s,
    )
    response.raise_for_status()
    raw = response.json()
    label = evidence = segment = None
    payload = behavior_input["payload"]
    try:
        reply = json.loads(raw["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError):
        reason = "malformed_response"
    except json.JSONDecodeError:
        reason = "reply_not_json"
    else:
        if not isinstance(reply, dict) or set(reply) != {
            "label",
            "evidence",
            "first_new_segment",
        }:
            reason = "reply_schema_mismatch"
        else:
            reason = label_error(reply["label"], reply["evidence"])
            if reason is None:
                reason = new_segment_error(
                    reply["first_new_segment"], reply["label"], payload
                )
            if reason is None:
                label, evidence = reply["label"], reply["evidence"]
                segment = reply["first_new_segment"]
    return {
        "sample_id": behavior_input["sample_id"],
        "input_hash": behavior_input["input_hash"],
        "rubric_version": behavior_input["rubric_version"],
        "status": "valid" if reason is None else "invalid",
        "reason": reason,
        "label": label,
        "evidence": evidence,
        "first_new_segment": segment,
        "provenance": {
            "kind": "llm_judge",
            "base_url": base_url,
            "model": model,
            "response_model": raw.get("model") if isinstance(raw, dict) else None,
            "api_key_env": api_key_env,
            "prompt_hash": canonical_hash(body),
        },
        "raw_response": raw,
    }


class Annotator(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Identifier
    kind: Literal["human", "model"]
    note: str | None = None


class OfflineJudgementRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: Identifier
    input_hash: Sha256
    rubric_version: Identifier
    label: str
    evidence: str
    annotator: Annotator
    first_new_segment: JsonValue = None


def load_offline_judgements(path: str | Path) -> list[dict[str, JsonValue]]:
    """Load caller-supplied JSONL judgements; bad labels stay invalid, not coerced.

    Segment evidence is checked against the matched payload in summarize_behavior.
    """
    judgements = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = OfflineJudgementRow.model_validate_json(line)
        except ValidationError as error:
            raise ValueError(
                f"{path}:{line_number}: malformed judgement: {error}"
            ) from error
        reason = label_error(row.label, row.evidence)
        judgements.append(
            {
                "sample_id": row.sample_id,
                "input_hash": row.input_hash,
                "rubric_version": row.rubric_version,
                "status": "valid" if reason is None else "invalid",
                "reason": reason,
                "label": row.label if reason is None else None,
                "submitted_label": row.label,
                "evidence": row.evidence,
                "first_new_segment": row.first_new_segment,
                "provenance": {
                    "kind": "offline_import",
                    "annotator": row.annotator.model_dump(),
                    "source": f"{path}:{line_number}",
                },
            }
        )
    return judgements


def summarize_behavior(
    behavior_inputs: list[dict[str, JsonValue]],
    judgements: list[dict[str, JsonValue]],
    selected: dict[str, str],
) -> dict[str, JsonValue]:
    """Count labels and scored coverage per category without inventing pass/fail."""
    for category in selected.values():
        check_category(category)
    inputs = index_records(behavior_inputs, selected, "behavior input")
    by_sample = defaultdict(list)
    for judgement in judgements:
        by_sample[judgement["sample_id"]].append(judgement)
    samples = []
    for sample_id in sorted(selected):
        item = inputs.get(sample_id)
        candidates = by_sample.get(sample_id, [])
        matching = [
            judgement
            for judgement in candidates
            if item is not None
            and item["status"] == "ready"
            and judgement["input_hash"] == item["input_hash"]
            and judgement["rubric_version"] == item["rubric_version"]
        ]
        # Note (wenyao): Offline rows only see the payload here, so recheck both sources.
        errors = [
            (
                judgement["reason"]
                if judgement["status"] != "valid"
                else label_error(judgement["label"], judgement["evidence"])
                or new_segment_error(
                    judgement["first_new_segment"], judgement["label"], item["payload"]
                )
            )
            for judgement in matching
        ]
        labels = {
            judgement["label"]
            for judgement, error in zip(matching, errors)
            if error is None
        }
        label = reason = None
        if item is None:
            reason = "missing_behavior_input"
        elif item["status"] != "ready":
            reason = item["reason"]
        elif not candidates:
            reason = "missing_judgement"
        elif not matching:
            reason = "judgement_input_mismatch"
        elif not labels:
            reason = "invalid_judgement"
        elif len(labels) > 1:
            reason = "conflicting_judgements"
        else:
            label = labels.pop()
        samples.append(
            {
                "sample_id": sample_id,
                "category": selected[sample_id],
                "label": label,
                "unscored_reason": reason,
                "judgement_errors": sorted({e for e in errors if e is not None}),
            }
        )
    categories = {}
    for category in CATEGORIES:
        rows = [row for row in samples if row["category"] == category]
        scored = [row for row in rows if row["label"] is not None]
        label_counts = Counter(row["label"] for row in scored)
        categories[category] = {
            "selected": len(rows),
            "scored": len(scored),
            "scored_coverage": len(scored) / len(rows) if rows else None,
            "label_counts": {label: label_counts[label] for label in BEHAVIOR_LABELS},
            "unscored_reasons": dict(
                Counter(row["unscored_reason"] for row in rows if row["label"] is None)
            ),
            "invalid_judgement_reasons": dict(
                Counter(
                    error
                    for row in rows
                    if row["unscored_reason"] == "invalid_judgement"
                    for error in row["judgement_errors"]
                )
            ),
        }
    return {
        "rubric_version": BEHAVIOR_RUBRIC_VERSION,
        "categories": categories,
        "orphan_judgements": sum(
            len(items) for key, items in by_sample.items() if key not in selected
        ),
        "samples": samples,
    }
