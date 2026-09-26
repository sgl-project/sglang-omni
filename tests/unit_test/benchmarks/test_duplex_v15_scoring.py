# SPDX-License-Identifier: Apache-2.0
import json

import httpx
import numpy as np
import pytest
import soundfile

from benchmarks.duplex.v15_behavior import (
    BEHAVIOR_LABELS,
    BEHAVIOR_RUBRIC,
    BEHAVIOR_RUBRIC_VERSION,
    build_behavior_input,
    call_openai_compatible_judge,
    load_offline_judgements,
    summarize_behavior,
)
from benchmarks.duplex.v15_scoring import (
    SILERO_VAD_CONFIG,
    TIMING_VERSION,
    score_event_timing,
    silero_speech_segments,
    summarize_timing,
)

SHA = "a" * 64


def timing(output_segments, **overrides):
    arguments = {
        "sample_id": "s1",
        "category": "interruption",
        "input_segments": [[0.5, 3.0], [4.0, 5.0]],
        "output_segments": output_segments,
        "event_start_s": 4.0,
        "event_end_s": 5.0,
        "input_duration_s": 6.0,
        "observed_end_s": 10.0,
        "protocol_valid": True,
        "timeline": "media",
        "segment_source": {"kind": "supplied"},
    }
    arguments.update(overrides)
    return score_event_timing(**arguments)


def test_ongoing_output_at_onset_stops_inside_event():
    record = timing([[3.5, 4.3], [6.0, 7.0]])
    assert record["version"] == TIMING_VERSION
    assert record["status"] == "eligible"
    assert record["speaking_at_onset"] is True
    assert record["stop"]["status"] == "stopped"
    assert record["stop"]["latency_s"] == pytest.approx(0.3)
    assert record["response"]["status"] == "responded"
    assert record["response"]["latency_s"] == pytest.approx(1.0)


def test_initial_silence_and_gap_do_not_count_as_overlap():
    record = timing([[1.0, 2.0], [4.2, 4.8], [5.0, 6.0]])
    assert record["speaking_at_onset"] is False
    assert record["stop"] == {
        "status": "not_speaking_at_onset",
        "latency_s": None,
        "span_end_s": None,
        "stopped_during_event": None,
    }
    assert record["response"]["status"] == "responded"
    assert record["response"]["latency_s"] == 0.0


def test_silent_model_is_never_rewarded_with_latency():
    record = timing([], observation_complete=True)
    assert record["has_output_speech"] is False
    assert record["stop"]["status"] == "not_speaking_at_onset"
    assert record["stop"]["latency_s"] is None
    assert record["response"]["status"] == "no_response"
    assert record["response"]["latency_s"] is None
    assert record["response"]["observed_after_event_s"] == 5.0


def test_stop_after_event_end_still_measures_onset_to_span_end():
    record = timing([[3.0, 5.5]])
    assert record["stop"]["status"] == "stopped"
    assert record["stop"]["latency_s"] == pytest.approx(1.5)
    assert record["stop"]["stopped_during_event"] is False
    assert record["response"]["status"] == "speaking_through_event_end"


@pytest.mark.parametrize("observation_complete", [False, True])
def test_speech_reaching_output_eof_is_censored(observation_complete):
    record = timing(
        [[3.0, 9.0]],
        output_duration_s=9.0,
        observed_end_s=12.0,
        observation_complete=observation_complete,
    )
    assert record["output_duration_s"] == 9.0
    assert record["stop"]["status"] == "right_censored"
    assert record["stop"]["latency_s"] is None
    default_eof = timing([[3.0, 9.0]], observed_end_s=9.0, observation_complete=True)
    assert default_eof["stop"]["status"] == "right_censored"


def test_stop_before_output_eof_is_measured():
    record = timing(
        [[3.0, 5.5]],
        output_duration_s=9.0,
        observed_end_s=12.0,
        observation_complete=True,
    )
    assert record["stop"]["status"] == "stopped"
    assert record["stop"]["latency_s"] == pytest.approx(1.5)
    assert record["response"]["status"] == "speaking_through_event_end"


@pytest.mark.parametrize("observed_end_s", [5.5, 30.0])
def test_missing_response_depends_on_completion_not_elapsed_time(observed_end_s):
    incomplete = timing([[3.5, 4.3]], observed_end_s=observed_end_s)
    assert incomplete["observation_complete"] is False
    assert incomplete["response"]["status"] == "right_censored"
    assert incomplete["response"]["latency_s"] is None
    complete = timing(
        [[3.5, 4.3]], observed_end_s=observed_end_s, observation_complete=True
    )
    assert complete["response"]["status"] == "no_response"
    assert complete["response"]["observed_after_event_s"] == pytest.approx(
        observed_end_s - 5.0
    )


def test_clean_reference_uses_anchor_without_event_speech():
    clean_input = [[0.5, 3.0]]
    overlap = timing([[3.5, 4.3]], input_segments=clean_input)
    assert overlap["status"] == "input_speech_absent"
    clean = timing(
        [[3.5, 4.9]], input_segments=clean_input, evaluation="clean_reference"
    )
    assert clean["status"] == "eligible"
    assert clean["evaluation"] == "clean_reference"
    assert clean["stop"]["latency_s"] == pytest.approx(0.9)


def test_ineligible_statuses_keep_metrics_none():
    failed = timing([[3.5, 4.3]], protocol_valid=False)
    absent = timing([[3.5, 4.3]], input_segments=[[0.5, 3.0]])
    unobserved = timing([[1.0, 2.0]], observed_end_s=3.0)
    for record, status in (
        (failed, "protocol_failure"),
        (absent, "input_speech_absent"),
        (unobserved, "not_observed"),
    ):
        assert record["status"] == status
        assert record["speaking_at_onset"] is None
        assert record["stop"]["latency_s"] is None
        assert record["response"]["latency_s"] is None


def test_time_translation_invariance():
    base = timing([[3.5, 4.3], [6.0, 7.0]])
    shift = 1.25
    shifted = timing(
        [[3.5 + shift, 4.3 + shift], [6.0 + shift, 7.0 + shift]],
        input_segments=[[0.5 + shift, 3.0 + shift], [4.0 + shift, 5.0 + shift]],
        event_start_s=4.0 + shift,
        event_end_s=5.0 + shift,
        input_duration_s=6.0 + shift,
        observed_end_s=10.0 + shift,
    )
    assert shifted["stop"]["latency_s"] == pytest.approx(base["stop"]["latency_s"])
    assert shifted["response"]["latency_s"] == pytest.approx(
        base["response"]["latency_s"]
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"output_segments": [[2.0, 1.0]]},
        {"output_segments": [[1.0, 3.0], [2.0, 4.0]]},
        {"output_segments": [[float("nan"), 1.0]]},
        {"output_segments": [[1.0, 11.0]]},
        {"input_segments": [[-0.1, 1.0]]},
        {"event_start_s": 5.0, "event_end_s": 4.0},
        {"event_end_s": 7.0},
        {"observed_end_s": float("inf")},
        {"category": "silence"},
        {"timeline": "acoustic"},
        {"evaluation": "clean"},
        {"output_duration_s": 11.0},
        {"output_duration_s": 0.0},
        {"output_duration_s": 4.0},
    ],
)
def test_malformed_timing_inputs_raise(overrides):
    arguments = {"output_segments": [[3.5, 4.3]], **overrides}
    with pytest.raises(ValueError):
        timing(**arguments)


def test_timing_summary_pairs_only_eligible_clean_and_overlap():
    selected = {"s1": "interruption", "s2": "interruption", "s3": "backchannel"}
    overlap = [
        timing([[3.5, 4.3], [6.0, 7.0]]),
        timing([[3.5, 4.5]], sample_id="s2"),
    ]
    clean_input = [[0.5, 3.0]]
    clean = [
        timing(
            [[3.5, 4.9], [5.5, 6.0]],
            input_segments=clean_input,
            evaluation="clean_reference",
        ),
        timing(
            [[3.5, 4.5]],
            sample_id="s2",
            input_segments=clean_input,
            evaluation="clean_reference",
            protocol_valid=False,
        ),
    ]
    summary = summarize_timing(overlap, clean, selected)
    interruption = summary["categories"]["interruption"]
    assert interruption["selected"] == 2
    assert interruption["eligible"] == 2
    assert interruption["paired_eligible"] == 1
    assert interruption["paired_stop_latency_delta_s"]["mean"] == pytest.approx(-0.6)
    assert interruption["paired_response_latency_delta_s"]["mean"] == pytest.approx(0.5)
    assert summary["categories"]["backchannel"]["missing"] == 1
    assert [pair["sample_id"] for pair in summary["pairs"]] == ["s1"]
    json.dumps(summary)
    with pytest.raises(ValueError):
        summarize_timing(overlap + overlap[:1], [], selected)
    with pytest.raises(ValueError):
        summarize_timing(overlap, overlap, selected)


def transcript(words, source="asr_aligned", duration=10.0):
    return {
        "transcript": {
            "text": " ".join(text for text, _, _ in words),
            "chunks": [
                {"text": text, "timestamp": [start, end]} for text, start, end in words
            ],
        },
        "timestamp_source": source,
        "duration_s": duration,
        "source_sha256": SHA,
    }


def transcripts(**overrides):
    values = {
        "clean_input": transcript(
            [("tell", 0.5, 0.8), ("me", 0.9, 1.0)], "provided_aligned"
        ),
        "noisy_input": transcript(
            [("tell", 0.5, 0.8), ("me", 0.9, 1.0), ("wait", 4.0, 4.4)],
            "provided_aligned",
        ),
        "clean_output": transcript([("sure", 3.0, 3.4), ("thing", 3.5, 3.9)]),
        "noisy_output": transcript(
            [
                ("sure", 3.0, 3.4),
                ("thing", 3.5, 4.1),
                ("yes", 5.5, 5.8),
                ("what", 5.9, 6.2),
            ]
        ),
    }
    values.update(overrides)
    return values


def behavior(sample_id="s1", category="interruption", **overrides):
    return build_behavior_input(
        sample_id=sample_id,
        category=category,
        event_start_s=4.0,
        event_end_s=5.0,
        transcripts=transcripts(**overrides),
        metadata={"timestamps": [4.0, 5.0]},
    )


def test_behavior_input_keeps_complete_transcripts_for_the_judge():
    item = behavior()
    assert item["status"] == "ready"
    payload = item["payload"]
    assert set(payload) == {
        "rubric_version",
        "sample_id",
        "category",
        "event",
        "metadata",
        "transcripts",
    }
    assert set(payload["transcripts"]) == {
        "clean_input",
        "noisy_input",
        "clean_output",
        "noisy_output",
    }
    # Note (wenyao): The word spanning onset stays; the judge decides continuation.
    assert [
        word["text"] for word in payload["transcripts"]["noisy_output"]["words"]
    ] == [
        "sure",
        "thing",
        "yes",
        "what",
    ]
    assert payload["event"] == {"start_s": 4.0, "end_s": 5.0}
    assert "first NEW semantic segment" in BEHAVIOR_RUBRIC
    assert "continue a sentence or idea already begun" in BEHAVIOR_RUBRIC
    assert behavior()["input_hash"] == item["input_hash"]
    changed = behavior(noisy_output=transcript([("no", 5.5, 5.8)]))
    assert changed["input_hash"] != item["input_hash"]


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"noisy_output": None}, "missing_transcript:noisy_output"),
        (
            {"clean_output": transcript([("hi", 1.0, 1.2)], "native_generated")},
            "timestamps_not_asr_aligned:clean_output",
        ),
        (
            {"noisy_output": transcript([("hi", 2.0, 1.0)])},
            "inconsistent_word_times:noisy_output",
        ),
        (
            {"noisy_output": transcript([("a", 3.0, 3.1), ("b", 1.0, 1.1)])},
            "inconsistent_word_times:noisy_output",
        ),
        (
            {"noisy_output": transcript([("a", 3.0, 12.0)])},
            "inconsistent_word_times:noisy_output",
        ),
        (
            {"noisy_output": {**transcript([]), "source_sha256": "bad"}},
            "malformed_transcript:noisy_output",
        ),
    ],
)
def test_unscorable_transcripts_name_the_gap(overrides, reason):
    item = behavior(**overrides)
    assert item["status"] == "unscorable"
    assert item["reason"] == reason


def test_nan_word_time_is_malformed():
    broken = transcript([("a", 1.0, 1.2)])
    broken["transcript"]["chunks"][0]["timestamp"] = [float("nan"), 1.2]
    assert (
        behavior(noisy_output=broken)["reason"] == "malformed_transcript:noisy_output"
    )


def reply(label, evidence="x", **fields):
    values = {
        "label": label,
        "evidence": evidence,
        "first_new_segment": {"text": "yes", "start_s": 5.5, "end_s": 5.8},
    }
    values.update(fields)
    return json.dumps(values)


def test_judge_may_return_null_segment_only_for_unknown(monkeypatch):
    judgement = ask_judge(reply("C_UNKNOWN", first_new_segment=None), monkeypatch)
    assert judgement["status"] == "valid"
    assert judgement["first_new_segment"] is None


def judge_client(content, captured):
    def handler(request):
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "model": "stub-judge",
                "choices": [{"message": {"role": "assistant", "content": content}}],
            },
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def ask_judge(content, monkeypatch, captured=None):
    monkeypatch.setenv("TEST_JUDGE_KEY", "secret")
    return call_openai_compatible_judge(
        behavior(),
        base_url="http://judge.test/v1/",
        model="judge-model",
        api_key_env="TEST_JUDGE_KEY",
        timeout_s=5.0,
        client=judge_client(content, [] if captured is None else captured),
    )


def test_judge_call_keeps_raw_response_and_hashes(monkeypatch):
    captured = []
    segment = {"text": "yes what", "start_s": 5.5, "end_s": 6.2}
    reply = json.dumps(
        {"label": "C_RESPOND", "evidence": "yes what", "first_new_segment": segment}
    )
    judgement = ask_judge(reply, monkeypatch, captured)
    assert judgement["status"] == "valid"
    assert judgement["label"] == "C_RESPOND"
    assert judgement["first_new_segment"] == segment
    assert judgement["input_hash"] == behavior()["input_hash"]
    assert judgement["provenance"]["response_model"] == "stub-judge"
    assert len(judgement["provenance"]["prompt_hash"]) == 64
    assert judgement["raw_response"]["choices"][0]["message"]["content"] == reply
    request = captured[0]
    assert str(request.url) == "http://judge.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer secret"
    body = json.loads(request.content)
    assert body["model"] == "judge-model"
    assert body["messages"][0] == {"role": "system", "content": BEHAVIOR_RUBRIC}
    assert json.loads(body["messages"][1]["content"]) == behavior()["payload"]
    json.dumps(judgement)


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ("```json\n{}\n```", "reply_not_json"),
        (reply("C_SILENCE"), "invalid_label"),
        (reply("C_RESUME", evidence=" "), "missing_evidence"),
        (json.dumps({"label": "C_RESUME", "evidence": "x"}), "reply_schema_mismatch"),
        (reply("C_RESUME", extra=1), "reply_schema_mismatch"),
        (reply("C_RESUME", first_new_segment=None), "missing_first_new_segment"),
        (
            reply(
                "C_RESUME",
                first_new_segment={"text": "thing", "start_s": 3.5, "end_s": 4.1},
            ),
            "pre_onset_first_new_segment",
        ),
        (
            reply(
                "C_RESUME",
                first_new_segment={"text": "yes", "start_s": 5.4, "end_s": 5.8},
            ),
            "segment_bounds_not_word_bounds",
        ),
        (
            reply(
                "C_RESUME",
                first_new_segment={"text": "no way", "start_s": 5.5, "end_s": 6.2},
            ),
            "segment_text_mismatch",
        ),
        (
            reply(
                "C_RESUME",
                first_new_segment={"text": "yes", "start_s": 5.5, "end_s": 6.2},
            ),
            "segment_text_mismatch",
        ),
        (
            reply(
                "C_RESUME",
                first_new_segment={"text": "  ", "start_s": 5.5, "end_s": 5.8},
            ),
            "segment_text_mismatch",
        ),
        (
            reply(
                "C_RESUME",
                first_new_segment={"text": "what", "start_s": 5.9, "end_s": 5.8},
            ),
            "segment_bounds_not_word_bounds",
        ),
        (
            reply("C_RESUME", first_new_segment={"text": "yes", "start_s": 5.5}),
            "invalid_first_new_segment",
        ),
        (None, "malformed_response"),
    ],
)
def test_judge_invalid_replies_stay_unscored(content, reason, monkeypatch):
    judgement = ask_judge(content, monkeypatch)
    assert judgement["status"] == "invalid"
    assert judgement["reason"] == reason
    assert judgement["label"] is None


def test_segment_text_ignores_only_whitespace(monkeypatch):
    punctuated = transcript([("sure", 3.0, 3.4), ("yes", 5.5, 5.8), (",", 5.8, 5.8)])
    monkeypatch.setenv("TEST_JUDGE_KEY", "secret")
    judgement = call_openai_compatible_judge(
        behavior(noisy_output=punctuated),
        base_url="http://judge.test/v1",
        model="judge-model",
        api_key_env="TEST_JUDGE_KEY",
        timeout_s=5.0,
        client=judge_client(
            reply(
                "C_RESPOND",
                first_new_segment={"text": "yes,", "start_s": 5.5, "end_s": 5.8},
            ),
            [],
        ),
    )
    assert judgement["status"] == "valid"


def test_judge_requires_caller_key(monkeypatch):
    monkeypatch.delenv("UNSET_JUDGE_KEY", raising=False)
    with pytest.raises(ValueError):
        call_openai_compatible_judge(
            behavior(),
            base_url="http://judge.test/v1",
            model="judge-model",
            api_key_env="UNSET_JUDGE_KEY",
            timeout_s=5.0,
            client=judge_client("{}", []),
        )


YES_WHAT = {"text": "yes what", "start_s": 5.5, "end_s": 6.2}


def offline_row(sample_id, input_hash, label="C_RESUME", **overrides):
    row = {
        "sample_id": sample_id,
        "input_hash": input_hash,
        "rubric_version": BEHAVIOR_RUBRIC_VERSION,
        "label": label,
        "evidence": "sure thing",
        "annotator": {"id": "rater-1", "kind": "human"},
        "first_new_segment": YES_WHAT,
    }
    row.update(overrides)
    return json.dumps(row)


def test_behavior_summary_accounts_for_every_selected_sample(tmp_path):
    selected = {
        "s1": "interruption",
        "s2": "interruption",
        "s3": "interruption",
        "s4": "backchannel",
        "s5": "talking_to_other",
        "s6": "background_speech",
    }
    ready = {key: behavior(key) for key in ("s1", "s2", "s3")}
    inputs = list(ready.values()) + [
        behavior("s4", "backchannel", noisy_output=None),
        behavior("s5", "talking_to_other"),
    ]
    path = tmp_path / "judgements.jsonl"
    path.write_text(
        "\n".join(
            [
                offline_row("s1", ready["s1"]["input_hash"]),
                offline_row("s2", "b" * 64),
                offline_row("s3", ready["s3"]["input_hash"], label="C_SILENCE"),
                offline_row("ghost", SHA),
            ]
        )
    )
    judgements = load_offline_judgements(path)
    assert judgements[0]["first_new_segment"]["start_s"] == 5.5
    assert judgements[2]["status"] == "invalid"
    assert judgements[2]["submitted_label"] == "C_SILENCE"
    summary = summarize_behavior(inputs, judgements, selected)
    interruption = summary["categories"]["interruption"]
    assert interruption["scored"] == 1
    assert interruption["scored_coverage"] == pytest.approx(1 / 3)
    assert interruption["label_counts"] == {
        label: int(label == "C_RESUME") for label in BEHAVIOR_LABELS
    }
    assert interruption["unscored_reasons"] == {
        "judgement_input_mismatch": 1,
        "invalid_judgement": 1,
    }
    assert summary["categories"]["backchannel"]["unscored_reasons"] == {
        "missing_transcript:noisy_output": 1
    }
    assert summary["categories"]["talking_to_other"]["unscored_reasons"] == {
        "missing_judgement": 1
    }
    assert summary["categories"]["background_speech"]["unscored_reasons"] == {
        "missing_behavior_input": 1
    }
    assert summary["orphan_judgements"] == 1
    json.dumps(summary)


def test_conflicting_valid_judgements_are_unscored(tmp_path):
    item = behavior()
    path = tmp_path / "judgements.jsonl"
    path.write_text(
        offline_row("s1", item["input_hash"])
        + "\n"
        + offline_row("s1", item["input_hash"], label="C_RESPOND")
    )
    summary = summarize_behavior(
        [item], load_offline_judgements(path), {"s1": "interruption"}
    )
    assert summary["samples"][0]["unscored_reason"] == "conflicting_judgements"


@pytest.mark.parametrize(
    ("label", "segment", "error"),
    [
        ("C_RESUME", None, "missing_first_new_segment"),
        (
            "C_RESUME",
            {"text": "thing", "start_s": 3.5, "end_s": 4.1},
            "pre_onset_first_new_segment",
        ),
        (
            "C_RESPOND",
            {"text": "invented", "start_s": 5.5, "end_s": 6.2},
            "segment_text_mismatch",
        ),
        ("C_RESPOND", {"text": "", "start_s": 5.5}, "invalid_first_new_segment"),
    ],
)
def test_offline_segment_evidence_is_checked_against_payload(
    tmp_path, label, segment, error
):
    item = behavior()
    path = tmp_path / "judgements.jsonl"
    path.write_text(
        offline_row("s1", item["input_hash"], label=label, first_new_segment=segment)
    )
    judgements = load_offline_judgements(path)
    assert judgements[0]["first_new_segment"] == segment
    summary = summarize_behavior([item], judgements, {"s1": "interruption"})
    assert summary["samples"][0]["label"] is None
    assert summary["samples"][0]["unscored_reason"] == "invalid_judgement"
    assert summary["samples"][0]["judgement_errors"] == [error]
    assert summary["categories"]["interruption"]["invalid_judgement_reasons"] == {
        error: 1
    }


def test_offline_unknown_without_segment_is_scored(tmp_path):
    item = behavior()
    path = tmp_path / "judgements.jsonl"
    path.write_text(
        offline_row("s1", item["input_hash"], label="C_UNKNOWN", first_new_segment=None)
    )
    summary = summarize_behavior(
        [item], load_offline_judgements(path), {"s1": "interruption"}
    )
    assert summary["samples"][0]["label"] == "C_UNKNOWN"


def test_offline_rows_require_annotator_provenance(tmp_path):
    path = tmp_path / "judgements.jsonl"
    path.write_text(offline_row("s1", SHA, annotator={"id": ""}))
    with pytest.raises(ValueError):
        load_offline_judgements(path)


def test_silero_path_records_provenance(tmp_path):
    pytest.importorskip("silero_vad")
    path = tmp_path / "silence.wav"
    soundfile.write(path, np.zeros(22050, dtype=np.float32), 22050, subtype="PCM_16")
    result = silero_speech_segments(path)
    assert result["segments"] == []
    assert result["duration_s"] == pytest.approx(1.0)
    assert result["vad"]["config"] == SILERO_VAD_CONFIG
    float_path = tmp_path / "float.wav"
    soundfile.write(
        float_path, np.zeros(16000, dtype=np.float32), 16000, subtype="FLOAT"
    )
    with pytest.raises(ValueError):
        silero_speech_segments(float_path)
