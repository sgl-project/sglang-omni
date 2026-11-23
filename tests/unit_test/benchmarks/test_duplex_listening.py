# SPDX-License-Identifier: Apache-2.0
"""Per-reply listening metrics: cutting replies from recorded sessions,
transcribing them, comparing completed ones with their transcripts, the
leave-one-out voice similarity and the per-category pools, without models."""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile
import torch
from pydantic import JsonValue

from benchmarks.duplex import listening_replies, listening_scores
from benchmarks.duplex.fdb.aggregate import render
from benchmarks.duplex.listening_replies import (
    REPLIES_FILE,
    extract_replies,
    transcribe_replies,
)
from benchmarks.duplex.listening_scores import (
    leave_one_out_similarity,
    run_score,
    summarize,
)
from benchmarks.duplex.reference_audio import SCHEMA as V15_MANIFEST_KIND
from benchmarks.duplex.reference_v10 import MANIFEST_KIND as V10_MANIFEST_KIND
from benchmarks.duplex.run_artifacts import file_sha256

OUTPUT_RATE = 24000
# note (luojiaxuan): a 3.5 s completed reply, a 1.5 s cancelled one, and a
# completed response.done whose reply never sent audio.
REPLY_SAMPLES = {"r1": 84000, "r2": 36000}


def done_event(response_id: str, status: str, transcript: str) -> JsonValue:
    return {
        "direction": "receive",
        "time_s": 1.0,
        "event": {
            "type": "response.done",
            "response": {
                "id": response_id,
                "status": status,
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_audio", "transcript": transcript}],
                    }
                ],
            },
        },
    }


def write_session(
    directory: Path, cuts: list[dict[str, JsonValue]] | None = None
) -> Path:
    """A recorded session: r1 completed, r2 cancelled, r3 completed without
    audio; r1's samples are 1000 and r2's 2000, played back to back from 0."""
    directory.mkdir(parents=True)
    media = np.concatenate(
        [
            np.full(REPLY_SAMPLES["r1"], 1000, dtype=np.int16),
            np.full(REPLY_SAMPLES["r2"], 2000, dtype=np.int16),
        ]
    )
    soundfile.write(
        directory / "output-media.wav", media, OUTPUT_RATE, subtype="PCM_16"
    )
    chunks = [
        {
            "response_id": "r1",
            "trace_line": 2,
            "media_start_sample": 0,
            "playout_start_sample": 0,
            "samples": 42000,
        },
        {
            "response_id": "r1",
            "trace_line": 3,
            "media_start_sample": 42000,
            "playout_start_sample": 42000,
            "samples": 42000,
        },
        {
            "response_id": "r2",
            "trace_line": 5,
            "media_start_sample": 84000,
            "playout_start_sample": 84000,
            "samples": 36000,
        },
    ]
    playout: dict[str, JsonValue] = {"chunks": chunks}
    if cuts is not None:
        playout["cuts"] = cuts
    else:
        pass
    (directory / "playout.json").write_text(json.dumps(playout))
    records = [
        {
            "direction": "send",
            "time_s": 0.0,
            "event": {"type": "input_audio_buffer.append"},
        },
        {
            "direction": "receive",
            "time_s": 0.5,
            "event": {"type": "response.output_audio.delta"},
        },
        {
            "direction": "receive",
            "time_s": 0.6,
            "event": {"type": "response.output_audio.delta"},
        },
        done_event("r1", "completed", "Hello there, friend."),
        {
            "direction": "receive",
            "time_s": 1.5,
            "event": {"type": "response.output_audio.delta"},
        },
        done_event("r2", "cancelled", "I was saying that the weather"),
        done_event("r3", "completed", "Bye."),
    ]
    (directory / "continuous.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    return directory


def source(directory: Path) -> dict[str, str]:
    return {
        "directory": str(directory),
        "trace_sha256": file_sha256(directory / "continuous.jsonl"),
    }


def write_manifests(root: Path) -> list[Path]:
    """A v1.5 export with an eligible overlap and an ineligible clean variant,
    and a v1.0 export with one eligible sample."""
    overlap = write_session(root / "recording" / "overlap")
    v10 = write_session(root / "recording" / "v10")
    v15_manifest = root / "reference-manifest.json"
    v15_manifest.write_text(
        json.dumps(
            {
                "kind": V15_MANIFEST_KIND,
                "samples": [
                    {
                        "sample_id": "user_interruption/1",
                        "category": "user_interruption",
                        "variants": {
                            "overlap": {"eligible": True, "source": source(overlap)},
                            "clean": {"eligible": False, "source": source(overlap)},
                        },
                    }
                ],
            }
        )
    )
    v10_manifest = root / "manifest.json"
    v10_manifest.write_text(
        json.dumps(
            {
                "kind": V10_MANIFEST_KIND,
                "samples": [
                    {
                        "sample_id": "candor_turn_taking/1",
                        "subset": "candor_turn_taking",
                        "eligible": True,
                        "source": source(v10),
                    }
                ],
            }
        )
    )
    return [v15_manifest, v10_manifest]


def test_replies_are_cut_per_response_in_start_order_at_16_khz(tmp_path: Path) -> None:
    manifests = write_manifests(tmp_path)
    out = tmp_path / "listening"
    records = extract_replies(manifests, out)
    assert [
        (
            record["dataset"],
            record["category"],
            record["variant"],
            record["response_id"],
        )
        for record in records
    ] == [
        ("v1.5", "user_interruption", "overlap", "r1"),
        ("v1.5", "user_interruption", "overlap", "r2"),
        ("v1.5", "user_interruption", "overlap", "r3"),
        ("v1.0", "candor_turn_taking", "input", "r1"),
        ("v1.0", "candor_turn_taking", "input", "r2"),
        ("v1.0", "candor_turn_taking", "input", "r3"),
    ]
    first, cancelled, silent = records[:3]
    assert (first["status"], first["transcript"]) == (
        "completed",
        "Hello there, friend.",
    )
    assert (cancelled["status"], silent["status"]) == ("cancelled", "completed")
    assert (first["duration_s"], cancelled["duration_s"], silent["duration_s"]) == (
        3.5,
        1.5,
        0.0,
    )
    assert silent["wav"] is None
    audio, rate = soundfile.read(out / first["wav"], dtype="int16")
    assert rate == 16000 and len(audio) == 56000
    assert abs(int(np.median(audio)) - 1000) <= 1
    audio, _ = soundfile.read(out / cancelled["wav"], dtype="int16")
    assert len(audio) == 24000 and abs(int(np.median(audio)) - 2000) <= 1


def test_a_trace_changed_after_the_export_is_refused(tmp_path: Path) -> None:
    manifests = write_manifests(tmp_path)
    trace = tmp_path / "recording" / "overlap" / "continuous.jsonl"
    trace.write_text(trace.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after the reference export"):
        extract_replies(manifests, tmp_path / "listening")


def test_a_barge_in_cut_keeps_only_what_the_playout_played(tmp_path: Path) -> None:
    # note (luojiaxuan): the cut after r2's delta stops playout 0.5 s into r2; a
    # cut before a delta leaves it whole.
    session = write_session(
        tmp_path / "cut",
        cuts=[
            {"trace_line": 1, "cut_sample": 0},
            {"trace_line": 6, "cut_sample": 84000 + 12000},
        ],
    )
    first, cancelled, _ = listening_replies.variant_replies(session)
    assert len(first["samples"]) == 84000
    assert len(cancelled["samples"]) == 12000


def test_a_reply_cut_shorter_than_one_asr_window_has_no_wav(tmp_path: Path) -> None:
    # note (luojiaxuan): 300 samples at 24 kHz is 200 at 16 kHz, under Parakeet's
    # 25 ms analysis window, which cannot be transcribed.
    session = write_session(
        tmp_path / "recording" / "v10",
        cuts=[{"trace_line": 6, "cut_sample": 84000 + 300}],
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "kind": V10_MANIFEST_KIND,
                "samples": [
                    {
                        "sample_id": "candor_turn_taking/1",
                        "subset": "candor_turn_taking",
                        "eligible": True,
                        "source": source(session),
                    }
                ],
            }
        )
    )
    first, cut, _ = extract_replies([manifest], tmp_path / "listening")
    assert first["wav"] is not None
    assert cut["wav"] is None
    assert cut["duration_s"] == pytest.approx(200 / 16000)


def test_transcribe_writes_each_spoken_reply_its_parakeet_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifests = write_manifests(tmp_path)
    nemo = tmp_path / "parakeet.nemo"
    nemo.write_bytes(b"checkpoint")
    transcribed = []

    class FakeParakeet:
        def transcribe(self, paths: list[str]) -> list[types.SimpleNamespace]:
            transcribed.extend(paths)
            return [
                types.SimpleNamespace(text=f"heard {Path(path).stem}") for path in paths
            ]

    monkeypatch.setattr(
        listening_replies, "load_nemo_model", lambda path, device: FakeParakeet()
    )
    out = tmp_path / "listening"
    with pytest.raises(SystemExit, match="pinned SHA-256"):
        transcribe_replies(manifests, out, nemo, "0" * 64, "cpu")
    transcribe_replies(manifests, out, nemo, file_sha256(nemo), "cpu")
    document = json.loads((out / REPLIES_FILE).read_text())
    assert document["asr"] == {
        "checkpoint_sha256": file_sha256(nemo),
        "device_type": "cpu",
    }
    assert [reply["asr_text"] for reply in document["replies"]] == [
        "heard 0",
        "heard 1",
        None,
    ] * 2
    assert len(transcribed) == 4


def test_leave_one_out_similarity_compares_each_reply_with_the_others() -> None:
    similarities = leave_one_out_similarity(
        torch.tensor([[2.0, 0.0], [1.0, 0.0], [0.0, 3.0]])
    )
    assert similarities[0] == pytest.approx(100 / np.sqrt(2))
    assert similarities[2] == pytest.approx(0.0)
    assert leave_one_out_similarity(torch.tensor([[1.0, 0.0]])) == [None]


def stub_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """UTMOS scores a reply by its loudness; speaker embeddings point one way
    for r1 and another for r2."""
    weights = tmp_path / "weights"
    weights.mkdir()
    for name in ("utmos.jit", "finetune.pth", "wavlm.pt"):
        (weights / name).write_bytes(name.encode())

    class FakeUTMOS:
        def __init__(self, device: str) -> None:
            pass

        def score_batch(self, paths: list[str]) -> list[float]:
            audio, _ = soundfile.read(paths[0], dtype="int16")
            return [float(np.median(audio)) / 1000]

    class FakeSpeaker:
        def __init__(self, **_: str) -> None:
            pass

        def embed(self, paths: list[str]) -> torch.Tensor:
            audio, _ = soundfile.read(paths[0], dtype="int16")
            return torch.tensor(
                [[1.0, 0.0]] if np.median(audio) < 1500 else [[1.0, 1.0]]
            )

    monkeypatch.setattr(listening_scores, "UTMOSScorer", FakeUTMOS)
    monkeypatch.setattr(listening_scores, "WavLMSpeakerSimilarity", FakeSpeaker)
    monkeypatch.setattr(
        listening_scores, "ensure_utmos_assets", lambda: weights / "utmos.jit"
    )
    monkeypatch.setattr(
        listening_scores,
        "ensure_speaker_similarity_assets",
        lambda: types.SimpleNamespace(
            finetune_checkpoint=weights / "finetune.pth",
            wavlm_base=weights / "wavlm.pt",
        ),
    )


def test_completed_replies_are_compared_and_long_enough_replies_are_scored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifests = write_manifests(tmp_path)
    out = tmp_path / "listening"
    out.mkdir()
    replies = extract_replies(manifests, out)
    heard = {"r1": "hello there", "r2": "I was saying", "r3": None}
    for reply in replies:
        reply["asr_text"] = heard[reply["response_id"]]
    (out / REPLIES_FILE).write_text(
        json.dumps({"asr": {"checkpoint_sha256": "a"}, "replies": replies})
    )
    stub_models(monkeypatch, tmp_path)
    run_score(out, "cpu")
    scored = json.loads((out / "scores.json").read_text())["replies"]
    first, cancelled, silent = scored[:3]
    # note (luojiaxuan): "friend" is missing from what Parakeet heard.
    assert first["word_errors"] == {
        "reference_words": 3,
        "substitutions": 0,
        "deletions": 1,
        "insertions": 0,
    }
    assert cancelled["word_errors"] is None
    assert (
        silent["word_errors"]["deletions"]
        == silent["word_errors"]["reference_words"]
        == 1
    )
    # note (luojiaxuan): the 1.5 s reply is too short to score; the v1.5 and v1.0
    # 3.5 s replies share one voice.
    assert (first["utmos"], cancelled["utmos"], silent["utmos"]) == (1.0, None, None)
    assert (first["voice_similarity"], cancelled["voice_similarity"]) == (
        pytest.approx(100.0),
        None,
    )
    window, rate = soundfile.read(out / "voice-window" / first["wav"], dtype="int16")
    assert rate == 16000 and len(window) == 48000
    summary = json.loads((out / "summary.json").read_text())
    assert summary["scored_duration_s"] == 3.0
    assert summary["models"]["utmos_sha256"] == file_sha256(
        tmp_path / "weights" / "utmos.jit"
    )
    interruption = summary["datasets"]["v1.5"]["user_interruption"]
    assert interruption == summary["datasets"]["v1.5"]["all"]
    assert interruption["statuses"] == {"cancelled": 1, "completed": 2}
    assert interruption["compared_replies"] == 2
    assert interruption["short_replies"] == 1
    assert interruption["omission_rate"] == pytest.approx(2 / 4)
    assert interruption["word_error_rate"] == pytest.approx(2 / 4)
    assert interruption["utmos_mean"] == pytest.approx(1.0)
    assert "all" not in summary["datasets"]["v1.0"]


def test_summary_counts_replies_that_never_finished_and_takes_low_percentiles() -> None:
    def reply(category: str, status: str | None, utmos: float | None) -> JsonValue:
        return {
            "dataset": "v1.0",
            "category": category,
            "status": status,
            "wav": None if utmos is None else "x.wav",
            "duration_s": 0.0 if utmos is None else 3.0,
            "utmos": utmos,
            "voice_similarity": None,
            "word_errors": None,
        }

    groups = summarize(
        [reply("icc_backchannel", "completed", 2.0)]
        + [reply("icc_backchannel", None, None)]
        + [
            reply("icc_backchannel", "completed", float(score))
            for score in range(3, 12)
        ]
    )["v1.0"]
    backchannel = groups["icc_backchannel"]
    assert backchannel["statuses"] == {"completed": 10, "missing": 1}
    assert backchannel["replies_with_audio"] == 10
    assert backchannel["utmos_p10"] == pytest.approx(2.9)
    assert backchannel["omission_rate"] is None
    assert backchannel["voice_similarity_mean"] is None


def test_aggregate_adds_a_listening_section_per_dataset(tmp_path: Path) -> None:
    group = {
        "replies": 4,
        "short_replies": 1,
        "compared_replies": 2,
        "omission_rate": 0.25,
        "word_error_rate": 0.5,
        "utmos_mean": 3.5,
        "utmos_p10": 2.0,
        "voice_similarity_mean": 80.0,
        "voice_similarity_p10": 60.0,
    }
    for repeat, utmos in ((1, 3.0), (2, 4.0)):
        listening = tmp_path / "run" / f"repeat-{repeat}" / "listening"
        listening.mkdir(parents=True)
        (listening / "summary.json").write_text(
            json.dumps(
                {
                    "scored_duration_s": 3.0,
                    "datasets": {
                        "v1.5": {"all": {**group, "utmos_mean": utmos}},
                        **({"v1.0": {"icc_backchannel": group}} if repeat == 1 else {}),
                    },
                }
            )
        )
        (listening.parent / "v10").mkdir()
        (listening.parent / "v10" / "sample-ids.txt").write_text("icc_backchannel/1\n")
    results = render(tmp_path / "run", "minicpmo", "qwen")
    section = results[results.index("## Listening quality") :]
    assert "Omission rate (lower is better)" in section
    assert "UTMOS (1-5, higher is better) scores replies of at least 3 s" in section
    assert "| all | 4 | 1 | 2 | 25.0 ± 0.0% | 50.0 ± 0.0% | 3.500 ± 0.707 |" in section
    assert "v1.0 cells come from repeat-1 only" in section
    assert (
        "| icc_backchannel | 4 | 1 | 2 | 25.0% | 50.0% | 3.500 | 2.000 | 80.000 | 60.000 |"
        in section
    )


def test_aggregate_refuses_repeats_scored_with_different_floors(tmp_path: Path) -> None:
    for repeat, floor_s in ((1, 3.0), (2, 2.0)):
        listening = tmp_path / "run" / f"repeat-{repeat}" / "listening"
        listening.mkdir(parents=True)
        (listening / "summary.json").write_text(
            json.dumps({"scored_duration_s": floor_s, "datasets": {}})
        )
        (listening.parent / "v10").mkdir()
        (listening.parent / "v10" / "sample-ids.txt").write_text("icc_backchannel/1\n")
    with pytest.raises(SystemExit, match="different shortest scored durations"):
        render(tmp_path / "run", "minicpmo", "qwen")
