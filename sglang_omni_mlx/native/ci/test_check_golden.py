# SPDX-License-Identifier: Apache-2.0
"""Unit tests for native golden gates."""

from __future__ import annotations

import hashlib
import json
import subprocess
import wave
from copy import deepcopy
from pathlib import Path
from typing import Literal

import check_golden
import cosyvoice3_golden
import music3_golden
import pytest
import speech_golden

CLIP = {
    "text": "hello",
    "language": "English",
    "generated_token_count": 2,
    "finish_reason": "stop",
}
METRICS = {"wer_en": 0.03, "cer_zh": 0.06, "mer_mixed": 0.16}
SPEECH = {"prompt_ids": "7 8 9", "finish_reason": "stop", "codes": ["1 2", "3 4"]}


def golden(clip_count: int = 10) -> dict:
    return {
        "model": "m",
        "baseline": METRICS,
        "tolerance": {"metric_pp": 1.5, "min_identical": 0.8},
        "devices": {
            "Apple M5": {
                "metrics": METRICS,
                "clips": {f"c{i}": CLIP for i in range(clip_count)},
            }
        },
    }


def results(changed: int, clip_count: int = 10) -> dict:
    return {
        f"c{i}": {**CLIP, "text": "other"} if i < changed else CLIP
        for i in range(clip_count)
    }


def test_a_recorded_chip_requires_every_clip() -> None:
    assert check_golden.check(golden(), "Apple M5", results(0), METRICS)[1] == []
    failures = check_golden.check(golden(), "Apple M5", results(1), METRICS)[1]
    assert failures == ["`c0` differs from its golden output"]


def test_another_chip_is_held_to_the_error_rates() -> None:
    assert check_golden.check(golden(), "Apple M4 Pro", results(2), METRICS)[1] == []
    worse = {**METRICS, "cer_zh": METRICS["cer_zh"] + 0.02}
    failures = check_golden.check(golden(), "Apple M4 Pro", results(2), worse)[1]
    assert failures == ["cer_zh is +2.00 pp from Apple M5's"]


def test_another_chip_needs_most_clips_identical() -> None:
    failures = check_golden.check(golden(), "Apple M4 Pro", results(3), METRICS)[1]
    assert failures == ["only 7/10 clips match Apple M5's outputs"]


def test_an_empty_golden_file_fails() -> None:
    assert check_golden.check(golden(0), "Apple M5", {}, METRICS)[1] == [
        "the golden file has no clips"
    ]


def test_moss_tags_are_dropped_before_scoring() -> None:
    text = "[0.00][S01] Hello [sniff] world.[1.25][1.30][S02]你好[2.00]"
    assert (
        check_golden.spoken_text(text, "moss_transcribe_diarize") == "Hello world. 你好"
    )
    assert check_golden.spoken_text("[S01] kept", None) == "[S01] kept"


def test_a_pending_baseline_reports_no_delta() -> None:
    pending = {**golden(), "baseline": {"source": "pending", **METRICS}}
    lines, failures = check_golden.check(pending, "Apple M5", results(0), METRICS)
    assert failures == []
    assert "| wer_en | pending | 3.00% | n/a |" in lines


def test_music3_golden_requires_complete_stereo_waveform_metadata() -> None:
    reference: music3_golden.Golden = {
        "model": "music3",
        "kind": "minimax_music3",
        "reference": "pinned Python MLX",
        "cases": [
            {
                "id": "cross_chunk",
                "input": "lyrics",
                "instructions": "caption",
                "seed": 42,
                "max_new_tokens": 250,
                "dit_steps": 30,
            }
        ],
        "devices": {},
    }
    waveform: music3_golden.AudioDigest = {
        "sha256": "a" * 64,
        "generated_frames": 250,
        "sample_count": 319879,
        "sample_rate": 32000,
    }
    assert music3_golden.validate_outputs(reference, {"cross_chunk": waveform}) == []
    assert music3_golden.validate_outputs(reference, {})
    assert music3_golden.validate_outputs(
        reference, {"cross_chunk": {**waveform, "sample_rate": 44100}}
    )
    assert music3_golden.validate_outputs(
        reference, {"cross_chunk": {**waveform, "generated_frames": 251}}
    )


@pytest.mark.parametrize(
    ("synthesis_outcome", "expected_failure"),
    [
        ("matching", None),
        ("different", "Music3 offload differs from resident waveforms"),
        ("resident_failure", "resident music3_synthesize exited 1: synthesis failed"),
        ("offload_failure", "offload music3_synthesize exited 1: synthesis failed"),
    ],
)
def test_music3_golden_records_only_matching_successful_waveforms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthesis_outcome: Literal[
        "matching", "different", "resident_failure", "offload_failure"
    ],
    expected_failure: str | None,
) -> None:
    reference_audio_path = tmp_path / "reference.wav"
    with wave.open(str(reference_audio_path), "wb") as reference_audio:
        reference_audio.setnchannels(2)
        reference_audio.setsampwidth(2)
        reference_audio.setframerate(32000)
        reference_audio.writeframes(b"\x01\x00\xff\xff" * 63902)
    reference_waveform: music3_golden.AudioDigest = {
        "sha256": hashlib.sha256(reference_audio_path.read_bytes()).hexdigest(),
        "generated_frames": 50,
        "sample_count": 63902,
        "sample_rate": 32000,
    }
    golden_record: music3_golden.Golden = {
        "model": "music3",
        "kind": "minimax_music3",
        "reference": "fixed stereo waveform",
        "cases": [
            {
                "id": "english_short",
                "input": "lyrics",
                "instructions": "caption",
                "seed": 42,
                "max_new_tokens": 50,
                "dit_steps": 30,
            }
        ],
        "devices": {
            "Apple M5": {"metrics": {}, "clips": {"english_short": reference_waveform}}
        },
    }
    golden_path = tmp_path / "golden.json"
    original_golden = json.dumps(golden_record, indent=1) + "\n"
    golden_path.write_text(original_golden)

    def run_synthesis(
        command: list[str], *, capture_output: bool, text: bool, check: bool
    ) -> subprocess.CompletedProcess[str]:
        is_offload = "--stage-offload-components" in command
        if (synthesis_outcome == "resident_failure" and not is_offload) or (
            synthesis_outcome == "offload_failure" and is_offload
        ):
            return subprocess.CompletedProcess(command, 1, "", "synthesis failed")
        else:
            audio_directory = Path(command[command.index("--output-directory") + 1])
            audio_directory.mkdir(parents=True)
            audio_path = audio_directory / "english_short.wav"
            if synthesis_outcome == "different" and is_offload:
                with wave.open(str(audio_path), "wb") as offload_audio:
                    offload_audio.setnchannels(2)
                    offload_audio.setsampwidth(2)
                    offload_audio.setframerate(32000)
                    offload_audio.writeframes(b"\x02\x00\xfe\xff" * 63902)
            else:
                audio_path.write_bytes(reference_audio_path.read_bytes())
            synthesis_result = {
                "id": "english_short",
                "file": str(audio_path),
                "generated_frames": 50,
                "sample_count": 63902,
                "sample_rate": 32000,
            }
            return subprocess.CompletedProcess(
                command, 0, json.dumps(synthesis_result) + "\n", ""
            )

    monkeypatch.setattr(music3_golden.subprocess, "run", run_synthesis)
    summary_lines, failures = music3_golden.check(
        golden_record, tmp_path, tmp_path, "Apple M5", tmp_path / "outputs", golden_path
    )
    if expected_failure is None:
        assert failures == []
        assert f"Recorded waveforms in {golden_path}." in summary_lines
        assert json.loads(golden_path.read_text()) == json.loads(original_golden)
    else:
        assert expected_failure in failures
        assert f"Recorded waveforms in {golden_path}." not in summary_lines
        assert golden_path.read_text() == original_golden


def speech_golden_file(case_ids: tuple[str, ...] = ("en", "fr")) -> dict:
    return {
        "model": "v",
        "kind": "voxtral_tts",
        "cases": {case_id: {} for case_id in case_ids},
        "devices": {"Apple M5": {"metrics": {}, "clips": {"en": SPEECH, "fr": SPEECH}}},
    }


def test_a_recorded_chip_requires_every_voxtral_code() -> None:
    same = {"en": SPEECH, "fr": SPEECH}
    assert speech_golden.check(speech_golden_file(), "Apple M5", same)[1] == []
    changed = {
        "en": {**SPEECH, "codes": ["1 2", "3 0"]},
        "fr": {**SPEECH, "codes": ["1 2"], "finish_reason": "length"},
    }
    failures = speech_golden.check(speech_golden_file(), "Apple M5", changed)[1]
    assert failures == ["`en` differs from frame 1", "`fr` has frame count 1, not 2"]


def test_a_recorded_chip_requires_the_same_decoded_audio() -> None:
    """Drama 3 and Qwen3-TTS goldens hold a hash of the decoded audio beside the
    codes; only a recorded chip is held to it."""
    recorded = {**SPEECH, "pcm_sha256": "ab"}
    golden = {
        **speech_golden_file(),
        "kind": "fishaudio_drama3",
        "devices": {
            "Apple M5": {"metrics": {}, "clips": {"en": recorded, "fr": recorded}}
        },
    }
    redecoded = {"en": {**recorded, "pcm_sha256": "cd"}, "fr": recorded}
    failures = speech_golden.check(golden, "Apple M5", redecoded)[1]
    assert failures == ["`en` decodes to other audio"]
    assert speech_golden.check(golden, "Apple M4 Pro", redecoded)[1] == []


def test_another_chip_is_held_to_voxtral_prompt_ids() -> None:
    sampled = {"en": {**SPEECH, "codes": ["5 6"]}, "fr": SPEECH}
    assert speech_golden.check(speech_golden_file(), "Apple M4 Pro", sampled)[1] == []
    retokenized = {"en": {**SPEECH, "prompt_ids": "7 8"}, "fr": SPEECH}
    failures = speech_golden.check(speech_golden_file(), "Apple M4 Pro", retokenized)[1]
    assert failures == ["`en` has other prompt ids than Apple M5's"]


@pytest.mark.parametrize("device", ["Apple M5", "Apple M4 Pro"])
def test_a_voxtral_case_without_golden_outputs_fails(device: str) -> None:
    results = {"en": SPEECH, "fr": SPEECH, "de": SPEECH}
    golden = speech_golden_file(("en", "fr", "de"))
    failures = speech_golden.check(golden, device, results)[1]
    assert failures == ["`de` has no golden output"]
    empty = {**speech_golden_file(), "devices": {}}
    assert speech_golden.check(empty, "Apple M5", results)[1] == [
        "the golden file has no outputs"
    ]


@pytest.mark.parametrize("device", ["Apple M5", "Apple M4 Pro"])
@pytest.mark.parametrize("field", ["mel_band_means", "mel_frame_means"])
@pytest.mark.parametrize(
    "source,mel_value,should_fail",
    [
        ("actual", float("nan"), True),
        ("expected", float("nan"), True),
        ("both", float("nan"), True),
        ("actual", float("inf"), True),
        ("expected", float("inf"), True),
        ("both", float("inf"), True),
        ("actual", -float("inf"), True),
        ("expected", -float("inf"), True),
        ("both", -float("inf"), True),
        ("actual", -2.0, False),
        ("actual", -1.975, False),
        ("actual", -1.0, True),
    ],
)
def test_cosyvoice3_golden_rejects_nonfinite_mel_means(
    device: str,
    field: Literal["mel_band_means", "mel_frame_means"],
    source: Literal["actual", "expected", "both"],
    mel_value: float,
    should_fail: bool,
) -> None:
    """NaN and matching infinities previously passed the mel tolerance gate."""
    expected = {
        **SPEECH,
        "mel_band_means": [-2.0, -3.0],
        "mel_frame_means": [-2.0, -3.0],
        "pcm_sha256": "same-waveform",
    }
    actual = deepcopy(expected)
    if source in ("actual", "both"):
        actual[field][0] = mel_value
    else:
        pass
    if source in ("expected", "both"):
        expected[field][0] = mel_value
    else:
        pass
    golden = {
        "model": "cosyvoice3",
        "kind": "cosyvoice3",
        "cases": {"vocode": {"vocode_tokens_from": "speech"}},
        "devices": {"Apple M5": {"clips": {"vocode": expected}}},
    }
    _, failures = cosyvoice3_golden.check(golden, device, {"vocode": actual})
    if should_fail:
        assert len(failures) == 1
        assert field in failures[0]
    else:
        assert failures == []


@pytest.mark.parametrize(
    "device,should_fail", [("Apple M5", True), ("Apple M4 Pro", False)]
)
def test_cosyvoice3_keeps_the_shared_per_chip_pcm_gate(
    device: str, should_fail: bool
) -> None:
    """The shared checker must preserve the recorded chip's CosyVoice3 PCM gate."""
    recorded = {
        **SPEECH,
        "pcm_sha256": "original",
        "mel_band_means": [-2.0],
        "mel_frame_means": [-2.0],
    }
    golden = {
        "model": "cosyvoice3",
        "kind": "cosyvoice3",
        "cases": {"vocode": {"vocode_tokens_from": "speech"}},
        "devices": {"Apple M5": {"clips": {"vocode": recorded}}},
    }
    _, failures = cosyvoice3_golden.check(
        golden, device, {"vocode": {**recorded, "pcm_sha256": "changed"}}
    )
    if should_fail:
        assert failures == ["`vocode` decodes to other audio"]
    else:
        assert failures == []
