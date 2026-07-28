# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from benchmarks.eval import (
    benchmark_asr_seedtts,
    benchmark_asr_stability,
    benchmark_whisper_translation,
)
from benchmarks.tasks.asr import FUN_ASR_MODEL_PATH, OMNI_WHISPER_MODEL_PATH


def test_consumer_asr_models_have_pinned_revisions() -> None:
    assert benchmark_asr_seedtts.MODEL_REVISIONS[FUN_ASR_MODEL_PATH] == (
        "854d88f94205cd17d2afdb24332130d86fbe654a"
    )
    assert benchmark_asr_seedtts.MODEL_REVISIONS[OMNI_WHISPER_MODEL_PATH] == (
        "06f233fe06e710322aca913c1bc4249a0d71fce1"
    )
    assert (
        benchmark_asr_stability.MODEL_REVISIONS == benchmark_asr_seedtts.MODEL_REVISIONS
    )


def test_stability_memory_validation_checks_headroom_and_retention() -> None:
    checkpoints = [
        {
            "label": "before_functional",
            "nvidia_smi_csv": "1000, 23564, 12.0, 0",
        },
        {
            "label": "after_cooldown",
            "nvidia_smi_csv": "1100, 23464, 15.0, 0",
        },
    ]
    resources = {"gpu_memory_free_mib": {"min": 4096.0}}

    result = benchmark_asr_stability._validate_memory_retention(
        checkpoints,
        resources,
        min_free_memory_mib=2048.0,
        max_retained_memory_mib=256.0,
    )

    assert result["passed"] is True
    assert result["retained_after_cooldown_mib"] == 100.0


def test_stability_memory_validation_rejects_missing_samples() -> None:
    result = benchmark_asr_stability._validate_memory_retention(
        [],
        {},
        min_free_memory_mib=2048.0,
        max_retained_memory_mib=256.0,
    )

    assert result["passed"] is False
    assert "unavailable" in result["error"]


def test_translation_quality_reports_exact_match() -> None:
    result = benchmark_whisper_translation._translation_quality(
        ["this is an exact translation match"],
        ["this is an exact translation match"],
    )

    assert result["corpus_wer"] == 0.0
    if result["bleu"] is not None:
        assert result["bleu"] > 99.0
        assert result["chrf"] > 99.0


def test_translation_audio_payload_uses_inline_bytes() -> None:
    audio_bytes, filename = benchmark_whisper_translation._audio_payload(
        {"bytes": b"mp3", "path": "/dataset/example.mp3"},
        index=0,
    )

    assert audio_bytes == b"mp3"
    assert filename == "example.mp3"
