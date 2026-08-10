from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.audar_tts.run_quality_benchmark import (
    DATASET_REPO,
    DATASET_REVISION,
    DATASET_SIZE,
    _load_targets,
)
from benchmarks.audar_tts.summarize_quality import _quality_metrics
from benchmarks.benchmarker.data import RequestResult
from benchmarks.eval.benchmark_tts_seedtts import (
    TtsSeedttsBenchmarkConfig,
    run_tts_seedtts_transcribe,
)
from benchmarks.tasks import tts


def _generation_result(samples: list[dict]) -> dict:
    return {
        "successful_samples": len(samples),
        "truncated_samples": 0,
        "samples": samples,
    }


def test_quality_runner_loads_pinned_hf_dataset(monkeypatch) -> None:
    class FakeDataset:
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def select(self, indices):
            return FakeDataset([self.rows[index] for index in indices])

        def __iter__(self):
            return iter(self.rows)

    calls = []

    def fake_load_dataset(repo, *, split, revision):
        calls.append((repo, split, revision))
        return FakeDataset(
            [
                {
                    "sample_id": f"sample-{index}",
                    "target_text": f"النص العربي {index}",
                }
                for index in range(DATASET_SIZE)
            ]
        )

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )
    targets = _load_targets(SimpleNamespace(samples=1))

    assert calls == [(DATASET_REPO, "test", DATASET_REVISION)]
    assert targets == [{"sample_id": "sample-0", "target_text": "النص العربي 0"}]


def test_arabic_quality_uses_target_and_asr_text_directly() -> None:
    generation = _generation_result(
        [
            {
                "sample_id": "one",
                "target_text": "مرحبا بكم في هذا العالم الجميل",
                "is_success": True,
                "reached_max_new_tokens": False,
            },
            {
                "sample_id": "two",
                "target_text": "هذا اختبار عربي بسيط وواضح جدا",
                "is_success": True,
                "reached_max_new_tokens": False,
            },
        ]
    )
    result = _quality_metrics(
        {
            "config": {"asr_model": "test-asr"},
            "summary": {"wer_corpus": 1 / 12},
            "per_sample": [
                {
                    "id": "one",
                    "is_success": True,
                    "ref_norm": "مرحبا بكم في هذا العالم الجميل",
                    "hyp_norm": "مرحبا بكم في هذا العالم الجميل",
                },
                {
                    "id": "two",
                    "is_success": True,
                    "ref_norm": "هذا اختبار عربي بسيط وواضح جدا",
                    "hyp_norm": "هذا اختبار عربي بسيط جدا",
                },
            ],
        },
        generation,
    )

    assert result["sample_count"] == 2
    assert result["asr_model"] == "test-asr"
    assert result["arabic_wer"] == pytest.approx(1 / 12)
    assert result["arabic_cer"] == pytest.approx(0.1)
    assert result["arabic_bleu"] == pytest.approx(80.6800859479275)
    assert result["arabic_chrf_pp"] == pytest.approx(86.42955282050909)


def test_generated_metadata_feeds_arabic_transcription(
    monkeypatch, tmp_path: Path
) -> None:
    targets = {
        "fleurs-ar-eg-0001": "إِنَّ ٱلْعَرَبِيَّةَ — جَمِيلَةٌ!",
        "fleurs-ar-eg-0002": "مرحبا بكم في هذا العالم الجميل",
    }
    generated = [
        {
            "sample_id": sample_id,
            "target_text": target_text,
            "wav_path": str(tmp_path / f"{sample_id}.wav"),
            "is_success": True,
            "latency_s": 1.0,
            "audio_duration_s": 2.0,
        }
        for sample_id, target_text in targets.items()
    ]
    (tmp_path / "generated.json").write_text(
        json.dumps(generated, ensure_ascii=False), encoding="utf-8"
    )

    async def fake_transcribe(samples, **kwargs):
        assert kwargs == {
            "port": 12345,
            "model_path": "test-asr",
            "lang": "ar",
            "concurrency": 1,
            "warmup": 2,
        }
        assert [sample.sample_id for sample in samples] == list(targets)
        assert [sample.target_text for sample in samples] == list(targets.values())
        assert [sample.ref_text for sample in samples] == list(targets.values())
        return (
            [
                RequestResult(
                    request_id=sample.sample_id,
                    text=sample.target_text,
                    is_success=True,
                    latency_s=0.1,
                )
                for sample in samples
            ],
            0.1,
        )

    monkeypatch.setattr(tts, "run_asr_transcription", fake_transcribe)
    run_tts_seedtts_transcribe(
        TtsSeedttsBenchmarkConfig(
            model="audarai/Audar-TTS-V1-Turbo",
            output_dir=str(tmp_path),
            meta="google/fleurs",
            lang="ar",
            asr_model_path="test-asr",
            asr_concurrency=1,
        ),
        asr_router_port=12345,
    )

    wer = json.loads((tmp_path / "wer_results.json").read_text(encoding="utf-8"))
    assert [sample["id"] for sample in wer["per_sample"]] == list(targets)
    assert [sample["ref_norm"] for sample in wer["per_sample"]] == [
        "ان العربية جميلة",
        "مرحبا بكم في هذا العالم الجميل",
    ]
    assert (
        _quality_metrics(
            wer,
            _generation_result(
                [
                    {
                        "sample_id": sample_id,
                        "target_text": target_text,
                        "is_success": True,
                        "reached_max_new_tokens": False,
                    }
                    for sample_id, target_text in targets.items()
                ]
            ),
        )["arabic_wer"]
        == 0.0
    )


def test_arabic_quality_rejects_asr_reference_not_derived_from_target() -> None:
    generation = _generation_result(
        [
            {
                "sample_id": sample_id,
                "target_text": target_text,
                "is_success": True,
                "reached_max_new_tokens": False,
            }
            for sample_id, target_text in (
                ("one", "النص العربي الأول هنا"),
                ("two", "النص العربي الثاني هنا"),
            )
        ]
    )
    wer = {
        "config": {"asr_model": "test-asr"},
        "summary": {"wer_corpus": 0.0},
        "per_sample": [
            {
                "id": "one",
                "is_success": True,
                "ref_norm": "مرجع خاطئ",
                "hyp_norm": "مرجع خاطئ",
            },
            {
                "id": "two",
                "is_success": True,
                "ref_norm": "النص العربي الثاني هنا",
                "hyp_norm": "النص العربي الثاني هنا",
            },
        ],
    }

    with pytest.raises(ValueError, match="ASR reference does not match target text"):
        _quality_metrics(wer, generation)
