from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import yaml

from benchmarks.dataset import prepare, seedtts

_MODELS_DIR = (
    Path(__file__).resolve().parents[3] / ".claude/skills/tune-ci-thresholds/models"
)


class _FakeDataset:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.column_names = list(rows[0].keys()) if rows else []
        self.selected_indices: list[int] | None = None

    def cast_column(self, _name: str, _audio_spec) -> "_FakeDataset":
        return self

    def select(self, indices: list[int]) -> "_FakeDataset":
        self.selected_indices = list(indices)
        return _FakeDataset([self._rows[i] for i in indices])

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)


def test_download_dataset_prewarms_all_mmmu_configs(monkeypatch) -> None:
    calls: list[tuple[str, str | None, str | None]] = []

    def fake_get_dataset_config_names(repo_id: str) -> list[str]:
        assert repo_id == "MMMU/MMMU"
        return ["Accounting", "Math"]

    def fake_load_dataset(
        repo_id: str, config_name: str | None = None, split: str | None = None
    ):
        calls.append((repo_id, config_name, split))
        return object()

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            get_dataset_config_names=fake_get_dataset_config_names,
            load_dataset=fake_load_dataset,
        ),
    )

    prepare.download_dataset("MMMU/MMMU", quiet=True)

    assert calls == [
        ("MMMU/MMMU", "Accounting", "validation"),
        ("MMMU/MMMU", "Math", "validation"),
    ]


def test_load_seedtts_samples_accepts_local_meta_lst(tmp_path: Path) -> None:
    meta_dir = tmp_path / "en"
    meta_dir.mkdir()
    ref_audio = meta_dir / "ref.wav"
    ref_audio.write_bytes(b"wav")
    meta_path = meta_dir / "meta.lst"
    meta_path.write_text(
        "sample-1|hello|ref.wav|target one\nsample-2|world|ref.wav|target two\n"
    )

    samples = seedtts.load_seedtts_samples(str(meta_path), max_samples=1)

    assert len(samples) == 1
    assert samples[0].sample_id == "sample-1"
    assert samples[0].ref_text == "hello"
    assert samples[0].ref_audio == str(ref_audio)
    assert samples[0].target_text == "target one"


def test_load_seedtts_samples_stages_only_selected_rows(
    monkeypatch, tmp_path: Path
) -> None:
    seedtts._STAGED_CACHE.clear()

    rows = [
        {
            "sample_id": f"sample-{idx}",
            "ref_text": f"ref-{idx}",
            "ref_audio_path": f"audio/{idx}.wav",
            "target_text": f"target-{idx}",
            "ref_audio": {"bytes": f"audio-{idx}".encode()},
        }
        for idx in range(5)
    ]
    dataset = _FakeDataset(rows)
    stage_dir = tmp_path / "seedtts_stage"
    stage_dir.mkdir()

    def fake_load_dataset(repo_id: str, split: str):
        assert repo_id == "zhaochenyang20/seed-tts-eval-arrow"
        assert split == "en"
        return dataset

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            Audio=lambda **kwargs: ("Audio", kwargs),
            load_dataset=fake_load_dataset,
        ),
    )
    monkeypatch.setattr(seedtts.tempfile, "mkdtemp", lambda prefix: str(stage_dir))
    monkeypatch.setattr(seedtts.atexit, "register", lambda *args, **kwargs: None)

    samples = seedtts.load_seedtts_samples(
        "zhaochenyang20/seed-tts-eval-arrow",
        max_samples=2,
        split="en",
    )

    assert dataset.selected_indices == [0, 1]
    assert [sample.sample_id for sample in samples] == ["sample-0", "sample-1"]
    assert sorted(
        path.relative_to(stage_dir).as_posix() for path in stage_dir.rglob("*.wav")
    ) == [
        "audio/0.wav",
        "audio/1.wav",
    ]

    seedtts._STAGED_CACHE.clear()


@pytest.mark.parametrize(
    ("ref_audio_path", "outside_name"),
    [
        ("../escape.wav", "escape.wav"),
        (None, "absolute.wav"),
    ],
)
def test_load_seedtts_samples_rejects_unsafe_audio_paths(
    monkeypatch, tmp_path: Path, ref_audio_path: str | None, outside_name: str
) -> None:
    seedtts._STAGED_CACHE.clear()

    stage_dir = tmp_path / "seedtts_stage"
    stage_dir.mkdir()
    outside_path = tmp_path / outside_name
    rows = [
        {
            "sample_id": "sample-0",
            "ref_text": "ref-0",
            "ref_audio_path": (
                ref_audio_path if ref_audio_path is not None else str(outside_path)
            ),
            "target_text": "target-0",
            "ref_audio": {"bytes": b"audio-0"},
        }
    ]

    def fake_load_dataset(repo_id: str, split: str):
        assert repo_id == "zhaochenyang20/seed-tts-eval-arrow"
        assert split == "en"
        return _FakeDataset(rows)

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            Audio=lambda **kwargs: ("Audio", kwargs),
            load_dataset=fake_load_dataset,
        ),
    )
    monkeypatch.setattr(seedtts.tempfile, "mkdtemp", lambda prefix: str(stage_dir))
    monkeypatch.setattr(seedtts.atexit, "register", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="Invalid ref_audio_path"):
        seedtts.load_seedtts_samples(
            "zhaochenyang20/seed-tts-eval-arrow",
            max_samples=1,
            split="en",
        )

    assert not outside_path.exists()
    assert list(stage_dir.rglob("*.wav")) == []

    seedtts._STAGED_CACHE.clear()


def test_tune_ci_threshold_configs_use_arrow_seedtts_datasets() -> None:
    for config_path in sorted(_MODELS_DIR.glob("*/config.yaml")):
        config = yaml.safe_load(config_path.read_text())
        for repo_id in config.get("hf_datasets", []):
            if "seed-tts" not in repo_id:
                continue
            assert repo_id.endswith(
                "-arrow"
            ), f"{config_path} still points to a non-arrow SeedTTS dataset: {repo_id}"


def test_tune_ci_threshold_asr_config_tracks_current_asr_ci_stages() -> None:
    config = yaml.safe_load((_MODELS_DIR / "asr/config.yaml").read_text())
    stages = yaml.safe_load((_MODELS_DIR / "asr/stages.yaml").read_text())

    assert config["test_globs"] == [
        "tests/test_model/test_asr_ci_multi_speaker.py",
        "tests/test_model/test_asr_ci_fun_asr.py",
    ]
    assert "tests/test_model/test_asr_ci.py" not in config["test_globs"]
    assert config["gpus_per_test"] == {
        "test_asr_ci_multi_speaker.py": 2,
        "test_asr_ci_fun_asr.py": 2,
    }
    assert config["hf_model_ids_by_test"] == {
        "test_asr_ci_multi_speaker.py": ["OpenMOSS-Team/MOSS-Transcribe-Diarize"],
        "test_asr_ci_fun_asr.py": ["FunAudioLLM/Fun-ASR-Nano-2512-hf"],
    }
    assert {
        "zhaochenyang20/movies800time",
        "zhaochenyang20/AISHELL4",
        "zhaochenyang20/googletime",
        "zhaochenyang20/seed-tts-eval-arrow",
    }.issubset(config["hf_datasets"])

    assert set(config["metric_sources"]) == {
        "test_asr_ci_multi_speaker.py",
        "test_asr_ci_fun_asr.py",
    }
    assert (
        config["metric_sources"]["test_asr_ci_multi_speaker.py"]["json_file"]
        == "test_moss_transcribe_diarize_m0/moss_transcribe_diarize_results.json"
    )
    assert (
        config["metric_sources"]["test_asr_ci_multi_speaker.py"]["paths"]["cer_percent"]
        == "diarization_metrics_percent.cer"
    )
    assert (
        config["metric_sources"]["test_asr_ci_fun_asr.py"]["variants"]["en"]["paths"][
            "corpus_wer"
        ]
        == "summary.corpus_wer"
    )

    assert set(stages) == {
        "aishell4_long_diarization",
        "aishell4_long_speed",
        "googletime_diarization",
        "googletime_speed",
        "multi_speaker_diarization",
        "multi_speaker_speed",
        "multi_speaker_stream_diarization",
        "multi_speaker_stream_speed",
        "fun_asr_en_wer",
        "fun_asr_en_speed",
        "fun_asr_zh_wer",
    }
    assert stages["multi_speaker_diarization"]["test"] == (
        "tests/test_model/test_asr_ci_multi_speaker.py"
    )
    assert stages["multi_speaker_diarization"]["expected_samples"] == 800
    assert "cer_percent" in stages["multi_speaker_diarization"]["metrics"]
    assert "throughput_qps" in stages["multi_speaker_speed"]["metrics"]
    assert stages["aishell4_long_diarization"]["expected_samples"] == 20
    assert (
        stages["aishell4_long_diarization"]["metrics"]["cer_percent"]["source"]
        == "AISHELL4_LONG_CER_PERCENT_REF"
    )
    assert (
        stages["aishell4_long_speed"]["metrics"]["throughput_qps"]["source"]
        == "AISHELL4_LONG_THROUGHPUT_QPS_REF"
    )
    assert (
        stages["aishell4_long_speed"]["metrics"]["throughput_qps"]["json_file"]
        == "test_moss_transcribe_diarize_m0/moss_transcribe_diarize_aishell4_long_results.json"
    )
    assert stages["googletime_diarization"]["expected_samples"] == 25
    assert (
        stages["googletime_diarization"]["metrics"]["cer_percent"]["source"]
        == "GOOGLETIME_CER_PERCENT_REF"
    )
    assert (
        stages["googletime_diarization"]["metrics"]["n_above_50_pct_cer"]["source"]
        == "GOOGLETIME_N_ABOVE_50_CER_REF"
    )
    assert (
        stages["googletime_diarization"]["metrics"]["cer_no_spk_below_50_corpus"][
            "source"
        ]
        == "GOOGLETIME_CER_NO_SPK_BELOW_50_PERCENT_REF"
    )
    assert (
        stages["googletime_speed"]["metrics"]["throughput_qps"]["source"]
        == "GOOGLETIME_THROUGHPUT_QPS_REF"
    )
    assert (
        stages["googletime_speed"]["metrics"]["throughput_qps"]["json_file"]
        == "test_moss_transcribe_diarize_m0/moss_transcribe_diarize_googletime_results.json"
    )
    assert stages["multi_speaker_stream_diarization"]["expected_samples"] == 800
    assert (
        stages["multi_speaker_stream_speed"]["metrics"]["text_ttft_p95_s"]["source"]
        == "MOSS_TD_STREAM_TEXT_TTFT_P95_S_REF"
    )
    assert (
        stages["multi_speaker_stream_speed"]["metrics"]["text_ttft_p95_s"]["json_file"]
        == "test_moss_transcribe_diarize_m0/moss_transcribe_diarize_stream_results.json"
    )
    assert stages["fun_asr_en_wer"]["test"] == "tests/test_model/test_asr_ci_fun_asr.py"
    assert stages["fun_asr_en_wer"]["expected_samples"] == 1088
    assert stages["fun_asr_zh_wer"]["expected_samples"] == 2020
    assert (
        stages["fun_asr_zh_wer"]["metrics"]["corpus_wer"]["json_file"]
        == "fun_asr_zh_results.json"
    )
    assert "throughput_qps" in stages["fun_asr_en_speed"]["metrics"]


def test_tune_ci_threshold_tts_config_owns_only_tts_stages() -> None:
    config = yaml.safe_load((_MODELS_DIR / "tts/config.yaml").read_text())
    stages = yaml.safe_load((_MODELS_DIR / "tts/stages.yaml").read_text())

    expected_tests = [
        "tests/test_model/test_tts_ci.py",
        "tests/test_model/test_tts_serving_ci.py",
    ]
    assert config["test_globs"] == expected_tests
    assert "test_asr_ci.py" not in config.get("gpus_per_test", {})
    assert "test_asr_ci.py" not in config.get("hf_model_ids_by_test", {})
    assert "test_asr_ci.py" not in config.get("metric_sources", {})
    assert {stage["test"] for stage in stages.values()} == set(expected_tests)
    assert not any(stage_key.startswith("qwen3_asr") for stage_key in stages)
