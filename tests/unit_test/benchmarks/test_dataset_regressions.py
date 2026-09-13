from __future__ import annotations

import io
import json
import struct
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from benchmarks.dataset import asr_longform, prepare, seedtts, stt_benchmark
from benchmarks.eval import (
    benchmark_asr_longform,
    benchmark_asr_seedtts,
    benchmark_asr_stt_benchmark,
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


def test_download_seedtts_uses_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict = {}

    def fake_load_dataset(repo_id: str, **kwargs):
        observed["repo_id"] = repo_id
        observed.update(kwargs)
        return object()

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            get_dataset_config_names=lambda *_args, **_kwargs: [],
            load_dataset=fake_load_dataset,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=lambda *_args, **_kwargs: None),
    )

    prepare.download_dataset(prepare.SEEDTTS_DATASET_ID, quiet=True)

    assert observed == {
        "repo_id": prepare.SEEDTTS_DATASET_ID,
        "revision": prepare.SEEDTTS_DATASET_REVISION,
    }


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


def test_local_seedtts_source_does_not_claim_huggingface_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    meta_path = tmp_path / "meta.lst"
    meta_path.write_text("sample-1|hello|ref.wav|target one\n")
    (tmp_path / "ref.wav").write_bytes(b"audio")
    output_path = tmp_path / "result.json"
    captured: dict = {}

    async def empty_sweep(*_args, **_kwargs):
        return []

    def capture_provenance(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_asr_seedtts",
            "--port",
            "8000",
            "--meta",
            str(meta_path),
            "--output",
            str(output_path),
        ],
    )
    monkeypatch.setattr(benchmark_asr_seedtts, "_sweep", empty_sweep)
    monkeypatch.setattr(
        benchmark_asr_seedtts,
        "collect_benchmark_provenance",
        capture_provenance,
    )

    benchmark_asr_seedtts.main()
    assert captured["dataset_revision"] is None
    assert captured["model_revision"] is None
    assert captured["server_config"]["quantization"] is None


def test_custom_seedtts_repo_does_not_use_canonical_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "result.json"
    loaded: dict = {}
    captured: dict = {}
    audio_path = tmp_path / "custom.wav"
    audio_path.write_bytes(b"audio")

    def capture_load(source: str, **kwargs):
        loaded["source"] = source
        loaded.update(kwargs)
        return [
            seedtts.SampleInput(
                sample_id="sample-1",
                ref_text="reference",
                ref_audio=str(audio_path),
                target_text="target",
            )
        ]

    async def empty_sweep(*_args, **_kwargs):
        return []

    def capture_provenance(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_asr_seedtts",
            "--port",
            "8000",
            "--meta",
            "example/custom-seedtts",
            "--output",
            str(output_path),
        ],
    )
    monkeypatch.setattr(
        benchmark_asr_seedtts,
        "load_seedtts_samples",
        capture_load,
    )
    monkeypatch.setattr(benchmark_asr_seedtts, "_sweep", empty_sweep)
    monkeypatch.setattr(
        benchmark_asr_seedtts,
        "collect_benchmark_provenance",
        capture_provenance,
    )

    benchmark_asr_seedtts.main()

    assert loaded["revision"] is None
    assert captured["dataset_revision"] is None


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--concurrencies", ""],
        ["--concurrencies", "0"],
        ["--repeats", "0"],
    ],
)
def test_asr_benchmark_cli_rejects_empty_work(
    monkeypatch: pytest.MonkeyPatch,
    extra_args: list[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark_asr_seedtts", "--port", "8000", *extra_args],
    )

    with pytest.raises(SystemExit):
        benchmark_asr_seedtts.parse_args()


def test_asr_benchmark_rejects_empty_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_asr_seedtts",
            "--port",
            "8000",
            "--meta",
            "example/empty",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    monkeypatch.setattr(
        benchmark_asr_seedtts,
        "load_seedtts_samples",
        lambda *_args, **_kwargs: [],
    )

    with pytest.raises(RuntimeError, match="No SeedTTS samples"):
        benchmark_asr_seedtts.main()


def test_evaluation_input_fingerprint_tracks_audio_content(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"first")
    samples = [
        seedtts.SampleInput(
            sample_id="sample-1",
            ref_text="reference",
            ref_audio=str(audio_path),
            target_text="target",
        )
    ]

    before = benchmark_asr_seedtts._evaluation_input_sha256(samples)
    audio_path.write_bytes(b"second")
    after = benchmark_asr_seedtts._evaluation_input_sha256(samples)

    assert before != after


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

    def fake_load_dataset(repo_id: str, split: str, revision: str | None = None):
        assert repo_id == "zhaochenyang20/seed-tts-eval-arrow"
        assert split == "en"
        assert revision == prepare.SEEDTTS_DATASET_REVISION
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

    def fake_load_dataset(repo_id: str, split: str, revision: str):
        assert repo_id == prepare.SEEDTTS_DATASET_ID
        assert split == "en"
        assert revision == prepare.SEEDTTS_DATASET_REVISION
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


def test_download_stt_benchmark_uses_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict = {}

    def fake_load_dataset(repo_id: str, **kwargs):
        observed["repo_id"] = repo_id
        observed.update(kwargs)
        return object()

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            get_dataset_config_names=lambda *_args, **_kwargs: [],
            load_dataset=fake_load_dataset,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=lambda *_args, **_kwargs: None),
    )

    prepare.download_dataset(prepare.DATASETS["stt-benchmark"], quiet=True)

    assert observed == {
        "repo_id": prepare.STT_BENCHMARK_DATASET_ID,
        "revision": prepare.STT_BENCHMARK_DATASET_REVISION,
    }


def _stt_wav_bytes(idx: int) -> bytes:
    """A valid minimal 16 kHz mono PCM WAV whose frames vary with *idx*."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(struct.pack("<4h", idx, -idx, idx + 1, 0))
    return buffer.getvalue()


def _stt_rows(count: int) -> list[dict]:
    return [
        {
            "sample_id": f"sample-{idx}",
            "audio": {"bytes": _stt_wav_bytes(idx), "path": None},
            "duration_seconds": 1.0 + idx,
            "transcription": f"Transcript {idx}.",
        }
        for idx in range(count)
    ]


def _install_fake_datasets(monkeypatch: pytest.MonkeyPatch, load_dataset) -> None:
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            Audio=lambda **kwargs: ("Audio", kwargs),
            load_dataset=load_dataset,
        ),
    )


def _stage_stt_into(monkeypatch: pytest.MonkeyPatch, stage_dir: Path) -> None:
    monkeypatch.setattr(
        stt_benchmark.tempfile, "mkdtemp", lambda prefix: str(stage_dir)
    )
    monkeypatch.setattr(stt_benchmark.atexit, "register", lambda *a, **k: None)


def test_load_stt_benchmark_samples_stages_selected_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stt_benchmark._STAGED_CACHE.clear()

    dataset = _FakeDataset(_stt_rows(5))
    stage_dir = tmp_path / "stt_stage"
    stage_dir.mkdir()

    def fake_load_dataset(repo_id: str, split: str, revision: str | None = None):
        assert repo_id == prepare.STT_BENCHMARK_DATASET_ID
        assert split == "train"
        assert revision == prepare.STT_BENCHMARK_DATASET_REVISION
        return dataset

    _install_fake_datasets(monkeypatch, fake_load_dataset)
    _stage_stt_into(monkeypatch, stage_dir)

    samples = stt_benchmark.load_stt_benchmark_samples(max_samples=2)

    assert dataset.selected_indices == [0, 1]
    assert [sample.sample_id for sample in samples] == ["sample-0", "sample-1"]
    assert samples[0].ref_text == "Transcript 0."
    assert samples[0].target_text == "Transcript 0."
    assert Path(samples[0].ref_audio) == stage_dir / "sample-0.wav"
    assert (stage_dir / "sample-0.wav").read_bytes() == _stt_wav_bytes(0)
    assert sorted(path.name for path in stage_dir.glob("*.wav")) == [
        "sample-0.wav",
        "sample-1.wav",
    ]

    # A repeat call with the same arguments comes from the staged cache.
    monkeypatch.setattr(
        stt_benchmark.tempfile,
        "mkdtemp",
        lambda prefix: pytest.fail("reloaded instead of using the cache"),
    )
    again = stt_benchmark.load_stt_benchmark_samples(max_samples=2)
    assert [sample.sample_id for sample in again] == ["sample-0", "sample-1"]

    stt_benchmark._STAGED_CACHE.clear()


def test_custom_stt_benchmark_repo_loads_default_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stt_benchmark._STAGED_CACHE.clear()
    observed: dict = {}

    def fake_load_dataset(repo_id: str, split: str, **kwargs):
        observed["repo_id"] = repo_id
        observed["split"] = split
        observed.update(kwargs)
        return _FakeDataset(_stt_rows(1))

    _install_fake_datasets(monkeypatch, fake_load_dataset)
    _stage_stt_into(monkeypatch, tmp_path)

    samples = stt_benchmark.load_stt_benchmark_samples(
        "example/custom-stt", split="validation"
    )

    assert len(samples) == 1
    assert observed == {"repo_id": "example/custom-stt", "split": "validation"}

    stt_benchmark._STAGED_CACHE.clear()


@pytest.mark.parametrize("sample_id", ["../escape", "nested/id", "", ".."])
def test_load_stt_benchmark_samples_rejects_unsafe_sample_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_id: str
) -> None:
    stt_benchmark._STAGED_CACHE.clear()

    stage_dir = tmp_path / "stt_stage"
    stage_dir.mkdir()
    rows = _stt_rows(1)
    rows[0]["sample_id"] = sample_id

    _install_fake_datasets(monkeypatch, lambda *a, **k: _FakeDataset(rows))
    _stage_stt_into(monkeypatch, stage_dir)

    with pytest.raises(ValueError, match="Invalid sample_id"):
        stt_benchmark.load_stt_benchmark_samples(max_samples=1)

    assert list(tmp_path.rglob("*.wav")) == []

    stt_benchmark._STAGED_CACHE.clear()


def _one_stt_sample(tmp_path: Path) -> list[seedtts.SampleInput]:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    return [
        seedtts.SampleInput(
            sample_id="sample-1",
            ref_text="Transcript.",
            ref_audio=str(audio_path),
            target_text="Transcript.",
        )
    ]


def _run_stt_benchmark_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    argv: list[str],
    samples: list[seedtts.SampleInput],
) -> tuple[dict, dict]:
    loaded: dict = {}
    captured: dict = {}

    def capture_load(repo_id: str, **kwargs):
        loaded["repo_id"] = repo_id
        loaded.update(kwargs)
        return samples

    async def empty_sweep(*_args, **_kwargs):
        return []

    def capture_provenance(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_asr_stt_benchmark",
            "--port",
            "8000",
            "--output",
            str(tmp_path / "result.json"),
            *argv,
        ],
    )
    monkeypatch.setattr(
        benchmark_asr_stt_benchmark, "load_stt_benchmark_samples", capture_load
    )
    monkeypatch.setattr(benchmark_asr_stt_benchmark, "_sweep", empty_sweep)
    monkeypatch.setattr(
        benchmark_asr_stt_benchmark,
        "collect_benchmark_provenance",
        capture_provenance,
    )
    benchmark_asr_stt_benchmark.main()
    return loaded, captured


def test_stt_benchmark_main_pins_canonical_revision_and_english(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded, captured = _run_stt_benchmark_main(
        monkeypatch, tmp_path, argv=[], samples=_one_stt_sample(tmp_path)
    )

    assert loaded == {
        "repo_id": prepare.STT_BENCHMARK_DATASET_ID,
        "max_samples": None,
        "split": "train",
        "revision": prepare.STT_BENCHMARK_DATASET_REVISION,
    }
    assert captured["dataset_id"] == prepare.STT_BENCHMARK_DATASET_ID
    assert captured["dataset_revision"] == prepare.STT_BENCHMARK_DATASET_REVISION
    assert captured["model_revision"] is None

    payload = json.loads((tmp_path / "result.json").read_text())
    assert payload["schema_version"] == 2
    assert payload["config"]["repo_id"] == prepare.STT_BENCHMARK_DATASET_ID
    assert payload["config"]["split"] == "train"
    assert payload["config"]["lang"] == "en"
    assert payload["config"]["num_samples"] == 1
    assert payload["results"] == []


def test_custom_stt_benchmark_repo_does_not_use_canonical_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded, captured = _run_stt_benchmark_main(
        monkeypatch,
        tmp_path,
        argv=["--repo-id", "example/custom-stt", "--max-samples", "3"],
        samples=_one_stt_sample(tmp_path),
    )

    assert loaded["revision"] is None
    assert loaded["max_samples"] == 3
    assert captured["dataset_revision"] is None


def test_stt_benchmark_main_rejects_empty_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(RuntimeError, match="No STT benchmark samples"):
        _run_stt_benchmark_main(monkeypatch, tmp_path, argv=[], samples=[])


def test_evaluation_input_fingerprint_is_namespaced(tmp_path: Path) -> None:
    samples = _one_stt_sample(tmp_path)
    fingerprint = benchmark_asr_seedtts._evaluation_input_sha256

    assert fingerprint(samples) == fingerprint(samples, namespace="seedtts")
    assert fingerprint(samples) != fingerprint(samples, namespace="stt-benchmark")


@pytest.mark.parametrize(
    ("dataset_name", "repo_id", "split", "revision"),
    [
        (
            "longlibriheavy-30",
            prepare.LONGLIBRIHEAVY_DATASET_ID,
            "llh_test_30",
            prepare.LONGLIBRIHEAVY_DATASET_REVISION,
        ),
        (
            "longlibriheavy-60",
            prepare.LONGLIBRIHEAVY_DATASET_ID,
            "llh_test_60",
            prepare.LONGLIBRIHEAVY_DATASET_REVISION,
        ),
        (
            "meanwhile",
            prepare.MEANWHILE_DATASET_ID,
            "test",
            prepare.MEANWHILE_DATASET_REVISION,
        ),
    ],
)
def test_download_asr_longform_dataset_uses_split_and_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
    dataset_name: str,
    repo_id: str,
    split: str,
    revision: str,
) -> None:
    observed: dict = {}

    def fake_load_dataset(observed_repo_id: str, **kwargs):
        observed["repo_id"] = observed_repo_id
        observed.update(kwargs)
        return object()

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            get_dataset_config_names=lambda *_args, **_kwargs: [],
            load_dataset=fake_load_dataset,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=lambda *_args, **_kwargs: None),
    )

    prepare.download_dataset(prepare.DATASETS[dataset_name], quiet=True)

    assert observed == {
        "repo_id": repo_id,
        "split": split,
        "revision": revision,
    }


def _longform_rows(count: int) -> list[dict]:
    return [
        {
            "audio": {"bytes": f"encoded-{index}".encode(), "path": None},
            "text": f"Transcript {index}.",
        }
        for index in range(count)
    ]


def _stage_longform_into(monkeypatch: pytest.MonkeyPatch, staging_dir: Path) -> None:
    monkeypatch.setattr(
        asr_longform.tempfile, "mkdtemp", lambda prefix: str(staging_dir)
    )
    monkeypatch.setattr(asr_longform.atexit, "register", lambda *a, **k: None)


def test_load_asr_longform_samples_selects_before_decode_and_stages_pcm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    asr_longform._STAGED_CACHE.clear()
    dataset = _FakeDataset(_longform_rows(5))
    staging_dir = tmp_path / "longform_stage"
    staging_dir.mkdir()
    decoded_sources: list[bytes | str] = []

    def fake_load_audio(source, **kwargs):
        decoded_sources.append(source)
        assert kwargs["target_sample_rate"] == 16000
        assert kwargs["mono"] is True
        return np.array([0.0, 0.25, -0.25, 0.0], dtype=np.float32)

    monkeypatch.setattr(asr_longform, "load_dataset", lambda *a, **k: dataset)
    monkeypatch.setattr(asr_longform, "Audio", lambda **kwargs: ("Audio", kwargs))
    monkeypatch.setattr(asr_longform, "load_audio", fake_load_audio)
    _stage_longform_into(monkeypatch, staging_dir)

    samples = asr_longform.load_asr_longform_samples("meanwhile", max_samples=2)

    assert dataset.selected_indices == [0, 1]
    assert decoded_sources == [b"encoded-0", b"encoded-1"]
    assert [sample.sample_id for sample in samples] == [
        "meanwhile-000000",
        "meanwhile-000001",
    ]
    assert samples[0].ref_text == "Transcript 0."
    assert samples[0].target_text == "Transcript 0."
    with wave.open(samples[0].ref_audio, "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16000
        assert wav_file.getnframes() == 4

    monkeypatch.setattr(
        asr_longform,
        "load_dataset",
        lambda *a, **k: pytest.fail("reloaded instead of using staged cache"),
    )
    again = asr_longform.load_asr_longform_samples("meanwhile", max_samples=2)
    assert [sample.sample_id for sample in again] == [
        "meanwhile-000000",
        "meanwhile-000001",
    ]

    asr_longform._STAGED_CACHE.clear()


def test_load_asr_longform_full_split_checks_canonical_sample_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asr_longform._STAGED_CACHE.clear()
    monkeypatch.setattr(
        asr_longform,
        "load_dataset",
        lambda *a, **k: _FakeDataset(_longform_rows(1)),
    )

    with pytest.raises(ValueError, match="Expected 64 samples"):
        asr_longform.load_asr_longform_samples("meanwhile")


def test_load_asr_longform_rejects_empty_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    asr_longform._STAGED_CACHE.clear()
    rows = _longform_rows(1)
    rows[0]["text"] = "  "
    staging_dir = tmp_path / "longform_stage"
    staging_dir.mkdir()
    monkeypatch.setattr(
        asr_longform, "load_dataset", lambda *a, **k: _FakeDataset(rows)
    )
    monkeypatch.setattr(asr_longform, "Audio", lambda **kwargs: ("Audio", kwargs))
    _stage_longform_into(monkeypatch, staging_dir)

    with pytest.raises(ValueError, match="Empty text"):
        asr_longform.load_asr_longform_samples("meanwhile", max_samples=1)


def test_asr_longform_main_records_registered_dataset_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded: dict = {}
    captured: dict = {}
    output_path = tmp_path / "result.json"

    def capture_load(dataset_name: str, **kwargs):
        loaded["dataset_name"] = dataset_name
        loaded.update(kwargs)
        return _one_stt_sample(tmp_path)

    async def empty_sweep(*_args, **_kwargs):
        return []

    def capture_provenance(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_asr_longform",
            "--dataset",
            "meanwhile",
            "--port",
            "8000",
            "--output",
            str(output_path),
        ],
    )
    monkeypatch.setattr(
        benchmark_asr_longform, "load_asr_longform_samples", capture_load
    )
    monkeypatch.setattr(benchmark_asr_longform, "_sweep", empty_sweep)
    monkeypatch.setattr(benchmark_asr_longform, "_print_table", lambda *_args: None)
    monkeypatch.setattr(
        benchmark_asr_longform,
        "collect_benchmark_provenance",
        capture_provenance,
    )

    benchmark_asr_longform.main()

    assert loaded == {
        "dataset_name": "meanwhile",
        "max_samples": None,
        "revision": prepare.MEANWHILE_DATASET_REVISION,
    }
    assert captured["dataset_id"] == prepare.MEANWHILE_DATASET_ID
    assert captured["dataset_revision"] == prepare.MEANWHILE_DATASET_REVISION
    payload = json.loads(output_path.read_text())
    assert payload["schema_version"] == 2
    assert payload["config"]["dataset"] == "meanwhile"
    assert payload["config"]["repo_id"] == prepare.MEANWHILE_DATASET_ID
    assert payload["config"]["split"] == "test"
    assert payload["config"]["lang"] == "en"
    assert payload["config"]["expected_num_samples"] == 64
    assert payload["results"] == []
