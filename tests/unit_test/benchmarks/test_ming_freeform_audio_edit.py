# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import io
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from benchmarks.dataset import ming_freeform_audio_edit as ming_edit
from benchmarks.dataset import prepare
from benchmarks.eval.benchmark_auk_audio_edit import (
    _build_auk_edit_request,
    _make_auk_edit_send_fn,
)
from benchmarks.metrics.speech_edit import (
    aggregate_signal_edit_scores,
    score_signal_edit,
)


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 10) -> None:
    pcm = np.asarray(np.clip(samples, -1, 1) * 32767, dtype="<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def _wav_bytes(samples: np.ndarray, sample_rate: int = 10) -> bytes:
    output = io.BytesIO()
    pcm = np.asarray(np.clip(samples, -1, 1) * 32767, dtype="<i2")
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return output.getvalue()


def _install_hub(
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, Path],
    calls: list[dict],
) -> None:
    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        return str(files[kwargs["filename"]])

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=fake_hf_hub_download),
    )


def test_load_time_stretch_uses_pinned_revision_and_scale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    meta = tmp_path / "metadata.csv"
    meta.write_text(
        "file_name|path|instruction|original_text|edited_text\n"
        "sample-1|wavs/sample-1.wav|adjusts the speed to 1.25|before|after\n",
        encoding="utf-8",
    )
    audio = tmp_path / "sample-1.wav"
    _write_wav(audio, np.array([0.1, -0.1], dtype=np.float32))
    calls: list[dict] = []
    _install_hub(
        monkeypatch,
        {
            "meta/time_stretch/meta_en_time_stretch.csv": meta,
            "wavs/sample-1.wav": audio,
        },
        calls,
    )

    samples = ming_edit.load_ming_freeform_samples("time_stretch")

    assert len(samples) == 1
    assert samples[0].scale == 1.25
    assert samples[0].source_audio == str(audio)
    assert samples[0].source_audio_url.endswith(
        f"/{prepare.MING_FREEFORM_AUDIO_EDIT_DATASET_REVISION}/wavs/sample-1.wav"
    )
    assert {call["revision"] for call in calls} == {
        prepare.MING_FREEFORM_AUDIO_EDIT_DATASET_REVISION
    }


def test_headerless_dialect_metadata_keeps_first_sample(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    meta = tmp_path / "dialect.csv"
    meta.write_text(
        "sample-1|wavs/sample-1.wav|Change the accent to Chengdu.|原文|原文\n",
        encoding="utf-8",
    )
    audio = tmp_path / "sample-1.wav"
    _write_wav(audio, np.array([0.1], dtype=np.float32))
    _install_hub(
        monkeypatch,
        {
            "meta/dialect/meta_zh_dialect.csv": meta,
            "wavs/sample-1.wav": audio,
        },
        [],
    )

    samples = ming_edit.load_ming_freeform_samples("dialect", language="zh")

    assert [sample.sample_id for sample in samples] == ["sample-1"]


def test_loader_rejects_audio_path_traversal_before_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    meta = tmp_path / "metadata.csv"
    meta.write_text(
        "file_name|path|instruction|original_text|edited_text\n"
        "sample-1|../secret.wav|Remove the first word.|before|after\n",
        encoding="utf-8",
    )
    calls: list[dict] = []
    _install_hub(
        monkeypatch,
        {"meta/del/meta_en_deletion_basic.csv": meta},
        calls,
    )

    with pytest.raises(ValueError, match="Invalid source audio path"):
        ming_edit.load_ming_freeform_samples("deletion")

    assert [call["filename"] for call in calls] == [
        "meta/del/meta_en_deletion_basic.csv"
    ]


def test_metadata_paths_cover_semantic_and_acoustic_variants() -> None:
    assert ming_edit.metadata_path("insertion", "zh", "basic") == (
        "meta/ins/meta_zh_insertion_basic.csv"
    )
    assert ming_edit.metadata_path("substitution", "en", "full") == (
        "meta/sub/meta_en_substitution.csv"
    )
    assert ming_edit.metadata_path("volume", "en") == "meta/vol/meta_en_vol.csv"
    with pytest.raises(ValueError, match="only in Chinese"):
        ming_edit.metadata_path("dialect", "en")


def test_time_stretch_signal_metrics_match_published_formula(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    generated = tmp_path / "generated.wav"
    _write_wav(source, np.ones(20, dtype=np.float32) * 0.25)
    _write_wav(generated, np.ones(12, dtype=np.float32) * 0.25)

    score = score_signal_edit(source, generated, task="time_stretch", scale=2.0)

    assert score["target_duration_s"] == pytest.approx(1.0)
    assert score["generated_duration_s"] == pytest.approx(1.2)
    assert score["absolute_duration_error_s"] == pytest.approx(0.2)
    assert score["relative_duration_error"] == pytest.approx(0.1)


def test_volume_signal_metrics_and_aggregate(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    generated = tmp_path / "generated.wav"
    _write_wav(source, np.array([0.25, -0.25], dtype=np.float32))
    _write_wav(generated, np.array([0.4, -0.4], dtype=np.float32))

    score = score_signal_edit(source, generated, task="volume", scale=2.0)
    summary = aggregate_signal_edit_scores([score])

    assert score["target_mean_absolute_amplitude"] == pytest.approx(0.5, abs=1e-4)
    assert score["absolute_amplitude_error"] == pytest.approx(0.1, abs=1e-4)
    assert score["relative_amplitude_error"] == pytest.approx(0.4, abs=1e-3)
    assert summary["evaluated"] == 1
    assert summary["relative_amplitude_error_mean"] == pytest.approx(
        score["relative_amplitude_error"]
    )


def test_prepare_uses_snapshot_download_for_raw_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict = {}
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            get_dataset_config_names=lambda *_args, **_kwargs: [],
            load_dataset=lambda *_args, **_kwargs: pytest.fail("used datasets loader"),
        ),
    )

    def fake_snapshot_download(**kwargs):
        observed.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            hf_hub_download=lambda *_args, **_kwargs: None,
            snapshot_download=fake_snapshot_download,
        ),
    )

    prepare.download_dataset(prepare.DATASETS["ming-freeform-audio-edit"], quiet=True)

    assert observed == {
        "repo_id": prepare.MING_FREEFORM_AUDIO_EDIT_DATASET_ID,
        "repo_type": "dataset",
        "allow_patterns": ["meta/**", "wavs/**"],
        "revision": prepare.MING_FREEFORM_AUDIO_EDIT_DATASET_REVISION,
    }


def test_auk_runner_sends_pinned_source_and_saves_wav(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _write_wav(source, np.ones(20, dtype=np.float32) * 0.25)
    sample = ming_edit.SpeechEditSample(
        sample_id="sample-1",
        task="time_stretch",
        language="en",
        source_audio=str(source),
        source_audio_repo_path="wavs/sample-1.wav",
        source_audio_url="https://example.test/revision/wavs/sample-1.wav",
        instruction="adjusts the speed to 2.0",
        original_text="before",
        edited_text="after",
        scale=2.0,
    )
    expected_audio = _wav_bytes(np.ones(10, dtype=np.float32) * 0.25)

    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def json(self):
            return {
                "audio": {
                    "data": base64.b64encode(expected_audio).decode("ascii"),
                    "format": "wav",
                },
                "meta_info": {"prompt_tokens": 4, "completion_tokens": 8},
            }

    class FakeSession:
        def __init__(self):
            self.url = None
            self.payload = None

        def post(self, url, *, json):
            self.url = url
            self.payload = json
            return FakeResponse()

    generated_dir = tmp_path / "generated"
    send_fn = _make_auk_edit_send_fn(
        api_url="http://localhost:8000/generate",
        model="tencent/AuK",
        seed=1234,
        generated_dir=generated_dir,
    )
    session = FakeSession()
    result = asyncio.run(send_fn(session, sample))

    assert result.is_success
    assert result.audio_duration_s == pytest.approx(1.0)
    assert result.prompt_tokens == 4
    assert result.completion_tokens == 8
    assert Path(result.wav_path).read_bytes() == expected_audio
    assert session.url == "http://localhost:8000/generate"
    assert session.payload == _build_auk_edit_request(
        sample, model="tencent/AuK", seed=1234
    )
    assert session.payload["metadata"] == {
        "tts_params": {"ref_audio": sample.source_audio_url}
    }
