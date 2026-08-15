# SPDX-License-Identifier: Apache-2.0
import hashlib
from pathlib import Path
from types import SimpleNamespace

import yaml

from scripts.ci.run_rocm_model_e2e import (
    IGNORED_SNAPSHOT_FILES,
    MING_OMNI_SOURCE_REVISION,
    ModelCase,
    _snapshot_ignore_patterns,
    _stage_auxiliary_assets,
    _tts_payload,
)
from sglang_omni.models.accelerator_support import iter_model_accelerator_support

MANIFEST = Path("scripts/ci/rocm_model_e2e_cases.yaml")


def _models() -> list[dict]:
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    return manifest["models"]


def test_e2e_manifest_covers_every_declared_rocm_architecture() -> None:
    expected = {entry.architecture for entry in iter_model_accelerator_support("rocm")}
    actual = {model["architecture"] for model in _models()}

    assert actual == expected
    assert len(actual) == 19


def test_e2e_manifest_has_unique_ids_and_pinned_revisions() -> None:
    models = _models()

    assert len({model["id"] for model in models}) == len(models)
    assert len({model["checkpoint"] for model in models}) == len(models)
    for model in models:
        revision = model["revision"]
        assert len(revision) == 40
        assert all(character in "0123456789abcdef" for character in revision)


def test_e2e_manifest_references_existing_configs() -> None:
    for model in _models():
        config = model.get("config")
        if config is not None:
            assert Path(config).is_file(), (model["id"], config)


def test_ming_runtime_onnx_artifacts_are_not_ignored() -> None:
    def case(model_id: str) -> ModelCase:
        return ModelCase(
            id=model_id,
            architecture="test",
            checkpoint="test/model",
            revision="0" * 40,
            mode="chat",
            required_gpus=1,
        )

    assert "*.onnx" in _snapshot_ignore_patterns(case("qwen3_omni"))
    assert "*.onnx" not in _snapshot_ignore_patterns(case("ming_tts"))
    assert "*.onnx_data" not in _snapshot_ignore_patterns(case("ming_omni"))
    assert _snapshot_ignore_patterns(case("qwen3_omni")) is IGNORED_SNAPSHOT_FILES


def test_ming_tts_e2e_request_does_not_send_unsupported_seed() -> None:
    case = ModelCase(
        id="ming_tts",
        architecture="test",
        checkpoint="test/model",
        revision="0" * 40,
        mode="tts",
        required_gpus=2,
    )

    assert "seed" not in _tts_payload(case)


def test_ming_omni_stages_checksummed_upstream_voice_assets(
    tmp_path: Path, monkeypatch
) -> None:
    payload = b"official voice asset"
    digest = hashlib.sha256(payload).hexdigest()
    case = ModelCase(
        id="ming_omni",
        architecture="test",
        checkpoint="test/model",
        revision="0" * 40,
        mode="chat_audio",
        required_gpus=5,
    )
    monkeypatch.setattr(
        "scripts.ci.run_rocm_model_e2e.MING_OMNI_AUXILIARY_ASSETS",
        (
            {
                "source": "data/test.bin",
                "destination": "talker/data/test.bin",
                "sha256": digest,
            },
        ),
    )
    calls = []

    def fake_get(url: str, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(content=payload, raise_for_status=lambda: None)

    monkeypatch.setattr("scripts.ci.run_rocm_model_e2e.httpx.get", fake_get)

    result = _stage_auxiliary_assets(case, tmp_path)

    assert result is not None
    assert result["revision"] == MING_OMNI_SOURCE_REVISION
    assert result["assets"][0]["sha256"] == digest
    assert (tmp_path / "talker/data/test.bin").read_bytes() == payload
    assert calls[0][0].endswith(f"/{MING_OMNI_SOURCE_REVISION}/data/test.bin")

    reused = _stage_auxiliary_assets(case, tmp_path)
    assert reused is not None and reused["assets"][0]["reused"] is True
    assert len(calls) == 1
