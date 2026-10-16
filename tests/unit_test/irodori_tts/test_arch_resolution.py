# SPDX-License-Identifier: Apache-2.0
"""Architecture discovery for local and Hub Irodori checkpoints."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang_omni.config import manager
from sglang_omni.config.manager import resolve_config_cls_for_model_path
from sglang_omni.models.irodori_tts.config import IrodoriTTSPipelineConfig
from sglang_omni.utils.hf import (
    IRODORI_ARCHITECTURE,
    try_resolve_arch_from_irodori_layout,
)


def write_model_card(path: Path, title: str) -> None:
    path.write_text(f"# {title}\n", encoding="utf-8")


def test_local_model_card_resolves_irodori_pipeline(tmp_path: Path) -> None:
    write_model_card(tmp_path / "README.md", "Irodori-TTS-v4.1-Small")
    (tmp_path / "model.safetensors").touch()

    assert resolve_config_cls_for_model_path(str(tmp_path)) is IrodoriTTSPipelineConfig
    assert try_resolve_arch_from_irodori_layout(str(tmp_path)) == IRODORI_ARCHITECTURE


def test_hub_model_id_resolves_irodori_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_card_path = tmp_path / "README.md"
    write_model_card(model_card_path, "Irodori-TTS-v4.1-Small")

    def fail_auto_config(repo_id: str, **kwargs: str) -> None:
        raise OSError("Irodori model card does not include config.json")

    def no_architecture(repo_id: str, revision: str | None = None) -> str | None:
        return None

    monkeypatch.setattr(
        manager, "AutoConfig", SimpleNamespace(from_pretrained=fail_auto_config)
    )
    for resolver_name in (
        "try_resolve_arch_from_raw_config",
        "try_resolve_arch_from_mistral_config",
        "try_resolve_arch_from_nemo_config",
        "try_resolve_arch_from_cosyvoice3_layout",
        "try_resolve_arch_from_auk_layout",
    ):
        monkeypatch.setattr(manager, resolver_name, no_architecture)

    downloaded_files: dict[str, str | None] = {}

    def fake_hub_download(
        repo_id: str, filename: str, revision: str | None = None
    ) -> str:
        downloaded_files.update(
            {"repo_id": repo_id, "filename": filename, "revision": revision}
        )
        return str(model_card_path)

    monkeypatch.setattr("sglang_omni.utils.hf.hf_hub_download", fake_hub_download)

    model_path = "Aratako/Irodori-TTS-v4.1-Small"
    assert resolve_config_cls_for_model_path(model_path) is IrodoriTTSPipelineConfig
    assert downloaded_files == {
        "repo_id": model_path,
        "filename": "README.md",
        "revision": None,
    }


def test_direct_local_checkpoint_uses_sibling_model_card(tmp_path: Path) -> None:
    from sglang_omni.models.irodori_tts.model_loader import checkpoint_file

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    write_model_card(tmp_path / "README.md", "Irodori-TTS-v4.1-Small")

    assert try_resolve_arch_from_irodori_layout(str(checkpoint)) == IRODORI_ARCHITECTURE
    assert (
        resolve_config_cls_for_model_path(str(checkpoint)) is IrodoriTTSPipelineConfig
    )
    assert checkpoint_file(str(checkpoint)) == checkpoint.resolve()


def test_checkpoint_file_uses_shared_hub_checkpoint_resolver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sglang_omni.models.irodori_tts.model_loader import checkpoint_file

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    checkpoint = snapshot / "model.safetensors"
    checkpoint.touch()
    resolved_model_ids: list[str] = []

    def fake_resolve_checkpoint(model_path: str) -> str:
        resolved_model_ids.append(model_path)
        return str(snapshot)

    monkeypatch.setattr(
        "sglang_omni.models.irodori_tts.model_loader.resolve_checkpoint",
        fake_resolve_checkpoint,
    )

    assert checkpoint_file("Aratako/Irodori-TTS-v4.1-Small") == checkpoint.resolve()
    assert resolved_model_ids == ["Aratako/Irodori-TTS-v4.1-Small"]


def test_unrelated_local_model_card_does_not_resolve_as_irodori(
    tmp_path: Path,
) -> None:
    write_model_card(tmp_path / "README.md", "Some Other Model")
    (tmp_path / "model.safetensors").touch()

    assert try_resolve_arch_from_irodori_layout(str(tmp_path)) is None
    with pytest.raises(ValueError, match="Could not resolve model architecture"):
        resolve_config_cls_for_model_path(str(tmp_path))


def test_runtime_loader_reads_checkpoint_metadata(tmp_path: Path) -> None:
    import json

    import torch
    from safetensors.torch import save_file

    from sglang_omni.models.irodori_tts.model_loader import checkpoint_configuration

    checkpoint = tmp_path / "model.safetensors"
    model_configuration = {"latent_dim": 32}
    text_configuration = {"model_type": "modernbert"}
    save_file(
        {"weight": torch.zeros(1)},
        str(checkpoint),
        metadata={
            "config_json": json.dumps(model_configuration),
            "text_encoder_config_json": json.dumps(text_configuration),
        },
    )

    assert checkpoint_configuration(checkpoint) == (
        model_configuration,
        text_configuration,
    )
