# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import huggingface_hub
import yaml

from sglang_omni.utils.checkpoint import resolve_checkpoint

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_resolve_checkpoint_returns_local_directory(tmp_path) -> None:
    assert resolve_checkpoint(str(tmp_path)) == str(tmp_path)


def test_resolve_checkpoint_downloads_latest_without_pin(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_snapshot_download(repo_id, revision=None):
        captured.update({"repo_id": repo_id, "revision": revision})
        return "/snapshots/latest"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    assert resolve_checkpoint("org/model") == "/snapshots/latest"
    assert captured == {"repo_id": "org/model", "revision": None}


def test_resolve_checkpoint_honors_pinned_revision(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_snapshot_download(repo_id, revision=None):
        captured.update({"repo_id": repo_id, "revision": revision})
        return "/snapshots/pinned"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    assert resolve_checkpoint("org/model@abc123") == "/snapshots/pinned"
    assert captured == {"repo_id": "org/model", "revision": "abc123"}


def test_pinned_spec_resolves_architecture_without_snapshot(
    monkeypatch, tmp_path
) -> None:
    from types import SimpleNamespace

    from sglang_omni.config import manager
    from sglang_omni.models.dots_tts.config import DotsTTSPipelineConfig

    def fail_snapshot(*args, **kwargs):
        raise AssertionError("architecture discovery must not download weights")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail_snapshot)
    captured: dict[str, object] = {}

    def fake_auto_config(path, revision=None, **kwargs):
        captured["auto"] = (path, revision)
        raise OSError("metadata offline in this test")

    monkeypatch.setattr(
        manager,
        "AutoConfig",
        SimpleNamespace(from_pretrained=fake_auto_config),
    )

    def fake_hub_download(repo_id, filename, revision=None, **kwargs):
        captured["raw"] = (repo_id, filename, revision)
        path = tmp_path / filename
        path.write_text('{"architectures": ["DotsTTSForConditionalGeneration"]}')
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hub_download)

    config_cls = manager.resolve_config_cls_for_model_path(
        "dots-studio/dots.tts-mf@c28105adc8228143392b4e346994ff613ee48a06"
    )

    assert config_cls is DotsTTSPipelineConfig
    revision = "c28105adc8228143392b4e346994ff613ee48a06"
    assert captured["auto"] == ("dots-studio/dots.tts-mf", revision)
    assert captured["raw"] == ("dots-studio/dots.tts-mf", "config.json", revision)


def test_local_model_path_skips_snapshot_resolution(monkeypatch, tmp_path) -> None:
    from sglang_omni.config import manager
    from sglang_omni.models.dots_tts.config import DotsTTSPipelineConfig

    def fail_snapshot(*args, **kwargs):
        raise AssertionError("local paths must not resolve a snapshot")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail_snapshot)
    (tmp_path / "config.json").write_text(
        '{"architectures": ["DotsTTSForConditionalGeneration"]}'
    )

    config_cls = manager.resolve_config_cls_for_model_path(str(tmp_path))

    assert config_cls is DotsTTSPipelineConfig


def test_dots_tts_canonical_config_pins_snapshot_revision() -> None:
    config = yaml.safe_load(
        (_REPO_ROOT / "examples" / "configs" / "dots_tts.yaml").read_text()
    )
    repo_id, _, revision = str(config["model_path"]).partition("@")

    assert repo_id == "dots-studio/dots.tts-mf"
    assert len(revision) == 40
    assert all(c in "0123456789abcdef" for c in revision)
