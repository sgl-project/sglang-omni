# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import sys

import pytest

_FLAG = "SGLANG_OMNI_USE_MODELSCOPE"
_HUB_MODULE = "sglang_omni.utils.hub"


@pytest.fixture(autouse=True)
def _restore_hub_module():
    previous = sys.modules.pop(_HUB_MODULE, None)
    try:
        yield
    finally:
        sys.modules.pop(_HUB_MODULE, None)
        if previous is not None:
            sys.modules[_HUB_MODULE] = previous


def _reimport_hub(monkeypatch, flag: str | None):
    """Re-import the hub wrapper so import-time backend selection runs fresh."""
    if flag is None:
        monkeypatch.delenv(_FLAG, raising=False)
    else:
        monkeypatch.setenv(_FLAG, flag)
    sys.modules.pop(_HUB_MODULE, None)
    return importlib.import_module(_HUB_MODULE)


def test_huggingface_backend_dispatches_to_hf_helpers(monkeypatch) -> None:
    hub = _reimport_hub(monkeypatch, None)
    calls: list[tuple[str, tuple, dict]] = []

    def fake_snapshot_download(*args, **kwargs):
        calls.append(("snapshot_download", args, kwargs))
        return "/hf/snapshot"

    def fake_hf_hub_download(*args, **kwargs):
        calls.append(("hf_hub_download", args, kwargs))
        return "/hf/config.json"

    def fake_cached_file(*args, **kwargs):
        calls.append(("cached_file", args, kwargs))
        return "/hf/cached/config.json"

    monkeypatch.setattr(hub, "_snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(hub, "_hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(hub, "_cached_file", fake_cached_file)

    assert hub.AutoConfig.__module__.startswith("transformers")
    assert hub.snapshot_download("repo/model", local_files_only=True) == "/hf/snapshot"
    assert (
        hub.hf_hub_download("repo/model", "config.json", revision="main")
        == "/hf/config.json"
    )
    assert (
        hub.cached_file("repo/model", "config.json", local_files_only=True)
        == "/hf/cached/config.json"
    )
    assert calls == [
        (
            "snapshot_download",
            ("repo/model",),
            {"local_files_only": True},
        ),
        (
            "hf_hub_download",
            (),
            {
                "repo_id": "repo/model",
                "filename": "config.json",
                "revision": "main",
            },
        ),
        (
            "cached_file",
            ("repo/model", "config.json"),
            {"local_files_only": True},
        ),
    ]


def test_modelscope_backend_dispatches_to_modelscope_helpers(monkeypatch) -> None:
    hub = _reimport_hub(monkeypatch, "true")
    calls: list[tuple[str, tuple, dict]] = []

    def fake_snapshot_download(*args, **kwargs):
        calls.append(("snapshot_download", args, kwargs))
        return "/modelscope/snapshot"

    def fake_model_file_download(*args, **kwargs):
        calls.append(("model_file_download", args, kwargs))
        return "/modelscope/config.json"

    monkeypatch.setattr(hub, "_snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(hub, "_model_file_download", fake_model_file_download)

    assert hub.AutoConfig.__module__.startswith("modelscope")
    assert (
        hub.snapshot_download("repo/model", local_files_only=True, force_download=True)
        == "/modelscope/snapshot"
    )
    assert (
        hub.hf_hub_download("repo/model", "config.json", revision="master")
        == "/modelscope/config.json"
    )
    assert (
        hub.cached_file("repo/model", "tokenizer.json", local_files_only=True)
        == "/modelscope/config.json"
    )
    assert calls == [
        (
            "snapshot_download",
            ("repo/model",),
            {"local_files_only": True},
        ),
        (
            "model_file_download",
            (),
            {
                "model_id": "repo/model",
                "file_path": "config.json",
                "revision": "master",
            },
        ),
        (
            "model_file_download",
            (),
            {
                "model_id": "repo/model",
                "file_path": "tokenizer.json",
                "local_files_only": True,
            },
        ),
    ]
