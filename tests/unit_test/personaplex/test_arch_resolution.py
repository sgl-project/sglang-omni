# SPDX-License-Identifier: Apache-2.0
"""A PersonaPlex checkpoint is recognised by its layout, locally or on the Hub."""

import httpx
import pytest
from huggingface_hub.errors import (
    LocalEntryNotFoundError,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
)
from huggingface_hub.utils import HFValidationError

from sglang_omni.config.manager import resolve_config_cls_for_model_path
from sglang_omni.models.personaplex.config import PersonaPlexPipelineConfig
from sglang_omni.utils.hf import (
    PERSONAPLEX_ARCHITECTURE,
    PERSONAPLEX_LAYOUT_MARKER,
    try_resolve_arch_from_layout_marker,
)

ARCH = "PersonaPlexForCausalLM"
NOT_FOUND = httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co"))


def resolve(model_path: str) -> str | None:
    return try_resolve_arch_from_layout_marker(
        model_path, PERSONAPLEX_LAYOUT_MARKER, PERSONAPLEX_ARCHITECTURE
    )


def test_local_tokenizer_marks_the_layout(tmp_path):
    assert resolve(str(tmp_path)) is None
    (tmp_path / "tokenizer_spm_32k_3.model").write_bytes(b"")
    assert resolve(str(tmp_path)) == ARCH


def test_local_checkpoint_selects_the_personaplex_pipeline(tmp_path):
    (tmp_path / "tokenizer_spm_32k_3.model").write_bytes(b"")
    assert resolve_config_cls_for_model_path(str(tmp_path)) is PersonaPlexPipelineConfig


def hub_download_raising(exc):
    def fake_hub_download(**_):
        raise exc

    return fake_hub_download


@pytest.mark.parametrize(
    "exc",
    [
        RemoteEntryNotFoundError("missing", response=NOT_FOUND),
        RepositoryNotFoundError("no repo", response=NOT_FOUND),
        LocalEntryNotFoundError("offline"),
        HFValidationError("bad repo id"),
    ],
)
def test_hub_lookup_misses_are_not_personaplex(monkeypatch, exc):
    monkeypatch.setattr(
        "sglang_omni.utils.hf.hf_hub_download", hub_download_raising(exc)
    )
    assert resolve("org/other-model") is None


def test_unexpected_hub_failures_propagate(monkeypatch):
    monkeypatch.setattr(
        "sglang_omni.utils.hf.hf_hub_download",
        hub_download_raising(RuntimeError("bug")),
    )
    with pytest.raises(RuntimeError, match="bug"):
        resolve("org/other-model")


def test_hub_marker_resolves(monkeypatch):
    monkeypatch.setattr(
        "sglang_omni.utils.hf.hf_hub_download", lambda **_: "/cache/marker"
    )
    assert resolve("nvidia/personaplex-7b-v1") == ARCH
