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

from sglang_omni.utils.hf import try_resolve_arch_from_personaplex_layout

ARCH = "PersonaPlexForCausalLM"
NOT_FOUND = httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co"))


def test_local_tokenizer_marks_the_layout(tmp_path):
    assert try_resolve_arch_from_personaplex_layout(str(tmp_path)) is None
    (tmp_path / "tokenizer_spm_32k_3.model").write_bytes(b"")
    assert try_resolve_arch_from_personaplex_layout(str(tmp_path)) == ARCH


def _raise(exc):
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
    monkeypatch.setattr("sglang_omni.utils.hf.hf_hub_download", _raise(exc))
    assert try_resolve_arch_from_personaplex_layout("org/other-model") is None


def test_unexpected_hub_failures_propagate(monkeypatch):
    monkeypatch.setattr(
        "sglang_omni.utils.hf.hf_hub_download", _raise(RuntimeError("bug"))
    )
    with pytest.raises(RuntimeError, match="bug"):
        try_resolve_arch_from_personaplex_layout("org/other-model")


def test_hub_marker_resolves(monkeypatch):
    monkeypatch.setattr(
        "sglang_omni.utils.hf.hf_hub_download", lambda **_: "/cache/marker"
    )
    assert try_resolve_arch_from_personaplex_layout("nvidia/personaplex-7b-v1") == ARCH
