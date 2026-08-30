# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.metadata
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from sglang_omni.models.dots_tts.compat import import_dots_tts

CHECKING_PACKAGE = """
import importlib.metadata as _md

torch_version = _md.version("torch")
torchaudio_version = _md.version("torchaudio")
if torch_version.split(".")[:2] != torchaudio_version.split(".")[:2]:
    raise RuntimeError(
        f"torch ({torch_version}) and torchaudio ({torchaudio_version}) "
        "minor versions do not match."
    )
"""


@pytest.fixture
def mismatched_stack(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    versions = {"torch": "2.13.0", "torchaudio": "2.11.0"}
    monkeypatch.setattr(importlib.metadata, "version", lambda name: versions[name])
    package = tmp_path / "dots_tts"
    package.mkdir()
    (package / "__init__.py").write_text(CHECKING_PACKAGE)
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [
        m for m in sys.modules if m == "dots_tts" or m.startswith("dots_tts.")
    ]:
        monkeypatch.delitem(sys.modules, name)
    yield package
    sys.modules.pop("dots_tts", None)


def test_import_dots_tts_passes_the_package_check_on_a_mismatched_stack(
    mismatched_stack: Path,
) -> None:
    module = import_dots_tts()

    assert module.torch_version == "2.13.0"
    assert module.torchaudio_version == "2.13.0"
    assert sys.modules["dots_tts"] is module
    assert import_dots_tts() is module
    assert importlib.metadata.version("torchaudio") == "2.11.0"


def test_import_dots_tts_restores_the_reader_when_the_package_fails(
    mismatched_stack: Path,
) -> None:
    (mismatched_stack / "__init__.py").write_text("raise ImportError('broken package')")

    with pytest.raises(ImportError, match="broken package"):
        import_dots_tts()

    assert "dots_tts" not in sys.modules
    assert importlib.metadata.version("torchaudio") == "2.11.0"
