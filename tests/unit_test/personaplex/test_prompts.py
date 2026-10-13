# SPDX-License-Identifier: Apache-2.0
"""Voice resolution and system tags, without weights."""

import errno
import tarfile

import pytest
import torch

from sglang_omni.models.personaplex import prompts
from sglang_omni.models.personaplex.architecture import TEXT_MARKER_IDS, TEXT_PAD_ID
from sglang_omni.models.personaplex.prompts import (
    decode_text,
    load_voice_prompt,
    resolve_voice_path,
    tokenize_text_prompt,
    wrap_system_tags,
)
from sglang_omni.models.personaplex.timeline import REFERENCE_CACHE_POSITIONS
from sglang_omni.serve.openai_errors import is_bad_request_error


class FakeTokenizer:
    def encode(self, text):
        return [len(word) for word in text.split()]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def test_system_tags_wrap_once():
    assert wrap_system_tags("  Be kind. ") == "<system> Be kind. <system>"
    assert wrap_system_tags("<system> x <system>") == "<system> x <system>"
    assert tokenize_text_prompt(FakeTokenizer(), "") == []
    assert tokenize_text_prompt(FakeTokenizer(), "Be kind") == [8, 2, 4, 8]


def test_decode_text_drops_frame_markers():
    ids = [TEXT_PAD_ID, 42, *sorted(TEXT_MARKER_IDS), 7]
    assert decode_text(FakeTokenizer(), ids) == "42 7"
    assert decode_text(FakeTokenizer(), [TEXT_PAD_ID]) == ""


def write_voices_archive(model_dir):
    voices = model_dir / "voices"
    voices.mkdir(parents=True)
    saved = {
        "embeddings": torch.randn(5, 1, 1, 8, dtype=torch.bfloat16),
        "cache": torch.full((1, 17, REFERENCE_CACHE_POSITIONS), 4, dtype=torch.long),
    }
    torch.save(saved, voices / "NATF2.pt")
    with tarfile.open(model_dir / "voices.tgz", "w:gz") as tar:
        tar.add(voices, arcname="voices")
    (voices / "NATF2.pt").unlink()
    voices.rmdir()


def test_voice_name_resolves_inside_the_packaged_archive(tmp_path):
    write_voices_archive(tmp_path)

    path = resolve_voice_path(tmp_path, "NATF2")
    assert path == tmp_path / "voices" / "NATF2.pt"
    prompt = load_voice_prompt(path, load_audio=None)
    assert prompt.frames == 6
    assert (
        prompt.embeddings.shape == (5, 8) and prompt.embeddings.dtype == torch.float32
    )
    assert prompt.tail_codes.shape == (2, 8)
    with pytest.raises(FileNotFoundError, match="packaged voices") as error:
        resolve_voice_path(tmp_path, "NOPE")
    assert is_bad_request_error(error.value)


def test_read_only_checkpoint_unpacks_voices_into_the_temp_dir_once(
    tmp_path, monkeypatch
):
    model_dir, temp_dir = tmp_path / "checkpoint", tmp_path / "tmp"
    temp_dir.mkdir()
    write_voices_archive(model_dir)
    make_staging = prompts.tempfile.mkdtemp
    unpacks = []

    def mkdtemp(prefix, dir):
        if dir == model_dir:
            raise OSError(errno.EROFS, "Read-only file system")
        unpacks.append(dir)
        return make_staging(prefix=prefix, dir=dir)

    monkeypatch.setattr(prompts.tempfile, "mkdtemp", mkdtemp)
    monkeypatch.setattr(prompts.tempfile, "gettempdir", lambda: str(temp_dir))

    path = resolve_voice_path(model_dir, "NATF2")
    assert path.parent.parent.parent == temp_dir
    assert path.is_file() and not (model_dir / "voices").exists()
    assert resolve_voice_path(model_dir, "NATF2") == path
    assert len(unpacks) == 1


def test_interrupted_unpack_leaves_no_partial_voices(tmp_path, monkeypatch):
    write_voices_archive(tmp_path)
    monkeypatch.setattr(prompts.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()

    def interrupted(self, path, **_):
        (prompts.Path(path) / "voices").mkdir()
        raise tarfile.ReadError("truncated archive")

    monkeypatch.setattr(tarfile.TarFile, "extractall", interrupted)
    with pytest.raises(tarfile.ReadError):
        prompts.voices_dir(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["tmp", "voices.tgz"]
