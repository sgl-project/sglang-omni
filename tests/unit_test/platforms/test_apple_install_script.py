# SPDX-License-Identifier: Apache-2.0
"""Exercise the Apple installer's checkout helper with real local Git remotes."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

INSTALLER = Path(__file__).resolve().parents[3] / "install.sh"
RELEASE = "v0.5.19"


def test_help_documents_bootstrap_and_override_controls():
    result = subprocess.run(
        ["bash", str(INSTALLER), "--help"],
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    assert "SGLANG_OMNI_EXTRAS" in result.stdout
    assert "qwen-tts" in INSTALLER.read_text()
    assert "SGLANG_OMNI_BOOTSTRAP_HOMEBREW=0" in result.stdout


def test_shared_apple_audio_formulas_are_installer_defaults():
    source = INSTALLER.read_text()
    assert "readonly APPLE_BREW_FORMULAS=(ffmpeg@7 sox uv)" in source
    assert "readonly APPLE_OMNI_EXTRAS=(fun-cosyvoice3)" in source
    assert "readonly APPLE_NO_DEPS_PACKAGES=(qwen-tts==0.1.1 einops)" in source


def git(repo: Path, *args: str, check: bool = True):
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=Installer Test",
            "-c",
            "user.email=installer-test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "tag.gpgsign=false",
            "-C",
            str(repo),
            *args,
        ],
        text=True,
        capture_output=True,
        check=check,
        timeout=15,
    )


def commit(repo: Path, text: str) -> str:
    (repo / "source.txt").write_text(text)
    git(repo, "add", "source.txt")
    git(repo, "commit", "-qm", text)
    return git(repo, "rev-parse", "HEAD").stdout.strip()


@pytest.fixture
def remote(tmp_path: Path, request):
    repo = tmp_path / "remote repo"
    repo.mkdir()
    git(repo, "init", "-q", "--initial-branch=main")
    commit(repo, "old source")
    git(repo, "tag", "v0.1.0")
    release_commit = commit(repo, "release source")
    if getattr(request, "param", "lightweight") == "annotated":
        git(repo, "tag", "-a", RELEASE, "-m", "Release")
    else:
        git(repo, "tag", RELEASE)
    main_commit = commit(repo, "development source")
    return repo, release_commit, main_commit


def checkout(destination: Path, ref: str, remote: Path):
    # Source only the checkout functions, so real Git is exercised without
    # running the macOS, Homebrew, Python, or model-runtime installation steps.
    functions = re.findall(
        r"^(?:fetch_checkout_ref|clone_or_reuse)\(\) \{\n.*?^\}",
        INSTALLER.read_text(),
        flags=re.MULTILINE | re.DOTALL,
    )
    script = "\n".join(
        [
            "set -Eeuo pipefail",
            "log() { :; }",
            'die() { printf "%s\\n" "$*" >&2; exit 1; }',
            *functions,
            'clone_or_reuse "$1" "$2" "$3" "test checkout"',
        ]
    )
    return subprocess.run(
        [
            "bash",
            "-c",
            script,
            "installer-test",
            str(destination),
            ref,
            remote.as_uri(),
        ],
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )


def assert_release(destination: Path, expected_commit: str):
    assert git(destination, "rev-parse", "HEAD").stdout.strip() == expected_commit
    assert (
        git(destination, "describe", "--tags", "--exact-match").stdout.strip()
        == RELEASE
    )
    assert git(destination, "rev-list", "--count", "HEAD").stdout.strip() == "1"
    assert git(destination, "status", "--porcelain").stdout == ""
    assert git(destination, "symbolic-ref", "-q", "HEAD", check=False).returncode == 1
    assert git(destination, "tag", "--list", "v0.1.0").stdout == ""


@pytest.mark.parametrize("remote", ["lightweight", "annotated"], indirect=True)
@pytest.mark.parametrize("ref", [RELEASE, f"refs/tags/{RELEASE}"])
def test_fresh_checkout_preserves_selected_release_tag(tmp_path, remote, ref):
    source, release_commit, _ = remote
    destination = tmp_path / "fresh checkout"
    checkout(destination, ref, source)
    assert_release(destination, release_commit)


@pytest.mark.parametrize("remote", ["lightweight", "annotated"], indirect=True)
@pytest.mark.parametrize("cached_ref", [RELEASE, "main"])
def test_reused_checkout_recovers_missing_tag(tmp_path, remote, cached_ref):
    source, release_commit, _ = remote
    destination = tmp_path / "cached checkout"
    destination.mkdir()
    git(destination, "init", "-q")
    git(destination, "remote", "add", "origin", source.as_uri())
    git(destination, "fetch", "--depth", "1", "--no-tags", "origin", cached_ref)
    git(destination, "checkout", "--detach", "--quiet", "FETCH_HEAD")
    assert git(destination, "tag", "--list").stdout == ""

    checkout(destination, RELEASE, source)
    assert_release(destination, release_commit)
    checkout(destination, RELEASE, source)
    assert_release(destination, release_commit)


@pytest.mark.parametrize("ref_kind", ["branch", "qualified_branch", "commit"])
def test_branch_and_commit_overrides_remain_untagged(tmp_path, remote, ref_kind):
    source, _, main_commit = remote
    ref = {
        "branch": "main",
        "qualified_branch": "refs/heads/main",
        "commit": main_commit,
    }[ref_kind]
    destination = tmp_path / "development checkout"
    checkout(destination, ref, source)
    assert git(destination, "rev-parse", "HEAD").stdout.strip() == main_commit
    assert git(destination, "tag", "--list").stdout == ""


def test_reused_branch_tracks_new_remote_commit(tmp_path, remote):
    source, _, _ = remote
    destination = tmp_path / "development checkout"
    checkout(destination, "main", source)
    updated = commit(source, "updated development source")
    checkout(destination, "main", source)
    assert git(destination, "rev-parse", "HEAD").stdout.strip() == updated
    assert (destination / "source.txt").read_text() == "updated development source"


def test_branch_ignores_suffix_only_tag_match(tmp_path, remote):
    """A tag named refs/tags/main must not redirect a main-branch checkout."""
    source, release_commit, main_commit = remote
    git(source, "tag", "refs/tags/main", release_commit)
    destination = tmp_path / "branch checkout"
    checkout(destination, "main", source)
    assert git(destination, "rev-parse", "HEAD").stdout.strip() == main_commit
    assert git(destination, "tag", "--list").stdout == ""
