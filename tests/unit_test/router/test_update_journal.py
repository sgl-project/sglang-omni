from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path

import pytest

from sglang_omni_router.python.update_journal import (
    STATE_DIR_ENV,
    JournalUnreadableError,
    JournalUnwritableError,
    UpdateJournal,
    build_journal,
    default_journal_path,
    ensure_state_dir,
    resolve_state_dir,
)


def _isolate_home(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    monkeypatch.setattr(
        os.path, "expanduser", lambda path: path.replace("~", str(home), 1)
    )


def test_begin_keep_clear_round_trip(tmp_path: Path) -> None:
    journal = UpdateJournal(str(tmp_path / "j.json"))
    assert journal.pending() == []
    journal.begin("/update_weights_from_disk", ["w1", "w0"])
    assert journal.pending() == ["w0", "w1"]
    assert journal.has_pending() is True
    journal.keep(["w0"])
    assert journal.pending() == ["w0"]
    journal.keep([])
    assert journal.pending() == []
    assert not (tmp_path / "j.json").exists()


def test_unreadable_journal_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "j.json"
    path.write_text("{ this is not valid json", encoding="utf-8")
    journal = UpdateJournal(str(path))
    with pytest.raises(JournalUnreadableError):
        journal.pending()
    # Note (Jiaxin Deng): has_pending must fail closed (assume a transaction is in
    # progress)
    assert journal.has_pending() is True
    # Note (Jiaxin Deng): discard must not silently clobber an unreadable journal
    journal.discard("w0")
    assert path.exists()


@pytest.mark.parametrize("payload", [[], {"worker_ids": [1]}])
def test_malformed_journal_schema_fails_closed(tmp_path: Path, payload) -> None:
    path = tmp_path / "j.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    journal = UpdateJournal(str(path))
    with pytest.raises(JournalUnreadableError):
        journal.pending()
    assert journal.has_pending() is True


def test_keep_retains_the_recorded_admin_path(tmp_path: Path) -> None:
    path = tmp_path / "j.json"
    journal = UpdateJournal(str(path))
    journal.begin("/update_weights_from_disk", ["w0", "w1"])
    journal.keep(["w1"])
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document == {
        "path": "/update_weights_from_disk",
        "worker_ids": ["w1"],
    }
    journal.discard("w1")
    assert journal.pending() == []


def test_discard_removes_one_entry(tmp_path: Path) -> None:
    journal = UpdateJournal(str(tmp_path / "j.json"))
    journal.begin("/x", ["w0", "w1"])
    journal.discard("w0")
    assert journal.pending() == ["w1"]
    journal.discard("w1")
    assert journal.pending() == []


def test_begin_fails_closed_when_the_write_is_not_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Note (Jiaxin Deng): a swallowed fsync error would let the update run with no
    # recovery record
    journal = UpdateJournal(str(tmp_path / "j.json"))

    def _fail(fd: int) -> None:
        raise OSError("fsync failed")

    monkeypatch.setattr(os, "fsync", _fail)
    with pytest.raises(JournalUnwritableError):
        journal.begin("/update_weights_from_disk", ["w0"])


@pytest.mark.skipif(os.name != "posix", reason="directory fsync is POSIX only")
def test_begin_fails_closed_when_the_directory_fsync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Note (Jiaxin Deng): the rename is durable only once the parent directory entry is
    # synced
    journal = UpdateJournal(str(tmp_path / "j.json"))
    real_fsync = os.fsync

    def _fail_on_directory(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("directory fsync failed")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _fail_on_directory)
    with pytest.raises(JournalUnwritableError):
        journal.begin("/update_weights_from_disk", ["w0"])


def test_discard_reports_failure_when_the_rewrite_is_not_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = UpdateJournal(str(tmp_path / "j.json"))
    journal.begin("/x", ["w0", "w1"])

    def _fail(fd: int) -> None:
        raise OSError("fsync failed")

    monkeypatch.setattr(os, "fsync", _fail)
    assert journal.discard("w0") is False


def test_begin_fails_closed_when_the_rename_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = UpdateJournal(str(tmp_path / "j.json"))

    def _fail(src: str, dst: str) -> None:
        raise OSError("rename failed")

    monkeypatch.setattr(os, "replace", _fail)
    with pytest.raises(JournalUnwritableError):
        journal.begin("/update_weights_from_disk", ["w0"])


def test_default_path_is_stable_by_endpoint() -> None:
    a = default_journal_path("0.0.0.0", 8000)
    b = default_journal_path("0.0.0.0", 8000)
    c = default_journal_path("0.0.0.0", 8001)
    assert a == b  # same endpoint -> same path across supervisor restarts
    assert a != c
    assert "8000" in a


def test_default_path_never_uses_a_temp_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Note (Jiaxin Deng): a temp directory survives a process restart but not a host
    # reboot, and the remote workers outlive the router host
    system_tmp = tempfile.gettempdir()
    sentinel = str(tmp_path / "temp-dir-sentinel")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: sentinel)
    monkeypatch.delenv(STATE_DIR_ENV, raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)

    path = default_journal_path("0.0.0.0", 8000)

    assert sentinel not in path  # the temp directory is never consulted
    assert not path.startswith(system_tmp)
    assert os.path.join(".local", "state", "sglang-omni-router") in path


def test_state_dir_resolution_priority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    _isolate_home(monkeypatch, home)
    monkeypatch.setenv(STATE_DIR_ENV, str(tmp_path / "env"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))

    assert resolve_state_dir(str(tmp_path / "explicit")) == str(tmp_path / "explicit")
    assert resolve_state_dir() == str(tmp_path / "env")

    monkeypatch.delenv(STATE_DIR_ENV)
    assert resolve_state_dir() == str(tmp_path / "xdg" / "sglang-omni-router")

    monkeypatch.delenv("XDG_STATE_HOME")
    assert resolve_state_dir() == str(home / ".local" / "state" / "sglang-omni-router")


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "127.0.0.1", "::1", "::", "router.internal", "fe80::1%eth0"],
)
def test_endpoint_key_is_one_safe_path_component(host: str, tmp_path: Path) -> None:
    path = default_journal_path(host, 8000, str(tmp_path))
    key = os.path.relpath(path, tmp_path).split(os.sep)[0]
    assert key not in {"", ".", ".."}
    assert not any(char in key for char in "/\\:%")


def test_distinct_endpoints_get_distinct_journals(tmp_path: Path) -> None:
    endpoints = [
        ("0.0.0.0", 8000),
        ("0.0.0.0", 8001),
        ("127.0.0.1", 8000),
        ("::1", 8000),
        ("::", 8000),
        ("router.internal", 8000),
    ]
    paths = {
        default_journal_path(host, port, str(tmp_path)) for host, port in endpoints
    }
    assert len(paths) == len(endpoints)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_created_state_directories_and_journal_are_owner_only(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    path = Path(default_journal_path("0.0.0.0", 8000, str(state_dir)))
    UpdateJournal(str(path)).begin("/update_weights_from_disk", ["w0"])

    assert stat.S_IMODE(state_dir.stat().st_mode) & 0o077 == 0
    assert stat.S_IMODE(path.parent.stat().st_mode) & 0o077 == 0
    assert stat.S_IMODE(path.stat().st_mode) & 0o177 == 0


@pytest.mark.skipif(os.name != "posix", reason="directory fsync is POSIX only")
def test_a_newly_created_endpoint_directory_is_made_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Note (Jiaxin Deng): syncing only the journal's own parent leaves the new
    # endpoint directory's entry in the state dir unsynced, so a host crash can
    # drop the whole directory and take the journal with it.
    state_dir = tmp_path / "state"
    journal = UpdateJournal(default_journal_path("0.0.0.0", 8000, str(state_dir)))
    synced: list[int] = []
    real_fsync = os.fsync

    def _record(fd: int) -> None:
        info = os.fstat(fd)
        if stat.S_ISDIR(info.st_mode):
            synced.append(info.st_ino)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _record)
    journal.begin("/update_weights_from_disk", ["w0"])

    assert state_dir.stat().st_ino in synced
    assert Path(journal.path).parent.stat().st_ino in synced

    # Note (Jiaxin Deng): only on creation; a rewrite must not pay for it.
    synced.clear()
    journal.keep(["w0"])
    assert state_dir.stat().st_ino not in synced


def test_ensure_state_dir_fails_closed_when_it_cannot_be_created(
    tmp_path: Path,
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    unusable = str(blocker / "state")

    with pytest.raises(JournalUnwritableError) as excinfo:
        ensure_state_dir(unusable)

    assert unusable in str(excinfo.value)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores mode bits"
)
def test_ensure_state_dir_fails_closed_when_it_is_not_writable(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    os.chmod(state_dir, 0o500)
    try:
        with pytest.raises(JournalUnwritableError) as excinfo:
            ensure_state_dir(str(state_dir))
    finally:
        os.chmod(state_dir, 0o700)
    assert str(state_dir) in str(excinfo.value)


def test_ensure_state_dir_survives_a_concurrently_removed_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Note (Jiaxin Deng): routers on different ports share one state directory; one
    # cleaning up first must not make the other refuse to start
    state_dir = tmp_path / "state"

    def _vanished(path: str) -> None:
        raise FileNotFoundError(path)

    monkeypatch.setattr(os, "unlink", _vanished)
    assert ensure_state_dir(str(state_dir)) == str(state_dir)


def test_an_unusable_state_dir_does_not_fall_back_to_a_temp_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Note (Jiaxin Deng): falling back would look durable while a reboot silently drops
    # the record
    fake_tmp = tmp_path / "tempdir"
    fake_tmp.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_tmp))
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    journal = UpdateJournal(
        default_journal_path("0.0.0.0", 8000, str(blocker / "state"))
    )

    with pytest.raises(JournalUnwritableError):
        journal.begin("/update_weights_from_disk", ["w0"])

    assert list(fake_tmp.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="directory fsync is POSIX only")
def test_startup_makes_the_state_directory_it_creates_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Note (Jiaxin Deng): the journal writer only syncs from the endpoint
    # directory down, so a state directory created at startup would keep an
    # unsynced entry in its own parent and could vanish with the journal.
    state_dir = tmp_path / "state"
    synced: list[int] = []
    real_fsync = os.fsync

    def _record(fd: int) -> None:
        info = os.fstat(fd)
        if stat.S_ISDIR(info.st_mode):
            synced.append(info.st_ino)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _record)
    ensure_state_dir(str(state_dir))

    assert tmp_path.stat().st_ino in synced


@pytest.mark.skipif(os.name != "posix", reason="directory fsync is POSIX only")
def test_clearing_an_already_absent_journal_still_syncs_the_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Note (Jiaxin Deng): an earlier unlink may have failed its directory
    # sync, so absence is not durability; a retry that skipped the sync would
    # report a resolution a crash can undo.
    journal = UpdateJournal(str(tmp_path / "j.json"))
    journal.begin("/x", ["w0"])
    journal.clear()
    synced: list[int] = []
    real_fsync = os.fsync

    def _record(fd: int) -> None:
        info = os.fstat(fd)
        if stat.S_ISDIR(info.st_mode):
            synced.append(info.st_ino)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _record)
    journal.clear()

    assert tmp_path.stat().st_ino in synced


def test_an_unresolvable_home_fails_closed_instead_of_using_a_relative_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Note (Jiaxin Deng): expanduser returns "~" unchanged under an injected
    # uid with no passwd entry, which would put the journal under the cwd.
    monkeypatch.delenv(STATE_DIR_ENV, raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setattr(os.path, "expanduser", lambda path: path)

    with pytest.raises(JournalUnwritableError) as excinfo:
        resolve_state_dir()

    assert "--router-state-dir" in str(excinfo.value)


def test_build_journal_refuses_updates_when_it_has_no_durable_home(
    tmp_path: Path,
) -> None:
    # Note (Jiaxin Deng): a router that never updates weights must still start,
    # so reads report no transaction; an update that cannot record its targets
    # must not run.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    journal = build_journal("0.0.0.0", 8000, str(blocker / "state"))

    assert journal.pending() == []
    assert journal.has_pending() is False
    assert journal.discard("w0") is True
    with pytest.raises(JournalUnwritableError):
        journal.begin("/update_weights_from_disk", ["w0"])
