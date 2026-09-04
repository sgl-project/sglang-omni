# SPDX-License-Identifier: Apache-2.0
"""Weight-update transaction journal (CP crash recovery).

A weight update is a distributed transaction: targets are disabled, the
barrier waits for the DPs, the broadcast runs, disabled state is restored. A
CP crash mid-transaction loses the in-memory registry, and a freshly built
replacement would re-enable workers whose weight versions may now differ.

The journal makes that failure closed. The target set is fsynced to a stable
per-endpoint path (keyed by host:port, so it survives a full supervisor
restart) before disabling, and cleared only on a known outcome: the broadcast
completed, or the ACK barrier aborted before anything was sent. A surviving
entry keeps its workers disabled until an operator verifies weight versions
and re-enables them (an admin-authenticated PUT /workers {"disabled": false}),
which discards the entry.

That path lives under a durable state directory, never a temp directory: the
workers are remote and outlive the router host, so a reboot that wiped the
journal would let a fresh router re-enable a pool whose weights are unknown.
An unusable state directory is an error rather than a temp-directory fallback,
which would look durable while silently losing the record.

Read and write both fail closed: an unreadable journal counts as "a
transaction may be in progress", and a write that could not be made durable
(including the parent directory fsync) raises rather than returning. Non-POSIX
hosts expose no directory handle to sync, so there durability is file-level
only (the multi-process router is Linux-only in any case).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re

logger = logging.getLogger(__name__)

STATE_DIR_ENV = "SGLANG_OMNI_ROUTER_STATE_DIR"
_STATE_DIR_NAME = "sglang-omni-router"
_STATE_DIR_MODE = 0o700
_JOURNAL_FILE_MODE = 0o600
_UNSAFE_KEY_CHARS = re.compile(r"[^A-Za-z0-9._-]")
# Note (Jiaxin Deng): these paths are ours alone, so a symlink sitting on one
# is an attempt to redirect the write, never a legitimate setup.
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_MAX_READABLE_KEY = 48


class JournalUnreadableError(Exception):
    """The journal exists but could not be read; recovery must fail closed."""


class JournalUnwritableError(Exception):
    """The journal could not be made durable; the caller must fail closed.

    Reporting a write that never reached the disk would let a weight update
    run with no crash-recovery record, which is the mixed-weight re-enable
    this journal exists to prevent.
    """


class UnavailableJournal:
    """Stand-in when no durable state directory could be resolved.

    A router that never updates weights must still start (the single-process
    relay predates the journal entirely), but an update with nowhere to record
    its target set must not run. So reads report no transaction and the write
    that would begin one refuses.
    """

    def __init__(self, reason: str) -> None:
        self._reason = reason

    @property
    def path(self) -> str:
        return f"<no durable state directory: {self._reason}>"

    def exists(self) -> bool:
        return False

    def pending(self) -> list[str]:
        return []

    def has_pending(self) -> bool:
        return False

    def begin(self, path: str, worker_ids: list[str]) -> None:
        raise JournalUnwritableError(self._reason)

    def keep(self, worker_ids: list[str]) -> None:
        raise JournalUnwritableError(self._reason)

    def discard(self, worker_id: str) -> bool:
        return True

    def clear(self) -> None:
        return None


class UpdateJournal:
    def __init__(self, path: str) -> None:
        self._path = path

    @property
    def path(self) -> str:
        return self._path

    def exists(self) -> bool:
        return os.path.exists(self._path)

    def _read_document(self) -> dict | None:
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, NotADirectoryError):
            # Note (Jiaxin Deng): POSIX reports ENOTDIR when an ancestor is a
            # regular file. No journal can exist under such a path, so treating
            # it as unreadable would fail closed and disable the pool for a
            # router that never wrote one.
            return None
        except (OSError, ValueError) as exc:
            raise JournalUnreadableError(str(exc)) from exc
        if not isinstance(data, dict):
            raise JournalUnreadableError("journal is not an object")
        return data

    def pending(self) -> list[str]:
        """Journaled target worker ids, or [] if there is no transaction.

        Raises JournalUnreadableError if the file is present but corrupt, so
        callers can fail closed instead of treating it as "no transaction".
        """
        data = self._read_document()
        if data is None:
            return []
        workers = data.get("worker_ids")
        if not isinstance(workers, list) or not all(
            isinstance(worker_id, str) for worker_id in workers
        ):
            raise JournalUnreadableError("worker_ids is not a list of strings")
        return workers

    def has_pending(self) -> bool:
        try:
            return bool(self.pending())
        except JournalUnreadableError:
            # Note (Jiaxin Deng): unreadable = assume a transaction is in
            # progress (fail closed).
            return True

    def begin(self, path: str, worker_ids: list[str]) -> None:
        self._write({"path": path, "worker_ids": sorted(worker_ids)})

    def keep(self, worker_ids: list[str]) -> None:
        """Persist the still-unresolved target set (empty clears)."""
        if not worker_ids:
            self.clear()
            return
        document: dict = {"worker_ids": sorted(worker_ids)}
        try:
            existing = self._read_document()
        except JournalUnreadableError:
            existing = None
        # Note (Jiaxin Deng): the admin path is the only durable clue to which
        # operation (disk reload, distributed update, pause) left the journal
        # behind; narrowing the target set must not erase it.
        if existing is not None and isinstance(existing.get("path"), str):
            document["path"] = existing["path"]
        self._write(document)

    def discard(self, worker_id: str) -> bool:
        """Remove one id; False when the entry could not be durably resolved.

        A False return means the file needs operator inspection; callers must
        surface that instead of reporting success while the 409 update gate
        stays closed.
        """
        try:
            remaining = [wid for wid in self.pending() if wid != worker_id]
        except JournalUnreadableError:
            # Note (Jiaxin Deng): cannot safely edit an unreadable journal;
            # leave it for an operator rather than partially clearing it.
            return False
        try:
            self.keep(remaining)
        except JournalUnwritableError:
            return False
        return True

    def clear(self) -> None:
        directory = os.path.dirname(self._path) or "."
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            # Note (Jiaxin Deng): an earlier unlink may have failed its
            # directory fsync, so absence is not durability; sync again rather
            # than report a resolution that a crash can undo.
            if os.path.isdir(directory):
                self._fsync_dir()
            return
        except OSError as exc:
            raise JournalUnwritableError(f"{self._path}: {exc}") from exc
        self._fsync_dir()

    def _write(self, data: dict) -> None:
        directory = os.path.dirname(self._path) or "."
        tmp_path = f"{self._path}.tmp"
        try:
            # Note (Jiaxin Deng): a directory this call creates is only durable
            # once its own entry in the parent is synced. Without this the whole
            # endpoint directory can vanish in a host crash and take the journal
            # with it, which is exactly the case the journal exists to survive.
            for created in reversed(_make_private_dir(directory)):
                _fsync_directory(os.path.dirname(created) or ".")
            fd = os.open(
                tmp_path,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _NOFOLLOW,
                _JOURNAL_FILE_MODE,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json.dumps(data))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._path)
        except OSError as exc:
            raise JournalUnwritableError(f"{self._path}: {exc}") from exc
        self._fsync_dir()

    def _fsync_dir(self) -> None:
        _fsync_directory(os.path.dirname(self._path) or ".")


def _fsync_directory(directory: str) -> None:
    # Note (Jiaxin Deng): a rename/unlink is only durable across a host crash
    # once the parent directory entry is synced; non-POSIX hosts expose no
    # directory handle, so there the file fsync is the cap.
    if os.name != "posix":
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError as exc:
        raise JournalUnwritableError(f"{directory}: {exc}") from exc
    try:
        os.fsync(fd)
    except OSError as exc:
        raise JournalUnwritableError(f"{directory}: {exc}") from exc
    finally:
        os.close(fd)


def _make_private_dir(path: str) -> list[str]:
    """makedirs; returns the directories this call created, innermost first.

    A pre-existing directory keeps its own permissions: the state directory may
    be a mount point an operator set up, and shared ancestors such as
    ~/.local must not be tightened underneath other tools.
    """
    if os.path.isdir(path):
        return []
    created: list[str] = []
    probe = path
    while probe and not os.path.isdir(probe):
        created.append(probe)
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    # Note (Jiaxin Deng): pass the mode so a permissive umask cannot leave the
    # directory world-writable in the window before the chmod lands.
    os.makedirs(path, mode=_STATE_DIR_MODE, exist_ok=True)
    if os.name == "posix":
        for directory in created:
            try:
                os.chmod(directory, _STATE_DIR_MODE)
            except OSError as exc:
                logger.warning(f"could not tighten {directory} to 0700: {exc}")
    return created


def resolve_state_dir(state_dir: str | None = None) -> str:
    """Where durable router state lives, highest precedence first."""
    for candidate in (state_dir, os.environ.get(STATE_DIR_ENV)):
        if candidate:
            return os.path.abspath(os.path.expanduser(candidate))
    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    if xdg_state_home:
        return os.path.join(
            os.path.abspath(os.path.expanduser(xdg_state_home)), _STATE_DIR_NAME
        )
    home = os.path.expanduser("~")
    if home.startswith("~"):
        # Note (Jiaxin Deng): expanduser returns the path unchanged when HOME
        # is unset and the uid has no passwd entry, which is an ordinary
        # container run under an injected uid. Returning it would put the
        # journal under the router's cwd, on the ephemeral layer.
        raise JournalUnwritableError(
            "cannot resolve a home directory for the router state directory "
            "(HOME is unset and this uid has no passwd entry); point "
            f"--router-state-dir or {STATE_DIR_ENV} at a writable persistent path"
        )
    return os.path.join(home, ".local", "state", _STATE_DIR_NAME)


def ensure_state_dir(state_dir: str) -> str:
    """Create and probe the state directory, or fail closed.

    Checked at startup so a misconfigured mount surfaces there instead of on
    the first weight update, when the pool is already disabled.
    """
    # Note (Jiaxin Deng): routers on different ports share one state directory,
    # so the probe is per-pid and its cleanup tolerates a vanished file; a
    # fixed name would let two starting routers unlink each other's probe and
    # both fail closed on a directory that is perfectly writable.
    probe_path = os.path.join(state_dir, f".write-probe.{os.getpid()}")
    try:
        for created in reversed(_make_private_dir(state_dir)):
            _fsync_directory(os.path.dirname(created) or ".")
        fd = os.open(
            probe_path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _NOFOLLOW,
            _JOURNAL_FILE_MODE,
        )
        os.close(fd)
        try:
            os.unlink(probe_path)
        except FileNotFoundError:
            pass
    except OSError as exc:
        raise JournalUnwritableError(
            f"router state directory {state_dir} is unusable ({exc}); it holds "
            "the weight-update journal that keeps a partially updated pool "
            "disabled across a host reboot. Point --router-state-dir (or "
            f"{STATE_DIR_ENV}) at a writable persistent path."
        ) from exc
    return state_dir


def _endpoint_key(host: str, port: int) -> str:
    """A filename-safe, per-endpoint directory name.

    Sanitizing alone would fold distinct endpoints together (`::1` and `__1`),
    so the readable part carries a digest of the exact host:port.
    """
    endpoint = f"{host}:{port}"
    readable = _UNSAFE_KEY_CHARS.sub("_", endpoint)[:_MAX_READABLE_KEY]
    digest = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{digest}"


def default_journal_path(host: str, port: int, state_dir: str | None = None) -> str:
    """A stable per-endpoint journal path that survives a host reboot.

    Neither the per-run workdir (random, lost on a full restart) nor a temp
    directory (erased by a reboot the remote workers survive); keyed by
    host:port so the same endpoint always finds its own transaction.
    """
    return os.path.join(
        resolve_state_dir(state_dir), _endpoint_key(host, port), "update_journal.json"
    )


def build_journal(
    host: str, port: int, state_dir: str | None, explicit_path: str | None = None
) -> UpdateJournal | UnavailableJournal:
    """The endpoint's journal, or a refusing stand-in if it has no home."""
    if explicit_path:
        return UpdateJournal(explicit_path)
    try:
        resolved = ensure_state_dir(resolve_state_dir(state_dir))
        return UpdateJournal(
            os.path.join(resolved, _endpoint_key(host, port), "update_journal.json")
        )
    except JournalUnwritableError as exc:
        logger.error(f"weight updates will be refused: {exc}")
        return UnavailableJournal(str(exc))
