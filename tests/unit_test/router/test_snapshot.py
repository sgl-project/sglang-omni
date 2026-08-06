from __future__ import annotations

import json
import os
from pathlib import Path

from sglang_omni_router.snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
    SnapshotReader,
    SnapshotWorker,
    SnapshotWriter,
)


def _workers(n: int = 2) -> list[SnapshotWorker]:
    return [
        SnapshotWorker(url=f"http://127.0.0.1:{8101 + i}", worker_id=f"w{i}")
        for i in range(n)
    ]


def test_snapshot_round_trip(tmp_path: Path) -> None:
    path = str(tmp_path / "snap.json")
    writer = SnapshotWriter(path, cp_epoch="epoch-a")
    published = writer.publish(_workers())

    reader = SnapshotReader(path)
    assert reader.maybe_reload() is True
    assert reader.snapshot == published
    assert reader.snapshot.version == SNAPSHOT_SCHEMA_VERSION
    assert [w.url for w in reader.snapshot.workers] == [
        "http://127.0.0.1:8101",
        "http://127.0.0.1:8102",
    ]
    assert reader.maybe_reload() is False


def test_snapshot_seq_is_monotonic_and_applied(tmp_path: Path) -> None:
    path = str(tmp_path / "snap.json")
    writer = SnapshotWriter(path, cp_epoch="epoch-a")
    writer.publish(_workers(1))
    writer.publish(_workers(3))
    assert writer.seq == 2

    reader = SnapshotReader(path)
    assert reader.maybe_reload() is True
    assert reader.snapshot.seq == 2
    assert len(reader.snapshot.workers) == 3


def test_reader_keeps_last_good_snapshot_on_torn_file(tmp_path: Path) -> None:
    path = str(tmp_path / "snap.json")
    writer = SnapshotWriter(path, cp_epoch="epoch-a")
    writer.publish(_workers())
    reader = SnapshotReader(path)
    assert reader.maybe_reload() is True
    good = reader.snapshot

    Path(path).write_text('{"version": 1, "seq": 2, "wor', encoding="utf-8")
    assert reader.maybe_reload() is False
    assert reader.snapshot == good

    writer.publish(_workers(1))
    assert reader.maybe_reload() is True
    assert reader.snapshot.seq == 2


def test_reader_ignores_stale_seq_within_the_same_epoch(tmp_path: Path) -> None:
    path = str(tmp_path / "snap.json")
    writer = SnapshotWriter(path, cp_epoch="epoch-a")
    writer.publish(_workers())
    writer.publish(_workers())
    reader = SnapshotReader(path)
    assert reader.maybe_reload() is True
    assert reader.snapshot.seq == 2

    stale = json.loads(Path(path).read_text(encoding="utf-8"))
    stale["seq"] = 1
    Path(path).write_text(json.dumps(stale), encoding="utf-8")
    assert reader.maybe_reload() is False
    assert reader.snapshot.seq == 2


def test_reader_accepts_a_new_cp_epoch_with_a_restarted_seq(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "snap.json")
    SnapshotWriter(path, cp_epoch="epoch-a").publish(_workers())
    reader = SnapshotReader(path)
    assert reader.maybe_reload() is True

    restarted = SnapshotWriter(path, cp_epoch="epoch-b")
    restarted.publish(_workers())
    restarted.publish(_workers(1))
    # Note (Jiaxin Deng): seq went 2 -> 2 across epochs; the new epoch must still be
    # applied
    assert reader.maybe_reload() is True
    assert reader.snapshot.cp_epoch == "epoch-b"
    assert reader.snapshot.seq == 2


def test_reader_reports_snapshot_age(tmp_path: Path) -> None:
    path = str(tmp_path / "snap.json")
    reader = SnapshotReader(path)
    assert reader.age_secs() is None
    SnapshotWriter(path, cp_epoch="epoch-a").publish(_workers())
    reader.maybe_reload()
    generated_at = reader.snapshot.generated_at
    assert reader.age_secs(now=generated_at + 7.5) == 7.5


def test_writer_unlink_is_idempotent(tmp_path: Path) -> None:
    path = str(tmp_path / "snap.json")
    writer = SnapshotWriter(path, cp_epoch="epoch-a")
    writer.publish(_workers())
    writer.unlink()
    writer.unlink()
    assert not Path(path).exists()


def test_reader_rejects_an_incompatible_schema_version(tmp_path: Path) -> None:
    # Note (Jiaxin Deng): valid JSON from another schema version must not replace the
    # good view
    path = str(tmp_path / "snap.json")
    writer = SnapshotWriter(path, cp_epoch="epoch-a")
    writer.publish(_workers())
    reader = SnapshotReader(path)
    assert reader.maybe_reload() is True
    good = reader.snapshot

    alien = json.loads(Path(path).read_text(encoding="utf-8"))
    alien["version"] = SNAPSHOT_SCHEMA_VERSION + 1
    alien["seq"] = 99
    Path(path).write_text(json.dumps(alien), encoding="utf-8")
    assert reader.maybe_reload() is False
    assert reader.snapshot == good


def test_writer_cleans_its_temp_file_when_the_replace_fails(
    tmp_path: Path, monkeypatch
) -> None:
    path = str(tmp_path / "snap.json")
    writer = SnapshotWriter(path, cp_epoch="epoch-a")

    def _boom(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr("sglang_omni_router.snapshot.os.replace", _boom)
    try:
        writer.publish(_workers())
    except OSError:
        pass
    leftovers = [p.name for p in tmp_path.iterdir()]
    assert leftovers == []


def test_reader_detects_rapid_republish_despite_coarse_mtime(
    tmp_path: Path,
) -> None:
    # Note (Jiaxin Deng): Kernel file timestamps are tick-granular: two publishes inside
    # one tick share an mtime, so mtime alone must not be the change signal.
    path = str(tmp_path / "snap.json")
    writer = SnapshotWriter(path, cp_epoch="epoch-a")
    writer.publish(_workers())
    reader = SnapshotReader(path)
    assert reader.maybe_reload() is True
    mtime_ns = os.stat(path).st_mtime_ns

    writer.publish(_workers(3))
    os.utime(path, ns=(mtime_ns, mtime_ns))  # simulate a same-tick republish
    assert reader.maybe_reload() is True
    assert reader.snapshot.seq == 2
