from __future__ import annotations

import contextlib
import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from .models import ActiveStatus, DispatchCandidate, PullRequestState, QueueItem, Stage, TerminalStatus
from .stage_registry import StageRegistry


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class QueueStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS pr_state (
                    repo TEXT NOT NULL,
                    pr_number INTEGER NOT NULL,
                    head_sha TEXT NOT NULL,
                    state TEXT NOT NULL,
                    draft INTEGER NOT NULL,
                    labels_json TEXT NOT NULL,
                    installation_id INTEGER,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (repo, pr_number)
                );

                CREATE TABLE IF NOT EXISTS queue_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repo TEXT NOT NULL,
                    pr_number INTEGER NOT NULL,
                    head_sha TEXT NOT NULL,
                    stage_id TEXT NOT NULL,
                    check_name TEXT NOT NULL,
                    check_run_id INTEGER,
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 1,
                    enqueue_time TEXT NOT NULL,
                    dispatch_time TEXT,
                    finish_time TEXT,
                    workflow_run_id INTEGER,
                    conclusion TEXT,
                    installation_id INTEGER,
                    updated_at TEXT NOT NULL,
                    UNIQUE (repo, pr_number, head_sha, stage_id, attempt)
                );

                CREATE INDEX IF NOT EXISTS idx_queue_status
                    ON queue_items (repo, status, head_sha);
                CREATE INDEX IF NOT EXISTS idx_queue_pr_sha
                    ON queue_items (repo, pr_number, head_sha);
                """
            )

    def upsert_pr_state(self, pr: PullRequestState) -> None:
        with self.transaction() as conn:
            self._upsert_pr_state(conn, pr)

    def _upsert_pr_state(self, conn: sqlite3.Connection, pr: PullRequestState) -> None:
        conn.execute(
            """
            INSERT INTO pr_state
                (repo, pr_number, head_sha, state, draft, labels_json, installation_id, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(repo, pr_number) DO UPDATE SET
                head_sha = excluded.head_sha,
                state = excluded.state,
                draft = excluded.draft,
                labels_json = excluded.labels_json,
                installation_id = excluded.installation_id,
                updated_at = excluded.updated_at
            """,
            (
                pr.repo,
                pr.pr_number,
                pr.head_sha,
                pr.state,
                int(pr.draft),
                json.dumps(sorted(pr.labels)),
                pr.installation_id,
                utc_now(),
            ),
        )

    def enqueue_pr_stages(
        self,
        pr: PullRequestState,
        registry: StageRegistry,
        run_label: str,
    ) -> list[QueueItem]:
        with self.transaction() as conn:
            self._upsert_pr_state(conn, pr)
            self._mark_stale_for_pr(conn, pr.repo, pr.pr_number, pr.head_sha)

            if pr.state != "open" or pr.draft or not pr.has_label(run_label):
                self._cancel_pending_for_pr(conn, pr.repo, pr.pr_number, pr.head_sha)
                return []

            now = utc_now()
            for stage in registry.all():
                conn.execute(
                    """
                    INSERT OR IGNORE INTO queue_items
                        (repo, pr_number, head_sha, stage_id, check_name, status,
                         attempt, enqueue_time, installation_id, updated_at)
                    VALUES (?, ?, ?, ?, ?, 'pending', 1, ?, ?, ?)
                    """,
                    (
                        pr.repo,
                        pr.pr_number,
                        pr.head_sha,
                        stage.stage_id,
                        stage.check_name,
                        now,
                        pr.installation_id,
                        now,
                    ),
                )
            return self._items_for_pr_sha(conn, pr.repo, pr.pr_number, pr.head_sha)

    def cancel_pending_for_pr(self, pr: PullRequestState) -> list[QueueItem]:
        with self.transaction() as conn:
            self._upsert_pr_state(conn, pr)
            self._cancel_pending_for_pr(conn, pr.repo, pr.pr_number, pr.head_sha)
            return self._items_for_pr_sha(conn, pr.repo, pr.pr_number, pr.head_sha)

    def mark_terminal(
        self,
        queue_item_id: int,
        status: str,
        conclusion: str | None = None,
        workflow_run_id: int | None = None,
    ) -> QueueItem | None:
        if status not in TerminalStatus:
            raise ValueError(f"{status!r} is not terminal")
        with self.transaction() as conn:
            updates = [status, conclusion, utc_now(), utc_now(), queue_item_id]
            conn.execute(
                """
                UPDATE queue_items
                SET status = ?, conclusion = ?, finish_time = ?, updated_at = ?
                WHERE id = ?
                """,
                updates,
            )
            if workflow_run_id is not None:
                conn.execute(
                    "UPDATE queue_items SET workflow_run_id = ?, updated_at = ? WHERE id = ?",
                    (workflow_run_id, utc_now(), queue_item_id),
                )
            return self._item_by_id(conn, queue_item_id)

    def mark_running(self, queue_item_id: int, workflow_run_id: int | None = None) -> QueueItem | None:
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE queue_items
                SET status = 'running',
                    workflow_run_id = COALESCE(?, workflow_run_id),
                    updated_at = ?
                WHERE id = ? AND status IN ('dispatched', 'running')
                """,
                (workflow_run_id, utc_now(), queue_item_id),
            )
            return self._item_by_id(conn, queue_item_id)

    def reset_failed_for_rerun(
        self,
        pr: PullRequestState,
        registry: StageRegistry,
        run_label: str,
    ) -> list[QueueItem]:
        """Create fresh pending attempts for failed/cancelled/timed-out stages."""

        with self.transaction() as conn:
            self._upsert_pr_state(conn, pr)
            if pr.state != "open" or pr.draft or not pr.has_label(run_label):
                return []

            failed = conn.execute(
                """
                SELECT stage_id, MAX(attempt) AS latest_attempt
                FROM queue_items
                WHERE repo = ? AND pr_number = ? AND head_sha = ?
                  AND status IN ('failed', 'cancelled', 'timed_out')
                GROUP BY stage_id
                """,
                (pr.repo, pr.pr_number, pr.head_sha),
            ).fetchall()

            now = utc_now()
            for row in failed:
                stage = registry.get(row["stage_id"])
                conn.execute(
                    """
                    INSERT OR IGNORE INTO queue_items
                        (repo, pr_number, head_sha, stage_id, check_name, status,
                         attempt, enqueue_time, installation_id, updated_at)
                    VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)
                    """,
                    (
                        pr.repo,
                        pr.pr_number,
                        pr.head_sha,
                        stage.stage_id,
                        stage.check_name,
                        int(row["latest_attempt"]) + 1,
                        now,
                        pr.installation_id,
                        now,
                    ),
                )
            return self._items_for_pr_sha(conn, pr.repo, pr.pr_number, pr.head_sha)

    def select_dispatch_candidates(
        self,
        registry: StageRegistry,
        *,
        run_label: str,
        high_priority_label: str,
        capacity: int,
        capacity_group: str = "self-hosted-gpu",
    ) -> list[DispatchCandidate]:
        with self.transaction() as conn:
            self._cancel_ineligible_locked(conn, run_label)
            active_count = self._active_count(conn, capacity_group, registry)
            slots = max(0, capacity - active_count)
            if slots <= 0:
                return []

            rows = conn.execute(
                """
                SELECT qi.*, ps.labels_json, ps.state AS pr_state, ps.draft AS pr_draft,
                       ps.head_sha AS current_head_sha
                FROM queue_items qi
                JOIN pr_state ps
                  ON ps.repo = qi.repo AND ps.pr_number = qi.pr_number
                WHERE qi.status = 'pending'
                """
            ).fetchall()

            candidates: list[DispatchCandidate] = []
            status_by_key = self._latest_status_by_stage(conn)
            for row in rows:
                stage = registry.get(row["stage_id"])
                if stage.capacity_group != capacity_group:
                    continue
                labels = set(json.loads(row["labels_json"]))
                if (
                    row["pr_state"] != "open"
                    or row["pr_draft"]
                    or row["head_sha"] != row["current_head_sha"]
                    or run_label not in labels
                ):
                    continue
                if not self._dependencies_passed(row, stage, status_by_key):
                    continue
                priority_rank = 0 if high_priority_label in labels else 1
                candidates.append(
                    DispatchCandidate(
                        item=self._row_to_item(row),
                        stage=stage,
                        priority_rank=priority_rank,
                    )
                )

            candidates.sort(
                key=lambda candidate: (
                    candidate.priority_rank,
                    candidate.item.enqueue_time,
                    candidate.item.pr_number,
                    candidate.stage.order,
                    candidate.stage.stage_id,
                )
            )

            selected = candidates[:slots]
            now = utc_now()
            for candidate in selected:
                conn.execute(
                    """
                    UPDATE queue_items
                    SET status = 'dispatched', dispatch_time = ?, updated_at = ?
                    WHERE id = ? AND status = 'pending'
                    """,
                    (now, now, candidate.item.id),
                )
            return selected

    def set_check_run_id(self, queue_item_id: int, check_run_id: int) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE queue_items SET check_run_id = ?, updated_at = ? WHERE id = ?",
                (check_run_id, utc_now(), queue_item_id),
            )

    def reset_to_pending(self, queue_item_id: int) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE queue_items
                SET status = 'pending', dispatch_time = NULL, updated_at = ?
                WHERE id = ?
                """,
                (utc_now(), queue_item_id),
            )

    def list_items_for_pr(self, repo: str, pr_number: int) -> list[QueueItem]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM queue_items
                WHERE repo = ? AND pr_number = ?
                ORDER BY head_sha, attempt, id
                """,
                (repo, pr_number),
            ).fetchall()
            return [self._row_to_item(row) for row in rows]

    def _cancel_pending_for_pr(
        self,
        conn: sqlite3.Connection,
        repo: str,
        pr_number: int,
        head_sha: str,
    ) -> None:
        conn.execute(
            """
            UPDATE queue_items
            SET status = 'cancelled', conclusion = 'cancelled', finish_time = ?, updated_at = ?
            WHERE repo = ? AND pr_number = ? AND head_sha = ? AND status = 'pending'
            """,
            (utc_now(), utc_now(), repo, pr_number, head_sha),
        )

    def _mark_stale_for_pr(
        self,
        conn: sqlite3.Connection,
        repo: str,
        pr_number: int,
        current_head_sha: str,
    ) -> None:
        conn.execute(
            """
            UPDATE queue_items
            SET status = 'stale', conclusion = 'cancelled', finish_time = ?, updated_at = ?
            WHERE repo = ? AND pr_number = ? AND head_sha != ?
              AND status IN ('pending', 'dispatched')
            """,
            (utc_now(), utc_now(), repo, pr_number, current_head_sha),
        )

    def _cancel_ineligible_locked(self, conn: sqlite3.Connection, run_label: str) -> None:
        rows = conn.execute(
            """
            SELECT qi.id, qi.status, ps.state, ps.draft, ps.labels_json, ps.head_sha AS current_head_sha,
                   qi.head_sha
            FROM queue_items qi
            JOIN pr_state ps
              ON ps.repo = qi.repo AND ps.pr_number = qi.pr_number
            WHERE qi.status = 'pending'
            """
        ).fetchall()
        for row in rows:
            labels = set(json.loads(row["labels_json"]))
            should_cancel = (
                row["state"] != "open"
                or row["draft"]
                or run_label not in labels
                or row["head_sha"] != row["current_head_sha"]
            )
            if should_cancel:
                status = "stale" if row["head_sha"] != row["current_head_sha"] else "cancelled"
                conn.execute(
                    """
                    UPDATE queue_items
                    SET status = ?, conclusion = 'cancelled', finish_time = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (status, utc_now(), utc_now(), row["id"]),
                )

    def _active_count(
        self,
        conn: sqlite3.Connection,
        capacity_group: str,
        registry: StageRegistry,
    ) -> int:
        rows = conn.execute(
            "SELECT stage_id FROM queue_items WHERE status IN ('dispatched', 'running')"
        ).fetchall()
        return sum(
            1
            for row in rows
            if registry.has(row["stage_id"]) and registry.get(row["stage_id"]).capacity_group == capacity_group
        )

    def _latest_status_by_stage(self, conn: sqlite3.Connection) -> dict[tuple[str, int, str, str], str]:
        rows = conn.execute(
            """
            SELECT repo, pr_number, head_sha, stage_id, status, MAX(attempt) AS attempt
            FROM queue_items
            GROUP BY repo, pr_number, head_sha, stage_id
            """
        ).fetchall()
        return {
            (row["repo"], int(row["pr_number"]), row["head_sha"], row["stage_id"]): row["status"]
            for row in rows
        }

    def _dependencies_passed(
        self,
        row: sqlite3.Row,
        stage: Stage,
        status_by_key: dict[tuple[str, int, str, str], str],
    ) -> bool:
        for dependency in stage.depends_on:
            key = (row["repo"], int(row["pr_number"]), row["head_sha"], dependency)
            if status_by_key.get(key) != "passed":
                return False
        return True

    def _items_for_pr_sha(
        self,
        conn: sqlite3.Connection,
        repo: str,
        pr_number: int,
        head_sha: str,
    ) -> list[QueueItem]:
        rows = conn.execute(
            """
            SELECT * FROM queue_items
            WHERE repo = ? AND pr_number = ? AND head_sha = ?
            ORDER BY attempt, id
            """,
            (repo, pr_number, head_sha),
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def _item_by_id(self, conn: sqlite3.Connection, queue_item_id: int) -> QueueItem | None:
        row = conn.execute("SELECT * FROM queue_items WHERE id = ?", (queue_item_id,)).fetchone()
        return self._row_to_item(row) if row else None

    def _row_to_item(self, row: sqlite3.Row) -> QueueItem:
        return QueueItem(
            id=int(row["id"]),
            repo=row["repo"],
            pr_number=int(row["pr_number"]),
            head_sha=row["head_sha"],
            stage_id=row["stage_id"],
            check_name=row["check_name"],
            check_run_id=int(row["check_run_id"]) if row["check_run_id"] is not None else None,
            status=row["status"],
            attempt=int(row["attempt"]),
            enqueue_time=row["enqueue_time"],
            dispatch_time=row["dispatch_time"],
            workflow_run_id=int(row["workflow_run_id"]) if row["workflow_run_id"] is not None else None,
            conclusion=row["conclusion"],
            installation_id=int(row["installation_id"]) if row["installation_id"] is not None else None,
        )
