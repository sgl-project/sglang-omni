from __future__ import annotations

import contextlib
import json
import sqlite3
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from domain.models import DagNode, PullRequestState, QueueItem, TERMINAL_STATUSES, WorkflowGraph
from domain.policy import count_active, latest_by_stage, select_next
from domain.transitions import validate_transition

from .sqlite_queries import SCHEMA, SELECT_ALL_ITEMS, SELECT_ITEMS_FOR_PR, SELECT_PENDING, SELECT_PR_STATES
from .store import QueueStore


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class SqliteQueueStore(QueueStore):
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
            conn.executescript(SCHEMA)

    def upsert_pr_state(self, pr: PullRequestState) -> None:
        with self.transaction() as conn:
            self._upsert_pr_state(conn, pr)

    def enqueue_pr(
        self,
        pr: PullRequestState,
        graph: WorkflowGraph,
        *,
        run_label: str,
    ) -> list[QueueItem]:
        with self.transaction() as conn:
            self._upsert_pr_state(conn, pr)
            self._mark_stale_for_pr(conn, pr.repo, pr.pr_number, pr.head_sha)

            if pr.state != "open" or pr.draft or not pr.has_label(run_label):
                self._cancel_pending_for_pr(conn, pr.repo, pr.pr_number, pr.head_sha)
                return []

            now = utc_now()
            for node in graph.nodes:
                if not node.is_executable:
                    continue
                conn.execute(
                    """
                    INSERT OR IGNORE INTO queue_items
                        (repo, pr_number, head_sha, stage_key, check_name, status,
                         attempt, enqueue_time, generated_workflow_path, source_hash,
                         installation_id, updated_at)
                    VALUES (?, ?, ?, ?, ?, 'pending', 1, ?, ?, ?, ?, ?)
                    """,
                    (
                        pr.repo,
                        pr.pr_number,
                        pr.head_sha,
                        node.key,
                        node.check_name,
                        now,
                        node.generated_workflow_path,
                        graph.source_hash,
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

    def reserve_next(
        self,
        graph: WorkflowGraph,
        *,
        run_label: str,
        high_priority_label: str,
        capacity: int,
        self_hosted_only: bool = True,
    ) -> QueueItem | None:
        with self.transaction() as conn:
            self._cancel_ineligible_locked(conn, run_label)
            all_items = [self._row_to_item(row) for row in conn.execute(SELECT_ALL_ITEMS).fetchall()]
            pr_states = self._pr_states_from_conn(conn)
            latest = latest_by_stage(all_items)
            pending = [item for item in all_items if item.status == "pending"]
            active = count_active(all_items, graph)

            selected_ids = select_next(
                pending,
                pr_states,
                latest,
                graph,
                run_label=run_label,
                high_priority_label=high_priority_label,
                active_count=active,
                capacity=capacity,
                self_hosted_only=self_hosted_only,
            )
            if not selected_ids:
                return None

            item_id = selected_ids[0]
            now = utc_now()
            cursor = conn.execute(
                """
                UPDATE queue_items
                SET status = 'reserved', dispatch_time = ?, updated_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (now, now, item_id),
            )
            if cursor.rowcount != 1:
                return None
            return self._item_by_id(conn, item_id)

    def mark_dispatched(
        self,
        queue_item_id: int,
        *,
        dispatch_id: str,
        workflow_run_id: int | None = None,
    ) -> QueueItem | None:
        with self.transaction() as conn:
            item = self._item_by_id(conn, queue_item_id)
            if item is None:
                return None
            validate_transition(item.status, "dispatched")
            conn.execute(
                """
                UPDATE queue_items
                SET status = 'dispatched', dispatch_id = ?, workflow_run_id = COALESCE(?, workflow_run_id),
                    updated_at = ?
                WHERE id = ? AND status = 'reserved'
                """,
                (dispatch_id, workflow_run_id, utc_now(), queue_item_id),
            )
            return self._item_by_id(conn, queue_item_id)

    def mark_running(self, queue_item_id: int, workflow_run_id: int | None = None) -> QueueItem | None:
        with self.transaction() as conn:
            item = self._item_by_id(conn, queue_item_id)
            if item is None:
                return None
            validate_transition(item.status, "running")
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

    def transition(
        self,
        queue_item_id: int,
        target_status: str,
        *,
        conclusion: str | None = None,
        workflow_run_id: int | None = None,
    ) -> QueueItem | None:
        with self.transaction() as conn:
            item = self._item_by_id(conn, queue_item_id)
            if item is None:
                return None
            validate_transition(item.status, target_status)
            conn.execute(
                """
                UPDATE queue_items
                SET status = ?, conclusion = ?, finish_time = ?, updated_at = ?
                WHERE id = ?
                """,
                (target_status, conclusion, utc_now(), utc_now(), queue_item_id),
            )
            if workflow_run_id is not None:
                conn.execute(
                    "UPDATE queue_items SET workflow_run_id = ?, updated_at = ? WHERE id = ?",
                    (workflow_run_id, utc_now(), queue_item_id),
                )
            return self._item_by_id(conn, queue_item_id)

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
                SET status = 'pending', dispatch_time = NULL, dispatch_id = NULL, updated_at = ?
                WHERE id = ?
                """,
                (utc_now(), queue_item_id),
            )

    def list_items_for_pr(self, repo: str, pr_number: int) -> list[QueueItem]:
        with self.connect() as conn:
            rows = conn.execute(SELECT_ITEMS_FOR_PR, (repo, pr_number)).fetchall()
            return [self._row_to_item(row) for row in rows]

    def list_all_items(self) -> list[QueueItem]:
        with self.connect() as conn:
            rows = conn.execute(SELECT_ALL_ITEMS).fetchall()
            return [self._row_to_item(row) for row in rows]

    def list_pr_states(self) -> dict[tuple[str, int], PullRequestState]:
        with self.connect() as conn:
            return self._pr_states_from_conn(conn)

    def reset_failed_for_rerun(
        self,
        pr: PullRequestState,
        graph: WorkflowGraph,
        *,
        run_label: str,
    ) -> list[QueueItem]:
        with self.transaction() as conn:
            self._upsert_pr_state(conn, pr)
            if pr.state != "open" or pr.draft or not pr.has_label(run_label):
                return []

            failed = conn.execute(
                """
                SELECT stage_key, MAX(attempt) AS latest_attempt
                FROM queue_items
                WHERE repo = ? AND pr_number = ? AND head_sha = ?
                  AND status IN ('failed', 'cancelled', 'timed_out')
                GROUP BY stage_key
                """,
                (pr.repo, pr.pr_number, pr.head_sha),
            ).fetchall()

            now = utc_now()
            for row in failed:
                node = graph.get(row["stage_key"])
                conn.execute(
                    """
                    INSERT OR IGNORE INTO queue_items
                        (repo, pr_number, head_sha, stage_key, check_name, status,
                         attempt, enqueue_time, generated_workflow_path, source_hash,
                         installation_id, updated_at)
                    VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pr.repo,
                        pr.pr_number,
                        pr.head_sha,
                        node.key,
                        node.check_name,
                        int(row["latest_attempt"]) + 1,
                        now,
                        node.generated_workflow_path,
                        graph.source_hash,
                        pr.installation_id,
                        now,
                    ),
                )
            return self._items_for_pr_sha(conn, pr.repo, pr.pr_number, pr.head_sha)

    def store_outputs(
        self,
        *,
        repo: str,
        head_sha: str,
        stage_key: str,
        attempt: int,
        outputs: dict[str, str],
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO stage_outputs (repo, head_sha, stage_key, attempt, outputs_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo, head_sha, stage_key, attempt) DO UPDATE SET
                    outputs_json = excluded.outputs_json,
                    updated_at = excluded.updated_at
                """,
                (repo, head_sha, stage_key, attempt, json.dumps(outputs, sort_keys=True), utc_now()),
            )

    def get_outputs(
        self,
        *,
        repo: str,
        head_sha: str,
        stage_key: str,
        attempt: int,
    ) -> dict[str, str]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT outputs_json FROM stage_outputs
                WHERE repo = ? AND head_sha = ? AND stage_key = ? AND attempt = ?
                """,
                (repo, head_sha, stage_key, attempt),
            ).fetchone()
            if row is None:
                return {}
            data = json.loads(row["outputs_json"])
            return {str(k): str(v) for k, v in data.items()}

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
              AND status IN ('pending', 'reserved', 'dispatched')
            """,
            (utc_now(), utc_now(), repo, pr_number, current_head_sha),
        )

    def _cancel_ineligible_locked(self, conn: sqlite3.Connection, run_label: str) -> None:
        rows = conn.execute(SELECT_PENDING).fetchall()
        for row in rows:
            labels = set(json.loads(row["labels_json"]))
            should_cancel = (
                row["pr_state"] != "open"
                or row["pr_draft"]
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

    def _pr_states_from_conn(self, conn: sqlite3.Connection) -> dict[tuple[str, int], PullRequestState]:
        states: dict[tuple[str, int], PullRequestState] = {}
        for row in conn.execute(SELECT_PR_STATES).fetchall():
            pr = PullRequestState(
                repo=row["repo"],
                pr_number=int(row["pr_number"]),
                head_sha=row["head_sha"],
                state=row["state"],
                draft=bool(row["draft"]),
                labels=frozenset(json.loads(row["labels_json"])),
                installation_id=int(row["installation_id"]) if row["installation_id"] is not None else None,
            )
            states[(pr.repo, pr.pr_number)] = pr
        return states

    def _row_to_item(self, row: sqlite3.Row) -> QueueItem:
        return QueueItem(
            id=int(row["id"]),
            repo=row["repo"],
            pr_number=int(row["pr_number"]),
            head_sha=row["head_sha"],
            stage_key=row["stage_key"],
            check_name=row["check_name"],
            check_run_id=int(row["check_run_id"]) if row["check_run_id"] is not None else None,
            status=row["status"],
            attempt=int(row["attempt"]),
            enqueue_time=row["enqueue_time"],
            dispatch_time=row["dispatch_time"],
            workflow_run_id=int(row["workflow_run_id"]) if row["workflow_run_id"] is not None else None,
            dispatch_id=row["dispatch_id"],
            generated_workflow_path=row["generated_workflow_path"],
            source_hash=row["source_hash"],
            conclusion=row["conclusion"],
            installation_id=int(row["installation_id"]) if row["installation_id"] is not None else None,
        )


def new_dispatch_id() -> str:
    return str(uuid.uuid4())
