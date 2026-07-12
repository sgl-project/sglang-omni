from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from domain.models import PullRequestState, QueueItem, WorkflowGraph


class QueueStore(ABC):
    @abstractmethod
    def upsert_pr_state(self, pr: PullRequestState) -> None: ...

    @abstractmethod
    def enqueue_pr(
        self,
        pr: PullRequestState,
        graph: WorkflowGraph,
        *,
        run_label: str,
    ) -> list[QueueItem]: ...

    @abstractmethod
    def cancel_pending_for_pr(self, pr: PullRequestState) -> list[QueueItem]: ...

    @abstractmethod
    def reserve_next(
        self,
        graph: WorkflowGraph,
        *,
        run_label: str,
        high_priority_label: str,
        capacity: int,
        self_hosted_only: bool = True,
    ) -> QueueItem | None: ...

    @abstractmethod
    def mark_dispatched(
        self,
        queue_item_id: int,
        *,
        dispatch_id: str,
        workflow_run_id: int | None = None,
    ) -> QueueItem | None: ...

    @abstractmethod
    def mark_running(self, queue_item_id: int, workflow_run_id: int | None = None) -> QueueItem | None: ...

    @abstractmethod
    def transition(
        self,
        queue_item_id: int,
        target_status: str,
        *,
        conclusion: str | None = None,
        workflow_run_id: int | None = None,
    ) -> QueueItem | None: ...

    @abstractmethod
    def set_check_run_id(self, queue_item_id: int, check_run_id: int) -> None: ...

    @abstractmethod
    def reset_to_pending(self, queue_item_id: int) -> None: ...

    @abstractmethod
    def list_items_for_pr(self, repo: str, pr_number: int) -> list[QueueItem]: ...

    @abstractmethod
    def list_all_items(self) -> list[QueueItem]: ...

    @abstractmethod
    def list_pr_states(self) -> dict[tuple[str, int], PullRequestState]: ...

    @abstractmethod
    def reset_failed_for_rerun(
        self,
        pr: PullRequestState,
        graph: WorkflowGraph,
        *,
        run_label: str,
    ) -> list[QueueItem]: ...

    @abstractmethod
    def store_outputs(
        self,
        *,
        repo: str,
        head_sha: str,
        stage_key: str,
        attempt: int,
        outputs: dict[str, str],
    ) -> None: ...

    @abstractmethod
    def get_outputs(
        self,
        *,
        repo: str,
        head_sha: str,
        stage_key: str,
        attempt: int,
    ) -> dict[str, str]: ...
