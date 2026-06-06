from __future__ import annotations

from .config import Settings
from .github import GitHubClient
from .models import PullRequestState, QueueItem
from .stage_registry import StageRegistry
from .store import QueueStore


CONCLUSION_BY_STATUS = {
    "passed": "success",
    "failed": "failure",
    "cancelled": "cancelled",
    "timed_out": "timed_out",
    "stale": "cancelled",
    "skipped": "skipped",
}


class CiScheduler:
    def __init__(
        self,
        *,
        settings: Settings,
        store: QueueStore,
        registry: StageRegistry,
        github: GitHubClient,
    ) -> None:
        self.settings = settings
        self.store = store
        self.registry = registry
        self.github = github

    async def enqueue_pr(self, pr: PullRequestState) -> list[QueueItem]:
        items = self.store.enqueue_pr_stages(pr, self.registry, self.settings.run_label)
        await self._ensure_check_runs(items)
        await self.dispatch_once()
        return items

    async def cancel_pr(self, pr: PullRequestState) -> None:
        items = self.store.cancel_pending_for_pr(pr)
        for item in items:
            if item.status == "cancelled" and item.check_run_id:
                await self.github.update_check_run(
                    repo=item.repo,
                    installation_id=item.installation_id,
                    check_run_id=item.check_run_id,
                    status="completed",
                    conclusion="cancelled",
                    output_title=item.check_name,
                    output_summary="CI stage was cancelled because the PR is no longer eligible.",
                )
        await self.dispatch_once()

    async def rerun_failed(self, pr: PullRequestState) -> list[QueueItem]:
        items = self.store.reset_failed_for_rerun(pr, self.registry, self.settings.run_label)
        await self._ensure_check_runs([item for item in items if item.status == "pending"])
        await self.dispatch_once()
        return items

    async def mark_running(self, queue_item_id: int, workflow_run_id: int | None = None) -> None:
        self.store.mark_running(queue_item_id, workflow_run_id)

    async def mark_terminal(
        self,
        queue_item_id: int,
        status: str,
        workflow_run_id: int | None = None,
    ) -> None:
        item = self.store.mark_terminal(
            queue_item_id,
            status,
            conclusion=CONCLUSION_BY_STATUS.get(status),
            workflow_run_id=workflow_run_id,
        )
        if item and item.check_run_id:
            await self.github.update_check_run(
                repo=item.repo,
                installation_id=item.installation_id,
                check_run_id=item.check_run_id,
                status="completed",
                conclusion=CONCLUSION_BY_STATUS.get(status, "failure"),
                output_title=item.check_name,
                output_summary=f"Scheduler marked stage `{item.stage_id}` as `{status}`.",
            )
        await self.dispatch_once()

    async def dispatch_once(self) -> list[int]:
        candidates = self.store.select_dispatch_candidates(
            self.registry,
            run_label=self.settings.run_label,
            high_priority_label=self.settings.high_priority_label,
            capacity=self.settings.runner_capacity,
        )
        dispatched: list[int] = []
        for candidate in candidates:
            item = candidate.item
            stage = candidate.stage
            try:
                if item.check_run_id:
                    await self.github.update_check_run(
                        repo=item.repo,
                        installation_id=item.installation_id,
                        check_run_id=item.check_run_id,
                        status="in_progress",
                        output_title=item.check_name,
                        output_summary="Scheduler dispatched this stage to a self-hosted GPU runner.",
                    )

                await self.github.dispatch_workflow(
                    repo=item.repo,
                    installation_id=item.installation_id,
                    workflow_id=stage.workflow_id or self.settings.dispatch_workflow_id,
                    ref=stage.workflow_ref or self.settings.dispatch_ref,
                    inputs={
                        "queue_item_id": str(item.id),
                        "pr_number": str(item.pr_number),
                        "head_sha": item.head_sha,
                        "stage_id": item.stage_id,
                        "check_run_id": str(item.check_run_id or ""),
                    },
                )
                dispatched.append(item.id)
            except Exception:
                self.store.reset_to_pending(item.id)
                if item.check_run_id:
                    await self.github.update_check_run(
                        repo=item.repo,
                        installation_id=item.installation_id,
                        check_run_id=item.check_run_id,
                        status="queued",
                        output_title=item.check_name,
                        output_summary="Dispatch failed; stage was returned to the queue.",
                    )
                raise
        return dispatched

    async def _ensure_check_runs(self, items: list[QueueItem]) -> None:
        for item in items:
            if item.check_run_id is not None:
                continue
            if item.status != "pending":
                continue
            check_run = await self.github.create_check_run(
                repo=item.repo,
                installation_id=item.installation_id,
                name=item.check_name,
                head_sha=item.head_sha,
                status="queued",
                external_id=f"ci-scheduler:{item.id}",
                output_title=item.check_name,
                output_summary="CI stage is queued in the global GPU scheduler.",
            )
            self.store.set_check_run_id(item.id, check_run.id)
