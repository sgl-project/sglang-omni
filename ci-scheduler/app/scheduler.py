from __future__ import annotations

import json
from pathlib import Path

from adapters.github_client import GitHubClient
from adapters.sqlite_store import SqliteQueueStore, new_dispatch_id
from adapters.store import QueueStore
from app.config import Settings
from compiler.graph import compile_all
from domain.models import (
    CONCLUSION_BY_STATUS,
    NeedState,
    PullRequestState,
    QueueItem,
    SchedulerContext,
    WorkflowGraph,
)
from domain.policy import latest_by_stage, select_hosted_ready, terminal_result


class CiScheduler:
    def __init__(
        self,
        *,
        settings: Settings,
        store: QueueStore,
        github: GitHubClient,
        graphs: dict[str, WorkflowGraph],
    ) -> None:
        self.settings = settings
        self.store = store
        self.github = github
        self.graphs = graphs

    def _primary_graph(self) -> WorkflowGraph:
        if "omni-ci" in self.graphs:
            return self.graphs["omni-ci"]
        return next(iter(self.graphs.values()))

    async def enqueue_pr(self, pr: PullRequestState) -> list[QueueItem]:
        graph = self._primary_graph()
        items = self.store.enqueue_pr(pr, graph, run_label=self.settings.run_label)
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
        if not self.settings.rerun_enabled:
            return []
        graph = self._primary_graph()
        items = self.store.reset_failed_for_rerun(
            pr, graph, run_label=self.settings.run_label
        )
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
        current = next(
            (item for item in self.store.list_all_items() if item.id == queue_item_id),
            None,
        )
        graph = self._primary_graph()
        if current and status == "passed":
            node = graph.get(current.stage_key)
            declared_outputs = node.job_def.get("outputs") or {}
            if declared_outputs:
                if workflow_run_id is None or not current.dispatch_id:
                    raise RuntimeError(
                        f"completed stage {current.stage_key} is missing workflow output identity"
                    )
                outputs = await self.github.get_scheduler_outputs(
                    repo=current.repo,
                    installation_id=current.installation_id,
                    workflow_run_id=workflow_run_id,
                    dispatch_id=current.dispatch_id,
                )
                missing = sorted(set(declared_outputs) - set(outputs))
                if missing:
                    raise RuntimeError(
                        f"stage {current.stage_key} omitted declared outputs: {', '.join(missing)}"
                    )
                self.store.store_outputs(
                    repo=current.repo,
                    head_sha=current.head_sha,
                    stage_key=current.stage_key,
                    attempt=current.attempt,
                    outputs={key: outputs[key] for key in declared_outputs},
                )

        item = self.store.transition(
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
                output_summary=f"Scheduler marked stage `{item.stage_key}` as `{status}`.",
            )
        await self.dispatch_once()

    async def dispatch_once(self) -> list[int]:
        graph = self._primary_graph()
        dispatched: list[int] = []

        hosted_ids = self._hosted_ready_ids(graph)
        for item_id in hosted_ids:
            item = self.store.transition(item_id, "reserved")
            if item is None:
                continue
            await self._dispatch_item(item, graph, consumes_capacity=False)
            dispatched.append(item_id)

        while True:
            reserved = self.store.reserve_next(
                graph,
                run_label=self.settings.run_label,
                high_priority_label=self.settings.high_priority_label,
                capacity=self.settings.runner_capacity,
                self_hosted_only=True,
            )
            if reserved is None:
                break
            try:
                await self._dispatch_item(reserved, graph, consumes_capacity=True)
                dispatched.append(reserved.id)
            except Exception:
                self.store.reset_to_pending(reserved.id)
                if reserved.check_run_id:
                    await self.github.update_check_run(
                        repo=reserved.repo,
                        installation_id=reserved.installation_id,
                        check_run_id=reserved.check_run_id,
                        status="queued",
                        output_title=reserved.check_name,
                        output_summary="Dispatch failed; stage was returned to the queue.",
                    )
                raise
            if self.settings.runner_capacity <= len(
                [i for i in self.store.list_all_items() if i.status in {"reserved", "dispatched", "running"}]
            ):
                break
        return dispatched

    def _hosted_ready_ids(self, graph: WorkflowGraph) -> list[int]:
        all_items = self.store.list_all_items()
        pr_states = self.store.list_pr_states()
        latest = latest_by_stage(all_items)
        pending = [item for item in all_items if item.status == "pending"]
        return list(
            select_hosted_ready(
                pending,
                pr_states,
                latest,
                graph,
                run_label=self.settings.run_label,
            )
        )

    def build_scheduler_context(self, item: QueueItem, graph: WorkflowGraph) -> SchedulerContext:
        all_items = self.store.list_all_items()
        pr_states = self.store.list_pr_states()
        pr = pr_states.get((item.repo, item.pr_number))
        labels = sorted(pr.labels) if pr else []
        latest = latest_by_stage(
            [i for i in all_items if i.repo == item.repo and i.head_sha == item.head_sha]
        )

        needs: dict[str, NeedState] = {}
        for dep in graph.needs(item.stage_key):
            dep_item = latest.get(dep)
            if dep_item is None:
                continue
            outputs = self.store.get_outputs(
                repo=dep_item.repo,
                head_sha=dep_item.head_sha,
                stage_key=dep_item.stage_key,
                attempt=dep_item.attempt,
            )
            needs[dep] = NeedState(result=terminal_result(dep_item.status), outputs=outputs)

        return SchedulerContext(
            source_event={
                "event_name": "pull_request",
                "number": item.pr_number,
                "pr_number": item.pr_number,
                "head_sha": item.head_sha,
                "head": {"sha": item.head_sha},
                "state": pr.state if pr else "open",
                "draft": pr.draft if pr else False,
                "labels": labels,
                "inputs": {},
            },
            needs=needs,
        )

    async def _dispatch_item(
        self,
        item: QueueItem,
        graph: WorkflowGraph,
        *,
        consumes_capacity: bool,
    ) -> None:
        dispatch_id = new_dispatch_id()
        existing = await self.github.find_workflow_run_by_dispatch_id(
            repo=item.repo,
            installation_id=item.installation_id,
            dispatch_id=dispatch_id,
        )
        if existing is not None:
            return

        context = self.build_scheduler_context(item, graph)
        workflow_path = item.generated_workflow_path
        if not workflow_path:
            raise RuntimeError(f"queue item {item.id} has no generated workflow path")

        workflow_id = Path(workflow_path).name
        if item.check_run_id:
            await self.github.update_check_run(
                repo=item.repo,
                installation_id=item.installation_id,
                check_run_id=item.check_run_id,
                status="in_progress",
                output_title=item.check_name,
                output_summary="Scheduler dispatched this stage.",
            )

        await self.github.dispatch_workflow(
            repo=item.repo,
            installation_id=item.installation_id,
            workflow_id=workflow_id,
            ref=self.settings.dispatch_ref,
            inputs={
                "queue_item_id": str(item.id),
                "dispatch_id": dispatch_id,
                "pr_number": str(item.pr_number),
                "head_sha": item.head_sha,
                "attempt": str(item.attempt),
                "scheduler_context": json.dumps(context.to_json_dict(), sort_keys=True),
                "source_hash": item.source_hash or graph.source_hash,
            },
        )
        self.store.mark_dispatched(item.id, dispatch_id=dispatch_id)

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
                output_summary="CI stage is queued in the global scheduler.",
            )
            self.store.set_check_run_id(item.id, check_run.id)


def load_graphs(workflows_dir: Path, *, roots: tuple[str, ...] | None = None) -> dict[str, WorkflowGraph]:
    from compiler.loader import DEFAULT_SCHEDULER_ROOTS

    return compile_all(
        workflows_dir,
        roots=roots or DEFAULT_SCHEDULER_ROOTS,
    )
