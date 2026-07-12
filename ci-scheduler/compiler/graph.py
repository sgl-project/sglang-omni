from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from domain.models import DagNode, WorkflowGraph

from .expressions import classify_runner
from .loader import (
    CompilerError,
    DEFAULT_SCHEDULER_ROOTS,
    UnsupportedFeatureError,
    is_reusable_workflow_call,
    load_yaml,
    normalize_needs,
    resolve_workflow_path,
    root_workflow_paths,
    source_hash,
    workflow_stem,
)


GENERATOR_VERSION = 1
GENERATED_PREFIX = "zz_generated_scheduler__"
GENERATED_WORKFLOWS_DIR = ".github/workflows/generated"


@dataclass
class _BuildState:
    nodes: list[DagNode]
    node_by_key: dict[str, DagNode]
    order_counter: int = 0
    visiting: set[str] | None = None


def _generated_workflow_filename(stage_key: str) -> str:
    safe = stage_key.replace("::", "__").replace("/", "_")
    return f"{GENERATED_PREFIX}{safe}.yaml"


def _make_key(prefix: str, job_id: str) -> str:
    if prefix:
        return f"{prefix}::{job_id}"
    return job_id


def _caller_prefix(root: str, caller_job_id: str, called_stem: str) -> str:
    return f"{root}::{caller_job_id}/{called_stem}"


def _map_caller_needs(root: str, caller_needs: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_make_key(root, dep) for dep in caller_needs)


def _terminal_inner_jobs(
    inner_job_ids: list[str],
    inner_needs: dict[str, tuple[str, ...]],
    prefix: str,
) -> tuple[str, ...]:
    depended_on = {dep for deps in inner_needs.values() for dep in deps}
    terminals = [job_id for job_id in inner_job_ids if job_id not in depended_on]
    return tuple(_make_key(prefix, job_id) for job_id in terminals)


def _expand_reusable_workflow(
    *,
    state: _BuildState,
    root: str,
    caller_job_id: str,
    caller_needs: tuple[str, ...],
    uses_path: Path,
    source_path: Path,
    workflow_name: str,
    caller_scope: str,
    workflows_dir: Path,
    repo_root: Path,
    seen_calls: set[tuple[str, str]],
) -> str:
    called_stem = workflow_stem(uses_path)
    virtual_key = _make_key(root, caller_job_id)

    call_sig = (virtual_key, str(uses_path))
    if call_sig in seen_calls:
        raise CompilerError(f"recursive reusable workflow call detected at {virtual_key}")
    seen_calls.add(call_sig)

    called = load_yaml(uses_path)
    inner_jobs = called.get("jobs")
    if not isinstance(inner_jobs, dict):
        raise CompilerError(f"{uses_path} is missing jobs mapping")

    prefix = _caller_prefix(root, caller_job_id, called_stem)
    mapped_caller_needs = _map_caller_needs(root, caller_needs)

    inner_job_ids = list(inner_jobs.keys())
    inner_needs: dict[str, tuple[str, ...]] = {}
    for inner_job_id, inner_job in inner_jobs.items():
        if not isinstance(inner_job, dict):
            raise CompilerError(f"job {inner_job_id!r} in {uses_path} must be a mapping")
        if is_reusable_workflow_call(inner_job):
            raise UnsupportedFeatureError(
                f"nested reusable workflow calls are not supported: {prefix}::{inner_job_id}"
            )
        inner_needs[inner_job_id] = normalize_needs(inner_job.get("needs"))

    for inner_job_id, inner_job in inner_jobs.items():
        local_needs = inner_needs[inner_job_id]
        if not local_needs:
            mapped_needs = mapped_caller_needs
        else:
            mapped_needs = tuple(_make_key(prefix, dep) for dep in local_needs)

        key = _make_key(prefix, inner_job_id)
        if key in state.node_by_key:
            raise CompilerError(f"duplicate stage key: {key}")

        runner_class = classify_runner(inner_job.get("runs-on"))
        if runner_class == "unsupported":
            raise UnsupportedFeatureError(f"unsupported runs-on for {key}")

        check_name = f"{workflow_name} / {inner_job.get('name', inner_job_id)}"
        generated_path = (
            f"{GENERATED_WORKFLOWS_DIR}/{_generated_workflow_filename(key)}"
            if runner_class in {"github-hosted", "self-hosted"}
            else None
        )

        state.order_counter += 1
        node = DagNode(
            key=key,
            root_workflow=root,
            job_id=inner_job_id,
            source_path=str(source_path),
            declaration_order=state.order_counter,
            needs=mapped_needs,
            is_virtual=False,
            is_executable=True,
            runner_class=runner_class,
            check_name=check_name,
            generated_workflow_path=generated_path,
            job_def=inner_job,
        )
        state.nodes.append(node)
        state.node_by_key[key] = node

    terminal_keys = _terminal_inner_jobs(inner_job_ids, inner_needs, prefix)
    state.order_counter += 1
    virtual_node = DagNode(
        key=virtual_key,
        root_workflow=root,
        job_id=caller_job_id,
        source_path=str(source_path),
        declaration_order=state.order_counter,
        needs=terminal_keys,
        is_virtual=True,
        is_executable=False,
        runner_class="github-hosted",
        check_name=f"{workflow_name} / {caller_job_id}",
        generated_workflow_path=None,
        job_def={"uses": str(uses_path)},
    )
    state.nodes.append(virtual_node)
    state.node_by_key[virtual_key] = virtual_node
    seen_calls.remove(call_sig)
    return virtual_key


def _expand_root_jobs(
    *,
    state: _BuildState,
    root: str,
    source_path: Path,
    workflow_data: dict[str, Any],
    workflows_dir: Path,
    repo_root: Path,
) -> None:
    jobs = workflow_data.get("jobs")
    if not isinstance(jobs, dict):
        raise CompilerError(f"{source_path} is missing jobs mapping")

    workflow_name = str(workflow_data.get("name", root))
    seen_calls: set[tuple[str, str]] = set()

    for job_id, job in jobs.items():
        if not isinstance(job, dict):
            raise CompilerError(f"job {job_id!r} in {source_path} must be a mapping")

        caller_needs = normalize_needs(job.get("needs"))

        if is_reusable_workflow_call(job):
            uses_path = resolve_workflow_path(repo_root, job["uses"])
            _expand_reusable_workflow(
                state=state,
                root=root,
                caller_job_id=job_id,
                caller_needs=caller_needs,
                uses_path=uses_path,
                source_path=source_path,
                workflow_name=workflow_name,
                caller_scope=root,
                workflows_dir=workflows_dir,
                repo_root=repo_root,
                seen_calls=seen_calls,
            )
            continue

        key = _make_key(root, job_id)
        if key in state.node_by_key:
            raise CompilerError(f"duplicate stage key: {key}")

        mapped_needs = tuple(_make_key(root, dep) for dep in caller_needs)
        runner_class = classify_runner(job.get("runs-on"))
        if runner_class == "unsupported":
            raise UnsupportedFeatureError(f"unsupported runs-on for {key}")

        check_name = f"{workflow_name} / {job.get('name', job_id)}"
        generated_path = (
            f"{GENERATED_WORKFLOWS_DIR}/{_generated_workflow_filename(key)}"
            if runner_class in {"github-hosted", "self-hosted"}
            else None
        )

        state.order_counter += 1
        node = DagNode(
            key=key,
            root_workflow=root,
            job_id=job_id,
            source_path=str(source_path),
            declaration_order=state.order_counter,
            needs=mapped_needs,
            is_virtual=False,
            is_executable=True,
            runner_class=runner_class,
            check_name=check_name,
            generated_workflow_path=generated_path,
            job_def=job,
        )
        state.nodes.append(node)
        state.node_by_key[key] = node


def _validate_graph(nodes: list[DagNode], node_by_key: dict[str, DagNode]) -> None:
    indegree: dict[str, int] = {node.key: 0 for node in nodes}
    adjacency: dict[str, list[str]] = {node.key: [] for node in nodes}

    for node in nodes:
        for dep in node.needs:
            if dep not in node_by_key:
                raise CompilerError(f"{node.key} depends on missing node {dep!r}")
            adjacency[dep].append(node.key)
            indegree[node.key] += 1

    ready = deque(
        sorted(
            (key for key, degree in indegree.items() if degree == 0),
            key=lambda key: node_by_key[key].declaration_order,
        )
    )
    visited = 0
    while ready:
        key = ready.popleft()
        visited += 1
        for child in sorted(adjacency[key], key=lambda k: node_by_key[k].declaration_order):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)

    if visited != len(nodes):
        raise CompilerError("workflow DAG contains a cycle")


def compile_workflow(
    source_path: Path,
    *,
    workflows_dir: Path | None = None,
) -> WorkflowGraph:
    source_path = source_path.resolve()
    if workflows_dir is None:
        workflows_dir = source_path.parent
    repo_root = workflows_dir.parent.parent
    root = workflow_stem(source_path)
    workflow_data = load_yaml(source_path)

    state = _BuildState(nodes=[], node_by_key={})
    _expand_root_jobs(
        state=state,
        root=root,
        source_path=source_path,
        workflow_data=workflow_data,
        workflows_dir=workflows_dir,
        repo_root=repo_root,
    )
    _validate_graph(state.nodes, state.node_by_key)
    return WorkflowGraph(
        root_workflow=root,
        source_path=str(source_path.relative_to(repo_root)),
        source_hash=source_hash(source_path),
        nodes=tuple(state.nodes),
        node_by_key=state.node_by_key,
    )


def compile_all(
    workflows_dir: Path,
    *,
    roots: tuple[str, ...] = DEFAULT_SCHEDULER_ROOTS,
) -> dict[str, WorkflowGraph]:
    graphs: dict[str, WorkflowGraph] = {}
    for path in root_workflow_paths(workflows_dir, roots=roots):
        if not path.exists():
            raise CompilerError(f"scheduler root workflow not found: {path}")
        graphs[workflow_stem(path)] = compile_workflow(path, workflows_dir=workflows_dir)
    return graphs
