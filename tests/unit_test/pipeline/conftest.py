# SPDX-License-Identifier: Apache-2.0
"""One stage-process set per linear topology, shared by the session tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
import pytest_asyncio

from tests.unit_test.fixtures.session_pipeline import (
    PipelineResources,
    event_log,
    pipeline,
)


@asynccontextmanager
async def reuse_pipeline(
    resources: PipelineResources,
) -> AsyncIterator[PipelineResources]:
    """Reuse live workers. Drop coordinator session bookkeeping between tests."""
    coordinator, events, processes = resources
    event_log(events)
    assert not coordinator.is_sessions_stopping
    assert all(process.is_alive() for process in processes)
    try:
        yield resources
    finally:
        for session in list(coordinator.sessions.values()):
            if session.is_closed:
                continue
            await coordinator.close_session(session.session_identity)
        coordinator.sessions.clear()
        coordinator.session_unavailable_stages.clear()
        event_log(events)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def shared_linear_pair(
    tmp_path_factory: pytest.TempPathFactory,
) -> AsyncIterator[PipelineResources]:
    async with pipeline(tmp_path_factory.mktemp("linear-pair")) as resources:
        yield resources


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def shared_linear_triple(
    tmp_path_factory: pytest.TempPathFactory,
) -> AsyncIterator[PipelineResources]:
    async with pipeline(
        tmp_path_factory.mktemp("linear-triple"), stage_count=3
    ) as resources:
        yield resources


@pytest_asyncio.fixture(loop_scope="session")
async def linear_pair(
    shared_linear_pair: PipelineResources,
) -> AsyncIterator[PipelineResources]:
    async with reuse_pipeline(shared_linear_pair) as resources:
        yield resources


@pytest_asyncio.fixture(loop_scope="session")
async def linear_triple(
    shared_linear_triple: PipelineResources,
) -> AsyncIterator[PipelineResources]:
    async with reuse_pipeline(shared_linear_triple) as resources:
        yield resources
