from __future__ import annotations

import httpx
import pytest
from pydantic import ValidationError

from sglang_omni_router.config import RouterConfig, WorkerConfig
from sglang_omni_router.health import HealthChecker
from sglang_omni_router.selector import NoEligibleWorkerError, WorkerSelector
from sglang_omni_router.worker import build_workers


@pytest.mark.parametrize(
    "url",
    [
        "ftp://127.0.0.1:8101",
        "http://user:pass@127.0.0.1:8101",
        "http://127.0.0.1:8101/path",
        "http://127.0.0.1:8101?x=1",
        "http://169.254.169.254",
        "http://169.254.1.1",
    ],
)
def test_worker_url_validation_rejects_invalid_urls(url: str) -> None:
    with pytest.raises(ValidationError):
        WorkerConfig(url=url)


def test_router_config_rejects_duplicate_urls_after_normalization() -> None:
    with pytest.raises(ValidationError, match="duplicate worker URLs"):
        RouterConfig(
            worker_urls=[
                WorkerConfig(url="HTTP://LOCALHOST:8101/"),
                WorkerConfig(url="http://localhost:8101"),
            ]
        )


def test_selector_filters_by_health_and_capability() -> None:
    workers = build_workers(
        [
            WorkerConfig(url="http://127.0.0.1:8101", capabilities={"speech"}),
            WorkerConfig(url="http://127.0.0.1:8102", capabilities={"chat"}),
            WorkerConfig(url="http://127.0.0.1:8103", capabilities={"speech"}),
        ]
    )
    workers[0].state = "unhealthy"
    workers[1].state = "healthy"
    workers[2].state = "healthy"

    selector = WorkerSelector("round_robin")

    assert selector.select(workers, capability="speech").url == "http://127.0.0.1:8103"


def test_round_robin_recomputes_candidates_after_health_change() -> None:
    workers = build_workers(
        [
            WorkerConfig(url="http://127.0.0.1:8101"),
            WorkerConfig(url="http://127.0.0.1:8102"),
        ]
    )
    for worker in workers:
        worker.state = "healthy"

    selector = WorkerSelector("round_robin")

    assert selector.select(workers, capability="speech").url == "http://127.0.0.1:8101"
    workers[0].state = "unhealthy"
    assert selector.select(workers, capability="speech").url == "http://127.0.0.1:8102"
    workers[1].state = "unhealthy"
    with pytest.raises(NoEligibleWorkerError):
        selector.select(workers, capability="speech")


def test_least_request_selects_lowest_active_request_count() -> None:
    workers = build_workers(
        [
            WorkerConfig(url="http://127.0.0.1:8101"),
            WorkerConfig(url="http://127.0.0.1:8102"),
        ]
    )
    for worker in workers:
        worker.state = "healthy"
    workers[0].active_requests = 2
    workers[1].active_requests = 2

    selector = WorkerSelector("least_request")

    assert selector.select(workers, capability="speech").url == "http://127.0.0.1:8101"

    workers[0].active_requests = 3
    assert selector.select(workers, capability="speech").url == "http://127.0.0.1:8102"


def test_worker_request_guard_cleans_up_count() -> None:
    worker = build_workers([WorkerConfig(url="http://127.0.0.1:8101")])[0]

    with pytest.raises(RuntimeError):
        with worker.request_guard():
            assert worker.active_requests == 1
            raise RuntimeError("boom")

    assert worker.active_requests == 0


@pytest.mark.asyncio
async def test_health_checker_uses_failure_and_success_thresholds() -> None:
    statuses = iter([500, 500, 200, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(next(statuses), request=request)

    worker = build_workers([WorkerConfig(url="http://worker.local:8101")])[0]
    config = RouterConfig(
        worker_urls=[WorkerConfig(url="http://worker.local:8101")],
        health_failure_threshold=2,
        health_success_threshold=2,
    )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        checker = HealthChecker(workers=[worker], config=config, client=client)

        await checker.check_once()
        assert worker.state == "unknown"
        await checker.check_once()
        assert worker.state == "unhealthy"
        await checker.check_once()
        assert worker.state == "unhealthy"
        await checker.check_once()
        assert worker.state == "healthy"
