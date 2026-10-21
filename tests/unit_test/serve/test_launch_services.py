# SPDX-License-Identifier: Apache-2.0
"""The services a launch.yaml declares: started in order, reused, stopped together."""

import os
import socket
import sys

import pytest

from sglang_omni.config.schema import PipelineConfig, ServiceConfig, StageConfig
from sglang_omni.serve import services as services_module
from sglang_omni.serve.services import ServiceProcesses, is_serving

FAKE_SERVICE = """
import http.server, sys, time
time.sleep(float(sys.argv[2]))
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/v1/models" else 404)
        self.end_headers()
    def log_message(self, *args):
        pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
"""


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def fake_service(name: str, port: int, delay_s: float = 0.0) -> ServiceConfig:
    return ServiceConfig(
        name=name,
        command=[sys.executable, "-c", FAKE_SERVICE, str(port), str(delay_s)],
        port=port,
    )


def exiting_service(name: str, port: int, code: int) -> ServiceConfig:
    return ServiceConfig(
        name=name,
        command=[sys.executable, "-c", f"import sys; sys.exit({code})"],
        port=port,
    )


def is_process_group_alive(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch):
    monkeypatch.setattr(services_module, "POLL_INTERVAL_S", 0.05)
    monkeypatch.setattr(services_module, "STOP_GRACE_S", 5.0)


@pytest.mark.asyncio
async def test_services_start_in_order_and_stop_together():
    first, second = fake_service("a", free_port(), 0.3), fake_service("b", free_port())
    async with ServiceProcesses([first, second]) as services:
        assert [service.name for service, _ in services.started] == ["a", "b"]
        assert is_serving(first.port) and is_serving(second.port)
        pids = [process.pid for _, process in services.started]
    assert not any(is_process_group_alive(pid) for pid in pids)
    assert not is_serving(first.port) and not is_serving(second.port)


@pytest.mark.asyncio
async def test_a_service_already_serving_is_reused_not_started():
    port = free_port()
    async with ServiceProcesses([fake_service("a", port)]) as running:
        async with ServiceProcesses([fake_service("a", port)]) as services:
            assert services.started == []
        assert is_serving(port)
        assert running.started[0][1].poll() is None


@pytest.mark.asyncio
async def test_a_service_exiting_before_ready_fails_the_start_and_stops_the_others():
    first = fake_service("a", free_port())
    broken = exiting_service("b", free_port(), 3)
    services = ServiceProcesses([first, broken])
    with pytest.raises(RuntimeError, match="'b' exited with code 3 before"):
        async with services:
            pass
    assert services.started == []
    assert not is_serving(first.port)


@pytest.mark.asyncio
async def test_watch_raises_when_a_started_service_exits():
    service = fake_service("a", free_port())
    async with ServiceProcesses([service]) as services:
        services.started[0][1].terminate()
        with pytest.raises(RuntimeError, match="'a' exited with code"):
            await services.watch()


def test_services_need_distinct_names_and_ports():
    def config(services):
        return PipelineConfig(
            model_path="m",
            stages=[StageConfig(name="s", factory_path="tests.fake:create_stage")],
            services=services,
        )

    base = {"command": ["true"], "port": 9000}
    with pytest.raises(ValueError, match="distinct names and ports"):
        config([{"name": "a", **base}, {"name": "a", **base, "port": 9001}])
    with pytest.raises(ValueError, match="distinct names and ports"):
        config([{"name": "a", **base}, {"name": "b", **base}])
    with pytest.raises(ValueError):
        ServiceConfig(name="a", command=[], port=9000)
