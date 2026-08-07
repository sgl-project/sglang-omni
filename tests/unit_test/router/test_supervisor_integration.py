"""Real-process supervisor integration (Linux only: fd passing + UDS).

Spawns the actual CP and DP children, so it exercises the shared listen
socket, the UDS internal channel, registration/heartbeat, crash respawn with
a bumped generation, and clean shutdown.
"""

from __future__ import annotations

import os
import signal
import sys
import time

import httpx
import pytest

from sglang_omni_router.config import RouterConfig, WorkerConfig
from sglang_omni_router.internal_channel import INTERNAL_TOKEN_HEADER
from sglang_omni_router.supervisor import RouterSupervisor

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="fd passing + UDS are Linux-only"
)


def _free_port() -> int:
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def _wait_until(predicate, timeout: float = 20.0, interval: float = 0.25):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(interval)
    raise AssertionError("condition not met within timeout")


@pytest.fixture()
def supervisor():
    config = RouterConfig(
        host="127.0.0.1",
        port=_free_port(),
        workers=[WorkerConfig(url="http://127.0.0.1:1")],  # never healthy; fine
        health_check_interval_secs=1,
    )
    instance = RouterSupervisor(config, router_processes=2)
    instance.start()
    try:
        yield instance, config
    finally:
        instance.shutdown()


def _internal_client(instance: RouterSupervisor) -> httpx.Client:
    context = instance.context
    return httpx.Client(
        transport=httpx.HTTPTransport(uds=context.internal_uds),
        base_url="http://internal",
        headers={INTERNAL_TOKEN_HEADER: context.internal_token},
        timeout=5.0,
    )


def _data_planes(instance: RouterSupervisor) -> list[dict]:
    try:
        with _internal_client(instance) as client:
            return client.get("/internal/data_planes").json()["data_planes"]
    except httpx.HTTPError:
        # Note (Jiaxin Deng): CP still booting (UDS not bound yet) or restarting
        return []


def test_two_dps_share_the_data_socket_and_register(supervisor) -> None:
    instance, config = supervisor
    url = f"http://127.0.0.1:{config.port}/live"

    def _live():
        try:
            return httpx.get(url, timeout=2.0).status_code == 200
        except httpx.HTTPError:
            return False

    _wait_until(_live)
    # Note (Jiaxin Deng): both DPs register and heartbeat over the UDS channel
    records = _wait_until(
        lambda: (lambda planes: planes if len(planes) == 2 else None)(
            _data_planes(instance)
        )
    )
    assert sorted(record["dp_index"] for record in records) == [0, 1]
    assert all(record["generation"] == 1 for record in records)

    # Note (Jiaxin Deng): the data port keeps answering while both children accept from
    # the shared queue
    for _ in range(20):
        assert httpx.get(url, timeout=2.0).status_code == 200


def test_killed_dp_is_respawned_with_a_bumped_generation(supervisor) -> None:
    instance, config = supervisor
    _wait_until(lambda: len(_data_planes(instance)) == 2)

    victim = instance.dp_slots[0].process
    os.kill(victim.pid, signal.SIGKILL)
    _wait_until(lambda: victim.poll() is not None)
    instance.poll_once()

    assert instance.dp_slots[0].generation == 2
    records = _wait_until(
        lambda: (
            lambda planes: (
                planes
                if any(r["dp_index"] == 0 and r["generation"] == 2 for r in planes)
                else None
            )
        )(_data_planes(instance))
    )
    assert any(r["generation"] == 2 for r in records)
    # Note (Jiaxin Deng): data port still serves after the respawn
    url = f"http://127.0.0.1:{config.port}/live"
    _wait_until(lambda: httpx.get(url, timeout=2.0).status_code == 200)


def test_shutdown_leaves_no_children(supervisor) -> None:
    instance, _ = supervisor
    pids = [slot.process.pid for slot in instance.dp_slots.values()]
    assert instance.cp_process is not None
    pids.append(instance.cp_process.pid)

    instance.shutdown()

    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_admin_crud_via_the_data_port_reaches_the_cp(supervisor) -> None:
    instance, config = supervisor
    _wait_until(lambda: len(_data_planes(instance)) == 2)
    base = f"http://127.0.0.1:{config.port}"

    created = httpx.post(
        f"{base}/workers", json={"url": "http://127.0.0.1:2"}, timeout=15.0
    )
    assert created.status_code == 200

    listed = httpx.get(f"{base}/workers", timeout=5.0)
    assert listed.status_code == 200
    urls = {worker["url"] for worker in listed.json()["workers"]}
    assert urls == {"http://127.0.0.1:1", "http://127.0.0.1:2"}

    # Note (Jiaxin Deng): the aggregate /health also rides the forwarding path to the CP
    health = httpx.get(f"{base}/health", timeout=5.0)
    assert health.status_code in (200, 503)
    assert "data_planes" in health.json()


def test_admission_shm_is_wired_end_to_end(supervisor) -> None:
    instance, config = supervisor
    _wait_until(lambda: len(_data_planes(instance)) == 2)
    base = f"http://127.0.0.1:{config.port}"

    def _health():
        try:
            return httpx.get(f"{base}/health", timeout=5.0).json()
        except httpx.HTTPError:
            return {}

    payload = _wait_until(
        lambda: (lambda p: p if p.get("admission_slots") else None)(_health())
    )
    assert [slot["generation"] for slot in payload["admission_slots"]] == [1, 1]
    assert payload["admission"]["inflight"] == 0
    assert payload["live_data_planes"] == 2

    # Note (Jiaxin Deng): SIGKILL one DP: the supervisor reclaims its slot after the
    # reap and the respawned generation-2 process claims it again
    victim = instance.dp_slots[0].process
    os.kill(victim.pid, signal.SIGKILL)
    _wait_until(lambda: victim.poll() is not None)
    instance.poll_once()

    _wait_until(
        lambda: _health().get("admission_slots", [{}])[0].get("generation") == 2
    )


def _has_exited(pid: int) -> bool:
    """True when the pid is gone or a zombie.

    Children orphaned by a dead supervisor are reparented to PID 1, which in a
    container often does not reap; an exited orphan then lingers as a zombie
    and both kill(pid, 0) and Popen.poll() still report it as running.
    """
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            return f.read().rsplit(") ", 1)[1].split(" ", 1)[0] == "Z"
    except FileNotFoundError:
        return True


def test_children_exit_on_supervisor_death_even_with_a_stuck_relay() -> None:
    # Note (Jiaxin Deng): a SIGKILLed supervisor runs no teardown, so each child must
    # exit on the death-pipe EOF; otherwise an orphan keeps serving on the inherited
    # public listener. A request that never completes must not hold that exit off.
    import socket
    import threading

    upstream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    upstream.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    upstream.bind(("127.0.0.1", 0))
    upstream.listen(8)
    held: list[socket.socket] = []
    stop = threading.Event()

    def _accept_and_never_answer() -> None:
        while not stop.is_set():
            try:
                conn, _ = upstream.accept()
            except OSError:
                return
            conn.recv(65536)  # read the request, never reply
            held.append(conn)

    threading.Thread(target=_accept_and_never_answer, daemon=True).start()
    config = RouterConfig(
        host="127.0.0.1",
        port=_free_port(),
        workers=[WorkerConfig(url=f"http://127.0.0.1:{upstream.getsockname()[1]}")],
    )
    instance = RouterSupervisor(config, router_processes=1)
    instance.start()
    children = [slot.process.pid for slot in instance.dp_slots.values()]
    children.append(instance.cp_process.pid)
    base = f"http://127.0.0.1:{config.port}"
    try:
        _wait_until(lambda: httpx.get(f"{base}/live", timeout=2.0).status_code == 200)
        stuck = threading.Thread(
            target=lambda: httpx.post(
                f"{base}/v1/chat/completions",
                json={"model": "m", "messages": []},
                timeout=60.0,
            ),
            daemon=True,
        )
        stuck.start()
        time.sleep(1.0)  # let the relay reach the upstream that never answers

        # Note (Jiaxin Deng): the supervisor holds the only write end; closing it is
        # exactly what its SIGKILL does to the children
        os.close(instance._death_pipe_write)
        instance._death_pipe_write = None
        _wait_until(
            lambda: all(_has_exited(pid) for pid in children) or None, timeout=30.0
        )
    finally:
        stop.set()
        upstream.close()
        instance.shutdown()
