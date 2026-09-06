# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang_omni.serve import boot_warmup


@pytest.mark.asyncio
async def test_boot_warmup_temporary_directory_failure_is_nonfatal(monkeypatch, caplog):
    def fail(**kwargs):
        raise OSError("No writable temporary directory")

    monkeypatch.setattr(boot_warmup.tempfile, "TemporaryDirectory", fail)
    await boot_warmup.run_boot_warmup(None, model_name="test", num_requests=1)
    assert "could not create temporary inputs" in caplog.text


@pytest.mark.asyncio
async def test_boot_warmup_runs_concurrent_distinct_requests_and_removes_inputs(caplog):
    caplog.set_level("INFO", logger=boot_warmup.__name__)
    entered, closed = [], []
    ready = asyncio.Event()

    async def generate(request, *, request_id):
        path = Path(request.metadata["audios"][0])
        entered.append((request, path, hashlib.sha256(path.read_bytes()).hexdigest()))
        if len(entered) == 3:
            ready.set()
        try:
            await asyncio.wait_for(ready.wait(), timeout=2)
            yield None
        finally:
            closed.append(request_id)

    await boot_warmup.run_boot_warmup(
        SimpleNamespace(generate=generate), model_name="test", num_requests=3
    )
    assert len(closed) == 3
    assert len({digest for _, _, digest in entered}) == 3
    assert len({request.messages[0].content for request, _, _ in entered}) == 3
    assert all(not path.exists() for _, path, _ in entered)
    assert "3/3 request(s)" in caplog.text
    for request, _, _ in entered:
        assert request.stream and request.output_modalities == ["text", "audio"]
        assert request.extra_params["talker_min_new_tokens"] == 32
        assert request.extra_params["talker_max_new_tokens"] == 32


@pytest.mark.asyncio
async def test_boot_warmup_stream_failure_closes_stream_and_keeps_serving(caplog):
    closed = []

    async def generate(request, *, request_id):
        try:
            yield None
            raise RuntimeError("synthetic request failure")
        finally:
            closed.append(request_id)

    await boot_warmup.run_boot_warmup(
        SimpleNamespace(generate=generate), model_name="test", num_requests=2
    )
    assert len(closed) == 2
    assert "every request failed" in caplog.text
    assert "synthetic request failure" in caplog.text


@pytest.mark.asyncio
async def test_boot_warmup_timeout_cancels_and_closes_all_streams(monkeypatch, caplog):
    monkeypatch.setattr(boot_warmup, "_TIMEOUT_S", 0.02)
    entered, closed = [], []
    never = asyncio.Event()

    async def generate(request, *, request_id):
        entered.append(request_id)
        try:
            await never.wait()
            yield None
        finally:
            closed.append(request_id)

    await boot_warmup.run_boot_warmup(
        SimpleNamespace(generate=generate), model_name="test", num_requests=2
    )
    assert len(entered) == 2 and set(closed) == set(entered)
    assert "serving anyway" in caplog.text


@pytest.mark.asyncio
async def test_boot_warmup_caller_cancellation_propagates_and_closes_streams():
    entered, closed, paths = [], [], []
    ready, never = asyncio.Event(), asyncio.Event()

    async def generate(request, *, request_id):
        entered.append(request_id)
        paths.append(Path(request.metadata["audios"][0]))
        if len(entered) == 2:
            ready.set()
        try:
            await never.wait()
            yield None
        finally:
            closed.append(request_id)

    task = asyncio.create_task(
        boot_warmup.run_boot_warmup(
            SimpleNamespace(generate=generate), model_name="test", num_requests=2
        )
    )
    try:
        await asyncio.wait_for(ready.wait(), timeout=2)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(entered) == 2 and set(closed) == set(entered)
    assert all(not path.exists() for path in paths)
