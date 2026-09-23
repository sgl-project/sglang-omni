import asyncio
import json
from contextlib import contextmanager

import pytest

from benchmarks.benchmarker import restage_trial
from benchmarks.benchmarker.data import RequestResult
from sglang_omni.restage.evaluation import SLO, QualityReport


@pytest.mark.asyncio
async def test_trial_stops_server_before_quality_and_saves_requests(
    tmp_path, monkeypatch
):
    running = False

    @contextmanager
    def server(**kwargs):
        nonlocal running
        assert kwargs["wait_for_gpu_release"] is False
        running = True
        try:
            yield
        finally:
            running = False

    monkeypatch.setattr(restage_trial, "managed_omni_server", server)

    def sender(url, audio_dir):
        assert url == "http://127.0.0.1:18000"

        async def send(session, sample):
            assert running
            return RequestResult(request_id=sample, is_success=True)

        return send

    async def quality(results):
        assert not running
        assert (tmp_path / "trial/requests.jsonl").exists()
        return QualityReport(
            {item.request_id: True for item in results}, 0.0, True, 0.2
        )

    result = await restage_trial.execute_trial(
        config_path=tmp_path / "candidate.yaml",
        model_path="dummy",
        samples=["a", "b"],
        send_factory=sender,
        quality=quality,
        slo=SLO(max_latency_s=1),
        rate=100,
        destination=tmp_path / "trial",
        port=18000,
        warmup=0,
    )
    assert result.feasible
    assert result.good_requests == 2
    saved = json.loads((tmp_path / "trial/result.json").read_text())
    assert saved["status"] == "complete"
    assert saved["evaluation"]["feasible"] is True


@pytest.mark.asyncio
async def test_trial_preserves_failure_and_stops_its_server(tmp_path, monkeypatch):
    events = []

    @contextmanager
    def server(**kwargs):
        events.append("start")
        try:
            yield
        finally:
            events.append("stop")

    monkeypatch.setattr(restage_trial, "managed_omni_server", server)

    def sender(url, audio_dir):
        async def send(session, sample):
            raise RuntimeError("sender failed")

        return send

    async def quality(results):
        raise AssertionError("quality must not run after failed dispatch")

    with pytest.raises(RuntimeError, match="sender failed"):
        await restage_trial.execute_trial(
            config_path=tmp_path / "candidate.yaml",
            model_path="dummy",
            samples=["a"],
            send_factory=sender,
            quality=quality,
            slo=SLO(max_latency_s=1),
            rate=100,
            destination=tmp_path / "trial",
            port=18000,
            warmup=0,
        )
    assert events == ["start", "stop"]
    assert (
        json.loads((tmp_path / "trial/result.json").read_text())["status"] == "failed"
    )


@pytest.mark.asyncio
async def test_trial_keeps_completed_request_evidence_after_later_failure(
    tmp_path, monkeypatch
):
    @contextmanager
    def server(**kwargs):
        yield

    monkeypatch.setattr(restage_trial, "managed_omni_server", server)

    def sender(url, audio_dir):
        async def send(session, sample):
            if sample == "fail":
                await asyncio.sleep(0.01)
                raise RuntimeError("later request failed")
            return RequestResult(request_id=sample, is_success=True)

        return send

    with pytest.raises(RuntimeError, match="later request failed"):
        await restage_trial.execute_trial(
            config_path=tmp_path / "candidate.yaml",
            model_path="dummy",
            samples=["completed", "fail"],
            send_factory=sender,
            quality=lambda results: None,
            slo=SLO(max_latency_s=1),
            rate=1000,
            destination=tmp_path / "trial",
            port=18000,
            warmup=0,
            arrival_seed=42,
        )
    rows = [
        json.loads(line)
        for line in (tmp_path / "trial/requests.jsonl").read_text().splitlines()
    ]
    assert [row["request_id"] for row in rows] == ["completed"]
    assert rows[0]["scheduled_s"] <= rows[0]["dispatched_s"] <= rows[0]["completed_s"]
