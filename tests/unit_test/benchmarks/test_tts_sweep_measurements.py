# SPDX-License-Identifier: Apache-2.0
"""TTS generation measurements used by the speculative comparison."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset.seedtts import SampleInput
from benchmarks.eval import benchmark_tts_seedtts as tts
from benchmarks.eval import speculative_tts
from benchmarks.eval.speculative_workloads import LoadSettings, TtsWorkload
from benchmarks.metrics.speculative import Leg, Side, compare, read_server_log
from benchmarks.metrics.wer import SampleOutput, calculate_wer_metrics
from benchmarks.tasks.tts import save_wer_results


@pytest.mark.asyncio
@pytest.mark.parametrize(("draft_rate", "expected_k"), [(0.12047, None), (0.12049, 1)])
async def test_k_uses_generation_wall_time_and_peak_uses_only_tts_repeats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    draft_rate: float,
    expected_k: int | None,
) -> None:
    """Both draft rates round to 0.12, but only one clears 1.2x the baseline."""
    sample = SampleInput("sample", "hello", "ref.wav", "hello")
    result = RequestResult(request_id="sample", is_success=True, latency_s=1.0)
    monitor = Mock()
    monitor.start.return_value = monitor
    monitor.stop.side_effect = [
        {"available": True, "gpu_memory_used_mib": {"max": 312}},
        {"available": True, "gpu_memory_used_mib": {"max": 210}},
        {"available": False, "error": "no samples"},
        {"available": False, "error": "no samples"},
    ]
    monkeypatch.setattr(tts, "ResourceMonitor", lambda: monitor)

    def transcribe(command: list[str], log_path: Path, *, check: bool) -> None:
        assert "--transcribe-only" in command
        assert monitor.start.call_count == monitor.stop.call_count
        sample_score = SampleOutput(
            sample_id="sample", is_success=True, is_transcribed=True, hits=1
        )
        save_wer_results(
            [sample_score],
            calculate_wer_metrics([sample_score], "en"),
            {},
            str(log_path.parent),
        )

    monkeypatch.setattr(speculative_tts, "run_client", transcribe)
    settings = LoadSettings(
        concurrencies=[1], repeats=2, warmup=False, max_samples=1, client_arguments=[]
    )
    legs = []
    sides: tuple[Side, Side] = ("baseline", "draft")
    for side, rate in zip(sides, (0.10040, draft_rate), strict=True):
        runner = Mock(wall_clock_s=1 / rate, run=AsyncMock(return_value=[result]))
        monkeypatch.setattr(tts, "BenchmarkRunner", Mock(return_value=runner))
        for repeat in (1, 2):
            directory = tmp_path / side / f"c1_r{repeat}"
            config = tts.TtsSeedttsBenchmarkConfig(
                model="tts",
                meta="seedtts",
                output_dir=str(directory),
                concurrency=1,
                measure_gpu_memory=True,
            )
            await tts.run_tts_seedtts_benchmark(config, samples=[sample])
        loads = speculative_tts.measure_tts(
            TtsWorkload(language="en", error_rate="WER"),
            settings,
            port=8700,
            model_path="tts",
            asr_model_path="asr",
            asr_concurrency=1,
            leg_dir=tmp_path / side,
        )
        assert loads[0].throughput_requests_per_s == pytest.approx(rate)
        assert loads[0].gpu_memory_peak_mib == (312 if side == "baseline" else None)
        legs.append(
            Leg(
                name=side,
                side=side,
                startup_s=0,
                server=read_server_log(""),
                loads=loads,
            )
        )
    assert compare(legs, k_gain=1.2).k == expected_k
    assert monitor.stop.call_count == 4


@pytest.mark.asyncio
async def test_failed_generation_stops_gpu_monitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monitor = Mock()
    monitor.start.return_value = monitor
    monkeypatch.setattr(tts, "ResourceMonitor", lambda: monitor)
    runner = Mock(run=AsyncMock(side_effect=RuntimeError("generation failed")))
    monkeypatch.setattr(tts, "BenchmarkRunner", Mock(return_value=runner))
    config = tts.TtsSeedttsBenchmarkConfig(
        model="tts", meta="seedtts", output_dir=str(tmp_path), measure_gpu_memory=True
    )
    with pytest.raises(RuntimeError, match="generation failed"):
        await tts.run_tts_seedtts_benchmark(config, samples=[])
    monitor.stop.assert_called_once_with()
