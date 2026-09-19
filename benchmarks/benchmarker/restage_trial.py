"""Run one Restage candidate using the shared serving benchmark machinery."""

import json
import math
from collections.abc import Awaitable, Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.restage import to_observation
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig, SendFn
from benchmarks.benchmarker.utils import managed_omni_server
from sglang_omni.restage.evaluation import SLO, Evaluation, QualityReport, evaluate


def _save_result(destination, metadata):
    (destination / "result.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


async def execute_trial(
    *,
    config_path: Path,
    model_path: str,
    samples: list[Any],
    send_factory: Callable[[str, Path], SendFn],
    quality: Callable[[list[RequestResult]], Awaitable[QualityReport]],
    slo: SLO,
    rate: float,
    destination: Path,
    port: int,
    warmup: int = 1,
    warmup_sample: Any | None = None,
    startup_timeout_s: int = 1800,
    request_timeout_s: int = 300,
    arrival_seed: int | None = None,
) -> Evaluation:
    """Measure an owned service, stop it, then evaluate quality and the joint SLO.

    The caller provides an admitted GPU allocation and a model-specific sender.
    The run is open-loop and quality is scored after the service has stopped,
    so evaluator work never competes with the measured configuration.
    """
    if not samples or not math.isfinite(rate) or rate <= 0:
        raise ValueError("A trial needs samples and a finite positive arrival rate")
    destination.mkdir(parents=True, exist_ok=False)
    audio_dir = destination / "audio"
    audio_dir.mkdir()
    metadata = {
        "status": "running",
        "config_path": str(config_path.resolve()),
        "model_path": model_path,
        "rate": rate,
        "expected_requests": len(samples),
        "slo": asdict(slo),
        "warmup": warmup,
        "arrival_seed": arrival_seed,
    }
    if warmup_sample is not None:
        metadata["warmup_sample"] = (
            asdict(warmup_sample) if is_dataclass(warmup_sample) else warmup_sample
        )
    _save_result(destination, metadata)
    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=0,
            request_rate=rate,
            warmup=warmup,
            disable_tqdm=True,
            timeout_s=request_timeout_s,
            arrival_seed=arrival_seed,
        )
    )
    try:
        send = send_factory(f"http://127.0.0.1:{port}", audio_dir)
        with (destination / "requests.jsonl").open("w", encoding="utf-8") as handle:

            def record(result):
                handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                handle.flush()

            runner.on_result = record
            with managed_omni_server(
                model_path=model_path,
                server_config=str(config_path.resolve()),
                port=port,
                host="127.0.0.1",
                log_file=destination / "server.log",
                timeout=startup_timeout_s,
                wait_for_gpu_release=False,
            ):
                results = await runner.run(samples, send, warmup_sample=warmup_sample)
        metadata.update(measurement_complete=True, elapsed_s=runner.wall_clock_s)
        report = await quality(results)
        (destination / "quality.json").write_text(
            json.dumps(asdict(report), indent=2), encoding="utf-8"
        )
        observations = [
            to_observation(result, quality_pass=report.verdicts.get(result.request_id))
            for result in results
        ]
        evaluation = evaluate(
            observations,
            slo,
            expected_requests=len(samples),
            elapsed_s=runner.wall_clock_s,
            corpus_wer=report.corpus_wer,
            corpus_quality_pass=report.corpus_pass,
        )
        metadata.update(status="complete", evaluation=asdict(evaluation))
        _save_result(destination, metadata)
        return evaluation
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _save_result(destination, metadata)
        raise
