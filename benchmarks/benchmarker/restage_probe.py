# SPDX-License-Identifier: Apache-2.0
"""Two probes on one GPU that calibrate the Restage capacity constants."""

from __future__ import annotations

import json
import statistics
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from benchmarks.benchmarker.restage_corpus import repeat_corpus
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import managed_omni_server
from benchmarks.dataset.seedtts import SampleInput
from benchmarks.tasks.asr import make_asr_send_fn
from benchmarks.tasks.tts import make_tts_send_fn
from sglang_omni.restage.calibration import Constants, StageConstants


def detect_gpu() -> tuple[str, float] | None:
    """Name and memory of the first visible GPU from nvidia-smi, or None."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        name, mib = out.stdout.strip().splitlines()[0].rsplit(",", 1)
        return name.strip(), float(mib) / 1024.0
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _sender(task, model_path, url, lang, stream, audio_dir, sender_options):
    if task == "asr":
        return make_asr_send_fn(
            model_path, f"{url}/v1/audio/transcriptions", lang=lang, stream=stream
        )
    return make_tts_send_fn(
        model_path,
        f"{url}/v1/audio/speech",
        save_audio_dir=str(audio_dir),
        stream=stream,
        **sender_options,
    )


def _percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


async def _closed_loop(runner_cfg, samples, send, on_result=None):
    runner = BenchmarkRunner(runner_cfg, on_result=on_result)
    results = await runner.run(samples, send)
    return results, runner.wall_clock_s


async def calibrate(
    *,
    config_path: Path,
    model_path: str,
    task: str,
    samples: list[SampleInput],
    stages: dict[str, dict[str, Any]],
    destination: Path,
    port: int,
    lang: str,
    slo_rtf: float = 1.0,
    context_tokens: int | None = None,
    single_requests: int = 3,
    concurrencies: tuple[int, ...] = (4, 8, 16),
    saturation_min_requests: int = 128,
    sender_options: dict[str, Any] | None = None,
    gpu: tuple[str, float] | None = None,
    startup_timeout_s: int = 1800,
    request_timeout_s: int = 300,
) -> Constants:
    """Measure the shipped pipeline once and write constants.json.

    The single-request probe gives the per-arrival stall (first output at
    concurrency 1) and the engine's share of service time from the
    ``X-Engine-Time`` header. The saturation probe runs closed-loop at each
    concurrency over at least ``saturation_min_requests`` requests and keeps
    the best throughput among those that completed every request inside the
    SLO. ``stages`` supplies the geometry the probes cannot see:
    ``weights_gib`` per GPU stage and ``kv_bytes_per_token`` on the engine.
    A stage marked ``"binding": false`` takes no share of the service time.
    """
    if task not in ("tts", "asr"):
        raise ValueError(f"Unsupported calibration task: {task}")
    if not samples or single_requests < 1 or not concurrencies:
        raise ValueError("Calibration needs samples, single requests and concurrencies")
    if gpu is None:
        gpu = detect_gpu()
    if gpu is None:
        raise ValueError("Pass gpu=(name, mem_gib): no GPU is visible to nvidia-smi")
    gpu_name, gpu_mem_gib = gpu
    destination.mkdir(parents=True, exist_ok=False)
    audio_dir = destination / "audio"
    audio_dir.mkdir()
    url = f"http://127.0.0.1:{port}"
    sender_options = sender_options or {}
    evidence: dict[str, Any] = {"single": [], "saturation": []}
    with managed_omni_server(
        model_path=model_path,
        server_config=str(config_path.resolve()),
        port=port,
        host="127.0.0.1",
        log_file=destination / "server.log",
        timeout=startup_timeout_s,
        wait_for_gpu_release=False,
    ):
        streaming = _sender(
            task, model_path, url, lang, True, audio_dir, sender_options
        )
        single, _ = await _closed_loop(
            RunConfig(
                max_concurrency=1,
                warmup=1,
                disable_tqdm=True,
                timeout_s=request_timeout_s,
            ),
            samples[:single_requests],
            streaming,
        )
        evidence["single"] = [asdict(r) for r in single]
        plain = _sender(task, model_path, url, lang, False, audio_dir, sender_options)
        # Note (Jiaxin Deng): X-Engine-Time is final only on non-streaming
        # responses, so the engine share needs a second concurrency-1 pass.
        single_plain, _ = await _closed_loop(
            RunConfig(
                max_concurrency=1,
                warmup=0,
                disable_tqdm=True,
                timeout_s=request_timeout_s,
            ),
            samples[:single_requests],
            plain,
        )
        evidence["single_plain"] = [asdict(r) for r in single_plain]
        repeats = max(1, -(-saturation_min_requests // len(samples)))
        corpus, _ = repeat_corpus(samples, repeats)
        for concurrency in concurrencies:
            results, elapsed = await _closed_loop(
                RunConfig(
                    max_concurrency=concurrency,
                    warmup=1,
                    disable_tqdm=True,
                    timeout_s=request_timeout_s,
                ),
                corpus,
                plain,
            )
            good = [r for r in results if r.is_success and r.audio_duration_s > 0]
            audio = sum(r.audio_duration_s for r in good)
            evidence["saturation"].append(
                {
                    "concurrency": concurrency,
                    "requests": len(results),
                    "successful": len(good),
                    "errors": sorted({r.error for r in results if not r.is_success})[
                        :5
                    ],
                    "elapsed_s": elapsed,
                    "throughput_audio_s_per_s": audio / elapsed if elapsed else 0.0,
                    "rtf_p99": (
                        _percentile(
                            [r.latency_s / r.audio_duration_s for r in good], 0.99
                        )
                        if good
                        else None
                    ),
                }
            )
    (destination / "probes.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    constants = _fit(
        evidence,
        stages=stages,
        model_path=model_path,
        gpu_name=gpu_name,
        gpu_mem_gib=gpu_mem_gib,
        slo_rtf=slo_rtf,
        context_tokens=context_tokens,
        samples=samples,
    )
    constants.save(destination / "constants.json")
    return constants


def _fit(
    evidence,
    *,
    stages,
    model_path,
    gpu_name,
    gpu_mem_gib,
    slo_rtf,
    context_tokens,
    samples,
):
    single = [
        r for r in evidence["single"] if r["is_success"] and r["audio_duration_s"] > 0
    ]
    if not single:
        raise ValueError("Every single-request probe failed; see probes.json")
    audio_seconds = statistics.median(r["audio_duration_s"] for r in single)
    stalls = [
        r[key]
        for r in single
        for key in ("audio_ttfp_s", "text_ttft_s")
        if r.get(key) is not None
    ]
    delta_s = statistics.median(stalls) if stalls else 0.0
    shares = [
        r["engine_time_s"] / r["latency_s"]
        for r in evidence.get("single_plain", single)
        if r["is_success"] and r["engine_time_s"] > 0 and r["latency_s"] > 0
    ]
    engine_share = min(1.0, statistics.median(shares)) if shares else 1.0
    feasible = [
        row
        for row in evidence["saturation"]
        if row["rtf_p99"] is not None
        and row["rtf_p99"] <= slo_rtf
        and row["successful"] == row["requests"]
    ]
    if feasible:
        # Note (Jiaxin Deng): throughput can fall past the knee while rtf still
        # fits, so the saturated value is the best feasible point, not the last.
        chosen = max(
            feasible,
            key=lambda row: (row["throughput_audio_s_per_s"], row["concurrency"]),
        )
        note = (
            "SLO boundary not bracketed above"
            if feasible[-1] is evidence["saturation"][-1]
            else "inside SLO"
        )
    else:
        chosen = evidence["saturation"][0]
        note = "no probe met the SLO, lowest concurrency kept"
    throughput = chosen["throughput_audio_s_per_s"]
    if throughput <= 0:
        raise ValueError("Saturation probe measured no throughput; see probes.json")
    provenance = f"MEASURED saturation c={chosen['concurrency']} rtf_p99={chosen['rtf_p99']:.3f} ({note})"
    engines = [name for name, spec in stages.items() if spec.get("kv_bytes_per_token")]
    if len(engines) != 1:
        raise ValueError("Exactly one stage must declare kv_bytes_per_token")
    engine = engines[0]
    others = [
        name
        for name, spec in stages.items()
        if name != engine and spec.get("binding", True)
    ]
    rows = {}
    for name, spec in stages.items():
        if name == engine:
            stage_throughput = throughput / engine_share
            share = engine_share
        elif name in others and engine_share < 1.0:
            # Note (Jiaxin Deng): the probe sees one non-engine share; several
            # tails split it evenly until a measured split cell refines them.
            share = (1.0 - engine_share) / len(others)
            stage_throughput = throughput / share
        else:
            share = 0.0
            stage_throughput = None
        rows[name] = StageConstants(
            throughput=stage_throughput,
            provenance=(
                f"PREDICTED share {share:.2f} of {provenance}"
                if stage_throughput is not None
                else "PRIOR non-binding"
            ),
            weights_gib=float(spec["weights_gib"]),
            kv_bytes_per_token=spec.get("kv_bytes_per_token"),
            delta_s=delta_s if name == engine else 0.0,
        )
    if context_tokens is None:
        context_tokens = max(
            32,
            round(
                statistics.mean(len(s.target_text or s.ref_text) for s in samples) / 3
            ),
        )
    return Constants(
        model_path=model_path,
        gpu_name=gpu_name,
        gpu_mem_gib=gpu_mem_gib,
        context_tokens=context_tokens,
        audio_seconds=audio_seconds,
        slo_rtf=slo_rtf,
        pipeline_throughput=throughput,
        pipeline_provenance=provenance,
        stages=rows,
    )


def is_media_reference(value) -> bool:
    return urlparse(str(value)).scheme in {"http", "https", "data", "file"}


def load_calibration_spec(path: Path) -> dict[str, Any]:
    """Load a calibration spec; sample and audio paths resolve relative to it."""
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
        base = path.resolve().parent
        options = {
            "model_path": spec["model_path"],
            "task": spec.get("task", "tts"),
            "stages": spec["stages"],
            "port": spec.get("port", 8140),
            "lang": spec.get("lang", "en"),
            "slo_rtf": spec.get("slo_rtf", 1.0),
            "context_tokens": spec.get("context_tokens"),
            "single_requests": spec.get("single_requests", 3),
            "concurrencies": tuple(spec.get("concurrencies", (4, 8, 16))),
            "saturation_min_requests": spec.get("saturation_min_requests", 128),
            "sender_options": spec.get("sender_options") or {},
        }
        if spec.get("gpu") is not None:
            options["gpu"] = (spec["gpu"]["name"], float(spec["gpu"]["mem_gib"]))
        samples = load_samples(spec["samples"], base)
        for sample in samples:
            if sample.ref_audio and not is_media_reference(sample.ref_audio):
                if not Path(sample.ref_audio).is_file():
                    raise ValueError(
                        f"Reference audio does not exist: {sample.ref_audio}"
                    )
        options["samples"] = samples
        return options
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Invalid calibration spec {path}: {exc}") from exc


def load_samples(value, base: Path) -> list[SampleInput]:
    """Explicit sample rows, or a SeedTTS source staged by the dataset loader."""
    if isinstance(value, dict):
        from benchmarks.dataset.seedtts import load_seedtts_samples

        source = value["source"]
        if not is_media_reference(source) and (base / source).exists():
            source = str(base / source)
        return load_seedtts_samples(
            source, value.get("max_samples"), split=value.get("split", "en")
        )
    samples = [SampleInput(**row) for row in value]
    for sample in samples:
        if sample.ref_audio and not is_media_reference(sample.ref_audio):
            sample.ref_audio = str(base / Path(sample.ref_audio).expanduser())
    return samples
