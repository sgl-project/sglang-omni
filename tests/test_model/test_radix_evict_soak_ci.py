# SPDX-License-Identifier: Apache-2.0
"""Nightly radix evict soak: Qwen3-ASR-1.7B under zero-hit saturation.

Salted zero-hit traffic (a seeded noise tail per request gives every clip a
unique fingerprint `extra_key`) fills a capped radix pool past saturation
(fill volume > 1.4x pool), then gates the post-saturation window against the
pre-saturation window of the same run:

- post/pre qps ratio: eviction-path slowdown (#1723) or upstream
  allocator/evict API drift.
- post/pre p99 ratio: eviction stalls surfacing as tail latency first.
- post decay ratio (last/first post loop): slow-leak degradation that
  worsens per loop.
- pre qps sanity floor: the run itself is broken; prevents vacuous passes.

Radix is force-enabled here (the pipeline default is off); the guard covers
deployments that keep it on.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.eval.benchmark_asr_seedtts import (
    QWEN3_ASR_MODEL_PATH,
    run_asr_seedtts_once,
)
from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    launch_managed_router,
)
from tests.utils import MetricCheckCollector

RADIX_POOL_TOKENS = 131072
RADIX_CI_CONCURRENCY = 32
BASE_SAMPLES = 1088
FILL_UNIQUE_REQUESTS = 24000  # >= 1.4x pool at 8 unique tokens/request
POST_LOOPS = 3
STARTUP_TIMEOUT = 600
NOISE_TAIL_SECONDS = 0.02
NOISE_AMPLITUDE = 1e-3

MIN_POST_PRE_QPS_RATIO = 0.90
MAX_POST_PRE_P99_RATIO = 1.5
MIN_POST_DECAY_RATIO = 0.90  # last post loop vs first: catches slow decay
MIN_SANE_PRE_QPS = 100.0

WORKER_EXTRA_ARGS = (
    "--asr.engine.disable_radix_cache false "
    f"--asr.engine.max_total_tokens {RADIX_POOL_TOKENS}"
)


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the radix evict soak CI")


@pytest.fixture(scope="module")
def base_samples() -> list[SampleInput]:
    return load_seedtts_samples(
        DATASETS["seedtts"],
        max_samples=BASE_SAMPLES,
        split="en",
    )


@pytest.fixture(scope="module")
def asr_router_server(
    tmp_path_factory: pytest.TempPathFactory,
) -> ManagedRouterHandle:
    with launch_managed_router(
        tmp_path_factory=tmp_path_factory,
        model_path=QWEN3_ASR_MODEL_PATH,
        model_name=QWEN3_ASR_MODEL_PATH,
        worker_extra_args=WORKER_EXTRA_ARGS,
        num_workers=1,
        wait_timeout=STARTUP_TIMEOUT,
        log_prefix="radix_evict_soak_router_logs",
    ) as router:
        yield router


def _salted_loop(
    base: list[SampleInput], out_dir: Path, loop_index: int
) -> list[SampleInput]:
    """Append a seeded noise tail to every clip: new fingerprint per request."""
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    salted = []
    for idx, sample in enumerate(base):
        audio, sr = sf.read(sample.ref_audio, dtype="float32")
        rng = np.random.default_rng(loop_index * 1_000_003 + idx)
        tail = (rng.standard_normal(int(sr * NOISE_TAIL_SECONDS)) * NOISE_AMPLITUDE)
        out_path = out_dir / f"loop{loop_index}_{idx}.wav"
        sf.write(out_path, np.concatenate([audio, tail.astype(np.float32)]), sr)
        salted.append(
            SampleInput(
                sample_id=f"{sample.sample_id}-l{loop_index}",
                ref_text=sample.ref_text,
                ref_audio=str(out_path),
                target_text=sample.target_text,
            )
        )
    return salted


def _run_loop(
    router: ManagedRouterHandle,
    samples: list[SampleInput],
    incomplete: list,
    label: str,
) -> dict:
    results = asyncio.run(
        run_asr_seedtts_once(
            samples,
            host="127.0.0.1",
            port=router.port,
            model_path=QWEN3_ASR_MODEL_PATH,
            lang="en",
            concurrency=RADIX_CI_CONCURRENCY,
        )
    )
    summary = results["summary"]
    if summary["evaluated"] != len(samples) or summary["skipped"] != 0:
        incomplete.append((label, summary["evaluated"], summary["skipped"]))
    return results["speed"]


def test_radix_evict_soak_throughput_holds(
    base_samples: list[SampleInput],
    asr_router_server: ManagedRouterHandle,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    _require_cuda()
    checks = MetricCheckCollector(label="Radix evict soak")
    audio_root = tmp_path_factory.mktemp("salted_audio")

    import shutil

    shutil.rmtree("/tmp/radix_evict_soak", ignore_errors=True)
    incomplete: list = []
    # Loop 0 absorbs warmup effects; loop 1 is the pre-saturation window.
    _run_loop(asr_router_server, _salted_loop(base_samples, audio_root / "l0", 0), incomplete, "warmup")
    pre = _run_loop(asr_router_server, _salted_loop(base_samples, audio_root / "l1", 1), incomplete, "pre")

    sent = 2 * len(base_samples)
    loop_index = 2
    while sent < FILL_UNIQUE_REQUESTS:
        _run_loop(
            asr_router_server,
            _salted_loop(base_samples, audio_root / f"l{loop_index}", loop_index),
            incomplete,
            f"fill{loop_index}",
        )
        sent += len(base_samples)
        loop_index += 1

    post_qps, post_p99 = [], []
    for _ in range(POST_LOOPS):
        speed = _run_loop(
            asr_router_server,
            _salted_loop(base_samples, audio_root / f"l{loop_index}", loop_index),
            incomplete,
            f"post{loop_index}",
        )
        post_qps.append(speed["throughput_samples_per_s"])
        post_p99.append(speed["latency_p99_s"])
        loop_index += 1

    pre_qps = pre["throughput_samples_per_s"]
    pre_p99 = pre["latency_p99_s"]
    avg_post_qps = sum(post_qps) / len(post_qps)
    max_post_p99 = max(post_p99)

    gate_rows = [
        ("pre qps sanity", pre_qps, f">= {MIN_SANE_PRE_QPS}"),
        ("post/pre qps", avg_post_qps / pre_qps, f">= {MIN_POST_PRE_QPS_RATIO}"),
        ("post/pre p99", max_post_p99 / pre_p99, f"<= {MAX_POST_PRE_P99_RATIO}"),
        ("post decay ratio (last/first post loop)", post_qps[-1] / post_qps[0], f">= {MIN_POST_DECAY_RATIO}"),
    ]
    print(
        f"radix evict soak: pre qps {pre_qps:.1f} p99 {pre_p99:.3f}s | "
        f"post qps {avg_post_qps:.1f} p99 {max_post_p99:.3f}s | "
        f"{sent + POST_LOOPS * len(base_samples)} unique requests, "
        f"pool {RADIX_POOL_TOKENS} tokens"
    )
    for name, value, bound in gate_rows:
        print(f"  gate {name}: {value:.3f} ({bound})")
    # metrics artifacts at a fixed path (latest attempt wins across retries)
    import json
    from pathlib import Path

    base = Path("/tmp/radix_evict_soak")
    base.mkdir(parents=True, exist_ok=True)
    md = ["| gate | value | bound | verdict |", "|---|---|---|---|"]
    for name, value, bound in gate_rows:
        op, thr = bound.split()
        ok = value >= float(thr) if op == ">=" else value <= float(thr)
        md.append(f"| {name} | {value:.3f} | {bound} | {'pass' if ok else '**FAIL**'} |")
    md.append("")
    md.append(
        f"pre {pre_qps:.1f} qps / p99 {pre_p99:.3f}s; post qps "
        + ", ".join(f"{q:.1f}" for q in post_qps)
        + f"; {sent + POST_LOOPS * len(base_samples)} unique requests, pool {RADIX_POOL_TOKENS} tokens"
    )
    (base / "results.md").write_text("\n".join(md))
    (base / "results.json").write_text(
        json.dumps(
            {
                "pool_tokens": RADIX_POOL_TOKENS,
                "unique_requests": sent + POST_LOOPS * len(base_samples),
                "pre": {"qps": pre_qps, "p99_s": pre_p99},
                "post_qps": post_qps,
                "post_p99": post_p99,
                "gates": {n: v for n, v, _ in gate_rows},
            }
        )
    )

    checks.check(
        len(post_qps) == POST_LOOPS and all(q > 0 for q in post_qps),
        f"post window incomplete: {post_qps} (refusing an empty comparison)",
    )
    checks.check(
        not incomplete,
        f"loops with failed requests: {incomplete}",
    )
    checks.check(
        pre_qps >= MIN_SANE_PRE_QPS,
        f"pre-saturation qps {pre_qps:.1f} below sanity floor {MIN_SANE_PRE_QPS}",
    )
    checks.check(
        avg_post_qps >= MIN_POST_PRE_QPS_RATIO * pre_qps,
        f"post-saturation qps {avg_post_qps:.1f} < "
        f"{MIN_POST_PRE_QPS_RATIO} x pre {pre_qps:.1f}",
    )
    checks.check(
        max_post_p99 <= MAX_POST_PRE_P99_RATIO * pre_p99,
        f"post-saturation p99 {max_post_p99:.3f}s > "
        f"{MAX_POST_PRE_P99_RATIO} x pre {pre_p99:.3f}s",
    )
    checks.check(
        post_qps[-1] >= MIN_POST_DECAY_RATIO * post_qps[0],
        f"post-saturation decay: last loop {post_qps[-1]:.1f} < "
        f"{MIN_POST_DECAY_RATIO} x first loop {post_qps[0]:.1f}",
    )
    checks.assert_all()
