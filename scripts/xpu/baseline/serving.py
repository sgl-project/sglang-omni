# SPDX-License-Identifier: Apache-2.0
"""Serve a model, and its judge when it has one, and sample it across one or more server lifetimes."""

from __future__ import annotations

import importlib
import os
import socket
import subprocess
import sys
from dataclasses import dataclass

from benchmarks.benchmarker.utils import start_server_from_cmd, stop_server
from scripts.xpu.baseline.cards import (
    allocate_cards,
    idle_after_run,
    idle_before_launch,
)
from scripts.xpu.baseline.flags import merged_flags
from scripts.xpu.baseline.gate import fold_launches, fold_samples
from scripts.xpu.baseline.models import MODEL_BY_KEY, BaselineModel, Record
from scripts.xpu.baseline.probes import take_sample
from sglang_omni.config.manager import resolve_config_cls_for_model_path

# SGLang's own enable_deterministic_inference is absent: with no backend named it picks fa3, which is CUDA-only.
DETERMINISM_ENGINE_ARGS = {
    "disable_radix_cache": "true",
    "random_seed": "0",
}

ENGINE_STAGE_CACHE: dict[str, tuple[str, ...]] = {}


@dataclass(frozen=True, kw_only=True)
class RunSettings:
    """How one model is measured on this host."""

    pool: list[int]
    family: str
    repeats: int
    warmups: int
    launches: int
    idle_mib: float
    idle_wait_s: int


def engine_stages(model_path: str, variant: str = "") -> tuple[str, ...]:
    """Stage names that own an SGLang engine, read from the model's pipeline config so an override names a real stage."""
    cache_key = f"{model_path}@{variant}"
    if cache_key in ENGINE_STAGE_CACHE:
        return ENGINE_STAGE_CACHE[cache_key]
    else:
        pass
    try:
        config_cls = resolve_config_cls_for_model_path(model_path)
        if variant:
            variants = getattr(
                importlib.import_module(config_cls.__module__), "Variants", {}
            )
            config_cls = variants.get(variant, config_cls)
        else:
            pass
        config = config_cls(model_path=model_path)
        stages = tuple(
            stage.name for stage in config.stages if getattr(stage, "engine", None)
        )
    except Exception as exc:
        print(f"  could not derive engine stages ({exc}); serving without them")
        stages = ()
    ENGINE_STAGE_CACHE[cache_key] = stages
    return stages


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def serve_command(
    model: BaselineModel, port: int, family: str, extra: tuple[str, ...] = ()
) -> list[str]:
    """The row's recipe for this card family, then the determinism args, then this host's flags; the later wins."""
    determinism = [
        token
        for stage in engine_stages(model.model_path, model.config_variant)
        for option, value in DETERMINISM_ENGINE_ARGS.items()
        for token in (f"--{stage}.engine.{option}", value)
    ]
    return [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model-path",
        model.model_path,
        *merged_flags(model.serve_args_on(family), determinism, extra),
    ]


def launch(
    model: BaselineModel, cards: str, family: str, extra: tuple[str, ...] = ()
) -> tuple[subprocess.Popen[bytes], str]:
    port = free_port()
    env = dict(os.environ, ZE_AFFINITY_MASK=cards, **model.env)
    env.setdefault("SGLANG_OMNI_STARTUP_TIMEOUT", str(model.startup_seconds))
    # The server strips proxy variables, so a Hub lookup there waits out a connect timeout instead of reading the cache.
    env.setdefault("HF_HUB_OFFLINE", "1")
    command = serve_command(model, port, family, extra)
    print(f"  serving {model.key} on card(s) {cards}, port {port}", flush=True)
    proc = start_server_from_cmd(
        command,
        None,
        port,
        timeout=model.startup_seconds,
        env=env,
        strip_proxy=True,
        health_path="/v1/models",
        health_body_contains=None,
    )
    return proc, f"http://127.0.0.1:{port}"


def measure(
    model: BaselineModel, settings: RunSettings, extras: dict[str, tuple[str, ...]]
) -> Record:
    """Serve the model, and its judge when it has one, then sample it warmups+repeats times."""
    servers = []
    taken: set[int] = set()
    cards = allocate_cards(settings.pool, model.cards, taken)
    judge = MODEL_BY_KEY[model.judge] if model.judge else None
    judge_cards = allocate_cards(settings.pool, judge.cards, taken) if judge else ""
    layout = ",".join(part for part in (cards, judge_cards) if part)
    busy_before = idle_before_launch(layout, settings.idle_mib, settings.idle_wait_s)
    judge_flags = extras.get(judge.key, ()) if judge else ()
    try:
        judge_url = ""
        if judge:
            judge_proc, judge_url = launch(
                judge, judge_cards, settings.family, judge_flags
            )
            servers.append(judge_proc)
        else:
            pass
        proc, base_url = launch(
            model, cards, settings.family, extras.get(model.key, ())
        )
        servers.append(proc)

        samples = []
        for index in range(settings.warmups + settings.repeats):
            sample = take_sample(model, base_url, judge_url)
            label = (
                f"warmup {index + 1}"
                if index < settings.warmups
                else f"rep {index - settings.warmups + 1}"
            )
            extra = (
                f" wer={sample['atmost']['round_trip_wer']:.4f}"
                if sample.get("atmost", {}).get("round_trip_wer") is not None
                else ""
            )
            print(f"    {label}: {sample['speed']}{extra}", flush=True)
            samples.append(sample)
    finally:
        for proc in reversed(servers):
            stop_server(proc)

    busy_after = idle_after_run(layout, settings.idle_mib)
    folded = fold_samples(samples[settings.warmups :])
    folded["cards"] = cards
    if judge:
        folded["judge_cards"] = judge_cards
        folded["judge_serve_extra"] = list(judge_flags)
    else:
        pass
    folded["warmups"] = settings.warmups
    folded["cards_used_mib"] = {"before": busy_before, "after": busy_after}
    return folded


def measure_launches(
    model: BaselineModel, settings: RunSettings, extras: dict[str, tuple[str, ...]]
) -> Record:
    """Measure across whole server restarts, the variance a gate has to survive and repeats cannot see."""
    runs = []
    for index in range(settings.launches):
        if settings.launches > 1:
            print(f"  launch {index + 1}/{settings.launches}", flush=True)
        else:
            pass
        runs.append(measure(model, settings, extras))
    return fold_launches(runs)
