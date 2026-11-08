# SPDX-License-Identifier: Apache-2.0
"""Sweep a speculative draft against no draft across concurrency.

One command for every draft: name its server config and a workload. The no-draft config is the same config with disable_speculative_decoding on every stage that runs a draft, so every other engine setting reaches both sides. The serving loader decides which stages run one, so a draft set through a shared selector or a model default is turned off too. Both turn off the prefix cache on the drafted stages unless --prefix-cache is set, because the clients send the same prompts in every pass and cache hits would stand in for prefill. Each leg starts a fresh server, runs the workload's client over every concurrency, and stops the server. Legs run one after another in the order given, ABBA by default, so drift between runs shows as a gap between the two legs of one side.

The summary gives, per concurrency, both sides' throughput, the gain, p95 latency and its change, first-output p95 when the client streams, the workload's accuracy metrics and failed requests; then k, the loads where the draft is worse than no draft, and per leg the startup time, KV pool, peak token usage, retractions, acceptance length and peak GPU memory. sweep.json holds the same comparison, and baseline.yaml and draft.yaml the configs each side served.

Set CUDA_VISIBLE_DEVICES to one GPU that nothing else uses: after each leg the managed server ends every process still using it. Arguments after -- go to the client, such as --stream to measure first output.

TTS workloads save generated audio at every load in every leg before starting one ASR server to score all saved audio. The ASR server uses the port after the last generation leg. Transcription is outside the TTS performance measurement and uses --asr-model-path and --asr-concurrency.

Usage:

    CUDA_VISIBLE_DEVICES=0 python -m benchmarks.eval.speculative_sweep \\
        --draft-config examples/configs/qwen3_asr_1_7b_dflash.yaml \\
        --workload seedtts-en \\
        --output-dir results/sweep/qwen3_asr_seedtts_en
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from benchmarks.benchmarker.utils import managed_omni_server
from benchmarks.eval.speculative_tts import generate_tts, measure_tts
from benchmarks.eval.speculative_workloads import (
    HOST,
    WORKLOADS,
    AsrWorkload,
    LoadSettings,
    TtsWorkload,
    measure_asr,
    measure_transcribe_diarize,
)
from benchmarks.metrics.speculative import (
    K_GAIN,
    Leg,
    Side,
    compare,
    read_server_log,
    render,
)
from benchmarks.tasks.asr import (
    DEFAULT_ASR_TRANSCRIBE_CONCURRENCY,
    QWEN3_ASR_MODEL_PATH,
)
from sglang_omni.cli.serve import apply_default_draft_engine_overrides
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.sources import patches_from_model_path_flag

logger = logging.getLogger(__name__)

SIDE_BY_LETTER: dict[str, Side] = {"A": "baseline", "B": "draft"}
SPECULATIVE_SETTING_PREFIX = "speculative_"
DEFAULT_CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64)
DRIVER_CLIENT_FLAGS = (
    "--host",
    "--port",
    "--model-path",
    "--output",
    "--output-dir",
    "--concurrencies",
    "--concurrency",
    # note (wilsonzheng0327): the transcribe-diarize client's alias of --concurrency.
    "--max-concurrency",
    "--repeats",
    "--warmup",
    "--max-samples",
    "--use-existing-server",
    "--dataset",
    "--lang",
    # note (wilsonzheng0327): these send a leg's requests to another server, replay saved outputs instead of sending them, or move the result files the sweep reads.
    "--base-url",
    "--reuse-asr-results",
    "--asr-results-file",
    "--speed-results-file",
)
TTS_DRIVER_CLIENT_FLAGS = (
    "--model",
    "--meta",
    "--testset",
    "--generate-only",
    "--transcribe-only",
    "--similarity-only",
    "--utmos-only",
    "--sustained-overshoot",
    "--request-rate",
    "--asr-model-path",
    "--asr-concurrency",
)
TTS_SERVER_CLIENT_FLAGS = (
    "--quantization",
    "--max-running-requests",
    "--max-queued-requests",
    "--cuda-graph-max-bs",
    "--server-config",
    "--server-timeout",
    "--skip-gpu-cleanup",
)


class StageConfigFile(BaseModel):
    model_config = ConfigDict(extra="allow")

    engine: dict[str, JsonValue] = Field(default_factory=dict)


class ServerConfigFile(BaseModel):
    """The part of a server config YAML the sweep reads; every other key passes through."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    model_path: str | None = None
    stages: dict[str, StageConfigFile] = Field(default_factory=dict)


def write_side_configs(
    draft_config_path: Path,
    output_dir: Path,
    *,
    model_path: str,
    keep_prefix_cache: bool,
) -> dict[Side, Path]:
    """Write the draft config, and the same config with the draft switched off on every stage that runs one.

    Each leg launches with --model-path, so that path, not the file's, decides the draft the model gets by default.
    """
    manager = ConfigManager.from_file(str(draft_config_path))
    resolved_config = apply_default_draft_engine_overrides(
        manager.merge_config(
            {}, extra_patches=patches_from_model_path_flag(model_path, manager.config)
        )
    )
    drafted_stages = [
        stage.name
        for stage in resolved_config.stages
        if stage.engine is not None
        and any(
            setting.startswith(SPECULATIVE_SETTING_PREFIX)
            for setting in stage.engine.overrides()
        )
    ]
    if not drafted_stages:
        raise ValueError(f"{draft_config_path} runs no stage with a draft")
    else:
        pass
    draft_config = ServerConfigFile.model_validate(
        yaml.safe_load(draft_config_path.read_text())
    )
    prefix_cache_settings: dict[str, JsonValue] = (
        {} if keep_prefix_cache else {"disable_radix_cache": True}
    )
    engine_settings: dict[Side, dict[str, JsonValue]] = {
        "baseline": {"disable_speculative_decoding": True, **prefix_cache_settings},
        "draft": prefix_cache_settings,
    }
    config_paths: dict[Side, Path] = {}
    for side, settings in engine_settings.items():
        side_config = draft_config.model_copy(deep=True)
        if settings:
            stages = dict(side_config.stages)
            for name in drafted_stages:
                stage = stages.setdefault(name, StageConfigFile())
                stage.engine = {**stage.engine, **settings}
            side_config.stages = stages
        else:
            pass
        config_paths[side] = output_dir / f"{side}.yaml"
        config_paths[side].write_text(
            yaml.safe_dump(side_config.model_dump(exclude_unset=True), sort_keys=False)
        )
    return config_paths


def run_leg(
    *,
    name: str,
    side: Side,
    port: int,
    model_path: str,
    server_config: Path,
    args: argparse.Namespace,
    settings: LoadSettings,
) -> Leg:
    workload = WORKLOADS[args.workload]
    leg_dir = args.output_dir / name
    leg_dir.mkdir()
    server_log = leg_dir / "server.log"
    started = time.monotonic()
    with managed_omni_server(
        model_path=model_path,
        port=port,
        host=HOST,
        log_file=server_log,
        server_config=str(server_config),
    ):
        startup_s = time.monotonic() - started
        logger.info(f"{name}: server ready in {startup_s:.1f} s")
        if isinstance(workload, AsrWorkload):
            loads = measure_asr(
                workload, settings, port=port, model_path=model_path, leg_dir=leg_dir
            )
        elif isinstance(workload, TtsWorkload):
            generate_tts(
                workload, settings, port=port, model_path=model_path, leg_dir=leg_dir
            )
            loads = []
        else:
            loads = measure_transcribe_diarize(
                workload, settings, port=port, model_path=model_path, leg_dir=leg_dir
            )
    return Leg(
        name=name,
        side=side,
        startup_s=startup_s,
        server=read_server_log(server_log.read_text(errors="replace")),
        loads=loads,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--draft-config",
        type=Path,
        required=True,
        help="Server config that adds the draft; the no-draft config is derived from it.",
    )
    parser.add_argument("--workload", choices=list(WORKLOADS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--asr-model-path",
        help="ASR model used to score saved TTS audio after all generation legs finish.",
    )
    parser.add_argument(
        "--asr-concurrency",
        type=int,
        help="Transcription concurrency for TTS quality evaluation.",
    )
    parser.add_argument(
        "--model-path", default=None, help="The config's model_path when omitted."
    )
    parser.add_argument(
        "--concurrencies",
        type=lambda value: [int(concurrency) for concurrency in value.split(",")],
        default=list(DEFAULT_CONCURRENCIES),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--no-warmup",
        dest="warmup",
        action="store_false",
        help="Skip the discarded warmup before each concurrency.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Run a subset, for a smoke test."
    )
    parser.add_argument(
        "--prefix-cache",
        action="store_true",
        help="Keep the prefix cache on, to measure repeated prompts.",
    )
    parser.add_argument(
        "--order", default="ABBA", help="A runs without the draft, B with it."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8700,
        help="The first leg's port; each later leg takes the next. TTS scoring uses the port after the last leg.",
    )
    if "--" in argv:
        separator = argv.index("--")
        args = parser.parse_args(argv[:separator])
        args.client_args = argv[separator + 1 :]
    else:
        args = parser.parse_args(argv)
        args.client_args = []
    visible_gpus = [
        gpu
        for gpu in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if gpu.strip()
    ]
    is_tts = isinstance(WORKLOADS[args.workload], TtsWorkload)
    if len(visible_gpus) != 1:
        parser.error("set CUDA_VISIBLE_DEVICES to the one GPU the sweep owns")
    elif set(args.order) != set(SIDE_BY_LETTER):
        parser.error("--order needs at least one A and one B, and no other letters")
    elif any(value <= 0 for value in args.concurrencies):
        parser.error("--concurrencies must contain positive integers")
    elif len(set(args.concurrencies)) != len(args.concurrencies):
        parser.error("--concurrencies must not contain duplicates")
    elif args.repeats < 1:
        parser.error("--repeats must be positive")
    elif not is_tts and (
        args.asr_model_path is not None or args.asr_concurrency is not None
    ):
        parser.error(
            "--asr-model-path and --asr-concurrency apply only to TTS workloads"
        )
    elif args.asr_concurrency is not None and args.asr_concurrency < 1:
        parser.error("--asr-concurrency must be positive")
    elif args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    elif is_tts and any(
        flag.startswith(argument.split("=")[0])
        for argument in args.client_args
        if argument.startswith("--")
        for flag in TTS_SERVER_CLIENT_FLAGS
    ):
        parser.error(
            "TTS server options cannot be passed after --. Set engine options in the "
            "--draft-config YAML under stages.<stage>.engine; select the YAML with "
            "--draft-config. Server timeout and GPU cleanup are controlled by the sweep."
        )
    elif is_tts and any(
        flag.startswith(argument.split("=")[0])
        for argument in args.client_args
        if argument.startswith("--")
        for flag in TTS_DRIVER_CLIENT_FLAGS
    ):
        parser.error(
            "TTS client arguments must not override sweep routing, workload, load or phase settings"
        )
    elif any(
        # note (wilsonzheng0327): argparse expands an unambiguous prefix, so --max-conc would reach the client as --max-concurrency.
        flag.startswith(argument.split("=")[0])
        for argument in args.client_args
        if argument.startswith("--")
        for flag in DRIVER_CLIENT_FLAGS
    ):
        parser.error(
            f"the client may not get {', '.join(DRIVER_CLIENT_FLAGS)} or an abbreviation of one: the sweep sets them itself, or they would change what it measures"
        )
    else:
        pass
    if is_tts:
        args.asr_model_path = (
            QWEN3_ASR_MODEL_PATH if args.asr_model_path is None else args.asr_model_path
        )
        args.asr_concurrency = (
            DEFAULT_ASR_TRANSCRIBE_CONCURRENCY
            if args.asr_concurrency is None
            else args.asr_concurrency
        )
    else:
        pass
    return args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    args = parse_args(sys.argv[1:])
    draft_config = ServerConfigFile.model_validate(
        yaml.safe_load(args.draft_config.read_text())
    )
    model_path = (
        args.model_path if args.model_path is not None else draft_config.model_path
    )
    if model_path is None:
        raise ValueError(f"{args.draft_config} names no model_path; pass --model-path")
    else:
        pass
    args.output_dir.mkdir(parents=True)
    config_paths = write_side_configs(
        args.draft_config,
        args.output_dir,
        model_path=model_path,
        keep_prefix_cache=args.prefix_cache,
    )
    settings = LoadSettings(
        concurrencies=args.concurrencies,
        repeats=args.repeats,
        warmup=args.warmup,
        max_samples=args.max_samples,
        client_arguments=args.client_args,
    )
    legs = []
    for index, letter in enumerate(args.order):
        side = SIDE_BY_LETTER[letter]
        legs.append(
            run_leg(
                name=f"{index + 1}_{side}",
                side=side,
                port=args.port + index,
                model_path=model_path,
                server_config=config_paths[side],
                args=args,
                settings=settings,
            )
        )
    workload = WORKLOADS[args.workload]
    if isinstance(workload, TtsWorkload):
        asr_port = args.port + len(args.order)
        with managed_omni_server(
            model_path=args.asr_model_path,
            port=asr_port,
            host=HOST,
            log_file=args.output_dir / "asr_server.log",
        ):
            for leg in legs:
                logger.info(f"{leg.name}: scoring saved TTS audio")
                leg.loads = measure_tts(
                    workload,
                    settings,
                    port=asr_port,
                    model_path=model_path,
                    asr_model_path=args.asr_model_path,
                    asr_concurrency=args.asr_concurrency,
                    leg_dir=args.output_dir / leg.name,
                )
    else:
        pass
    comparison = compare(legs, k_gain=K_GAIN)
    summary = render(
        comparison, legs, title=f"{args.draft_config.stem} on {args.workload}"
    )
    (args.output_dir / "summary.md").write_text(summary)
    sweep = {
        "arguments": vars(args),
        "model_path": model_path,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "legs": [
            {"name": leg.name, "startup_s": leg.startup_s, **asdict(leg.server)}
            for leg in legs
        ],
        "loads": {leg.name: [asdict(load) for load in leg.loads] for leg in legs},
        "comparison": asdict(comparison),
    }
    (args.output_dir / "sweep.json").write_text(
        json.dumps(sweep, indent=2, default=str)
    )
    print(summary)


if __name__ == "__main__":
    main()
