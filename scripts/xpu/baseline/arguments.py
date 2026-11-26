# SPDX-License-Identifier: Apache-2.0
"""The baseline's command line."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.xpu.baseline.models import DEFAULT_BASELINE


def parse_arguments() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description="Record and re-check per-model accuracy and speed for the models Intel XPU serves."
    )
    add = parser.add_argument
    add("--models", default="all", help="comma-separated keys, or 'all'")
    add("--baseline", type=Path, default=DEFAULT_BASELINE)
    add(
        "--record",
        action="store_true",
        help="write this environment's entry instead of checking",
    )
    add(
        "--calibrate",
        action="store_true",
        help="report observed spread and exit without checking",
    )
    add(
        "--repeats",
        type=int,
        default=5,
        help="scored samples per model, taken after the warmups",
    )
    add(
        "--launches",
        type=int,
        default=1,
        help="whole server restarts per model; above 1, speed is anchored at the slowest launch and exact fields must agree across restarts",
    )
    add(
        "--warmups",
        type=int,
        default=None,
        help="samples taken and discarded before scoring begins (default 2 when recording; a check repeats its entry's)",
    )
    add(
        "--fresh",
        action="store_true",
        help="re-measure models already recorded for this environment instead of resuming",
    )
    add(
        "--card-pool",
        default="",
        help="cards this run may claim, by physical id; each model takes what its topology needs (default: every card this process can see, within any inherited ZE_AFFINITY_MASK)",
    )
    add(
        "--close-tolerance",
        type=float,
        default=None,
        help="relative band for loudness, which moves when kernels change even under a seed",
    )
    add(
        "--absolute-tolerance",
        type=float,
        default=None,
        help="absolute headroom for a rate such as WER",
    )
    add(
        "--speed-tolerance",
        type=float,
        default=None,
        help="lower bound on the speed regression allowed, before per-field spread is considered",
    )
    add(
        "--speed-headroom",
        type=float,
        default=None,
        help="multiple of a field's recorded spread to allow; the larger of this and the floor wins",
    )
    add(
        "--speed-ceiling",
        type=float,
        default=None,
        help="the most speed regression any field may be allowed, however wide its recorded spread",
    )
    add(
        "--serve-extra",
        action="append",
        default=[],
        metavar="KEY=ARGS",
        help="extra serve args for one model on this host, recorded with its entry, e.g. qwen3-tts=--vocoder.factory.async_decode false",
    )
    add(
        "--idle-mib",
        type=float,
        default=1024.0,
        help="refuse to measure a card already holding more than this; a neighbour's work reads as a regression otherwise",
    )
    add(
        "--idle-wait-s",
        type=int,
        default=0,
        help="seconds to wait for a busy card before skipping the model",
    )
    add(
        "--prune-hf-cache",
        action="store_true",
        help="delete cached checkpoints this measurement does not need; destructive, and on some hosts the cache holds other users' work",
    )
    add(
        "--prefetch",
        action="store_true",
        help="fetch weights before launching, for hosts whose server has no network",
    )
    add(
        "--host-label",
        default="",
        help="identify the machine as this instead of by hostname, for runs inside a container whose hostname names the container and not the machine",
    )
    add(
        "--max-checkpoint-gib",
        type=float,
        default=100.0,
        help="skip a model whose checkpoint is larger than this, cached or not; it keeps its row and the run still exits 0",
    )
    add("--list", action="store_true")
    return parser, parser.parse_args()
