#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record and re-check per-model accuracy and speed for the models Intel XPU serves.

Run from the repository root: python -m scripts.xpu.xpu_model_baseline [--list | --record | --calibrate] [--models a,b].
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

from scripts.xpu.baseline.arguments import parse_arguments
from scripts.xpu.baseline.cards import CardsBusy, visible_cards
from scripts.xpu.baseline.environment import (
    card_family,
    environment_fingerprint,
    fingerprint_key,
    recorded_from_commit,
)
from scripts.xpu.baseline.gate import (
    card_pool,
    compare,
    resolve_tolerances,
    warmups_for,
    with_recorded_judge_flags,
)
from scripts.xpu.baseline.hub import (
    download_blocked,
    prefetch_checkpoints,
    prune_hub_cache,
)
from scripts.xpu.baseline.models import (
    MODEL_BY_KEY,
    MODELS,
    BaselineModel,
    Environment,
    Record,
    Tolerances,
)
from scripts.xpu.baseline.serving import RunSettings, measure_launches
from scripts.xpu.baseline.store import load_baseline, write_entry


@dataclass
class Tally:
    results: dict[str, Record] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    gains: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def line(self) -> str:
        return (
            f"  running tally: {len(self.results)} ok, {len(self.failures)} failed, "
            f"{len(self.skipped)} skipped"
        )


def list_models(path: Path) -> int:
    recorded_on: dict[str, list[str]] = {}
    for environment in load_baseline(path)["environments"].values():
        for key in environment.get("models", {}):
            recorded_on.setdefault(key, []).append(environment["fingerprint"]["host"])
    print(f"{'key':22s} {'cards':>5s} {'judge':10s} {'recorded on':22s} model")
    for model in MODELS:
        where = (
            "skip"
            if model.unservable
            else ",".join(sorted(recorded_on.get(model.key, []))) or "-"
        )
        print(
            f"{model.key:22s} {model.cards:5d} {model.judge or '-':10s} {where:22s} {model.model_path}"
        )
    return 0


def parse_serve_extra(
    items: list[str], parser: argparse.ArgumentParser
) -> dict[str, tuple[str, ...]]:
    extras: dict[str, tuple[str, ...]] = {}
    for item in items:
        key, _, argument_text = item.partition("=")
        if key not in MODEL_BY_KEY or not argument_text:
            parser.error(f"--serve-extra wants <model key>=<args>, got {item!r}")
        else:
            extras[key] = tuple(shlex.split(argument_text))
    return extras


def skip_reason(model: BaselineModel, done: set[str], pool_size: int) -> str:
    """Why this model is not measured here, before anything touches the cache; empty when it is."""
    judge_cards = MODEL_BY_KEY[model.judge].cards if model.judge else 0
    if model.unservable:
        return model.unservable
    elif model.key in done:
        return "already recorded for this environment; --fresh to redo"
    elif model.cards + judge_cards > pool_size:
        return (
            f"needs {model.cards} card(s)"
            + (f" and {judge_cards} for its judge" if judge_cards else "")
            + f", pool has {pool_size}"
        )
    else:
        return ""


def judged(
    key: str, entry: Environment, observed: Record, tolerances: Tolerances, tally: Tally
) -> None:
    if key not in entry.get("models", {}):
        tally.failures.append(
            f"{key}: absent from this environment's baseline; re-record to add it"
        )
        return
    else:
        failures, gains, notes = compare(
            key, entry["models"][key], observed, tolerances
        )
    tally.failures += failures
    tally.gains += gains
    tally.notes += notes
    for message in gains:
        print(f"  IMPROVED {message.split(': ', 1)[1]}", flush=True)
    for message in notes:
        print(f"  NOTE {message.split(': ', 1)[1]}", flush=True)


def report(args: argparse.Namespace, tally: Tally) -> int:
    if args.calibrate:
        print("\nobserved spread per field, as a fraction of the median:")
        for key, observed in tally.results.items():
            for name, span in sorted(observed["spread"].items()):
                print(f"  {key:22s} {name:46s} {span:.2%}")
        print(
            "\nSet --speed-tolerance above the largest speed spread, or the gate will be flaky."
        )
        for message in tally.failures:
            print(f"FAILED      {message}")
        return 1 if tally.failures else 0
    else:
        print()
    # While recording there is nothing to regress against: a failure is a model that would not serve or answer.
    label = "FAILED     " if args.record else "REGRESSION "
    for message in tally.failures:
        print(f"{label} {message}")
    for message in tally.gains:
        print(f"IMPROVED    {message}")
    if tally.gains:
        print(
            "  these beat the baseline: re-record so the floor rises with the code, "
            "or the gate stops catching anything."
        )
    else:
        pass
    for message in tally.notes:
        print(f"NOTE        {message}")
    for message in tally.skipped:
        print(f"SKIPPED     {message}")
    print(
        f"{len(tally.results)} measured, {len(tally.skipped)} skipped, {len(tally.failures)} "
        f"{'failure(s)' if args.record else 'regression(s)'}"
    )
    return 1 if tally.failures else 0


def main() -> int:
    parser, args = parse_arguments()
    if args.list:
        return list_models(args.baseline)
    else:
        keys = (
            [model.key for model in MODELS]
            if args.models == "all"
            else args.models.split(",")
        )
    unknown = [key for key in keys if key not in MODEL_BY_KEY]
    if unknown:
        parser.error(f"unknown model keys: {unknown}; see --list")
    else:
        extras = parse_serve_extra(args.serve_extra, parser)

    mask = os.environ.get("ZE_AFFINITY_MASK", "").strip()
    try:
        visible = visible_cards()
    except ValueError:
        parser.error(f"ZE_AFFINITY_MASK={mask!r} names sub-devices; give whole cards")
    try:
        pool = card_pool(args.card_pool, visible)
    except ValueError as exc:
        parser.error(f"{exc}" + (f" (ZE_AFFINITY_MASK={mask})" if mask else ""))
    fingerprint = environment_fingerprint(args.host_label)
    try:
        family = card_family(fingerprint)
    except ValueError as exc:
        parser.error(str(exc))
    env_key = fingerprint_key(fingerprint)
    print(f"card pool {pool}, {family} recipe", flush=True)
    print(f"environment {env_key}: {json.dumps(fingerprint)}")

    stored = load_baseline(args.baseline)
    entry: Environment = stored["environments"].get(env_key, {})
    if not args.record and not args.calibrate and not entry:
        known = [env.get("fingerprint", {}) for env in stored["environments"].values()]
        print(
            f"no baseline for environment {env_key} in {args.baseline}.\n"
            f"  known environments: {known or 'none'}\n"
            "  run with --record to add this one.",
            file=sys.stderr,
        )
        return 2
    else:
        pass

    # A check measures the recipe its entry was recorded with; a flag given here still wins.
    explicit = set(extras)
    if not args.record:
        restored = {
            key: tuple(value)
            for key, value in entry.get("serve_extra", {}).items()
            if key not in extras
        }
        for key, value in sorted(restored.items()):
            print(f"serve flags for {key}, from this entry: {shlex.join(value)}")
        extras = {**restored, **extras}
    else:
        pass

    tolerances, sources = resolve_tolerances(
        {
            "close": args.close_tolerance,
            "absolute": args.absolute_tolerance,
            "speed": args.speed_tolerance,
            "speed_headroom": args.speed_headroom,
            "speed_ceiling": args.speed_ceiling,
        },
        entry.get("tolerances", {}),
    )
    print(f"speed floor {tolerances['speed']:.1%}, from {sources['speed']}")
    base_settings = RunSettings(
        pool=pool,
        family=family,
        repeats=args.repeats,
        warmups=0,
        launches=args.launches,
        idle_mib=args.idle_mib,
        idle_wait_s=args.idle_wait_s,
    )

    tally = Tally()
    done = set(entry.get("models", {})) if args.record and not args.fresh else set()
    for index, key in enumerate(keys, start=1):
        model = MODEL_BY_KEY[key]
        print(f"\n== [{index}/{len(keys)}] {model.name} ({key}) ==", flush=True)
        reason = skip_reason(model, done, len(pool))
        if not reason and args.prune_hf_cache:
            print(f"  {prune_hub_cache(model)}", flush=True)
            reason = download_blocked(
                model.model_path, args.max_checkpoint_gib, args.prefetch
            )
        elif not reason:
            reason = download_blocked(
                model.model_path, args.max_checkpoint_gib, args.prefetch
            )
        else:
            pass
        if reason:
            print(f"  SKIPPED: {reason}", flush=True)
            tally.skipped.append(f"{key}: {reason}")
            continue
        elif args.prefetch:
            print(f"  {prefetch_checkpoints(model)}", flush=True)
        else:
            pass
        recorded: Record = {} if args.record else entry.get("models", {}).get(key, {})
        settings = replace(base_settings, warmups=warmups_for(recorded, args.warmups))
        if args.warmups is None and "warmups" in recorded:
            print(f"  warmups {settings.warmups}, from this entry", flush=True)
        else:
            pass
        model_extras = (
            extras
            if args.record
            else with_recorded_judge_flags(model, recorded, extras, explicit)
        )
        try:
            observed = measure_launches(model, settings, model_extras)
        except CardsBusy as exc:
            print(f"  SKIPPED: {exc}", flush=True)
            tally.skipped.append(f"{key}: {exc}")
            continue
        except Exception as exc:
            tally.failures.append(f"{key}: did not produce a measurement: {exc}")
            print(f"  FAILED: {exc}", flush=True)
            print(tally.line(), flush=True)
            continue
        tally.results[key] = observed
        print(f"  median  {json.dumps(observed['speed'])}")
        print(
            f"  accuracy {json.dumps(observed['exact'])} {json.dumps(observed.get('atmost', {}))}"
        )
        print(f"  spread  {json.dumps(observed['spread'])}")
        if args.record:
            write_entry(
                args.baseline,
                env_key,
                {
                    "fingerprint": fingerprint,
                    "recorded_from_commit": recorded_from_commit(),
                    "tolerances": tolerances,
                },
                extras,
                tally.results,
            )
        elif args.calibrate:
            pass
        else:
            judged(key, entry, observed, tolerances, tally)
        print(tally.line(), flush=True)

    if args.record and tally.results:
        print(
            f"\nwrote {len(tally.results)} model(s) for environment {env_key} to {args.baseline}"
        )
    elif args.record:
        print("\nnothing measured; leaving the baseline untouched")
    else:
        pass
    return report(args, tally)


if __name__ == "__main__":
    sys.exit(main())
