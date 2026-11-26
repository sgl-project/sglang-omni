# SPDX-License-Identifier: Apache-2.0
"""How samples fold into a record and how a record is judged against its baseline; pure functions with no device."""

from __future__ import annotations

import shlex
import statistics

from scripts.xpu.baseline.models import BaselineModel, Record, Sample, Tolerances

COMPARED_GROUPS = ("exact", "atmost", "close", "speed")
DEFAULT_WARMUPS = 2

BUILTIN_TOLERANCES: Tolerances = {
    "close": 0.03,
    "absolute": 0.05,
    "speed": 0.10,
    "speed_headroom": 3.0,
    # Wider than this a gate cannot see a doubling, however noisy the entry was when recorded.
    "speed_ceiling": 0.5,
}


def fold_samples(samples: list[Sample] | list[Record]) -> Record:
    """Collapse repeats: exact fields must agree, a ceiling takes the worst sample, the rest the median."""
    folded: Record = {group: {} for group in COMPARED_GROUPS}
    folded["spread"] = {}
    for group in COMPARED_GROUPS:
        for name in samples[0].get(group, {}):
            values = [sample[group][name] for sample in samples]
            if group == "exact":
                unique = set(values)
                if len(unique) != 1:
                    raise AssertionError(
                        f"{name} is not reproducible across repeats: {sorted(unique)}"
                    )
                else:
                    folded[group][name] = values[0]
                continue
            else:
                folded[group][name] = (
                    max(values) if group == "atmost" else statistics.median(values)
                )
            if len(values) > 1 and folded[group][name]:
                span = (max(values) - min(values)) / folded[group][name]
                folded["spread"][f"{group}.{name}"] = round(span, 4)
                trimmed = trimmed_span(values, folded[group][name])
                if group == "speed" and trimmed is not None:
                    folded.setdefault("trimmed_spread", {})[f"{group}.{name}"] = trimmed
                else:
                    pass
                drift = warming_drift(values) if group == "speed" else None
                if drift:
                    early, late = drift
                    print(
                        f"    still warming on {name}: the run's two halves sit apart "
                        f"({early:.6g} then {late:.6g}); raise --warmups before trusting the spread"
                    )
                else:
                    pass
            else:
                pass
    if samples[0].get("observed"):
        folded["observed"] = {}
        folded["observed_span"] = {}
        for name in samples[0]["observed"]:
            values = [sample["observed"][name] for sample in samples]
            middle = statistics.median(values)
            folded["observed"][name] = middle
            folded["observed_span"][name] = round(
                (max(values) - min(values)) / (middle or 1), 4
            )
    else:
        pass
    return folded


def trimmed_span(values: list[float], middle: float) -> float | None:
    """Spread with the best and worst sample dropped, which is what a gate is sized from; None below five samples."""
    if len(values) < 5:
        return None
    else:
        inner = sorted(values)[1:-1]
        return round((max(inner) - min(inner)) / middle, 4)


def warming_drift(values: list[float]) -> tuple[float, float] | None:
    """The two half-medians when they sit more than 3 percent apart, which means the run is still settling."""
    if len(values) < 4:
        return None
    else:
        middle = statistics.median(values)
        half = len(values) // 2
        early = statistics.median(values[:half])
        late = statistics.median(values[-half:])
    if middle and abs(late - early) / abs(middle) > 0.03:
        return (early, late)
    else:
        return None


def pessimistic_speed(runs: list[Record]) -> dict[str, float]:
    """The slowest value each speed field reached across launches: these models are modal, not noisy."""
    worst: dict[str, float] = {}
    for run in runs:
        for name, value in run.get("speed", {}).items():
            if name not in worst:
                worst[name] = value
            elif "latency" in name:
                worst[name] = max(worst[name], value)
            else:
                worst[name] = min(worst[name], value)
    return worst


def fold_launches(runs: list[Record]) -> Record:
    """One record from several server lifetimes: accuracy must agree, speed is the slowest launch."""
    if len(runs) == 1:
        return runs[0]
    else:
        folded = fold_samples(runs)
    folded["speed"] = pessimistic_speed(runs)
    for field_name in ("cards", "judge_cards"):
        layouts = sorted({run[field_name] for run in runs if run.get(field_name)})
        if layouts:
            folded[field_name] = " | ".join(layouts)
        else:
            pass
    # The worst trimmed within-run dispersion, which sizes the tolerance once the value is the slowest launch.
    worst_within: dict[str, float] = {}
    for run in runs:
        per_run = run.get("trimmed_spread") or run.get("spread", {})
        for name, value in per_run.items():
            worst_within[name] = max(worst_within.get(name, 0.0), value)
    folded["within_run_spread"] = worst_within
    folded["launches"] = len(runs)
    for carried in ("judge_serve_extra", "warmups"):
        if carried in runs[0]:
            folded[carried] = runs[0][carried]
        else:
            pass
    return folded


def speed_tolerance_for(baseline: Record, name: str, tolerances: Tolerances) -> float:
    """The larger of the floor and headroom times the recorded spread, never past the ceiling."""
    # A multi-launch entry already holds its slowest launch, so its launch-to-launch spread is not charged twice.
    if baseline.get("launches", 1) > 1:
        spread = baseline.get("within_run_spread", {}).get(f"speed.{name}")
    else:
        spread = baseline.get("trimmed_spread", {}).get(f"speed.{name}")
    if spread is None:
        spread = baseline.get("spread", {}).get(f"speed.{name}")
    else:
        pass
    ceiling = tolerances.get("speed_ceiling", BUILTIN_TOLERANCES["speed_ceiling"])
    if spread is None:
        return min(ceiling, tolerances["speed"])
    else:
        return min(
            ceiling, max(tolerances["speed"], spread * tolerances["speed_headroom"])
        )


def card_notes(key: str, baseline: Record, observed: Record) -> list[str]:
    recorded_cards = baseline.get("cards")
    observed_cards = observed.get("cards")
    if not recorded_cards:
        return [
            f"{key}: recorded without its card assignment, so a change of card cannot be "
            "detected; re-record this entry"
        ]
    elif observed_cards and recorded_cards != observed_cards:
        return [
            f"{key}: recorded on card(s) {recorded_cards}, measured on {observed_cards}; "
            "speed is not comparable across cards on a host with uneven PCIe links"
        ]
    else:
        return []


def compare(
    key: str, baseline: Record, observed: Record, tolerances: Tolerances
) -> tuple[list[str], list[str], list[str]]:
    """Return regressions, improvements and notes; only a regression fails the run."""
    failures: list[str] = []
    gains: list[str] = []
    notes = card_notes(key, baseline, observed)

    for name, want in baseline.get("exact", {}).items():
        got = observed.get("exact", {}).get(name)
        if got != want:
            failures.append(
                f"{key}: {name} changed\n    baseline {want!r}\n    observed {got!r}"
            )
        else:
            pass

    for name, want in baseline.get("atmost", {}).items():
        got = observed.get("atmost", {}).get(name)
        ceiling = want + tolerances["absolute"]
        if got is None:
            failures.append(f"{key}: {name} missing from this run")
        elif got > ceiling:
            failures.append(
                f"{key}: {name} rose to {got:.4f} (baseline {want:.4f}, ceiling {ceiling:.4f})"
            )
        elif got < want:
            gains.append(f"{key}: {name} fell to {got:.4f} from {want:.4f}")
        else:
            pass

    for name, want in baseline.get("close", {}).items():
        got = observed.get("close", {}).get(name)
        if got is None:
            failures.append(f"{key}: {name} missing from this run")
        elif want and abs(got - want) / want > tolerances["close"]:
            failures.append(
                f"{key}: {name} moved {(got - want) / want:+.2%} "
                f"(baseline {want:.6g}, observed {got:.6g}, allowed +-{tolerances['close']:.1%})"
            )
        else:
            pass

    for name, want in baseline.get("speed", {}).items():
        got = observed.get("speed", {}).get(name)
        if got is None:
            failures.append(f"{key}: {name} missing from this run")
            continue
        else:
            allowed = speed_tolerance_for(baseline, name, tolerances)
        # Latency must not grow; a rate must not shrink.
        regression = (got - want) / want if "latency" in name else (want - got) / want
        if regression > allowed:
            failures.append(
                f"{key}: {name} regressed {regression:.2%} "
                f"(baseline {want:.6g}, observed {got:.6g}, allowed {allowed:.1%})"
            )
        elif -regression > allowed:
            gains.append(
                f"{key}: {name} improved {-regression:.2%} "
                f"(baseline {want:.6g}, observed {got:.6g})"
            )
        else:
            pass
    return failures, gains, notes


def resolve_tolerances(
    flags: dict[str, float | None], recorded: Tolerances | dict[str, float]
) -> tuple[Tolerances, dict[str, str]]:
    """An explicit flag wins, then what the environment recorded, then the built-in; returns where each came from."""
    tolerances: Tolerances = dict(BUILTIN_TOLERANCES)
    sources: dict[str, str] = {}
    for name in BUILTIN_TOLERANCES:
        flagged = flags.get(name)
        if flagged is not None:
            tolerances[name], sources[name] = flagged, "the flag"
        elif name in recorded:
            tolerances[name], sources[name] = recorded[name], "this entry"
        else:
            sources[name] = "the built-in default"
    return tolerances, sources


def card_pool(spec: str, visible: list[int]) -> list[int]:
    """The physical cards a run may claim; raises ValueError for a card it cannot see or one named twice."""
    pool = [int(card) for card in spec.split(",") if card.strip()] if spec else visible
    outside = sorted(set(pool) - set(visible))
    if outside:
        raise ValueError(
            f"--card-pool {spec} names card(s) {outside} this process cannot use; it can use {visible}"
        )
    elif len(set(pool)) != len(pool):
        raise ValueError(f"--card-pool {spec} names a card more than once")
    else:
        return pool


def with_recorded_judge_flags(
    model: BaselineModel,
    recorded: Record,
    extras: dict[str, tuple[str, ...]],
    explicit: set[str],
) -> dict[str, tuple[str, ...]]:
    """A judge runs with the flags its dependent was recorded with, unless this run names its own."""
    if (
        not model.judge
        or model.judge in explicit
        or "judge_serve_extra" not in recorded
    ):
        return extras
    else:
        flags = tuple(recorded["judge_serve_extra"])
    print(
        f"  judge {model.judge} flags, from this entry: {shlex.join(flags) or 'none'}"
    )
    return {**extras, model.judge: flags}


def warmups_for(recorded: Record, explicit: int | None) -> int:
    """A check discards as many warmups as its entry did, unless this run names its own."""
    if explicit is not None:
        return explicit
    else:
        return recorded.get("warmups", DEFAULT_WARMUPS)


def merged_serve_extra(
    previous: dict[str, list[str]],
    extras: dict[str, tuple[str, ...]],
    measured: set[str],
) -> dict[str, list[str]]:
    """Provenance belongs to the model it was measured with, so a re-record can clear a workaround."""
    kept = {key: value for key, value in previous.items() if key not in measured}
    kept.update({key: list(value) for key, value in extras.items() if key in measured})
    return kept
