"""Repeated candidate measurements and an evidence-scoped recommendation."""

import argparse
import asyncio
import hashlib
import json
import math
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path

from benchmarks.benchmarker.restage_asr import execute_asr_trial
from benchmarks.benchmarker.restage_probe import is_media_reference, load_samples
from benchmarks.benchmarker.restage_tts import execute_tts_trial
from benchmarks.dataset.seedtts import SampleInput
from sglang_omni.restage.evaluation import SLO, Evaluation
from sglang_omni.restage.search import search_rates
from sglang_omni.restage.selection import Selection, select_candidate

# Note (Jiaxin Deng): bracket the baseline's predicted rate and the best
# candidate's, so both the shipped layout and the winner are measured near
# their own knees on one shared grid.
BASELINE_BRACKET = (0.5, 0.75, 1.0)
BEST_BRACKET = (0.5, 0.75, 1.0)


def predicted_rate_grid(predicted, baseline):
    best = max(predicted.values())
    rates = {round(predicted[baseline] * f, 3) for f in BASELINE_BRACKET}
    rates |= {round(best * f, 3) for f in BEST_BRACKET}
    return sorted(rates)


async def execute_campaign(
    *,
    configs: dict[str, Path],
    baseline: str,
    rates: list[float],
    repeats: int,
    arrival_seed: int,
    destination: Path,
    trial_options: dict,
    task: str = "tts",
    run_identity: str | None = None,
    resume: bool = False,
    predicted: dict[str, float] | None = None,
    identity_samples: object = None,
) -> Selection:
    """Measure supplied candidates with identical workload/SLO and paired arrivals.

    The caller supplies admitted hardware and model-specific options. Any trial
    exception stops the campaign, retaining completed trial evidence. Resume
    reuses completed trials and requires the recorded campaign identity;
    ``identity_samples`` stands in for staged sample paths in that identity.
    """
    runners = {"tts": execute_tts_trial, "asr": execute_asr_trial}
    if task not in runners:
        raise ValueError(f"Unsupported campaign task: {task}")
    if resume and not run_identity:
        raise ValueError("Resume requires an explicit frozen run_identity")
    if not rates or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("Rates must be nonempty, finite and positive")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if baseline not in configs:
        raise ValueError("Include the baseline configuration")
    run_trial = runners[task]
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    config_bytes = {key: path.read_bytes() for key, path in configs.items()}
    snapshots = {
        key: destination / f"candidate-{index:05d}.yaml"
        for index, key in enumerate(configs)
    }
    metadata = {
        "schema_version": 2,
        "run_identity": run_identity,
        "trial_inputs": _input_identity(
            trial_options
            if identity_samples is None
            else {**trial_options, "samples": identity_samples}
        ),
        "task": task,
        "baseline": baseline,
        "rates": rates,
        "repeats": repeats,
        "arrival_seed": arrival_seed,
        "predicted_requests_per_s": predicted or {},
        "configs": {
            key: {
                "file": path.name,
                "sha256": hashlib.sha256(config_bytes[key]).hexdigest(),
            }
            for key, path in snapshots.items()
        },
    }
    manifest = destination / "campaign.json"
    completed = {}
    checkpoint = destination / "completed-trials.json"

    def write_trial_log():
        _atomic_write(
            destination / "trials.jsonl",
            "".join(json.dumps(row) + "\n" for row in completed.values()),
        )

    if resume:
        if json.loads(manifest.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Campaign identity differs from the recorded run")
        if checkpoint.exists():
            for row in json.loads(checkpoint.read_text(encoding="utf-8")):
                completed[row["candidate"], row["rate"], row["repeat"]] = row
        write_trial_log()
    else:
        destination.mkdir(parents=True, exist_ok=False)
        for key, path in snapshots.items():
            path.write_bytes(config_bytes[key])
        manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    results = {}
    base_port = trial_options.get("port")
    for index, (key, path) in enumerate(snapshots.items()):

        async def trial(rate, repeat):
            if (key, rate, repeat) in completed:
                return Evaluation(**completed[key, rate, repeat]["evaluation"])
            base_dir = (
                destination / f"candidate-{index:05d}-rate-{rate.hex()}-repeat-{repeat}"
            )
            trial_dir = base_dir
            attempt = 0
            while trial_dir.exists():
                attempt += 1
                trial_dir = base_dir.with_name(f"{base_dir.name}-attempt-{attempt:05d}")
            options = dict(trial_options)
            if base_port is not None:
                # Note (Jiaxin Deng): rotate ports so a stopped server's TIME_WAIT
                # sockets never block the next trial on the same host.
                options["port"] = base_port + (len(completed) % 32)
            try:
                evaluation = await run_trial(
                    config_path=path,
                    destination=trial_dir,
                    rate=rate,
                    arrival_seed=arrival_seed + repeat,
                    **options,
                )
            except BaseException as exc:
                _atomic_write(
                    destination / "failure.json",
                    json.dumps(
                        {
                            "candidate": key,
                            "rate": rate,
                            "repeat": repeat,
                            "directory": trial_dir.name,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        indent=2,
                    ),
                )
                raise
            row = {
                "candidate": key,
                "rate": rate,
                "repeat": repeat,
                "arrival_seed": arrival_seed + repeat,
                "directory": trial_dir.name,
                "evaluation": asdict(evaluation),
            }
            _atomic_write(checkpoint, json.dumps([*completed.values(), row], indent=2))
            completed[key, rate, repeat] = row
            write_trial_log()
            return evaluation

        results[key] = await search_rates(
            trial, [float(rate) for rate in rates], repeats=repeats
        )
        (destination / f"candidate-{index:05d}-search.json").write_text(
            json.dumps(asdict(results[key]), indent=2), encoding="utf-8"
        )
    selection = select_candidate(results, baseline=baseline, predicted=predicted)
    (destination / "selection.json").write_text(
        json.dumps(asdict(selection), indent=2), encoding="utf-8"
    )
    if selection.recommended is not None:
        shutil.copyfile(
            snapshots[selection.recommended], destination / "recommended.yaml"
        )
    return selection


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value.resolve())
    raise TypeError(f"Unsupported campaign option: {type(value).__name__}")


def _atomic_write(path, text):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _input_identity(options):
    return json.loads(json.dumps(options, default=_json_default, allow_nan=False))


def load_campaign_spec(path: Path) -> dict:
    """Load a campaign with local asset/config paths relative to its spec.

    A spec names either ``configs`` (key to YAML) with explicit ``rates`` and
    an optional ``predicted`` requests/s per key, or a ``plan_directory``
    written by ``autotune plan``; the plan supplies the
    baseline, the exported candidates, their predicted rates and, when
    ``rates`` is absent, a bracket below the best predicted rate.
    """
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
        base = path.resolve().parent
        predicted = None
        if "plan_directory" in spec:
            if "configs" in spec or "baseline" in spec:
                raise ValueError(
                    "A plan_directory supplies configs and baseline itself"
                )
            from sglang_omni.restage.plan import load_plan

            spec["configs"], rows = load_plan(base / spec.pop("plan_directory"))
            spec["baseline"] = "baseline"
            predicted = {key: row["requests_per_s"] for key, row in rows.items()}
            if "rates" not in spec:
                spec["rates"] = predicted_rate_grid(predicted, "baseline")
        else:
            spec["configs"] = {
                key: base / value for key, value in spec["configs"].items()
            }
            predicted = spec.pop("predicted", None)
            if predicted is not None and set(predicted) != set(spec["configs"]):
                raise ValueError("predicted must name exactly the configs")
        spec["predicted"] = predicted
        options = spec["trial_options"]
        task = spec.get("task", "tts")
        sender = options.get("sender_options") or {}
        needs_audio = task == "asr" or not sender.get("no_ref_audio", False)
        # Note (Jiaxin Deng): a dataset source stages audio into a fresh temp
        # dir per run, so the identity keeps the source spec, not the paths.
        if isinstance(options["samples"], dict):
            spec["identity_samples"] = options["samples"]
        options["samples"] = load_samples(options["samples"], base)
        inputs = list(options["samples"])
        if options.get("warmup_sample") is not None:
            options["warmup_sample"] = SampleInput(**options["warmup_sample"])
            inputs.append(options["warmup_sample"])
        for sample in inputs:
            media_reference = is_media_reference(sample.ref_audio)
            if sample.ref_audio and not media_reference:
                sample.ref_audio = str(base / Path(sample.ref_audio).expanduser())
            if needs_audio:
                if task == "asr" and media_reference:
                    raise ValueError("ASR samples require local audio files")
                if not media_reference and (
                    not sample.ref_audio or not Path(sample.ref_audio).is_file()
                ):
                    raise ValueError(
                        f"Reference audio file does not exist: {sample.ref_audio}"
                    )
        options["slo"] = SLO(**options["slo"])
        if task == "tts":
            options["asr_config_path"] = base / options["asr_config_path"]
            if not options["asr_config_path"].is_file():
                raise ValueError(
                    f"ASR configuration file does not exist: {options['asr_config_path']}"
                )
        return spec
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Invalid campaign spec {path}: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    spec = load_campaign_spec(args.spec)
    selection = asyncio.run(
        execute_campaign(destination=args.output, resume=args.resume, **spec)
    )
    print(json.dumps(asdict(selection), indent=2))


if __name__ == "__main__":
    main()
