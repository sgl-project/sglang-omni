# SPDX-License-Identifier: Apache-2.0
"""Resolve the CI stages and models that pull request labels select.

A stage runs with its stage label or run-all-ci. A model label picks the model of
a selected stage; a selected stage without one draws its model at random.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

RUN_ALL_LABEL = "run-all-ci"
OMNI_CI_WORKFLOW = "Omni CI"


@dataclass(frozen=True, kw_only=True)
class CiStage:
    name: str
    label: str
    workflow: str
    preset_by_model_label: dict[str, str]
    random_presets: tuple[str, ...]


CI_STAGES: tuple[CiStage, ...] = (
    CiStage(
        name="unit_test",
        label="run-unit-test-ci-only",
        workflow=OMNI_CI_WORKFLOW,
        preset_by_model_label={},
        random_presets=(),
    ),
    CiStage(
        name="asr",
        label="run-asr-ci",
        workflow=OMNI_CI_WORKFLOW,
        preset_by_model_label={
            "run-asr-fun-asr-nano-2512": "fun",
            "run-asr-qwen3-asr-1.7b": "qwen3",
            "run-asr-whisper-large-v3": "whisper",
        },
        random_presets=("fun", "qwen3", "whisper"),
    ),
    CiStage(
        name="tts",
        label="run-tts-ci",
        workflow=OMNI_CI_WORKFLOW,
        preset_by_model_label={
            "run-tts-higgs-tts-3-4b": "higgs",
            "run-tts-moss-tts-local-transformer-v1.5": "moss",
            "run-tts-qwen3-tts-12hz-1.7b-base": "qwen3-tts",
            "run-tts-qwen3-tts-12hz-1.7b-customvoice": "qwen3-tts-custom-voice",
            "run-tts-fun-cosyvoice3-0.5b-2512": "cosyvoice3",
        },
        # note (chenyang): qwen3-tts-custom-voice gates nothing until it is
        # calibrated on the CI host, so only its model label selects it.
        random_presets=("higgs", "moss", "qwen3-tts", "cosyvoice3"),
    ),
    CiStage(
        name="mps",
        label="run-mps-ci",
        workflow=OMNI_CI_WORKFLOW,
        preset_by_model_label={
            "run-mps-higgs-tts-3-4b": "higgs",
            "run-mps-moss-tts-local-transformer-v1.5": "moss",
        },
        random_presets=("higgs", "moss"),
    ),
    CiStage(
        name="omni",
        label="run-omni-ci",
        workflow=OMNI_CI_WORKFLOW,
        preset_by_model_label={
            "run-omni-qwen3-omni-30b-a3b-instruct": "qwen3-omni",
            "run-omni-minicpm-o-4.5": "minicpmo",
        },
        random_presets=("qwen3-omni", "minicpmo"),
    ),
    CiStage(
        name="intel_cpu",
        label="run-intel-cpu-ci",
        workflow="CPU CI",
        preset_by_model_label={},
        random_presets=(),
    ),
    CiStage(
        name="intel_xpu",
        label="run-intel-xpu-ci",
        workflow="XPU CI",
        preset_by_model_label={},
        random_presets=(),
    ),
    CiStage(
        name="mac",
        label="run-mac-ci",
        workflow="Voxt Mac CI",
        preset_by_model_label={},
        random_presets=(),
    ),
)
MPS_CONFIG_BY_PRESET = {
    "higgs": "examples/mps_dp/configs/higgs_h100_dp3.yaml",
    "moss": "examples/mps_dp/configs/moss_local_h100_dp2.yaml",
}
GATED_WORKFLOWS = frozenset(stage.workflow for stage in CI_STAGES)


@dataclass(frozen=True, kw_only=True)
class CiPlan:
    enabled_stage_names: frozenset[str]
    preset_by_stage_name: dict[str, str]
    ignored_model_labels: tuple[str, ...]


def select_stage_names(labels: frozenset[str]) -> frozenset[str]:
    if RUN_ALL_LABEL in labels:
        return frozenset(stage.name for stage in CI_STAGES)
    else:
        return frozenset(stage.name for stage in CI_STAGES if stage.label in labels)


def select_workflows(labels: frozenset[str]) -> frozenset[str]:
    """Every selected stage also opts Omni CI in, which runs the unit tests."""
    enabled_stage_names = select_stage_names(labels)
    if enabled_stage_names:
        return frozenset(
            {OMNI_CI_WORKFLOW}
            | {
                stage.workflow
                for stage in CI_STAGES
                if stage.name in enabled_stage_names
            }
        )
    else:
        return frozenset()


def resolve_ci_plan(labels: frozenset[str], seed: str) -> CiPlan:
    """Raises ValueError without a stage label or with two model labels in a stage."""
    enabled_stage_names = select_stage_names(labels)
    if not enabled_stage_names:
        stage_labels = ", ".join(stage.label for stage in CI_STAGES)
        raise ValueError(
            f"No CI label found. Add {RUN_ALL_LABEL} or a stage label: {stage_labels}."
        )
    else:
        pass

    preset_by_stage_name: dict[str, str] = {}
    ignored_model_labels: list[str] = []
    for stage in CI_STAGES:
        model_labels = sorted(stage.preset_by_model_label.keys() & labels)
        if len(model_labels) > 1:
            raise ValueError(
                f"Keep only one {stage.name} model label: {', '.join(model_labels)}."
            )
        elif stage.name not in enabled_stage_names:
            ignored_model_labels.extend(model_labels)
        elif model_labels:
            preset_by_stage_name[stage.name] = stage.preset_by_model_label[
                model_labels[0]
            ]
        elif stage.random_presets:
            preset_by_stage_name[stage.name] = random.Random(
                f"{seed}-{stage.name}"
            ).choice(stage.random_presets)
        else:
            pass
    return CiPlan(
        enabled_stage_names=enabled_stage_names,
        preset_by_stage_name=preset_by_stage_name,
        ignored_model_labels=tuple(ignored_model_labels),
    )


def format_github_outputs(plan: CiPlan) -> list[str]:
    output_lines: list[str] = []
    for stage in CI_STAGES:
        is_enabled = stage.name in plan.enabled_stage_names
        output_lines.append(f"run_{stage.name}={str(is_enabled).lower()}")
        if stage.preset_by_model_label:
            preset = plan.preset_by_stage_name.get(stage.name, "")
            output_lines.append(f"{stage.name}_ci_model={preset}")
        else:
            pass
    mps_preset = plan.preset_by_stage_name.get("mps", "")
    output_lines.append(f"mps_ci_config={MPS_CONFIG_BY_PRESET.get(mps_preset, '')}")
    return output_lines


def parse_labels(labels_json: str) -> frozenset[str]:
    labels = json.loads(labels_json)
    if isinstance(labels, list) and all(isinstance(label, str) for label in labels):
        return frozenset(labels)
    else:
        raise ValueError(f"Expected a JSON list of label names, got {labels_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    resolve_parser = subparsers.add_parser(
        "resolve", help="Append the selected stages and models to GITHUB_OUTPUT."
    )
    resolve_parser.add_argument("--labels-json", required=True)
    resolve_parser.add_argument(
        "--seed",
        required=True,
        help="Seeds the random model draw; the run id keeps it stable across reruns.",
    )
    resolve_parser.add_argument("--github-output", type=Path, required=True)
    cancel_parser = subparsers.add_parser(
        "workflows-to-cancel",
        help="Print, as a JSON list, the gated workflows the labels no longer opt into.",
    )
    cancel_parser.add_argument("--labels-json", required=True)
    arguments = parser.parse_args()

    labels = parse_labels(arguments.labels_json)
    if arguments.command == "workflows-to-cancel":
        print(json.dumps(sorted(GATED_WORKFLOWS - select_workflows(labels))))
        return
    else:
        pass

    try:
        plan = resolve_ci_plan(labels, arguments.seed)
    except ValueError as error:
        print(f"::error::{error}")
        sys.exit(1)
    for model_label in plan.ignored_model_labels:
        print(
            f"::warning::Ignoring {model_label}: its stage label and "
            f"{RUN_ALL_LABEL} are both missing."
        )
    output_lines = format_github_outputs(plan)
    print("\n".join(output_lines))
    with arguments.github_output.open("a", encoding="utf-8") as output_file:
        output_file.write("".join(f"{line}\n" for line in output_lines))


if __name__ == "__main__":
    main()
