# SPDX-License-Identifier: Apache-2.0
"""Resolve the CI stages and models that pull request labels select.

A stage runs with its stage label or run-all-ci. run-lint-only satisfies the
gate and runs no self-hosted job. A model label names a model, not a stage.
It picks that model for every selected stage that accepts it, and never
selects a stage. Higgs and MOSS labels are shared by TTS and MPS. A selected
stage without a model label draws its model at random.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

RUN_ALL_LABEL = "run-all-ci"
RUN_LINT_ONLY_LABEL = "run-lint-only"
HIGGS_TTS_MODEL_LABEL = "run-higgs-tts-3-4b"
MOSS_TTS_MODEL_LABEL = "run-moss-tts-local-transformer-v1.5"
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
            "run-fun-asr-nano-2512": "fun",
            "run-qwen3-asr-1.7b": "qwen3",
            "run-whisper-large-v3": "whisper",
        },
        random_presets=("fun", "qwen3", "whisper"),
    ),
    CiStage(
        name="tts",
        label="run-tts-ci",
        workflow=OMNI_CI_WORKFLOW,
        preset_by_model_label={
            HIGGS_TTS_MODEL_LABEL: "higgs",
            MOSS_TTS_MODEL_LABEL: "moss",
            "run-qwen3-tts-12hz-1.7b-base": "qwen3-tts",
            "run-qwen3-tts-12hz-1.7b-customvoice": "qwen3-tts-custom-voice",
            "run-fun-cosyvoice3-0.5b-2512": "cosyvoice3",
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
            HIGGS_TTS_MODEL_LABEL: "higgs",
            MOSS_TTS_MODEL_LABEL: "moss",
        },
        random_presets=("higgs", "moss"),
    ),
    CiStage(
        name="omni",
        label="run-omni-ci",
        workflow=OMNI_CI_WORKFLOW,
        preset_by_model_label={
            "run-qwen3-omni-30b-a3b-instruct": "qwen3-omni",
            "run-minicpm-o-4.5": "minicpmo",
        },
        random_presets=("qwen3-omni", "minicpmo"),
    ),
    CiStage(
        name="full_duplex",
        label="run-full-duplex",
        workflow=OMNI_CI_WORKFLOW,
        preset_by_model_label={},
        random_presets=(),
    ),
    CiStage(
        name="turn_based",
        label="run-turn-based",
        workflow=OMNI_CI_WORKFLOW,
        preset_by_model_label={},
        random_presets=(),
    ),
    CiStage(
        name="cascaded",
        label="run-cascaded",
        workflow=OMNI_CI_WORKFLOW,
        preset_by_model_label={},
        random_presets=(),
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
    """Raises ValueError without a CI label or with two model labels in a stage."""
    enabled_stage_names = select_stage_names(labels)
    if not enabled_stage_names and RUN_LINT_ONLY_LABEL not in labels:
        stage_labels = ", ".join(stage.label for stage in CI_STAGES)
        raise ValueError(
            "No CI label found. Add "
            f"{RUN_ALL_LABEL}, {RUN_LINT_ONLY_LABEL}, "
            f"or a stage label: {stage_labels}."
        )
    else:
        pass

    preset_by_stage_name: dict[str, str] = {}
    unused_model_labels: list[str] = []
    used_model_labels: set[str] = set()
    for stage in CI_STAGES:
        model_labels = sorted(stage.preset_by_model_label.keys() & labels)
        if len(model_labels) > 1:
            raise ValueError(
                f"Keep only one {stage.name} model label: {', '.join(model_labels)}."
            )
        elif stage.name not in enabled_stage_names:
            unused_model_labels.extend(model_labels)
        elif model_labels:
            selected_model_label = model_labels[0]
            preset_by_stage_name[stage.name] = stage.preset_by_model_label[
                selected_model_label
            ]
            used_model_labels.add(selected_model_label)
        elif stage.random_presets:
            preset_by_stage_name[stage.name] = random.Random(
                f"{seed}-{stage.name}"
            ).choice(stage.random_presets)
        else:
            pass
    ignored_model_labels = tuple(
        dict.fromkeys(
            label for label in unused_model_labels if label not in used_model_labels
        )
    )
    return CiPlan(
        enabled_stage_names=enabled_stage_names,
        preset_by_stage_name=preset_by_stage_name,
        ignored_model_labels=ignored_model_labels,
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
    is_self_hosted = bool(plan.enabled_stage_names)
    output_lines.append(f"run_self_hosted={str(is_self_hosted).lower()}")
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
