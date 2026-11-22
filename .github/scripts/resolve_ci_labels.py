# SPDX-License-Identifier: Apache-2.0
"""Resolve the CI stages and models that pull request labels select.

A stage runs with its own stage label. A model label picks its model for every
selected stage that accepts it. When no stage that accepts it is selected, an
ASR, TTS or Omni model label selects the stages that test its model:
run-asr-ci covers two stages (MOSS-TD multi-speaker and SeedTTS), and a model
label alone runs only its own. TTS runs Higgs serving stress with run-tts-ci,
or when the Higgs label selects it. A model label never selects MPS, so Higgs
and MOSS labels, shared by TTS and MPS, select only TTS. Every stage except
Mac also runs PR Test on H100. run-lint-only satisfies the gate and runs no
self-hosted job. A selected stage without a model label draws its model at
random.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

RUN_LINT_ONLY_LABEL = "run-lint-only"
ASR_STAGE_LABEL = "run-asr-ci"
TTS_STAGE_LABEL = "run-tts-ci"
HIGGS_TTS_MODEL_LABEL = "run-higgs-tts-3-4b"
MOSS_TTS_MODEL_LABEL = "run-moss-tts-local-transformer-v1.5"
OMNI_CI_WORKFLOW = "Omni CI"


@dataclass(frozen=True, kw_only=True)
class CiStage:
    name: str
    label: str
    workflow: str
    runs_pr_test: bool
    preset_by_model_label: dict[str, str]
    random_presets: tuple[str, ...]
    is_selected_by_model_label: bool = False
    # note (0xtoward): directories under sglang_omni/models that a stage with one fixed model loads.
    model_directories: tuple[str, ...] = ()


CI_STAGES: tuple[CiStage, ...] = (
    CiStage(
        name="unit_test",
        label="run-unit-test-ci-only",
        workflow=OMNI_CI_WORKFLOW,
        runs_pr_test=True,
        preset_by_model_label={},
        random_presets=(),
    ),
    CiStage(
        name="asr_multi_speaker",
        label=ASR_STAGE_LABEL,
        workflow=OMNI_CI_WORKFLOW,
        runs_pr_test=True,
        preset_by_model_label={"run-moss-transcribe-diarize": "moss-td"},
        random_presets=("moss-td",),
        is_selected_by_model_label=True,
    ),
    CiStage(
        name="asr_seedtts",
        label=ASR_STAGE_LABEL,
        workflow=OMNI_CI_WORKFLOW,
        runs_pr_test=True,
        preset_by_model_label={
            "run-ark-asr-3b": "ark",
            "run-fun-asr-nano-2512": "fun",
            "run-qwen3-asr-1.7b": "qwen3",
            "run-whisper-large-v3": "whisper",
        },
        random_presets=("ark", "fun", "qwen3", "whisper"),
        is_selected_by_model_label=True,
    ),
    CiStage(
        name="tts",
        label=TTS_STAGE_LABEL,
        workflow=OMNI_CI_WORKFLOW,
        runs_pr_test=True,
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
        is_selected_by_model_label=True,
    ),
    CiStage(
        name="mps",
        label="run-mps-ci",
        workflow=OMNI_CI_WORKFLOW,
        runs_pr_test=True,
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
        runs_pr_test=True,
        preset_by_model_label={
            "run-qwen3-omni-30b-a3b-instruct": "qwen3-omni",
            "run-minicpm-o-4.5": "minicpmo",
        },
        random_presets=("qwen3-omni", "minicpmo"),
        is_selected_by_model_label=True,
    ),
    CiStage(
        name="full_duplex",
        label="run-native-duplex",
        workflow=OMNI_CI_WORKFLOW,
        runs_pr_test=True,
        preset_by_model_label={},
        random_presets=(),
        model_directories=("minicpm_o",),
    ),
    CiStage(
        name="turn_based",
        label="run-turn-based-duplex",
        workflow=OMNI_CI_WORKFLOW,
        runs_pr_test=True,
        preset_by_model_label={},
        random_presets=(),
        model_directories=("qwen3_omni", "vad"),
    ),
    CiStage(
        name="cascaded",
        label="run-cascaded-duplex",
        workflow=OMNI_CI_WORKFLOW,
        runs_pr_test=True,
        preset_by_model_label={},
        random_presets=(),
        model_directories=("streaming_cascade", "vad"),
    ),
    CiStage(
        name="intel_cpu",
        label="run-intel-cpu-ci",
        workflow="CPU CI",
        runs_pr_test=True,
        preset_by_model_label={},
        random_presets=(),
    ),
    CiStage(
        name="intel_xpu",
        label="run-intel-xpu-ci",
        workflow="XPU CI",
        runs_pr_test=True,
        preset_by_model_label={},
        random_presets=(),
    ),
    CiStage(
        name="mac",
        label="run-mac-ci",
        workflow="Mac CI",
        runs_pr_test=False,
        preset_by_model_label={},
        random_presets=(),
    ),
)
MPS_CONFIG_BY_PRESET = {
    "higgs": "examples/mps_dp/configs/higgs_h100_dp3.yaml",
    "moss": "examples/mps_dp/configs/moss_local_h100_dp2.yaml",
}
GATED_WORKFLOWS = frozenset(stage.workflow for stage in CI_STAGES)
SHORT_TARGET_BY_SHARED_STAGE_LABEL = {ASR_STAGE_LABEL: "asr", TTS_STAGE_LABEL: "tts"}
SOURCE_ROOT = "sglang_omni/"
MODEL_DIRECTORY_ROOT = "sglang_omni/models/"
MODEL_DIRECTORIES_BY_PRESET = {
    "moss-td": ("moss_transcribe_diarize",),
    "fun": ("fun_asr",),
    "qwen3": ("qwen3_asr",),
    "whisper": ("whisper_asr",),
    "higgs": ("higgs_tts",),
    # note (0xtoward): MOSS-TTS Local builds on the audio tokenizer and modules of moss_tts.
    "moss": ("moss_tts_local", "moss_tts"),
    "qwen3-tts": ("qwen3_tts",),
    "qwen3-tts-custom-voice": ("qwen3_tts",),
    "cosyvoice3": ("fun_cosyvoice3",),
    "qwen3-omni": ("qwen3_omni",),
    "minicpmo": ("minicpm_o",),
}
MAX_TARGETS_PER_SUGGESTION = 2


@dataclass(frozen=True, kw_only=True)
class CiPlan:
    enabled_stage_names: frozenset[str]
    preset_by_stage_name: dict[str, str]
    runs_tts_serving: bool


@dataclass(frozen=True, kw_only=True)
class StageSuggestion:
    model_directory: str
    targets: tuple[str, ...]
    is_in_command: bool
    stage_model_directories: tuple[str, ...]


def select_stage_names(labels: frozenset[str]) -> frozenset[str]:
    """A model label adds its stages only if no stage accepting it is selected.

    So run-mps-ci with the Higgs label pins MPS's model and adds no TTS stage.
    """
    stage_names = {stage.name for stage in CI_STAGES if stage.label in labels}
    for model_label in labels:
        accepting_stages = [
            stage for stage in CI_STAGES if model_label in stage.preset_by_model_label
        ]
        if not any(stage.name in stage_names for stage in accepting_stages):
            stage_names.update(
                stage.name
                for stage in accepting_stages
                if stage.is_selected_by_model_label
            )
        else:
            pass
    return frozenset(stage_names)


def should_run_pr_test(enabled_stage_names: frozenset[str]) -> bool:
    return any(
        stage.runs_pr_test for stage in CI_STAGES if stage.name in enabled_stage_names
    )


def select_workflows(labels: frozenset[str]) -> frozenset[str]:
    """A selected stage that runs PR Test also opts Omni CI in."""
    enabled_stage_names = select_stage_names(labels)
    stage_workflows = frozenset(
        stage.workflow for stage in CI_STAGES if stage.name in enabled_stage_names
    )
    if should_run_pr_test(enabled_stage_names):
        return stage_workflows | {OMNI_CI_WORKFLOW}
    else:
        return stage_workflows


def build_labels_by_target() -> dict[str, frozenset[str]]:
    """The labels that each /tag-and-rerun-ci target adds."""
    labels_by_target: dict[str, frozenset[str]] = {
        "lint": frozenset({RUN_LINT_ONLY_LABEL}),
    }
    for stage in CI_STAGES:
        if stage.label in SHORT_TARGET_BY_SHARED_STAGE_LABEL:
            stage_target = SHORT_TARGET_BY_SHARED_STAGE_LABEL[stage.label]
        else:
            stage_target = stage.name.replace("_", "-")
        labels_by_target[stage_target] = frozenset({stage.label})
        for label, preset in stage.preset_by_model_label.items():
            if stage.label == ASR_STAGE_LABEL:
                target = f"{preset}-asr"
            elif stage.name == "mps":
                target = f"mps-{preset}"
            else:
                target = preset
            if stage.is_selected_by_model_label:
                labels_by_target[target] = frozenset({label})
            else:
                labels_by_target[target] = frozenset({stage.label, label})
    return labels_by_target


def labels_after_tagging(
    current_labels: frozenset[str], requested_labels: frozenset[str]
) -> frozenset[str]:
    """A requested model label replaces the other model labels of each stage that accepts it."""
    removed_labels: set[str] = set()
    for stage in CI_STAGES:
        model_labels = stage.preset_by_model_label.keys()
        if requested_labels & model_labels:
            removed_labels.update((current_labels & model_labels) - requested_labels)
        else:
            pass
    return (current_labels - removed_labels) | requested_labels


def is_model_directory_tested(
    stage: CiStage, model_directory: str, labels: frozenset[str]
) -> bool:
    """Whether the stage is selected and the one model it runs lives in the directory."""
    if stage.name not in select_stage_names(labels):
        return False
    elif model_directory in stage.model_directories:
        return True
    else:
        pinned_presets = [
            preset
            for label, preset in stage.preset_by_model_label.items()
            if label in labels
        ]
        runnable_presets = pinned_presets or list(stage.random_presets)
        return bool(runnable_presets) and all(
            model_directory in MODEL_DIRECTORIES_BY_PRESET[preset]
            for preset in runnable_presets
        )


def suggest_stage_targets(
    changed_paths: list[str], labels: frozenset[str]
) -> list[StageSuggestion]:
    """The /tag-and-rerun-ci targets that make a stage test each model directory a pull request changes.

    A stage runs one model, so of several changed models the first enters the command and each other one comes with its own targets, whatever the labels select. The stages are the ones a model label selects and the ones with a fixed model. The targets are searched with the label rules of this module, so a rule change needs no edit here.
    """
    changed_model_directories = {
        path.removeprefix(MODEL_DIRECTORY_ROOT).split("/")[0]
        for path in changed_paths
        if path.startswith(MODEL_DIRECTORY_ROOT)
    }
    labels_by_target = build_labels_by_target()
    target_sets = [
        targets
        for target_count in range(1, MAX_TARGETS_PER_SUGGESTION + 1)
        for targets in combinations(sorted(labels_by_target), target_count)
    ]
    suggestions: list[StageSuggestion] = []
    command_targets: tuple[str, ...] = ()
    for stage in CI_STAGES:
        served_model_directories = set(stage.model_directories)
        if stage.is_selected_by_model_label:
            served_model_directories.update(
                model_directory
                for preset in (
                    *stage.preset_by_model_label.values(),
                    *stage.random_presets,
                )
                for model_directory in MODEL_DIRECTORIES_BY_PRESET[preset]
            )
        else:
            pass
        changed_served_directories = sorted(
            served_model_directories & changed_model_directories
        )
        tagged_labels_by_targets = {
            targets: labels_after_tagging(
                labels,
                frozenset().union(
                    *(
                        labels_by_target[target]
                        for target in (*command_targets, *targets)
                    )
                ),
            )
            for targets in ([(), *target_sets] if changed_served_directories else [()])
        }
        untested_model_directories = [
            model_directory
            for model_directory in changed_served_directories
            if not is_model_directory_tested(
                stage, model_directory, tagged_labels_by_targets[()]
            )
        ]
        is_command_needed = untested_model_directories == changed_served_directories
        suggested_target_sets: set[tuple[str, ...]] = set()
        for model_directory in untested_model_directories:
            targets = min(
                (
                    targets
                    for targets, tagged_labels in tagged_labels_by_targets.items()
                    if is_model_directory_tested(stage, model_directory, tagged_labels)
                ),
                key=lambda targets: (
                    len(targets),
                    len(select_stage_names(tagged_labels_by_targets[targets])),
                    targets,
                ),
            )
            # note (0xtoward): directories of one preset share its targets, and one run tests them all.
            if targets in suggested_target_sets:
                pass
            else:
                is_in_command = is_command_needed and not suggested_target_sets
                suggested_target_sets.add(targets)
                suggestions.append(
                    StageSuggestion(
                        model_directory=model_directory,
                        targets=targets,
                        is_in_command=is_in_command,
                        stage_model_directories=tuple(changed_served_directories),
                    )
                )
                if is_in_command:
                    command_targets = (*command_targets, *targets)
                else:
                    pass
    return suggestions


def suggest_shared_code_targets(
    changed_paths: list[str], labels: frozenset[str]
) -> list[str]:
    """The stage targets for a change to shared code alone, when the labels select no model stage."""
    source_paths = [path for path in changed_paths if path.startswith(SOURCE_ROOT)]
    changes_shared_code = bool(source_paths) and not any(
        "/" in path.removeprefix(MODEL_DIRECTORY_ROOT)
        for path in source_paths
        if path.startswith(MODEL_DIRECTORY_ROOT)
    )
    selected_stage_names = select_stage_names(labels)
    is_model_stage_selected = any(
        stage.name in selected_stage_names
        and bool(stage.preset_by_model_label or stage.model_directories)
        for stage in CI_STAGES
    )
    if changes_shared_code and not is_model_stage_selected:
        stage_labels = {
            stage.label for stage in CI_STAGES if stage.is_selected_by_model_label
        }
        return [
            target
            for target, target_labels in build_labels_by_target().items()
            if target_labels <= stage_labels
        ]
    else:
        return []


def resolve_ci_plan(labels: frozenset[str], seed: str) -> CiPlan:
    """Raises ValueError without a CI label or with two model labels in a stage."""
    enabled_stage_names = select_stage_names(labels)
    if not enabled_stage_names and RUN_LINT_ONLY_LABEL not in labels:
        stage_labels = ", ".join(dict.fromkeys(stage.label for stage in CI_STAGES))
        raise ValueError(
            f"No CI label found. Add {RUN_LINT_ONLY_LABEL}, a model label or a stage "
            f"label: {stage_labels}."
        )
    else:
        pass

    preset_by_stage_name: dict[str, str] = {}
    for stage in CI_STAGES:
        model_labels = sorted(stage.preset_by_model_label.keys() & labels)
        if len(model_labels) > 1:
            raise ValueError(
                f"Keep only one {stage.name} model label: {', '.join(model_labels)}."
            )
        elif stage.name not in enabled_stage_names:
            pass
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
    runs_tts_serving = TTS_STAGE_LABEL in labels or (
        "tts" in enabled_stage_names and HIGGS_TTS_MODEL_LABEL in labels
    )
    return CiPlan(
        enabled_stage_names=enabled_stage_names,
        preset_by_stage_name=preset_by_stage_name,
        runs_tts_serving=runs_tts_serving,
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
    output_lines.append(f"run_tts_serving={str(plan.runs_tts_serving).lower()}")
    is_self_hosted = should_run_pr_test(plan.enabled_stage_names)
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
    output_lines = format_github_outputs(plan)
    print("\n".join(output_lines))
    with arguments.github_output.open("a", encoding="utf-8") as output_file:
        output_file.write("".join(f"{line}\n" for line in output_lines))


if __name__ == "__main__":
    main()
