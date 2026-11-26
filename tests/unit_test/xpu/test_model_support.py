# SPDX-License-Identifier: Apache-2.0
"""Per-model contract for every family Intel XPU serves, with no weights and no accelerator.

The registry logs and swallows a failed model import, so a broken config module only shows here as a missing architecture.
"""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

import pytest
import yaml

from scripts.xpu.baseline.models import DEFAULT_BASELINE, MODEL_BY_KEY
from scripts.xpu.make_baseline_readme import render
from sglang_omni.config.manager import resolve_config_cls_for_model_path
from sglang_omni.config.schema import PipelineConfig
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.platforms import current_platform
from sglang_omni.utils.imports import import_string
from tests.test_ci.xpu_model.conftest import holds_a_checkpoint
from tests.test_ci.xpu_model.exercises import EXERCISES
from tests.test_ci.xpu_model.model_table import (
    KNOWN_FAMILIES,
    XPU_MODELS,
    XpuModel,
    device_family,
    model_named,
    xpu_pipelines,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALL_GUIDE = REPO_ROOT / "docs" / "other_platforms" / "installation_xpu.md"
README = REPO_ROOT / "scripts" / "xpu" / "README.md"

PIPELINES = xpu_pipelines()
PIPELINE_IDS = [model.pipeline_key for model in PIPELINES]

# One flag per capture that loses a D270 device rather than falling back to eager.
D270_FLAGS = {
    "qwen3-tts": ["--tts_engine.engine.disable_cuda_graph", "true"],
    "auk": ["--auk_engine.factory.enable_dit_cuda_graph", "false"],
    "auk-flash": ["--auk_engine.factory.enable_dit_cuda_graph", "false"],
}


def config_cls(model: XpuModel) -> type[PipelineConfig]:
    module = importlib.import_module(f"sglang_omni.models.{model.package}.config")
    if model.variant is None:
        return module.EntryClass
    else:
        variants = getattr(module, "Variants", {})
    assert (
        model.variant in variants
    ), f"{model.package} exposes no {model.variant!r} variant; Variants={sorted(variants)}"
    return variants[model.variant]


@pytest.mark.parametrize("model", PIPELINES, ids=PIPELINE_IDS)
def test_the_config_module_imports_without_a_swallowed_error(model: XpuModel) -> None:
    """A direct import raises where the registry would only log and move on."""
    module = importlib.import_module(f"sglang_omni.models.{model.package}.config")
    assert hasattr(module, "EntryClass"), f"{model.package}.config has no EntryClass"


@pytest.mark.parametrize("model", PIPELINES, ids=PIPELINE_IDS)
def test_the_architecture_reaches_the_pipeline_config_registry(model: XpuModel) -> None:
    """An architecture missing here is a checkpoint no operator can serve."""
    registered = PIPELINE_CONFIG_REGISTRY.get_supported_archs()
    assert model.architecture in registered, (
        f"{model.architecture} is not registered, so a {model.package} checkpoint "
        "cannot be served. The registry swallows model import errors: import "
        f"sglang_omni.models.{model.package}.config directly to see the cause."
    )


@pytest.mark.parametrize("model", PIPELINES, ids=PIPELINE_IDS)
def test_the_pipeline_builds_the_stage_table_xpu_serves(model: XpuModel) -> None:
    """The topology is pinned, so a stage gained or lost surfaces here."""
    config = config_cls(model)(model_path="unused")
    assert tuple(stage.name for stage in config.stages) == model.stages
    if config.entry_stage is not None:
        assert config.entry_stage in model.stages
    else:
        pass


@pytest.mark.parametrize("model", PIPELINES, ids=PIPELINE_IDS)
def test_no_stage_pins_a_device_this_host_cannot_serve(model: XpuModel) -> None:
    """A stage leaves its device unset for the platform to resolve, or names this platform's own."""
    config = config_cls(model)(model_path="unused")
    live = current_platform.device_type
    for stage in config.stages:
        device = getattr(stage.factory, "device", None)
        if device is None:
            continue
        else:
            assert str(device).split(":")[0] == live, (
                f"{model.package} stage {stage.name!r} pins device={device!r}, which "
                f"this host ({live}) cannot serve"
            )


@pytest.mark.parametrize("model", PIPELINES, ids=PIPELINE_IDS)
def test_every_stage_factory_resolves_to_a_callable(model: XpuModel) -> None:
    """A renamed or moved factory fails at stage launch, minutes into a load."""
    config = config_cls(model)(model_path="unused")
    for stage in config.stages:
        assert stage.factory_path, f"stage {stage.name!r} declares no factory_path"
        assert callable(
            import_string(stage.factory_path)
        ), f"{stage.factory_path} is not callable"


@pytest.mark.parametrize("model", PIPELINES, ids=PIPELINE_IDS)
def test_a_checkpoint_resolves_to_the_family_that_serves_it(
    model: XpuModel, tmp_path: Path
) -> None:
    """The model path finds the family from checkpoint metadata alone."""
    checkpoint = tmp_path / model.mirror_dirname
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"architectures": [model.architecture]})
    )
    module = importlib.import_module(f"sglang_omni.models.{model.package}.config")
    assert resolve_config_cls_for_model_path(str(checkpoint)) is module.EntryClass


def test_every_model_in_the_table_is_exercised_by_the_serving_lane() -> None:
    """The serving lane dispatches on kind and would raise KeyError on an unknown one, but only on a host with weights."""
    uncovered = sorted(
        f"{model.key} (kind={model.kind})"
        for model in XPU_MODELS
        if model.kind not in EXERCISES
    )
    assert not uncovered, f"tests/test_ci/xpu_model has no exercise for: {uncovered}"


def test_no_exercise_is_dead_code() -> None:
    """An exercise no row claims is a dropped model or a mistyped kind, and both read as coverage that is not there."""
    claimed = {model.kind for model in XPU_MODELS}
    assert not sorted(
        set(EXERCISES) - claimed
    ), f"no row uses: {sorted(set(EXERCISES) - claimed)}"


def test_each_row_is_a_distinct_checkpoint() -> None:
    keys = [model.key for model in XPU_MODELS]
    assert len(keys) == len(set(keys)), "duplicate key in XPU_MODELS"
    envs = [model.checkpoint_env for model in XPU_MODELS]
    assert len(envs) == len(set(envs)), "two rows derive the same checkpoint_env"


def test_a_tensor_parallel_row_claims_a_card_per_rank() -> None:
    """The lane claims a row's cards, and a tensor-parallel stage needs one per rank."""
    short = {m.key: (m.tp_size, m.cards) for m in XPU_MODELS if m.tp_size > m.cards}
    assert not short, f"rows claiming fewer cards than their TP degree: {short}"
    assert any(
        1 < m.tp_size <= m.cards <= 4 for m in XPU_MODELS
    ), "no tensor-parallel row fits a four-card host"


def test_every_model_the_install_guide_documents_has_a_row() -> None:
    """A model the guide documents with no row is one nothing checks still serves; a /path/to/ path names its repo."""
    documented = set(
        re.findall(r"--model-path[ =]+([^\s\\`'\"]+)", INSTALL_GUIDE.read_text())
    )
    repos = {model.repo_id for model in XPU_MODELS}
    names = {model.mirror_dirname for model in XPU_MODELS}
    missing = sorted(
        path
        for path in documented
        if path.partition("@")[0] not in repos and Path(path).name not in names
    )
    assert (
        not missing
    ), f"{INSTALL_GUIDE.name} documents {missing}, which have no row in model_table.py"


@pytest.mark.parametrize("model", XPU_MODELS, ids=[m.key for m in XPU_MODELS])
def test_a_stage_scoped_serve_argument_names_a_stage_this_row_has(
    model: XpuModel,
) -> None:
    """A flag aimed at a stage the pipeline does not have is rejected only when the engine is built, minutes into a load."""
    names = set(model.stages)
    arguments = model.serve_args
    for family in KNOWN_FAMILIES:
        arguments += model.serve_args_on(family)
    for argument in arguments:
        if not argument.startswith("--") or "." not in argument:
            continue
        else:
            stage = argument[2:].split(".", 1)[0]
        if stage in names or stage in PipelineConfig.model_fields or "-" in stage:
            continue
        else:
            pytest.fail(
                f"{model.key} passes {argument!r}, but its stages are "
                f"{sorted(names)}"
            )


@pytest.mark.parametrize("model", XPU_MODELS, ids=[m.key for m in XPU_MODELS])
def test_a_row_the_device_lane_skips_says_why(model: XpuModel) -> None:
    """A silent skip looks the same as a pass, so a skipped row must say why."""
    if model.serving:
        assert len(model.serving) > 20, (
            f"{model.key} declines to serve with an unhelpfully short reason: "
            f"{model.serving!r}"
        )
    else:
        assert model.kind, f"{model.key} claims to be servable with no kind"


def test_the_code2wav_stage_takes_the_graph_policy_from_the_platform() -> None:
    speech = config_cls(model_named("omni-speech"))(model_path="unused")
    kwargs = speech.stage_factory_kwargs("code2wav")
    assert kwargs["enable_cuda_graph"] is current_platform.enable_code2wav_graph()


def test_a_d270_card_gets_the_d270_recipe_and_an_unknown_card_the_base_one() -> None:
    """A misread name serves D270 with a capture that loses the device, or a B60 without a graph it records fine."""
    assert device_family("Intel(R) Graphics [0x674c]") == "d270"
    assert device_family("Intel(R) Arc(TM) Pro B60 Graphics") == "bmg"
    assert device_family("Intel(R) Data Center GPU Max 1550") == "unknown"
    assert device_family("Intel(R) Arc(TM) Pro B60 Graphics", "d270") == "d270"


@pytest.mark.parametrize("override", ["D270", " d270 ", "Bmg"])
def test_a_forced_family_is_read_case_blind(override: str) -> None:
    assert (
        device_family("Intel(R) Arc(TM) Pro B60 Graphics", override) in KNOWN_FAMILIES
    )


@pytest.mark.parametrize("override", ["d-270", "d27O", "b60"])
def test_a_forced_family_the_lane_cannot_name_is_refused(override: str) -> None:
    """A typo would otherwise serve the base recipe on the card the override exists for."""
    with pytest.raises(ValueError, match="names no family"):
        device_family("Intel(R) Graphics [0x674c]", override)


@pytest.mark.parametrize("model", XPU_MODELS, ids=[m.key for m in XPU_MODELS])
def test_a_row_pins_the_checkpoint_its_config_file_pins(model: XpuModel) -> None:
    """The row's checkpoint outranks the config's model_path, so a bumped pin in one is silently ignored by the other."""
    arguments = list(model.serve_args)
    if "--config" not in arguments:
        return
    else:
        config = yaml.safe_load(
            (REPO_ROOT / arguments[arguments.index("--config") + 1]).read_text()
        )
    configured = str(config.get("model_path", ""))
    if configured.partition("@")[0] == model.repo_id:
        assert (
            configured == model.checkpoint_spec
        ), f"{model.key} serves {model.checkpoint_spec}, but its config pins {configured}"
    else:
        pass


@pytest.mark.parametrize("model", XPU_MODELS, ids=[m.key for m in XPU_MODELS])
def test_device_args_name_a_family_the_lane_can_detect(model: XpuModel) -> None:
    """A misspelt family is a workaround that never applies, first seen as the device it protects going down."""
    for family, args in model.device_args:
        assert (
            family in KNOWN_FAMILIES
        ), f"{model.key} sets device_args for {family!r}; known: {KNOWN_FAMILIES}"
        assert args, f"{model.key} sets an empty device_args entry for {family!r}"


def test_the_d270_recipe_turns_off_exactly_the_captures_that_lose_the_device() -> None:
    """A flag added without a failing capture slows D270 for nothing, and one dropped takes the card down."""
    added = {
        model.key: list(model.serve_args_on("d270")[len(model.serve_args) :])
        for model in XPU_MODELS
        if model.serve_args_on("d270") != model.serve_args
    }
    assert added == D270_FLAGS


def test_the_committed_baseline_holds_b60_environments_only() -> None:
    """D270 numbers are recorded to a baseline outside the tree and are not published in this file."""
    with DEFAULT_BASELINE.open() as handle:
        environments = json.load(handle)["environments"].values()
    families = {
        device_family(environment["fingerprint"].get("device", ""))
        for environment in environments
    }
    assert families == {"bmg"}, f"non-B60 environments: {families}"


def test_every_model_the_committed_baseline_holds_is_one_it_still_measures() -> None:
    """An entry for a model the table no longer has would be gated against nothing and never re-recorded."""
    with DEFAULT_BASELINE.open() as handle:
        environments = json.load(handle)["environments"].values()
    orphaned = sorted(
        key
        for environment in environments
        for key in environment["models"]
        if key not in MODEL_BY_KEY
    )
    assert not orphaned, f"baseline entries with no model: {orphaned}"


def test_the_committed_readme_is_what_its_generator_writes() -> None:
    """The page is generated from the baseline and the table, so a hand edit or a stale page fails here."""
    assert README.read_text() == render(), (
        "scripts/xpu/README.md is stale; regenerate it with "
        "python -m scripts.xpu.make_baseline_readme > scripts/xpu/README.md"
    )


def test_only_a_complete_checkpoint_is_handed_to_the_server(tmp_path: Path) -> None:
    """Each case is a directory a host really held, which the server was given and failed on minutes later."""

    def checkpoint(name: str, files: dict[str, str]) -> Path:
        root = tmp_path / name
        for relative, text in files.items():
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            (root / relative).write_text(text)
        root.mkdir(exist_ok=True)
        return root

    index = json.dumps({"weight_map": {"a": "m-1.safetensors", "b": "m-2.safetensors"}})
    assert not holds_a_checkpoint(checkpoint("placeholder", {}))
    assert not holds_a_checkpoint(checkpoint("config_only", {"config.json": "{}"}))
    assert not holds_a_checkpoint(
        checkpoint(
            "half_fetched",
            {
                "config.json": "{}",
                "m.safetensors.index.json": index,
                "m-1.safetensors": "",
            },
        )
    )
    assert holds_a_checkpoint(
        checkpoint(
            "complete",
            {
                "config.json": "{}",
                "m.safetensors.index.json": index,
                "m-1.safetensors": "",
                "m-2.safetensors": "",
            },
        )
    )
    assert holds_a_checkpoint(
        checkpoint("nested", {"config.json": "{}", "ar/model.safetensors": ""})
    )
    assert holds_a_checkpoint(
        checkpoint("yaml_config", {"config.yaml": "", "auk_base.safetensors": ""})
    )
