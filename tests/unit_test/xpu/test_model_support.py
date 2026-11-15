# SPDX-License-Identifier: Apache-2.0
"""Per-model contract for every family Intel XPU serves (no weights, no accelerator).

The serving lane (tests/test_ci/xpu_model/) needs checkpoints and cards, so it
cannot gate a PR; this covers the CPU-side wiring that a PR can break.

Registry membership matters more than it looks: registry.py logs and swallows a
failed model import, so a broken config module does not raise -- the architecture
vanishes and first surfaces as ``KeyError`` for whoever names the checkpoint.

Device resolution is asserted declaratively here; proving a factory *resolves* an
unset device needs the factory to run, which is
tests/unit_test/test_stage_device_contract.py.
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.platforms import current_platform
from sglang_omni.utils.imports import import_string
from tests.test_ci.xpu_model.model_table import (
    KNOWN_FAMILIES,
    XPU_MODELS,
    XpuModel,
    device_family,
    model_named,
    xpu_pipelines,
)

_PIPELINES = xpu_pipelines()
_PIPELINE_IDS = [model.pipeline_key for model in _PIPELINES]


def config_cls(model: XpuModel) -> type:
    module = importlib.import_module(f"sglang_omni.models.{model.package}.config")
    if model.variant is None:
        return module.EntryClass
    variants = getattr(module, "Variants", {})
    assert model.variant in variants, (
        f"{model.package} exposes no {model.variant!r} variant; "
        f"Variants={sorted(variants)}"
    )
    return variants[model.variant]


@pytest.mark.parametrize("model", _PIPELINES, ids=_PIPELINE_IDS)
def test_the_config_module_imports_without_a_swallowed_error(model: XpuModel) -> None:
    """A direct import raises where the registry would only log and move on."""
    module = importlib.import_module(f"sglang_omni.models.{model.package}.config")
    assert hasattr(module, "EntryClass"), f"{model.package}.config has no EntryClass"


@pytest.mark.parametrize("model", _PIPELINES, ids=_PIPELINE_IDS)
def test_the_architecture_reaches_the_pipeline_config_registry(model: XpuModel) -> None:
    """An architecture missing here is a checkpoint no operator can serve."""
    registered = PIPELINE_CONFIG_REGISTRY.get_supported_archs()
    assert model.architecture in registered, (
        f"{model.architecture} is not registered, so a {model.package} checkpoint "
        "cannot be served. The registry swallows model import errors: import "
        f"sglang_omni.models.{model.package}.config directly to see the cause."
    )


@pytest.mark.parametrize("model", _PIPELINES, ids=_PIPELINE_IDS)
def test_the_pipeline_builds_the_stage_table_xpu_serves(model: XpuModel) -> None:
    """The topology is pinned, so a stage gained or lost surfaces here."""
    config = config_cls(model)(model_path="unused")
    assert tuple(stage.name for stage in config.stages) == model.stages
    if config.entry_stage is not None:
        assert config.entry_stage in model.stages


@pytest.mark.parametrize("model", _PIPELINES, ids=_PIPELINE_IDS)
def test_no_stage_pins_a_device_this_host_cannot_serve(model: XpuModel) -> None:
    """Unset is intended -- the worker resolves it from the platform -- so the
    only other thing a stage may say is this platform's own device."""
    config = config_cls(model)(model_path="unused")
    live = current_platform.device_type
    for stage in config.stages:
        device = getattr(stage.factory, "device", None)
        if device is None:
            continue
        assert str(device).split(":")[0] == live, (
            f"{model.package} stage {stage.name!r} pins device={device!r}, which "
            f"this host ({live}) cannot serve"
        )


@pytest.mark.parametrize("model", _PIPELINES, ids=_PIPELINE_IDS)
def test_every_stage_factory_resolves_to_a_callable(model: XpuModel) -> None:
    """A renamed or moved factory fails at stage launch, minutes into a load."""
    config = config_cls(model)(model_path="unused")
    for stage in config.stages:
        assert stage.factory_path, f"stage {stage.name!r} declares no factory_path"
        factory = import_string(stage.factory_path)
        assert callable(factory), f"{stage.factory_path} is not callable"


@pytest.mark.parametrize("model", _PIPELINES, ids=_PIPELINE_IDS)
def test_a_checkpoint_resolves_to_the_family_that_serves_it(
    model: XpuModel, tmp_path: Path
) -> None:
    """``--model-path`` finds the family from checkpoint metadata alone."""
    from sglang_omni.config.manager import resolve_config_cls_for_model_path

    checkpoint = tmp_path / model.mirror_dirname
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"architectures": [model.architecture]})
    )

    resolved = resolve_config_cls_for_model_path(str(checkpoint))

    module = importlib.import_module(f"sglang_omni.models.{model.package}.config")
    assert resolved is module.EntryClass


def test_every_model_in_the_table_is_exercised_by_the_serving_lane() -> None:
    """The serving lane dispatches on ``kind`` and would raise KeyError on an
    unknown one, but only on a host with weights in an opt-in lane."""
    from tests.test_ci.xpu_model.exercises import EXERCISES

    uncovered = sorted(
        f"{model.key} (kind={model.kind})"
        for model in XPU_MODELS
        if model.kind not in EXERCISES
    )
    assert not uncovered, f"tests/test_ci/xpu_model has no exercise for: {uncovered}"


def test_no_exercise_is_dead_code() -> None:
    """An exercise no row claims is either a dropped model or a typo'd kind, and
    both read as coverage that is not there."""
    from tests.test_ci.xpu_model.exercises import EXERCISES

    claimed = {model.kind for model in XPU_MODELS}
    assert not sorted(
        set(EXERCISES) - claimed
    ), f"no row uses: {sorted(set(EXERCISES) - claimed)}"


def test_each_row_is_a_distinct_checkpoint() -> None:
    """Two rows for one checkpoint would serve it twice and gate it once."""
    keys = [model.key for model in XPU_MODELS]
    assert len(keys) == len(set(keys)), "duplicate key in XPU_MODELS"
    envs = [model.checkpoint_env for model in XPU_MODELS]
    assert len(envs) == len(set(envs)), "two rows derive the same checkpoint_env"


@pytest.mark.parametrize("model", XPU_MODELS, ids=[m.key for m in XPU_MODELS])
def test_a_stage_scoped_serve_argument_names_a_stage_this_row_has(
    model: XpuModel,
) -> None:
    """``--thinker.gpu`` against a pipeline with no thinker is rejected only when
    the engine is built, minutes into a load, and the row would have been wrong
    the whole time."""
    names = set(model.stages)
    arguments = model.serve_args
    for family in KNOWN_FAMILIES:
        arguments += model.serve_args_on(family)
    for argument in arguments:
        if not argument.startswith("--") or "." not in argument:
            continue
        else:
            stage = argument[2:].split(".", 1)[0]
        if stage in names or "-" in stage:
            continue
        else:
            pytest.fail(
                f"{model.key} passes {argument!r}, but its stages are "
                f"{sorted(names)}"
            )


@pytest.mark.parametrize("model", XPU_MODELS, ids=[m.key for m in XPU_MODELS])
def test_a_row_the_device_lane_skips_says_why(model: XpuModel) -> None:
    """The reason is the whole value of the row: a family this image cannot
    serve still belongs in the table, but silently skipping it looks the same as
    passing."""
    if model.serving:
        assert len(model.serving) > 20, (
            f"{model.key} declines to serve with an unhelpfully short reason: "
            f"{model.serving!r}"
        )
    else:
        assert model.kind, f"{model.key} claims to be servable with no kind"


def test_the_code2wav_stage_takes_the_graph_policy_from_the_platform() -> None:
    """The policy has to travel to the stage for either answer to mean anything."""
    speech = config_cls(model_named("omni-speech"))(model_path="unused")
    kwargs = speech.stage_factory_kwargs("code2wav")
    assert kwargs["enable_cuda_graph"] is current_platform.enable_code2wav_graph()


def test_a_d270_card_gets_the_d270_recipe_and_an_unknown_card_the_base_one() -> None:
    """The recipe follows the hardware, so a misread name serves D270 with a capture
    that loses the device, or a B60 without a graph it records fine."""
    assert device_family("Intel(R) Graphics [0x674c]") == "d270"
    assert device_family("Intel(R) Arc(TM) Pro B60 Graphics") == "bmg"
    assert device_family("Intel(R) Data Center GPU Max 1550") == "unknown"
    assert device_family("Intel(R) Arc(TM) Pro B60 Graphics", "d270") == "d270"


@pytest.mark.parametrize("model", XPU_MODELS, ids=[m.key for m in XPU_MODELS])
def test_device_args_name_a_family_the_lane_can_detect(model: XpuModel) -> None:
    """A misspelt family is a workaround that never applies, and the first sign is
    the device it was meant to protect going down."""
    for family, args in model.device_args:
        assert (
            family in KNOWN_FAMILIES
        ), f"{model.key} sets device_args for {family!r}; known: {KNOWN_FAMILIES}"
        assert args, f"{model.key} sets an empty device_args entry for {family!r}"


# The perf baseline serves the same models from its own table. The two tables are
# the same fact written twice, so these tests make a drift between them fail here
# rather than as one lane quietly serving a recipe the other has fixed.
REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_SCRIPT = REPO_ROOT / "scripts" / "xpu" / "xpu_model_baseline.py"
BASELINE_DATA = REPO_ROOT / "scripts" / "xpu" / "xpu_model_baseline.json"
# This lane's key for a row the baseline names differently.
BASELINE_KEY = {"omni-text": "qwen3-omni-text"}


def baseline_rows() -> dict[str, dict]:
    """The baseline script's model table, read without running the script."""
    rows = {}
    for node in ast.walk(ast.parse(BASELINE_SCRIPT.read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "XpuModel":
            row = {}
            for keyword in node.keywords:
                if keyword.arg in ("key", "serve_args", "env", "unservable"):
                    row[keyword.arg] = ast.literal_eval(keyword.value)
                else:
                    continue
            rows[row["key"]] = row
        else:
            continue
    return rows


def shared_rows() -> list[tuple[XpuModel, dict]]:
    rows = baseline_rows()
    pairs = []
    for model in XPU_MODELS:
        key = BASELINE_KEY.get(model.key, model.key)
        if key in rows:
            pairs.append((model, rows[key]))
        else:
            continue
    return pairs


def test_every_model_the_baseline_measures_is_a_row_here() -> None:
    """A model the baseline gates on speed but this lane never serves loses its
    functional check the day it regresses."""
    rows = baseline_rows()
    here = {BASELINE_KEY.get(model.key, model.key) for model in XPU_MODELS}
    assert not sorted(set(rows) - here), f"missing rows: {sorted(set(rows) - here)}"


@pytest.mark.parametrize(
    "model,row", shared_rows(), ids=[model.key for model, _ in shared_rows()]
)
def test_this_lane_and_the_baseline_serve_a_model_the_same_way(
    model: XpuModel, row: dict
) -> None:
    """The base recipe and environment, which every host serves with. A model
    the baseline cannot serve anywhere must be skipped here too; the reverse does
    not hold, since this lane also skips a checkpoint too large for a runner."""
    assert tuple(row.get("serve_args", ())) == model.serve_args
    assert dict(row.get("env", {})) == dict(model.env)
    if row.get("unservable"):
        assert model.serving, f"the baseline cannot serve {model.key}, this lane tries"
    else:
        pass


# What D270 serves on top of each recipe: one flag per capture that loses the device
# there rather than falling back to eager.
D270_FLAGS = {
    "qwen3-tts": ["--tts_engine.engine.disable_cuda_graph", "true"],
    "auk": ["--auk_engine.factory.enable_dit_cuda_graph", "false"],
    "auk-flash": ["--auk_engine.factory.enable_dit_cuda_graph", "false"],
}


def test_the_d270_recipe_turns_off_exactly_the_captures_that_lose_the_device() -> None:
    """Each flag gives up a graph, so one added without a failing capture slows D270
    for nothing and one dropped takes the card down. The flags follow request shape:
    Qwen3-TTS CustomVoice and VoiceDesign never send ref_audio, so they keep theirs."""
    added = {
        model.key: list(model.serve_args_on("d270")[len(model.serve_args) :])
        for model in XPU_MODELS
        if model.serve_args_on("d270") != model.serve_args
    }
    assert added == D270_FLAGS


def test_the_committed_baseline_holds_b60_environments_only() -> None:
    """D270 numbers are recorded to a baseline outside the tree (--baseline PATH)
    and are not published in this file."""
    with BASELINE_DATA.open() as handle:
        environments = json.load(handle)["environments"].values()
    hosts = {
        environment["fingerprint"].get("host"): device_family(
            environment["fingerprint"].get("device", "")
        )
        for environment in environments
    }
    assert hosts, "the committed baseline holds no environment"
    assert set(hosts.values()) == {"bmg"}, f"non-B60 environments: {hosts}"


def test_only_a_complete_checkpoint_is_handed_to_the_server(tmp_path: Path) -> None:
    """Each case is a directory a host really held: the server was given it and
    failed minutes later naming a path that exists."""
    from tests.test_ci.xpu_model.conftest import holds_a_checkpoint

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
