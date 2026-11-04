# SPDX-License-Identifier: Apache-2.0
"""Every draft the docs, example configs and model defaults serve is pinned in the registry."""

from __future__ import annotations

import importlib
import re
import shlex
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.schema import StageConfig
from sglang_omni.models.registry import resolve_variant
from sglang_omni.speculative.draft_registry import DRAFT_REGISTRY, PinnedDraft

REPO_ROOT = Path(__file__).resolve().parents[3]
BASH_BLOCK = re.compile(r"```(?:bash|sh|shell)\n(.*?)```", re.DOTALL)
ENGINE_DRAFT_FLAG = re.compile(r"--(\w+)\.engine\.(speculative_\w+)")


@dataclass(frozen=True, kw_only=True)
class ServedDraft:
    source: str
    model_path: str | None
    stage: str
    engine_keys: dict[str, str]


def registered_keys(draft: PinnedDraft) -> dict[str, str]:
    return {key: str(value) for key, value in draft.engine_overrides().items()}


def stage_draft_keys(stage: StageConfig) -> dict[str, str]:
    if stage.engine is None:
        return {}
    else:
        return {
            key: str(value)
            for key, value in stage.engine.overrides().items()
            if key.startswith("speculative_")
        }


def drafts_in_docs() -> list[ServedDraft]:
    served = []
    for doc_path in sorted((REPO_ROOT / "docs").rglob("*.md")):
        for block in BASH_BLOCK.findall(doc_path.read_text(encoding="utf-8")):
            for command in block.replace("\\\n", " ").splitlines():
                if ".engine.speculative_" in command:
                    tokens = [
                        part
                        for token in shlex.split(command, comments=True)
                        for part in token.split("=", 1)
                    ]
                    keys_by_stage: dict[str, dict[str, str]] = {}
                    for flag, value in zip(tokens, tokens[1:]):
                        match = ENGINE_DRAFT_FLAG.fullmatch(flag)
                        if match is not None:
                            keys_by_stage.setdefault(match[1], {})[match[2]] = value
                        else:
                            pass
                    model_path = dict(zip(tokens, tokens[1:])).get("--model-path")
                    served.extend(
                        ServedDraft(
                            source=str(doc_path.relative_to(REPO_ROOT)),
                            model_path=model_path,
                            stage=stage,
                            engine_keys=engine_keys,
                        )
                        for stage, engine_keys in keys_by_stage.items()
                    )
                else:
                    pass
    return served


def drafts_in_example_configs() -> list[ServedDraft]:
    served = []
    for config_path in sorted((REPO_ROOT / "examples").rglob("*.yaml")):
        if "speculative_" in config_path.read_text(encoding="utf-8"):
            config = ConfigManager.from_file(str(config_path)).config
            for stage in config.stages:
                engine_keys = stage_draft_keys(stage)
                if engine_keys:
                    served.append(
                        ServedDraft(
                            source=str(config_path.relative_to(REPO_ROOT)),
                            model_path=config.model_path,
                            stage=stage.name,
                            engine_keys=engine_keys,
                        )
                    )
                else:
                    pass
        else:
            pass
    return served


SERVED_DRAFTS = drafts_in_docs() + drafts_in_example_configs()


def test_registry_pins_each_draft_and_turns_on_at_most_one_per_stage() -> None:
    for draft in DRAFT_REGISTRY:
        assert re.fullmatch(r"[0-9a-f]{40}", draft.draft_model_revision), draft
        assert all(key.startswith("speculative_") for key in draft.engine_settings)
    assert len(
        {(draft.model_path, draft.stage, draft.algorithm) for draft in DRAFT_REGISTRY}
    ) == len(DRAFT_REGISTRY)
    default_on = Counter(
        (draft.model_path, draft.stage)
        for draft in DRAFT_REGISTRY
        if draft.is_default_on
    )
    assert all(count == 1 for count in default_on.values()), default_on


@pytest.mark.parametrize(
    "served",
    SERVED_DRAFTS,
    ids=lambda served: f"{served.source}:{served.stage}:{served.engine_keys.get('speculative_algorithm')}",
)
def test_served_draft_gives_its_registry_entry_keys(served: ServedDraft) -> None:
    registered = {
        (draft.model_path, draft.stage, draft.algorithm): draft
        for draft in DRAFT_REGISTRY
    }
    draft = registered.get(
        (
            served.model_path,
            served.stage,
            served.engine_keys.get("speculative_algorithm"),
        )
    )
    assert draft is not None, (
        f"{served.source} serves a draft for {served.model_path} {served.stage} that "
        "sglang_omni/speculative/draft_registry.py does not pin"
    )
    assert served.engine_keys == registered_keys(draft)


def test_every_draft_off_by_default_is_served_in_a_doc_or_example_config() -> None:
    served_drafts = {
        (
            served.model_path,
            served.stage,
            served.engine_keys.get("speculative_algorithm"),
        )
        for served in SERVED_DRAFTS
    }
    assert {
        (draft.model_path, draft.stage, draft.algorithm)
        for draft in DRAFT_REGISTRY
        if not draft.is_default_on
    } <= served_drafts


def test_model_defaults_serve_exactly_the_default_on_drafts() -> None:
    default_drafts = set()
    for config_module_path in sorted(
        (REPO_ROOT / "sglang_omni" / "models").glob("*/config.py")
    ):
        module = importlib.import_module(
            f"sglang_omni.models.{config_module_path.parent.name}.config"
        )
        for variant in (module.EntryClass, *vars(module).get("Variants", {}).values()):
            config_cls = resolve_variant(variant)
            # note (Haoling Pu): a config whose stages come only from a profile file has no defaults.
            if config_cls.model_fields["stages"].is_required():
                pass
            else:
                for stage in config_cls(model_path="unused").stages:
                    engine_keys = stage_draft_keys(stage)
                    if engine_keys:
                        default_drafts.add((stage.name, frozenset(engine_keys.items())))
                    else:
                        pass
    assert default_drafts == {
        (draft.stage, frozenset(registered_keys(draft).items()))
        for draft in DRAFT_REGISTRY
        if draft.is_default_on
    }
