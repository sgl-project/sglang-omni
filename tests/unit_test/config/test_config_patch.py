# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the normalized patch representation and its precedence."""

from __future__ import annotations

import pytest

from sglang_omni.config.patch import (
    ConfigPatch,
    ConfigPatchSet,
    ConfigSource,
    DuplicatePatchError,
    Layer,
    SourceKind,
    Specificity,
)
from sglang_omni.config.path import ConfigPathError

MEM_FRACTION = "stages.thinker.engine.mem_fraction_static"

YAML = ConfigSource(SourceKind.YAML_FILE, "configs/omni.yaml")
CLI = ConfigSource(SourceKind.CLI_DOTTED, "command line")
CLI_FLAG = ConfigSource(SourceKind.CLI_FLAG, "--mem-fraction-static")


@pytest.fixture
def patch(pipeline_config):
    root = type(pipeline_config)

    def _patch(path, value, source, **kwargs):
        return ConfigPatch.create(path, value, source, root=root, **kwargs)

    return _patch


class TestConstruction:
    def test_layer_defaults_to_the_source_kind(self, patch):
        assert patch(MEM_FRACTION, "0.8", YAML).layer is Layer.USER_FILE
        assert patch(MEM_FRACTION, "0.8", CLI).layer is Layer.CLI

    def test_value_is_coerced_to_the_declared_type(self, patch):
        assert patch("stages.thinker.tp_size", "4", CLI).value == 4
        assert patch(MEM_FRACTION, "0.8", CLI).value == 0.8

    def test_coercion_can_be_disabled_for_already_typed_sources(self, patch):
        assert patch("stages.thinker.tp_size", "4", YAML, coerce=False).value == "4"

    def test_unknown_path_fails_at_construction(self, patch):
        with pytest.raises(ConfigPathError):
            patch("stages.thinker.nope", 1, CLI)

    def test_a_free_form_group_key_is_not_an_unknown_path(self, patch):
        """Group vocabularies belong to their consumers, so construction
        accepts any key under them and coerces the text best-effort."""
        created = patch("stages.thinker.engine.disable_radix_cache", "true", CLI)
        assert created.value is True


class TestPrecedence:
    def test_cli_beats_yaml_regardless_of_insertion_order(self, patch):
        patchset = ConfigPatchSet()
        patchset.add(patch(MEM_FRACTION, 0.9, CLI))
        patchset.add(patch(MEM_FRACTION, 0.5, YAML))
        assert [p.value for p in patchset.ordered()] == [0.5, 0.9]
        assert patchset.winners()[MEM_FRACTION].value == 0.9

    def test_an_explicit_path_beats_a_broadcast_flag_inside_one_layer(self, patch):
        """``--mem-fraction-static`` fans out at BROADCAST specificity, so a
        dotted per-stage spelling must win without being a conflict."""
        patchset = ConfigPatchSet()
        patchset.add(
            patch(MEM_FRACTION, 0.3, CLI_FLAG, specificity=Specificity.BROADCAST)
        )
        patchset.add(patch(MEM_FRACTION, 0.6, CLI))
        assert patchset.winners()[MEM_FRACTION].value == 0.6
        patchset.require_no_conflicts()

    def test_container_patch_applies_before_a_nested_one(self, patch):
        env = "stages.preprocessing.env"
        patchset = ConfigPatchSet()
        patchset.add(patch(f"{env}.OMP_NUM_THREADS", "8", YAML))
        patchset.add(patch(env, {"EXTRA_FLAG": "1"}, YAML))
        assert [p.key for p in patchset.ordered()] == [
            env,
            f"{env}.OMP_NUM_THREADS",
        ]

    def test_insertion_order_is_the_last_tie_break(self, patch):
        patchset = ConfigPatchSet()
        patchset.add(patch("stages.thinker.tp_size", 2, YAML))
        patchset.add(patch("stages.preprocessing.tp_size", 4, YAML))
        assert [p.key for p in patchset.ordered()] == [
            "stages.thinker.tp_size",
            "stages.preprocessing.tp_size",
        ]


class TestConflicts:
    def test_same_rank_disagreement_is_a_conflict(self, patch):
        """A dotted flag and a typed flag writing one path sit at the same
        layer and specificity; nothing breaks the tie, so the set says so."""
        patchset = ConfigPatchSet()
        patchset.add(patch("model_path", "/models/a", CLI))
        patchset.add(
            patch(
                "model_path",
                "/models/b",
                ConfigSource(SourceKind.CLI_FLAG, "--model-path"),
            )
        )
        assert len(patchset.conflicts()) == 1
        with pytest.raises(DuplicatePatchError) as excinfo:
            patchset.require_no_conflicts()
        message = str(excinfo.value)
        assert "cli dotted" in message
        assert "cli flag" in message

    def test_agreeing_sources_are_not_a_conflict(self, patch):
        patchset = ConfigPatchSet()
        patchset.add(patch(MEM_FRACTION, 0.5, YAML))
        patchset.add(patch(MEM_FRACTION, 0.5, YAML))
        assert patchset.conflicts() == []
        patchset.require_no_conflicts()

    def test_different_layers_are_not_a_conflict(self, patch):
        patchset = ConfigPatchSet()
        patchset.add(patch(MEM_FRACTION, 0.5, YAML))
        patchset.add(patch(MEM_FRACTION, 0.9, CLI))
        patchset.require_no_conflicts()

    def test_different_paths_are_not_a_conflict(self, patch):
        patchset = ConfigPatchSet()
        patchset.add(patch("stages.thinker.tp_size", 2, YAML))
        patchset.add(patch("stages.preprocessing.tp_size", 4, YAML))
        patchset.require_no_conflicts()


class TestBookkeeping:
    def test_merge_preserves_both_sides(self, patch):
        first = ConfigPatchSet().add(patch("stages.thinker.tp_size", 2, YAML))
        second = ConfigPatchSet().add(patch("stages.thinker.tp_size", 4, CLI))
        merged = first.merge(second)
        assert len(merged) == 2
        assert len(first) == 1
        assert merged.winners()["stages.thinker.tp_size"].value == 4
