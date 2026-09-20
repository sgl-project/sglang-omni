# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the resolver: application, provenance, and diffing.

The resolver is the only code that writes into a configuration; these are
the hand-written cases worth reading. Dotted-key canonicalization (the CLI's
implied ``stages.`` prefix and the one legacy spelling that still parses)
is covered at the bottom.
"""

from __future__ import annotations

import pytest

from sglang_omni.config.compat import canonicalize_dotted_key
from sglang_omni.config.patch import (
    ConfigPatch,
    ConfigPatchSet,
    ConfigSource,
    SourceKind,
)
from sglang_omni.config.path import ConfigPathError
from sglang_omni.config.resolver import ConfigResolver, diff_configs
from sglang_omni.config.schema import PipelineConfig

MEM_FRACTION = "stages.thinker.engine.mem_fraction_static"

YAML = ConfigSource(SourceKind.YAML_FILE, "configs/omni.yaml")
CLI = ConfigSource(SourceKind.CLI_DOTTED, "command line")
MODEL = ConfigSource(SourceKind.MODEL_DEFAULT, "FakeOmniConfig")


def patchset(*patches: ConfigPatch) -> ConfigPatchSet:
    return ConfigPatchSet(list(patches))


def make(config, path, value, source=CLI, **kwargs):
    return ConfigPatch.create(path, value, source, root=type(config), **kwargs)


class TestResolve:
    def test_applies_a_leaf_patch(self, pipeline_config: PipelineConfig):
        resolved = ConfigResolver(pipeline_config).resolve(
            patchset(make(pipeline_config, MEM_FRACTION, "0.8"))
        )
        assert resolved.value(MEM_FRACTION) == 0.8

    def test_baseline_is_untouched(self, pipeline_config: PipelineConfig):
        ConfigResolver(pipeline_config).resolve(
            patchset(make(pipeline_config, MEM_FRACTION, "0.8"))
        )
        assert pipeline_config.stages[1].engine.mem_fraction_static is None

    def test_result_is_validated(self, pipeline_config: PipelineConfig):
        """mem_fraction_static must stay in (0, 1); the resolver rebuilds the
        model rather than assigning onto it, so validators still fire."""
        with pytest.raises(ValueError, match="mem_fraction_static"):
            ConfigResolver(pipeline_config).resolve(
                patchset(make(pipeline_config, MEM_FRACTION, "1.5"))
            )

    def test_container_patch_deep_merges(self, pipeline_config: PipelineConfig):
        env = "stages.preprocessing.env"
        resolved = ConfigResolver(pipeline_config).resolve(
            patchset(
                make(pipeline_config, f"{env}.OMP_NUM_THREADS", "8"),
                make(pipeline_config, env, {"EXTRA_FLAG": "1"}, YAML),
            )
        )
        # The container came from a weaker layer, so the leaf patch survives it.
        assert resolved.value(f"{env}.OMP_NUM_THREADS") == "8"
        assert resolved.value(f"{env}.EXTRA_FLAG") == "1"

    def test_conflicting_patches_are_refused(self, pipeline_config: PipelineConfig):
        with pytest.raises(ValueError, match="same precedence"):
            ConfigResolver(pipeline_config).resolve(
                patchset(
                    make(pipeline_config, MEM_FRACTION, 0.5, YAML),
                    make(pipeline_config, MEM_FRACTION, 0.9, YAML),
                )
            )

    def test_tp_size_text_is_coerced_and_applied(self, pipeline_config: PipelineConfig):
        resolved = ConfigResolver(pipeline_config).resolve(
            patchset(
                make(pipeline_config, "stages.thinker.tp_size", "4"),
                make(pipeline_config, "stages.thinker.gpu", "[0, 1, 2, 3]"),
            )
        )
        assert resolved.config.stages[1].tp_size == 4


class TestProvenance:
    def test_explains_the_full_chain(self, pipeline_config: PipelineConfig):
        resolved = ConfigResolver(pipeline_config).resolve(
            patchset(
                make(pipeline_config, MEM_FRACTION, 0.5, MODEL),
                make(pipeline_config, MEM_FRACTION, 0.7, YAML),
                make(pipeline_config, MEM_FRACTION, 0.9, CLI),
            )
        )
        text = resolved.provenance.explain(MEM_FRACTION)
        assert "0.9" in text and "[winner]" in text
        assert text.count("[superseded]") == 2
        assert "configs/omni.yaml" in text

    def test_records_the_pre_patch_value(self, pipeline_config: PipelineConfig):
        resolved = ConfigResolver(pipeline_config).resolve(
            patchset(
                make(pipeline_config, "stages.thinker.tp_size", 4),
                make(pipeline_config, "stages.thinker.gpu", [0, 1, 2, 3]),
            )
        )
        assert resolved.provenance.baseline["stages.thinker.tp_size"] == 1

    def test_untouched_paths_are_not_explained(self, pipeline_config: PipelineConfig):
        """`config explain` asks touched() first and reports the default itself."""
        resolved = ConfigResolver(pipeline_config).resolve(
            patchset(
                make(pipeline_config, "stages.thinker.tp_size", 4),
                make(pipeline_config, "stages.thinker.gpu", [0, 1, 2, 3]),
            )
        )
        assert not resolved.provenance.touched("stages.thinker.process")
        assert resolved.provenance.winner("stages.thinker.process") is None

    def test_entries_carry_the_contributing_patch(
        self, pipeline_config: PipelineConfig
    ):
        """An entry is the patch plus the verdict, not a copy of its fields."""
        patch = make(pipeline_config, MEM_FRACTION, 0.9)
        resolved = ConfigResolver(pipeline_config).resolve(patchset(patch))
        winner = resolved.provenance.winner(MEM_FRACTION)
        assert winner is not None
        assert winner.patch is patch
        assert winner.winning is True
        assert winner.value == 0.9
        assert winner.source is CLI

    def test_no_rewrite_note_when_the_winning_value_sticks(
        self, pipeline_config: PipelineConfig
    ):
        resolved = ConfigResolver(pipeline_config).resolve(
            patchset(make(pipeline_config, MEM_FRACTION, "0.8"))
        )
        text = resolved.provenance.explain(MEM_FRACTION)
        assert text.splitlines()[0] == f"{MEM_FRACTION} = 0.8"
        assert "(resolved to" not in text


class TestDiff:
    def test_identical_configs_have_no_differences(
        self, pipeline_config: PipelineConfig
    ):
        assert diff_configs(pipeline_config, pipeline_config) == []

    def test_difference_is_reported_by_stage_name(
        self, pipeline_config: PipelineConfig
    ):
        changed = ConfigResolver(pipeline_config).resolve(
            patchset(make(pipeline_config, "stages.thinker.factory.max_seq_len", 99))
        )
        differences = diff_configs(pipeline_config, changed.config)
        assert [d.path for d in differences] == ["stages.thinker.factory.max_seq_len"]
        assert differences[0].expected == 8192
        assert differences[0].actual == 99


class TestDottedKeyCanonicalization:
    def test_a_stage_name_head_gets_the_stages_prefix(
        self, pipeline_config: PipelineConfig
    ):
        assert (
            canonicalize_dotted_key("thinker.tp_size", pipeline_config)
            == "stages.thinker.tp_size"
        )

    def test_a_pasted_canonical_path_is_accepted(self, pipeline_config: PipelineConfig):
        assert (
            canonicalize_dotted_key("stages.thinker.tp_size", pipeline_config)
            == "stages.thinker.tp_size"
        )

    def test_a_top_level_field_passes_through(self, pipeline_config: PipelineConfig):
        assert canonicalize_dotted_key("model_path", pipeline_config) == "model_path"

    def test_an_unknown_head_is_refused_with_candidates(
        self, pipeline_config: PipelineConfig
    ):
        with pytest.raises(ConfigPathError) as excinfo:
            canonicalize_dotted_key("thinkr.tp_size", pipeline_config)
        assert "thinker" in str(excinfo.value)
