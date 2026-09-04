# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the canonical write surfaces.

There are exactly two ways to set a per-stage value, and they are the same
path written twice: the YAML ``stages:`` mapping starts from the stage name,
and a dotted CLI flag starts from the stage name too, with the ``stages.``
prefix implied. What these tests pin down:

* both spellings resolve through the same patch machinery -- same paths, same
  conflict rules, same provenance;
* the CLI coerces text by the declared type while YAML values keep their
  native types untouched;
* the ``engine``/``factory`` groups accept keys beyond their
  declared fields and pass them through untouched -- whether a key exists is
  the consuming module's call, not the parser's;
* writing one path twice at the same precedence stays an error rather than
  quietly becoming last-one-wins.
"""

from __future__ import annotations

import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.patch import (
    ConfigPatchSet,
    DuplicatePatchError,
    Layer,
    Specificity,
)
from sglang_omni.config.path import ConfigPathError
from sglang_omni.config.resolver import ConfigResolver
from sglang_omni.config.schema import PipelineConfig
from sglang_omni.config.sources import (
    patches_from_dotted_cli,
    patches_from_model_path_flag,
    patches_from_shared_block,
    patches_from_stages_mapping,
)

MAX_SEQ_LEN = "stages.thinker.factory.max_seq_len"


def resolve(config: PipelineConfig, *patchsets) -> PipelineConfig:
    """Apply several patch sets the way a single entry point would."""
    merged = patchsets[0]
    for patchset in patchsets[1:]:
        merged = merged.merge(patchset)
    return ConfigResolver(config).resolve(merged).config


def stage(config: PipelineConfig, name: str):
    return config.stage_named(name)


class TestDottedCli:
    def test_text_is_coerced_to_the_declared_type(self, pipeline_config):
        patches = patches_from_dotted_cli(
            [("thinker.tp_size", "2"), ("thinker.gpu", "[0, 1]")], pipeline_config
        )
        resolved = resolve(pipeline_config, patches)
        assert stage(resolved, "thinker").tp_size == 2

    def test_the_stage_name_head_gets_the_implied_prefix(self, pipeline_config):
        patches = patches_from_dotted_cli(
            [("thinker.factory.max_seq_len", "4096")], pipeline_config
        )
        (patch,) = list(patches)
        assert patch.path.raw == MAX_SEQ_LEN
        assert patch.layer is Layer.CLI
        assert patch.specificity is Specificity.EXPLICIT

    def test_the_explicit_stages_prefix_is_refused(self, pipeline_config):
        with pytest.raises(ConfigPathError, match="implied"):
            patches_from_dotted_cli([("stages.thinker.tp_size", "2")], pipeline_config)

    def test_a_whole_stage_flag_is_refused(self, pipeline_config):
        with pytest.raises(ConfigPathError, match="one field below"):
            patches_from_dotted_cli([("thinker", "2")], pipeline_config)

    def test_an_unknown_head_suggests_stages_and_top_level_fields(
        self, pipeline_config
    ):
        with pytest.raises(ConfigPathError) as excinfo:
            patches_from_dotted_cli([("thinkr.tp_size", "2")], pipeline_config)
        assert "thinker" in str(excinfo.value)

    def test_an_unknown_field_inside_a_stage_gets_suggestions(self, pipeline_config):
        with pytest.raises(ConfigPathError, match="tp_size"):
            patches_from_dotted_cli([("thinker.tp_siz", "2")], pipeline_config)

    def test_the_none_sentinel_clears_a_field(self, pipeline_config):
        patches = patches_from_dotted_cli(
            [("thinker.factory.max_seq_len", "none")], pipeline_config
        )
        resolved = resolve(pipeline_config, patches)
        assert stage(resolved, "thinker").factory.max_seq_len is None

    def test_a_top_level_field_passes_through(self, pipeline_config):
        patches = patches_from_dotted_cli(
            [("model_path", "/models/other")], pipeline_config
        )
        resolved = resolve(pipeline_config, patches)
        assert resolved.model_path == "/models/other"

    def test_engine_on_a_non_engine_stage_is_refused(self, pipeline_config):
        with pytest.raises(ConfigPathError, match="not an engine stage"):
            patches_from_dotted_cli(
                [("preprocessing.engine.mem_fraction_static", "0.5")],
                pipeline_config,
            )

    def test_a_group_path_is_refused_with_leaf_guidance(self, pipeline_config):
        """The CLI writes one leaf at a time; a whole group as one flag value
        would replace the container and reset siblings other sources set."""
        with pytest.raises(ConfigPathError, match="one field below"):
            patches_from_dotted_cli(
                [("thinker.factory", '{"device": "cpu"}')], pipeline_config
            )

    def test_a_group_path_is_refused_regardless_of_the_value(self, pipeline_config):
        with pytest.raises(ConfigPathError, match="one field below"):
            patches_from_dotted_cli(
                [("thinker.engine", "not-a-mapping")], pipeline_config
            )


class TestGroupPassThrough:
    """Keys the groups do not declare travel to the consumer untouched."""

    def test_a_free_form_engine_key_reaches_the_overrides(self, pipeline_config):
        patches = patches_from_dotted_cli(
            [("thinker.engine.disable_radix_cache", "true")], pipeline_config
        )
        resolved = resolve(pipeline_config, patches)
        assert (
            stage(resolved, "thinker").engine.overrides()["disable_radix_cache"] is True
        )

    def test_free_form_scheduler_and_model_keys_are_kept(self, pipeline_config):
        patches = patches_from_dotted_cli(
            [("thinker.factory.made_up_knob", "7"), ("thinker.factory.lookahead", "9")],
            pipeline_config,
        )
        resolved = resolve(pipeline_config, patches)
        thinker = stage(resolved, "thinker")
        assert thinker.factory.model_extra["made_up_knob"] == 7
        assert thinker.factory.model_extra["lookahead"] == 9

    def test_a_declared_key_still_validates_eagerly(self, pipeline_config):
        patches = patches_from_dotted_cli(
            [("thinker.factory.max_concurrency", "0")], pipeline_config
        )
        with pytest.raises(Exception, match="max_concurrency"):
            resolve(pipeline_config, patches)


class TestStagesMapping:
    def test_a_known_name_merges_as_leaf_patches(self, pipeline_config):
        patches = patches_from_stages_mapping(
            {"thinker": {"tp_size": 2, "engine": {"mem_fraction_static": 0.6}}},
            type(pipeline_config),
            ["preprocessing", "thinker"],
        )
        paths = {patch.path.raw for patch in patches}
        assert paths == {
            "stages.thinker.tp_size",
            "stages.thinker.engine.mem_fraction_static",
        }
        assert all(patch.layer is Layer.USER_FILE for patch in patches)

    def test_yaml_values_keep_their_native_types(self, pipeline_config):
        patches = patches_from_stages_mapping(
            {"thinker": {"engine": {"mem_fraction_static": 0.6}}},
            type(pipeline_config),
            ["thinker"],
        )
        (patch,) = list(patches)
        assert patch.value == 0.6 and isinstance(patch.value, float)

    def test_the_mapping_resolves_onto_the_config(self, pipeline_config):
        patches = patches_from_stages_mapping(
            {
                "thinker": {
                    "engine": {"disable_radix_cache": True},
                    "factory": {"lookahead": 9},
                }
            },
            type(pipeline_config),
            ["thinker"],
        )
        resolved = resolve(pipeline_config, patches)
        thinker = stage(resolved, "thinker")
        assert thinker.engine.overrides()["disable_radix_cache"] is True
        assert thinker.factory.model_extra["lookahead"] == 9

    def test_an_unknown_name_is_refused_with_the_real_names(self, pipeline_config):
        """Stage topology lives in the model's config class; a config file
        cannot introduce stages."""
        with pytest.raises(ValueError, match="thinker"):
            patches_from_stages_mapping(
                {"decode": {"factory_path": "x:y", "terminal": True}},
                type(pipeline_config),
                ["preprocessing", "thinker"],
            )

    def test_name_inside_a_body_is_refused(self, pipeline_config):
        with pytest.raises(ValueError, match="mapping key is the stage's name"):
            patches_from_stages_mapping(
                {"thinker": {"name": "thinker"}},
                type(pipeline_config),
                ["thinker"],
            )

    def test_the_list_form_is_refused_with_guidance(self, pipeline_config):
        with pytest.raises(ValueError, match="mapping"):
            patches_from_stages_mapping(
                [{"name": "thinker"}], type(pipeline_config), ["thinker"]
            )

    def test_a_non_mapping_stage_body_is_refused(self, pipeline_config):
        with pytest.raises(ValueError, match="must be a mapping"):
            patches_from_stages_mapping(
                {"thinker": 4}, type(pipeline_config), ["thinker"]
            )


class TestStageDefaults:
    """``shared`` fans one body out to the selected stages."""

    def test_the_selection_expands_to_one_patch_per_stage(self, pipeline_config):
        patches = patches_from_shared_block(
            [
                {
                    "select": {"stages": ["preprocessing", "thinker"]},
                    "factory": {"max_seq_len": 4096},
                }
            ],
            type(pipeline_config),
            ["preprocessing", "thinker"],
        )
        assert {patch.path.raw for patch in patches.ordered()} == {
            "stages.preprocessing.factory.max_seq_len",
            "stages.thinker.factory.max_seq_len",
        }
        assert all(patch.specificity is Specificity.ROLE for patch in patches.ordered())

    def test_an_explicit_stage_entry_overrides_the_selection(self, pipeline_config):
        defaults = patches_from_shared_block(
            [
                {
                    "select": {"stages": ["preprocessing", "thinker"]},
                    "factory": {"max_seq_len": 4096},
                }
            ],
            type(pipeline_config),
            ["preprocessing", "thinker"],
        )
        explicit = patches_from_stages_mapping(
            {"thinker": {"factory": {"max_seq_len": 8192}}},
            type(pipeline_config),
            ["preprocessing", "thinker"],
        )
        resolved = resolve(pipeline_config, defaults, explicit)
        assert stage(resolved, "thinker").factory.max_seq_len == 8192
        assert stage(resolved, "preprocessing").factory.max_seq_len == 4096

    def test_the_engine_selector_matches_engine_stages_only(self, pipeline_config):
        patches = patches_from_shared_block(
            [{"select": {"engine": True}, "engine": {"mem_fraction_static": 0.5}}],
            type(pipeline_config),
            ["preprocessing", "thinker"],
        )
        assert [patch.path.raw for patch in patches.ordered()] == [
            "stages.thinker.engine.mem_fraction_static"
        ]

    def test_exclude_removes_matched_stages(self, pipeline_config):
        patches = patches_from_shared_block(
            [
                {
                    "select": {
                        "stages": ["preprocessing", "thinker"],
                        "exclude": ["thinker"],
                    },
                    "factory": {"max_seq_len": 4096},
                }
            ],
            type(pipeline_config),
            ["preprocessing", "thinker"],
        )
        assert [patch.path.raw for patch in patches.ordered()] == [
            "stages.preprocessing.factory.max_seq_len"
        ]

    def test_two_entries_writing_one_leaf_conflict(self, pipeline_config):
        patches = patches_from_shared_block(
            [
                {"select": {"stages": ["thinker"]}, "factory": {"max_seq_len": 4096}},
                {
                    "select": {"stages": ["preprocessing", "thinker"]},
                    "factory": {"max_seq_len": 2048},
                },
            ],
            type(pipeline_config),
            ["preprocessing", "thinker"],
        )
        with pytest.raises(DuplicatePatchError):
            resolve(pipeline_config, patches)

    def test_an_unknown_stage_name_is_refused_with_the_real_names(
        self, pipeline_config
    ):
        with pytest.raises(ValueError, match="thinker"):
            patches_from_shared_block(
                [{"select": {"stages": ["thinkr"]}, "factory": {"max_seq_len": 1}}],
                type(pipeline_config),
                ["preprocessing", "thinker"],
            )

    def test_an_empty_selection_is_refused(self, pipeline_config):
        with pytest.raises(ValueError, match="matches no stage"):
            patches_from_shared_block(
                [
                    {
                        "select": {"engine": True, "exclude": ["thinker"]},
                        "engine": {"mem_fraction_static": 0.5},
                    }
                ],
                type(pipeline_config),
                ["preprocessing", "thinker"],
            )

    def test_a_bodyless_entry_is_refused(self, pipeline_config):
        with pytest.raises(ValueError, match="writes nothing"):
            patches_from_shared_block(
                [{"select": {"stages": ["thinker"]}}],
                type(pipeline_config),
                ["preprocessing", "thinker"],
            )

    def test_an_unknown_selector_key_is_refused(self, pipeline_config):
        with pytest.raises(ValueError, match="capability"):
            patches_from_shared_block(
                [
                    {
                        "select": {"capability": "sglang"},
                        "engine": {"mem_fraction_static": 0.5},
                    }
                ],
                type(pipeline_config),
                ["preprocessing", "thinker"],
            )


class TestPrecedence:
    def test_the_command_line_beats_the_file(self, pipeline_config):
        file_patches = patches_from_stages_mapping(
            {"thinker": {"tp_size": 2}}, type(pipeline_config), ["thinker"]
        )
        cli_patches = patches_from_dotted_cli(
            [("thinker.tp_size", "4"), ("thinker.gpu", "[0, 1, 2, 3]")],
            pipeline_config,
        )
        resolved = resolve(pipeline_config, file_patches, cli_patches)
        assert stage(resolved, "thinker").tp_size == 4

    def test_the_model_path_flag_beats_the_file_value(self, pipeline_config):
        flag = patches_from_model_path_flag("/models/flag", pipeline_config)
        resolved = resolve(pipeline_config, flag)
        assert resolved.model_path == "/models/flag"


class TestManager:
    def test_merge_config_applies_dotted_pairs(self, pipeline_config):
        manager = ConfigManager(pipeline_config)
        merged = manager.merge_config(
            [("thinker.tp_size", "2"), ("thinker.gpu", "[0, 1]")]
        )
        assert stage(merged, "thinker").tp_size == 2

    def test_a_flag_repeated_with_different_values_is_refused(self, pipeline_config):
        manager = ConfigManager(pipeline_config)
        with pytest.raises(DuplicatePatchError):
            manager.merge_config([("thinker.tp_size", "2"), ("thinker.tp_size", "4")])

    def test_a_flag_repeated_with_the_same_value_is_tolerated(self, pipeline_config):
        manager = ConfigManager(pipeline_config)
        merged = manager.merge_config(
            [
                ("thinker.tp_size", "2"),
                ("thinker.tp_size", "2"),
                ("thinker.gpu", "[0, 1]"),
            ]
        )
        assert stage(merged, "thinker").tp_size == 2

    def test_extra_patches_join_the_same_conflict_check(self, pipeline_config):
        manager = ConfigManager(pipeline_config)
        extra = ConfigPatchSet().merge(
            patches_from_model_path_flag("/models/flag", pipeline_config)
        )
        merged = manager.merge_config(
            [("thinker.tp_size", "2"), ("thinker.gpu", "[0, 1]")],
            extra_patches=extra,
        )
        assert merged.model_path == "/models/flag"
        assert stage(merged, "thinker").tp_size == 2


class TestPerStageTypedGroup:
    """A model may subclass a group to type one stage's free-form keys.

    The subclassed field becomes a typed path: static constraints enforce at
    resolution and the lossless conversion rule applies -- identical
    treatment to a field declared on the shared group, scoped to one stage."""

    @staticmethod
    def _pipeline_cls():
        from typing import ClassVar

        from pydantic import Field

        from sglang_omni.config.schema import FactoryArgs, StageConfig

        class VocoderFactoryArgs(FactoryArgs):
            stream_slots: int | None = Field(default=None, ge=1, le=64)

        class VocoderStageConfig(StageConfig):
            factory: VocoderFactoryArgs = Field(default_factory=VocoderFactoryArgs)

        class TypedGroupPipelineConfig(PipelineConfig):
            stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
                "vocoder": VocoderStageConfig,
            }
            stages: list[StageConfig] = Field(
                default_factory=lambda: [
                    VocoderStageConfig(
                        name="vocoder",
                        process="p",
                        factory_path="tests.fake:create_vocoder",
                        terminal=True,
                    )
                ]
            )

        return TypedGroupPipelineConfig

    def test_the_declared_range_is_enforced_at_resolution(self):
        cls = self._pipeline_cls()
        config = cls(model_path="dummy")
        with pytest.raises(ValueError, match="stream_slots"):
            ConfigManager(config).merge_config(
                [("vocoder.factory.stream_slots", "128")]
            )

    def test_the_lossless_conversion_rule_applies(self):
        cls = self._pipeline_cls()
        config = cls(model_path="dummy")
        with pytest.raises(ConfigPathError, match="got a boolean"):
            ConfigManager(config).merge_config(
                [("vocoder.factory.stream_slots", "true")]
            )

    def test_a_legal_value_lands_on_the_typed_field(self):
        cls = self._pipeline_cls()
        config = cls(model_path="dummy")
        merged = ConfigManager(config).merge_config(
            [("vocoder.factory.stream_slots", "8")]
        )
        assert merged.stage_named("vocoder").factory.stream_slots == 8
