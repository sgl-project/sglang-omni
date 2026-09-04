# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the canonical configuration path compiler."""

from __future__ import annotations

import pytest

from sglang_omni.config.path import (
    ConfigPath,
    ConfigPathError,
    PathVisibility,
    iter_schema_paths,
)
from sglang_omni.config.schema import PipelineConfig

MEM_FRACTION = "stages.thinker.engine.mem_fraction_static"


class TestParsing:
    def test_typed_leaf_carries_its_declared_type(self, pipeline_config):
        path = ConfigPath.parse(
            "stages.thinker.factory.max_seq_len", type(pipeline_config)
        )
        assert path.parts == ("stages", "thinker", "factory", "max_seq_len")
        # The declared type plus its static Field constraints (gt=0), so
        # coerce enforces the same rule the rebuild does.
        import typing
        from typing import get_args, get_origin

        assert get_origin(path.value_type) is typing.Annotated
        assert get_args(path.value_type)[0] == (int | None)
        assert path.is_leaf
        assert path.stage_name == "thinker"

    def test_container_paths_are_not_leaves(self, pipeline_config):
        cls = type(pipeline_config)
        assert not ConfigPath.parse("stages.thinker.factory", cls).is_leaf
        assert not ConfigPath.parse("stages", cls).is_leaf
        assert not ConfigPath.parse("stages.thinker.env", cls).is_leaf

    def test_scalar_list_is_a_leaf_replaced_as_a_whole(self, pipeline_config):
        cls = type(pipeline_config)
        assert ConfigPath.parse("stages.thinker.stream_to", cls).is_leaf
        with pytest.raises(ConfigPathError, match="replaced as a whole"):
            ConfigPath.parse("stages.thinker.stream_to.0", cls)

    def test_free_mapping_accepts_any_key(self, pipeline_config):
        path = ConfigPath.parse(
            "stages.preprocessing.env.OMP_NUM_THREADS", type(pipeline_config)
        )
        assert path.value_type is str

    def test_the_engine_group_exists_only_on_engine_stages(self, pipeline_config):
        cls = type(pipeline_config)
        ConfigPath.parse(MEM_FRACTION, cls)
        with pytest.raises(ConfigPathError, match="not an engine stage"):
            ConfigPath.parse("stages.preprocessing.engine.mem_fraction_static", cls)

    def test_group_keys_beyond_the_declared_fields_parse_as_free_form(
        self, pipeline_config
    ):
        """The groups' vocabularies belong to their consumers; the parser
        accepts any key and leaves legality to the module that reads it."""
        cls = type(pipeline_config)
        for raw in (
            "stages.thinker.engine.disable_radix_cache",
            "stages.thinker.factory.made_up_knob",
            "stages.thinker.factory.lookahead",
        ):
            # Parses without a did-you-mean refusal; the value type is open
            # because only the consumer knows what the key means.
            assert ConfigPath.parse(raw, cls).value_type is not None


class TestErrors:
    def test_unknown_field_suggests_neighbours(self, pipeline_config):
        with pytest.raises(ConfigPathError) as excinfo:
            ConfigPath.parse("stages.thinker.tp_siz", type(pipeline_config))
        message = str(excinfo.value)
        assert "tp_size" in message
        assert "did you mean" in message

    def test_positional_stage_index_is_refused_with_a_hint(self, pipeline_config):
        with pytest.raises(ConfigPathError) as excinfo:
            ConfigPath.parse("stages.1.tp_size", type(pipeline_config))
        assert "addressed by name" in str(excinfo.value)

    def test_descending_below_a_scalar_is_refused(self, pipeline_config):
        with pytest.raises(ConfigPathError, match="leaf of type"):
            ConfigPath.parse("stages.thinker.tp_size.value", type(pipeline_config))

    def test_empty_and_malformed_paths(self, pipeline_config):
        with pytest.raises(ConfigPathError):
            ConfigPath.parse("", type(pipeline_config))
        with pytest.raises(ConfigPathError, match="empty segment"):
            ConfigPath.parse("stages..tp_size", type(pipeline_config))

    def test_unknown_top_level_field(self, pipeline_config):
        with pytest.raises(ConfigPathError) as excinfo:
            ConfigPath.parse("stagez", type(pipeline_config))
        assert "stages" in str(excinfo.value)


class TestVisibility:
    def test_the_engine_leaf_is_public(self, pipeline_config):
        path = ConfigPath.parse(MEM_FRACTION, type(pipeline_config))
        assert path.visibility is PathVisibility.PUBLIC
        path.require_writable()

    @pytest.mark.parametrize(
        "raw", ["stages", "entry_stage", "config_cls", "stages.thinker.name"]
    )
    def test_internal_paths_are_not_writable(self, pipeline_config, raw):
        """Stage topology and derived identity belong to the config class; no
        user-facing source may write them."""
        path = ConfigPath.parse(raw, type(pipeline_config))
        assert path.visibility is PathVisibility.INTERNAL
        assert not path.is_public()
        assert path.visibility_reason
        with pytest.raises(ConfigPathError, match="cannot be set"):
            path.require_writable()


class TestCoercion:
    @pytest.mark.parametrize(
        "raw,text,expected",
        [
            ("stages.thinker.tp_size", "4", 4),
            (MEM_FRACTION, "0.75", 0.75),
            ("stages.thinker.terminal", "true", True),
            ("stages.thinker.process", "gen", "gen"),
            ("stages.thinker.gpu", "0", 0),
            ("stages.thinker.gpu", "[0, 1]", [0, 1]),
            ("stages.thinker.stream_to", '["a", "b"]', ["a", "b"]),
            ("stages.thinker.factory.max_seq_len", "32768", 32768),
        ],
    )
    def test_typed_coercion(self, pipeline_config, raw, text, expected):
        assert ConfigPath.parse(raw, type(pipeline_config)).coerce(text) == expected

    def test_string_fields_are_not_guessed_into_numbers(self, pipeline_config):
        path = ConfigPath.parse(
            "stages.preprocessing.env.OMP_NUM_THREADS", type(pipeline_config)
        )
        assert path.coerce("4") == "4"

    def test_non_string_values_pass_through(self, pipeline_config):
        path = ConfigPath.parse("stages.thinker.tp_size", type(pipeline_config))
        assert path.coerce(4) == 4

    def test_free_form_group_keys_fall_back_to_scalar_parsing(self, pipeline_config):
        path = ConfigPath.parse(
            "stages.thinker.factory.made_up_knob", type(pipeline_config)
        )
        assert path.coerce("true") is True
        assert path.coerce("7") == 7
        assert path.coerce("bar") == "bar"


class TestReadWrite:
    def test_read_from_model_and_from_dump(self, pipeline_config: PipelineConfig):
        path = ConfigPath.parse("stages.thinker.tp_size", type(pipeline_config))
        assert path.read(pipeline_config) == 1
        assert path.read(pipeline_config.model_dump()) == 1

    def test_read_optional_none(self, pipeline_config: PipelineConfig):
        path = ConfigPath.parse(MEM_FRACTION, type(pipeline_config))
        assert path.read(pipeline_config) is None

    def test_write_round_trips_through_validation(
        self, pipeline_config: PipelineConfig
    ):
        cls = type(pipeline_config)
        data = pipeline_config.model_dump()
        ConfigPath.parse(MEM_FRACTION, cls).write(data, 0.8)
        ConfigPath.parse("stages.thinker.factory.max_seq_len", cls).write(data, 4096)
        rebuilt = cls(**data)
        thinker = rebuilt.stage_named("thinker")
        assert thinker.engine.mem_fraction_static == 0.8
        assert thinker.factory.max_seq_len == 4096

    def test_write_creates_missing_optional_container(
        self, pipeline_config: PipelineConfig
    ):
        cls = type(pipeline_config)
        data = pipeline_config.model_dump()
        assert data["stages"][1]["comm"] is None
        ConfigPath.parse("stages.thinker.comm.credits", cls).write(data, 8)
        rebuilt = cls(**data)
        assert rebuilt.stage_named("thinker").comm.credits == 8

    def test_write_new_mapping_key(self, pipeline_config: PipelineConfig):
        cls = type(pipeline_config)
        data = pipeline_config.model_dump()
        ConfigPath.parse("stages.thinker.env.CUDA_LAUNCH_BLOCKING", cls).write(
            data, "1"
        )
        rebuilt = cls(**data)
        assert rebuilt.stage_named("thinker").env["CUDA_LAUNCH_BLOCKING"] == "1"

    def test_unknown_stage_lists_the_real_names(self, pipeline_config: PipelineConfig):
        data = pipeline_config.model_dump()
        with pytest.raises(ConfigPathError) as excinfo:
            ConfigPath.parse("stages.talker.tp_size", type(pipeline_config)).write(
                data, 2
            )
        message = str(excinfo.value)
        assert "no entry named 'talker'" in message
        assert "thinker" in message

    def test_write_does_not_touch_siblings(self, pipeline_config: PipelineConfig):
        cls = type(pipeline_config)
        data = pipeline_config.model_dump()
        ConfigPath.parse("stages.thinker.factory.max_seq_len", cls).write(data, 128)
        assert data["stages"][0]["factory"].get("max_seq_len") is None
        assert data["stages"][1]["factory"]["max_concurrency"] == 4


class TestSchemaEnumeration:
    def test_public_enumeration_hides_internal_paths(self, pipeline_config):
        cls = type(pipeline_config)
        public = iter_schema_paths(cls)
        every = iter_schema_paths(cls, include_non_public=True)
        assert MEM_FRACTION.replace("thinker", "*") in public
        assert "config_cls" not in public
        assert "stages" not in public
        assert "config_cls" in every
        assert "stages" in every

    def test_every_enumerated_path_parses(self, pipeline_config):
        cls = type(pipeline_config)
        for candidate in iter_schema_paths(cls, include_non_public=True):
            ConfigPath.parse(candidate.replace("*", "thinker"), cls)


class TestLosslessNumericCoercion:
    """Conversions only go the lossless way, whatever the spelling.

    An int fits a float field and 0/1 fit a bool field; a bool never fits a
    numeric field (true would silently become 1) and a float never fits an
    int field (32.0 would be truncated into shape). CLI text is parsed to a
    scalar first, so both it and a native YAML scalar answer to the rule."""

    def _path(self, pipeline_config, text: str) -> ConfigPath:
        return ConfigPath.parse(text, type(pipeline_config))

    @pytest.mark.parametrize("value", ["true", True, "2.5", 32.0])
    def test_lossy_values_are_refused_on_an_int_field(
        self, pipeline_config, value
    ) -> None:
        path = self._path(pipeline_config, "stages.thinker.tp_size")
        with pytest.raises(ConfigPathError, match="tp_size expects"):
            path.coerce(value)

    def test_a_boolean_is_refused_on_a_float_field(self, pipeline_config) -> None:
        path = self._path(
            pipeline_config, "stages.thinker.factory.prefill_coalesce_wait_ms"
        )
        with pytest.raises(ConfigPathError, match="got a boolean"):
            path.coerce("true")

    def test_an_int_still_fits_a_float_field(self, pipeline_config) -> None:
        path = self._path(
            pipeline_config, "stages.thinker.factory.prefill_coalesce_wait_ms"
        )
        assert path.coerce("40") == 40.0

    def test_zero_and_one_still_fit_a_bool_field(self, pipeline_config) -> None:
        path = self._path(pipeline_config, "stages.thinker.factory.enable_async_decode")
        assert path.coerce("1") is True
        assert path.coerce("0") is False

    def test_numeric_text_stays_text_on_a_string_field(self, pipeline_config) -> None:
        path = self._path(pipeline_config, "stages.thinker.factory.device")
        assert path.coerce("123") == "123"
