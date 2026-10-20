# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the resolver: application, provenance, and diffing.

The resolver is the only code that writes into a configuration; these are
the hand-written cases worth reading. Dotted-key canonicalization (the CLI's
implied ``stages.`` prefix and the one legacy spelling that still parses)
is covered at the bottom.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from sglang_omni.config.base import ConfigModel
from sglang_omni.config.compat import canonicalize_dotted_key
from sglang_omni.config.patch import (
    ConfigPatch,
    ConfigPatchSet,
    ConfigSource,
    SourceKind,
)
from sglang_omni.config.path import ConfigPathError
from sglang_omni.config.resolver import ConfigResolver, diff_configs
from sglang_omni.config.schema import EngineArgs, PipelineConfig

MEM_FRACTION = "stages.thinker.engine.mem_fraction_static"

YAML = ConfigSource(SourceKind.YAML_FILE, "configs/omni.yaml")
CLI = ConfigSource(SourceKind.CLI_DOTTED, "command line")
MODEL = ConfigSource(SourceKind.MODEL_DEFAULT, "FakeOmniConfig")


class EngineWrapper(BaseModel):
    engine: EngineArgs


class NestedConfig(ConfigModel):
    children: (
        EngineWrapper
        | list[EngineWrapper]
        | dict[str, tuple[EngineWrapper, ...]]
        | None
    )
    reject_parent: bool = False

    def check_local_conflicts(self) -> None:
        if self.reject_parent:
            raise ValueError("parent conflict")
        else:
            pass


@pytest.mark.parametrize("container", ["direct", "list", "dict_tuple"])
def test_conflicts_recurse_through_models_and_containers(container):
    """Nested engine conflicts must be found without parent forwarding methods."""
    engine = EngineArgs(kv_cache_bytes="2GiB", mem_fraction_static=0.7)
    wrapper = EngineWrapper(engine=engine)
    children = {
        "direct": wrapper,
        "list": [wrapper],
        "dict_tuple": {"group": (wrapper,)},
    }[container]
    config = NestedConfig(children=children, reject_parent=True)
    before = config.model_dump()
    with pytest.raises(ValueError, match="engine.kv_cache_bytes"):
        config.check_conflicts()
    assert config.model_dump() == before

    engine.mem_fraction_static = None
    with pytest.raises(ValueError, match="parent conflict"):
        config.check_conflicts()
    config.reject_parent = False
    config.check_conflicts()
    config.children = None
    config.check_conflicts()


def patchset(*patches: ConfigPatch) -> ConfigPatchSet:
    return ConfigPatchSet(list(patches))


def make(config, path, value, source=CLI, **kwargs):
    return ConfigPatch.create(path, value, source, root=type(config), **kwargs)


class TestResolve:
    @pytest.mark.parametrize(
        ("first_values", "later_values", "error"),
        [
            (
                {"stages.thinker.tp_size": 2},
                {"stages.thinker.gpu": [0, 1]},
                "gpu is required",
            ),
            (
                {
                    "stages.thinker.engine.kv_cache_bytes": "2GiB",
                    "stages.thinker.engine.mem_fraction_static": 0.7,
                },
                {"stages.thinker.engine.mem_fraction_static": None},
                "cannot be set together",
            ),
        ],
    )
    def test_later_source_can_complete_relationships(
        self, pipeline_config, first_values, later_values, error
    ):
        """A file's incomplete relationships must survive until CLI overrides."""
        intermediate = ConfigResolver(pipeline_config).resolve(
            patchset(
                *(
                    make(pipeline_config, key, value, YAML)
                    for key, value in first_values.items()
                )
            )
        )
        with pytest.raises(ValueError, match=error):
            intermediate.config.check_conflicts()
        final = ConfigResolver(intermediate.config).resolve(
            patchset(
                *(
                    make(intermediate.config, key, value, CLI)
                    for key, value in later_values.items()
                )
            )
        )
        final.config.check_conflicts()
        for key, value in later_values.items():
            assert final.value(key) == value

    @pytest.mark.parametrize(
        ("path", "value"), [("stages.thinker.gpu", -1), ("stages.thinker.tp_size", 0)]
    )
    def test_invalid_field_is_rejected_before_final_validation(
        self, pipeline_config, path, value
    ):
        with pytest.raises(ValueError):
            ConfigResolver(pipeline_config).resolve(
                patchset(make(pipeline_config, path, value))
            )

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
