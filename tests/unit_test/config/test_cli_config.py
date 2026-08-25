# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``sgl-omni config resolve`` and ``sgl-omni config explain``.

These commands exist to answer two questions a launch cannot: *what
configuration would this launch actually use*, and *which source set this
value*. Both answers are only worth printing if they match what the server
would compute, so the tests check the printed result against
``ConfigManager`` -- the same entry point ``sgl-omni serve`` calls -- rather
than against the command's own internals.

The fixtures drive the commands through ``--config`` rather than
``--model-path``: resolving a model path reads the model's own config from
disk, which these tests have no weights for.
"""

from __future__ import annotations

from unittest import mock

import pytest
import yaml
from typer.testing import CliRunner

from sglang_omni.cli.config import config_app
from sglang_omni.cli.serve import patches_from_broadcast_flags
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.sources import dump_user_config


def output_of(result) -> str:
    """Everything the command wrote, whichever click version is installed.

    click 8.2 separates stderr from stdout; older versions fold it in and
    raise on ``result.stderr``.
    """
    text = result.stdout
    try:
        text += result.stderr
    except (AttributeError, ValueError):  # pragma: no cover - version dependent
        pass
    return text


@pytest.fixture
def runner() -> CliRunner:
    """A runner whose ``result.stdout`` is stdout alone, colors off.

    click < 8.2 folds stderr into stdout unless asked not to, which would let
    a stderr notice land in the middle of a document meant for redirection.
    click >= 8.2 always separates the two and dropped the argument. NO_COLOR
    keeps rich from wrapping flag names in escape codes on color-forcing
    terminals (CI), where the codes land inside ``--flag`` and break substring
    assertions on error text.
    """
    env = {"NO_COLOR": "1", "TERM": "dumb"}
    try:
        return CliRunner(mix_stderr=False, env=env)
    except TypeError:  # pragma: no cover - version dependent
        return CliRunner(env=env)


@pytest.fixture
def base_config():
    """A shipped pipeline config, so the paths are ones users actually type."""
    module = pytest.importorskip("sglang_omni.models.moss_tts.config")
    return module.MossTTSPipelineConfig(model_path="dummy")


@pytest.fixture
def stage(base_config) -> str:
    """The pipeline's first engine stage: the one the fixtures address."""
    cls = type(base_config)
    for entry in base_config.stages:
        if cls.stage_config_cls(entry.name).engine_stage:
            return entry.name
    raise AssertionError("fixture pipeline has no engine stage")


@pytest.fixture
def fraction(stage) -> str:
    return f"stages.{stage}.engine.mem_fraction_static"


@pytest.fixture
def config_file(tmp_path, base_config, stage):
    """A config file whose ``stages:`` mapping sets one engine field."""
    data = {
        "config_cls": base_config.config_cls,
        "model_path": "dummy",
        "stages": {stage: {"engine": {"mem_fraction_static": 0.55}}},
    }
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


@pytest.fixture
def plain_config_file(tmp_path, base_config):
    """The same pipeline with no stage entries at all."""
    data = {"config_cls": base_config.config_cls, "model_path": "dummy"}
    path = tmp_path / "plain.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


class TestResolve:
    def test_prints_the_config_a_launch_would_use(self, runner, config_file):
        """The whole point: this is what `serve` with these arguments builds."""
        result = runner.invoke(config_app, ["resolve", "--config", str(config_file)])

        assert result.exit_code == 0, output_of(result)
        expected = ConfigManager.from_file(str(config_file)).config
        assert yaml.safe_load(result.stdout) == dump_user_config(expected)

    def test_dotted_arguments_are_applied_like_serve_applies_them(
        self, runner, plain_config_file, stage
    ):
        argv = [f"--{stage}.engine.mem-fraction-static", "0.35"]
        result = runner.invoke(
            config_app, ["resolve", "--config", str(plain_config_file), *argv]
        )

        assert result.exit_code == 0, output_of(result)
        manager = ConfigManager.from_file(str(plain_config_file))
        expected = manager.merge_config(manager.parse_extra_args(argv))
        assert yaml.safe_load(result.stdout) == dump_user_config(expected)

    def test_stdout_is_a_usable_config_file(self, runner, config_file, stage):
        """`config resolve > pipeline.yaml` must produce a loadable document
        in the same shape the input was written in: a ``stages:`` mapping."""
        result = runner.invoke(config_app, ["resolve", "--config", str(config_file)])

        assert result.exit_code == 0, output_of(result)
        printed = yaml.safe_load(result.stdout)
        assert printed["stages"][stage]["engine"]["mem_fraction_static"] == 0.55

    def test_diff_lists_only_what_the_sources_changed(
        self, runner, config_file, fraction
    ):
        result = runner.invoke(
            config_app, ["resolve", "--config", str(config_file), "--show", "diff"]
        )

        assert result.exit_code == 0, output_of(result)
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        assert lines == [f"{fraction}: None -> 0.55"]

    def test_diff_is_empty_when_no_source_says_anything(
        self, runner, plain_config_file
    ):
        result = runner.invoke(
            config_app,
            ["resolve", "--config", str(plain_config_file), "--show", "diff"],
        )

        assert result.exit_code == 0, output_of(result)
        assert "No configuration source changed" in result.stdout

    def test_provenance_shows_the_file_losing_to_the_command_line(
        self, runner, config_file, stage, fraction
    ):
        result = runner.invoke(
            config_app,
            [
                "resolve",
                "--config",
                str(config_file),
                "--show",
                "provenance",
                f"--{stage}.engine.mem_fraction_static",
                "0.35",
            ],
        )

        assert result.exit_code == 0, output_of(result)
        assert f"{fraction} = 0.35" in result.stdout
        assert "0.55  <- yaml file" in result.stdout
        assert "[superseded]" in result.stdout
        assert "0.35  <- cli dotted (command line)  [winner]" in result.stdout

    def test_model_path_is_required_without_a_config_file(self, runner):
        result = runner.invoke(config_app, ["resolve"])

        assert result.exit_code != 0
        assert "--model-path is required" in output_of(result)

    def test_the_explicit_stages_prefix_is_refused_without_a_traceback(
        self, runner, plain_config_file, stage
    ):
        """Flags start at the stage name; the refusal says the prefix is
        implied instead of dumping the parse internals."""
        result = runner.invoke(
            config_app,
            [
                "resolve",
                "--config",
                str(plain_config_file),
                f"--stages.{stage}.tp_size",
                "2",
            ],
        )

        assert result.exit_code != 0
        output = output_of(result)
        assert "implied" in output
        assert "Traceback" not in output

    def test_an_unknown_stage_head_names_the_real_stages(
        self, runner, plain_config_file, stage
    ):
        result = runner.invoke(
            config_app,
            [
                "resolve",
                "--config",
                str(plain_config_file),
                "--nonexistent_stage.tp_size",
                "2",
            ],
        )

        assert result.exit_code != 0
        output = output_of(result)
        assert stage in output
        assert "Traceback" not in output


class TestExplain:
    def test_names_the_winning_source_and_what_it_overrode(
        self, runner, config_file, stage, fraction
    ):
        result = runner.invoke(
            config_app,
            [
                "explain",
                fraction,
                "--config",
                str(config_file),
                f"--{stage}.engine.mem_fraction_static",
                "0.35",
            ],
        )

        assert result.exit_code == 0, output_of(result)
        assert f"{fraction} = 0.35" in result.stdout
        assert "None  <- model default" in result.stdout
        assert "0.55  <- yaml file" in result.stdout
        assert "0.35  <- cli dotted (command line)  [winner]" in result.stdout

    def test_the_cli_spelling_of_a_path_is_accepted(
        self, runner, config_file, stage, fraction
    ):
        """The question can be asked the way the flag is written: stage name
        first, ``stages.`` implied."""
        result = runner.invoke(
            config_app,
            [
                "explain",
                f"{stage}.engine.mem_fraction_static",
                "--config",
                str(config_file),
            ],
        )

        assert result.exit_code == 0, output_of(result)
        assert f"{fraction} = 0.55" in result.stdout

    def test_untouched_path_reports_the_model_default(self, runner, config_file, stage):
        """A path no source mentions is a normal question, not an error."""
        result = runner.invoke(
            config_app,
            ["explain", f"stages.{stage}.tp_size", "--config", str(config_file)],
        )

        assert result.exit_code == 0, output_of(result)
        assert f"stages.{stage}.tp_size = 1" in result.stdout
        assert "no source touched this path" in result.stdout

    def test_unknown_path_suggests_instead_of_raising(self, runner, config_file, stage):
        result = runner.invoke(
            config_app,
            ["explain", f"stages.{stage}.tp_sizee", "--config", str(config_file)],
        )

        assert result.exit_code != 0
        assert "did you mean" in output_of(result).lower()

    def test_a_positional_stage_index_is_refused_readably(self, runner, config_file):
        """Stages are addressed by name; the refusal says so, tracelessly."""
        result = runner.invoke(
            config_app,
            ["explain", "stages.9.tp_size", "--config", str(config_file)],
        )

        assert result.exit_code != 0
        output = output_of(result)
        assert "addressed by name" in output
        assert "Traceback" not in output

    def test_without_a_path_it_lists_every_touched_path(
        self, runner, config_file, fraction
    ):
        result = runner.invoke(config_app, ["explain", "--config", str(config_file)])

        assert result.exit_code == 0, output_of(result)
        assert f"{fraction} = 0.55  <- yaml file" in result.stdout

    def test_a_top_level_file_value_names_the_file(self, runner, tmp_path, base_config):
        """model_path written at the file's top level is the file's write,
        not the model default."""
        data = {
            "config_cls": base_config.config_cls,
            "model_path": "from/the/file",
        }
        path = tmp_path / "pipeline.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False))

        result = runner.invoke(
            config_app, ["explain", "model_path", "--config", str(path)]
        )

        assert result.exit_code == 0, output_of(result)
        assert "model_path = 'from/the/file'" in result.stdout
        assert "yaml file" in result.stdout
        assert "[winner]" in result.stdout


class TestTensorParallelDerivation:
    """The post-merge TP derivation: preview shows it, users outrank it."""

    @pytest.fixture
    def ming_config_file(self, tmp_path):
        pytest.importorskip("sglang_omni.models.ming_omni.config")
        data = {
            "config_cls": "MingOmniPipelineConfig",
            "model_path": "dummy",
            "stages": {"thinker": {"tp_size": 2, "gpu": [0, 1]}},
        }
        path = tmp_path / "ming.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return path

    def test_resolve_previews_the_derived_engine_override(
        self, runner, ming_config_file
    ):
        with mock.patch(
            "sglang_omni.cli.serve.should_disable_custom_all_reduce_for_gpus",
            return_value=True,
        ):
            result = runner.invoke(
                config_app,
                ["resolve", "--config", str(ming_config_file), "--show", "config"],
            )

        assert result.exit_code == 0, output_of(result)
        assert "disable_custom_all_reduce: true" in result.stdout

    def test_an_explicit_engine_value_outranks_the_derivation(
        self, runner, ming_config_file
    ):
        with mock.patch(
            "sglang_omni.cli.serve.should_disable_custom_all_reduce_for_gpus",
            return_value=True,
        ):
            result = runner.invoke(
                config_app,
                [
                    "resolve",
                    "--config",
                    str(ming_config_file),
                    "--thinker.engine.disable_custom_all_reduce",
                    "false",
                    "--show",
                    "config",
                ],
            )

        assert result.exit_code == 0, output_of(result)
        assert "disable_custom_all_reduce: false" in result.stdout


class TestModelPathWithConfig:
    """``--config`` plus ``--model-path``: the flag wins, as it does in serve."""

    def test_resolve_prefers_the_flag_over_the_file(self, runner, plain_config_file):
        result = runner.invoke(
            config_app,
            [
                "resolve",
                "--config",
                str(plain_config_file),
                "--model-path",
                "local/checkout",
            ],
        )

        assert result.exit_code == 0, output_of(result)
        assert yaml.safe_load(result.stdout)["model_path"] == "local/checkout"

    def test_explain_names_the_flag_as_the_winner(self, runner, plain_config_file):
        result = runner.invoke(
            config_app,
            [
                "explain",
                "model_path",
                "--config",
                str(plain_config_file),
                "--model-path",
                "local/checkout",
            ],
        )

        assert result.exit_code == 0, output_of(result)
        assert "model_path = 'local/checkout'" in result.stdout
        assert "cli flag (--model-path)  [winner]" in result.stdout


class TestBroadcastFlagOnConfigCommands:
    """config resolve/explain accept the broadcast flag exactly as serve does,
    so a launch can be rehearsed with the same argument list."""

    def test_resolve_applies_the_broadcast(self, runner, plain_config_file, stage):
        result = runner.invoke(
            config_app,
            [
                "resolve",
                "--config",
                str(plain_config_file),
                "--mem-fraction-static",
                "0.5",
            ],
        )

        assert result.exit_code == 0, output_of(result)
        printed = yaml.safe_load(result.stdout)
        assert printed["stages"][stage]["engine"]["mem_fraction_static"] == 0.5

    def test_explain_shows_the_dotted_path_beating_the_broadcast(
        self, runner, plain_config_file, stage, fraction
    ):
        result = runner.invoke(
            config_app,
            [
                "explain",
                fraction,
                "--config",
                str(plain_config_file),
                "--mem-fraction-static",
                "0.5",
                f"--{stage}.engine.mem_fraction_static",
                "0.6",
            ],
        )

        assert result.exit_code == 0, output_of(result)
        assert f"{fraction} = 0.6" in result.stdout
        assert "cli flag (--mem-fraction-static)  [superseded]" in result.stdout
        assert "cli dotted (command line)  [winner]" in result.stdout


class TestBroadcastFlags:
    """The three convenience flags fan out as patches, below explicit paths."""

    def test_mem_fraction_reaches_every_engine_stage(self, base_config, stage):
        patches = patches_from_broadcast_flags(
            base_config,
            mem_fraction_static=0.4,
        )
        keys = {patch.key for patch in patches.ordered()}
        assert f"stages.{stage}.engine.mem_fraction_static" in keys
        cls = type(base_config)
        engine_stages = [
            entry.name
            for entry in base_config.stages
            if cls.stage_config_cls(entry.name).engine_stage
        ]
        assert keys == {
            f"stages.{name}.engine.mem_fraction_static" for name in engine_stages
        }

    def test_an_explicit_dotted_path_beats_the_fan_out(self, base_config, stage):
        flag_patches = patches_from_broadcast_flags(
            base_config,
            mem_fraction_static=0.4,
        )
        merged = ConfigManager(base_config).merge_config(
            [(f"{stage}.engine.mem_fraction_static", "0.6")],
            extra_patches=flag_patches,
        )
        assert merged.stage_named(stage).engine.mem_fraction_static == 0.6

    def test_an_out_of_range_fraction_is_refused(self, base_config):
        # Range is the schema's rule: the flag builds patches, and resolution
        # refuses the out-of-range value with the schema's message.
        patches = patches_from_broadcast_flags(base_config, mem_fraction_static=1.5)
        with pytest.raises(ValueError, match="mem_fraction_static"):
            ConfigManager(base_config).merge_config([], extra_patches=patches)

    def test_without_the_flag_no_patch_is_built(self, base_config):
        patches = patches_from_broadcast_flags(
            base_config,
            mem_fraction_static=None,
        )
        assert len(patches) == 0


class TestServeErrors:
    """`sgl-omni serve` fails the same way the config commands do: readably.

    Every case here stops in the config-merging prologue, long before
    ``launch_server`` is reached, so no server is started.
    """

    @pytest.fixture
    def app(self):
        from sglang_omni.cli import app as cli_app

        return cli_app

    def test_the_explicit_stages_prefix_is_refused_without_a_traceback(
        self, app, runner, plain_config_file, stage
    ):
        result = runner.invoke(
            app,
            [
                "serve",
                "--config",
                str(plain_config_file),
                f"--stages.{stage}.tp_size",
                "2",
            ],
        )

        assert result.exit_code != 0
        output = output_of(result)
        assert "implied" in output
        assert "Traceback" not in output

    def test_a_path_written_twice_is_refused_without_a_traceback(
        self, app, runner, plain_config_file, stage
    ):
        """The refusal is only useful if the user can read it."""
        result = runner.invoke(
            app,
            [
                "serve",
                "--config",
                str(plain_config_file),
                f"--{stage}.tp_size",
                "8",
                f"--{stage}.tp_size",
                "4",
            ],
        )

        assert result.exit_code != 0
        output = output_of(result)
        assert "is set twice at the same precedence" in output
        assert "Pick one." in output
        assert "Traceback" not in output

    def test_a_flag_without_a_value_is_refused_readably(
        self, app, runner, plain_config_file, stage
    ):
        result = runner.invoke(
            app,
            [
                "serve",
                "--config",
                str(plain_config_file),
                f"--{stage}.tp_size",
            ],
        )

        assert result.exit_code != 0
        output = output_of(result)
        assert "Missing value" in output
        assert "Traceback" not in output
