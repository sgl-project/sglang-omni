# SPDX-License-Identifier: Apache-2.0
"""Tests for legacy -> v1 CLI dispatch."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import typer

from sglang_omni.cli.serve import _build_v1_exec_argv, serve


def test_build_v1_exec_argv_strips_split_version_flag() -> None:
    argv = [
        "sgl-omni",
        "serve",
        "--model-path",
        "dummy",
        "--version",
        "v1",
        "--thinker-cuda-graph",
        "on",
    ]

    assert _build_v1_exec_argv(argv) == [
        sys.executable,
        "-m",
        "sglang_omni_v1.cli",
        "serve",
        "--model-path",
        "dummy",
        "--thinker-cuda-graph",
        "on",
    ]


def test_build_v1_exec_argv_strips_inline_version_flag() -> None:
    argv = [
        "sgl-omni",
        "serve",
        "--model-path",
        "dummy",
        "--version=v1",
        "--talker-torch-compile",
        "on",
    ]

    assert _build_v1_exec_argv(argv) == [
        sys.executable,
        "-m",
        "sglang_omni_v1.cli",
        "serve",
        "--model-path",
        "dummy",
        "--talker-torch-compile",
        "on",
    ]


def test_serve_rejects_legacy_only_flags_for_v1() -> None:
    with pytest.raises(typer.BadParameter, match="legacy server"):
        serve(
            ctx=SimpleNamespace(args=[]),
            model_path="dummy",
            config=None,
            text_only=False,
            host="0.0.0.0",
            port=8000,
            model_name=None,
            mem_fraction_static=0.75,
            thinker_mem_fraction_static=None,
            talker_mem_fraction_static=None,
            thinker_max_seq_len=None,
            version="v1",
            log_level="info",
        )


@patch("sglang_omni.cli.serve._dispatch_to_v1_cli")
@patch("sglang_omni.cli.serve.ConfigManager.from_model_path")
def test_serve_dispatches_to_v1_before_loading_legacy_config(
    from_model_path, dispatch_to_v1_cli
) -> None:
    serve(
        ctx=SimpleNamespace(args=[]),
        model_path="dummy",
        config=None,
        text_only=False,
        host="0.0.0.0",
        port=8000,
        model_name=None,
        mem_fraction_static=None,
        thinker_mem_fraction_static=None,
        talker_mem_fraction_static=None,
        thinker_max_seq_len=None,
        version="v1",
        log_level="info",
    )

    dispatch_to_v1_cli.assert_called_once_with()
    from_model_path.assert_not_called()
