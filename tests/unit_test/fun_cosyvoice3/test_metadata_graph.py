# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace
from unittest.mock import Mock

from sglang_omni.models.fun_cosyvoice3 import metadata_graph


def _with_flag(monkeypatch, value):
    envs = SimpleNamespace()
    if value is not None:
        envs.SGLANG_ENABLE_METADATA_GLUE_GRAPH = SimpleNamespace(get=lambda: value)
    monkeypatch.setitem(sys.modules, "sglang.srt.environ", SimpleNamespace(envs=envs))


def test_capture_override_is_scoped_and_idempotent(monkeypatch):
    _with_flag(monkeypatch, True)
    graph = Mock()
    torch = SimpleNamespace(cuda=SimpleNamespace(graph=graph))
    module = SimpleNamespace(torch=torch)
    monkeypatch.setattr(metadata_graph.importlib, "import_module", lambda name: module)

    metadata_graph.patch_metadata_capture()
    patched = module.torch
    metadata_graph.patch_metadata_capture()
    assert module.torch is patched
    assert torch.cuda.graph is graph

    patched.cuda.graph("g", stream="s")
    graph.assert_called_once_with("g", stream="s", capture_error_mode="thread_local")
    patched.cuda.graph("g", capture_error_mode="global")
    graph.assert_called_with("g", capture_error_mode="global")


def test_no_patch_when_flag_is_off_or_absent(monkeypatch):
    load = Mock()
    monkeypatch.setattr(metadata_graph.importlib, "import_module", load)
    for value in (None, False):
        _with_flag(monkeypatch, value)
        metadata_graph.patch_metadata_capture()
    load.assert_not_called()
