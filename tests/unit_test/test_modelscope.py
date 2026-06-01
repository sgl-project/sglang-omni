# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import sys

from sglang_omni.environ import OMNIENV

_FLAG = "SGLANG_OMNI_USE_MODELSCOPE"


def _reimport_hub(monkeypatch, flag: str | None):
    """Re-import sglang_omni.utils.hub with the flag set/unset so its
    import-time backend selection runs fresh."""
    if flag is None:
        monkeypatch.delenv(_FLAG, raising=False)
    else:
        monkeypatch.setenv(_FLAG, flag)
    sys.modules.pop("sglang_omni.utils.hub", None)
    return importlib.import_module("sglang_omni.utils.hub")


def test_env_var_defaults_to_false(monkeypatch) -> None:
    monkeypatch.delenv(_FLAG, raising=False)
    assert OMNIENV.SGLANG_OMNI_USE_MODELSCOPE.get() is False


def test_autoconfig_backend_is_transformers_when_flag_off(monkeypatch) -> None:
    hub = _reimport_hub(monkeypatch, None)
    assert hub.AutoConfig.__module__.startswith("transformers")


def test_autoconfig_backend_is_modelscope_when_flag_on(monkeypatch) -> None:
    hub = _reimport_hub(monkeypatch, "true")
    assert hub.AutoConfig.__module__.startswith("modelscope")
