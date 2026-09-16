# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import textwrap
from pathlib import Path


def test_cosyvoice_is_required_only_for_model_loading() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import builtins

                original_import = builtins.__import__

                def without_cosyvoice(name, *args, **kwargs):
                    if name == "cosyvoice" or name.startswith("cosyvoice."):
                        raise ModuleNotFoundError(name, name=name)
                    return original_import(name, *args, **kwargs)

                builtins.__import__ = without_cosyvoice

                from sglang_omni.models.fun_cosyvoice3 import stages

                try:
                    stages.load_cosyvoice3_flow_hift("unused", device="cpu")
                except RuntimeError as exc:
                    assert str(exc) == stages.COSYVOICE_INSTALL_HINT
                else:
                    raise AssertionError("model loading must require CosyVoice")
                """
            ),
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
