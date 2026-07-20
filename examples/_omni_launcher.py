# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
from collections.abc import Sequence

if __package__:
    from examples.launchers._common import LauncherPreset
    from examples.launchers.ming_omni import PRESETS as MING_PRESETS
    from examples.launchers.qwen3_omni import PRESETS as QWEN3_PRESETS
else:
    from launchers._common import LauncherPreset
    from launchers.ming_omni import PRESETS as MING_PRESETS
    from launchers.qwen3_omni import PRESETS as QWEN3_PRESETS

PRESETS: dict[str, LauncherPreset] = {
    **QWEN3_PRESETS,
    **MING_PRESETS,
}


def parse_preset_args(
    preset_name: str,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    return PRESETS[preset_name].build_parser().parse_args(argv)


def run_preset(
    preset_name: str,
    argv: Sequence[str] | None = None,
) -> None:
    preset = PRESETS[preset_name]
    if preset.spawn:
        mp.set_start_method("spawn", force=True)
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", preset.default_log_level).upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    preset.run(parse_preset_args(preset_name, argv))


def run_cli(argv: Sequence[str] | None = None) -> None:
    argv = list(argv) if argv is not None else sys.argv[1:]
    if argv and argv[0] in PRESETS:
        run_preset(argv[0], argv[1:])
        return
    selector = argparse.ArgumentParser(
        description="Run an Omni example through a reusable launcher preset."
    )
    selector.add_argument("preset", choices=sorted(PRESETS))
    selector.parse_args(argv)
