# SPDX-License-Identifier: Apache-2.0
"""The shipped replica example configs must load and compile as documented."""

from __future__ import annotations

from pathlib import Path

import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.topology import compile_logical_processes

_REPO_ROOT = Path(__file__).resolve().parents[3]

_REPLICA_EXAMPLES = [
    "examples/configs/qwen3_omni_speech_replica2.yaml",
    "examples/configs/qwen3_omni_speech_code2wav_replica2_ci.yaml",
]


@pytest.mark.parametrize("relative_path", _REPLICA_EXAMPLES)
def test_replica_example_loads_and_declares_its_placement(relative_path: str) -> None:
    config = ConfigManager.from_file(str(_REPO_ROOT / relative_path)).config

    plan, _ = compile_logical_processes(config)
    replicated = [process for process in plan.processes if process.is_replicated]
    assert replicated, relative_path
    # Note (Jiaxin Deng): replica_devices colocation is only valid with a
    # declared budget on every GPU stage it places, so the examples must carry
    # gpu_memory_fraction for each replicated GPU stage.
    for process in replicated:
        for stage_name in process.stage_names:
            stage = next(s for s in config.stages if s.name == stage_name)
            if stage.gpu is not None:
                assert stage.gpu_memory_fraction is not None, (
                    relative_path,
                    stage_name,
                )
