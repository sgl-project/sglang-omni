# SPDX-License-Identifier: Apache-2.0
"""Partition selected pytest files by estimated execution time."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sglang.test.ci.ci_register import CIRegistry, HWBackend, auto_partition


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--auto-partition-id", type=int, required=True)
    parser.addoption("--auto-partition-size", type=int, required=True)
    parser.addoption("--partition-model-file", type=Path, required=True)


def pytest_configure(config: pytest.Config) -> None:
    partition_id = config.getoption("--auto-partition-id")
    partition_size = config.getoption("--auto-partition-size")
    if partition_size <= 0:
        raise pytest.UsageError("--auto-partition-size must be positive.")
    elif not 0 <= partition_id < partition_size:
        raise pytest.UsageError(
            f"--auto-partition-id must be in range [0, {partition_size})."
        )
    else:
        pass


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    duration_path: Path = config.getoption("--partition-model-file")
    with duration_path.open(encoding="utf-8") as duration_file:
        duration_seconds_by_file: dict[str, float] = json.load(duration_file)
    file_paths = {item.path.relative_to(config.rootpath).as_posix() for item in items}
    test_files = [
        CIRegistry(
            backend=HWBackend.CPU,
            filename=file_path,
            est_time=duration_seconds_by_file.get(file_path, 1.0),
        )
        for file_path in sorted(file_paths)
    ]
    selected_paths = {
        test_file.filename
        for test_file in auto_partition(
            test_files,
            config.getoption("--auto-partition-id"),
            config.getoption("--auto-partition-size"),
        )
    }
    selected_items: list[pytest.Item] = []
    deselected_items: list[pytest.Item] = []
    for item in items:
        if item.path.relative_to(config.rootpath).as_posix() in selected_paths:
            selected_items.append(item)
        else:
            deselected_items.append(item)
    config.hook.pytest_deselected(items=deselected_items)
    items[:] = selected_items
