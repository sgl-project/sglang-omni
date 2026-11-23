# SPDX-License-Identifier: Apache-2.0
"""Fixtures of the native runtime's protocol tests.

NATIVE_STUB_SERVER is the stub_server binary; NATIVE_RUNTIME_BIN is an
installed runtime's bin directory; NATIVE_MING3_MTP_PROBE is the
ming3_mtp_probe binary.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from native_server import SupervisedServer


@pytest.fixture(scope="session")
def stub_server_binary() -> Path:
    return Path(os.environ["NATIVE_STUB_SERVER"])


@pytest.fixture(scope="session")
def ming3_mtp_probe_binary() -> Path:
    return Path(os.environ["NATIVE_MING3_MTP_PROBE"])


@pytest.fixture(scope="session")
def runtime_bin() -> Path:
    return Path(os.environ["NATIVE_RUNTIME_BIN"])


@pytest.fixture(scope="session")
def model_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Where a stand-in model loads from; stub_server reads nothing there."""
    return tmp_path_factory.mktemp("stub-model")


@pytest.fixture(scope="module")
def asr_server(
    stub_server_binary: Path, model_directory: Path
) -> Iterator[SupervisedServer]:
    server = SupervisedServer(stub_server_binary, "stub_asr", model_directory)
    yield server
    server.stop()


@pytest.fixture(scope="module")
def speech_server(
    stub_server_binary: Path, model_directory: Path
) -> Iterator[SupervisedServer]:
    server = SupervisedServer(stub_server_binary, "stub_speech", model_directory)
    yield server
    server.stop()


@pytest.fixture(scope="module")
def ming_speech_server(
    stub_server_binary: Path, model_directory: Path
) -> Iterator[SupervisedServer]:
    server = SupervisedServer(stub_server_binary, "stub_ming_tts", model_directory)
    yield server
    server.stop()


@pytest.fixture(scope="module")
def chat_server(
    stub_server_binary: Path, model_directory: Path
) -> Iterator[SupervisedServer]:
    server = SupervisedServer(stub_server_binary, "stub_chat", model_directory)
    try:
        yield server
    finally:
        server.stop()
