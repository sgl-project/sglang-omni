# SPDX-License-Identifier: Apache-2.0
"""Unified lifecycle tests for v1 relay implementations."""

import pytest
import torch

from sglang_omni_v1.relay.nixl import NixlRelay


@pytest.fixture(params=["nixl"])
def relay_class(request):
    if request.param == "nixl":
        return NixlRelay


@pytest.fixture
def relay_config(relay_class):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return "worker0", device


def _create_connector(relay_class, config):
    try:
        return relay_class(config[0], device=config[1])
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"Failed to initialize {relay_class.__name__}: {e}")


class TestRelayUnified:
    def test_health(self, relay_class, relay_config):
        connector = _create_connector(relay_class, relay_config)
        try:
            if hasattr(connector, "health"):
                health = connector.health()
                assert isinstance(health, dict)
        finally:
            if hasattr(connector, "close"):
                connector.close()

    def test_cleanup(self, relay_class, relay_config):
        connector = _create_connector(relay_class, relay_config)
        try:
            if hasattr(connector, "cleanup"):
                connector.cleanup("test_request_id")
        finally:
            if hasattr(connector, "close"):
                connector.close()
