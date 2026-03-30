# SPDX-License-Identifier: Apache-2.0
"""Shared test utilities."""

from __future__ import annotations

import socket


def find_free_port() -> int:
    """Find and return an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class DummyBatchPlanner:
    """No-op batch planner for scheduler unit tests."""

    def select_requests(self, waiting_reqs, running_reqs, resource_manager):
        del waiting_reqs, running_reqs, resource_manager
        return []

    def build_batch(self, selected):
        del selected
        return None


class DummyResourceManager:
    """No-op resource manager for scheduler unit tests."""

    def free(self, request):
        del request


class DummyIterationController:
    """No-op iteration controller for scheduler unit tests."""

    def update_request(self, request, output):
        del request, output

    def is_finished(self, request, output):
        del request, output
        return False
