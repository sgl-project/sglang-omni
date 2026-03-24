# SPDX-License-Identifier: Apache-2.0
"""Shared test dummy implementations."""

from __future__ import annotations


class DummyBatchPlanner:
    def __init__(self, *args, **kwargs):
        pass

    def select_requests(self, *args, **kwargs):
        return []

    def build_batch(self, *args, **kwargs):
        return None


class DummyResourceManager:
    def __init__(self, *args, **kwargs):
        pass

    def free(self, request):
        del request


class DummyIterationController:
    def __init__(self, *args, **kwargs):
        pass

    def update_request(self, request, output):
        del request, output

    def is_finished(self, request, output):
        del request, output
        return False
