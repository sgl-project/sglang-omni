# SPDX-License-Identifier: Apache-2.0
"""Async engine wrapper for plain torch modules."""

from __future__ import annotations

import asyncio
from typing import Any

import torch
import torch.nn as nn

from sglang_omni.engines.base import Engine


class AsyncModuleEngine(Engine):
    """Run a torch module asynchronously per request."""

    def __init__(self, module: nn.Module):
        self._module = module
        self._futures: dict[str, asyncio.Future[Any]] = {}
        self._aborted: set[str] = set()

    async def add_request(self, request_id: str, data: Any) -> None:
        if request_id in self._aborted:
            return
        if not isinstance(data, dict):
            raise TypeError(f"AsyncModuleEngine expects dict inputs, got {type(data)}")
        loop = asyncio.get_running_loop()
        if data.get("_skip"):
            future: asyncio.Future[Any] = loop.create_future()
            future.set_result(data.get("_result"))
            self._futures[request_id] = future
            return
        future = loop.run_in_executor(None, self._run_forward, data)
        self._futures[request_id] = future

    def _run_forward(self, data: dict[str, Any]) -> Any:
        data = {k: v for k, v in data.items() if not k.startswith("_")}
        with torch.inference_mode():
            return self._module(**data)

    async def get_result(self, request_id: str) -> Any:
        if request_id in self._aborted:
            raise asyncio.CancelledError(f"Request {request_id} was aborted")
        future = self._futures.pop(request_id, None)
        if future is None:
            raise KeyError(f"No pending future for request {request_id}")
        return await future

    async def abort(self, request_id: str) -> None:
        self._aborted.add(request_id)
        future = self._futures.pop(request_id, None)
        if future is not None:
            future.cancel()
