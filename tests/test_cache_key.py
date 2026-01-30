# SPDX-License-Identifier: Apache-2.0
"""Tests for engine cache functionality."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from sglang_omni.engines.omni import create_encoder_engine
from sglang_omni.engines.omni.runtime.encoder import EncoderRequestData


class CountingModel(nn.Module):
    """Mock model that counts forward calls."""

    def __init__(self):
        super().__init__()
        self.forward_count = 0

    def forward(self, **kwargs) -> Any:
        self.forward_count += 1
        if kwargs.get("_skip_all"):
            return {}
        # Return mock encoder output
        if "input_ids" in kwargs:
            batch_size, seq_len = kwargs["input_ids"].shape
            output = MagicMock()
            output.last_hidden_state = torch.randn(batch_size, seq_len, 64)
            return output
        return {
            "image_embeds": torch.randn(1, 10, 64),
            "image_token_counts": torch.tensor([10]),
        }


@pytest.fixture
def engine_with_cache():
    """Create engine with cache enabled."""
    model = CountingModel()
    engine = create_encoder_engine(
        model=model, device="cpu", use_cache=True, cache_size=100
    )
    return engine, model


@pytest.fixture
def engine_without_cache():
    """Create engine with cache disabled."""
    model = CountingModel()
    engine = create_encoder_engine(model=model, device="cpu", use_cache=False)
    return engine, model


async def run_request(
    engine, request_id: str, cache_key: str, input_ids=None, input_dict=None
):
    """Helper to run a single request."""
    if input_ids is None:
        input_ids = torch.tensor([1, 2, 3])
    data = EncoderRequestData(
        input_ids=input_ids, input_dict=input_dict, cache_key=cache_key
    )
    await engine.add_request(request_id, data)
    return await engine.get_result(request_id)


class TestCacheHit:
    """Test cache hit/miss behavior."""

    @pytest.mark.asyncio
    async def test_same_key_hits_cache(self, engine_with_cache):
        """Same cache_key should skip model execution."""
        engine, model = engine_with_cache
        await engine.start()
        try:
            await run_request(engine, "req-1", cache_key="key-A")
            await run_request(engine, "req-2", cache_key="key-A")  # Same key
            assert model.forward_count == 1, "Second request should hit cache"
        finally:
            await engine.stop()

    @pytest.mark.asyncio
    async def test_different_key_misses_cache(self, engine_with_cache):
        """Different cache_key should call model."""
        engine, model = engine_with_cache
        await engine.start()
        try:
            await run_request(engine, "req-1", cache_key="key-A")
            await run_request(engine, "req-2", cache_key="key-B")  # Different key
            assert model.forward_count == 2, "Different key should miss cache"
        finally:
            await engine.stop()

    @pytest.mark.asyncio
    async def test_no_cache_always_calls_model(self, engine_without_cache):
        """Without cache, model always called."""
        engine, model = engine_without_cache
        await engine.start()
        try:
            await run_request(engine, "req-1", cache_key="key-A")
            await run_request(
                engine, "req-2", cache_key="key-A"
            )  # Same key but no cache
            assert model.forward_count == 2, "Without cache, model always called"
        finally:
            await engine.stop()

    @pytest.mark.asyncio
    async def test_multiple_hits(self, engine_with_cache):
        """Multiple requests with same key should all hit cache."""
        engine, model = engine_with_cache
        await engine.start()
        try:
            for i in range(5):
                await run_request(engine, f"req-{i}", cache_key="same-key")
            assert (
                model.forward_count == 1
            ), "All 5 requests should hit cache after first"
        finally:
            await engine.stop()


class TestCacheEviction:
    """Test LRU eviction."""

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Old entries evicted when cache full."""
        model = CountingModel()
        engine = create_encoder_engine(
            model=model, device="cpu", use_cache=True, cache_size=2
        )
        await engine.start()
        try:
            # Fill cache
            await run_request(engine, "req-0", cache_key="key-0")
            await run_request(engine, "req-1", cache_key="key-1")
            assert model.forward_count == 2

            # Add third entry - evicts key-0
            await run_request(engine, "req-2", cache_key="key-2")
            assert model.forward_count == 3

            # key-1 still cached
            await run_request(engine, "req-3", cache_key="key-1")
            assert model.forward_count == 3

            # key-0 was evicted
            await run_request(engine, "req-4", cache_key="key-0")
            assert model.forward_count == 4
        finally:
            await engine.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
