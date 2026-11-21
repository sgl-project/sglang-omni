# SPDX-License-Identifier: Apache-2.0
"""Warm MiniCPM-o native realtime sampling before the server reports ready."""
from __future__ import annotations

import asyncio
from dataclasses import replace

from sglang_omni.serve.realtime.control import Closed, Drained, Failure
from sglang_omni.serve.realtime.runtime import SessionRuntime
from sglang_omni.serve.realtime.types import AdapterFactory, Capabilities, RuntimeLimits

# Four native units are the verified short-session warmup shape, not a minimum or full graph-coverage claim.
WARMUP_UNIT_COUNT = 4
WARMUP_MAX_NEW_TOKENS_PER_UNIT = 2


async def warm_up_minicpm_session(
    capabilities: Capabilities,
    adapter_factory: AdapterFactory,
    limits: RuntimeLimits,
    timeout_seconds: float,
) -> None:
    """Run a short audio session whose first unit takes the non-forced sampler path."""
    runtime = SessionRuntime(
        "minicpm-o",
        capabilities,
        adapter_factory,
        replace(
            limits,
            session_update_timeout_s=min(
                limits.session_update_timeout_s, timeout_seconds
            ),
        ),
    )

    async def drain(until: type[Drained] | type[Closed]) -> None:
        async for envelope in runtime.outputs():
            if isinstance(envelope.event, Failure):
                raise RuntimeError(envelope.event.message)
            elif isinstance(envelope.event, until):
                return
            else:
                pass

    drain_task = asyncio.create_task(drain(Drained))
    try:
        await runtime.update(
            {
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": capabilities.input_sample_rate_hz,
                        }
                    }
                },
                "sglang": {
                    "sampling": {
                        "force_listen_count": 0,
                        "listen_prob_scale": 0.0,
                        "max_new_tokens_per_unit": WARMUP_MAX_NEW_TOKENS_PER_UNIT,
                    }
                },
            },
            "minicpm-warmup-configure",
        )
        await runtime.append(
            bytes(capabilities.native_unit_bytes * WARMUP_UNIT_COUNT),
            0,
            None,
            "minicpm-warmup-audio",
        )
        await runtime.end("minicpm-warmup-end")
        await drain_task
        await runtime.close("minicpm-warmup-finished")
        await drain(Closed)
    finally:
        await runtime.close("minicpm-warmup-finished")
        drain_task.cancel()
        await asyncio.gather(drain_task, return_exceptions=True)
