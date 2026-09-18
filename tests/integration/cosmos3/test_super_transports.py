# SPDX-License-Identifier: Apache-2.0
"""Valid-input action smoke and text understanding on both public transports."""

import os

import pytest

from tests.integration.cosmos3.super_harness import deployment, transports
from tests.integration.cosmos3.test_super_gpu import (
    _check_action,
    _check_media,
    _config,
    _generation_case,
)
from tests.test_model.test_cosmos3_super_generator_ci import _devices, _resolve_model

pytestmark = pytest.mark.skipif(
    os.environ.get("COSMOS3_SUPER_RUN_GPU") != "1",
    reason="Set COSMOS3_SUPER_RUN_GPU=1 on a GPU node",
)


@pytest.mark.parametrize("transport", transports())
@pytest.mark.asyncio
async def test_action_smoke(tmp_path, transport):
    config = _config(
        "generation", str(_resolve_model()), _devices(), tmp_path / "media"
    )
    async with deployment(config, transport, tmp_path / transport) as client:
        for mode in ("policy", "inverse_dynamics", "forward_dynamics"):
            inputs = _generation_case(mode, tmp_path)
            chunk = await client.generate(inputs)
            assert chunk.media and len(chunk.media) == 1
            if mode == "forward_dynamics":
                _check_media(
                    chunk.media[0],
                    inputs["num_frames"],
                    inputs["width"],
                    inputs["height"],
                )
            else:
                _check_action(
                    chunk.media[0],
                    mode,
                    inputs["num_frames"] - 1,
                    inputs["raw_action_dim"],
                )


@pytest.mark.parametrize("transport", transports())
@pytest.mark.asyncio
async def test_reasoner_text(tmp_path, transport):
    config = _config("reasoner", str(_resolve_model("reasoner")), _devices(), tmp_path)
    async with deployment(config, transport, tmp_path / transport) as client:
        result = await client.completion(
            {
                "model": config.name,
                "messages": [
                    {
                        "role": "user",
                        "content": "What is two plus two? Answer with one digit.",
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 128,
                "stream": False,
            }
        )
        assert result["choices"][0]["message"]["content"].strip().endswith("4")
