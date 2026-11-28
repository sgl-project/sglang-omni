# SPDX-License-Identifier: Apache-2.0
"""Minimal functional smoke: every model Intel XPU serves answers one request.

pytest tests/test_ci/xpu_model -m xpu_serving -v [-k "asr or tts"] [-m "xpu_serving and xpu_tp"]
"""

from __future__ import annotations

import os

import pytest

from tests.test_ci.xpu_model.conftest import (
    ClaimCards,
    ServeModel,
    require_card_count,
    resolve_checkpoint,
)
from tests.test_ci.xpu_model.exercises import EXERCISES
from tests.test_ci.xpu_model.model_table import ALLOW_DOWNLOAD_ENV, XPU_MODELS, XpuModel

# accelerator keeps it out of the unit lanes, which deselect it; xpu_serving selects it.
pytestmark = [pytest.mark.accelerator, pytest.mark.xpu_serving]

ROWS = [
    pytest.param(
        model, id=model.key, marks=[pytest.mark.xpu_tp] if model.tp_size > 1 else []
    )
    for model in XPU_MODELS
]


def serve_plan(
    model: XpuModel, claim: ClaimCards, family: str
) -> tuple[list[str], dict[str, str]]:
    """Claim the row's cards and return its serve arguments and environment on a card of the family."""
    cards = claim(model.cards)
    env = {"ZE_AFFINITY_MASK": ",".join(str(card) for card in cards)}
    env["SGLANG_OMNI_STARTUP_TIMEOUT"] = str(model.startup_seconds)
    if os.environ.get(ALLOW_DOWNLOAD_ENV) != "1":
        # The server strips proxy variables, so a Hub lookup waits out a connect timeout instead.
        env["HF_HUB_OFFLINE"] = "1"
    else:
        pass
    env.update(dict(model.env))
    return list(model.serve_args_on(family)), env


@pytest.mark.parametrize("model", ROWS)
def test_the_model_serves_and_answers_on_xpu(
    model: XpuModel,
    xpu_host: None,
    xpu_device_family: str,
    serve_xpu_model: ServeModel,
    claim_cards: ClaimCards,
) -> None:
    """Serve the row on the XPU, send one request, and check what came back."""
    if model.serving:
        pytest.skip(model.serving)
    else:
        pass
    require_card_count(model.cards)
    checkpoint = resolve_checkpoint(model)
    extra_args, env = serve_plan(model, claim_cards, xpu_device_family)

    with serve_xpu_model(model, checkpoint, extra_args, env) as base_url:
        verdict = EXERCISES[model.kind](base_url, checkpoint, model)

    print(f"\n{model.key} ({xpu_device_family}): {verdict}")
