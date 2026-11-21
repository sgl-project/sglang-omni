# SPDX-License-Identifier: Apache-2.0
"""Minimal functional smoke: every model Intel XPU serves answers one request.

Content is checked, not just status; the request shapes and their assertions live
in exercises.py. No WER, MOS, or throughput floors -- those need the eval extra
and belong in tests/test_model/.

    pytest tests/test_ci/xpu_model -v                     # every model with weights
    pytest tests/test_ci/xpu_model -v -k "asr or tts"     # a subset
    REQUIRE_FREE_GPU=0 pytest tests/test_ci/xpu_model -v  # ignore who else is on the card
"""

from __future__ import annotations

import os
from collections.abc import Callable

import pytest

from tests.test_ci.xpu_model.conftest import resolve_checkpoint
from tests.test_ci.xpu_model.exercises import EXERCISES
from tests.test_ci.xpu_model.model_table import ALLOW_DOWNLOAD_ENV, XPU_MODELS, XpuModel

# accelerator keeps it out of the unit lanes, which deselect it; xpu_serving selects it.
pytestmark = [pytest.mark.accelerator, pytest.mark.xpu_serving]


def serve_plan(
    model: XpuModel, claim: Callable[[int], list[int]], family: str
) -> tuple[list[str], dict[str, str]]:
    """The serve arguments and environment this row needs on a ``family`` card.

    The flags themselves come from the row, which carries its section of the XPU
    install guide; this only claims the cards and hands the row's own
    environment over. A stage's gpu id indexes into ZE_AFFINITY_MASK rather than
    the host, which is why the mask is set here and the rows count lanes from
    zero.
    """
    cards = claim(model.cards)
    env = {"ZE_AFFINITY_MASK": ",".join(str(card) for card in cards)}
    env["SGLANG_OMNI_STARTUP_TIMEOUT"] = str(model.startup_seconds)
    if os.environ.get(ALLOW_DOWNLOAD_ENV) != "1":
        # The server runs with proxy variables stripped, so it cannot reach the Hub,
        # and every lookup there waits out a connect timeout instead of reading the
        # cache. AuK loads a second model by repo id and spent 30 minutes in one.
        env["HF_HUB_OFFLINE"] = "1"
    else:
        pass
    env.update(dict(model.env))
    return list(model.serve_args_on(family)), env


@pytest.mark.parametrize("model", XPU_MODELS, ids=[m.key for m in XPU_MODELS])
def test_the_model_serves_and_answers_on_xpu(
    model: XpuModel,
    xpu_host: None,
    xpu_device_family: str,
    serve_xpu_model,
    claim_cards,
) -> None:
    """Serve ``model`` on the XPU, send one request, and check what came back."""
    if model.serving:
        pytest.skip(model.serving)
    else:
        pass
    checkpoint = resolve_checkpoint(model)
    extra_args, env = serve_plan(model, claim_cards, xpu_device_family)

    with serve_xpu_model(model, checkpoint, extra_args, env) as base_url:
        verdict = EXERCISES[model.kind](base_url, checkpoint, model)

    print(f"\n{model.key} ({xpu_device_family}): {verdict}")
