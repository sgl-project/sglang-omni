# SPDX-License-Identifier: Apache-2.0
"""The speech thinker path must vet prefill CUDA graphs like every other path.

Speech defers CUDA graph capture (hidden-state capture needs the graphs built
after the hooks are installed), so it drives capture itself instead of letting
``create_sglang_infrastructure`` do it. That detour used to skip two things the
normal path gets:

  - ``enable_prefill_input_embeds``, which is what makes the prefill graph
    runner allocate the ``input_embeds`` replay slot. Without it the slot only
    exists when ``model_config.is_multimodal`` happens to be true, so an
    embedding-injected batch would silently fall back to eager.
  - ``init_sglang_cuda_graphs``, the wrapper that applies that view, and the
    post-capture attestation that the declared backend and buckets actually
    materialized.
"""

from __future__ import annotations

import inspect

from sglang_omni.models.qwen3_omni import bootstrap as qwen3_omni_bootstrap


def _source() -> str:
    return inspect.getsource(qwen3_omni_bootstrap)


def test_speech_path_requests_the_prefill_embeds_slot() -> None:
    src = _source()
    assert "enable_prefill_input_embeds=" in src, (
        "the speech thinker must ask for the prefill input_embeds slot "
        "explicitly instead of relying on model_config.is_multimodal"
    )


def test_deferred_capture_uses_the_omni_wrapper() -> None:
    src = _source()
    assert "init_sglang_cuda_graphs(" in src, (
        "deferred capture must go through init_sglang_cuda_graphs so the "
        "prefill embeds view is applied"
    )
    assert (
        "model_runner.init_cuda_graphs()" not in src
    ), "raw init_cuda_graphs() bypasses the omni prefill embeds view"


def test_speech_path_attests_prefill_graphs() -> None:
    src = _source()
    assert "attest_prefill_cuda_graphs" in src, (
        "an operator-locked prefill backend must be attested after capture on "
        "the speech path too"
    )
