# SPDX-License-Identifier: Apache-2.0
"""Ming thinker continuation preserves text output without encoder payloads."""

from unittest.mock import Mock

import pytest
import torch
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.ming_omni.bootstrap import make_thinker_scheduler_adapters
from sglang_omni.models.ming_omni.config import MingOmniPDPipelineConfig
from sglang_omni.models.ming_omni.pd import MingThinkerPDAdapter
from sglang_omni.models.ming_omni.pipeline.usage import build_text_usage
from sglang_omni.scheduling.pd_utils import (
    DecodeContinuation,
    continuation_from_req,
    req_from_continuation,
)
from tests.unit_test.pipeline.test_pd_utils import make_allocation, prefill_req


@pytest.mark.parametrize("priority", [None, -7, 0, 7])
def test_ming_roundtrip_preserves_output_and_drops_encoder_inputs(
    priority: int | None,
) -> None:
    tokenizer = Mock(spec=PreTrainedTokenizerBase)
    tokenizer.eos_token_id = 2
    adapter = MingThinkerPDAdapter(tokenizer=tokenizer)
    request = prefill_req()
    request.priority = priority
    request.omni_data.stage_payload.request.inputs = torch.ones(4)
    request.omni_data.stage_payload.data = {
        "prompt": {"input_ids": torch.tensor([10, 11, 12])},
        "thinker_inputs": {"model_inputs": {"image_embeds": torch.ones(4)}},
        "encoder_outs": {"audio": torch.ones(4)},
        "stream_state": {"token_ids": [], "text": ""},
    }
    continuation = DecodeContinuation.decode(
        continuation_from_req(request, "transfer", adapter.build_state).encode()
    )
    assert continuation.stage_payload["request"]["inputs"] is None
    assert not continuation.stage_payload["data"].get("thinker_inputs")
    assert not continuation.stage_payload["data"].get("encoder_outs")
    pool = ReqToTokenPool(
        size=1, max_context_len=32, device="cpu", enable_memory_saver=False
    )
    restored = req_from_continuation(
        continuation,
        make_allocation(),
        req_to_token_pool=pool,
        state_restorer=adapter.restore_state,
    )
    assert restored.priority == priority
    assert restored.tokenizer is tokenizer
    assert restored.omni_model_inputs is None
    assert restored.omni_data.req is restored
    assert list(restored.output_ids) == [42]
    _, result_adapter = make_thinker_scheduler_adapters(
        tokenizer=tokenizer, vocab_size=128
    )
    original = result_adapter(request.omni_data)
    resumed = result_adapter(restored.omni_data)
    assert resumed.data["thinker_out"] == original.data["thinker_out"]
    assert build_text_usage(resumed.data) == build_text_usage(original.data)
    pool.free(restored)
    assert pool.available_size() == 1


@pytest.mark.parametrize("modalities", [["audio"], ["text", "audio"]])
def test_ming_rejects_speech_handoff(modalities: list[str]) -> None:
    request = prefill_req()
    request.omni_data.stage_payload.request.metadata["output_modalities"] = modalities
    adapter = MingThinkerPDAdapter(tokenizer=Mock(spec=PreTrainedTokenizerBase))
    with pytest.raises(ValueError, match="text output only"):
        adapter.build_state(request)


def test_ming_rejects_unknown_resume_schema() -> None:
    request = prefill_req()
    adapter = MingThinkerPDAdapter(tokenizer=Mock(spec=PreTrainedTokenizerBase))
    with pytest.raises(ValueError, match="resume state"):
        adapter.restore_state(request, request.omni_data, {"schema": "other"})


def test_ming_explicit_pd_topology() -> None:
    config = MingOmniPDPipelineConfig(model_path="local-model")
    stages = {stage.name: stage for stage in config.stages}
    assert stages["mm_aggregate"].next == "thinker"
    assert stages["thinker"].next == "thinker_decode"
    assert stages["thinker_decode"].next == "decode"
    assert stages["thinker"].stream_to == []
    assert stages["thinker_decode"].stream_to == ["decode"]
    assert stages["decode"].terminal
    for name in ("thinker", "thinker_decode"):
        assert stages[name].engine.disable_radix_cache
        assert stages[name].engine.page_size == 1
        assert stages[name].tp_size == 1
