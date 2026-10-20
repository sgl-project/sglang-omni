# SPDX-License-Identifier: Apache-2.0
"""Model continuation and explicit PD topology contracts."""

from unittest.mock import Mock

import pytest
import torch
from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from transformers import PreTrainedTokenizerBase

from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.models.qwen3_omni.config import (
    Qwen3OmniPDPipelineConfig,
    build_server_warmup_request,
)
from sglang_omni.models.qwen3_omni.pd import QwenThinkerPDAdapter
from sglang_omni.scheduling.pd_utils import (
    DecodeContinuation,
    continuation_from_req,
    req_from_continuation,
)
from tests.unit_test.pipeline.test_pd_utils import make_allocation, prefill_req


@pytest.mark.parametrize("priority", [None, -7, 0, 7])
def test_model_continuation_keeps_priority_without_media(priority: int | None) -> None:
    tokenizer = Mock(spec=PreTrainedTokenizerBase)
    adapter = QwenThinkerPDAdapter(tokenizer=tokenizer)
    request = prefill_req()
    request.priority = priority
    request.omni_data.stage_payload.request.inputs = torch.ones(4)
    request.omni_data.stage_payload.data = {
        "prompt": {"input_ids": torch.tensor([10, 11, 12])},
        "thinker_inputs": {"model_inputs": {"image_embeds": torch.ones(4)}},
        "stream_state": {"token_ids": [], "text": ""},
    }
    request.multimodal_inputs = MultimodalInputs(mm_items=[])
    request.multimodal_inputs.mrope_position_delta = torch.tensor([[5]])
    continuation = DecodeContinuation.decode(
        continuation_from_req(request, "transfer", adapter.build_state).encode()
    )
    assert continuation.stage_payload["request"]["inputs"] is None
    assert not continuation.stage_payload["data"].get("thinker_inputs")
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
    assert list(restored.output_ids) == [42]
    assert restored.omni_data.req is restored
    assert restored.multimodal_inputs.mrope_position_delta.tolist() == [[5]]
    pool.free(restored)
    assert pool.available_size() == 1


@pytest.mark.parametrize(
    "position_delta",
    [
        [[1, 2]],
        [[float("nan")]],
        [[float("inf")]],
        [[1e100]],
        [[2**63]],
        [[True]],
        [True],
        "5",
    ],
)
def test_qwen_rejects_invalid_position_delta(position_delta: object) -> None:
    request = prefill_req()
    adapter = QwenThinkerPDAdapter(tokenizer=Mock(spec=PreTrainedTokenizerBase))
    with pytest.raises(ValueError, match="mRoPE"):
        adapter.restore_state(
            request,
            request.omni_data,
            {
                "schema": adapter.schema,
                "mrope_position_delta": position_delta,
            },
        )


def test_pd_topology_keeps_detokenizer_separate() -> None:
    configuration = Qwen3OmniPDPipelineConfig(model_path="local-model")
    stages = {stage.name: stage for stage in configuration.stages}
    assert stages["mm_aggregate"].next == "thinker"
    assert stages["thinker"].next == "thinker_decode"
    assert stages["thinker"].factory.pd_role == "prefill"
    assert stages["thinker_decode"].factory.pd_role == "decode"
    assert stages["thinker_decode"].next == "decode"
    assert stages["decode"].terminal
    assert not stages["thinker"].stream_to
    assert stages["thinker_decode"].stream_to == ["decode"]
    assert stages["thinker"].process != stages["thinker_decode"].process
    for name in ("thinker", "thinker_decode"):
        assert stages[name].engine.disable_radix_cache
        assert stages[name].engine.page_size == 1
        assert stages[name].tp_size == 1


@pytest.mark.parametrize("context_length", [None, 16384])
def test_pd_context_uses_checkpoint_default_or_explicit_override(
    context_length: int | None,
) -> None:
    configuration = Qwen3OmniPDPipelineConfig(model_path="local-model")
    for stage in configuration.stages:
        if stage.name in ("preprocessing", "thinker", "thinker_decode"):
            assert "max_seq_len" not in stage.factory.model_dump(exclude_unset=True)
            if context_length is not None:
                stage.factory = stage.factory.model_copy(
                    update={"max_seq_len": context_length}
                )
            resolved = resolve_stage_factory_args(stage, configuration, gpu_id=0)
            assert resolved.get("max_seq_len") == context_length


def test_qwen_pd_inherits_multimodal_text_warmup() -> None:
    configuration = Qwen3OmniPDPipelineConfig(model_path="local-model")
    assert configuration.server_warmup_request_factory.endswith(
        "build_server_warmup_request"
    )
    request = build_server_warmup_request(configuration)
    assert request.modalities == ["text"]
    assert request.talker_max_new_tokens is None
    assert {part["type"] for part in request.messages[0].content} == {
        "image_url",
        "input_audio",
        "text",
    }
