# SPDX-License-Identifier: Apache-2.0
"""Check real Qwen position outputs across the model PD resume boundary."""
from array import array
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor import forward_batch_info
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.qwen3_omni.pd import QwenThinkerPDAdapter
from sglang_omni.scheduling.pd_utils import (
    DecodeContinuation,
    ReservedKV,
    continuation_from_req,
    req_from_continuation,
)
from tests.unit_test.pipeline.test_pd_utils import prefill_req
from tests.unit_test.qwen3_omni.test_mrope_positions import (
    audio_span,
    image_span,
    oracle_and_fast,
    video_span,
)


@pytest.mark.parametrize("modality", ["text", "audio", "image", "video"])
def test_real_qwen_delta_survives_pd(
    modality: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_grid = None
    video_grid = None
    seconds = None
    audio_lengths = None
    if modality == "image":
        tokens = [10, 11] + image_span([1, 4, 4]) + [12, 13]
        image_grid = torch.tensor([[1, 4, 4]])
    elif modality == "video":
        tokens = [10] + video_span([2, 4, 4]) + [11]
        video_grid = torch.tensor([[2, 4, 4]])
        seconds = torch.tensor([0.5])
    elif modality == "audio":
        tokens = [10] + audio_span(200) + [11]
        audio_lengths = torch.tensor([200])
    else:
        tokens = [10, 11, 12]
    oracle, actual = oracle_and_fast(
        torch.tensor([tokens]),
        image_grid_thw=image_grid,
        video_grid_thw=video_grid,
        second_per_grid_ts=seconds,
        audio_seqlens=audio_lengths,
    )
    positions, delta = actual
    assert torch.equal(oracle[0].float(), positions.float())
    assert torch.equal(oracle[1].float(), delta.float())
    request = prefill_req()
    request.origin_input_ids = array("q", tokens)
    request.origin_input_ids_unpadded = array("q", tokens)
    request.omni_data.stage_payload.data = {
        "prompt": {"input_ids": torch.tensor(tokens)}
    }
    request.multimodal_inputs = MultimodalInputs(mm_items=[])
    request.multimodal_inputs.mrope_position_delta = delta
    request.multimodal_inputs.mrope_positions = positions.squeeze(1)
    adapter = QwenThinkerPDAdapter(tokenizer=Mock(spec=PreTrainedTokenizerBase))
    payload, resume, prompt_tokens = adapter.build_state(request)
    assert prompt_tokens == tokens
    assert payload["data"]["prompt"]["input_ids"] == tokens
    continuation = continuation_from_req(request, "mrope-transfer", adapter.build_state)
    decoded = DecodeContinuation.decode(continuation.encode())
    assert decoded.multimodal_resume == resume
    pool = ReqToTokenPool(
        size=1, max_context_len=len(tokens) + 8, device="cpu", enable_memory_saver=False
    )
    allocation = ReservedKV(
        slots=torch.arange(len(tokens), dtype=torch.int64),
        page_indices=tuple(range(len(tokens))),
        seq_len=len(tokens),
    )
    restored = req_from_continuation(
        decoded,
        allocation,
        req_to_token_pool=pool,
        state_restorer=adapter.restore_state,
    )
    restored_delta = restored.multimodal_inputs.mrope_position_delta
    assert restored_delta.dtype == delta.dtype
    assert torch.equal(restored_delta, delta)

    execution = SimpleNamespace(deterministic=SimpleNamespace(rl_on_policy_target=None))
    monkeypatch.setattr(forward_batch_info, "get_exec", lambda: execution)
    decode_positions = []
    for multimodal_inputs in (request.multimodal_inputs, restored.multimodal_inputs):
        forward_batch = object.__new__(forward_batch_info.ForwardBatch)
        forward_batch.seq_lens_cpu = torch.tensor([len(tokens) + 1])
        forward_batch.seq_lens = forward_batch.seq_lens_cpu.clone()
        forward_batch._compute_mrope_positions_decode(  # noqa: leading-underscore -- pinned SGLang consumer
            SimpleNamespace(device="cpu"),
            SimpleNamespace(multimodal_inputs=[multimodal_inputs]),
        )
        decode_positions.append(forward_batch.mrope_positions)
    assert torch.equal(decode_positions[0], decode_positions[1])
    assert list(restored.origin_input_ids) == tokens
    assert list(restored.output_ids) == [42]
    pool.free(restored)
    assert pool.available_size() == 1
