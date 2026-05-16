# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.fishaudio_s2_pro.config import S2ProPipelineConfig
from sglang_omni.models.fishaudio_s2_pro.payload_types import S2ProState
from sglang_omni.models.fishaudio_s2_pro.request_builders import (
    apply_tts_result,
    build_sglang_tts_request,
    make_tts_scheduler_adapters,
)
from sglang_omni.models.fishaudio_s2_pro.tokenizer import (
    Reference,
    S2ProTokenizerAdapter,
)
from tests.unit_test.fixtures.fish_fakes import (
    FakeFishTokenizer,
    make_s2pro_payload,
    make_s2pro_state,
)


@pytest.fixture(autouse=True)
def fast_sampling_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.normalize",
        lambda self, tokenizer: None,
    )
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
        lambda self, vocab_size: None,
    )


def test_fish_config_state_and_tokenizer_prompt_contracts() -> None:
    """Preserves S2-Pro topology, state tensor round-trip, and prompt VQ layout."""
    config = S2ProPipelineConfig(model_path="model")
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "tts_engine",
        "vocoder",
    ]
    assert config.terminal_stages == ["vocoder"]
    assert config.gpu_placement == {"tts_engine": 0, "vocoder": 0}

    state = S2ProState(
        input_ids=torch.tensor([1, 2, 3]),
        vq_mask_tokens=torch.tensor([False, True, False]),
        vq_parts=[torch.tensor([[10, 11], [20, 21]])],
        output_codes=torch.tensor([[100, 101], [1, 2], [3, 4]]),
    )
    restored = S2ProState.from_dict(state.to_dict())
    assert restored.input_ids == [1, 2, 3]
    assert torch.equal(restored.vq_parts[0], torch.tensor([[10, 11], [20, 21]]))
    assert torch.equal(
        restored.output_codes, torch.tensor([[100, 101], [1, 2], [3, 4]])
    )

    tokenizer = FakeFishTokenizer()
    adapter = S2ProTokenizerAdapter(tokenizer)
    prompt = adapter.build_prompt(
        "target",
        references=[
            Reference(
                audio_bytes=b"",
                text="ref",
                vq_codes=torch.tensor([[0, 1], [10, 11]], dtype=torch.long),
            )
        ],
        num_codebooks=2,
        speaker="alice",
    )
    assert adapter.eos_token_ids == [99]
    assert prompt["vq_mask_tokens"].dtype == torch.bool
    assert prompt["vq_mask_tokens"].sum().item() == 2
    assert torch.equal(prompt["vq_parts"][0], torch.tensor([[0, 1], [10, 11]]))
    assert any("<|speaker:alice|>target" in text for text in tokenizer.encoded_texts)


def test_fish_tts_request_and_result_adapters_preserve_tensor_contracts() -> None:
    """Preserves TTS request tensor fields and result adapter output-code shape."""
    tokenizer = FakeFishTokenizer()
    state = make_s2pro_state(
        input_ids=[10, 11, 12],
        vq_mask_tokens=[False, True, True],
        vq_parts=[[[1, 2], [3, 4]]],
        max_new_tokens=6,
        temperature=0.6,
    )

    req_data = build_sglang_tts_request(state, tokenizer, request_id="req-1")
    assert torch.equal(req_data.input_ids, torch.tensor([10, 11, 12]))
    assert req_data.vq_mask_tokens.dtype == torch.bool
    assert torch.equal(req_data.vq_parts[0], torch.tensor([[1, 2], [3, 4]]))
    assert req_data.req.eos_token_ids == {99}

    req_data.output_codes = [
        torch.tensor([[100], [1], [2]], dtype=torch.long),
        torch.tensor([[101], [3], [4]], dtype=torch.long),
    ]
    apply_tts_result(state, req_data)
    assert torch.equal(
        state.output_codes,
        torch.tensor([[100, 101], [1, 3], [2, 4]], dtype=torch.long),
    )
    assert state.prompt_tokens == 3
    assert state.completion_tokens == 2

    payload = make_s2pro_payload(request_id="req-2")
    request_builder, result_adapter = make_tts_scheduler_adapters(tokenizer=tokenizer)
    adapted = request_builder(payload)
    adapted.output_codes = [torch.tensor([[100], [1], [2]], dtype=torch.long)]
    result_payload = result_adapter(adapted)
    assert adapted.stage_payload is payload
    assert result_payload.request is payload.request
    assert result_payload.data["output_codes"] == [[100], [1], [2]]
