# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for Audio8's native two-part generation step."""

from collections import deque
from types import SimpleNamespace

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput

from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.models.audio8_tts.fast_ar import CodebookSampler
from sglang_omni.models.audio8_tts.hf_config import ArkttsConfig
from sglang_omni.models.audio8_tts.model_runner import (
    Audio8ModelRunner,
    Audio8RequestData,
)
from sglang_omni.scheduling.types import SchedulerRequest
from sglang_omni.vendor.sglang.core import GenerationBatchResult


class CodebookModel:
    """A model boundary with visible dependence on the selected row and token."""

    def __init__(self) -> None:
        self.config: ArkttsConfig = ArkttsConfig(
            semantic_begin_id=20, semantic_end_id=23, eos_token_id=2
        )
        self.decode_input_embeds: torch.Tensor | None = None

    def generate_codebooks(
        self,
        hidden_states: torch.Tensor,
        semantic_token_ids: torch.Tensor,
        sample_codebook: CodebookSampler,
    ) -> torch.Tensor:
        return torch.stack(
            (semantic_token_ids - 20, sample_codebook(hidden_states)), dim=1
        )

    def embed_tokens(
        self, input_ids: torch.Tensor, codebook_ids: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat((input_ids.unsqueeze(1), codebook_ids), dim=1)


def test_current_frames_follow_their_requests_through_decode_and_retraction() -> None:
    """A fast-AR frame must use the current sample and survive batch reordering."""
    runner = object.__new__(Audio8ModelRunner)
    runner.model = CodebookModel()
    first = Audio8RequestData(
        req=SimpleNamespace(inflight_middle_chunks=0),
        prompt_codes=torch.tensor([[8, 0, 0]]),
        semantic_history_token_ids=deque(maxlen=10),
    )
    second = Audio8RequestData(
        req=SimpleNamespace(inflight_middle_chunks=0),
        prompt_codes=torch.tensor([[9, 0, 0]]),
        semantic_history_token_ids=deque(maxlen=10),
    )
    first_request = SchedulerRequest("first", data=first)
    second_request = SchedulerRequest("second", data=second)
    logits = torch.full((2, 40), -10.0)
    logits[0, 21] = logits[1, 23] = 5.0
    logits[:, 30] = 100.0
    prefill_result = GenerationBatchResult(
        logits_output=LogitsProcessorOutput(
            next_token_logits=logits,
            hidden_states=torch.tensor([[1.0, 4.0], [9.0, 3.0]]),
        )
    )
    runner.post_prefill(
        prefill_result,
        None,
        SimpleNamespace(is_prefill_only=False),
        [first_request, second_request],
    )
    assert prefill_result.next_token_ids.tolist() == [21, 23]
    assert first.generated_codec_frames[0].tolist() == [1, 1]
    assert second.generated_codec_frames[0].tolist() == [3, 0]

    decode_batch = SimpleNamespace(input_ids=torch.tensor([23, 21]))
    runner.before_decode(decode_batch, None, [second_request, first_request])
    assert runner.model.decode_input_embeds.tolist() == [[23, 3, 0], [21, 1, 1]]

    decode_logits = torch.full((2, 40), -10.0)
    decode_logits[0, 2] = decode_logits[1, 20] = 5.0
    decode_result = GenerationBatchResult(
        logits_output=LogitsProcessorOutput(
            next_token_logits=decode_logits,
            hidden_states=torch.tensor([[9.0, 1.0], [10.0, 9.0]]),
        )
    )
    runner.post_decode(decode_result, None, None, [second_request, first_request])
    assert decode_result.next_token_ids.tolist() == [2, 20]
    assert len(second.generated_codec_frames) == 1
    assert first.generated_codec_frames[-1].tolist() == [0, 0]

    replay_batch = SimpleNamespace(
        input_ids=torch.tensor([8, 21, 20]),
        extend_prefix_lens_cpu=[0],
        extend_seq_lens_cpu=[3],
        replace_embeds=None,
    )
    runner.before_prefill(replay_batch, None, [first_request])
    assert get_omni_prefill_inputs(replay_batch).input_embeds.tolist() == [
        [8, 0, 0],
        [21, 1, 1],
        [20, 0, 0],
    ]
