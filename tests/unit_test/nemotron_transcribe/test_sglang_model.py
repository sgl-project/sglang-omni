# SPDX-License-Identifier: Apache-2.0
"""The decoder forward without speculative decoding, on tiny CPU modules."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.managers import mm_schedule, mm_utils
from sglang.srt.managers.mm_utils import (
    general_mm_embed_routine,
    init_mm_embedding_cache,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from torch import nn

from sglang_omni.models.nemotron_transcribe.sglang_model import (
    NemotronTranscribeForConditionalGeneration,
)

HIDDEN_SIZE = 4
VOCAB_SIZE = 16
PROMPT_TOKENS = 2
AUDIO_TOKENS = 3
WAVEFORM = torch.tensor([0.5, -1.0, 2.0])
PRE_LM_EMBEDDINGS = (
    torch.arange(AUDIO_TOKENS * HIDDEN_SIZE, dtype=torch.bfloat16).reshape(
        AUDIO_TOKENS, HIDDEN_SIZE
    )
    / 8
)


@dataclass(kw_only=True)
class LanguageModelCall:
    input_ids: torch.Tensor | None
    input_embeds: torch.Tensor
    positions: torch.Tensor


class RecordingLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16)
        self.pp_group = SimpleNamespace(is_first_rank=True)
        self.calls: list[LanguageModelCall] = []

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor | None,
        forward_batch: SimpleNamespace,
        input_embeds: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        self.calls.append(
            LanguageModelCall(
                input_ids=input_ids,
                input_embeds=input_embeds.clone(),
                positions=positions,
            )
        )
        return input_embeds.float().sum(dim=-1)


class WaveformPerception(nn.Module):
    """One embedding per waveform sample, so placeholder counts follow sample counts."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(1, HIDDEN_SIZE)
        self.calls = 0

    def forward(
        self, input_signal: torch.Tensor, input_signal_length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls += 1
        return self.proj(input_signal.unsqueeze(-1)), input_signal_length


def make_model() -> NemotronTranscribeForConditionalGeneration:
    torch.manual_seed(0)
    model = NemotronTranscribeForConditionalGeneration.__new__(
        NemotronTranscribeForConditionalGeneration
    )
    nn.Module.__init__(model)
    model.language_model = RecordingLanguageModel()
    model.perception = WaveformPerception()
    model.encoder_max_batch_size = 8
    return model


def make_forward_batch(audio_source: str) -> tuple[torch.Tensor, SimpleNamespace]:
    if audio_source == "decode":
        return torch.tensor([5, 6]), SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            contains_mm_inputs=lambda: False,
            mm_inputs=None,
            input_embeds=None,
        )
    else:
        pass
    audio_item = MultimodalDataItem(
        modality=Modality.AUDIO,
        hash=1234,
        feature=WAVEFORM.clone(),
        model_specific_data={"num_audio_tokens": AUDIO_TOKENS},
    )
    audio_item.set_pad_value()
    if audio_source == "pre_lm_encoder":
        audio_item.precomputed_embeddings = PRE_LM_EMBEDDINGS.clone()
        audio_item.feature = None
    else:
        pass
    audio_item.offsets = [(PROMPT_TOKENS, PROMPT_TOKENS + AUDIO_TOKENS - 1)]
    input_ids = torch.tensor([1, 2] + [audio_item.pad_value] * AUDIO_TOKENS + [3])
    mm_inputs = [MultimodalInputs(mm_items=[audio_item])]
    return input_ids, SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        contains_mm_inputs=lambda: True,
        mm_inputs=mm_inputs,
        extend_prefix_lens_cpu=[0],
        extend_seq_lens_cpu=[input_ids.numel()],
        spec_algorithm=None,
        input_embeds=None,
        mm_input_embeds=None,
    )


@pytest.fixture(autouse=True)
def single_process_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scheduler publishes server args and the embedding cache; size 0 is the stage default."""
    monkeypatch.setattr(mm_utils, "get_server_args", lambda: None)
    monkeypatch.setattr(mm_schedule, "embedding_cache", None)
    init_mm_embedding_cache(0)


@pytest.mark.parametrize(
    ("audio_source", "perception_calls"),
    [("pre_lm_encoder", 0), ("forward_encoder", 1), ("decode", 0)],
)
def test_forward_without_speculation_matches_the_general_embed_routine(
    audio_source: str, perception_calls: int
) -> None:
    """forward via prepare_mm_inputs must match the previous general_mm_embed_routine exactly."""
    routine_model = make_model()
    input_ids, forward_batch = make_forward_batch(audio_source)
    positions = torch.arange(input_ids.numel())
    routine_output = general_mm_embed_routine(
        input_ids=input_ids.clone(),
        forward_batch=forward_batch,
        language_model=routine_model.language_model,
        data_embedding_funcs={Modality.AUDIO: routine_model.get_audio_feature},
        positions=positions,
    )
    forward_model = make_model()
    input_ids, forward_batch = make_forward_batch(audio_source)
    forward_output = NemotronTranscribeForConditionalGeneration.forward(
        forward_model,
        input_ids=input_ids.clone(),
        positions=positions,
        forward_batch=forward_batch,
    )

    (routine_call,) = routine_model.language_model.calls
    (forward_call,) = forward_model.language_model.calls
    assert forward_call.input_ids is None
    assert forward_call.positions is positions
    torch.testing.assert_close(
        forward_call.input_embeds, routine_call.input_embeds, rtol=0, atol=0
    )
    torch.testing.assert_close(forward_output, routine_output, rtol=0, atol=0)
    assert forward_model.perception.calls == perception_calls
    assert routine_model.perception.calls == perception_calls
    if audio_source == "pre_lm_encoder":
        audio_embeddings = PRE_LM_EMBEDDINGS
    elif audio_source == "forward_encoder":
        with torch.no_grad():
            audio_embeddings = forward_model.perception.proj(WAVEFORM.unsqueeze(-1)).to(
                torch.bfloat16
            )
    else:
        audio_embeddings = None
    if audio_embeddings is not None:
        torch.testing.assert_close(
            forward_call.input_embeds[PROMPT_TOKENS : PROMPT_TOKENS + AUDIO_TOKENS],
            audio_embeddings,
        )
    else:
        pass
