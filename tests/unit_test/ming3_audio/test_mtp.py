# SPDX-License-Identifier: Apache-2.0
"""Audio-conditioned MTP inputs and prompt replay."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.models import bailing_moe_nextn
from sglang.srt.models.bailing_moe_v3 import BailingMoeV3ForCausalLM
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from torch import nn

from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.modules.ming_audio.hf_config import Ming3AudioConfig
from sglang_omni.modules.ming_audio.thinker import Ming3AudioModelRunner, Ming3AudioMTP


class DraftInputRecorder(nn.Module):
    """Replace GPU transformer computation while retaining its tensor inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding.from_pretrained(
            torch.arange(32, dtype=torch.float32).reshape(16, 2)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: SimpleNamespace,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.word_embeddings(input_ids)
        else:
            pass
        return torch.cat((input_embeds, forward_batch.spec_info.hidden_states), dim=-1)


@pytest.fixture
def draft() -> Ming3AudioMTP:
    model = Ming3AudioMTP.__new__(Ming3AudioMTP)
    nn.Module.__init__(model)
    model.model = DraftInputRecorder()
    model.lm_head = nn.Identity()
    model.logits_processor = (
        lambda input_ids, hidden_states, head, forward_batch: hidden_states
    )
    return model


def test_native_prefill_rotates_audio_and_mtp_fills_every_request_tail(
    draft: Ming3AudioMTP,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native prefill and the audio draft must agree on per-request alignment."""
    prompt_embeddings = torch.tensor(
        [
            [0, 0],
            [101, 102],
            [103, 104],
            [6, 7],
            [8, 9],
            [105, 106],
        ],
        dtype=torch.float32,
    )
    original_embeddings = prompt_embeddings.clone()
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    batch = SimpleNamespace(
        input_ids=torch.arange(6),
        forward_mode=ForwardMode.EXTEND,
        extend_lens=[3, 1, 2],
        chunked_req_next_prompt_token=None,
    )
    monkeypatch.setattr(
        ForwardBatch,
        "init_new",
        Mock(
            side_effect=lambda batch, runner, **options: SimpleNamespace(
                input_ids=batch.input_ids,
                positions=torch.arange(6),
                forward_mode=batch.forward_mode,
                extend_start_loc=torch.tensor([0, 3, 4]),
                extend_seq_lens=torch.tensor(batch.extend_lens),
                spec_info=batch.spec_info,
            )
        ),
    )

    class DraftForwardReached(Exception):
        pass

    forward = Mock(side_effect=DraftForwardReached)
    worker = EagleDraftWorker.__new__(EagleDraftWorker)
    worker.speculative_algorithm = SpeculativeAlgorithm.EAGLE
    worker.seed_dsa_topk_from_draft_extend = False
    worker.draft_runner = SimpleNamespace(canary_manager=None, forward=forward)
    with pytest.raises(DraftForwardReached):
        worker._draft_extend_for_prefill(  # noqa: leading-underscore - Native worker entry point.
            batch, hidden_states, torch.tensor([7, 8, 9]), prompt_embeddings
        )
    forward_batch = forward.call_args.args[0]
    assert forward_batch.input_ids.tolist() == [1, 2, 7, 8, 5, 9]
    result = draft(forward_batch.input_ids, forward_batch.positions, forward_batch)
    expected_embeddings = torch.tensor(
        [[101, 102], [103, 104], [14, 15], [16, 17], [105, 106], [18, 19]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(result[:, :2], expected_embeddings)
    torch.testing.assert_close(result[:, 2:], hidden_states)
    torch.testing.assert_close(prompt_embeddings, original_embeddings)


@pytest.mark.parametrize(
    "forward_mode",
    [
        ForwardMode.EXTEND,
        ForwardMode.DECODE,
        ForwardMode.DRAFT_EXTEND_V2,
        ForwardMode.IDLE,
    ],
)
def test_mtp_without_audio_uses_native_token_embeddings(
    draft: Ming3AudioMTP, forward_mode: ForwardMode
) -> None:
    token_count = 0 if forward_mode.is_idle() else 2
    input_ids = torch.arange(token_count)
    hidden_states = torch.ones(token_count, 2)
    batch = SimpleNamespace(
        mm_input_embeds=None,
        forward_mode=forward_mode,
        spec_info=SimpleNamespace(hidden_states=hidden_states),
    )

    result = draft(input_ids, torch.arange(token_count), batch)

    torch.testing.assert_close(result[:, :2], draft.model.word_embeddings(input_ids))
    torch.testing.assert_close(result[:, 2:], hidden_states)


def test_reprefill_preserves_audio_conditioning_and_respects_prefix_offsets() -> None:
    """A retracted request must reconstruct audio rows from the retained prompt."""
    runner = Ming3AudioModelRunner.__new__(Ming3AudioModelRunner)
    embeddings = nn.Embedding.from_pretrained(torch.arange(32).float().reshape(16, 2))
    runner.model = SimpleNamespace(get_input_embeddings=lambda: embeddings)
    audio_embeddings = torch.tensor([[101.0, 102.0], [103.0, 104.0]])
    conditioning = {
        "audio_positions": torch.tensor([1, 2]),
        "audio_embeddings": audio_embeddings,
    }
    requests = [
        SimpleNamespace(data=SimpleNamespace(model_inputs=conditioning)),
        SimpleNamespace(
            data=SimpleNamespace(
                model_inputs={
                    "audio_positions": torch.empty(0, dtype=torch.long),
                    "audio_embeddings": torch.empty(0, 2),
                }
            )
        ),
    ]
    for prefix_length in (0, 2, 0):
        batch = SimpleNamespace(
            input_ids=torch.tensor(list(range(prefix_length, 4)) + [5, 6]),
            extend_seq_lens_cpu=[4 - prefix_length, 2],
            extend_prefix_lens_cpu=[prefix_length, 0],
            replace_embeds=None,
        )
        runner.before_prefill(batch, None, requests)
        expected = embeddings(batch.input_ids).clone()
        for prompt_position, audio_row in zip((1, 2), audio_embeddings, strict=True):
            if prompt_position >= prefix_length:
                expected[prompt_position - prefix_length] = audio_row
            else:
                pass
        torch.testing.assert_close(
            get_omni_prefill_inputs(batch).input_embeds, expected
        )
    torch.testing.assert_close(conditioning["audio_embeddings"], audio_embeddings)
    assert conditioning["audio_positions"].tolist() == [1, 2]


def test_native_constructor_selects_v3_and_loads_only_the_mtp_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audio MTP must select the V3 decoder and its bundled checkpoint layer."""
    configuration = Ming3AudioConfig(
        text_config={
            "num_hidden_layers": 42,
            "num_nextn_predict_layers": 1,
            "short_conv_kernel_size": 4,
            "hidden_size": 4,
            "vocab_size": 16,
            "num_experts": 1,
            "num_shared_experts": 0,
        }
    )
    configuration.model_type = "bailing_hybrid"
    decoder = nn.Module()
    decoder.attention = nn.Module()
    decoder.attention.o_proj = nn.Linear(4, 4, bias=False)
    decoder.input_layernorm = nn.LayerNorm(4, bias=False)
    monkeypatch.setattr(
        bailing_moe_nextn, "BailingMoeV3DecoderLayer", lambda config, **options: decoder
    )
    monkeypatch.setattr(
        bailing_moe_nextn,
        "VocabParallelEmbedding",
        lambda size, width, **options: nn.Embedding(size, width),
    )
    monkeypatch.setattr(
        bailing_moe_nextn,
        "ParallelLMHead",
        lambda size, width, **options: nn.Linear(width, size, bias=False),
    )
    monkeypatch.setattr(
        bailing_moe_nextn,
        "ReplicatedLinear",
        lambda width, output_width, **options: nn.Linear(
            width, output_width, bias=False
        ),
    )
    monkeypatch.setattr(
        bailing_moe_nextn, "LogitsProcessor", lambda config: nn.Identity()
    )
    monkeypatch.setattr(bailing_moe_nextn, "is_dp_attention_enabled", lambda: False)
    monkeypatch.setattr(bailing_moe_nextn, "get_num_fused_shared_experts", lambda: 0)
    monkeypatch.setattr(
        bailing_moe_nextn,
        "get_parallel",
        lambda: SimpleNamespace(enable_dp_lm_head=False),
    )
    monkeypatch.setattr(
        BailingMoeV3ForCausalLM, "log_shared_experts_fusion", lambda self: None
    )

    draft = Ming3AudioMTP(configuration)
    assert draft.model.is_hybrid
    assert draft.model.decoder is decoder
    assert draft.base_load_weights_func is BailingMoeV3ForCausalLM.load_weights
    assert draft.post_load_weights_func is BailingMoeV3ForCausalLM.post_load_weights

    projection = torch.arange(32).reshape(4, 8).float()
    attention = torch.arange(16).reshape(4, 4).float()
    normalization = torch.tensor([1.0, 2.0, 3.0, 4.0])
    draft.load_weights(
        [
            ("audio.encoder.weight", torch.zeros(1)),
            ("model.layers.41.input_layernorm.weight", torch.zeros(4)),
            ("model.layers.42.eh_proj.weight", projection),
            ("model.layers.42.attention.dense.weight", attention),
            ("model.layers.42.input_layernorm.weight", normalization),
        ]
    )
    torch.testing.assert_close(draft.model.eh_proj.weight, projection)
    torch.testing.assert_close(decoder.attention.o_proj.weight, attention)
    torch.testing.assert_close(decoder.input_layernorm.weight, normalization)
