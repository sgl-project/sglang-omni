# SPDX-License-Identifier: Apache-2.0
"""Audio-conditioned MTP inputs and prompt replay."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from torch import nn

from sglang_omni.models.ming3_asr import mtp
from sglang_omni.models.ming3_asr.mtp import Ming3ASRMTP
from sglang_omni.modules.ming_audio.thinker import Ming3AudioMTP


def test_asr_mtp_loads_fine_tuned_prediction_weights_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ASR draft replaces the prediction block and retains shared verifier weights."""
    prediction_name = "model.layers.24.eh_proj.weight"
    fine_tuned = torch.full((2, 4), 3.0)
    checkpoint = tmp_path / "model.safetensors"
    save_file({prediction_name: fine_tuned}, checkpoint)
    download = Mock(return_value=str(checkpoint))
    monkeypatch.setattr(mtp, "hf_hub_download", download)
    shared = {
        "model.word_embeddings.weight": torch.ones(4, 2),
        "lm_head.weight": torch.ones(4, 2),
        "model.layers.23.input_layernorm.weight": torch.ones(2),
    }
    loaded: dict[str, torch.Tensor] = {}

    def load_native(
        self: Ming3AudioMTP, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> None:
        loaded.update(weights)

    monkeypatch.setattr(Ming3AudioMTP, "load_weights", load_native)
    draft = Ming3ASRMTP.__new__(Ming3ASRMTP)
    nn.Module.__init__(draft)
    draft.config = SimpleNamespace(num_hidden_layers=24)
    draft.load_weights(iter([*shared.items(), (prediction_name, torch.zeros(2, 4))]))
    assert set(loaded) == {*shared, prediction_name}
    assert all(loaded[name] is weight for name, weight in shared.items())
    torch.testing.assert_close(loaded[prediction_name], fine_tuned)
    download.assert_called_once_with(
        mtp.MTP_MODEL_ID,
        "model.safetensors",
        revision=mtp.MTP_MODEL_REVISION,
    )


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
