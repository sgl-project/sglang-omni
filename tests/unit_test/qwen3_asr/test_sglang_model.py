from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang_omni.models.qwen3_asr.sglang_model import Qwen3ASRForConditionalGeneration


class _RecordingAudioTower(nn.Module):
    dtype = torch.float32

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.empty(0))
        self.seen_features: torch.Tensor | None = None
        self.seen_lengths: torch.Tensor | None = None

    def forward(
        self,
        input_features: torch.Tensor,
        *,
        feature_lens: torch.Tensor,
    ) -> SimpleNamespace:
        self.seen_features = input_features
        self.seen_lengths = feature_lens
        return SimpleNamespace(last_hidden_state=input_features)


def test_get_audio_feature_preserves_masks_in_mixed_batch() -> None:
    tower = _RecordingAudioTower()
    model = SimpleNamespace(audio_tower=tower)
    items = [
        SimpleNamespace(
            feature=torch.tensor([[[1.0, 2.0, 90.0, 91.0]]]),
            feature_attention_mask=torch.tensor([[1, 1, 0, 0]]),
        ),
        SimpleNamespace(
            feature=torch.tensor([[[3.0, 4.0]]]),
            feature_attention_mask=None,
        ),
    ]

    output = Qwen3ASRForConditionalGeneration.get_audio_feature(model, items)

    assert torch.equal(tower.seen_lengths, torch.tensor([2, 2]))
    assert torch.equal(tower.seen_features, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    assert torch.equal(output, tower.seen_features)


def test_get_audio_feature_rejects_mismatched_mask_shape() -> None:
    model = SimpleNamespace(audio_tower=_RecordingAudioTower())
    items = [
        SimpleNamespace(
            feature=torch.ones((1, 2, 4)),
            feature_attention_mask=torch.ones((1, 3), dtype=torch.long),
        )
    ]

    with pytest.raises(ValueError, match="feature_attention_mask shape"):
        Qwen3ASRForConditionalGeneration.get_audio_feature(model, items)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_get_audio_feature_normalizes_cpu_masks_for_cuda_features() -> None:
    tower = _RecordingAudioTower().cuda()
    model = SimpleNamespace(audio_tower=tower)
    items = [
        SimpleNamespace(
            feature=torch.tensor([[[1.0, 2.0, 90.0, 91.0]]], device="cuda"),
            feature_attention_mask=torch.tensor([[1, 1, 0, 0]]),
        ),
        SimpleNamespace(
            feature=torch.tensor([[[3.0, 4.0]]], device="cuda"),
            feature_attention_mask=None,
        ),
    ]

    Qwen3ASRForConditionalGeneration.get_audio_feature(model, items)

    assert tower.seen_lengths is not None
    assert tower.seen_lengths.device.type == "cuda"
    assert torch.equal(tower.seen_lengths.cpu(), torch.tensor([2, 2]))
    assert tower.seen_features is not None
    assert torch.equal(
        tower.seen_features.cpu(),
        torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
    )
