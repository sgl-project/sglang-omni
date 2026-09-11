# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.auk.flow_matching import fuse_hidden_states


class RecordingTransformer(torch.nn.Module):
    latent_dim = 8

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return torch.zeros_like(kwargs["x"])

    def clear_cache(self):
        pass


def test_fusion_matches_upstream_layerwise_normalization():
    torch.manual_seed(42)
    hidden = torch.randn(2, 5, 7, 16)
    weights = torch.randn(4)
    scale = torch.tensor([1.5])
    normalized = torch.stack(
        [F.layer_norm(layer, [16]) for layer in hidden[:, 1:].unbind(1)], dim=1
    )
    expected = (normalized * weights.softmax(0)[None, :, None, None]).sum(1) * scale
    torch.testing.assert_close(
        fuse_hidden_states(hidden, weights, scale), expected, rtol=0, atol=0
    )


@pytest.mark.parametrize("cfg_strength", [0.0, 2.0])
def test_batch_matches_individual_sampling_with_different_lengths(cfg_strength):
    from sglang_omni.models.auk.dit import AuKDit
    from sglang_omni.models.auk.flow_matching import AuKFlowMatching, AuKSampleItem

    torch.manual_seed(42)
    flow = AuKFlowMatching(
        AuKDit(
            dim=32,
            heads=2,
            dim_head=16,
            latent_dim=8,
            text_hidden_dim=16,
            num_layers=1,
            num_single_layers=1,
        ),
        num_llm_layers=2,
    ).eval()
    for parameter in flow.parameters():
        torch.nn.init.uniform_(parameter, -0.2, 0.2)
    items = [
        AuKSampleItem(
            torch.randn(text, 16),
            torch.ones(text, dtype=torch.bool),
            frames,
            torch.randn(ref, 8) if ref else None,
            seed=seed,
            ref_length=max(0, ref - 1),
        )
        for text, frames, ref, seed in [(5, 19, 4, 1), (8, 11, 7, 2), (3, 15, 0, 3)]
    ]
    sampling = dict(steps=3, cfg_strength=cfg_strength)
    rng = torch.random.get_rng_state()
    expected = [flow.sample(item, **sampling) for item in items]
    actual = flow.sample_batch(items, **sampling)
    assert torch.equal(torch.random.get_rng_state(), rng)
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-6)


def test_bf16_backbone_integrates_in_fp32_and_tracks_fp32_backbone():
    from sglang_omni.models.auk.dit import AuKDit
    from sglang_omni.models.auk.flow_matching import AuKFlowMatching, AuKSampleItem

    torch.manual_seed(7)
    flow = AuKFlowMatching(
        AuKDit(
            dim=32,
            heads=2,
            dim_head=16,
            latent_dim=8,
            text_hidden_dim=16,
            num_layers=1,
            num_single_layers=1,
        ),
        num_llm_layers=2,
    ).eval()
    for parameter in flow.parameters():
        torch.nn.init.uniform_(parameter, -0.2, 0.2)
    items = [
        AuKSampleItem(
            torch.randn(text, 16),
            torch.ones(text, dtype=torch.bool),
            frames,
            torch.randn(ref, 8) if ref else None,
            seed=seed,
            ref_length=ref,
        )
        for text, frames, ref, seed in [(5, 19, 4, 1), (3, 15, 0, 3)]
    ]
    sampling = dict(steps=3, cfg_strength=2.0)
    expected = flow.sample_batch(items, **sampling)
    flow.transformer.to(torch.bfloat16)
    actual = flow.sample_batch(items, **sampling)
    for output, reference in zip(actual, expected):
        assert output.dtype == torch.float32
        assert torch.isfinite(output).all()
        cosine = torch.nn.functional.cosine_similarity(
            output.flatten(), reference.flatten(), dim=0
        )
        assert cosine > 0.99, cosine


def test_singleton_elision_passes_no_masks_to_transformer():
    from sglang_omni.models.auk.flow_matching import AuKFlowMatching, AuKSampleItem

    transformer = RecordingTransformer()
    flow = AuKFlowMatching(transformer, num_llm_layers=2)
    item = AuKSampleItem(
        conditioning=torch.zeros(4, 16),
        text_mask=None,
        target_frames=5,
        ref_latent=torch.zeros(3, 8),
        ref_length=3,
    )
    flow.sample_batch([item], steps=1, cfg_strength=0, elide_singleton_masks=True)
    call = transformer.calls[0]
    assert call["mask"] is None
    assert call["c_mask"] is None
    assert call["ref_mask"] is None
    assert call["all_valid_masks"] is True


def test_enabled_multi_request_batch_retains_padded_masks():
    from sglang_omni.models.auk.flow_matching import AuKFlowMatching, AuKSampleItem

    transformer = RecordingTransformer()
    flow = AuKFlowMatching(transformer, num_llm_layers=2)
    items = [
        AuKSampleItem(
            conditioning=torch.zeros(text, 16),
            text_mask=torch.ones(text, dtype=torch.bool),
            target_frames=frames,
            ref_latent=torch.zeros(reference, 8),
            ref_length=reference,
        )
        for text, frames, reference in ((4, 5, 3), (2, 3, 1))
    ]
    flow.sample_batch(items, steps=1, cfg_strength=0, elide_singleton_masks=True)
    call = transformer.calls[0]
    assert call["mask"].tolist() == [[True] * 5, [True] * 3 + [False] * 2]
    assert call["c_mask"].tolist() == [[True] * 4, [True] * 2 + [False] * 2]
    assert call["ref_mask"].tolist() == [[True] * 3, [True, False, False]]
    assert call["all_valid_masks"] is False


def test_real_dit_singleton_elision_matches_all_valid_masks():
    from sglang_omni.models.auk.dit import AuKDit
    from sglang_omni.models.auk.flow_matching import AuKFlowMatching, AuKSampleItem

    torch.manual_seed(42)
    flow = AuKFlowMatching(
        AuKDit(
            dim=32,
            heads=2,
            dim_head=16,
            latent_dim=8,
            text_hidden_dim=16,
            num_layers=1,
            num_single_layers=1,
        ),
        num_llm_layers=2,
    ).eval()
    for parameter in flow.parameters():
        torch.nn.init.uniform_(parameter, -0.2, 0.2)
    conditioning = torch.randn(4, 16)
    reference = torch.randn(3, 8)
    baseline = AuKSampleItem(
        conditioning=conditioning,
        text_mask=torch.ones(4, dtype=torch.bool),
        target_frames=5,
        ref_latent=reference,
        ref_length=3,
        seed=7,
    )
    elided = AuKSampleItem(
        conditioning=conditioning,
        text_mask=None,
        target_frames=5,
        ref_latent=reference,
        ref_length=3,
        seed=7,
    )
    expected = flow.sample_batch([baseline], steps=2, cfg_strength=2)[0]
    actual = flow.sample_batch(
        [elided], steps=2, cfg_strength=2, elide_singleton_masks=True
    )[0]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
