# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.auk.flow_matching import fuse_hidden_states


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


class ShapePadding:
    """The step-graph runner's padding contract, with nothing to replay.

    It pads to one declared shape, and only for a batch that shape covers on
    every axis, which is the runner's own lookup. A bind returning None is the
    runner's fallback for a shape it holds no graph for, so the padded batch
    still runs through the eager backbone.
    """

    batch = 8
    shape = (32, 16, 8)

    def pad_lengths(self, *, frames, ref, text, batch):
        covers = (
            batch <= self.batch
            and frames <= self.shape[0]
            and ref <= self.shape[1]
            and text <= self.shape[2]
        )
        return self.shape if covers else None

    def bind(self, *args, **kwargs):
        return None


@pytest.mark.parametrize("count", [1, 3])
def test_shape_padding_does_not_change_the_sampled_latents(monkeypatch, count):
    """Padded rows must reach the backbone and contribute nothing.

    The masks carry the real lengths and the rope positions are built from them,
    so rounding target frames, reference frames, and text tokens up to a declared
    capture shape is the same trajectory the unpadded batch integrates.
    """
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
            ref_length=ref,
        )
        for text, frames, ref, seed in [(5, 19, 4, 1), (8, 11, 7, 2), (3, 15, 0, 3)][
            :count
        ]
    ]
    widths = []
    original = type(flow.transformer).forward

    def forward(self, x, *args, **kwargs):
        widths.append(x.shape[1])
        return original(self, x, *args, **kwargs)

    monkeypatch.setattr(type(flow.transformer), "forward", forward)

    sampling = dict(steps=3, cfg_strength=2.0)
    expected = flow.sample_batch(items, **sampling)
    unpadded = set(widths)
    widths.clear()
    actual = flow.sample_batch(items, **sampling, step_graph=ShapePadding())

    assert unpadded == {max(item.target_frames for item in items)}
    assert set(widths) == {32}
    for output, reference in zip(actual, expected):
        assert output.shape == reference.shape
        torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-6)


def test_a_batch_the_runner_declines_is_not_padded(monkeypatch):
    """Padding is only worth its waste on a batch that gets a graph."""
    from sglang_omni.models.auk.dit import AuKDit
    from sglang_omni.models.auk.flow_matching import AuKFlowMatching, AuKSampleItem

    torch.manual_seed(0)
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
    items = [
        AuKSampleItem(
            torch.randn(5, 16), torch.ones(5, dtype=torch.bool), frames, seed=index
        )
        for index, frames in enumerate((19, 11, 15))
    ]
    widths = []
    original = type(flow.transformer).forward

    def forward(self, x, *args, **kwargs):
        widths.append(x.shape[1])
        return original(self, x, *args, **kwargs)

    monkeypatch.setattr(type(flow.transformer), "forward", forward)
    declining = ShapePadding()
    declining.batch = 2
    flow.sample_batch(items, steps=2, cfg_strength=2.0, step_graph=declining)
    assert set(widths) == {19}
