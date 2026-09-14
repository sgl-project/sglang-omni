# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect

import pytest
import torch

from sglang_omni.models.auk.config import ENGINE_STAGE, AuKPipelineConfig
from sglang_omni.models.auk.dit import AuKDit
from sglang_omni.models.auk.flow_matching import AuKFlowMatching, AuKSampleItem
from sglang_omni.models.auk.stages import create_auk_engine_executor


def _flow() -> AuKFlowMatching:
    torch.manual_seed(7)
    flow = AuKFlowMatching(
        AuKDit(
            dim=16,
            heads=1,
            dim_head=16,
            latent_dim=4,
            text_hidden_dim=8,
            num_layers=1,
            num_single_layers=1,
            dropout=0.0,
        ),
        num_llm_layers=2,
    ).eval()
    for parameter in flow.parameters():
        torch.nn.init.uniform_(parameter, -0.2, 0.2)
    return flow


def _item(*, text: int = 3, frames: int = 5, ref: int = 2, seed: int = 11):
    return AuKSampleItem(
        conditioning=torch.randn(text, 8),
        text_mask=torch.ones(text, dtype=torch.bool),
        target_frames=frames,
        ref_latent=torch.randn(ref, 4) if ref else None,
        seed=seed,
        ref_length=ref,
    )


@pytest.mark.parametrize("cfg_strength", [0.0, 2.0])
@pytest.mark.parametrize("ref", [0, 2])
@pytest.mark.parametrize("shape", [(3, 5), (4, 7)])
def test_cache_preserves_exact_cpu_output(cfg_strength, ref, shape):
    flow = _flow()
    item = _item(text=shape[0], frames=shape[1], ref=ref)

    baseline = flow.sample(
        item,
        steps=3,
        cfg_strength=cfg_strength,
        cache_reference_audio_embedding=False,
    )
    cached = flow.sample(
        item,
        steps=3,
        cfg_strength=cfg_strength,
        cache_reference_audio_embedding=True,
    )

    assert torch.equal(cached, baseline)


def test_cfg_cache_reduces_only_reference_embedding_calls(monkeypatch):
    flow = _flow()
    item = _item(frames=5, ref=2)
    counts = {"dynamic": 0, "reference": 0}
    original = flow.transformer.audio_embed._embed

    def counted(value, mask=None):
        kind = "dynamic" if value.shape[1] == item.target_frames else "reference"
        counts[kind] += 1
        return original(value, mask=mask)

    monkeypatch.setattr(flow.transformer.audio_embed, "_embed", counted)
    flow.sample(
        item,
        steps=32,
        cfg_strength=2.0,
        cache_reference_audio_embedding=True,
    )

    assert counts == {"dynamic": 64, "reference": 2}


def test_first_step_populates_and_next_step_reuses_reference_tensors(monkeypatch):
    dit = _flow().transformer
    inputs = dict(
        text=torch.randn(1, 3, 8),
        time=torch.tensor(0.0),
        ref=torch.randn(1, 2, 4),
        cfg_infer=True,
        cache=True,
        cache_reference_audio_embedding=True,
    )
    reference_calls = 0
    original = dit.audio_embed._embed

    def counted(value, mask=None):
        nonlocal reference_calls
        if value.shape[1] == 2:
            reference_calls += 1
        return original(value, mask=mask)

    monkeypatch.setattr(dit.audio_embed, "_embed", counted)
    dit(x=torch.randn(1, 5, 4), **inputs)
    cached = (dit.reference_audio_cond, dit.reference_audio_uncond)
    assert reference_calls == 2
    assert all(value is not None for value in cached)

    dit(x=torch.randn(1, 5, 4), **inputs)
    assert reference_calls == 2
    assert cached[0] is dit.reference_audio_cond
    assert cached[1] is dit.reference_audio_uncond


def test_cache_is_cleared_after_each_shape_and_after_exception(monkeypatch):
    flow = _flow()
    first = _item(text=3, frames=5, ref=2)
    second = _item(text=4, frames=7, ref=3, seed=12)

    flow.sample(
        first,
        steps=2,
        cfg_strength=2.0,
        cache_reference_audio_embedding=True,
    )
    assert flow.transformer.reference_audio_cond is None
    assert flow.transformer.reference_audio_uncond is None
    flow.sample(
        second,
        steps=2,
        cfg_strength=2.0,
        cache_reference_audio_embedding=True,
    )
    assert flow.transformer.reference_audio_cond is None
    assert flow.transformer.reference_audio_uncond is None

    original = flow.transformer.forward

    def fail_after_forward(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected failure")

    monkeypatch.setattr(flow.transformer, "forward", fail_after_forward)
    with pytest.raises(RuntimeError, match="injected failure"):
        flow.sample(
            first,
            steps=2,
            cfg_strength=2.0,
            cache_reference_audio_embedding=True,
        )
    assert flow.transformer.reference_audio_cond is None
    assert flow.transformer.reference_audio_uncond is None

    monkeypatch.setattr(flow.transformer, "forward", original)
    assert torch.isfinite(
        flow.sample(
            second,
            steps=2,
            cfg_strength=2.0,
            cache_reference_audio_embedding=True,
        )
    ).all()


def test_cache_is_default_false_and_plumbed_to_engine_factory():
    config = AuKPipelineConfig(model_path="tencent/AuK")
    engine = next(stage for stage in config.stages if stage.name == ENGINE_STAGE)
    assert engine.factory.cache_reference_audio_embedding is False
    parameter = inspect.signature(create_auk_engine_executor).parameters[
        "cache_reference_audio_embedding"
    ]
    assert parameter.default is False
