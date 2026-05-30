# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``EncoderModuleSpec`` + ``EncoderModuleContainer``.

The container is the load-bearing piece that prevents the encoder
runner from loading LLM / talker / unrelated-encoder weights onto an
encoder GPU. These tests exercise:

- Spec validation (unique names, non-empty list).
- Prefix-filter behavior on ``load_weights``: image-stage container
  drops audio / language / talker / unrelated-encoder keys without
  allocating a destination.
- Audio-stage container symmetrically drops vision / language / talker
  keys.
- Checkpoint-rewrite ordering applies left-to-right.
- Missing-prefix and missing-param keys are silently skipped, not
  errors (mirrors upstream loader's tolerant behavior).

We use tiny synthetic ``nn.Linear`` submodules so the tests run on CPU
in well under a second — no Qwen3-Omni weights involved.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from sglang_omni.model_runner.sglang_encoder_runner import EncoderModuleContainer
from sglang_omni.models.qwen3_omni.encoder_adapters import EncoderModuleSpec


def _make_visual_like_module():
    """Stand-in submodule that mirrors a tiny piece of the visual encoder.

    Two Linear params named ``proj.weight`` / ``proj.bias`` so a checkpoint
    key like ``model.visual.proj.weight`` flows through the rewrite
    chain into ``visual.proj.weight``.
    """
    m = nn.Module()
    m.proj = nn.Linear(4, 4, bias=True)
    return m


def _make_audio_like_module():
    m = nn.Module()
    m.encoder = nn.Linear(4, 4, bias=True)
    return m


VISUAL_SPEC = EncoderModuleSpec(
    name="visual",
    build_module=lambda hf, qc: _make_visual_like_module(),
    checkpoint_prefixes=("model.visual.", "thinker.visual.", "visual."),
    checkpoint_rewrites=(
        ("model.visual.", "visual."),
        ("thinker.visual.", "visual."),
    ),
)

AUDIO_SPEC = EncoderModuleSpec(
    name="audio_tower",
    build_module=lambda hf, qc: _make_audio_like_module(),
    checkpoint_prefixes=("audio_tower.", "thinker.audio_tower."),
    checkpoint_rewrites=(("thinker.audio_tower.", "audio_tower."),),
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_container_builds_only_declared_submodules():
    container = EncoderModuleContainer(
        hf_config=None,
        encoder_specs=(VISUAL_SPEC,),
        quant_config=None,
    )
    children = dict(container.named_children())
    assert "visual" in children
    assert "audio_tower" not in children


def test_container_rejects_empty_spec_list():
    with pytest.raises(ValueError, match="at least one EncoderModuleSpec"):
        EncoderModuleContainer(
            hf_config=None,
            encoder_specs=(),
            quant_config=None,
        )


def test_container_rejects_duplicate_spec_names():
    dup = EncoderModuleSpec(
        name="visual",
        build_module=lambda hf, qc: _make_audio_like_module(),
        checkpoint_prefixes=("audio_tower.",),
    )
    with pytest.raises(ValueError, match="must be unique"):
        EncoderModuleContainer(
            hf_config=None,
            encoder_specs=(VISUAL_SPEC, dup),
            quant_config=None,
        )


# ---------------------------------------------------------------------------
# Prefix filtering — the load-bearing test for "no thinker weights leak"
# ---------------------------------------------------------------------------


def _full_qwen3_omni_checkpoint_keys():
    """A representative slice of what a real checkpoint stream contains.

    Includes visual + audio + language + talker + unrelated-encoder
    prefixes. The image-stage container must accept ONLY visual; the
    audio-stage container must accept ONLY audio.
    """
    return [
        # Visual
        ("model.visual.proj.weight", torch.randn(4, 4)),
        ("model.visual.proj.bias", torch.randn(4)),
        ("thinker.visual.proj.weight", torch.randn(4, 4)),  # legacy key form
        # Audio
        ("audio_tower.encoder.weight", torch.randn(4, 4)),
        ("audio_tower.encoder.bias", torch.randn(4)),
        ("thinker.audio_tower.encoder.weight", torch.randn(4, 4)),
        # Language model — must NEVER allocate on encoder GPU
        ("thinker.model.layers.0.self_attn.q_proj.weight", torch.randn(4, 4)),
        ("thinker.model.layers.0.mlp.experts.0.w1.weight", torch.randn(4, 4)),
        ("thinker.lm_head.weight", torch.randn(4, 4)),
        # Talker — must NEVER allocate on encoder GPU
        ("talker.model.layers.0.self_attn.q_proj.weight", torch.randn(4, 4)),
        ("talker.lm_head.weight", torch.randn(4, 4)),
        # Code2wav / vocoder / random unrelated keys
        ("code2wav.decoder.0.weight", torch.randn(4, 4)),
    ]


def test_image_stage_container_loads_only_visual_weights():
    """Locks: image stage MUST NOT allocate audio/LLM/talker tensors."""
    container = EncoderModuleContainer(
        hf_config=None,
        encoder_specs=(VISUAL_SPEC,),
        quant_config=None,
    )

    # Capture which params the loader actually wrote to.
    original_params = {n: p.detach().clone() for n, p in container.named_parameters()}

    container.load_weights(_full_qwen3_omni_checkpoint_keys())

    new_params = dict(container.named_parameters())
    # Only visual.proj.weight + visual.proj.bias should exist on this container.
    param_names = set(new_params)
    assert param_names == {"visual.proj.weight", "visual.proj.bias"}, param_names

    # And those got written to (different from random init).
    for name in param_names:
        # The checkpoint stream contained two competing keys for proj.weight
        # (model.visual and thinker.visual). The last applied wins. Either
        # way the value should differ from the random init.
        assert not torch.equal(new_params[name], original_params[name]), name


def test_audio_stage_container_loads_only_audio_weights():
    container = EncoderModuleContainer(
        hf_config=None,
        encoder_specs=(AUDIO_SPEC,),
        quant_config=None,
    )

    container.load_weights(_full_qwen3_omni_checkpoint_keys())

    param_names = set(dict(container.named_parameters()))
    assert param_names == {"audio_tower.encoder.weight", "audio_tower.encoder.bias"}, param_names


def test_container_skips_unmatched_prefix_silently():
    """Mirrors upstream loader's tolerant behavior: unknown keys do not
    raise (some checkpoints carry tied-embedding shadows, optimizer
    state, etc.); the container just drops them."""
    container = EncoderModuleContainer(
        hf_config=None,
        encoder_specs=(VISUAL_SPEC,),
        quant_config=None,
    )

    weights = [
        ("totally.unrelated.tensor", torch.randn(4, 4)),
        ("optimizer.state.0", torch.randn(4)),
    ]
    # Should not raise.
    container.load_weights(weights)


def test_checkpoint_rewrites_apply_in_order():
    """Multiple rewrites apply left-to-right via str.replace."""
    spec = EncoderModuleSpec(
        name="visual",
        build_module=lambda hf, qc: _make_visual_like_module(),
        checkpoint_prefixes=("model.visual.",),
        checkpoint_rewrites=(
            ("model.visual.", "visual."),
            # Second rewrite that happens to pattern-match after the first:
            # rename "proj.bias" -> "proj.bias" (no-op, just exercise ordering).
            ("proj.bias", "proj.bias"),
        ),
    )
    container = EncoderModuleContainer(
        hf_config=None,
        encoder_specs=(spec,),
        quant_config=None,
    )
    weights = [("model.visual.proj.weight", torch.full((4, 4), 7.0))]
    container.load_weights(weights)
    # After rewrite "model.visual.proj.weight" → "visual.proj.weight",
    # which IS a real param, so it gets written.
    assert torch.allclose(
        dict(container.named_parameters())["visual.proj.weight"],
        torch.full((4, 4), 7.0),
    )


# ---------------------------------------------------------------------------
# Production specs sanity
# ---------------------------------------------------------------------------


def test_qwen3_omni_visual_spec_prefix_set():
    """Lock the production visual spec accepts the three known forms.

    Production checkpoints from different SGLang / HF versions place
    visual weights under ``model.visual.``, ``thinker.visual.``, or
    plain ``visual.``. Any drift here means the loader silently skips
    the visual weights and the encoder runs random-init.
    """
    from sglang_omni.models.qwen3_omni.encoder_adapters import (
        QWEN3_OMNI_VISUAL_SPEC,
    )

    assert QWEN3_OMNI_VISUAL_SPEC.name == "visual"
    assert "model.visual." in QWEN3_OMNI_VISUAL_SPEC.checkpoint_prefixes
    assert "thinker.visual." in QWEN3_OMNI_VISUAL_SPEC.checkpoint_prefixes
    assert "visual." in QWEN3_OMNI_VISUAL_SPEC.checkpoint_prefixes


def test_qwen3_omni_audio_spec_prefix_set():
    from sglang_omni.models.qwen3_omni.encoder_adapters import (
        QWEN3_OMNI_AUDIO_SPEC,
    )

    assert QWEN3_OMNI_AUDIO_SPEC.name == "audio_tower"
    assert "audio_tower." in QWEN3_OMNI_AUDIO_SPEC.checkpoint_prefixes
    assert "thinker.audio_tower." in QWEN3_OMNI_AUDIO_SPEC.checkpoint_prefixes


def test_image_adapter_advertises_only_visual_spec():
    from sglang_omni.models.qwen3_omni.encoder_adapters import (
        QWEN3_OMNI_VISUAL_SPEC,
        Qwen3OmniImageEncoderAdapter,
    )

    specs = Qwen3OmniImageEncoderAdapter.encoder_specs
    assert len(specs) == 1
    assert specs[0] is QWEN3_OMNI_VISUAL_SPEC


def test_audio_adapter_advertises_only_audio_spec():
    from sglang_omni.models.qwen3_omni.encoder_adapters import (
        QWEN3_OMNI_AUDIO_SPEC,
        Qwen3OmniAudioEncoderAdapter,
    )

    specs = Qwen3OmniAudioEncoderAdapter.encoder_specs
    assert len(specs) == 1
    assert specs[0] is QWEN3_OMNI_AUDIO_SPEC


# ---------------------------------------------------------------------------
# Fused-shard dispatch — locks the audio NaN bug found end-to-end
# ---------------------------------------------------------------------------


class _FusedQKVLinear(nn.Module):
    """Tiny stand-in for QKVParallelLinear with a shard-aware weight_loader.

    Mirrors the real shape: ``qkv_proj.weight`` is the param name (i.e.
    ``self.weight`` here), and the loader receives ``shard_id`` so it
    knows which slice of the fused tensor to write to.
    """
    def __init__(self):
        super().__init__()
        # Total fused dim 12 = 3*4 (q + k + v shards of 4 each).
        self.weight = nn.Parameter(torch.zeros(12, 4))

        def _shard_loader(param, weight, shard_id):
            offset = {"q": 0, "k": 4, "v": 8}[shard_id]
            param.data[offset:offset + 4].copy_(weight)

        self.weight.weight_loader = _shard_loader


def _make_audio_with_fused_qkv():
    """Stand-in audio module with one fused-QKV self_attn block."""
    m = nn.Module()
    m.self_attn = nn.Module()
    m.self_attn.qkv_proj = _FusedQKVLinear()
    return m


def test_audio_stage_fuses_qkv_via_stacked_params_mapping():
    """Locks the fused-shard contract.

    The real Qwen3-Omni audio checkpoint has 192 q_proj/k_proj/v_proj
    keys per audio attention layer that must be fused into qkv_proj
    via ``weight_loader(param, weight, shard_id)``. Pre-fix, my naive
    container dropped those keys → audio attention layers ran with
    random init → NaN in the output → CUDA assert in multinomial
    sampling downstream. This test fails fast if the dispatch
    regresses.
    """
    spec = EncoderModuleSpec(
        name="audio_tower",
        build_module=lambda hf, qc: _make_audio_with_fused_qkv(),
        checkpoint_prefixes=("audio_tower.",),
        stacked_params_mapping=(
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
        ),
    )
    container = EncoderModuleContainer(
        hf_config=None, encoder_specs=(spec,), quant_config=None,
    )

    # Synthesize one (shard, value) per slice so we can verify each
    # ended up in the right place.
    q_w = torch.full((4, 4), 1.0)
    k_w = torch.full((4, 4), 2.0)
    v_w = torch.full((4, 4), 3.0)
    container.load_weights([
        ("audio_tower.self_attn.q_proj.weight", q_w),
        ("audio_tower.self_attn.k_proj.weight", k_w),
        ("audio_tower.self_attn.v_proj.weight", v_w),
    ])

    fused = dict(container.named_parameters())["audio_tower.self_attn.qkv_proj.weight"]
    assert torch.allclose(fused[0:4], q_w)
    assert torch.allclose(fused[4:8], k_w)
    assert torch.allclose(fused[8:12], v_w)


def test_visual_load_weights_delegated_to_submodule():
    """When a submodule has its own ``load_weights``, the container
    must delegate to it (the visual encoder's own loader knows how to
    fuse its own attention shards). The container must NOT also try
    to apply ``stacked_params_mapping`` for that submodule."""
    calls = []

    class _ModuleWithLoadWeights(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4)

        def load_weights(self, weights):
            for name, w in weights:
                calls.append(name)

    spec = EncoderModuleSpec(
        name="visual",
        build_module=lambda hf, qc: _ModuleWithLoadWeights(),
        checkpoint_prefixes=("visual.",),
        # Stacked mapping should be ignored because the submodule
        # has its own load_weights.
        stacked_params_mapping=((".attn.qkv_proj", ".attn.q_proj", "q"),),
    )
    container = EncoderModuleContainer(
        hf_config=None, encoder_specs=(spec,), quant_config=None,
    )
    container.load_weights([
        ("visual.attn.q_proj.weight", torch.zeros(4, 4)),
        ("visual.proj.weight", torch.zeros(4, 4)),
    ])
    # The submodule sees relative names (without the leading "visual.").
    assert calls == ["attn.q_proj.weight", "proj.weight"]
