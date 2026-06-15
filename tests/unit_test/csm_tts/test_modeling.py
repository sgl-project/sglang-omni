# SPDX-License-Identifier: Apache-2.0
"""Unit tests for csm_tts.modeling (CPU, no GPU, no sglang engine).

 The depth-decoder greedy parity test vs HF pins the
position/offset/head off-by-ones (the #1 bug class here) and the
codebooks-head orientation before any GPU work — a hard correctness
blocker.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.csm_tts import modeling

torch.manual_seed(0)

# Real CSM-1B depth rope (config.json depth_decoder_config) — used
# on BOTH sides of the HF parity test so our llama3-rope port is pinned too.
_DEPTH_ROPE_SCALING = {
    "rope_type": "llama3",
    "factor": 32.0,
    "low_freq_factor": 0.001953125,
    "high_freq_factor": 0.0078125,
    "original_max_position_embeddings": 16,
}


def _tiny_depth_cfg(
    num_codebooks: int = 8,
    vocab_size: int = 17,
    hidden_size: int = 16,
    backbone_hidden_size: int = 32,
) -> SimpleNamespace:
    """Tiny depth config — same field set as hf_config.build_depth_decoder_config."""
    return SimpleNamespace(
        hidden_size=hidden_size,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=8,
        num_key_value_heads=1,
        intermediate_size=hidden_size * 2,
        max_position_embeddings=33,
        backbone_hidden_size=backbone_hidden_size,
        vocab_size=vocab_size,
        num_codebooks=num_codebooks,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        rope_theta=500000.0,
        rope_scaling=dict(_DEPTH_ROPE_SCALING),
    )


def _randomize(module: torch.nn.Module, std: float = 0.05) -> None:
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(0, std)


def test_frame_embedding_matches_hand_rolled_reference() -> None:
    """``CsmFrameEmbedding.forward(codes_B32)`` equals the hand-rolled
    ``Σ_k embed[c_k + k*2051]`` over the one fused table (random weights,
    fp32, exact)."""
    fe = modeling.CsmFrameEmbedding(
        num_codebooks=32, codebook_vocab=2051, hidden_size=64
    )
    _randomize(fe, std=0.02)
    codes = torch.randint(0, 2051, (3, 32))
    with torch.no_grad():
        got = fe(codes)
        ref = torch.stack(
            [
                sum(fe.weight[codes[b, k] + k * 2051] for k in range(32))
                for b in range(3)
            ]
        )
    assert got.shape == (3, 64)
    assert torch.allclose(got, ref, atol=1e-5)


def test_embed_codebook_offset_correctness() -> None:
    """``embed_codebook(codes_B, k)`` reads rows at exactly ``codes + k*2051``
    for every k in 0..31."""
    fe = modeling.CsmFrameEmbedding(
        num_codebooks=32, codebook_vocab=2051, hidden_size=8
    )
    _randomize(fe, std=0.02)
    codes = torch.randint(0, 2051, (4,))
    with torch.no_grad():
        for k in range(32):
            got = fe.embed_codebook(codes, k)
            assert torch.equal(
                got, fe.weight[codes + k * 2051]
            ), f"offset wrong at k={k}"


def test_codebooks_head_index_mapping_and_orientation() -> None:
    """``CsmCodebooksHead`` head *k* predicts codebook *k+1*: ``forward(h, k)``
    equals ``h @ weight[k]`` ≡ ``F.linear(h, weight[k].T)`` with the weight
    kept in CHECKPOINT layout ``[31, hidden, vocab]`` (orientation trap,
    review)."""
    head = modeling.CsmCodebooksHead(num_heads=31, hidden_size=16, codebook_vocab=23)
    _randomize(head, std=0.02)
    assert head.weight.shape == (31, 16, 23)  # checkpoint layout, NOT transposed
    h = torch.randn(5, 16)
    with torch.no_grad():
        for step in (0, 13, 30):
            got = head(h, step)
            assert got.shape == (5, 23)
            assert torch.allclose(got, h @ head.weight[step], atol=1e-6)
            assert torch.allclose(got, F.linear(h, head.weight[step].T), atol=1e-6)


def test_generate_frame_shapes_with_random_weights() -> None:
    """``generate_frame`` at random weights, greedy: shape ``[B, 32]`` int64,
    values in ``[0, vocab)``, cb0 passes through unchanged, pool-sized param
    buffers are sliced ``[:bs]``; poisoned depth KV cannot leak into a frame
    (per-frame reset is structurally free)."""
    cfg = _tiny_depth_cfg(num_codebooks=32, vocab_size=51)
    fe = modeling.CsmFrameEmbedding(
        cfg.num_codebooks, cfg.vocab_size, cfg.backbone_hidden_size
    )
    dd = modeling.CsmDepthDecoder(cfg, fe, max_slots=4)
    _randomize(dd)
    bs = 2
    h = torch.randn(bs, cfg.backbone_hidden_size)
    cb0 = torch.randint(0, cfg.vocab_size, (bs,))
    temps = torch.zeros(4)  # pool-sized on purpose; greedy
    topks = torch.full((4,), cfg.vocab_size, dtype=torch.long)
    with torch.no_grad():
        frame = dd.generate_frame(h, cb0, temps, topks, bs=bs)
    assert frame.shape == (bs, 32)
    assert frame.dtype == torch.int64
    assert (frame >= 0).all() and (frame < cfg.vocab_size).all()
    assert torch.equal(frame[:, 0], cb0)

    with torch.no_grad():
        dd.k_cache.copy_(torch.randn_like(dd.k_cache) * 50)
        dd.v_cache.copy_(torch.randn_like(dd.v_cache) * 50)
        frame2 = dd.generate_frame(h, cb0, temps, topks, bs=bs)
    assert torch.equal(frame2, frame)


def _dense_layer_reference(layer, x, rope_cos, rope_sin):
    """Independent full-sequence reimplementation (no cache, naive attention)."""
    attn = layer.self_attn
    B, T, _ = x.shape
    h = layer.input_layernorm(x)
    q = attn.q_proj(h).view(B, T, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(h).view(B, T, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(h).view(B, T, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
    cos, sin = rope_cos[:T], rope_sin[:T]
    rot = modeling._rotate_half
    q = q * cos + rot(q) * sin
    k = k * cos + rot(k) * sin
    rep = attn.num_heads // attn.num_kv_heads
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(attn.head_dim)
    scores = scores + torch.full((T, T), float("-inf")).triu(1)
    o = (scores.softmax(-1) @ v).transpose(1, 2).reshape(B, T, -1)
    x = x + attn.o_proj(o)
    return x + layer.mlp(layer.post_attention_layernorm(x))


def test_incremental_static_kv_matches_dense_reference() -> None:
    """The incremental 33-slot static-KV path equals an independent dense
    full-recompute reference (pins KV indexing, rope positions, causal mask,
    GQA mapping)."""
    cfg = _tiny_depth_cfg()
    fe = modeling.CsmFrameEmbedding(
        cfg.num_codebooks, cfg.vocab_size, cfg.backbone_hidden_size
    )
    dd = modeling.CsmDepthDecoder(cfg, fe, max_slots=3)
    _randomize(dd)
    bs = 2
    h = torch.randn(bs, cfg.backbone_hidden_size)
    cb0 = torch.randint(0, cfg.vocab_size, (bs,))
    temps = torch.zeros(3)
    topks = torch.full((3,), cfg.vocab_size, dtype=torch.long)
    with torch.no_grad():
        frame = dd.generate_frame(h, cb0, temps, topks, bs=bs)

        # Dense reference: re-run the whole prefix per step, no KV cache.
        embeds = [h]
        codes = [cb0]
        prev = cb0
        for p in range(1, cfg.num_codebooks):
            embeds.append(dd.frame_embedding.embed_codebook(prev, p - 1))
            x = dd.inputs_embeds_projector(torch.stack(embeds, dim=1))
            for layer in dd.layers:
                x = _dense_layer_reference(layer, x, dd.rope_cos, dd.rope_sin)
            logits = dd.codebooks_head(dd.norm(x[:, -1]), p - 1).float()
            prev = logits.argmax(-1)
            codes.append(prev)
        ref = torch.stack(codes, dim=1)
    assert torch.equal(frame, ref)


def test_depth_decoder_greedy_parity_vs_hf() -> None:
    """Single-frame greedy parity vs HF ``CsmDepthDecoderForCausalLM`` (tiny
       config, random weights, temp=0, both fp32, EXACT code match) — pins the
    head orientation + the position/offset scheme + our llama3-rope port.
       Hard correctness blocker."""
    pytest.importorskip(
        "transformers", reason="needs transformers >= 4.52 (native CSM)"
    )
    try:
        from transformers.models.csm import CsmDepthDecoderConfig
        from transformers.models.csm.modeling_csm import CsmDepthDecoderForCausalLM
    except ImportError:
        pytest.skip("installed transformers lacks the native CSM depth decoder")

    torch.manual_seed(7)
    cfg = _tiny_depth_cfg()
    fe = modeling.CsmFrameEmbedding(
        cfg.num_codebooks, cfg.vocab_size, cfg.backbone_hidden_size
    )
    dd = modeling.CsmDepthDecoder(cfg, fe, max_slots=4)
    with torch.no_grad():
        fe.weight.normal_(0, 0.05)
    _randomize(dd)

    hf_cfg = CsmDepthDecoderConfig(
        num_codebooks=cfg.num_codebooks,
        vocab_size=cfg.vocab_size,
        backbone_hidden_size=cfg.backbone_hidden_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        max_position_embeddings=cfg.max_position_embeddings,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        rope_scaling=dict(cfg.rope_scaling),
    )
    hf = CsmDepthDecoderForCausalLM(hf_cfg).eval()

    # Copy OUR random weights into the HF reference.
    state = {
        "model.embed_tokens.weight": fe.weight,
        "model.inputs_embeds_projector.weight": dd.inputs_embeds_projector.weight,
        "model.norm.weight": dd.norm.weight,
        "codebooks_head.weight": dd.codebooks_head.weight,
    }
    for i, layer in enumerate(dd.layers):
        pre = f"model.layers.{i}."
        attn, mlp = layer.self_attn, layer.mlp
        state[pre + "self_attn.q_proj.weight"] = attn.q_proj.weight
        state[pre + "self_attn.k_proj.weight"] = attn.k_proj.weight
        state[pre + "self_attn.v_proj.weight"] = attn.v_proj.weight
        state[pre + "self_attn.o_proj.weight"] = attn.o_proj.weight
        state[pre + "mlp.gate_proj.weight"] = mlp.gate_proj.weight
        state[pre + "mlp.up_proj.weight"] = mlp.up_proj.weight
        state[pre + "mlp.down_proj.weight"] = mlp.down_proj.weight
        state[pre + "input_layernorm.weight"] = layer.input_layernorm.weight
        state[pre + "post_attention_layernorm.weight"] = (
            layer.post_attention_layernorm.weight
        )
    missing, unexpected = hf.load_state_dict(
        {k: v.detach().clone() for k, v in state.items()}, strict=False
    )
    assert not unexpected, f"HF reference got unexpected tensors: {unexpected}"
    assert all(
        "rotary" in name or "inv_freq" in name for name in missing
    ), f"HF reference is missing real weights: {missing}"

    bs = 2
    h = torch.randn(bs, cfg.backbone_hidden_size)
    cb0 = torch.tensor([3, 11])
    temps = torch.zeros(4)
    topks = torch.full((4,), cfg.vocab_size, dtype=torch.long)
    with torch.no_grad():
        ours = dd.generate_frame(h, cb0, temps, topks, bs=bs)

    # HF greedy drive: depth prefill [pos0=h, pos1=cb0] then 1-token steps.
    # HF embeds position p with codebook-table offset clamp(p-1, 0) and head
    # W[p-1] (modeling_csm.py CsmDepthDecoderModel.forward + CsmCodebooksHead).
    # ``cache_position`` must be passed EXPLICITLY: the LM wrapper hands it to
    # CsmCodebooksHead for head selection, and without it the head silently
    # falls back to W[0] for every step (as HF's own generate path passes it).
    with torch.no_grad():
        ids = torch.stack([torch.zeros_like(cb0), cb0], dim=1)
        out = hf(
            input_ids=ids,
            backbone_last_hidden_state=h,
            use_cache=True,
            logits_to_keep=1,
            cache_position=torch.tensor([0, 1]),
        )
        past = out.past_key_values
        prev = out.logits[:, -1].argmax(-1)
        codes = [cb0, prev]
        for p in range(2, cfg.num_codebooks):
            out = hf(
                input_ids=prev.unsqueeze(1),
                past_key_values=past,
                use_cache=True,
                logits_to_keep=1,
                cache_position=torch.tensor([p]),
            )
            past = out.past_key_values
            prev = out.logits[:, -1].argmax(-1)
            codes.append(prev)
        ref = torch.stack(codes, dim=1)

    assert torch.equal(
        ours, ref
    ), f"greedy depth divergence:\nours={ours.tolist()}\nhf  ={ref.tolist()}"
