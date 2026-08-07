# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch
from dots_tts.models.dots_tts.config import _DiTConfig, _EncoderConfig
from dots_tts.modules.backbone.dit import DiT
from dots_tts.modules.backbone.encoder import VAESemanticEncoder

from sglang_omni.models.dots_tts import tail

FM_HIDDEN = 32
LATENT_DIM = 6
PATCH_SIZE = 2
NFE = 2


def test_batched_tail_mask_hides_padding_and_preserves_causality() -> None:
    from sglang_omni.models.dots_tts.tail import batched_causal_update_mask

    mask = batched_causal_update_mask(
        capacity_tokens=4,
        valid_persistent=torch.tensor([1, 3]),
        prev_len=2,
        current_len=2,
    )

    assert mask.shape == (2, 1, 4, 8)
    assert mask[0, 0].tolist() == [
        [True, False, False, False, True, False, False, False],
        [True, False, False, False, True, True, False, False],
        [True, False, False, False, True, True, True, True],
        [True, False, False, False, True, True, True, True],
    ]
    assert mask[1, 0].tolist() == [
        [True, True, True, False, True, False, False, False],
        [True, True, True, False, True, True, False, False],
        [True, True, True, False, True, True, True, True],
        [True, True, True, False, True, True, True, True],
    ]


class _TailModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.velocity_field_predictor = DiT(
            in_dim=FM_HIDDEN,
            out_dim=LATENT_DIM,
            transformer_config=_DiTConfig(
                num_layers=2,
                num_heads=2,
                hidden_size=FM_HIDDEN,
                ffn_hidden_size=64,
                modulation=True,
                qk_norm=True,
                rotary_bias=True,
            ),
            mode="meanflow",
        )
        self.coordinate_proj = torch.nn.Linear(LATENT_DIM, FM_HIDDEN)
        self.latent_proj = torch.nn.Linear(LATENT_DIM, FM_HIDDEN)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.normal_(0.0, 0.2)


def _patch_encoder() -> VAESemanticEncoder:
    encoder_config = _EncoderConfig(
        num_layers=1,
        num_heads=2,
        hidden_size=FM_HIDDEN,
        ffn_hidden_size=64,
        causal=True,
    )
    config = type(
        "_EncoderConfigStub",
        (),
        {"patch_size": PATCH_SIZE, "PatchEncoder": encoder_config},
    )()
    return VAESemanticEncoder(in_dim=LATENT_DIM, out_dim=FM_HIDDEN, config=config)


def _build_tail(model: _TailModel, *, slots: int):
    encoder = _patch_encoder()
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.normal_(0.0, 0.2)
    return tail.DotsTtsAcousticTail(
        dit=tail.fuse_dit_for_inference(model),
        coordinate_proj=model.coordinate_proj,
        patch_encoder=encoder,
        spec=tail.DotsTtsTailSpec(
            nfe=NFE,
            patch_capacity=8,
            num_slots=slots,
            hidden_patch_size=1,
            latent_patch_size=PATCH_SIZE,
            latent_dim=LATENT_DIM,
            fm_hidden_size=FM_HIDDEN,
        ),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _reference_meanflow(
    dit: torch.nn.Module,
    coordinate_proj: torch.nn.Module,
    sequence: torch.Tensor,
    fm_seq_len: int,
    g_cond: torch.Tensor,
) -> torch.Tensor:
    total = fm_seq_len + PATCH_SIZE
    x_base = sequence.new_zeros(1, total, FM_HIDDEN)
    x_base[:, :fm_seq_len] = sequence[:, :fm_seq_len]
    mask = torch.zeros((1, total, total), dtype=torch.bool)
    block_start = fm_seq_len - 1
    if block_start:
        mask[:, :block_start, :block_start] = torch.ones(
            block_start, block_start, dtype=torch.bool
        ).tril()
    mask[:, block_start:fm_seq_len, :fm_seq_len] = True
    mask[:, block_start:fm_seq_len, fm_seq_len:] = True
    mask[:, fm_seq_len:, :] = True
    positions = torch.arange(total, dtype=torch.float32).reshape(1, total)
    latent = torch.randn(1, PATCH_SIZE, LATENT_DIM)
    times = torch.linspace(0.0, 1.0, NFE + 1)
    for step in range(NFE):
        value = x_base.clone()
        value[:, fm_seq_len:] = coordinate_proj(latent)
        duration = (times[step + 1] - times[step]).expand(1)
        velocity = dit(
            x=value,
            timesteps=times[step].expand(1),
            duration=duration,
            attn_mask=mask,
            pos_ids=positions,
            g_cond=g_cond,
        )[:, fm_seq_len:]
        latent = (latent + duration.reshape(1, 1, 1) * velocity).clone()
    return latent


def test_kv_cached_tail_matches_full_recompute() -> None:
    torch.manual_seed(1234)
    model = _TailModel().eval()
    acoustic_tail = _build_tail(model, slots=1)
    unit = acoustic_tail.spec.unit_len
    g_cond = torch.randn(1, FM_HIDDEN)
    grid = torch.linspace(0.0, 1.0, NFE + 1)
    mods = acoustic_tail.dit.build_mods(
        grid[:-1], duration=grid[1:] - grid[:-1], g_cond=g_cond
    )
    prompt_rows = torch.randn(3 * unit, FM_HIDDEN)
    slot = acoustic_tail.acquire_slot()
    acoustic_tail.seed_fm_history(slot, fm_rows=prompt_rows, all_mods=mods)
    sequence = torch.zeros(1, acoustic_tail.spec.dit_cache_tokens + unit, FM_HIDDEN)
    sequence[0, : prompt_rows.size(0)] = prompt_rows
    sequence_len = prompt_rows.size(0)

    hidden = torch.randn(1, FM_HIDDEN)
    sequence[0, sequence_len] = hidden[0]
    sequence_len += 1
    torch.manual_seed(9)
    expected = _reference_meanflow(
        acoustic_tail.dit,
        model.coordinate_proj,
        sequence,
        sequence_len,
        g_cond,
    )
    torch.manual_seed(9)
    actual = acoustic_tail.sample_patches(
        [slot], fm_hidden_rows=hidden, latent_proj=model.latent_proj
    )

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)


def test_tail_slots_are_bounded_and_reusable() -> None:
    acoustic_tail = _build_tail(_TailModel().eval(), slots=2)
    first = acoustic_tail.acquire_slot()
    acoustic_tail.acquire_slot()
    try:
        acoustic_tail.acquire_slot()
    except RuntimeError as error:
        assert "ran out of slots" in str(error)
    else:
        raise AssertionError("slot exhaustion must fail")
    acoustic_tail.release_slot(first)
    assert acoustic_tail.acquire_slot() == first


def test_request_release_forgets_slot_before_it_can_be_reused() -> None:
    from sglang_omni.models.dots_tts.flow_head import DotsTTSFlowHead

    released = []
    flow = SimpleNamespace(_tail=SimpleNamespace(release_slot=released.append))
    state = SimpleNamespace(slot=3)

    DotsTTSFlowHead.release_request(flow, state)
    DotsTTSFlowHead.release_request(flow, state)

    assert state.slot is None
    assert released == [3]


def test_fused_dit_builds_modulations_with_bfloat16_weights() -> None:
    model = _TailModel().eval().to(torch.bfloat16)
    dit = tail.fuse_dit_for_inference(model)
    steps = torch.tensor([0.0, 0.5], dtype=torch.bfloat16)

    mods = dit.build_mods(steps, duration=torch.full_like(steps, 0.5))

    assert mods.dtype == torch.bfloat16
