# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
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


def _build_tail(
    model: _TailModel,
    *,
    slots: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    patch_capacity: int = 8,
    optimize: bool = False,
):
    encoder = _patch_encoder().to(device=device, dtype=dtype)
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.normal_(0.0, 0.2)
    return tail.DotsTtsAcousticTail(
        dit=tail.fuse_dit_for_inference(model),
        coordinate_proj=model.coordinate_proj,
        latent_proj=model.latent_proj,
        patch_encoder=encoder,
        spec=tail.DotsTtsTailSpec(
            nfe=NFE,
            patch_capacity=patch_capacity,
            num_slots=slots,
            hidden_patch_size=1,
            latent_patch_size=PATCH_SIZE,
            latent_dim=LATENT_DIM,
            fm_hidden_size=FM_HIDDEN,
        ),
        device=device,
        dtype=dtype,
        optimize=optimize,
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


@pytest.mark.parametrize("slots", [1, 2])
def test_kv_cached_tail_matches_full_recompute(slots: int) -> None:
    torch.manual_seed(1234)
    model = _TailModel().eval()
    acoustic_tail = _build_tail(model, slots=slots)
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
    actual = acoustic_tail.sample_patches([slot], fm_hidden_rows=hidden)

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    assert acoustic_tail._dit_contiguous_view_steps == (NFE if slots == 1 else 0)


def test_tail_slots_are_bounded_and_reusable() -> None:
    acoustic_tail = _build_tail(_TailModel().eval(), slots=2)
    first = acoustic_tail.acquire_slot()
    acoustic_tail.acquire_slot()
    try:
        acoustic_tail.acquire_slot()
    except RuntimeError as error:
        message = str(error)
        assert "ran out of slots" in message
        assert "admission failed" in message
        assert "does not silently shrink" in message
        assert "raise max_running_requests" in message
    else:
        raise AssertionError("slot exhaustion must fail")
    acoustic_tail.release_slot(first)
    assert acoustic_tail.acquire_slot() == first


def test_estimate_acoustic_pool_bytes_matches_allocated_tensors() -> None:
    acoustic_tail = _build_tail(_TailModel().eval(), slots=2, patch_capacity=8)
    estimate = acoustic_tail._pool_memory_estimate(acoustic_tail._mods_width)
    assert estimate.total_bytes == acoustic_tail._allocated_pool_bytes()
    assert estimate.num_slots == 2
    assert estimate.patch_capacity == 8
    assert estimate.bytes_per_slot == estimate.total_bytes // 2
    # note (guozhihao-224): pool bytes scale linearly with slot count at fixed capacity.
    double = tail.estimate_acoustic_pool_bytes(
        spec=tail.DotsTtsTailSpec(
            nfe=NFE,
            patch_capacity=8,
            num_slots=4,
            hidden_patch_size=1,
            latent_patch_size=PATCH_SIZE,
            latent_dim=LATENT_DIM,
            fm_hidden_size=FM_HIDDEN,
        ),
        dit_layers=acoustic_tail._dit_layers,
        dit_heads=acoustic_tail._dit_heads,
        dit_head_dim=acoustic_tail._dit_head_dim,
        encoder_layers=acoustic_tail._encoder_layers,
        encoder_heads=acoustic_tail._encoder_heads,
        encoder_head_dim=acoustic_tail._encoder_head_dim,
        encoder_block=acoustic_tail._encoder_block,
        encoder_conv_channels=int(acoustic_tail._encoder.ds_proj.in_channels),
        encoder_conv_padding=int(acoustic_tail._encoder.ds_proj.left_padding),
        mods_width=acoustic_tail._mods_width,
        dtype=acoustic_tail.dtype,
    )
    assert double.total_bytes == 2 * estimate.total_bytes


def test_validate_acoustic_pool_memory_rejects_when_vram_is_tight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimate = tail.AcousticPoolMemoryEstimate(
        dit_kv_bytes=8 << 30,
        encoder_kv_bytes=2 << 30,
        scratch_bytes=1 << 30,
        aux_bytes=1 << 30,
        total_bytes=12 << 30,
        num_slots=16,
        patch_capacity=501,
        nfe=4,
        dtype=torch.bfloat16,
    )
    device = torch.device("cuda:0")
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda _device=None: (4 << 30, 80 << 30),
    )
    with pytest.raises(ValueError, match="admission failed at startup") as caught:
        tail.validate_acoustic_pool_memory(estimate, device=device)
    message = str(caught.value)
    assert "Parameters are not changed automatically" in message
    assert "Lower max_running_requests" in message
    assert "about 4 full-length slot(s)" in message

    # Enough free memory passes.
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda _device=None: (40 << 30, 80 << 30),
    )
    tail.validate_acoustic_pool_memory(estimate, device=device)

    # Non-CUDA devices skip the gate.
    tail.validate_acoustic_pool_memory(estimate, device=torch.device("cpu"))


def test_validate_acoustic_pool_memory_releases_cached_blocks_before_sampling_free_vram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimate = tail.AcousticPoolMemoryEstimate(
        dit_kv_bytes=10,
        encoder_kv_bytes=0,
        scratch_bytes=0,
        aux_bytes=0,
        total_bytes=10,
        num_slots=1,
        patch_capacity=1,
        nfe=1,
        dtype=torch.uint8,
    )
    memory = {"free": 10}
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: memory.update(free=12),
    )
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda _device=None: (memory["free"], 20),
    )

    tail.validate_acoustic_pool_memory(
        estimate,
        device=torch.device("cuda:0"),
    )


def test_permuted_full_pool_matches_fragmented_gather_fallback() -> None:
    torch.manual_seed(1234)
    direct = _build_tail(_TailModel().eval(), slots=2)
    torch.manual_seed(1234)
    fallback = _build_tail(_TailModel().eval(), slots=3)
    direct_slots = [direct.acquire_slot(), direct.acquire_slot()][::-1]
    fallback_slots = [
        fallback.acquire_slot(),
        fallback.acquire_slot(),
        fallback.acquire_slot(),
    ]
    fallback.release_slot(fallback_slots.pop(1))

    grid = torch.linspace(0.0, 1.0, NFE + 1)
    for row, units in enumerate((3, 2)):
        g_cond = torch.randn(1, FM_HIDDEN)
        mods = direct.dit.build_mods(
            grid[:-1], duration=grid[1:] - grid[:-1], g_cond=g_cond
        )
        history = torch.randn(units * direct.spec.unit_len, FM_HIDDEN)
        for acoustic_tail, slot in (
            (direct, direct_slots[row]),
            (fallback, fallback_slots[row]),
        ):
            acoustic_tail.seed_fm_history(slot, fm_rows=history, all_mods=mods)
            acoustic_tail.initialize_slot_rng(slot, 100 + row)

    hidden = torch.randn(2, FM_HIDDEN)
    actual = direct.sample_patches(direct_slots, fm_hidden_rows=hidden)
    expected = fallback.sample_patches(fallback_slots, fm_hidden_rows=hidden)

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    assert direct._dit_contiguous_view_steps == NFE
    assert fallback._dit_contiguous_view_steps == 0


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


@pytest.mark.gpu
def test_batched_tail_cuda_graph_matches_eager_for_dynamic_slot_order() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(1234)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    eager_model = _TailModel().eval().to(device=device, dtype=dtype)
    graph_model = copy.deepcopy(eager_model)
    torch.manual_seed(9)
    eager = _build_tail(
        eager_model,
        slots=8,
        device=device,
        dtype=dtype,
        patch_capacity=33,
    )
    torch.manual_seed(9)
    graph = _build_tail(
        graph_model,
        slots=8,
        device=device,
        dtype=dtype,
        patch_capacity=33,
        optimize=True,
    )

    for name in (
        "_dit_k",
        "_dit_v",
        "_encoder_k",
        "_encoder_v",
        "_encoder_conv_tail",
        "_window",
        "_all_mods",
    ):
        eager_value = getattr(eager, name)
        eager_value.normal_(0, 0.05)
        getattr(graph, name).copy_(eager_value)
    for slot in range(8):
        eager._fm_seq_len[slot] = graph._fm_seq_len[slot] = 15
        eager._encoder_seq_len[slot] = graph._encoder_seq_len[slot] = 4
        eager.initialize_slot_rng(slot, 100 + slot)
        graph.initialize_slot_rng(slot, 100 + slot)

    slots = [7, 2, 5, 0, 6, 1, 4, 3]
    hidden = torch.randn(8, FM_HIDDEN, device=device, dtype=dtype)
    eager_latent = eager.sample_patches(slots, fm_hidden_rows=hidden)
    graph_latent = graph.sample_patches(slots, fm_hidden_rows=hidden)
    torch.testing.assert_close(graph_latent, eager_latent, rtol=2e-2, atol=2e-2)

    latent = torch.randn(8, PATCH_SIZE, LATENT_DIM, device=device, dtype=dtype)
    eager_feedback = eager.encode_feedback(slots, latent)
    graph_feedback = graph.encode_feedback(slots, latent)
    torch.testing.assert_close(graph_feedback, eager_feedback, rtol=2e-2, atol=2e-2)
    assert graph._graph_replays == {"meanflow": 1, "semantic_encoder": 1}
    assert not graph._graph_misses
    assert graph._dit_contiguous_view_steps == NFE
