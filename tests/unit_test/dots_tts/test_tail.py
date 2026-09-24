# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import logging
from collections import Counter
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sglang_omni.models.dots_tts.compat import import_dots_tts

import_dots_tts()

from dots_tts.models.dots_tts.config import _DiTConfig, _EncoderConfig
from dots_tts.modules.backbone.dit import DiT
from dots_tts.modules.backbone.encoder import VAESemanticEncoder

from sglang_omni.models.dots_tts import tail

FM_HIDDEN = 32
LATENT_DIM = 6
PATCH_SIZE = 2
NFE = 2
SLOT_DIMS = {
    "dit_k": 2,
    "dit_v": 2,
    "encoder_k": 1,
    "encoder_v": 1,
    "encoder_conv_tail": 0,
    "window": 0,
    "all_mods": 1,
}


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
    pad_to_bucket: bool = False,
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
        pad_to_bucket=pad_to_bucket,
    )


def _copy_live_state(source, destination) -> None:
    slots = source.spec.num_slots
    for name, slot_dim in SLOT_DIMS.items():
        source_value = getattr(source, name)
        source_value.normal_(0, 0.05)
        destination_value = getattr(destination, name)
        live = [slice(None)] * source_value.ndim
        live[slot_dim] = slice(0, slots)
        destination_value[tuple(live)].copy_(source_value)


def _assert_live_state_close(actual, expected) -> None:
    slots = expected.spec.num_slots
    for name, slot_dim in SLOT_DIMS.items():
        actual_value = getattr(actual, name)
        expected_value = getattr(expected, name)
        live = [slice(None)] * expected_value.ndim
        live[slot_dim] = slice(0, slots)
        torch.testing.assert_close(
            actual_value[tuple(live)],
            expected_value,
            rtol=2e-2,
            atol=2e-2,
        )
    for slot in range(slots):
        assert torch.equal(
            actual.generators[slot].get_state(),
            expected.generators[slot].get_state(),
        )
    assert actual._fm_seq_len == expected._fm_seq_len
    assert actual.encoder_seq_len == expected.encoder_seq_len


def _fill_reserved_state(acoustic_tail, value: float) -> None:
    slot = acoustic_tail.spec.num_slots
    for name, slot_dim in SLOT_DIMS.items():
        tensor = getattr(acoustic_tail, name)
        if tensor.size(slot_dim) > slot:
            tensor.select(slot_dim, slot).fill_(value)


@pytest.mark.parametrize(
    ("slots", "enabled", "expected"),
    [
        (1, False, (1,)),
        (2, True, (1, 2)),
        (4, False, (1, 4)),
        (8, False, (1, 4, 8)),
        (12, False, (1, 4, 8)),
        (12, True, (1, 4, 8, 12)),
        (24, False, (1, 4, 8, 16)),
        (24, True, (1, 4, 8, 16, 24)),
        (32, True, (1, 4, 8, 16, 32)),
    ],
)
def test_graph_batch_buckets_include_deployment_maximum(
    slots: int, enabled: bool, expected: tuple[int, ...]
) -> None:
    assert tail.graph_batch_buckets(slots, include_maximum=enabled) == expected


@pytest.mark.parametrize("optimize", [False, True])
def test_padding_request_does_not_allocate_on_cpu(optimize: bool) -> None:
    acoustic_tail = _build_tail(
        _TailModel().eval(),
        slots=12,
        patch_capacity=33,
        optimize=optimize,
        pad_to_bucket=True,
    )

    assert acoustic_tail.cuda_graph_enabled is False
    assert acoustic_tail.pad_to_bucket is False
    assert acoustic_tail.graph_batch_buckets == (1, 4, 8)
    for name, slot_dim in SLOT_DIMS.items():
        assert getattr(acoustic_tail, name).shape[slot_dim] == 12
    estimate = acoustic_tail.pool_memory_estimate(acoustic_tail.mods_width)
    assert estimate.num_slots == 12
    assert estimate.total_bytes == acoustic_tail.allocated_pool_bytes()


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
    assert acoustic_tail.dit_contiguous_view_steps == (NFE if slots == 1 else 0)


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
    estimate = acoustic_tail.pool_memory_estimate(acoustic_tail.mods_width)
    assert estimate.total_bytes == acoustic_tail.allocated_pool_bytes()
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
        dit_layers=acoustic_tail.dit_layers,
        dit_heads=acoustic_tail.dit_heads,
        dit_head_dim=acoustic_tail.dit_head_dim,
        encoder_layers=acoustic_tail.encoder_layers,
        encoder_heads=acoustic_tail.encoder_heads,
        encoder_head_dim=acoustic_tail.encoder_head_dim,
        encoder_block=acoustic_tail.encoder_block,
        encoder_conv_channels=int(acoustic_tail.encoder.ds_proj.in_channels),
        encoder_conv_padding=int(acoustic_tail.encoder.ds_proj.left_padding),
        mods_width=acoustic_tail.mods_width,
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
    assert direct.dit_contiguous_view_steps == NFE
    assert fallback.dit_contiguous_view_steps == 0


def test_request_release_forgets_slot_before_it_can_be_reused() -> None:
    from sglang_omni.models.dots_tts.flow_head import DotsTTSFlowHead

    released = []
    flow = SimpleNamespace(tail=SimpleNamespace(release_slot=released.append))
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


@pytest.mark.accelerator
@pytest.mark.parametrize("slots", [1, 8])
def test_batched_tail_cuda_graph_matches_eager_for_dynamic_slot_order(
    slots: int,
) -> None:
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
        slots=slots,
        device=device,
        dtype=dtype,
        patch_capacity=33,
    )
    torch.manual_seed(9)
    graph = _build_tail(
        graph_model,
        slots=slots,
        device=device,
        dtype=dtype,
        patch_capacity=33,
        optimize=True,
    )

    _copy_live_state(eager, graph)
    for slot in range(slots):
        eager._fm_seq_len[slot] = graph._fm_seq_len[slot] = 15
        eager.encoder_seq_len[slot] = graph.encoder_seq_len[slot] = 4
        eager.initialize_slot_rng(slot, 100 + slot)
        graph.initialize_slot_rng(slot, 100 + slot)

    slot_order = [0] if slots == 1 else [7, 2, 5, 0, 6, 1, 4, 3]
    hidden = torch.randn(slots, FM_HIDDEN, device=device, dtype=dtype)
    eager_latent = eager.sample_patches(slot_order, fm_hidden_rows=hidden)
    graph_latent = graph.sample_patches(slot_order, fm_hidden_rows=hidden)
    torch.testing.assert_close(graph_latent, eager_latent, rtol=2e-2, atol=2e-2)

    latent = torch.randn(slots, PATCH_SIZE, LATENT_DIM, device=device, dtype=dtype)
    eager_feedback = eager.encode_feedback(slot_order, latent)
    graph_feedback = graph.encode_feedback(slot_order, latent)
    torch.testing.assert_close(graph_feedback, eager_feedback, rtol=2e-2, atol=2e-2)
    assert graph.graph_replays == {"meanflow": 1, "semantic_encoder": 1}
    assert not graph.graph_misses
    assert graph.dit_contiguous_view_steps == NFE


@pytest.mark.accelerator
@pytest.mark.parametrize(("slots", "first_rows"), [(8, 5), (12, 9), (24, 17)])
def test_padded_tail_replay_matches_eager_and_bin_slot_stays_reusable(
    slots: int, first_rows: int
) -> None:
    """Padded, exact, and uncaptured-context execution preserve request state."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(1234)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    eager_model = _TailModel().eval().to(device=device, dtype=dtype)
    graph_model = copy.deepcopy(eager_model)
    torch.manual_seed(9)
    eager = _build_tail(
        eager_model, slots=slots, device=device, dtype=dtype, patch_capacity=40
    )
    torch.manual_seed(9)
    graph = _build_tail(
        graph_model,
        slots=slots,
        device=device,
        dtype=dtype,
        patch_capacity=40,
        optimize=True,
        pad_to_bucket=True,
    )

    _copy_live_state(eager, graph)
    assert [eager.acquire_slot() for _ in range(slots)] == list(range(slots))
    assert [graph.acquire_slot() for _ in range(slots)] == list(range(slots))
    for slot in range(slots):
        fm_length = 52 if slot in {slots - 3, slots - 1} else 15
        encoder_length = (
            18 if slot in {slots - 3, slots - 1} else 4
        ) * graph.encoder_block
        eager._fm_seq_len[slot] = graph._fm_seq_len[slot] = fm_length
        eager.encoder_seq_len[slot] = graph.encoder_seq_len[slot] = encoder_length
        eager.initialize_slot_rng(slot, 100 + slot)
        graph.initialize_slot_rng(slot, 100 + slot)

    # note (0xtoward): A uses the maximum-batch gather twin; B changes live
    # members and uses the largest permitted filler count (2 -> 4).
    first = list(reversed(range(first_rows)))
    second = [slots - 1, slots - 2]
    full = list(reversed(range(slots - 4, slots)))
    shared_graphs = (
        graph.meanflow_graphs[(4, 32 * graph.spec.unit_len)],
        graph.encoder_graphs[(4, 32 * graph.encoder_block)],
    )
    schedules = (first, second, first, full, second, [0, 1])
    for replay_index, slot_order in enumerate(schedules):
        if replay_index == 5:
            # note (0xtoward): These histories fit the pool but exceed every
            # captured context, so even a legal 2 -> 4 batch must run eagerly.
            for acoustic_tail in (eager, graph):
                for slot in slot_order:
                    acoustic_tail._fm_seq_len[slot] = (
                        32 * acoustic_tail.spec.unit_len + acoustic_tail.spec.window_len
                    )
                    acoustic_tail.encoder_seq_len[slot] = (
                        32 * acoustic_tail.encoder_block + 1
                    )
        _fill_reserved_state(graph, 0.25 + replay_index)
        hidden = torch.randn(len(slot_order), FM_HIDDEN, device=device, dtype=dtype)
        eager_latent = eager.sample_patches(slot_order, fm_hidden_rows=hidden)
        graph_latent = graph.sample_patches(slot_order, fm_hidden_rows=hidden)
        torch.testing.assert_close(graph_latent, eager_latent, rtol=2e-2, atol=2e-2)

        latent = torch.randn(
            len(slot_order), PATCH_SIZE, LATENT_DIM, device=device, dtype=dtype
        )
        eager_feedback = eager.encode_feedback(slot_order, latent)
        graph_feedback = graph.encode_feedback(slot_order, latent)
        torch.testing.assert_close(graph_feedback, eager_feedback, rtol=2e-2, atol=2e-2)
        _assert_live_state_close(graph, eager)
        if replay_index == 3:
            for captured in shared_graphs:
                assert captured.inputs["slots"].tolist() == full
        elif replay_index == 4:
            # note (0xtoward): Reusing the same full -> partial capture must
            # replace every stale live suffix before replaying filler rows.
            for captured in shared_graphs:
                assert captured.inputs["slots"].tolist() == second + [slots, slots]
                for name, value in captured.inputs.items():
                    if name != "slots":
                        assert torch.count_nonzero(value[2:]).item() == 0

    assert graph.graph_replays == {"meanflow": 5, "semantic_encoder": 5}
    assert graph.graph_padded_replays == {"meanflow": 4, "semantic_encoder": 4}
    assert graph.graph_misses == {"meanflow": 1, "semantic_encoder": 1}
    assert graph.dit_contiguous_view_steps == 0

    # note (0xtoward): Filler writes stay outside every allocatable slot.
    assert graph.pad_bin_slot == slots
    for name, slot_dim in SLOT_DIMS.items():
        expected_rows = (
            slots if name in {"dit_k", "dit_v", "encoder_k", "encoder_v"} else slots + 1
        )
        assert getattr(graph, name).size(slot_dim) == expected_rows
    assert eager.window.shape[0] == slots
    estimate = graph.pool_memory_estimate(graph.mods_width)
    assert estimate.num_slots == slots
    assert estimate.total_bytes == graph.allocated_pool_bytes()
    bystander = slots - 1
    assert bystander not in slot_order
    hidden_one = torch.randn(1, FM_HIDDEN, device=device, dtype=dtype)
    eager_one = eager.sample_patches([bystander], fm_hidden_rows=hidden_one)
    graph_one = graph.sample_patches([bystander], fm_hidden_rows=hidden_one)
    torch.testing.assert_close(graph_one, eager_one, rtol=2e-2, atol=2e-2)

    # note (0xtoward): Reacquiring a real row must not inherit filler state.
    eager.release_slot(bystander)
    graph.release_slot(bystander)
    assert eager.acquire_slot() == graph.acquire_slot() == bystander
    grid = torch.linspace(0.0, 1.0, NFE + 1, device=device, dtype=dtype)
    g_cond = torch.randn(1, FM_HIDDEN, device=device, dtype=dtype)
    mods = eager.dit.build_mods(grid[:-1], duration=grid[1:] - grid[:-1], g_cond=g_cond)
    history = torch.randn(
        2 * eager.spec.unit_len, FM_HIDDEN, device=device, dtype=dtype
    )
    for acoustic_tail in (eager, graph):
        acoustic_tail.seed_fm_history(bystander, fm_rows=history, all_mods=mods)
        acoustic_tail.initialize_slot_rng(bystander, 999)
    hidden_one = torch.randn(1, FM_HIDDEN, device=device, dtype=dtype)
    torch.testing.assert_close(
        graph.sample_patches([bystander], fm_hidden_rows=hidden_one),
        eager.sample_patches([bystander], fm_hidden_rows=hidden_one),
        rtol=2e-2,
        atol=2e-2,
    )
    _assert_live_state_close(graph, eager)


@pytest.mark.accelerator
@pytest.mark.parametrize(
    ("slots", "optimize", "pad_to_bucket", "buckets"),
    [
        (12, False, True, (1, 4, 8)),
        (12, True, False, (1, 4, 8)),
        (1, True, True, (1,)),
        (2, True, True, (1, 2)),
    ],
)
def test_inactive_padding_preserves_cuda_pool_storage(
    slots: int, optimize: bool, pad_to_bucket: bool, buckets: tuple[int, ...]
) -> None:
    """Disabled padding or buckets without gaps allocate no reserved row."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    model = _TailModel().eval().to(device=device, dtype=dtype)
    graph_model = copy.deepcopy(model)
    baseline = _build_tail(
        model, slots=slots, device=device, dtype=dtype, patch_capacity=33
    )
    actual = _build_tail(
        graph_model,
        slots=slots,
        device=device,
        dtype=dtype,
        patch_capacity=33,
        optimize=optimize,
        pad_to_bucket=pad_to_bucket,
    )

    assert not actual.pad_to_bucket
    assert actual.graph_batch_buckets == buckets
    assert not actual.meanflow_pad_graphs
    assert actual.allocated_pool_bytes() == baseline.allocated_pool_bytes()
    assert [pool.shape for pool in actual.pool_tensors()] == [
        pool.shape for pool in baseline.pool_tensors()
    ]


def _seed_single_slot_tail(patch_capacity: int) -> tuple[Any, int]:
    torch.manual_seed(7)
    model = _TailModel().eval()
    acoustic_tail = _build_tail(model, slots=1, patch_capacity=patch_capacity)
    unit = acoustic_tail.spec.unit_len
    g_cond = torch.randn(1, FM_HIDDEN)
    grid = torch.linspace(0.0, 1.0, NFE + 1)
    mods = acoustic_tail.dit.build_mods(
        grid[:-1], duration=grid[1:] - grid[:-1], g_cond=g_cond
    )
    slot = acoustic_tail.acquire_slot()
    acoustic_tail.seed_fm_history(
        slot, fm_rows=torch.randn(3 * unit, FM_HIDDEN), all_mods=mods
    )
    return acoustic_tail, slot


def _counter_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if "tail graph counters" in r.getMessage()]


@pytest.mark.parametrize(
    "graph_kind", ["meanflow_graphs", "meanflow_pad_graphs", "encoder_graphs"]
)
def test_tail_logs_graph_counters_every_50_steps(caplog, graph_kind: str) -> None:
    acoustic_tail, slot = _seed_single_slot_tail(patch_capacity=60)
    # A captured batch-8 bucket that batch-1 decode can never select: every
    # cycle is a real miss, which is exactly what the counters report.
    getattr(acoustic_tail, graph_kind)[(8, 16)] = object()

    with caplog.at_level(logging.DEBUG, logger=tail.logger.name):
        for step in range(50):
            latent = acoustic_tail.sample_patches(
                [slot], fm_hidden_rows=torch.randn(1, FM_HIDDEN)
            )
            acoustic_tail.encode_feedback([slot], latent)
            acoustic_tail.note_decode_cycle()
            if step == 48:
                assert not _counter_records(caplog)
        [periodic] = _counter_records(caplog)
        caplog.clear()
        acoustic_tail.log_graph_counters()
        [shutdown] = _counter_records(caplog)

    assert periodic.levelno == logging.DEBUG
    assert shutdown.levelno == logging.INFO
    record = periodic
    message = record.getMessage()
    assert "steps=50" in message
    assert "meanflow_replays=0" in message
    assert "meanflow_misses=50" in message
    assert "semantic_encoder_replays=0" in message
    assert "semantic_encoder_misses=50" in message
    assert "meanflow_padded_replays=0" in message
    assert "semantic_encoder_padded_replays=0" in message
    assert acoustic_tail.graph_misses == Counter(
        {"meanflow": 50, "semantic_encoder": 50}
    )
    assert acoustic_tail.graph_replays == Counter()


def test_tail_without_captured_graphs_logs_no_counters(caplog) -> None:
    acoustic_tail, slot = _seed_single_slot_tail(patch_capacity=60)
    assert not acoustic_tail.has_captured_graphs

    with caplog.at_level(logging.DEBUG, logger=tail.logger.name):
        for _ in range(50):
            latent = acoustic_tail.sample_patches(
                [slot], fm_hidden_rows=torch.randn(1, FM_HIDDEN)
            )
            acoustic_tail.encode_feedback([slot], latent)
            acoustic_tail.note_decode_cycle()
        acoustic_tail.log_graph_counters()

    assert acoustic_tail.tail_steps == 50
    assert acoustic_tail.graph_misses == Counter(
        {"meanflow": 50, "semantic_encoder": 50}
    )
    assert not _counter_records(caplog)
