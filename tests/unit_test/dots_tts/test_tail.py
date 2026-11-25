# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang_omni.models.dots_tts.compat import import_dots_tts
from sglang_omni.platforms import current_platform

import_dots_tts()

from dots_tts.models.dots_tts.config import _DiTConfig, _EncoderConfig
from dots_tts.modules.backbone.dit import DiT
from dots_tts.modules.backbone.encoder import VAESemanticEncoder

from sglang_omni.models.dots_tts import tail

FM_HIDDEN = 32
LATENT_DIM = 6
PATCH_SIZE = 2
NFE = 2


@pytest.mark.parametrize(
    "platform_type,device_type",
    [
        ("cpu", "cpu"),
        ("cuda", "cuda"),
        ("rocm", "cuda"),
        ("xpu", "xpu"),
        ("npu", "npu"),
        ("npu", "xpu"),
    ],
)
def test_attention_context_preserves_policy_outside_npu(
    monkeypatch: pytest.MonkeyPatch, platform_type: str, device_type: str
) -> None:
    monkeypatch.setattr(
        tail,
        "current_platform",
        SimpleNamespace(
            is_npu=lambda: platform_type == "npu", device_type=platform_type
        ),
    )
    npu_context = platform_type == "npu" and device_type == "npu"
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        with tail.attention_context(SimpleNamespace(type=device_type)):
            assert torch.backends.cuda.flash_sdp_enabled() == npu_context
            assert torch.backends.cuda.mem_efficient_sdp_enabled() != npu_context
            assert torch.backends.cuda.math_sdp_enabled() != npu_context
        assert torch.backends.cuda.flash_sdp_enabled()
        assert not torch.backends.cuda.mem_efficient_sdp_enabled()
        assert not torch.backends.cuda.math_sdp_enabled()


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


@pytest.mark.parametrize(
    "device_type",
    [
        "cpu",
        pytest.param(
            "npu",
            marks=[
                pytest.mark.accelerator,
                pytest.mark.skipif(
                    not current_platform.is_npu(), reason="requires NPU"
                ),
            ],
        ),
    ],
)
def test_tail_rotary_preserves_float32_without_changing_other_instances(
    monkeypatch: pytest.MonkeyPatch, device_type: str
) -> None:
    unrelated_rotary = tail.RotaryEmbedding(8)
    original_forward = type(unrelated_rotary).forward
    device = torch.device(device_type)
    acoustic_tail = build_tail(TailModel().to(device), slots=1, device=device)
    if device_type == "cpu":
        assert acoustic_tail.dit_rotary.forward.__func__ is original_forward
        return
    else:
        pass
    assert acoustic_tail.dit_rotary.forward.__func__ is tail.rotary_forward
    positions = torch.tensor(
        [0, 255, 256, 257, 4095, 4096, 4097], dtype=torch.float32, device=device
    )
    expected = original_forward(acoustic_tail.dit_rotary, positions)
    original_einsum = torch.einsum
    autocast_states: list[bool] = []

    def observed_einsum(
        equation: str, position_values: torch.Tensor, frequencies: torch.Tensor
    ) -> torch.Tensor:
        autocast_states.append(torch.is_autocast_enabled(position_values.device.type))
        return original_einsum(equation, position_values, frequencies)

    monkeypatch.setattr(torch, "einsum", observed_einsum)
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        actual = acoustic_tail.dit_rotary(positions)
    assert autocast_states == [False]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert type(unrelated_rotary).forward is original_forward
    assert unrelated_rotary.forward.__func__ is original_forward


class TailModel(torch.nn.Module):
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


def patch_encoder() -> VAESemanticEncoder:
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


def build_tail(
    model: TailModel,
    *,
    slots: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    patch_capacity: int = 8,
    optimize: bool = False,
):
    encoder = patch_encoder().to(device=device, dtype=dtype)
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


def reference_meanflow(
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
    model = TailModel().eval()
    acoustic_tail = build_tail(model, slots=slots)
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
    expected = reference_meanflow(
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
    acoustic_tail = build_tail(TailModel().eval(), slots=2)
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
    acoustic_tail = build_tail(TailModel().eval(), slots=2, patch_capacity=8)
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
    device_module = SimpleNamespace(
        device=lambda _device: nullcontext(),
        empty_cache=lambda: None,
        mem_get_info=lambda _device=None: (4 << 30, 80 << 30),
    )
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda _device: device_module,
    )
    with pytest.raises(ValueError, match="admission failed at startup") as caught:
        tail.validate_acoustic_pool_memory(estimate, device=device)
    message = str(caught.value)
    assert "Parameters are not changed automatically" in message
    assert "Lower max_running_requests" in message
    assert "about 4 full-length slot(s)" in message

    # Enough free memory passes.
    monkeypatch.setattr(
        device_module,
        "mem_get_info",
        lambda _device=None: (40 << 30, 80 << 30),
    )
    tail.validate_acoustic_pool_memory(estimate, device=device)

    # CPU devices skip the accelerator memory gate.
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
    device_module = SimpleNamespace(
        device=lambda _device: nullcontext(),
        empty_cache=lambda: memory.update(free=12),
        mem_get_info=lambda _device=None: (memory["free"], 20),
    )
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda _device: device_module,
    )

    tail.validate_acoustic_pool_memory(
        estimate,
        device=torch.device("cuda:0"),
    )


def test_permuted_full_pool_matches_fragmented_gather_fallback() -> None:
    torch.manual_seed(1234)
    direct = build_tail(TailModel().eval(), slots=2)
    torch.manual_seed(1234)
    fallback = build_tail(TailModel().eval(), slots=3)
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
    model = TailModel().eval().to(torch.bfloat16)
    dit = tail.fuse_dit_for_inference(model)
    steps = torch.tensor([0.0, 0.5], dtype=torch.bfloat16)

    mods = dit.build_mods(steps, duration=torch.full_like(steps, 0.5))

    assert mods.dtype == torch.bfloat16


@pytest.mark.accelerator
@pytest.mark.skipif(not current_platform.is_npu(), reason="requires NPU")
def test_npu_eager_feedback_matches_full_encoder() -> None:
    device_module = torch.get_device_module(current_platform.device_type)
    if not device_module.is_available():
        pytest.skip("NPU is unavailable")
    else:
        pass
    device = torch.device(current_platform.device_type, device_module.current_device())
    torch.manual_seed(1234)
    model = TailModel().eval().to(device=device, dtype=torch.bfloat16)
    acoustic_tail = build_tail(model, slots=2, device=device, dtype=torch.bfloat16)
    reference_encoder = copy.deepcopy(acoustic_tail.encoder)
    slot = acoustic_tail.acquire_slot()
    latent_history: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(3):
            latent_patch = torch.randn(
                1, PATCH_SIZE, LATENT_DIM, device=device, dtype=torch.bfloat16
            )
            latent_history.append(latent_patch)
            expected = reference_encoder(torch.cat(latent_history, dim=1))[:, -1]
            actual = acoustic_tail.encode_feedback([slot], latent_patch)
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.accelerator
def test_batched_tail_cuda_graph_matches_eager_for_dynamic_slot_order() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(1234)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    eager_model = TailModel().eval().to(device=device, dtype=dtype)
    graph_model = copy.deepcopy(eager_model)
    torch.manual_seed(9)
    eager = build_tail(
        eager_model,
        slots=8,
        device=device,
        dtype=dtype,
        patch_capacity=33,
    )
    torch.manual_seed(9)
    graph = build_tail(
        graph_model,
        slots=8,
        device=device,
        dtype=dtype,
        patch_capacity=33,
        optimize=True,
    )

    for name in (
        "dit_k",
        "dit_v",
        "encoder_k",
        "encoder_v",
        "encoder_conv_tail",
        "window",
        "all_mods",
    ):
        eager_value = getattr(eager, name)
        eager_value.normal_(0, 0.05)
        getattr(graph, name).copy_(eager_value)
    for slot in range(8):
        eager._fm_seq_len[slot] = graph._fm_seq_len[slot] = (
            15  # noqa: leading-underscore  # production name
        )
        eager.encoder_seq_len[slot] = graph.encoder_seq_len[slot] = 4
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
    assert graph.graph_replays == {"meanflow": 1, "semantic_encoder": 1}
    assert not graph.graph_misses
    assert graph.dit_contiguous_view_steps == NFE


def test_validate_acoustic_pool_memory_gates_xpu_through_xpu_memory_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimate = tail.AcousticPoolMemoryEstimate(
        dit_kv_bytes=11 << 30,
        encoder_kv_bytes=2 << 30,
        scratch_bytes=4 << 30,
        aux_bytes=0,
        total_bytes=17 << 30,
        num_slots=16,
        patch_capacity=501,
        nfe=4,
        dtype=torch.bfloat16,
    )
    memory = {"free": 16 << 30, "released": 0}

    def cuda_untouched(device: torch.device | None = None) -> None:
        raise AssertionError("the XPU gate must not query CUDA")

    monkeypatch.setattr(torch.cuda, "device", cuda_untouched)
    monkeypatch.setattr(torch.cuda, "empty_cache", cuda_untouched)
    monkeypatch.setattr(torch.cuda, "mem_get_info", cuda_untouched)
    monkeypatch.setattr(torch.xpu, "device", lambda _device: nullcontext())
    monkeypatch.setattr(
        torch.xpu,
        "empty_cache",
        lambda: memory.update(released=memory["released"] + 1),
    )
    monkeypatch.setattr(
        torch.xpu,
        "mem_get_info",
        lambda _device=None: (memory["free"], 24 << 30),
    )
    device = torch.device("xpu:0")

    with pytest.raises(ValueError, match="admission failed at startup") as caught:
        tail.validate_acoustic_pool_memory(estimate, device=device)
    assert memory["released"] == 1
    assert "only 16.00 GiB is free on xpu:0" in str(caught.value)
    assert "Lower max_running_requests" in str(caught.value)

    memory["free"] = 40 << 30
    tail.validate_acoustic_pool_memory(estimate, device=device)
    assert memory["released"] == 2


def install_eager_graphs(
    acoustic_tail: tail.DotsTtsAcousticTail,
    batch_sizes: tuple[int, ...],
    patches: int = 16,
) -> None:
    """Stand-in graphs that rerun the captured step eagerly on replay.

    They read the same static inputs a CUDA graph would, so the selection,
    padding and row mapping around a replay run on any device.
    """

    def install(
        graphs: dict[tuple[int, int], tail.CapturedTailGraph],
        batch_size: int,
        capacity: int,
        kind: str,
    ) -> None:
        inputs, run = acoustic_tail.build_graph_step(batch_size, capacity, kind=kind)
        output = torch.zeros_like(run())
        graphs[(batch_size, capacity)] = tail.CapturedTailGraph(
            SimpleNamespace(replay=lambda: output.copy_(run())), inputs, output
        )

    with torch.no_grad():
        for batch_size in batch_sizes:
            install(
                acoustic_tail.meanflow_graphs,
                batch_size,
                patches * acoustic_tail.spec.unit_len,
                "meanflow",
            )
            install(
                acoustic_tail.encoder_graphs,
                batch_size,
                patches * acoustic_tail.encoder_block,
                "semantic_encoder",
            )


def paired_tails(
    slots: int,
    *,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    optimize: bool = False,
    eager_graph_batches: tuple[int, ...] = (),
) -> tuple[tail.DotsTtsAcousticTail, tail.DotsTtsAcousticTail]:
    """An eager reference tail and a graph tail with the same weights."""
    torch.manual_seed(1234)
    model = TailModel().eval().to(device=device, dtype=dtype)
    tails = []
    for use_graphs in (False, True):
        torch.manual_seed(9)
        tails.append(
            build_tail(
                copy.deepcopy(model),
                slots=slots,
                device=device,
                dtype=dtype,
                patch_capacity=33,
                optimize=optimize and use_graphs,
            )
        )
    reference, padded = tails
    install_eager_graphs(padded, eager_graph_batches)
    return reference, padded


def prime_tails(
    reference: tail.DotsTtsAcousticTail,
    padded: tail.DotsTtsAcousticTail,
    live: list[int],
) -> None:
    """Admit the live slots on both tails and give them the same history."""
    unit = reference.spec.unit_len
    block = reference.encoder_block
    for acoustic_tail in (reference, padded):
        for slot in range(acoustic_tail.spec.num_slots):
            acoustic_tail.release_slot(slot)
        for slot in [
            acoustic_tail.acquire_slot() for _ in range(acoustic_tail.spec.num_slots)
        ]:
            if slot in live:
                pass
            else:
                acoustic_tail.release_slot(slot)
    for name in (
        "dit_k",
        "dit_v",
        "encoder_k",
        "encoder_v",
        "encoder_conv_tail",
        "window",
        "all_mods",
    ):
        value = getattr(reference, name)
        value.normal_(0, 0.05)
        getattr(padded, name).copy_(value)
    for row, slot in enumerate(live):
        for acoustic_tail in (reference, padded):
            # Rows of one batch sit at different history lengths.
            acoustic_tail._fm_seq_len[slot] = (
                3 + row % 3
            ) * unit  # noqa: leading-underscore  # production name
            acoustic_tail.encoder_seq_len[slot] = (1 + row % 3) * block
            acoustic_tail.initialize_slot_rng(slot, 100 + slot)


def assert_cycle_matches(
    reference: tail.DotsTtsAcousticTail,
    padded: tail.DotsTtsAcousticTail,
    slots: list[int],
    *,
    tolerance: float,
) -> None:
    hidden = torch.randn(
        len(slots), FM_HIDDEN, device=reference.device, dtype=reference.dtype
    )
    expected = reference.sample_patches(slots, fm_hidden_rows=hidden)
    actual = padded.sample_patches(slots, fm_hidden_rows=hidden)
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    # Both tails encode the reference latent so one cycle's rounding does not
    # compound into the next comparison.
    expected_feedback = reference.encode_feedback(slots, expected)
    actual_feedback = padded.encode_feedback(slots, expected)
    torch.testing.assert_close(
        actual_feedback, expected_feedback, rtol=tolerance, atol=tolerance
    )


def test_graph_selection_pads_a_partial_batch_with_idle_slots() -> None:
    _, padded = paired_tails(8, eager_graph_batches=(4, 8))
    graphs = padded.meanflow_graphs
    capacity = 4 * padded.spec.unit_len

    # One live row would be padded 4x, past GRAPH_BATCH_PADDING_FACTOR.
    assert padded.select_graph(graphs, [5], capacity) is None

    graph, row_index = padded.select_graph(graphs, [5, 2], capacity)
    members = graph.inputs["slots"].tolist()
    assert len(members) == 4
    assert members == sorted(set(members))
    assert [members[row] for row in row_index.tolist()] == [5, 2]

    graph, row_index = padded.select_graph(graphs, [5, 2, 1, 0, 3], capacity)
    assert graph.inputs["slots"].tolist() == list(range(8))
    assert row_index.tolist() == [5, 2, 1, 0, 3]

    # A history past the largest captured context bucket has no graph.
    assert padded.select_graph(graphs, [5, 2], 17 * padded.spec.unit_len) is None


@pytest.mark.parametrize(
    "live,pad_rows",
    [
        ([6, 1, 3], 1),  # gathers into the batch 4 graph
        ([7, 2, 5, 0, 6], 3),  # fills the whole-pool batch 8 graph
        ([4, 1, 6, 3], 0),  # exact batch, no padding
    ],
)
def test_padded_graph_replay_matches_eager_for_a_partial_batch(
    live: list[int], pad_rows: int
) -> None:
    reference, padded = paired_tails(8, eager_graph_batches=(4, 8))
    prime_tails(reference, padded, live)

    cycles = 3
    for _ in range(cycles):
        assert_cycle_matches(reference, padded, live, tolerance=2e-4)

    assert padded.graph_replays == {"meanflow": cycles, "semantic_encoder": cycles}
    assert not padded.graph_misses
    assert padded.graph_pad_rows == {
        "meanflow": cycles * pad_rows,
        "semantic_encoder": cycles * pad_rows,
    }
    assert not reference.graph_replays
    for pool in padded.pool_tensors():
        assert torch.isfinite(pool).all()


def test_slot_that_padded_a_replay_serves_the_next_request_cleanly() -> None:
    reference, padded = paired_tails(4, eager_graph_batches=(4,))
    live = [2, 0, 1]
    prime_tails(reference, padded, live)
    for _ in range(2):
        assert_cycle_matches(reference, padded, live, tolerance=2e-4)
    assert padded.graph_pad_rows["meanflow"] == 2

    # Slot 3 padded both cycles on the graph tail and never ran on the
    # reference tail. A request admitted into it must not see the difference.
    grid = torch.linspace(0.0, 1.0, NFE + 1)
    mods = reference.dit.build_mods(
        grid[:-1], duration=grid[1:] - grid[:-1], g_cond=torch.randn(1, FM_HIDDEN)
    )
    history = torch.randn(2 * reference.spec.unit_len, FM_HIDDEN)
    for acoustic_tail in (reference, padded):
        assert acoustic_tail.acquire_slot() == 3
        acoustic_tail.seed_fm_history(3, fm_rows=history, all_mods=mods)
        acoustic_tail.initialize_slot_rng(3, 7)

    for _ in range(2):
        assert_cycle_matches(reference, padded, [3, 0, 2, 1], tolerance=2e-4)
    assert not padded.graph_misses


def test_admitted_slot_that_sits_a_step_out_is_never_used_as_padding() -> None:
    reference, padded = paired_tails(4, eager_graph_batches=(4,))
    prime_tails(reference, padded, [0, 1, 2, 3])
    waiting_window = padded.window[3].clone()
    waiting_conv_tail = padded.encoder_conv_tail[3].clone()
    waiting_keys = padded.dit_k[:, :, 3].clone()

    # Three rows fit the batch 4 graph, but the only other slot is owned by a
    # request that is not decoding this step, so the step must run eagerly.
    assert_cycle_matches(reference, padded, [0, 1, 2], tolerance=2e-4)

    assert padded.graph_misses == {"meanflow": 1, "semantic_encoder": 1}
    assert not padded.graph_replays
    assert torch.equal(padded.window[3], waiting_window)
    assert torch.equal(padded.encoder_conv_tail[3], waiting_conv_tail)
    assert torch.equal(padded.dit_k[:, :, 3], waiting_keys)


@pytest.mark.accelerator
def test_batched_tail_cuda_graph_pads_partial_batches_to_a_captured_bucket() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    reference, padded = paired_tails(
        16, device=device, dtype=torch.bfloat16, optimize=True
    )
    assert {batch for batch, _ in padded.meanflow_graphs} == {1, 2, 4, 8, 16}

    order = [9, 3, 14, 0, 7, 12, 5, 1, 15, 8, 2, 11, 6, 13, 4, 10]
    cycles = 2
    expected_pad_rows = 0
    row_counts = (1, 2, 3, 5, 8, 11, 13, 16)
    for rows in row_counts:
        live = order[:rows]
        prime_tails(reference, padded, live)
        for _ in range(cycles):
            assert_cycle_matches(reference, padded, live, tolerance=2e-2)
        bucket = min(batch for batch in (1, 2, 4, 8, 16) if batch >= rows)
        expected_pad_rows += cycles * (bucket - rows)

    replays = cycles * len(row_counts)
    assert padded.graph_replays == {"meanflow": replays, "semantic_encoder": replays}
    assert not padded.graph_misses
    assert padded.graph_pad_rows == {
        "meanflow": expected_pad_rows,
        "semantic_encoder": expected_pad_rows,
    }
    for pool in padded.pool_tensors():
        assert torch.isfinite(pool.float()).all()
