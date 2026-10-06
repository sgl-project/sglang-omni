# SPDX-License-Identifier: Apache-2.0
"""Incremental AudioVAE decoding from per-slot history matches the windowed decode."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from sglang_omni.models.dots_tts.alias_free import (
    FILTER_TAPS,
    alias_free_channels_last,
    install_alias_free_fusion,
)
from sglang_omni.models.dots_tts.incremental_codec import DotsIncrementalDecoder
from sglang_omni.models.dots_tts.incremental_codec_cuda_graph import (
    DotsIncrementalCodecCudaGraphRunner,
)
from sglang_omni.models.dots_tts.vocoder_slot_pool import DotsVocoderSlotPool
from tests.unit_test.fixtures.accelerator import require_cuda

if TYPE_CHECKING:
    from dots_tts.modules.vocoder.vocoder_inference import VocoderInference
else:
    pass

PATCH = 4
MERGE = 4


def tiny_inference() -> VocoderInference:
    try:
        from sglang_omni.models.dots_tts.compat import import_dots_tts

        import_dots_tts()
        from dots_tts.modules.vocoder.bigvgan import AudioVAE
        from dots_tts.modules.vocoder.config import AudioVAEConfig
        from dots_tts.modules.vocoder.vocoder_inference import VocoderInference
    except ImportError as exc:
        pytest.skip(f"dots_tts unavailable: {exc}")
    else:
        pass
    torch.manual_seed(0)
    config = AudioVAEConfig(
        sample_rate=1600,
        upsample_rates=[4, 2],
        upsample_kernel_sizes=[8, 4],
        upsample_initial_channel=32,
        resblock="1",
        resblock_kernel_sizes=[3, 5],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
        downsample_rates=[2, 4],
        downsample_channels=[4, 8, 16],
        latent_dim=8,
        causal=True,
        mi_num_layers=1,
        causal_encoder=True,
        use_bias_at_final=False,
        use_tanh_at_final=False,
    )
    vocoder = AudioVAE(config).eval()
    vocoder.remove_weight_norm()
    return VocoderInference(vocoder)


def attach_incremental_codec(
    pool: DotsVocoderSlotPool, inference: VocoderInference, max_batch_size: int
) -> DotsIncrementalCodecCudaGraphRunner:
    decoder = DotsIncrementalDecoder(inference)
    pool.incremental_codec = DotsIncrementalCodecCudaGraphRunner(
        decoder,
        decoder.new_state_arena(pool.num_slots),
        max_batch_size=max_batch_size,
        warm_fresh_frames=[PATCH * patches for patches in range(1, MERGE + 1)],
        cold_window_frames=sorted({8, 16, 24, 32, pool.window_size}),
        capture_graphs=False,
    )
    return pool.incremental_codec


def schedule(total_patches: int) -> list[int]:
    """Two single-patch steps for first audio, then merged steps, like the coalesced pump."""
    steps: list[int] = []
    taken = 0
    while taken < total_patches:
        size = 1 if len(steps) < 2 else min(MERGE, total_patches - taken)
        steps.append(size)
        taken += size
    return steps


def decode(
    pool: DotsVocoderSlotPool,
    streams: list[torch.Tensor],
    *,
    fuse_final: bool = False,
) -> list[torch.Tensor]:
    """Run staggered streams through one pool; rows of different ages share equal-length steps.

    fuse_final runs each stream's last latents as a final step of its own.
    """
    plans = [schedule(stream.shape[1] // PATCH) for stream in streams]
    slots = [pool.acquire() for _ in streams]
    cursor = [0] * len(streams)
    progress = [0] * len(streams)
    chunks: list[list[torch.Tensor]] = [[] for _ in streams]
    round_index = 0
    while any(cursor[index] < len(plans[index]) for index in range(len(streams))):
        active = [
            index
            for index in range(len(streams))
            if index <= round_index and cursor[index] < len(plans[index])
        ]
        by_size: dict[int, list[int]] = {}
        for index in active:
            by_size.setdefault(plans[index][cursor[index]], []).append(index)
        for size, members in sorted(by_size.items()):
            groups = [
                [index for index in members if cursor[index] + 1 < len(plans[index])],
                [index for index in members if cursor[index] + 1 == len(plans[index])],
            ]
            if not fuse_final:
                groups = [members, []]
            else:
                pass
            for final, group in enumerate(groups):
                if not group:
                    continue
                else:
                    pass
                latents: dict[int, torch.Tensor] = {}
                for index in group:
                    frames = size * PATCH
                    latents[slots[index]] = streams[index][
                        :, progress[index] : progress[index] + frames
                    ]
                    progress[index] += frames
                    cursor[index] += 1
                out = pool.step(latents, final=bool(final))
                for index in group:
                    chunks[index].append(out[slots[index]].reshape(-1))
        round_index += 1
    for index, slot in enumerate(slots):
        chunks[index].append(pool.flush(slot).reshape(-1))
        pool.release(slot)
    return [torch.cat(parts) for parts in chunks]


@torch.no_grad()
def test_incremental_codec_matches_window_decode() -> None:
    inference = tiny_inference()
    window_pool = DotsVocoderSlotPool(inference, num_slots=2, chunk_size=PATCH * MERGE)
    incremental_pool = DotsVocoderSlotPool(
        inference, num_slots=2, chunk_size=PATCH * MERGE
    )
    codec = attach_incremental_codec(incremental_pool, inference, max_batch_size=2)
    generator = torch.Generator().manual_seed(1)
    latent_dim = int(inference.vocoder.h.latent_dim)
    streams = [
        torch.randn(1, frames, latent_dim, generator=generator) for frames in (96, 60)
    ]

    expected = decode(window_pool, streams)
    observed = decode(incremental_pool, streams)

    for reference, candidate in zip(expected, observed, strict=True):
        assert candidate.shape == reference.shape
        error = (candidate - reference).norm() / reference.norm()
        assert error < 1e-5, error
    assert codec.decoder.warm_history_frames < 96 // 2


@torch.no_grad()
def test_incremental_codec_final_step_matches_window_decode() -> None:
    inference = tiny_inference()
    window_pool = DotsVocoderSlotPool(inference, num_slots=2, chunk_size=PATCH * MERGE)
    incremental_pool = DotsVocoderSlotPool(
        inference, num_slots=2, chunk_size=PATCH * MERGE
    )
    attach_incremental_codec(incremental_pool, inference, max_batch_size=2)
    generator = torch.Generator().manual_seed(4)
    latent_dim = int(inference.vocoder.h.latent_dim)
    streams = [
        torch.randn(1, frames, latent_dim, generator=generator) for frames in (92, 68)
    ]

    expected = decode(window_pool, streams)
    observed = decode(incremental_pool, streams, fuse_final=True)

    for reference, candidate in zip(expected, observed, strict=True):
        assert candidate.shape == reference.shape
        error = (candidate - reference).norm() / reference.norm()
        assert error < 1e-5, error


@pytest.mark.parametrize("cancel_early", [False, True])
@torch.no_grad()
def test_incremental_codec_reused_slot_matches_window_decode(
    cancel_early: bool,
) -> None:
    inference = tiny_inference()
    window_pool = DotsVocoderSlotPool(inference, num_slots=1, chunk_size=PATCH * MERGE)
    incremental_pool = DotsVocoderSlotPool(
        inference, num_slots=1, chunk_size=PATCH * MERGE
    )
    attach_incremental_codec(incremental_pool, inference, max_batch_size=1)
    generator = torch.Generator().manual_seed(2)
    latent_dim = int(inference.vocoder.h.latent_dim)
    prior_frames = PATCH if cancel_early else 96
    prior_stream = torch.randn(1, prior_frames, latent_dim, generator=generator)
    for pool in (window_pool, incremental_pool):
        slot = pool.acquire()
        consumed_frames = 0
        for patches in schedule(prior_frames // PATCH):
            frames = patches * PATCH
            pool.step(
                {slot: prior_stream[:, consumed_frames : consumed_frames + frames]}
            )
            consumed_frames += frames
        if cancel_early:
            # note (0xtoward): abort releases the slot without flushing its pending tail.
            pass
        else:
            pool.flush(slot)
        pool.release(slot)

    stream = torch.randn(1, 60, latent_dim, generator=generator)
    [reference] = decode(window_pool, [stream])
    [candidate] = decode(incremental_pool, [stream])

    assert candidate.shape == reference.shape
    error = (candidate - reference).norm() / reference.norm()
    assert error < 1e-5, error


def cuda_fused_inference(monkeypatch: pytest.MonkeyPatch) -> VocoderInference:
    require_cuda()
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    inference = tiny_inference()
    inference.vocoder.to("cuda")
    assert install_alias_free_fusion(inference.vocoder.decoder) > 0
    return inference


@pytest.mark.accelerator
@pytest.mark.parametrize("padded", [False, True])
@torch.no_grad()
def test_channels_last_alias_free_matches_fused_kernel(
    monkeypatch: pytest.MonkeyPatch, padded: bool
) -> None:
    inference = cuda_fused_inference(monkeypatch)
    activation = inference.vocoder.decoder.resblocks[0].activations[0]
    channels = activation.frozen_alpha.numel()
    generator = torch.Generator(device="cuda").manual_seed(3)
    value = torch.randn(2, channels, 70, device="cuda", generator=generator)
    bias = torch.randn(channels, device="cuda", generator=generator)
    for folded_bias in (None, bias):
        fused = activation(value, bias=folded_bias)
        # note (0xtoward): without padding, the first FILTER_TAPS - 1 outputs would
        # read frames before the input start, so they are not produced.
        expected = fused if padded else fused[..., FILTER_TAPS - 1 :]
        observed = alias_free_channels_last(
            activation, value.transpose(1, 2).contiguous(), folded_bias, padded=padded
        ).transpose(1, 2)
        assert torch.equal(observed, expected)


@pytest.mark.accelerator
@torch.no_grad()
def test_channels_last_warm_step_matches_ncl_warm_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference = cuda_fused_inference(monkeypatch)
    ncl_decoder = DotsIncrementalDecoder(inference, channels_last=False)
    channels_last_decoder = DotsIncrementalDecoder(inference)
    assert channels_last_decoder.use_channels_last
    assert not ncl_decoder.use_channels_last
    generator = torch.Generator(device="cuda").manual_seed(5)
    ncl_arena = ncl_decoder.new_state_arena(3)
    for history in ncl_arena.tensors():
        history.copy_(torch.randn(history.shape, device="cuda", generator=generator))
    channels_last_arena = channels_last_decoder.new_state_arena(3)
    for target, source in zip(channels_last_arena.tensors(), ncl_arena.tensors()):
        target.copy_(source)
    slot_index = torch.tensor([2, 0, 1], device="cuda")
    for frames_count in (PATCH, 3 * PATCH):
        frames = torch.randn(
            3,
            ncl_decoder.latent_channels,
            frames_count,
            device="cuda",
            generator=generator,
        )
        expected = ncl_decoder.warm_forward(ncl_arena, frames, slot_index)
        observed = channels_last_decoder.warm_forward(
            channels_last_arena, frames, slot_index
        )
        torch.testing.assert_close(observed, expected, rtol=1e-4, atol=1e-5)
    for observed, expected in zip(channels_last_arena.tensors(), ncl_arena.tensors()):
        torch.testing.assert_close(observed, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.accelerator
@torch.no_grad()
def test_channels_last_cold_step_matches_ncl_cold_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference = cuda_fused_inference(monkeypatch)
    ncl_decoder = DotsIncrementalDecoder(inference, channels_last=False)
    channels_last_decoder = DotsIncrementalDecoder(inference)
    assert channels_last_decoder.use_channels_last
    assert not ncl_decoder.use_channels_last
    generator = torch.Generator(device="cuda").manual_seed(7)
    ncl_arena = ncl_decoder.new_state_arena(3)
    channels_last_arena = channels_last_decoder.new_state_arena(3)
    slot_index = torch.tensor([2, 0, 1], device="cuda")
    window_frames = 6 * PATCH
    valid_frames = torch.tensor([window_frames, 2, 4 * PATCH + 1], device="cuda")
    # note (0xtoward): pool windows are left-aligned and zero past each row's valid frames.
    real_frames = torch.arange(window_frames, device="cuda") < valid_frames.unsqueeze(1)
    window = torch.randn(
        3,
        ncl_decoder.latent_channels,
        window_frames,
        device="cuda",
        generator=generator,
    ) * real_frames.unsqueeze(1)
    stable_frames = valid_frames - ncl_decoder.lookahead
    expected = ncl_decoder.cold_forward(
        ncl_arena, window, slot_index, stable_frames, valid_frames
    )
    observed = channels_last_decoder.cold_forward(
        channels_last_arena, window, slot_index, stable_frames, valid_frames
    )
    torch.testing.assert_close(observed, expected, rtol=1e-4, atol=1e-5)
    for observed_history, expected_history in zip(
        channels_last_arena.tensors(), ncl_arena.tensors()
    ):
        torch.testing.assert_close(
            observed_history, expected_history, rtol=1e-4, atol=1e-5
        )
