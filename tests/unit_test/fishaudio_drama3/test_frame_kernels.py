# SPDX-License-Identifier: Apache-2.0
"""The fused frame embeddings, decode-row staging and in-frame decode step
reproduce the PyTorch paths they replace bit for bit."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from sglang_omni.models.fishaudio_drama3 import frame_kernels
from sglang_omni.models.fishaudio_drama3.audio_decoder import (
    CODEBOOK_TOP_K,
    AudioDecoder,
)
from sglang_omni.models.fishaudio_drama3.decode_state import Drama3DecodeStatePool
from sglang_omni.models.fishaudio_drama3.sampling import (
    RAS_WINDOW_FRAMES,
    sample_semantic_with_ras,
    sample_top_k_top_p,
)
from sglang_omni.models.fishaudio_drama3.sampling_kernels import (
    FIRST_CODEBOOK_HEAD_STREAM,
    CodebookSampler,
)
from sglang_omni.platforms import current_platform
from tests.unit_test.fishaudio_drama3.graph_fixtures import bits
from tests.unit_test.fishaudio_drama3.test_tower_kernels import (
    PagedTestAttention,
    build_decoder,
    require_validated_gpu,
    tower_steps,
)

ROW_COUNTS = [1, 5, 16]
accelerator = pytest.mark.skipif(
    not current_platform.is_cuda() or not torch.cuda.is_available(),
    reason="the frame kernels run on CUDA only",
)


def fused_decoder(device: torch.device) -> AudioDecoder:
    decoder = build_decoder(device)
    decoder.enable_fused_layers(device)
    assert decoder.fused_frame_embeddings
    return decoder


def random_frames(
    decoder: AudioDecoder, row_count: int, generator: torch.Generator
) -> torch.Tensor:
    """Frames with semantic and text main tokens and codes outside the table."""
    device = decoder.codebook_embeddings.weight.device
    frames = torch.randint(
        -3,
        decoder.codebook_size + 3,
        (row_count, 1 + decoder.num_codebooks),
        device=device,
        generator=generator,
    )
    semantic = torch.randint(
        decoder.semantic_begin_id,
        decoder.semantic_end_id + 1,
        (row_count,),
        device=device,
        generator=generator,
    )
    text = torch.randint(
        0, decoder.semantic_begin_id, (row_count,), device=device, generator=generator
    )
    is_semantic = (torch.rand(row_count, device=device, generator=generator) < 0.7) | (
        torch.arange(row_count, device=device) == 0
    )
    frames[:, 0] = torch.where(is_semantic, semantic, text)
    return frames


@accelerator
@pytest.mark.accelerator
@pytest.mark.parametrize("row_count", ROW_COUNTS)
def test_fused_frame_embeddings_match_the_plain_path_bit_for_bit(
    row_count: int,
) -> None:
    require_validated_gpu()
    device = torch.device("cuda")
    decoder = fused_decoder(device)
    generator = torch.Generator(device=device).manual_seed(row_count)
    frames = random_frames(decoder, row_count, generator)
    text_embedding = nn.Embedding(
        decoder.semantic_end_id + 1,
        decoder.codebook_embeddings.embedding_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        fused_backbone = decoder.backbone_input(frames, text_embedding)
        fused_shared = decoder.shared_input(frames)
        decoder.fused_frame_embeddings = False
        plain_backbone = decoder.backbone_input(frames, text_embedding)
        plain_shared = decoder.shared_input(frames)
    assert torch.equal(bits(fused_backbone), bits(plain_backbone))
    assert torch.equal(bits(fused_shared), bits(plain_shared))
    assert torch.equal(
        bits(plain_backbone),
        bits(decoder.embed_backbone(frames, text_embedding(frames[:, 0]))),
    )


@accelerator
@pytest.mark.accelerator
def test_fused_frame_embeddings_read_strided_frame_rows() -> None:
    require_validated_gpu()
    device = torch.device("cuda")
    decoder = fused_decoder(device)
    generator = torch.Generator(device=device).manual_seed(3)
    padded = random_frames(decoder, 12, generator)[::2]
    with torch.no_grad():
        fused = decoder.shared_input(padded)
        decoder.fused_frame_embeddings = False
        plain = decoder.shared_input(padded)
    assert torch.equal(bits(fused), bits(plain))


def staged_pool(
    decoder: AudioDecoder, slot_count: int, generator: torch.Generator
) -> Drama3DecodeStatePool:
    device = decoder.codebook_embeddings.weight.device
    pool = Drama3DecodeStatePool.allocate(
        request_slot_count=slot_count,
        frame_width=1 + decoder.num_codebooks,
        device=device,
    )
    pool.frames.copy_(random_frames(decoder, slot_count + 1, generator))
    pool.ras_windows.copy_(
        torch.randint(
            0, 1000, pool.ras_windows.shape, device=device, generator=generator
        )
    )
    pool.temperature.copy_(
        torch.rand(slot_count + 1, device=device, generator=generator)
    )
    pool.top_k.copy_(
        torch.randint(1, 100, (slot_count + 1,), device=device, generator=generator)
    )
    pool.top_p.copy_(torch.rand(slot_count + 1, device=device, generator=generator))
    pool.epsilon_cutoff.copy_(
        torch.rand(slot_count + 1, device=device, generator=generator)
    )
    return pool


def clone_pool(pool: Drama3DecodeStatePool) -> Drama3DecodeStatePool:
    return Drama3DecodeStatePool(
        frames=pool.frames.clone(),
        ras_windows=pool.ras_windows.clone(),
        temperature=pool.temperature.clone(),
        top_k=pool.top_k.clone(),
        top_p=pool.top_p.clone(),
        epsilon_cutoff=pool.epsilon_cutoff.clone(),
        row_indices=pool.row_indices.clone(),
    )


@accelerator
@pytest.mark.accelerator
@pytest.mark.parametrize("row_count", ROW_COUNTS)
def test_decode_row_staging_matches_index_select_and_index_copy(
    row_count: int,
) -> None:
    device = torch.device("cuda")
    decoder = build_decoder(device)
    slot_count = 24
    generator = torch.Generator(device=device).manual_seed(40 + row_count)
    pool = staged_pool(decoder, slot_count, generator)
    rows = (
        1 + torch.randperm(slot_count, device=device, generator=generator)[:row_count]
    )
    pool.row_indices[:row_count].copy_(rows)
    row_indices = pool.row_indices[:row_count]
    width = 1 + decoder.num_codebooks
    buffers = {
        "frames": torch.zeros((slot_count, width), dtype=torch.int64, device=device),
        "ras_windows": torch.zeros(
            (slot_count, RAS_WINDOW_FRAMES), dtype=torch.int64, device=device
        ),
        "temperature": torch.zeros(slot_count, device=device),
        "top_k": torch.zeros(slot_count, dtype=torch.int64, device=device),
        "top_p": torch.zeros(slot_count, device=device),
        "epsilon_cutoff": torch.zeros(slot_count, device=device),
    }
    frame_kernels.gather_decode_rows(
        row_indices=row_indices,
        state_frames=pool.frames,
        state_ras_windows=pool.ras_windows,
        state_temperature=pool.temperature,
        state_top_k=pool.top_k,
        state_top_p=pool.top_p,
        state_epsilon_cutoff=pool.epsilon_cutoff,
        **buffers,
    )
    for name, buffer in buffers.items():
        expected = getattr(pool, name).index_select(0, row_indices)
        assert torch.equal(buffer[:row_count], expected), name
        assert not torch.any(buffer[row_count:]), name

    sampled = random_frames(decoder, slot_count, generator)
    reference = clone_pool(pool)
    published = frame_kernels.publish_decode_frames(
        row_indices=row_indices,
        sampled_frames=sampled[:row_count],
        state_frames=pool.frames,
        state_ras_windows=pool.ras_windows,
    )
    expected_frames = sampled[:row_count].clone()
    windows = reference.ras_windows.index_select(0, row_indices)
    reference.frames.index_copy_(0, row_indices, expected_frames)
    reference.ras_windows.index_copy_(
        0, row_indices, torch.cat((windows[:, 1:], expected_frames[:, :1]), dim=1)
    )
    assert torch.equal(published, expected_frames)
    assert published.data_ptr() != sampled.data_ptr()
    assert torch.equal(pool.frames, reference.frames)
    assert torch.equal(pool.ras_windows, reference.ras_windows)


def previous_decode_step(
    decoder: AudioDecoder,
    sampler: CodebookSampler,
    semantic_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    input_frames: torch.Tensor,
    previous_main_tokens: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    attention: PagedTestAttention,
) -> torch.Tensor:
    """decode_step as it ran before the samplers wrote into the frame."""
    sampler.step.add_(1)
    main_tokens, codebook_zero = sample_semantic_with_ras(
        semantic_logits,
        decoder.main_token_candidate_ids,
        previous_main_tokens,
        temperature,
        top_k,
        top_p,
        decoder.semantic_begin_id,
        decoder.codebook_size,
        sampler,
    )
    label_is_semantic = decoder.is_semantic(main_tokens)
    hidden_states = decoder.run_shared_layers(
        hidden_states,
        codebook_zero,
        label_is_semantic,
        decoder.shared_input(input_frames),
        positions,
        attention,
        True,
    )
    codebook_top_k = top_k.clamp(max=CODEBOOK_TOP_K)
    codes = [codebook_zero]
    for residual_index, (layer, head) in enumerate(
        zip(decoder.residual_layers, decoder.codebook_heads)
    ):
        hidden_states = decoder.run_layer(
            layer,
            len(decoder.shared_layers) + residual_index,
            hidden_states,
            codes[-1],
            label_is_semantic,
            None,
            positions,
            attention,
            True,
        )
        codes.append(
            sample_top_k_top_p(
                head(hidden_states),
                temperature,
                codebook_top_k,
                top_p,
                sampler,
                FIRST_CODEBOOK_HEAD_STREAM + residual_index,
            )
        )
    return torch.stack([main_tokens, *codes], dim=-1)


@accelerator
@pytest.mark.accelerator
@pytest.mark.parametrize("row_count", ROW_COUNTS)
def test_the_in_frame_decode_step_samples_the_previous_frames(row_count: int) -> None:
    require_validated_gpu()
    device = torch.device("cuda")
    decoder = fused_decoder(device)
    sampler = CodebookSampler(step=decoder.sampling_step, seed=12345)
    decoder.sampler = sampler
    generator = torch.Generator(device=device).manual_seed(70 + row_count)
    step = tower_steps(decoder, row_count, seed=row_count, device=device)[1]
    candidate_count = decoder.main_token_candidate_ids.shape[0]
    semantic_logits = 3.0 * torch.randn(
        (row_count, candidate_count), device=device, generator=generator
    )
    input_frames = random_frames(decoder, row_count, generator)
    previous_main_tokens = decoder.main_token_candidate_ids[
        torch.randint(
            0,
            candidate_count,
            (row_count, RAS_WINDOW_FRAMES),
            device=device,
            generator=generator,
        )
    ]
    temperature = 0.5 + torch.rand(row_count, device=device, generator=generator)
    top_k = torch.randint(1, 80, (row_count,), device=device, generator=generator)
    top_p = 0.5 + 0.5 * torch.rand(row_count, device=device, generator=generator)
    arguments = (
        semantic_logits,
        step.hidden_states,
        step.positions,
        input_frames,
        previous_main_tokens,
        temperature,
        top_k,
        top_p,
    )
    results = []
    with torch.no_grad():
        for run in ("previous", "in frame", "in frame into out"):
            attention = PagedTestAttention(decoder=decoder, device=device)
            attention.bind(step.slots, step.history, step.history_lengths)
            sampler.step.fill_(7)
            if run == "previous":
                frames = previous_decode_step(decoder, sampler, *arguments, attention)
            elif run == "in frame":
                frames = decoder.decode_step(*arguments, attention)
            else:
                out = torch.full(
                    (row_count, 1 + decoder.num_codebooks),
                    -9,
                    dtype=torch.int64,
                    device=device,
                )
                frames = decoder.decode_step(*arguments, attention, out=out)
                assert frames is out
            assert int(sampler.step) == 8
            results.append(
                (
                    frames.clone(),
                    attention.key_caches.clone(),
                    attention.value_caches.clone(),
                )
            )
    for frames, keys, values in results[1:]:
        assert torch.equal(frames, results[0][0])
        assert torch.equal(bits(keys), bits(results[0][1]))
        assert torch.equal(bits(values), bits(results[0][2]))
    assert torch.all(results[0][0][:, 2:] < decoder.codebook_size)
    assert not math.isnan(float(results[0][0].float().sum()))
