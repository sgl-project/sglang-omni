# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses

import torch
from torch import nn

from sglang_omni.models.easymagpie_tts.decode_buffers import EasyMagpieDecodeBuffers
from sglang_omni.models.easymagpie_tts.sglang_model import patch_silu_shared_experts


def decode_buffers(
    talker, batch: int, *, top_k: int = 5, seed: int = 3, audio_valid=None
) -> EasyMagpieDecodeBuffers:
    buffers = EasyMagpieDecodeBuffers.allocate(
        talker.tts_config,
        max_batch=batch,
        max_top_k=top_k,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    buffers.stage(
        conditioning=torch.zeros(batch, 8),
        audio_valid=(
            torch.ones(batch, dtype=torch.bool) if audio_valid is None else audio_valid
        ),
        temperatures=torch.full((batch,), 0.8),
        top_ks=torch.full((batch,), top_k),
        seeds=torch.full((batch,), seed),
        positions=torch.arange(batch) * 4,
    )
    return buffers


def sample(talker, hidden: torch.Tensor, **kwargs) -> torch.Tensor:
    buffers = decode_buffers(talker, hidden.shape[0], **kwargs)
    return talker.heads.sample_codes(
        hidden,
        temperatures=buffers.temperatures,
        top_ks=buffers.top_ks,
        seeds=buffers.seeds,
        positions=buffers.positions,
        max_top_k=buffers.max_top_k,
    )


def test_sampling_emits_one_code_per_stacked_codebook(talker) -> None:
    hidden = torch.randn(3, 8)
    codes = sample(talker, hidden)
    assert codes.shape == (3, 4)
    allowed = set(range(16)) | {talker.tts_config.audio_eos_id}
    assert set(codes.flatten().tolist()) <= allowed


def test_sampling_is_reproducible_from_seed_and_position(talker) -> None:
    hidden = torch.randn(2, 8)
    torch.testing.assert_close(
        sample(talker, hidden, top_k=18), sample(talker, hidden, top_k=18)
    )


def test_top_k_one_is_greedy_over_allowed_codes(talker) -> None:
    head = talker.heads.local_transformer_out_projections[0]
    nn.init.zeros_(head.weight)
    with torch.no_grad():
        head.bias.copy_(torch.arange(24, dtype=torch.float32))
    codes = sample(talker, torch.randn(2, 8), top_k=1)
    # The highest logits sit on special ids; only EOS (17) is allowed.
    assert codes[:, 0].tolist() == [17, 17]


def test_phoneme_prediction_falls_back_to_unk_when_underconfident(talker) -> None:
    hidden = torch.zeros(1, 8)
    assert talker.heads.predict_phonemes(hidden).shape == (1, 1)
    talker.heads.config = dataclasses.replace(
        talker.tts_config, phoneme_confidence_unk_threshold=0.99
    )
    assert talker.heads.predict_phonemes(hidden).tolist() == [[19]]


def test_conditioning_masks_each_stream_independently(talker) -> None:
    heads = talker.heads
    heads.text_embedding.weight.data.fill_(1)
    for table in heads.phoneme_embeddings:
        table.weight.data.fill_(2)
    for table in heads.audio_embeddings:
        table.weight.data.fill_(3)
    combined = talker.compose_conditioning(
        text_tokens=torch.tensor([1, 2]),
        text_valid=torch.tensor([True, False]),
        phoneme_tokens=torch.tensor([[1], [2]]),
        phoneme_valid=torch.tensor([False, True]),
        previous_audio_codes=torch.zeros((2, 4), dtype=torch.long),
        audio_valid=torch.tensor([True, True]),
    )
    assert torch.all(combined[0] == 4)
    assert torch.all(combined[1] == 5)


def test_acoustic_eos_drives_stop_logits_only_on_audio_rows(talker) -> None:
    codes = torch.zeros((3, 4), dtype=torch.long)
    codes[0, 2] = talker.tts_config.audio_eos_id
    codes[2, 1] = talker.tts_config.audio_eos_id
    talker.heads.sample_codes = lambda *args, **kwargs: codes
    buffers = decode_buffers(talker, 3, audio_valid=torch.tensor([True, True, False]))
    hidden = torch.zeros(3, 8)

    eos = talker.decode_tts_heads(hidden, buffers)
    logits = talker.make_stop_logits(hidden, eos)

    assert buffers.codes.tolist() == codes.tolist()
    assert buffers.phonemes.shape == (3, 1)
    assert buffers.eos.tolist() == eos.tolist() == [True, False, False]
    assert logits[:, 1].tolist() == [30.0, -30.0, -30.0]
    assert logits[:, 0].tolist() == [0.0, 0.0, 0.0]


def test_shared_experts_are_patched_to_silu() -> None:
    layers = nn.ModuleList()
    for has_shared in (True, False):
        layer = nn.Module()
        layer.mixer = nn.Module()
        if has_shared:
            layer.mixer.shared_experts = nn.Module()
            layer.mixer.shared_experts.act_fn = nn.ReLU()
        else:
            pass
        layers.append(layer)
    backbone = nn.Module()
    backbone.model = nn.Module()
    backbone.model.layers = layers

    assert patch_silu_shared_experts(backbone) == 1
    values = torch.tensor([-1.0, 1.0])
    torch.testing.assert_close(
        layers[0].mixer.shared_experts.act_fn(values), nn.functional.silu(values)
    )
