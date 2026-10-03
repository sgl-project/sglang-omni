# SPDX-License-Identifier: Apache-2.0
"""Talker RVQ code generation runs on a schedule precomputed at load time."""

import pytest
import torch

from sglang_omni.models.nemotron_voicechat.mog_head import MoGHead
from sglang_omni.models.nemotron_voicechat.talker import (
    EarTtsTalker,
    GraphCodeGenerator,
)
from sglang_omni.platforms import current_platform

NUM_QUANTIZERS = 31
CODEBOOK_SIZE = 16
HIDDEN_SIZE = 32
LATENT_SIZE = 8
TOP_P = 0.9
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def make_talker() -> EarTtsTalker:
    talker = EarTtsTalker(
        dict(
            hidden_size=HIDDEN_SIZE,
            vocab_size=10,
            char_vocab_size=8,
            num_quantizers=NUM_QUANTIZERS,
            codebook_size=CODEBOOK_SIZE,
            latent_size=LATENT_SIZE,
            char_encoder_config={
                "encoder": dict(
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                )
            },
        )
    )
    talker.rvq_embs.normal_()
    return talker


def make_mog_head() -> MoGHead:
    mog_head = MoGHead(
        dict(
            hidden_size=HIDDEN_SIZE,
            intermediate_size=64,
            num_layers=3,
            num_predictions=CODEBOOK_SIZE,
            out_size=LATENT_SIZE,
            low_rank=4,
        )
    )
    mog_head.low_mat.data.normal_()
    return mog_head


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("num_iter", [1, 4, 8, 40])
@pytest.mark.parametrize("exponent", [0.5, 1.0, 2.0, 3.0])
def test_level_schedule_assigns_every_quantizer_once(device, num_iter, exponent):
    level_schedule = make_talker().build_level_schedule(
        num_iter, exponent, torch.device(device)
    )

    assert len(level_schedule) <= num_iter
    next_level = 0
    for first_level, level_count in level_schedule:
        assert first_level == next_level
        assert level_count > 0
        next_level += level_count
    assert next_level == NUM_QUANTIZERS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_generate_codes_does_not_sync_with_host():
    device = torch.device("cuda")
    talker = make_talker().to(device)
    mog_head = make_mog_head().to(device)
    level_schedule = talker.build_level_schedule(8, 2.0, device)
    hidden_TD = torch.randn(1, HIDDEN_SIZE, device=device)

    torch.cuda.set_sync_debug_mode("error")
    try:
        codes_TQ = talker.generate_codes(
            hidden_TD, mog_head, level_schedule=level_schedule, top_p=TOP_P
        )
    finally:
        torch.cuda.set_sync_debug_mode("default")

    assert codes_TQ.shape == (1, NUM_QUANTIZERS)
    assert int(codes_TQ.min()) >= 0
    assert int(codes_TQ.max()) < CODEBOOK_SIZE


def make_graph_code_generator(talker, mog_head, level_schedule):
    device = torch.device("cuda")
    return GraphCodeGenerator(
        talker,
        mog_head,
        backend=current_platform.get_device_graph_backend(device),
        level_schedule=level_schedule,
        top_p=TOP_P,
        noise_scale=1.0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_graph_codes_match_eager_step_for_step():
    device = torch.device("cuda")
    talker = make_talker().to(device)
    mog_head = make_mog_head().to(device)
    level_schedule = talker.build_level_schedule(8, 2.0, device)
    graph_code_generator = make_graph_code_generator(talker, mog_head, level_schedule)
    hidden_steps = [torch.randn(1, HIDDEN_SIZE, device=device) for _ in range(5)]

    torch.manual_seed(7)
    with torch.inference_mode():
        eager_steps = [
            talker.generate_codes(
                hidden_TD, mog_head, level_schedule=level_schedule, top_p=TOP_P
            )
            for hidden_TD in hidden_steps
        ]
    torch.manual_seed(7)
    graph_steps = [graph_code_generator(hidden_TD) for hidden_TD in hidden_steps]

    for eager_codes, graph_codes in zip(eager_steps, graph_steps, strict=True):
        assert torch.equal(eager_codes, graph_codes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_graph_replays_draw_fresh_noise():
    device = torch.device("cuda")
    talker = make_talker().to(device)
    mog_head = make_mog_head().to(device)
    graph_code_generator = make_graph_code_generator(
        talker, mog_head, talker.build_level_schedule(8, 2.0, device)
    )
    hidden_TD = torch.randn(1, HIDDEN_SIZE, device=device)

    replays = [graph_code_generator(hidden_TD) for _ in range(8)]

    assert any(not torch.equal(replays[0], codes) for codes in replays[1:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_graph_replay_does_not_sync_with_host():
    device = torch.device("cuda")
    talker = make_talker().to(device)
    mog_head = make_mog_head().to(device)
    graph_code_generator = make_graph_code_generator(
        talker, mog_head, talker.build_level_schedule(8, 2.0, device)
    )
    hidden_TD = torch.randn(1, HIDDEN_SIZE, device=device)

    torch.cuda.set_sync_debug_mode("error")
    try:
        codes_TQ = graph_code_generator(hidden_TD)
    finally:
        torch.cuda.set_sync_debug_mode("default")

    assert codes_TQ.shape == (1, NUM_QUANTIZERS)
    assert not codes_TQ.is_inference()
