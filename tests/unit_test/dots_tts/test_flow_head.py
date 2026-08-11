# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.dots_tts.flow_head import DotsTTSFlowHead

LLM_HIDDEN = 48
FM_HIDDEN = 32
LATENT_DIM = 6
PATCH_SIZE = 2
NFE = 2


def _flow_head(tmp_path) -> DotsTTSFlowHead:
    torch.save(
        {"mean": torch.zeros(LATENT_DIM), "var": torch.ones(LATENT_DIM)},
        tmp_path / "latent_stats.pt",
    )
    config = {
        "latent_dim": LATENT_DIM,
        "patch_size": PATCH_SIZE,
        "PatchEncoder": {
            "num_layers": 1,
            "num_heads": 2,
            "hidden_size": FM_HIDDEN,
            "ffn_hidden_size": 64,
            "causal": True,
        },
        "DiT": {
            "num_layers": 2,
            "num_heads": 2,
            "hidden_size": FM_HIDDEN,
            "ffn_hidden_size": 64,
            "modulation": True,
            "qk_norm": True,
            "rotary_bias": True,
        },
        "vocoder": {"sample_rate": 48000},
        "meanflow": {"enabled": True, "use_duration_embedding": True},
    }
    flow = DotsTTSFlowHead(
        config,
        llm_hidden_size=LLM_HIDDEN,
        latent_stats_path=str(tmp_path / "latent_stats.pt"),
        optimize=False,
    )
    with torch.no_grad():
        for parameter in flow.parameters():
            parameter.normal_(0.0, 0.2)
    return flow.eval()


def test_single_stream_decode_batch_accepts_2d_hidden(tmp_path) -> None:
    torch.manual_seed(1234)
    flow = _flow_head(tmp_path)
    state, prompt_embeddings = flow.new_request(
        max_audio_patch_count=8,
        prompt_latents=None,
        speaker_embedding=torch.randn(1, 512),
        speaker_scale=1.5,
        rng=None,
    )
    assert prompt_embeddings is None
    flow.append_hidden(state, torch.randn(1, 1, LLM_HIDDEN))

    def _decode(*, append_hidden: bool):
        # The model runner passes rank-2 [batch, hidden] rows for decode.
        return flow.decode_batch(
            [state],
            hidden_states=torch.randn(1, LLM_HIDDEN),
            num_steps=[NFE],
            ode_methods=["euler"],
            guidance_scales=[1.2],
            eos_thresholds=[2.0],
            append_hidden=append_hidden,
        )

    [first] = _decode(append_hidden=False)
    [second] = _decode(append_hidden=True)

    for step in (first, second):
        assert step.latent_patch.shape == (1, PATCH_SIZE, LATENT_DIM)
        assert step.feedback_embedding.shape[-1] == LLM_HIDDEN
        assert step.emit
        assert not step.finished
    assert state.decoded_patches == 2


def test_append_hidden_uses_bias_for_null_projection(tmp_path) -> None:
    flow = _flow_head(tmp_path)
    state, _ = flow.new_request(
        max_audio_patch_count=2,
        prompt_latents=None,
        speaker_embedding=None,
        speaker_scale=1.0,
        rng=None,
    )
    projection_calls = []
    hook = flow.hidden_proj.register_forward_hook(
        lambda _module, inputs, _output: projection_calls.append(inputs[0])
    )
    try:
        flow.append_hidden(state, torch.randn(1, 1, LLM_HIDDEN))
    finally:
        hook.remove()

    assert len(projection_calls) == 1
    torch.testing.assert_close(
        state.fm_cfg_sequence[0, 0], flow.hidden_proj.bias, rtol=0, atol=0
    )


def test_single_stream_seed_survives_rematerialization(tmp_path) -> None:
    torch.manual_seed(1618)
    flow = _flow_head(tmp_path)
    prefill_hidden = torch.randn(1, 1, LLM_HIDDEN)
    next_hidden = torch.randn(1, LLM_HIDDEN)
    schedule = torch.tensor([[0, 1]])

    uninterrupted, _ = flow.new_request(
        max_audio_patch_count=6,
        prompt_latents=None,
        speaker_embedding=None,
        speaker_scale=1.0,
        rng=41,
    )
    retracted, _ = flow.new_request(
        max_audio_patch_count=6,
        prompt_latents=None,
        speaker_embedding=None,
        speaker_scale=1.0,
        rng=41,
    )
    for state in (uninterrupted, retracted):
        flow.initialize_history(
            state,
            hidden_states=prefill_hidden,
            prompt_span_positions=torch.empty(0, dtype=torch.long),
            audio_span_token_ids={1},
            generation_schedule=schedule,
            prefill_end=1,
            decoded_latent_patches=[],
        )

    [first_expected] = flow.decode_batch(
        [uninterrupted],
        hidden_states=prefill_hidden[:, -1],
        num_steps=[NFE],
        ode_methods=["euler"],
        guidance_scales=[1.0],
        eos_thresholds=[2.0],
        append_hidden=False,
    )
    [first_actual] = flow.decode_batch(
        [retracted],
        hidden_states=prefill_hidden[:, -1],
        num_steps=[NFE],
        ode_methods=["euler"],
        guidance_scales=[1.0],
        eos_thresholds=[2.0],
        append_hidden=False,
    )
    torch.testing.assert_close(first_actual.latent_patch, first_expected.latent_patch)

    rng_state = flow.suspend_request(retracted)
    assert rng_state is not None
    rematerialized, _ = flow.new_request(
        max_audio_patch_count=6,
        prompt_latents=None,
        speaker_embedding=None,
        speaker_scale=1.0,
        rng=rng_state,
    )
    flow.replay_feedback(rematerialized, [first_actual.latent_patch])
    flow.initialize_history(
        rematerialized,
        hidden_states=torch.cat([prefill_hidden, next_hidden.unsqueeze(1)], dim=1),
        prompt_span_positions=torch.empty(0, dtype=torch.long),
        audio_span_token_ids={1},
        generation_schedule=schedule,
        prefill_end=1,
        decoded_latent_patches=[first_actual.latent_patch],
    )
    [expected] = flow.decode_batch(
        [uninterrupted],
        hidden_states=next_hidden,
        num_steps=[NFE],
        ode_methods=["euler"],
        guidance_scales=[1.0],
        eos_thresholds=[2.0],
        append_hidden=True,
    )
    [actual] = flow.decode_batch(
        [rematerialized],
        hidden_states=next_hidden,
        num_steps=[NFE],
        ode_methods=["euler"],
        guidance_scales=[1.0],
        eos_thresholds=[2.0],
        append_hidden=False,
    )
    torch.testing.assert_close(actual.latent_patch, expected.latent_patch)
    torch.testing.assert_close(actual.feedback_embedding, expected.feedback_embedding)


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16],
    ids=["float32", "bfloat16"],
)
def test_flow_rematerialization_matches_uninterrupted_next_step(
    tmp_path, dtype: torch.dtype
) -> None:
    torch.manual_seed(1618)
    flow = _flow_head(tmp_path).to(dtype=dtype)
    flow.init_batched_tail(num_slots=2, nfe=NFE, max_audio_patches=8)
    prompt_latents = torch.randn(1, 2 * PATCH_SIZE, LATENT_DIM, dtype=dtype)
    prefill_hidden = torch.randn(1, 3, LLM_HIDDEN, dtype=dtype)
    prompt_positions = torch.tensor([1, 2])
    schedule = torch.tensor([[0, 1, 1, 1]])
    tolerance = 3e-4 if dtype == torch.float32 else 2e-2

    uninterrupted, _ = flow.new_request(
        max_audio_patch_count=6,
        prompt_latents=prompt_latents,
        speaker_embedding=None,
        speaker_scale=1.0,
        rng=41,
    )
    retracted, _ = flow.new_request(
        max_audio_patch_count=6,
        prompt_latents=prompt_latents,
        speaker_embedding=None,
        speaker_scale=1.0,
        rng=41,
    )
    for state in (uninterrupted, retracted):
        flow.initialize_history(
            state,
            hidden_states=prefill_hidden,
            prompt_span_positions=prompt_positions,
            audio_span_token_ids={1},
            generation_schedule=schedule,
            prefill_end=3,
            decoded_latent_patches=[],
        )

    first_steps = flow.decode_batch(
        [uninterrupted, retracted],
        hidden_states=prefill_hidden[:, -1].expand(2, -1),
        num_steps=[NFE, NFE],
        ode_methods=["euler", "euler"],
        guidance_scales=[1.0, 1.0],
        eos_thresholds=[2.0, 2.0],
        append_hidden=False,
    )
    assert flow.resolve_batched_eos() == [False, False]
    torch.testing.assert_close(
        first_steps[0].latent_patch,
        first_steps[1].latent_patch,
        rtol=tolerance,
        atol=tolerance,
    )

    rng_state = flow.suspend_request(retracted)
    rematerialized, _ = flow.new_request(
        max_audio_patch_count=6,
        prompt_latents=prompt_latents,
        speaker_embedding=None,
        speaker_scale=1.0,
        rng=rng_state,
    )
    replayed_feedback = flow.replay_feedback(
        rematerialized,
        [first_steps[1].latent_patch],
    )
    torch.testing.assert_close(
        replayed_feedback[0],
        first_steps[1].feedback_embedding,
        rtol=tolerance,
        atol=tolerance,
    )

    next_hidden = torch.randn(1, LLM_HIDDEN, dtype=dtype)
    flow.initialize_history(
        rematerialized,
        hidden_states=torch.cat([prefill_hidden, next_hidden.unsqueeze(1)], dim=1),
        prompt_span_positions=prompt_positions,
        audio_span_token_ids={1},
        generation_schedule=schedule,
        prefill_end=3,
        decoded_latent_patches=[first_steps[1].latent_patch],
    )
    [expected] = flow.decode_batch(
        [uninterrupted],
        hidden_states=next_hidden,
        num_steps=[NFE],
        ode_methods=["euler"],
        guidance_scales=[1.0],
        eos_thresholds=[2.0],
        append_hidden=True,
    )
    assert flow.resolve_batched_eos() == [False]
    [actual] = flow.decode_batch(
        [rematerialized],
        hidden_states=next_hidden,
        num_steps=[NFE],
        ode_methods=["euler"],
        guidance_scales=[1.0],
        eos_thresholds=[2.0],
        append_hidden=False,
    )
    assert flow.resolve_batched_eos() == [False]

    torch.testing.assert_close(
        actual.latent_patch,
        expected.latent_patch,
        rtol=tolerance,
        atol=tolerance,
    )
    torch.testing.assert_close(
        actual.feedback_embedding,
        expected.feedback_embedding,
        rtol=tolerance,
        atol=tolerance,
    )


def test_validate_request_batched_gates_prompt_and_span_budget() -> None:
    flow = SimpleNamespace(
        is_batched=True,
        _batched_nfe=4,
        _tail=SimpleNamespace(spec=SimpleNamespace(patch_capacity=9)),
    )
    validate = DotsTTSFlowHead.validate_request

    validate(
        flow, num_steps=4, ode_method="euler", prompt_patch_count=3, total_span_count=9
    )

    with pytest.raises(ValueError, match="reference audio"):
        validate(
            flow,
            num_steps=4,
            ode_method="euler",
            prompt_patch_count=0,
            total_span_count=9,
        )
    with pytest.raises(ValueError, match="audio spans"):
        validate(
            flow,
            num_steps=4,
            ode_method="euler",
            prompt_patch_count=3,
            total_span_count=10,
        )

    single = SimpleNamespace(is_batched=False)
    validate(
        single,
        num_steps=8,
        ode_method="rk4",
        prompt_patch_count=0,
        total_span_count=10**6,
    )


def test_flow_matching_checkpoint_runs_the_single_request_solver(tmp_path) -> None:
    torch.save(
        {"mean": torch.zeros(LATENT_DIM), "var": torch.ones(LATENT_DIM)},
        tmp_path / "latent_stats.pt",
    )
    config = {
        "latent_dim": LATENT_DIM,
        "patch_size": PATCH_SIZE,
        "PatchEncoder": {
            "num_layers": 1,
            "num_heads": 2,
            "hidden_size": FM_HIDDEN,
            "ffn_hidden_size": 64,
            "causal": True,
        },
        "DiT": {
            "num_layers": 2,
            "num_heads": 2,
            "hidden_size": FM_HIDDEN,
            "ffn_hidden_size": 64,
            "modulation": True,
            "qk_norm": True,
            "rotary_bias": True,
        },
        "vocoder": {"sample_rate": 48000},
        # note (luojiaxuan): SOAR and base ship without a meanflow block.
        "meanflow": None,
    }
    flow = DotsTTSFlowHead(
        config,
        llm_hidden_size=LLM_HIDDEN,
        latent_stats_path=str(tmp_path / "latent_stats.pt"),
        optimize=False,
    )
    with torch.no_grad():
        for parameter in flow.parameters():
            parameter.normal_(0.0, 0.2)
    flow = flow.eval()

    assert flow.mode == "flow_matching"
    assert not flow.is_batched
    # note (luojiaxuan): single-request serving keeps per-request num_steps and CFG.
    flow.validate_request(num_steps=10, ode_method="euler")

    with pytest.raises(ValueError, match="max_running_requests=1"):
        flow.init_batched_tail(num_slots=16, nfe=10, max_audio_patches=8)

    state, _ = flow.new_request(
        max_audio_patch_count=8,
        prompt_latents=None,
        speaker_embedding=torch.randn(1, 512),
        speaker_scale=1.5,
        rng=None,
    )
    flow.append_hidden(state, torch.randn(1, 1, LLM_HIDDEN))
    step = flow.decode_next(
        state,
        hidden_states=torch.randn(1, 1, LLM_HIDDEN),
        num_steps=2,
        ode_method="euler",
        guidance_scale=1.2,
        eos_threshold=0.8,
    )
    assert step.latent_patch.shape == (1, PATCH_SIZE, LATENT_DIM)
    assert torch.isfinite(step.latent_patch).all()
    assert torch.isfinite(step.feedback_embedding).all()


def test_batched_eos_resolve_reads_staged_flags(tmp_path) -> None:
    torch.manual_seed(7)
    flow = _flow_head(tmp_path)
    flow.init_batched_tail(num_slots=2, nfe=NFE, max_audio_patches=8)
    prompt_latents = torch.randn(1, 2 * PATCH_SIZE, LATENT_DIM)
    states = []
    for seed in (11, 12):
        state, _ = flow.new_request(
            max_audio_patch_count=6,
            prompt_latents=prompt_latents,
            speaker_embedding=None,
            speaker_scale=1.0,
            rng=seed,
        )
        flow.initialize_history(
            state,
            hidden_states=torch.randn(1, 3, LLM_HIDDEN),
            prompt_span_positions=torch.tensor([1, 2]),
            audio_span_token_ids={1},
            generation_schedule=torch.tensor([[0, 1, 1, 1]]),
            prefill_end=3,
            decoded_latent_patches=[],
        )
        states.append(state)

    denormalize_shapes = []
    denormalize = flow.io.denormalize

    def record_denormalize(value):
        denormalize_shapes.append(tuple(value.shape))
        return denormalize(value)

    flow.io.denormalize = record_denormalize
    steps = flow.decode_batch(
        states,
        hidden_states=torch.randn(2, LLM_HIDDEN),
        num_steps=[NFE, NFE],
        ode_methods=["euler", "euler"],
        guidance_scales=[1.0, 1.0],
        eos_thresholds=[2.0, 2.0],
        append_hidden=False,
    )
    assert denormalize_shapes == [(2, PATCH_SIZE, LATENT_DIM)]
    assert all(not step.finished for step in steps)
    assert flow.has_pending_batched_eos
    assert flow.resolve_batched_eos() == [False, False]
    assert not flow.has_pending_batched_eos
    assert flow.resolve_batched_eos() == []


def test_batched_eos_staging_requires_resolve_before_reuse(tmp_path) -> None:
    torch.manual_seed(8)
    flow = _flow_head(tmp_path)
    flow.init_batched_tail(num_slots=1, nfe=NFE, max_audio_patches=8)
    state, _ = flow.new_request(
        max_audio_patch_count=6,
        prompt_latents=torch.randn(1, 2 * PATCH_SIZE, LATENT_DIM),
        speaker_embedding=None,
        speaker_scale=1.0,
        rng=3,
    )
    flow.initialize_history(
        state,
        hidden_states=torch.randn(1, 3, LLM_HIDDEN),
        prompt_span_positions=torch.tensor([1, 2]),
        audio_span_token_ids={1},
        generation_schedule=torch.tensor([[0, 1, 1, 1]]),
        prefill_end=3,
        decoded_latent_patches=[],
    )
    flow.decode_batch(
        [state],
        hidden_states=torch.randn(1, LLM_HIDDEN),
        num_steps=[NFE],
        ode_methods=["euler"],
        guidance_scales=[1.0],
        eos_thresholds=[2.0],
        append_hidden=False,
    )
    slot = state.slot
    assert slot is not None
    tail = flow._tail
    fm_seq_len = tail.fm_seq_len(slot)
    encoder_seq_len = tail._encoder_seq_len[slot]
    rng_state = tail.slot_rng_state(slot)
    assert rng_state is not None
    decoded_patches = state.decoded_patches

    with pytest.raises(RuntimeError, match="before resolve_batched_eos"):
        flow.decode_batch(
            [state],
            hidden_states=torch.randn(1, LLM_HIDDEN),
            num_steps=[NFE],
            ode_methods=["euler"],
            guidance_scales=[1.0],
            eos_thresholds=[2.0],
            append_hidden=True,
        )
    assert tail.fm_seq_len(slot) == fm_seq_len
    assert tail._encoder_seq_len[slot] == encoder_seq_len
    actual_rng_state = tail.slot_rng_state(slot)
    assert actual_rng_state is not None
    torch.testing.assert_close(actual_rng_state, rng_state, rtol=0, atol=0)
    assert state.decoded_patches == decoded_patches
    assert flow.resolve_batched_eos() == [False]


def test_batched_eos_suppresses_first_check_until_resolve(tmp_path) -> None:
    torch.manual_seed(9)
    flow = _flow_head(tmp_path)
    flow.init_batched_tail(num_slots=1, nfe=NFE, max_audio_patches=8)
    state, _ = flow.new_request(
        max_audio_patch_count=6,
        prompt_latents=torch.randn(1, 2 * PATCH_SIZE, LATENT_DIM),
        speaker_embedding=None,
        speaker_scale=1.0,
        rng=5,
    )
    flow.initialize_history(
        state,
        hidden_states=torch.randn(1, 3, LLM_HIDDEN),
        prompt_span_positions=torch.tensor([1, 2]),
        audio_span_token_ids={1},
        generation_schedule=torch.tensor([[0, 1, 1, 1]]),
        prefill_end=3,
        decoded_latent_patches=[],
    )

    state.suppress_first_eos_check = True
    state.decoded_patches = 0
    with torch.no_grad():
        final = flow.eos_proj[-1]
        final.weight.zero_()
        final.bias[:] = torch.tensor([-10.0, 10.0])
    flow.decode_batch(
        [state],
        hidden_states=torch.randn(1, LLM_HIDDEN),
        num_steps=[NFE],
        ode_methods=["euler"],
        guidance_scales=[1.0],
        eos_thresholds=[0.1],
        append_hidden=False,
    )
    assert flow.resolve_batched_eos() == [False]
    flow.decode_batch(
        [state],
        hidden_states=torch.randn(1, LLM_HIDDEN),
        num_steps=[NFE],
        ode_methods=["euler"],
        guidance_scales=[1.0],
        eos_thresholds=[0.1],
        append_hidden=True,
    )
    assert flow.resolve_batched_eos() == [True]
