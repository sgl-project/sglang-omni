# SPDX-License-Identifier: Apache-2.0
"""Batched stage hand-offs and checkpoint sampling recipes."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch
from transformers import Qwen2_5OmniTextConfig
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerTextModel,
)

from sglang_omni.models.auk import constants as C
from sglang_omni.models.auk.hf_config import AuKRuntimeConfig
from sglang_omni.models.auk.payload_types import AuKState
from sglang_omni.models.auk.reference_encode import AuKConditionEncoder, build_messages
from sglang_omni.models.auk.stages import (
    condition_batch,
    create_auk_engine_executor,
    decode_batch,
    sample_batch,
    warmup_flow,
)
from sglang_omni.models.auk.vae import BigVGANFlowVAE
from sglang_omni.pipeline.control_plane import deserialize_message, serialize_message
from sglang_omni.proto import CompleteMessage, OmniRequest, StagePayload


@pytest.mark.parametrize("token_lengths", [[0], [-1, 32]])
def test_conditioning_graph_refuses_nonpositive_lengths(
    token_lengths: list[int],
) -> None:
    encoder = AuKConditionEncoder.__new__(AuKConditionEncoder)
    with pytest.raises(ValueError, match="must be positive"):
        encoder.capture_text_graphs(token_lengths, compute_dtype=torch.bfloat16)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph requires CUDA")
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_conditioning_graph_preserves_changed_requests_and_fallbacks(
    weight_dtype: torch.dtype,
) -> None:
    torch.manual_seed(21)
    configuration = Qwen2_5OmniTextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1000000.0,
            "mrope_section": [4, 6, 6],
        },
    )
    encoder = AuKConditionEncoder.__new__(AuKConditionEncoder)
    encoder.device = torch.device("cuda", torch.cuda.current_device())
    encoder.model = (
        Qwen2_5OmniThinkerTextModel(configuration)
        .to(device=encoder.device, dtype=weight_dtype)
        .eval()
        .requires_grad_(False)
    )
    encoder.processor = Mock()
    encoder.processor.apply_chat_template.return_value = ["unused"]
    encoder.text_graphs = {}
    encoder.capture_text_graphs([31, 32], compute_dtype=torch.bfloat16)
    previous_hidden = previous_snapshot = None
    for batch_size, token_length in [(1, 31), (1, 32), (1, 31), (1, 33), (2, 32)]:
        token_ids = torch.randint(
            1,
            configuration.vocab_size,
            (batch_size, token_length),
            device=encoder.device,
        )
        attention_mask = torch.ones_like(token_ids)
        if batch_size > 1:
            attention_mask[-1, -3:] = 0
        else:
            pass
        encoder.processor.return_value = {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
        }
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = torch.stack(
                encoder.model(
                    input_ids=token_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                ).hidden_states,
                dim=1,
            )
            actual = encoder.encode_batch(
                [build_messages("Read this text.", False)] * batch_size,
                [None] * batch_size,
            )
        for index, (hidden, mask) in enumerate(actual):
            valid_tokens = attention_mask[index].bool()
            torch.testing.assert_close(
                hidden, expected[index, :, valid_tokens], rtol=0, atol=0
            )
            assert mask.all()
        if previous_hidden is not None:
            torch.testing.assert_close(
                previous_hidden, previous_snapshot, rtol=0, atol=0
            )
        else:
            pass
        previous_hidden = actual[0][0]
        previous_snapshot = previous_hidden.clone()
    with torch.autocast("cuda", enabled=False):
        expected = torch.stack(
            encoder.model(
                input_ids=token_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            ).hidden_states,
            dim=1,
        )
        actual = encoder.encode_batch(
            [build_messages("Read this text.", False)] * batch_size,
            [None] * batch_size,
        )
    torch.testing.assert_close(actual[0][0], expected[0], rtol=0, atol=0)


def test_batched_generation_preserves_request_boundaries_and_serializes_audio():
    device = torch.device("cpu")

    class PosteriorVAE(torch.nn.Module):
        encoding_and_normalization = BigVGANFlowVAE.encoding_and_normalization

        def __init__(self):
            super().__init__()
            self.hop_size = 480
            self.audio_encoder = torch.nn.Conv1d(1, 128, 1, stride=self.hop_size)
            self.register_buffer("global_mean", torch.zeros(64))
            self.register_buffer("global_log_std", torch.ones(64))
            self.denormalize = lambda latent: latent
            self.inference_from_latents = Mock(
                side_effect=lambda latent: torch.full(
                    (latent.shape[0], 1, latent.shape[-1] * 480), 0.25
                )
            )

    vae = PosteriorVAE()
    encoder = Mock()
    encoder.encode_batch.return_value = [
        (torch.zeros(3, 6, 16), torch.ones(6, dtype=torch.bool)) for _ in range(3)
    ]
    fusion = (torch.zeros(2), torch.ones(1))
    flow = Mock()
    flow.sample_batch.side_effect = lambda items, **kwargs: [
        torch.zeros(item.target_frames, 64) for item in items
    ]
    payloads = [
        StagePayload(
            request_id=str(index),
            request=OmniRequest(inputs="hello"),
            data=AuKState(
                instruction="Say hello",
                gen_frames=frames,
                seed=11 if index < 2 else 12,
                ref_audio=np.zeros(24001, dtype=np.float32),
            ).to_dict(),
        )
        for index, frames in enumerate((151, 75, 151))
    ]

    rng = torch.random.get_rng_state()
    conditioned = condition_batch(payloads, encoder, vae, fusion, device, torch.float32)
    states = [AuKState.from_dict(payload.data) for payload in conditioned]
    assert torch.equal(states[0].ref_latent, states[1].ref_latent)
    assert not torch.equal(states[0].ref_latent, states[2].ref_latent)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert states[0].ref_length == 50
    assert states[0].ref_latent.stride() == (1, 51)
    sampled = sample_batch(conditioned, flow, device, torch.float32, 1500, {})
    assert len(flow.sample_batch.call_args.args[0]) == 3
    results = decode_batch(sampled, vae, device)

    assert [
        call.args[0].shape[0] for call in vae.inference_from_latents.call_args_list
    ] == [2, 1]
    for index, (frames, result) in enumerate(zip((151, 75, 151), results)):
        restored = deserialize_message(
            serialize_message(
                CompleteMessage(
                    request_id=result.request_id,
                    from_stage="decode",
                    success=True,
                    result=result.data,
                )
            )
        )
        assert restored.request_id == str(index)
        assert restored.result["audio_waveform_shape"] == [frames * 480]
        waveform = np.frombuffer(restored.result["audio_waveform"], dtype=np.float32)
        np.testing.assert_array_equal(waveform, np.full(frames * 480, 0.25))
        assert restored.result["usage"]["completion_tokens"] == frames


@pytest.mark.parametrize("flash", [False, True])
def test_engine_uses_checkpoint_sampling_recipe(monkeypatch, flash):
    from sglang_omni.models.auk import stages

    monkeypatch.setattr(stages, "resolve_checkpoint", lambda path: path)
    config = AuKRuntimeConfig(model_path="stub", name="AuK-Flash" if flash else "AuK")
    monkeypatch.setattr(stages, "make_runtime_config", lambda path: config)
    flow = Mock()
    flow.sample_batch.return_value = [torch.zeros(10, 64)]
    monkeypatch.setattr(stages, "load_flow", lambda *args: flow)
    scheduler = create_auk_engine_executor("stub", device="cpu", nfe=8, cfg_strength=3)
    state = AuKState(
        gen_frames=10,
        conditioning=torch.zeros(6, 16),
        text_mask=torch.ones(6, dtype=torch.bool),
    )
    scheduler.fn(
        StagePayload(
            request_id="test", request=OmniRequest(inputs="hello"), data=state.to_dict()
        )
    )
    recipe = flow.sample_batch.call_args.kwargs
    assert recipe == dict(
        steps=4 if flash else 8,
        cfg_strength=0 if flash else 3,
        sway_sampling_coef=None if flash else -1,
        t_grid=C.FLASH_T_GRID if flash else None,
    )


@pytest.fixture
def stages(monkeypatch):
    from sglang_omni.models.auk import stages

    monkeypatch.setattr(stages, "resolve_checkpoint", lambda path: path)
    monkeypatch.setattr(
        stages,
        "make_runtime_config",
        lambda path: AuKRuntimeConfig(model_path="stub", name="AuK"),
    )
    return stages


@pytest.mark.parametrize("field", ["dtype", "weight_dtype"])
def test_unknown_dtype_names_are_rejected_before_the_checkpoint_is_resolved(field):
    with pytest.raises(
        ValueError,
        match=rf"AuK {field} must be one of float32, float16, bfloat16, got 'bf16'",
    ):
        create_auk_engine_executor("stub", device="cpu", **{field: "bf16"})


def test_backbone_dtype_is_chosen_when_the_flow_is_loaded(stages, monkeypatch):
    """A later executor must not inherit an earlier one's cast backbone."""
    requested = []
    autocast = []

    def sample_batch(items, **kwargs):
        autocast.append(torch.is_autocast_enabled("cpu"))
        return [torch.zeros(item.target_frames, 64) for item in items]

    def load_flow(checkpoint, device, backbone_dtype, compile_blocks):
        requested.append(backbone_dtype)
        flow = Mock()
        flow.sample_batch.side_effect = sample_batch
        return flow

    monkeypatch.setattr(stages, "load_flow", load_flow)
    for weight_dtype in ("bfloat16", "float32"):
        scheduler = create_auk_engine_executor(
            "stub", device="cpu", dtype="bfloat16", weight_dtype=weight_dtype
        )
        scheduler.fn(
            StagePayload(
                request_id="test",
                request=OmniRequest(inputs="hello"),
                data=AuKState(
                    gen_frames=10,
                    conditioning=torch.zeros(6, 16),
                    text_mask=torch.ones(6, dtype=torch.bool),
                ).to_dict(),
            )
        )
    assert requested == [torch.bfloat16, torch.float32]
    assert autocast == [False, True]


def engine_payload():
    return StagePayload(
        request_id="test",
        request=OmniRequest(inputs="hello"),
        data=AuKState(
            gen_frames=10,
            conditioning=torch.zeros(6, 16),
            text_mask=torch.ones(6, dtype=torch.bool),
        ).to_dict(),
    )


def stub_flow():
    flow = Mock()
    flow.transformer.attn_mask_enabled = True
    flow.sample_batch.return_value = [torch.zeros(10, 64)]
    return flow


def test_block_compilation_is_chosen_when_the_flow_is_loaded(stages, monkeypatch):
    """Compiling mutates the backbone, so a cached flow must not be reused."""
    requested = []
    warmups = []

    def load_flow(checkpoint, device, backbone_dtype, compile_blocks):
        requested.append(compile_blocks)
        return stub_flow()

    monkeypatch.setattr(stages, "load_flow", load_flow)
    monkeypatch.setattr(stages, "warmup_flow", lambda *args: warmups.append(args[0]))
    for compile_blocks in (True, False):
        create_auk_engine_executor(
            "stub", device="cpu", enable_dit_torch_compile=compile_blocks
        )
    assert requested == [True, False]
    assert len(warmups) == 1


def test_the_step_graph_is_skipped_where_the_platform_records_none(stages, monkeypatch):
    flow = stub_flow()
    monkeypatch.setattr(stages, "load_flow", lambda *args: flow)
    scheduler = create_auk_engine_executor(
        "stub", device="cpu", enable_dit_cuda_graph=True
    )
    scheduler.fn(engine_payload())
    assert "step_graph" not in flow.sample_batch.call_args.kwargs


def test_the_step_graph_is_refused_without_the_attention_bias(stages, monkeypatch):
    """Padding is only neutral because masked keys are biased to -inf."""
    flow = stub_flow()
    flow.transformer.attn_mask_enabled = False
    monkeypatch.setattr(stages, "load_flow", lambda *args: flow)
    with pytest.raises(ValueError, match="attn_mask_enabled"):
        create_auk_engine_executor("stub", device="cpu", enable_dit_cuda_graph=True)


def test_the_warmup_covers_a_request_that_carries_no_reference():
    """An instruction-only request guards on no bias, and would recompile both blocks."""
    flow = stub_flow()
    flow.transformer.txt_proj.in_features = 16
    flow.transformer.latent_dim = 64

    warmup_flow(flow, torch.device("cpu"), torch.float32, dict(steps=8))

    batches = [items for (items,), _ in flow.sample_batch.call_args_list]
    lone = [
        items[0] for items in batches if len(items) == 1 and not items[0].ref_length
    ]
    assert lone and lone[0].ref_latent is None
    # The other two guards dynamo separates: a reference, and a padded batch.
    assert {(len(items), bool(items[0].ref_length)) for items in batches} == {
        (1, False),
        (1, True),
        (2, True),
    }
