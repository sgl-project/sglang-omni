# SPDX-License-Identifier: Apache-2.0
"""Batched stage hand-offs and checkpoint sampling recipes."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from sglang_omni.models.auk import constants as C
from sglang_omni.models.auk.hf_config import AuKRuntimeConfig
from sglang_omni.models.auk.payload_types import AuKState
from sglang_omni.models.auk.stages import (
    _condition_batch,
    _decode_batch,
    _sample_batch,
    create_auk_engine_executor,
)
from sglang_omni.models.auk.vae import BigVGANFlowVAE
from sglang_omni.pipeline.control_plane import deserialize_message, serialize_message
from sglang_omni.proto import CompleteMessage, OmniRequest, StagePayload


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
    conditioned = _condition_batch(
        payloads, encoder, vae, fusion, device, torch.float32
    )
    states = [AuKState.from_dict(payload.data) for payload in conditioned]
    assert torch.equal(states[0].ref_latent, states[1].ref_latent)
    assert not torch.equal(states[0].ref_latent, states[2].ref_latent)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert states[0].ref_length == 50
    assert states[0].ref_latent.stride() == (1, 51)
    sampled = _sample_batch(conditioned, flow, device, torch.float32, 1500, {})
    assert len(flow.sample_batch.call_args.args[0]) == 3
    results = _decode_batch(sampled, vae, device)

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
    monkeypatch.setattr(stages, "_load_flow", lambda *args: flow)
    scheduler = create_auk_engine_executor("stub", device="cpu", nfe=8, cfg_strength=3)
    state = AuKState(
        gen_frames=10,
        conditioning=torch.zeros(6, 16),
        text_mask=torch.ones(6, dtype=torch.bool),
    )
    scheduler._fn(
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
        cache_reference_audio_embedding=False,
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

    def load_flow(checkpoint, device, backbone_dtype):
        requested.append(backbone_dtype)
        flow = Mock()
        flow.sample_batch.side_effect = sample_batch
        return flow

    monkeypatch.setattr(stages, "_load_flow", load_flow)
    for weight_dtype in ("bfloat16", "float32"):
        scheduler = create_auk_engine_executor(
            "stub", device="cpu", dtype="bfloat16", weight_dtype=weight_dtype
        )
        scheduler._fn(
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
