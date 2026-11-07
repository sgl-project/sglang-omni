# SPDX-License-Identifier: Apache-2.0
"""Ming 3 TTS acoustic-step capture on whichever accelerator the host provides."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from sglang_omni.models.ming3_tts.acoustic import FLOW_STEPS, AcousticInputs
from sglang_omni.models.ming3_tts.sglang_model import SpeechHeadModel
from sglang_omni.modules.ming_audio.flow.aggregator import Aggregator
from sglang_omni.modules.ming_audio.flow.cfm import get_epss_timesteps
from sglang_omni.modules.ming_audio.flow.flow_loss import FlowLoss
from sglang_omni.platforms import current_platform
from sglang_omni.utils.device import resolve_concrete_device

LATENT_DIM = 8
HIDDEN_SIZE = 32
MAX_BATCH_SIZE = 4


class HeadConfig:
    def __init__(self) -> None:
        self.tts_config: dict[str, int] = {"latent_dim": LATENT_DIM}
        self.hidden_size: int = HIDDEN_SIZE


class AcousticHeads(SpeechHeadModel, nn.Module):
    def __init__(self, device: torch.device) -> None:
        nn.Module.__init__(self)
        self.flowloss = FlowLoss(
            z_channels=LATENT_DIM,
            llm_cond_dim=HIDDEN_SIZE,
            hidden_size=HIDDEN_SIZE,
            depth=1,
            num_heads=2,
            attn_backend="torch",
        )
        self.linear_proj_audio = Aggregator(
            in_channels=LATENT_DIM,
            hidden_size=HIDDEN_SIZE,
            depth=1,
            num_heads=2,
            llm_input_dim=HIDDEN_SIZE,
            attn_backend="torch",
        )
        self.stop_head = nn.Linear(HIDDEN_SIZE, 2)
        self.acoustic_graphs = {}
        self.enable_fused_flow_head = False
        self.flow_head_path_logged = True
        self.flow_head_projections = None
        self.config = HeadConfig()
        self.to(device=device, dtype=torch.float32)
        self.flow_timesteps = get_epss_timesteps(FLOW_STEPS, device, torch.float32)


def build_acoustic_inputs(batch_size: int, device: torch.device) -> AcousticInputs:
    generator = torch.Generator(device=device).manual_seed(17)
    return AcousticInputs(
        hidden_states=torch.randn(
            (batch_size, HIDDEN_SIZE), device=device, generator=generator
        ),
        latent_history=torch.randn(
            (batch_size, 1, LATENT_DIM), device=device, generator=generator
        ),
        initial_noise=torch.randn(
            (batch_size, LATENT_DIM, 1), device=device, generator=generator
        ),
        solver_noise=torch.randn(
            (FLOW_STEPS - 1, batch_size, 1, LATENT_DIM),
            device=device,
            generator=generator,
        ),
        temperatures=torch.full((batch_size,), 1.5, device=device),
    )


def accelerator_heads() -> tuple[AcousticHeads, torch.device]:
    device = resolve_concrete_device(None)
    if current_platform.get_device_graph_backend(device) is None:
        pytest.skip(f"{device.type} records no model-owned graphs")
    else:
        pass
    torch.manual_seed(0)
    return AcousticHeads(device).eval(), device


@pytest.mark.accelerator
def test_capture_records_an_exact_bucket_per_batch_size() -> None:
    heads, _device = accelerator_heads()

    heads.capture_acoustic_graphs(MAX_BATCH_SIZE)

    assert sorted(heads.acoustic_graphs) == [1, 2, 3, 4]


@pytest.mark.accelerator
def test_a_replayed_batch_returns_its_own_rows() -> None:
    heads, device = accelerator_heads()
    heads.capture_acoustic_graphs(MAX_BATCH_SIZE)

    step = heads.sample_acoustic_step(build_acoustic_inputs(3, device))

    assert step.latents.shape == (3, 1, LATENT_DIM)
    assert step.feedback_embeddings.shape == (3, HIDDEN_SIZE)
    assert step.stop_probabilities.shape == (3,)


@pytest.mark.accelerator
def test_a_replayed_row_is_detached_from_the_graph_output_buffer() -> None:
    heads, device = accelerator_heads()
    heads.capture_acoustic_graphs(MAX_BATCH_SIZE)

    first = heads.sample_acoustic_step(build_acoustic_inputs(1, device))
    kept = first.latents.clone()
    heads.sample_acoustic_step(build_acoustic_inputs(2, device))

    assert torch.equal(first.latents, kept)


def test_a_device_with_no_graph_backend_stays_eager() -> None:
    device = torch.device("cpu")
    assert current_platform.get_device_graph_backend(device) is None
    torch.manual_seed(0)
    heads = AcousticHeads(device).eval()

    heads.capture_acoustic_graphs(MAX_BATCH_SIZE)
    step = heads.sample_acoustic_step(build_acoustic_inputs(2, device))

    assert heads.acoustic_graphs == {}
    assert step.latents.shape == (2, 1, LATENT_DIM)
