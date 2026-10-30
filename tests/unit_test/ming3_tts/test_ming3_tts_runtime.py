# SPDX-License-Identifier: Apache-2.0
"""Runtime contracts of the Ming 3 TTS acoustic step, vocoder graphs, and references."""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.model_loader.utils import set_default_torch_dtype
from torch import nn

from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.models.ming3_tts import stages
from sglang_omni.models.ming3_tts.acoustic import FLOW_STEPS, AcousticInputs
from sglang_omni.models.ming3_tts.kernels import cfg_solver_step
from sglang_omni.models.ming3_tts.model_config import build_audio_aggregator
from sglang_omni.models.ming3_tts.model_runner import (
    LaunchedAcousticStep,
    Ming3TTSModelRunner,
    Ming3TTSRequestData,
)
from sglang_omni.models.ming3_tts.sglang_model import (
    Ming3DenseTTSConfig,
    Ming3TTSConfig,
    SpeechHeadModel,
    TinySpeechHeadModel,
)
from sglang_omni.models.ming3_tts.stages import read_reference_audio
from sglang_omni.models.ming3_tts.vocoder import DecoderGraphs
from sglang_omni.modules.ming_audio.flow.cfm import get_epss_timesteps
from sglang_omni.modules.ming_audio.flow.flow_loss import FlowLoss
from sglang_omni.modules.ming_audio.vae.configuration_audio_vae import AudioVAEconfig
from sglang_omni.modules.ming_audio.vae.modeling_audio_vae import AudioVAE
from sglang_omni.scheduling.types import SchedulerRequest
from tests.unit_test.fixtures.accelerator import require_cuda

HIDDEN_SIZE = 64
LATENT_DIM = 8
FLOW_CONFIG = {"hidden_size": 128, "depth": 2, "num_heads": 2}


def randomize_linear_layers(module: nn.Module, seed: int) -> None:
    """Replace zero-initialized output layers so every path reaches the outputs."""
    generator = torch.Generator(device=next(module.parameters()).device)
    generator.manual_seed(seed)
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            layer.weight.data.normal_(0.0, layer.in_features**-0.5, generator=generator)
            layer.bias.data.normal_(0.0, 0.1, generator=generator)
        else:
            pass


class TinySpeechHeads(SpeechHeadModel, nn.Module):
    """The speech heads at test width, built like the engine's but outside SGLang."""

    def __init__(self, device: torch.device) -> None:
        nn.Module.__init__(self)
        self.config = Ming3DenseTTSConfig(
            llm_config={"hidden_size": HIDDEN_SIZE},
            audio_tokenizer_config={"enc_kwargs": {"latent_dim": LATENT_DIM}},
        )
        with set_default_torch_dtype(torch.bfloat16), device:
            self.flowloss = FlowLoss(
                z_channels=LATENT_DIM,
                llm_cond_dim=HIDDEN_SIZE,
                attn_backend="torch",
                **FLOW_CONFIG,
            )
            self.linear_proj_audio = build_audio_aggregator(
                HIDDEN_SIZE, LATENT_DIM, FLOW_CONFIG
            )
            self.stop_head = nn.Linear(HIDDEN_SIZE, 2)
        randomize_linear_layers(self, seed=0)
        self.eval()
        self.flow_timesteps = get_epss_timesteps(FLOW_STEPS, device, torch.float32)
        self.acoustic_graphs = {}
        self.flow_head_projections = None
        self.enable_fused_flow_head = True
        self.flow_head_path_logged = False
        self.weights_loaded = False


def make_inputs(batch_size: int, device: torch.device) -> AcousticInputs:
    generator = torch.Generator(device=device).manual_seed(batch_size)
    return AcousticInputs(
        hidden_states=torch.randn(
            (batch_size, HIDDEN_SIZE),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
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
        temperatures=torch.full((batch_size,), 0.7, device=device),
    )


class NativeTinySpeechHeads(TinySpeechHeadModel, TinySpeechHeads):
    """Exercise Tiny's native arithmetic with the same randomized speech heads."""


@pytest.mark.accelerator
@pytest.mark.parametrize("native_precision", [False, True])
def test_acoustic_graph_replays_the_declared_guidance_after_allocator_reuse(
    native_precision: bool,
) -> None:
    """A captured step once read its guidance scales from freed memory."""
    require_cuda()
    device = torch.device("cuda", torch.cuda.current_device())
    heads = (
        NativeTinySpeechHeads(device) if native_precision else TinySpeechHeads(device)
    )
    with torch.no_grad():
        heads.capture_acoustic_graphs(4)
        # Small allocations take any block the capture left free.
        overwrites = [torch.full((64,), -7.0, device=device) for _ in range(256)]
        for batch_size in (1, 2, 4, 5):
            inputs = make_inputs(batch_size, device)
            replayed = heads.sample_acoustic_step(inputs)
            eager = heads.compute_acoustic_step(
                inputs,
                heads.build_sampling_scales(batch_size),
                heads.build_flow_constants(batch_size),
            )
            assert torch.equal(replayed.latents, eager.latents)
            assert torch.equal(replayed.feedback_embeddings, eager.feedback_embeddings)
            assert torch.equal(replayed.stop_probabilities, eager.stop_probabilities)
    assert len(overwrites) == 256


def test_retraction_restores_the_resolved_launch_before_a_newer_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A newer lookahead launch once overwrote an older step's rollback state."""
    runner = Ming3TTSModelRunner.__new__(Ming3TTSModelRunner)
    generator = torch.Generator().manual_seed(42)
    original_noise_state = generator.get_state()
    original_latents = torch.zeros(1, LATENT_DIM)
    original_feedback = torch.zeros(HIDDEN_SIZE)
    request_data = Ming3TTSRequestData(
        speaker_embedding=torch.zeros(192),
        speaker_position=0,
        sampling_temperature=0.0,
        noise_generator=generator,
        latent_history=original_latents,
        next_feedback=original_feedback,
        req=SimpleNamespace(is_retracted=False),
    )
    requests = [SchedulerRequest(request_id="retracted", data=request_data)]
    result = SimpleNamespace(next_token_ids=torch.ones(1, dtype=torch.long))

    def launch_step(result, requests) -> LaunchedAcousticStep:
        torch.rand(1, generator=generator)
        request_data.launched_latents += 1
        request_data.latent_history = request_data.latent_history + 1
        request_data.next_feedback = request_data.next_feedback + 1
        return LaunchedAcousticStep(
            latents=request_data.latent_history[None],
            feedback_embeddings=request_data.next_feedback[None],
            is_middle_chunk=[False],
            host_token_ids=None,
        )

    monkeypatch.setattr(runner, "launch_acoustic_step", launch_step)
    monkeypatch.setattr(
        runner,
        "next_host_staging",
        lambda shape, dtype: torch.empty(shape, dtype=dtype),
    )
    first_launch = runner.post_decode_launch(result, None, requests)
    runner.post_decode_launch(result, None, requests)
    assert request_data.launched_latents == 2
    request_data.req.is_retracted = True
    runner.post_decode_resolve(first_launch, result, None, None, requests)
    assert request_data.latent_history is original_latents
    assert request_data.next_feedback is original_feedback
    assert request_data.launched_latents == 0
    assert torch.equal(generator.get_state(), original_noise_state)
    assert request_data.latents == []
    assert request_data.feedback_embeddings == []


@pytest.mark.accelerator
def test_fused_flow_head_tracks_the_module_forward() -> None:
    require_cuda()
    device = torch.device("cuda", torch.cuda.current_device())
    heads = TinySpeechHeads(device)
    batch_size = 4
    with torch.no_grad():
        inputs = make_inputs(batch_size, device)
        scales = heads.build_sampling_scales(batch_size)
        reference = heads.compute_acoustic_step(inputs, scales, None)
        constants = heads.build_flow_constants(batch_size)
        assert constants is not None
        fused = heads.compute_acoustic_step(inputs, scales, constants)
        after_packing = heads.compute_acoustic_step(inputs, scales, None)
    # Packing the projections leaves the module forward untouched.
    assert torch.equal(reference.latents, after_packing.latents)
    assert torch.equal(reference.feedback_embeddings, after_packing.feedback_embeddings)
    assert torch.equal(reference.stop_probabilities, fused.stop_probabilities)
    drift = (reference.latents - inputs.initial_noise.transpose(1, 2)).abs().mean()
    latent_error = (fused.latents - reference.latents).abs().mean()
    assert latent_error < 0.05 * drift
    feedback_error = (
        (fused.feedback_embeddings.float() - reference.feedback_embeddings.float())
        .abs()
        .mean()
    )
    assert feedback_error < 0.05 * reference.feedback_embeddings.float().abs().mean()


def make_audio_vae(device: torch.device) -> AudioVAE:
    backbone = {
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 4,
        "max_window_layers": 0,
        "use_sliding_window": True,
        "sliding_window": 16,
        "is_causal": True,
        "vocab_size": 1,
        "_attn_implementation": "sdpa",
    }
    vae = AudioVAE(
        AudioVAEconfig(
            enc_kwargs={
                "backbone": backbone,
                "input_dim": 32,
                "hop_size": 32,
                "latent_dim": LATENT_DIM,
            },
            dec_kwargs={
                "backbone": backbone,
                "output_dim": 32,
                "latent_dim": LATENT_DIM,
            },
            patch_size=4,
        )
    ).eval()
    vae.to(device=device, dtype=torch.bfloat16)
    return vae


@pytest.mark.parametrize(
    "use_cuda", [pytest.param(True, marks=pytest.mark.accelerator), False]
)
def test_vocoder_graphs_match_eager_decode_bit_for_bit(use_cuda: bool) -> None:
    if use_cuda:
        require_cuda()
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    vae = make_audio_vae(device)
    decoder = vae.decoder
    # Captured lengths cover one, two, and five latents; others run eagerly.
    graphs = DecoderGraphs(
        decoder.decoder, [4, 8, 20] if use_cuda else [], HIDDEN_SIZE, device
    )
    generator = torch.Generator(device=device).manual_seed(0)
    with torch.inference_mode():
        for latent_steps in (2, 5, 1, 3, 5):
            latents = torch.randn(
                (1, latent_steps, LATENT_DIM),
                device=device,
                dtype=torch.bfloat16,
                generator=generator,
            )
            eager, _, _ = vae.decode(latents)
            frames = decoder.upsampling.upsampler(
                decoder.fc1(latents).transpose(1, 2)
            ).transpose(1, 2)
            hidden_states = graphs.run(frames)
            # The head's kernels depend on its input layout, so the replay
            # must hand over the strides eager decode produces.
            assert (
                hidden_states.stride()
                == decoder.decoder(
                    inputs_embeds=frames, use_cache=False
                ).last_hidden_state.stride()
            )
            replayed, _, _, _ = decoder.head(
                hidden_states,
                streaming=False,
                audio_buffer=None,
                window_buffer=None,
                last_chunk=False,
            )
            assert torch.equal(eager, replayed)


def test_reference_audio_reads_bytes_and_keeps_remote_urls(tmp_path: Path) -> None:
    audio = b"RIFF0000WAVEfmt "
    path = tmp_path / "reference.wav"
    path.write_bytes(audio)
    encoded = base64.b64encode(audio).decode()
    assert read_reference_audio({"data": encoded}) == audio
    assert (
        read_reference_audio({"audio_path": f"data:audio/wav;base64,{encoded}"})
        == audio
    )
    assert read_reference_audio({"audio_path": str(path)}) == audio
    assert read_reference_audio({"audio_path": f"file://{path}"}) == audio
    remote = "https://example.com/reference.wav"
    assert read_reference_audio({"audio_path": remote}) == remote


@pytest.mark.accelerator
@pytest.mark.parametrize("add_noise", [True, False])
def test_solver_kernel_matches_eager_arithmetic_exactly(add_noise: bool) -> None:
    """Guidance, timestep and noise arithmetic must not hide under transformer tolerances."""
    require_cuda()
    generator = torch.Generator(device="cuda").manual_seed(327)
    velocity = torch.randn(
        (6, 1, LATENT_DIM), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    sample = torch.randn((3, 1, LATENT_DIM), device="cuda", generator=generator)
    guidance = torch.tensor([0.0, 1.25, 2.0], device="cuda")
    noise_scale = torch.tensor([0.0, 0.25, 0.75], device="cuda")
    step_size = torch.tensor([-0.125], device="cuda")
    noise_step_scale = torch.tensor([0.375], device="cuda")
    noise = torch.randn(sample.shape, device="cuda", generator=generator)
    conditional, unconditional = velocity.chunk(2)
    expected = sample + step_size * (
        conditional + (conditional - unconditional) * guidance[:, None, None]
    )
    if add_noise:
        expected = expected + noise_scale[:, None, None] * noise_step_scale * noise
    else:
        pass
    actual = cfg_solver_step(
        velocity,
        sample,
        guidance,
        noise_scale,
        step_size,
        noise_step_scale,
        noise,
        add_noise,
    )
    assert torch.equal(actual, expected)


@pytest.mark.accelerator
@pytest.mark.parametrize("temperature", [0.0, 0.25, 1.0])
def test_zero_velocity_solve_keeps_noise_timestep_and_temperature_scaling(
    temperature: float,
) -> None:
    """Noise uses each current timestep and is omitted on the last solver step."""
    require_cuda()
    device = torch.device("cuda", torch.cuda.current_device())
    heads = TinySpeechHeads(device)
    heads.flowloss.cfm.model.final_layer.linear.weight.data.zero_()
    heads.flowloss.cfm.model.final_layer.linear.bias.data.zero_()
    inputs = make_inputs(2, device)
    inputs.temperatures.fill_(temperature)
    inputs.solver_noise.copy_(
        torch.arange(1, FLOW_STEPS, device=device)[:, None, None, None]
    )
    timesteps = heads.flow_timesteps
    timesteps = timesteps - (torch.cos(torch.pi / 2 * timesteps) - 1 + timesteps)
    expected = inputs.initial_noise.transpose(1, 2).clone()
    for step in range(FLOW_STEPS - 1):
        expected = (
            expected
            + 0.25
            * (temperature**0.5)
            * abs(timesteps[step + 1] - timesteps[step]) ** 0.5
            * inputs.solver_noise[step]
        )
    with torch.no_grad():
        scales = heads.build_sampling_scales(2)
        constants = heads.build_flow_constants(2)
        actual = heads.compute_acoustic_step(inputs, scales, constants)
    assert torch.equal(actual.latents, expected)


def test_live_weight_updates_are_rejected_before_mutating_weights() -> None:
    """Engine-only updates would leave preprocessing and captured constants stale."""
    heads = TinySpeechHeads(torch.device("cpu"))
    heads.weights_loaded = True
    original = heads.stop_head.weight.clone()
    with pytest.raises(RuntimeError, match="restart the pipeline"):
        heads.load_weights([("stop_head.weight", torch.zeros_like(original))])
    assert torch.equal(original, heads.stop_head.weight)


@pytest.mark.parametrize(
    "use_cuda", [pytest.param(True, marks=pytest.mark.accelerator), False]
)
def test_seeded_reference_encoding_matches_fresh_posterior_sample(
    use_cuda: bool,
) -> None:
    """A private seeded posterior must equal a fresh encode without changing global RNG state."""
    if use_cuda:
        require_cuda()
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    vae = make_audio_vae(device)
    waveform = torch.linspace(-0.5, 0.5, 1024)
    with torch.inference_mode():
        original_state = torch.random.get_rng_state()
        first = stages.encode_reference_latents(
            vae, waveform, device, torch.Generator(device=device).manual_seed(42)
        )
        repeated = stages.encode_reference_latents(
            vae, waveform, device, torch.Generator(device=device).manual_seed(42)
        )
        changed = stages.encode_reference_latents(
            vae, waveform, device, torch.Generator(device=device).manual_seed(43)
        )
    assert torch.equal(first, repeated)
    assert not torch.equal(first, changed)
    assert torch.equal(original_state, torch.random.get_rng_state())


def test_tiny_prefill_preserves_token_ids_without_a_dense_graph_table() -> None:
    """Dense prefill graph staging must not allocate or index a table for Tiny."""
    runner = Ming3TTSModelRunner.__new__(Ming3TTSModelRunner)
    runner.device = torch.device("cpu")
    embeddings = nn.Embedding(16, HIDDEN_SIZE)
    runner.model = SimpleNamespace(
        config=Ming3TTSConfig(llm_config={"hidden_size": HIDDEN_SIZE}),
        get_input_embeddings=lambda: embeddings,
    )
    input_ids = torch.tensor([1, 2, 3])
    request = SimpleNamespace(
        input_ids=input_ids,
        latent_history=torch.zeros((1, LATENT_DIM)),
        speaker_position=-1,
        prompt_rows=None,
        feedback_embeddings=[],
    )
    batch = SimpleNamespace(
        input_ids=input_ids.clone(),
        extend_prefix_lens_cpu=[0],
        extend_seq_lens_cpu=[3],
        replace_embeds=None,
    )
    runner.before_prefill(
        batch, None, [SchedulerRequest(request_id="tiny", data=request)]
    )
    assert torch.equal(batch.input_ids, input_ids)
    assert torch.equal(
        get_omni_prefill_inputs(batch).input_embeds, embeddings(input_ids)
    )
