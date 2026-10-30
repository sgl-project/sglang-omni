# SPDX-License-Identifier: Apache-2.0
"""Pipeline registration and backbone configuration contracts."""

import base64
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import sglang_omni.models.ming3_tts.tiny_model_runner as ming3_runner
from sglang_omni.models.ming3_tts import stages
from sglang_omni.models.ming3_tts.config import Ming3TTSPipelineConfig
from sglang_omni.models.ming3_tts.sglang_model import Ming3TTSConfig
from sglang_omni.models.ming3_tts.stages import (
    PromptConditioning,
    PromptLatentHook,
    PromptLatentInput,
)
from sglang_omni.models.ming3_tts.vocoder import create_vocoder_executor
from sglang_omni.models.ming3_tts.vocoder_graph import VocoderGraphRunner
from sglang_omni.models.model_capabilities import get_model_capabilities
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.modules.ming_audio.vae.modeling_audio_vae import AudioVAE
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.reference_encoder import ReferenceEncodeService

CHECKPOINT_ARCHITECTURE = "LingV3TTSForConditionalGeneration"
HIDDEN_SIZE = 64
LATENT_DIM = 8


def test_checkpoint_architecture_resolves_to_ming3_tts_pipeline() -> None:
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config(CHECKPOINT_ARCHITECTURE)
        is Ming3TTSPipelineConfig
    )
    capabilities = get_model_capabilities(CHECKPOINT_ARCHITECTURE)
    assert capabilities is not None
    assert capabilities.supports_reference_audio is True


def test_default_pipeline_ends_at_vocoder() -> None:
    config = Ming3TTSPipelineConfig(model_path="inclusionAI/Ming-3.0-tts-tiny")
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "tts_engine",
        "vocoder",
    ]
    assert config.terminal_stages == ["vocoder"]


def test_backbone_config_flattens_llm_config_and_keeps_speech_sections() -> None:
    config = Ming3TTSConfig(
        llm_config={
            "hidden_size": HIDDEN_SIZE,
            "model_type": "bailing_moe_v3",
        },
        tts_config={"latent_dim": 4},
        flow_config={"depth": 1},
        aggregator_config={"depth": 1},
    )
    assert config.model_type == "lingv3_tts"
    assert config.hidden_size == HIDDEN_SIZE
    assert config.tts_config == {"latent_dim": 4}
    assert config.flow_config == {"depth": 1}
    assert config.aggregator_config == {"depth": 1}


class LinearVocoderDecoder(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(4, 1, bias=False, device="cuda")
        self.latent_dim = 4
        with torch.no_grad():
            self.weight.fill_(1)


class LinearAudioVAE(AudioVAE):
    """Use a small decoder to exercise real graph storage without checkpoints."""

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.decoder = LinearVocoderDecoder()

    def decode(self, latents: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        return self.decoder(latents).squeeze(-1), None, None


@pytest.mark.accelerator
def test_vocoder_replay_owns_results_and_falls_back_for_uncaptured_lengths() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA graphs require a CUDA device")
    else:
        pass
    runner = VocoderGraphRunner(LinearAudioVAE(), maximum_decode_steps=2)
    retained = runner.decode(torch.ones(1, 2, 4, device="cuda"))
    subsequent = runner.decode(torch.zeros(1, 2, 4, device="cuda"))
    torch.testing.assert_close(retained, torch.full((1, 2), 4.0, device="cuda"))
    torch.testing.assert_close(subsequent, torch.zeros(1, 2, device="cuda"))
    fallback = runner.decode(torch.ones(1, 3, 4, device="cuda"))
    runner.decode(torch.zeros(1, 1, 4, device="cuda"))
    torch.testing.assert_close(fallback, torch.full((1, 3), 4.0, device="cuda"))


@pytest.mark.accelerator
def test_kda_startup_warms_both_branches_without_advancing_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cold multi-request prefill compiled a second KDA branch during serving."""
    if not torch.cuda.is_available() or ming3_runner.chunk_kda is None:
        pytest.skip("KDA warmup requires CUDA and Triton")
    else:
        pass
    native_prefill = ming3_runner.chunk_kda
    token_counts: list[int] = []

    def observe_prefill(
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        initial_state_indices: torch.Tensor,
        use_qk_l2norm_in_kernel: bool,
        cu_seqlens: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        lower_bound: float | None,
        beta_is_raw: bool,
    ) -> torch.Tensor:
        assert A_log.dtype == dt_bias.dtype == torch.float32
        token_counts.append(q.shape[1])
        output = native_prefill(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            beta_is_raw=beta_is_raw,
        )
        assert torch.isfinite(output).all().item()
        return output

    monkeypatch.setattr(ming3_runner, "chunk_kda", observe_prefill)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        ming3_runner.warmup_kda_prefill(
            head_count=16,
            key_channel_count=128,
            value_channel_count=128,
            lower_bound=-5.0,
            activation_dtype=torch.bfloat16,
            state_dtype=torch.float32,
            device=torch.device("cuda"),
        )
    finally:
        torch.set_default_dtype(default_dtype)
    assert token_counts == [64, 1088]
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())


@pytest.mark.parametrize("dense", [True, False])
def test_checkpoint_defaults_preserve_tiny_acceleration(
    tmp_path: Path, dense: bool
) -> None:
    """Both checkpoints retain their accelerated scheduling and graph defaults."""
    tmp_path = tmp_path / "checkpoint@local"
    tmp_path.mkdir()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "dense" if dense else "lingv3_tts",
                "ditar_config": {},
                "audio_tokenizer_config": {},
                "architectures": ["BailingMMNativeForConditionalGeneration"],
            }
        )
    )
    config = Ming3TTSPipelineConfig(model_path=str(tmp_path))
    preprocessing, engine, vocoder = config.stages
    assert preprocessing.gpu == 0
    assert engine.factory.enable_async_decode is True
    assert engine.factory.enable_fused_flow_head is dense
    assert bool(engine.factory.prefill_graph_token_counts)
    assert bool(vocoder.factory.graph_latent_steps)
    assert config.mps == "auto"
    rebuilt = Ming3TTSPipelineConfig.model_validate(config.model_dump())
    assert rebuilt.model_dump() == config.model_dump()


def test_prompt_cache_respects_content_and_seed_and_matches_fresh_encode() -> None:
    """A prompt cached with another seed must not replace a fresh seeded encode."""
    encodes: list[tuple[bytes, int]] = []

    def encode(reference_audio: bytes, seed: int) -> PromptConditioning:
        encodes.append((reference_audio, seed))
        generator = torch.Generator().manual_seed(seed)
        latents = torch.randn((2, LATENT_DIM), generator=generator)
        return PromptConditioning(prompt_latents=latents, prompt_rows=latents * 2)

    service = ReferenceEncodeService(
        PromptLatentHook(encode, model_identity="test"), max_bytes=1024
    )
    reference = PromptLatentInput(
        reference_key="same-content", reference_audio=b"audio", seed=42
    )
    first = service.get_or_encode(reference)
    repeated = service.get_or_encode(dict(reference))
    changed = service.get_or_encode({**reference, "seed": 43})
    assert encodes == [(b"audio", 42), (b"audio", 43)]
    fresh = encode(b"audio", 42)
    assert torch.equal(first["prompt_latents"], repeated["prompt_latents"])
    assert torch.equal(first["prompt_latents"], fresh["prompt_latents"])
    assert torch.equal(first["prompt_rows"], fresh["prompt_rows"])
    assert not torch.equal(first["prompt_latents"], changed["prompt_latents"])


@pytest.mark.parametrize(
    "disable_flag", ["disable_cuda_graph", "disable_prefill_cuda_graph", None]
)
def test_engine_factory_honors_operator_prefill_and_mixed_chunk_overrides(
    monkeypatch: pytest.MonkeyPatch, disable_flag: str | None
) -> None:
    """Stage defaults once re-enabled graphs and mixed chunk after explicit opt-outs."""
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda path: path)
    monkeypatch.setattr(
        stages,
        "load_checkpoint_config",
        lambda path: {
            "model_type": "dense",
            "audio_tokenizer_config": {"enc_kwargs": {"latent_dim": LATENT_DIM}},
        },
    )
    overrides = {
        "enable_mixed_chunk": False,
        "chunked_prefill_size": -1,
        "cuda_graph_config": {
            "decode": {"max_bs": 16},
            "prefill": {"backend": "breakable"},
        },
    }
    if disable_flag:
        overrides[disable_flag] = True
        overrides["cuda_graph_config"]["prefill"].pop("backend")
    else:
        pass
    captured = {}

    class StopBeforeInfrastructure(Exception):
        pass

    def build_server_args(checkpoint: str, **arguments: stages.JsonValue) -> None:
        captured.update(arguments)
        raise StopBeforeInfrastructure

    monkeypatch.setattr(stages, "build_sglang_server_args", build_server_args)
    with pytest.raises(StopBeforeInfrastructure):
        stages.create_tts_engine_executor(
            "checkpoint",
            max_seq_len=4096,
            enable_acoustic_graph=True,
            prefill_graph_token_counts=[64, 128],
            prefill_graph_max_requests=32,
            enable_async_decode=True,
            enable_fused_flow_head=True,
            device="cpu",
            server_args_overrides=overrides,
        )
    assert captured["enable_mixed_chunk"] is False
    assert captured["chunked_prefill_size"] == -1
    assert "cuda_graph_backend_prefill" not in captured
    assert captured["cuda_graph_config"]["decode"] == {"max_bs": 16}
    if disable_flag:
        assert "cuda_graph_bs_prefill" not in captured
    else:
        assert captured["cuda_graph_config"]["prefill"]["backend"] == "breakable"


@pytest.mark.parametrize("cache_entries", [0, 16])
@pytest.mark.parametrize("device", [None, "cpu"])
def test_preprocessing_cache_bypasses_unseeded_prompts_and_urls(
    monkeypatch: pytest.MonkeyPatch, cache_entries: int, device: str | None
) -> None:
    """Unseeded/remote references must encode afresh; zero entries must bypass both caches."""
    prompt_encodes: list[int | None] = []
    speaker_encodes: list[int] = []
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda path: path)
    monkeypatch.setattr(
        stages,
        "load_checkpoint_config",
        lambda path: {
            "model_type": "dense",
            "audio_tokenizer_config": {"sample_rate": 44100},
        },
    )
    monkeypatch.setattr(stages, "get_tokenizer", lambda *arguments, **keywords: None)
    monkeypatch.setattr(stages, "validate_dense_tokenizer", lambda tokenizer: None)

    def load_audio_vae(checkpoint, resolved_device, *, component):
        assert resolved_device == torch.device("cpu")
        return None, 44100

    monkeypatch.setattr(stages, "load_audio_vae", load_audio_vae)
    monkeypatch.setattr(
        stages, "load_prompt_aggregator", lambda *arguments: nn.Identity()
    )
    monkeypatch.setattr(
        stages,
        "load_audio",
        lambda *arguments, **keywords: np.zeros(1000, dtype=np.float32),
    )
    monkeypatch.setattr(
        stages, "cached_fbank", lambda *arguments, **keywords: torch.zeros((10, 80))
    )
    monkeypatch.setattr(torch.backends.cuda, "enable_cudnn_sdp", lambda enabled: None)

    class SpeakerSession:
        def get_inputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="audio")]

        def run(self, outputs: None, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
            speaker_encodes.append(1)
            return [np.ones(192, dtype=np.float32)]

    monkeypatch.setattr(
        stages.onnxruntime,
        "InferenceSession",
        lambda *arguments, **keywords: SpeakerSession(),
    )

    def encode(
        vae: None,
        waveform: torch.Tensor,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        seed = generator.initial_seed() if generator else None
        prompt_encodes.append(seed)
        return torch.full((2, LATENT_DIM), float(seed or 0))

    monkeypatch.setattr(stages, "encode_reference_latents", encode)
    scheduler = stages.create_preprocessing_executor(
        "checkpoint",
        max_concurrency=1,
        reference_cache_entries=cache_entries,
        reference_cache_bytes=1024,
        max_reference_seconds=120,
        device=device,
    )
    reference = {"data": base64.b64encode(b"audio").decode(), "text": "prompt"}
    seeds = [42, 42, 43, None, None, 42, 42]
    try:
        for index, seed in enumerate(seeds):
            audio = (
                reference
                if index < 5
                else {
                    "audio_path": "https://example.invalid/audio.wav",
                    "text": "prompt",
                }
            )
            payload = StagePayload(
                request_id=str(index),
                request=OmniRequest(
                    inputs={"text": "speech", "references": [audio]},
                    params={"seed": seed},
                ),
                data=None,
            )
            result = scheduler.fn(payload)
            assert torch.equal(
                result.data["prompt"]["prompt_latents"],
                torch.full((2, LATENT_DIM), float(seed or 0)),
            )
    finally:
        scheduler.stop()
    assert prompt_encodes == ([42, 43, None, None, 42, 42] if cache_entries else seeds)
    assert len(speaker_encodes) == (3 if cache_entries else 7)


def test_legacy_explicit_stage_factories_accept_previous_arguments() -> None:
    """Saved explicit stage configurations must remain callable after an upgrade."""

    inspect.signature(stages.create_preprocessing_executor).bind(
        model_path="checkpoint"
    )
    inspect.signature(stages.create_tts_engine_executor).bind(
        model_path="checkpoint", max_seq_len=4096, enable_acoustic_graph=True
    )
    inspect.signature(stages.create_vocoder_executor).bind(model_path="checkpoint")
    assert stages.create_vocoder_executor is create_vocoder_executor
