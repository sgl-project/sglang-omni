# SPDX-License-Identifier: Apache-2.0
"""Continuous autoregressive speech synthesis for Ming 3 audio."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import onnxruntime
import soundfile
import torch
import torch.nn.functional as functional
import torchaudio
from pydantic import BaseModel
from safetensors.torch import load_file
from torch import nn
from transformers import (
    AutoTokenizer,
    DynamicCache,
    PreTrainedTokenizerBase,
    Qwen2Config,
    Qwen2Model,
)

from sglang_omni.modules.ming_audio.flow.aggregator import Aggregator
from sglang_omni.modules.ming_audio.flow.cfm import CFM, get_epss_timesteps
from sglang_omni.modules.ming_audio.flow.dit import DiT
from sglang_omni.modules.ming_audio.vae.configuration_audio_vae import AudioVAEconfig
from sglang_omni.modules.ming_audio.vae.modeling_audio_vae import AudioVAE
from sglang_omni.utils.audio_features import cached_fbank

TTS_PROMPT = "Please generate speech based on the following description.\n"
MINIMUM_DECODE_STEPS = 3
STOP_PROBABILITY_THRESHOLD = 0.5
SPEAKER_SAMPLE_RATE = 16000
SPEAKER_MEL_BINS = 80
SPEAKER_EMBEDDING_DIM = 192
DIFFUSION_NOISE_SCALE = 0.25
DIFFUSION_TEMPERATURE = 0.0  # zero temperature drops the SDE noise term
SPEAKER_ENCODER_INTRA_OP_THREADS = 2
GRAPH_WARMUP_STEPS = 3


@dataclass(kw_only=True)
class AcousticGraph:
    graph: torch.cuda.CUDAGraph
    hidden: torch.Tensor
    history: torch.Tensor
    noise: torch.Tensor
    timesteps: torch.Tensor
    diffusion_parameters: torch.Tensor
    diffusion_noise: torch.Tensor
    latent: torch.Tensor
    embeddings: torch.Tensor
    stop_probability: torch.Tensor


class AcousticModuleConfig(BaseModel):
    in_channels: int
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: int = 4
    dropout: float = 0.1
    attn_backend: str = "flash_attn"
    attn_mask_enabled: bool = True
    pe_attn_head: int | None = None
    qk_norm: str | None = None


class TalkerConfig(BaseModel):
    aggregator: AcousticModuleConfig
    flowmodel: AcousticModuleConfig
    patch_size: int
    history_patch_size: int
    steps: int
    cfg_strength: float


class Ming3Talker(nn.Module):
    def __init__(
        self,
        config: TalkerConfig,
        language_config: Qwen2Config,
        enable_cuda_graph: bool,
    ) -> None:
        super().__init__()
        self.config: TalkerConfig = config
        self.model: Qwen2Model = Qwen2Model(language_config)
        self.cfm: CFM = CFM(
            DiT(
                llm_cond_dim=language_config.hidden_size,
                **config.flowmodel.model_dump(),
            ),
            steps=config.steps,
        )
        self.aggregator: Aggregator = Aggregator(
            llm_input_dim=language_config.hidden_size,
            **config.aggregator.model_dump(),
        )
        self.stop_head: nn.Linear = nn.Linear(language_config.hidden_size, 2)
        self.spk_head: nn.Linear = nn.Linear(
            SPEAKER_EMBEDDING_DIM, language_config.hidden_size
        )
        self.enable_cuda_graph: bool = enable_cuda_graph
        self.acoustic_graph: AcousticGraph | None = None

    def decode_acoustic(
        self,
        hidden: torch.Tensor,
        history: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        diffusion_parameters: torch.Tensor,
        diffusion_noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.cfm.sample(
            hidden, history, noise, timesteps, diffusion_parameters, diffusion_noise
        )
        embeddings = self.aggregator(latent)
        stop_probability = self.stop_head(hidden[:, -1]).softmax(dim=-1)[0, 1]
        return latent, embeddings, stop_probability

    def capture_acoustic_graph(
        self,
        hidden: torch.Tensor,
        history: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        diffusion_parameters: torch.Tensor,
        diffusion_noise: torch.Tensor,
    ) -> AcousticGraph:
        inputs = tuple(
            tensor.clone()
            for tensor in (
                hidden,
                history,
                noise,
                timesteps,
                diffusion_parameters,
                diffusion_noise,
            )
        )
        stream = torch.cuda.Stream(device=hidden.device)
        stream.wait_stream(torch.cuda.current_stream(hidden.device))
        with torch.cuda.stream(stream):
            for _ in range(GRAPH_WARMUP_STEPS):
                self.decode_acoustic(*inputs)
        torch.cuda.synchronize(hidden.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
            latent, embeddings, stop_probability = self.decode_acoustic(*inputs)
        return AcousticGraph(
            graph=graph,
            hidden=inputs[0],
            history=inputs[1],
            noise=inputs[2],
            timesteps=inputs[3],
            diffusion_parameters=inputs[4],
            diffusion_noise=inputs[5],
            latent=latent,
            embeddings=embeddings,
            stop_probability=stop_probability,
        )

    def generate_latents(
        self,
        embeddings: torch.Tensor,
        reference_latents: torch.Tensor,
        max_decode_steps: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        config = self.config
        history = functional.pad(
            reference_latents[:, -config.history_patch_size :],
            (0, 0, max(0, config.history_patch_size - reference_latents.shape[1]), 0),
        )
        timesteps = get_epss_timesteps(
            config.steps, embeddings.device, embeddings.dtype
        )
        diffusion_parameters = embeddings.new_tensor(
            [config.cfg_strength, DIFFUSION_NOISE_SCALE, DIFFUSION_TEMPERATURE]
        )
        cache = DynamicCache(config=self.model.config)
        generated: list[torch.Tensor] = []
        for step in range(max_decode_steps):
            hidden = self.model(
                inputs_embeds=embeddings, past_key_values=cache, use_cache=True
            ).last_hidden_state[:, -1:]
            noise = torch.randn(
                (1, config.patch_size, history.shape[-1]),
                device=embeddings.device,
                dtype=embeddings.dtype,
                generator=generator,
            )
            diffusion_noise = torch.randn(
                (config.steps, *noise.shape),
                device=noise.device,
                dtype=noise.dtype,
                generator=generator,
            )
            if self.enable_cuda_graph:
                if self.acoustic_graph is None:
                    self.acoustic_graph = self.capture_acoustic_graph(
                        hidden,
                        history,
                        noise,
                        timesteps,
                        diffusion_parameters,
                        diffusion_noise,
                    )
                else:
                    pass
                captured = self.acoustic_graph
                captured.hidden.copy_(hidden)
                captured.history.copy_(history)
                captured.noise.copy_(noise)
                captured.timesteps.copy_(timesteps)
                captured.diffusion_parameters.copy_(diffusion_parameters)
                captured.diffusion_noise.copy_(diffusion_noise)
                captured.graph.replay()
                # note (Codex): Returned patches outlive the graph's reusable output storage.
                latent = captured.latent.clone()
                embeddings = captured.embeddings
                stop_probability = captured.stop_probability
            else:
                latent, embeddings, stop_probability = self.decode_acoustic(
                    hidden,
                    history,
                    noise,
                    timesteps,
                    diffusion_parameters,
                    diffusion_noise,
                )
            history = torch.cat([history[:, config.patch_size :], latent], dim=1)
            generated.append(latent)
            if step > MINIMUM_DECODE_STEPS and bool(
                stop_probability > STOP_PROBABILITY_THRESHOLD
            ):
                break
            else:
                pass
        return torch.cat(generated, dim=1)


class Ming3SpeechGenerator:
    def __init__(
        self,
        model_path: str,
        device: torch.device,
        max_decode_steps: int,
        enable_cuda_graph: bool,
    ) -> None:
        path = Path(model_path) / "talker"
        configuration = TalkerConfig.model_validate_json(
            (path / "config.json").read_text()
        )
        language_config = Qwen2Config.from_pretrained(
            path / "llm", attn_implementation="sdpa"
        )
        self.talker: Ming3Talker = Ming3Talker(
            configuration, language_config, enable_cuda_graph
        ).eval()
        self.talker.load_state_dict(load_file(str(path / "model.safetensors")))
        self.talker.to(device=device, dtype=torch.bfloat16)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            path / "llm", trust_remote_code=False
        )
        vae_configuration = json.loads((path / "vae/config.json").read_text())
        for name in ("enc_kwargs", "dec_kwargs"):
            vae_configuration[name]["backbone"]["_attn_implementation"] = "sdpa"
        self.vae: AudioVAE = AudioVAE(AudioVAEconfig(**vae_configuration)).eval()
        self.vae.load_state_dict(load_file(str(path / "vae/model.safetensors")))
        self.vae.to(device=device, dtype=torch.bfloat16)
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = SPEAKER_ENCODER_INTRA_OP_THREADS
        self.speaker_encoder: onnxruntime.InferenceSession = (
            onnxruntime.InferenceSession(
                str(path / "campplus.onnx"),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
        )
        self.device: torch.device = device
        self.sample_rate: int = vae_configuration["sample_rate"]
        self.max_decode_steps: int = max_decode_steps

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        reference_audio: bytes | None,
        reference_text: str,
        seed: int,
    ) -> torch.Tensor:
        talker = self.talker
        tokenizer = self.tokenizer
        latent_channels = talker.aggregator.in_channels
        reference_latents = torch.empty(
            (1, 0, latent_channels), device=self.device, dtype=torch.bfloat16
        )
        reference_embeddings = torch.empty(
            (1, 0, talker.model.config.hidden_size),
            device=self.device,
            dtype=torch.bfloat16,
        )
        speaker_embedding: torch.Tensor | None = None
        if reference_audio is not None:
            waveform, sample_rate = soundfile.read(
                io.BytesIO(reference_audio), dtype="float32", always_2d=True
            )
            waveform = torch.from_numpy(waveform.mean(axis=1)).unsqueeze(0)
            speaker_waveform = torchaudio.functional.resample(
                waveform, sample_rate, SPEAKER_SAMPLE_RATE
            )
            features = cached_fbank(
                speaker_waveform,
                num_mel_bins=SPEAKER_MEL_BINS,
                sample_frequency=SPEAKER_SAMPLE_RATE,
            )
            features -= features.mean(dim=0, keepdim=True)
            speaker_input_name = self.speaker_encoder.get_inputs()[0].name
            speaker_vector = self.speaker_encoder.run(
                None, {speaker_input_name: features.unsqueeze(0).numpy()}
            )[0]
            speaker_embedding = talker.spk_head(
                torch.from_numpy(speaker_vector).to(self.device, dtype=torch.bfloat16)
            )
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.sample_rate
            )
            samples_per_patch = (
                self.vae.encoder.hop_size
                * self.vae.encoder.patch_size
                * talker.config.patch_size
            )
            waveform = functional.pad(
                waveform, ((-waveform.shape[-1]) % samples_per_patch, 0)
            )
            # note (Codex): VAE posterior sampling consumes the reference RNG before AR decoding.
            with torch.random.fork_rng(devices=[self.device]):
                torch.manual_seed(seed)
                reference_latents, _ = self.vae.encode_latent(
                    waveform.to(self.device, dtype=torch.bfloat16),
                    torch.tensor([waveform.shape[-1]], device=self.device),
                )
            patches = reference_latents.reshape(
                -1, talker.config.patch_size, latent_channels
            )
            reference_embeddings = talker.aggregator(patches).reshape(
                1, -1, talker.model.config.hidden_size
            )
        else:
            pass

        speaker_prompt = (
            "  speaker_1:<|vision_start|><|vision_pad|><|vision_end|>\n"
            if speaker_embedding is not None
            else ""
        )
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{TTS_PROMPT}{speaker_prompt} Text input:\n"
            f"{reference_text}{text}<|im_end|>\n<|im_start|>assistant\n<audio>"
        )
        prompt_ids = tokenizer.encode(prompt)
        reference_count = reference_embeddings.shape[1] if reference_text else 0
        prompt_ids.extend(
            [tokenizer.convert_tokens_to_ids("<audioPatch>")] * reference_count
        )
        token_ids = torch.tensor([prompt_ids], device=self.device)
        embeddings = talker.model.embed_tokens(token_ids)
        if speaker_embedding is not None:
            speaker_position = (
                prompt_ids.index(tokenizer.convert_tokens_to_ids("<|vision_start|>"))
                + 1
            )
            embeddings[0, speaker_position] = speaker_embedding[0]
        else:
            pass
        if reference_count:
            embeddings[:, -reference_count:] = reference_embeddings
        else:
            pass
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = talker.generate_latents(
            embeddings, reference_latents, self.max_decode_steps, generator
        )
        waveform, _, _ = self.vae.decode(latents, use_cache=False, last_chunk=True)
        return waveform.flatten().float().cpu()
