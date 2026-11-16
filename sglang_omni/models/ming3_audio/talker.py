# SPDX-License-Identifier: Apache-2.0
"""Continuous autoregressive speech synthesis for Ming 3 audio."""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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

from sglang_omni.models.ming3_audio.text_processing import (
    DEFAULT_ENABLE_TEXT_SEGMENTATION,
    DEFAULT_MAX_FRAGMENT_CHARS,
    DEFAULT_MAX_FRAGMENTS,
    DEFAULT_MAX_INPUT_CHARS,
    DEFAULT_MAX_SEMANTIC_LENGTH,
    DEFAULT_MAX_TOTAL_DECODE_STEPS,
    TextTooLargeError,
    maximum_admitted_fragment_count,
    plan_speech_text,
)
from sglang_omni.modules.ming_audio.flow.aggregator import Aggregator
from sglang_omni.modules.ming_audio.flow.cfm import CFM, get_epss_timesteps
from sglang_omni.modules.ming_audio.flow.dit import DiT
from sglang_omni.modules.ming_audio.vae.configuration_audio_vae import AudioVAEconfig
from sglang_omni.modules.ming_audio.vae.modeling_audio_vae import AudioVAE
from sglang_omni.utils.audio_features import cached_fbank

logger = logging.getLogger(__name__)

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
GenerationStopReason = Literal["natural_stop", "max_decode_steps_reached"]


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


@dataclass(frozen=True, kw_only=True)
class ReferenceConditioning:
    """Immutable per-request voice and acoustic prompt conditioning."""

    reference_latents: torch.Tensor
    reference_embeddings: torch.Tensor
    speaker_embedding: torch.Tensor | None


@dataclass(frozen=True, kw_only=True)
class GenerationOutcome:
    """Autoregressive latents and the Talker stop-head outcome."""

    latents: torch.Tensor
    actual_decode_steps: int
    stop_reason: GenerationStopReason


class MaxDecodeStepsReachedError(RuntimeError):
    """Raised when segmented synthesis exhausts a fragment's decode budget."""


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
        return self.generate_latents_with_status(
            embeddings, reference_latents, max_decode_steps, generator
        ).latents

    def generate_latents_with_status(
        self,
        embeddings: torch.Tensor,
        reference_latents: torch.Tensor,
        max_decode_steps: int,
        generator: torch.Generator,
    ) -> GenerationOutcome:
        """Generate latents while retaining the actual stop-head outcome."""
        if max_decode_steps <= 0:
            raise ValueError("max_decode_steps must be greater than zero")
        else:
            pass
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
        stop_reason: GenerationStopReason = "max_decode_steps_reached"
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
                stop_reason = "natural_stop"
                break
            else:
                pass
        return GenerationOutcome(
            latents=torch.cat(generated, dim=1),
            actual_decode_steps=len(generated),
            stop_reason=stop_reason,
        )


class Ming3SpeechGenerator:
    def __init__(
        self,
        model_path: str,
        device: torch.device,
        max_decode_steps: int,
        enable_cuda_graph: bool,
        *,
        enable_text_segmentation: bool = DEFAULT_ENABLE_TEXT_SEGMENTATION,
        max_semantic_length: int = DEFAULT_MAX_SEMANTIC_LENGTH,
        max_fragment_characters: int = DEFAULT_MAX_FRAGMENT_CHARS,
        max_fragments: int = DEFAULT_MAX_FRAGMENTS,
        max_input_characters: int = DEFAULT_MAX_INPUT_CHARS,
        max_total_decode_steps: int = DEFAULT_MAX_TOTAL_DECODE_STEPS,
    ) -> None:
        self.max_decode_steps: int = max_decode_steps
        self.enable_text_segmentation: bool = enable_text_segmentation
        self.max_semantic_length: int = max_semantic_length
        self.max_fragment_characters: int = max_fragment_characters
        self.max_fragments: int = max_fragments
        self.max_input_characters: int = max_input_characters
        self.max_total_decode_steps: int = max_total_decode_steps
        if self.enable_text_segmentation:
            self.get_max_admitted_fragment_count()
        else:
            pass
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

    def get_max_admitted_fragment_count(self) -> int:
        """Return the safe fragment ceiling from the configured worst-case budget."""
        if self.max_fragments <= 0:
            raise ValueError("max_fragments must be greater than zero")
        else:
            pass
        maximum_fragment_count = maximum_admitted_fragment_count(
            self.max_decode_steps, self.max_total_decode_steps
        )
        return min(self.max_fragments, maximum_fragment_count)

    def validate_decode_budget(self, fragment_count: int) -> None:
        """Reject a request whose worst-case fragment budget cannot fit.

        The ceiling stays conservative because emitted steps are not tracked here.
        """
        if fragment_count <= 0:
            raise ValueError("fragment_count must be greater than zero")
        else:
            pass
        maximum_admitted_fragment_count = self.get_max_admitted_fragment_count()
        if fragment_count > maximum_admitted_fragment_count:
            raise TextTooLargeError(
                f"planned {fragment_count} fragments at up to "
                f"{self.max_decode_steps} decode steps each; limit is "
                f"{self.max_total_decode_steps} decode steps and admits at most "
                f"{maximum_admitted_fragment_count} fragments"
            )
        else:
            pass

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        reference_audio: bytes | None,
        reference_text: str,
        seed: int,
    ) -> torch.Tensor:
        if self.enable_text_segmentation:
            text_plan = plan_speech_text(
                text,
                max_semantic_length=self.max_semantic_length,
                max_fragment_chars=self.max_fragment_characters,
                max_fragments=self.max_fragments,
                max_input_chars=self.max_input_characters,
            )
            fragments = text_plan.fragments
            self.validate_decode_budget(len(fragments))
        else:
            fragments = (text,)
        logger.info(
            f"Ming3 speech synthesis fragments={len(fragments)} "
            f"length_min={min(map(len, fragments))} "
            f"length_max={max(map(len, fragments))} "
            f"segmentation_enabled={self.enable_text_segmentation}"
        )
        conditioning = self.prepare_reference(reference_audio, seed)
        waveforms: list[torch.Tensor] = []
        fragment_count = len(fragments)
        for fragment_index, fragment in enumerate(fragments, start=1):
            try:
                waveform = self.synthesize_fragment(
                    fragment,
                    reference_text,
                    conditioning,
                    seed,
                    fragment_index,
                    fragment_count,
                )
            except MaxDecodeStepsReachedError:
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"Ming3 synthesis failed for fragment {fragment_index}/"
                    f"{fragment_count}"
                ) from exc
            else:
                pass
            if waveform.ndim != 1 or waveform.numel() == 0:
                raise RuntimeError(
                    f"Ming3 synthesis returned an empty or non-flat waveform for "
                    f"fragment {fragment_index}/{fragment_count}"
                )
            else:
                pass
            if not bool(torch.isfinite(waveform).all()):
                raise RuntimeError(
                    f"Ming3 synthesis returned non-finite audio for fragment "
                    f"{fragment_index}/{fragment_count}"
                )
            else:
                pass
            waveforms.append(waveform)
        if len(waveforms) == 1:
            return waveforms[0]
        else:
            pass
        return torch.cat(waveforms, dim=0)

    def prepare_reference(
        self, reference_audio: bytes | None, seed: int
    ) -> ReferenceConditioning:
        """Prepare voice conditioning once for the current user request."""
        talker = self.talker
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

        logger.info(
            f"Ming3 reference conditioning prepared "
            f"has_reference_audio={reference_audio is not None}"
        )
        return ReferenceConditioning(
            reference_latents=reference_latents,
            reference_embeddings=reference_embeddings,
            speaker_embedding=speaker_embedding,
        )

    def synthesize_fragment(
        self,
        text: str,
        reference_text: str,
        conditioning: ReferenceConditioning,
        seed: int,
        fragment_index: int,
        fragment_count: int,
    ) -> torch.Tensor:
        """Generate and decode one fragment with fresh autoregressive state."""
        talker = self.talker
        tokenizer = self.tokenizer
        speaker_prompt = (
            "  speaker_1:<|vision_start|><|vision_pad|><|vision_end|>\n"
            if conditioning.speaker_embedding is not None
            else ""
        )
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{TTS_PROMPT}{speaker_prompt} Text input:\n"
            f"{reference_text}{text}<|im_end|>\n<|im_start|>assistant\n<audio>"
        )
        prompt_ids = tokenizer.encode(prompt)
        reference_count = (
            conditioning.reference_embeddings.shape[1] if reference_text else 0
        )
        prompt_ids.extend(
            [tokenizer.convert_tokens_to_ids("<audioPatch>")] * reference_count
        )
        token_ids = torch.tensor([prompt_ids], device=self.device)
        embeddings = talker.model.embed_tokens(token_ids)
        if conditioning.speaker_embedding is not None:
            speaker_position = (
                prompt_ids.index(tokenizer.convert_tokens_to_ids("<|vision_start|>"))
                + 1
            )
            embeddings[0, speaker_position] = conditioning.speaker_embedding[0]
        else:
            pass
        if reference_count:
            embeddings[:, -reference_count:] = conditioning.reference_embeddings
        else:
            pass
        generator = torch.Generator(device=self.device).manual_seed(seed)
        generation_outcome = talker.generate_latents_with_status(
            embeddings,
            conditioning.reference_latents,
            self.max_decode_steps,
            generator,
        )
        logger.info(
            f"Ming3 fragment {fragment_index}/{fragment_count} "
            f"characters={len(text)} "
            f"actual_decode_steps={generation_outcome.actual_decode_steps} "
            f"stop_reason={generation_outcome.stop_reason} "
            f"cuda_graph_enabled={talker.enable_cuda_graph}"
        )
        if generation_outcome.stop_reason == "max_decode_steps_reached":
            if self.enable_text_segmentation:
                raise MaxDecodeStepsReachedError(
                    f"Ming3 fragment {fragment_index}/{fragment_count} reached "
                    f"max_decode_steps after "
                    f"{generation_outcome.actual_decode_steps} actual decode steps "
                    f"(stop_reason={generation_outcome.stop_reason})"
                )
            else:
                logger.warning(
                    f"Ming3 raw one-shot fragment {fragment_index}/{fragment_count} "
                    f"reached max_decode_steps after "
                    f"{generation_outcome.actual_decode_steps} actual decode steps "
                    f"(stop_reason={generation_outcome.stop_reason}); "
                    "preserving legacy segmentation-off audio output"
                )
        else:
            pass
        waveform, _, _ = self.vae.decode(
            generation_outcome.latents, use_cache=False, last_chunk=True
        )
        return waveform.flatten().float().cpu()
