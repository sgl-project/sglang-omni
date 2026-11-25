# SPDX-License-Identifier: Apache-2.0
# Adapted from Stability-AI/stable-audio-3 at commit 790793b89ddceea612be9ce42ff6830d18f6fa57.
# Copyright (c) 2026 Stability AI. The MIT license notice is retained in licenses/.
"""Stable Audio 3 loading and batched text-to-audio generation with ping-pong sampling."""

import math
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import torch
from safetensors import safe_open
from torch._dynamo import mark_dynamic
from torch._dynamo.decorators import mark_unbacked
from transformers import PreTrainedTokenizerBase, T5GemmaEncoderModel

from sglang_omni.models.stable_audio3.autoencoder import AudioDecoder, DecoderBlock
from sglang_omni.models.stable_audio3.conditioning import Conditioner, load_text_encoder
from sglang_omni.models.stable_audio3.cuda_graphs import (
    DiTCudaGraphRunner,
    PromptEncoderCudaGraphRunner,
)
from sglang_omni.models.stable_audio3.dit import (
    DiffusionTransformer,
    SamplingConditions,
    TransformerBlock,
)
from sglang_omni.models.stable_audio3.layers import AttentionBackend
from sglang_omni.models.stable_audio3.model_config import (
    MAX_DURATION_SECONDS,
    MIN_DURATION_SECONDS,
    StableAudio3Config,
)

# note (ruilin): the checkpoint is distilled for 8 ping-pong steps, and its own inference pads every clip by 6 s.
SAMPLING_STEPS = 8
HEADROOM_SECONDS = 6.0
# note (ruilin): on an H200 one more generate call (step launches, prompt encoding, graph binding) costs about as much as 40 s of padded audio; it sets how finely a batch is split.
GROUP_OVERHEAD_SECONDS = 40.0
# note (ruilin): the checkpoint's sampling shift has rate 0, so the schedule does not depend on length.
LOG_SNR_START = -6.2
LOG_SNR_END = 2.0
CHECKPOINT_FILENAME = "model.safetensors"
COMPONENT_PREFIXES = {
    "dit": "model.model.",
    "conditioner": "conditioner.",
    "decoder": "pretransform.model.",
}
# note (ruilin): the encoder and the bottleneck's encode-side tensors serve audio-to-audio only.
UNUSED_PREFIXES = (
    "pretransform.model.encoder.",
    "pretransform.model.bottleneck.scaling_factor",
    "pretransform.model.bottleneck.bias",
    "pretransform.model.bottleneck.noise_scaling_factor",
)


@dataclass(frozen=True, kw_only=True)
class GenerationRequest:
    prompt: str
    duration_seconds: float
    seed: int


def sampling_schedule(device: torch.device) -> torch.Tensor:
    """Noise levels from 1 to 0, evenly spaced in log-SNR between the checkpoint's bounds."""
    timesteps = torch.linspace(1, 0, SAMPLING_STEPS + 1, device=device)
    log_snr_start = torch.tensor(LOG_SNR_START, device=device)
    log_snr = LOG_SNR_END - timesteps * (LOG_SNR_END - log_snr_start)
    noise_levels = torch.sigmoid(-log_snr)
    noise_levels = torch.where(
        timesteps <= 0, torch.zeros_like(noise_levels), noise_levels
    )
    return torch.where(timesteps >= 1, torch.ones_like(noise_levels), noise_levels)


def plan_length_groups(
    latent_lengths: list[int], overhead_latents: int
) -> list[list[int]]:
    """Cut a batch, sorted by length, into runs that each pad only to their own longest request.

    The cut minimizes the padded latents plus a fixed overhead per run, so a batch of equal lengths stays whole. Returns indices into latent_lengths.
    """
    order = sorted(range(len(latent_lengths)), key=lambda index: latent_lengths[index])
    best_cost = [0] + [math.inf] * len(order)
    run_start = [0] * (len(order) + 1)
    for end in range(1, len(order) + 1):
        for start in range(end):
            cost = (
                best_cost[start]
                + overhead_latents
                + (end - start) * latent_lengths[order[end - 1]]
            )
            if cost < best_cost[end]:
                best_cost[end] = cost
                run_start[end] = start
            else:
                pass
    groups = []
    end = len(order)
    while end > 0:
        groups.append(order[run_start[end] : end])
        end = run_start[end]
    return groups[::-1]


class StableAudio3Runtime:
    def __init__(
        self,
        *,
        config: StableAudio3Config,
        dit: DiffusionTransformer,
        conditioner: Conditioner,
        decoder: AudioDecoder,
        tokenizer: PreTrainedTokenizerBase,
        text_encoder: T5GemmaEncoderModel,
        device: torch.device,
    ) -> None:
        if config.sample_size < MAX_DURATION_SECONDS * config.sample_rate:
            raise ValueError(
                f"sample_size {config.sample_size} is shorter than {MAX_DURATION_SECONDS} s"
            )
        else:
            pass
        self.config = config
        self.dit = dit
        self.conditioner = conditioner
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
        self.dtype = next(dit.parameters()).dtype
        self.noise_levels = sampling_schedule(device)
        encoder = config.model.pretransform.config.encoder.config
        self.latent_alignment = encoder.chunk_size // encoder.strides[0]
        self.latent_channels = config.model.diffusion.config.io_channels
        self.dit_graphs: DiTCudaGraphRunner | None = None
        self.prompt_graphs: PromptEncoderCudaGraphRunner | None = None

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: Path,
        device: torch.device,
        dtype: torch.dtype,
        *,
        attention_backend: AttentionBackend,
    ) -> "StableAudio3Runtime":
        config = StableAudio3Config.from_checkpoint(checkpoint_dir)
        with torch.device("meta"):
            dit = DiffusionTransformer(
                config.model.diffusion.config, backend=attention_backend
            )
            conditioner = Conditioner(config.model.conditioning)
            decoder = AudioDecoder(
                config.model.pretransform.config, backend=attention_backend
            )
        modules: dict[str, DiffusionTransformer | Conditioner | AudioDecoder] = {
            "dit": dit,
            "conditioner": conditioner,
            "decoder": decoder,
        }
        with safe_open(
            checkpoint_dir / CHECKPOINT_FILENAME, framework="pt", device="cpu"
        ) as archive:
            for name, module in modules.items():
                prefix = COMPONENT_PREFIXES[name]
                weights = {
                    weight_name.removeprefix(prefix): archive.get_tensor(weight_name)
                    for weight_name in archive.keys()
                    if weight_name.startswith(prefix)
                    and not weight_name.startswith(UNUSED_PREFIXES)
                }
                module.load_state_dict(weights, strict=True, assign=True)
                module.to(device=device, dtype=dtype).eval().requires_grad_(False)
        tokenizer, text_encoder = load_text_encoder(
            checkpoint_dir / config.model.conditioning.configs[0].config.subfolder,
            device,
        )
        return cls(
            config=config,
            dit=dit,
            conditioner=conditioner,
            decoder=decoder,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            device=device,
        )

    def compile_blocks(self) -> None:
        """Compiles the DiT and decoder blocks, one compiled forward per stack, folding their norms, modulation and rotary into fewer kernels."""
        compiled_dit_block = torch.compile(type(self.dit.transformer.layers[0]).forward)
        compiled_decoder_block = torch.compile(
            type(self.decoder.decoder.resampling.transformers[0]).forward
        )

        def dit_block(
            block: TransformerBlock,
            hidden_states: torch.Tensor,
            *,
            context_keys_values: tuple[torch.Tensor, torch.Tensor],
            adaln_modulation: torch.Tensor,
            rotary_angles: torch.Tensor,
            key_lengths: torch.Tensor,
            local_embedding: torch.Tensor,
            num_memory_tokens: int,
        ) -> torch.Tensor:
            keys, values = context_keys_values
            # note (ruilin): an unbacked batch size never specializes to one, so a row's bits do not depend on the rows batched with it.
            for batched in (hidden_states, adaln_modulation, key_lengths, keys, values):
                mark_unbacked(batched, 0)
            mark_dynamic(hidden_states, 1)
            mark_dynamic(rotary_angles, 0)
            return compiled_dit_block(
                block,
                hidden_states,
                context_keys_values=context_keys_values,
                adaln_modulation=adaln_modulation,
                rotary_angles=rotary_angles,
                key_lengths=key_lengths,
                local_embedding=local_embedding,
                num_memory_tokens=num_memory_tokens,
            )

        def decoder_block(
            block: DecoderBlock, hidden_states: torch.Tensor, window_tokens: int
        ) -> torch.Tensor:
            mark_dynamic(hidden_states, [0, 1])
            return compiled_decoder_block(block, hidden_states, window_tokens)

        # note (ruilin): every block calls the one compiled forward of its stack, so the stack compiles once and its dynamic dimensions are marked before each call.
        for block in self.dit.transformer.layers:
            block.forward = MethodType(dit_block, block)
        for block in self.decoder.decoder.resampling.transformers:
            block.forward = MethodType(decoder_block, block)

    @torch.inference_mode()
    def capture_graphs(self) -> int:
        """Records every graph bucket a request can reach, so no request waits on a capture; returns how many graphs exist."""
        graph_count = 0
        if self.prompt_graphs is None:
            pass
        else:
            self.prompt_graphs.capture_buckets()
            graph_count += len(self.prompt_graphs.graphs)
        if self.dit_graphs is None:
            pass
        else:
            shortest_latents = self.latent_length(MIN_DURATION_SECONDS)
            # note (ruilin): the DiT graphs record against condition buffers sized by one loaded batch; any batch's conditions give them their shapes.
            self.dit_graphs.load_conditions(
                self.sampling_conditions(
                    [
                        GenerationRequest(
                            prompt="Warm piano chords",
                            duration_seconds=MIN_DURATION_SECONDS,
                            seed=0,
                        )
                    ],
                    shortest_latents,
                ),
                batch_size=1,
            )
            self.dit_graphs.capture_buckets(shortest_latents)
            graph_count += len(self.dit_graphs.graphs)
        return graph_count

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.prompt_graphs is None:
            return self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
        else:
            return self.prompt_graphs.encode(input_ids, attention_mask)

    def latent_length(self, duration_seconds: float) -> int:
        """Latents for the request plus headroom, aligned to the autoencoder and capped at the model's sample size."""
        samples_per_latent = self.decoder.downsampling_ratio
        alignment_samples = samples_per_latent * self.latent_alignment
        samples = int((duration_seconds + HEADROOM_SECONDS) * self.config.sample_rate)
        samples = math.ceil(samples / alignment_samples) * alignment_samples
        return min(samples, self.config.sample_size) // samples_per_latent

    def valid_latent_length(self, duration_seconds: float, latent_length: int) -> int:
        """Latents attention reads: the clip plus headroom, within the padded length."""
        samples_per_latent = self.decoder.downsampling_ratio
        headroom_latents = int(
            HEADROOM_SECONDS * self.config.sample_rate / samples_per_latent
        )
        content_latents = math.ceil(
            int(duration_seconds * self.config.sample_rate) / samples_per_latent
        )
        return min(content_latents + headroom_latents, latent_length)

    def noise(
        self, generator: torch.Generator, latent_length: int, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.randn(
            (1, self.latent_channels, latent_length),
            generator=generator,
            device=self.device,
            dtype=dtype,
        )

    def generate_in_length_groups(
        self, requests: list[GenerationRequest]
    ) -> list[torch.Tensor]:
        """Like generate, but requests of similar length run together, so a short request is not padded to a long one."""
        overhead_latents = round(
            GROUP_OVERHEAD_SECONDS
            * self.config.sample_rate
            / self.decoder.downsampling_ratio
        )
        groups = plan_length_groups(
            [self.latent_length(request.duration_seconds) for request in requests],
            overhead_latents,
        )
        audio: dict[int, torch.Tensor] = {}
        for group in groups:
            waveforms = self.generate([requests[index] for index in group])
            for index, waveform in zip(group, waveforms, strict=True):
                audio[index] = waveform
        return [audio[index] for index in range(len(requests))]

    def sampling_conditions(
        self, requests: list[GenerationRequest], latent_length: int
    ) -> SamplingConditions:
        """Prompt and duration conditions for a batch padded to latent_length; every sampling step reuses them."""
        seconds_embedding = self.conditioner.embed_seconds(
            [request.duration_seconds for request in requests]
        )
        condition_tokens = torch.cat(
            [
                self.conditioner.embed_prompts(
                    [request.prompt for request in requests],
                    self.tokenizer,
                    self.encode_text,
                ),
                seconds_embedding[:, None],
            ],
            dim=1,
        )
        valid_lengths = torch.tensor(
            [
                self.valid_latent_length(
                    request.duration_seconds,
                    self.latent_length(request.duration_seconds),
                )
                for request in requests
            ],
            device=self.device,
        )
        return self.dit.prepare(
            condition_tokens,
            seconds_embedding,
            valid_lengths=valid_lengths,
            latent_length=latent_length,
        )

    @torch.inference_mode()
    def generate(self, requests: list[GenerationRequest]) -> list[torch.Tensor]:
        """Stereo audio per request in the decoder's precision; each request draws its noise from its own seed, so batching does not change it."""
        latent_lengths = [
            self.latent_length(request.duration_seconds) for request in requests
        ]
        conditions = self.sampling_conditions(requests, max(latent_lengths))
        generators = [
            torch.Generator(self.device).manual_seed(request.seed)
            for request in requests
        ]
        latents = torch.zeros(
            len(requests),
            self.latent_channels,
            max(latent_lengths),
            device=self.device,
            dtype=self.dtype,
        )
        for row, (generator, length) in enumerate(
            zip(generators, latent_lengths, strict=True)
        ):
            latents[row, :, :length] = self.noise(generator, length, torch.float32)
        graphed_step = (
            None
            if self.dit_graphs is None
            else self.dit_graphs.bind(conditions, len(requests), latents.shape[-1])
        )
        for step in range(SAMPLING_STEPS):
            timestep = self.noise_levels[step].to(self.dtype)
            next_timestep = self.noise_levels[step + 1].to(self.dtype)
            if graphed_step is None:
                velocity = self.dit(latents, timestep.expand(len(requests)), conditions)
            else:
                velocity = graphed_step(latents, timestep)
            denoised = latents - timestep * velocity
            noise = torch.zeros_like(latents)
            for row, (generator, length) in enumerate(
                zip(generators, latent_lengths, strict=True)
            ):
                noise[row, :, :length] = self.noise(generator, length, self.dtype)
            latents = (1 - next_timestep) * denoised + next_timestep * noise
        audio: list[torch.Tensor] = []
        for row, (request, generator, length) in enumerate(
            zip(requests, generators, latent_lengths, strict=True)
        ):
            decoded = self.decoder(latents[row : row + 1, :, :length], generator)
            samples = int(request.duration_seconds * self.config.sample_rate)
            audio.append(decoded[0, :, :samples].clamp(-1, 1))
        return audio
