# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modifications: retain MiniCPM-o inference only; local imports and typing.
"""Flow for MiniCPM-o."""

from __future__ import annotations

import math
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from sglang_omni.models.minicpm_o.components.token2wav.conformer import (
    UpsampleConformerEncoderV2,
    make_pad_mask,
)
from sglang_omni.models.minicpm_o.components.token2wav.conformer_state import (
    ConformerState,
)
from sglang_omni.models.minicpm_o.components.token2wav.dit import DiT, DiTState
from sglang_omni.models.minicpm_o.components.token2wav.fixed_packed import (
    FixedPackedLayout,
    build_fixed_packed_layout,
)
from sglang_omni.models.minicpm_o.components.token2wav.flow_cuda_graph import (
    CapturedPackedFlowGraph,
    FlowCudaGraphRunner,
)
from sglang_omni.models.minicpm_o.components.token2wav.packed_dit_cuda_graph import (
    CapturedPackedDiTGraph,
)

# note (Junnan Li): Classifier-free guidance runs the estimator on all conditioned rows, then all unconditioned rows, so a stream's estimator caches hold its two rows on this axis.
ESTIMATOR_GUIDANCE_AXIS = 2


@dataclass(frozen=True, kw_only=True)
class RaggedLayout:
    """Each row's real new frames and history frames, right-padded to padded_frames and padded_history."""

    frame_counts: list[int]
    history_counts: list[int]
    padded_frames: int
    padded_history: int

    def attention_mask(self, device: torch.device) -> torch.Tensor:
        """Keys are [new frames, history] as Attention concatenates them; a row sees only its own real keys."""
        padded_frames = self.padded_frames
        keys = torch.arange(padded_frames + self.padded_history, device=device)
        frame_counts = torch.tensor(self.frame_counts, device=device)
        history_counts = torch.tensor(self.history_counts, device=device)
        visible = (keys[None, :] < frame_counts[:, None]) | (
            (keys[None, :] >= padded_frames)
            & (keys[None, :] < padded_frames + history_counts[:, None])
        )
        return visible[:, None, :].expand(-1, padded_frames, -1)


@dataclass(frozen=True, kw_only=True)
class ChunkGraph:
    """One captured Euler loop over static inputs for a fixed stream count, chunk length and history capacity."""

    graph: torch.cuda.CUDAGraph
    stream_count: int
    frame_count: int
    history_capacity: int
    noise: torch.Tensor
    mu: torch.Tensor
    speaker_embeddings: torch.Tensor
    convolution_cache: torch.Tensor
    attention_window: torch.Tensor
    attention_mask: torch.Tensor
    valid_frame_counts: torch.Tensor
    mel: torch.Tensor
    next_convolution_cache: torch.Tensor


class CausalConditionalCFM(torch.nn.Module):

    def __init__(self, estimator: DiT, inference_cfg_rate: float = 0.7) -> None:
        super().__init__()
        self.estimator = estimator
        self.inference_cfg_rate = inference_cfg_rate
        self.out_channels = estimator.out_channels
        self.graph_runner: FlowCudaGraphRunner | None = None
        self.register_buffer(
            "rand_noise",
            torch.randn([1, self.out_channels, 50 * 600]),
            persistent=False,
        )
        # note (Junnan Li): Sorted by stream count, then chunk length, so the first that fits a forward is the smallest.
        self.chunk_graphs: list[ChunkGraph] = []

    def solve_euler(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        mel_conditioning: torch.Tensor,
        states: list[DiTState] | None = None,
        *,
        packed_valid_frames: int | None = None,
    ) -> tuple[torch.Tensor, list[DiTState] | None]:
        t = t_span[0].expand(x.size(0))
        dt = t_span[1] - t_span[0]
        assert self.inference_cfg_rate > 0, "inference_cfg_rate better > 0"
        paired_mask = torch.cat([mask, mask], dim=0)
        paired_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)
        paired_speaker_embeddings = torch.cat(
            [speaker_embeddings, torch.zeros_like(speaker_embeddings)], dim=0
        )
        paired_mel_conditioning = torch.cat(
            [mel_conditioning, torch.zeros_like(mel_conditioning)], dim=0
        )
        graph_runner = self.graph_runner if states is None else None
        packed_runner = self.estimator.packed_graph_runner if states is None else None
        next_states: list[DiTState] = []
        is_packed = self.estimator.enable_variable_length and x.shape[0] > 1
        packed_capacity = (
            packed_runner.fit(x.shape[0], x.shape[2], packed_valid_frames)
            if packed_runner is not None and is_packed
            else None
        )
        graph_shape = (
            graph_runner.fit(x, t_span)
            if graph_runner is not None
            and (not is_packed or packed_capacity is not None)
            else None
        )
        packed_layout = (
            build_fixed_packed_layout(
                paired_mask.bool().squeeze(1).sum(dim=1, dtype=torch.int32),
                graph_shape[1] if graph_shape is not None else x.shape[2],
                packed_capacity,
                self.estimator.blocks[0].conv.kernel_size - 1,
            )
            if packed_capacity is not None
            else None
        )
        with (
            graph_runner.lock if graph_shape is not None else nullcontext(),
            packed_runner.lock if packed_layout is not None else nullcontext(),
        ):
            captured_flow = (
                graph_runner.prepare(
                    graph_shape,
                    x,
                    paired_mu,
                    paired_mask,
                    paired_speaker_embeddings,
                    paired_mel_conditioning,
                    packed_layout,
                )
                if graph_shape is not None
                else None
            )
            captured_packed = (
                packed_runner.prepare(packed_layout)
                if packed_layout is not None
                else None
            )
            for step in range(1, len(t_span)):
                if states is not None:
                    derivative, next_state = self.estimator.forward_chunk(
                        torch.cat([x, x], dim=0),
                        paired_mu,
                        torch.cat([t, t], dim=0),
                        paired_speaker_embeddings,
                        paired_mel_conditioning,
                        states[step - 1],
                    )
                    next_states.append(next_state)
                    conditional, unconditional = derivative.chunk(2, dim=0)
                    x = x + dt * (
                        (1.0 + self.inference_cfg_rate) * conditional
                        - self.inference_cfg_rate * unconditional
                    )
                elif isinstance(captured_flow, CapturedPackedFlowGraph):
                    graph_runner.run_packed_step(
                        captured_flow, packed_runner, captured_packed, t, dt
                    )
                elif captured_flow is not None:
                    graph_runner.run_step(captured_flow, t, dt)
                else:
                    x = self.euler_step(
                        x,
                        t,
                        dt,
                        paired_mu,
                        paired_mask,
                        paired_speaker_embeddings,
                        paired_mel_conditioning,
                        packed_layout=packed_layout,
                        packed_graph=captured_packed,
                    )
                t = t + dt
                if step < len(t_span) - 1:
                    dt = t_span[step + 1] - t_span[step]
                else:
                    pass
            output = (
                captured_flow.output[:, :, : x.shape[2]].clone()
                if captured_flow is not None
                else x
            )
            return output, next_states if states is not None else None

    def euler_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        paired_mu: torch.Tensor,
        paired_mask: torch.Tensor,
        paired_speaker_embeddings: torch.Tensor,
        paired_mel_conditioning: torch.Tensor,
        *,
        packed_layout: FixedPackedLayout | None = None,
        packed_graph: CapturedPackedDiTGraph | None = None,
    ) -> torch.Tensor:
        derivative = self.estimator.forward(
            torch.cat([x, x], dim=0),
            paired_mask,
            paired_mu,
            torch.cat([t, t], dim=0),
            paired_speaker_embeddings,
            paired_mel_conditioning,
            packed_layout=packed_layout,
            packed_graph=packed_graph,
        )
        conditional_derivative, unconditional_derivative = derivative.chunk(2, dim=0)
        guided_derivative = (
            (1.0 + self.inference_cfg_rate) * conditional_derivative
            - self.inference_cfg_rate * unconditional_derivative
        )
        return x + dt * guided_derivative

    @torch.inference_mode()
    def forward(
        self,
        mu: torch.Tensor,
        mask: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        mel_conditioning: torch.Tensor,
        n_timesteps: int = 10,
        temperature: float = 1.0,
        states: list[DiTState] | None = None,
        noise_offsets: list[int] | None = None,
        *,
        packed_valid_frames: int | None = None,
    ) -> tuple[torch.Tensor, list[DiTState] | None]:
        if n_timesteps <= 0:
            raise ValueError("n_timesteps must be positive")
        else:
            pass
        if noise_offsets is None:
            noise_offsets = [0] * mu.size(0)
        else:
            pass
        return self.solve_euler(
            self.draw_noise(noise_offsets, mu.size(2), temperature),
            self.time_span(n_timesteps, mu.device, mu.dtype),
            mu,
            mask,
            speaker_embeddings,
            mel_conditioning,
            states,
            packed_valid_frames=packed_valid_frames,
        )

    def draw_noise(
        self, noise_offsets: list[int], frames: int, temperature: float
    ) -> torch.Tensor:
        if max(noise_offsets) + frames > self.rand_noise.size(2):
            raise ValueError(
                "Combined reference and generated audio exceed 600 seconds"
            )
        else:
            pass
        return (
            torch.cat(
                [
                    self.rand_noise[:, :, offset : offset + frames]
                    for offset in noise_offsets
                ]
            )
            * temperature
        )

    def time_span(
        self, n_timesteps: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=device, dtype=dtype)
        return 1 - torch.cos(t_span * 0.5 * torch.pi)

    def capture_chunk_graphs(
        self,
        *,
        stream_counts: tuple[int, ...],
        frame_counts: tuple[int, ...],
        history_capacity: int,
        convolution_cache: torch.Tensor,
        attention_cache: torch.Tensor,
    ) -> None:
        """Capture a stream continuation's Euler loop for every stream count and chunk length.

        All graphs share one attention window storage and one memory pool, since a forward replays one at a time.
        """
        device = attention_cache.device
        steps, depth, _, heads, _, key_value_width = attention_cache.shape
        convolution_channels, convolution_history = convolution_cache.shape[3:]
        activation_dtype = attention_cache.dtype
        channels = self.out_channels
        window_storage = torch.empty(
            steps
            * depth
            * 2
            * max(stream_counts)
            * heads
            * (max(frame_counts) + history_capacity)
            * key_value_width,
            dtype=activation_dtype,
            device=device,
        )
        pool = torch.cuda.graph_pool_handle()
        for stream_count in sorted(stream_counts, reverse=True):
            for frame_count in sorted(frame_counts, reverse=True):
                guided_rows = 2 * stream_count
                window_shape = (
                    steps,
                    depth,
                    guided_rows,
                    heads,
                    frame_count + history_capacity,
                    key_value_width,
                )
                attention_window = (
                    window_storage[: math.prod(window_shape)].view(window_shape).zero_()
                )
                noise = torch.zeros(
                    stream_count,
                    channels,
                    frame_count,
                    dtype=self.rand_noise.dtype,
                    device=device,
                )
                mu = torch.zeros(
                    stream_count,
                    channels,
                    frame_count,
                    dtype=activation_dtype,
                    device=device,
                )
                speaker_embeddings = torch.zeros(
                    stream_count, channels, dtype=activation_dtype, device=device
                )
                convolution = torch.zeros(
                    steps,
                    depth,
                    guided_rows,
                    convolution_channels,
                    convolution_history,
                    dtype=convolution_cache.dtype,
                    device=device,
                )
                attention_mask = (
                    RaggedLayout(
                        frame_counts=[frame_count] * stream_count,
                        history_counts=[0] * stream_count,
                        padded_frames=frame_count,
                        padded_history=history_capacity,
                    )
                    .attention_mask(device)
                    .repeat(2, 1, 1)
                )
                valid_frame_counts = torch.full(
                    (guided_rows,), frame_count, device=device
                )
                states = [
                    DiTState(
                        convolution=convolution[step],
                        attention=attention_window[step],
                        attention_mask=attention_mask,
                        valid_frame_counts=valid_frame_counts,
                    )
                    for step in range(steps)
                ]

                def run_chunk() -> tuple[torch.Tensor, torch.Tensor]:
                    # note (Junnan Li): Constant inputs are made inside the run, so the graph's pool owns them.
                    mel, next_states = self.solve_euler(
                        noise,
                        self.time_span(steps, device, activation_dtype),
                        mu,
                        torch.ones_like(mu[:, :1]),
                        speaker_embeddings,
                        torch.zeros_like(mu),
                        states,
                    )
                    assert next_states is not None
                    return mel, torch.stack(
                        [state.convolution for state in next_states]
                    )

                current_stream = torch.cuda.current_stream(device)
                side_stream = torch.cuda.Stream(device)
                side_stream.wait_stream(current_stream)
                with torch.cuda.stream(side_stream):
                    run_chunk()
                current_stream.wait_stream(side_stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    mel, next_convolution_cache = run_chunk()
                self.chunk_graphs.append(
                    ChunkGraph(
                        graph=graph,
                        stream_count=stream_count,
                        frame_count=frame_count,
                        history_capacity=history_capacity,
                        noise=noise,
                        mu=mu,
                        speaker_embeddings=speaker_embeddings,
                        convolution_cache=convolution,
                        attention_window=attention_window,
                        attention_mask=attention_mask,
                        valid_frame_counts=valid_frame_counts,
                        mel=mel,
                        next_convolution_cache=next_convolution_cache,
                    )
                )
        self.chunk_graphs.sort(
            key=lambda graph: (graph.stream_count, graph.frame_count)
        )

    def chunk_graph(
        self, stream_count: int, frame_count: int, history_count: int
    ) -> ChunkGraph | None:
        """The smallest captured graph a forward fits in; None runs the forward eagerly."""
        for graph in self.chunk_graphs:
            if (
                graph.stream_count >= stream_count
                and graph.frame_count >= frame_count
                and graph.history_capacity >= history_count
            ):
                return graph
            else:
                pass
        return None

    def replay_chunk(
        self,
        graph: ChunkGraph,
        mu: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        convolution_cache: torch.Tensor,
        layout: RaggedLayout,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return copies of the mel and the next convolution caches of a graph whose attention window holds the histories.

        Inputs and layout cover the graph's padded rows.
        """
        graph.noise.copy_(
            self.draw_noise(layout.history_counts, graph.frame_count, temperature=1.0)
        )
        graph.mu.copy_(mu)
        graph.speaker_embeddings.copy_(speaker_embeddings)
        graph.convolution_cache.copy_(convolution_cache)
        graph.attention_mask.copy_(layout.attention_mask(mu.device).repeat(2, 1, 1))
        graph.valid_frame_counts.copy_(
            torch.tensor(layout.frame_counts, device=mu.device).repeat(2)
        )
        graph.graph.replay()
        return graph.mel.clone(), graph.next_convolution_cache.clone()

    @torch.inference_mode()
    def forward_chunk(
        self,
        mu: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        mel_conditioning: torch.Tensor,
        n_timesteps: int = 10,
        temperature: float = 1.0,
        convolution_cache: torch.Tensor | None = None,
        attention_window: torch.Tensor | None = None,
        layout: RaggedLayout | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the mel, the next convolution caches and the attention window filled with the new frames.

        attention_window holds [new frames, history] per step, history written; None starts a stream.
        """
        if attention_window is None:
            noise_offsets = [0] * mu.size(0)
            states = [DiTState() for _ in range(n_timesteps)]
        else:
            assert convolution_cache is not None
            if layout is None:
                noise_offsets = [attention_window.shape[4] - mu.size(2)] * mu.size(0)
                attention_mask = None
                valid_frame_counts = None
            else:
                # note (Junnan Li): stepaudio2 starts a chunk's noise at its own history length.
                noise_offsets = layout.history_counts
                attention_mask = layout.attention_mask(mu.device).repeat(2, 1, 1)
                valid_frame_counts = torch.tensor(
                    layout.frame_counts, device=mu.device
                ).repeat(2)
            states = [
                DiTState(
                    convolution=convolution_cache[index],
                    attention=attention_window[index],
                    attention_mask=attention_mask,
                    valid_frame_counts=valid_frame_counts,
                )
                for index in range(n_timesteps)
            ]
        result, next_states = self.forward(
            mu,
            torch.ones_like(mu[:, :1]),
            speaker_embeddings,
            mel_conditioning,
            n_timesteps,
            temperature,
            states,
            noise_offsets,
        )
        assert next_states is not None
        if attention_window is None:
            next_attention_window = torch.stack(
                [state.attention for state in next_states]
            )
        else:
            next_attention_window = attention_window
        return (
            result,
            torch.stack([state.convolution for state in next_states]),
            next_attention_window,
        )


class CausalMaskedDiffWithXvec(torch.nn.Module):

    def __init__(
        self,
        encoder: UpsampleConformerEncoderV2,
        decoder: CausalConditionalCFM,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: Literal["mel"] = "mel",
        vocab_size: int = 6561,
    ) -> None:
        super().__init__()
        if output_type != "mel":
            raise ValueError("MiniCPM-o flow output must be mel")
        else:
            pass
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.pre_lookahead_len = int(encoder.pre_lookahead_layer.pre_lookahead_len)
        self.up_rate = int(encoder.up_layer.stride)
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.speaker_embedding_projection = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_dim, output_size)
        self.decoder = decoder

    @torch.inference_mode()
    def inference(
        self,
        speech_tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        prompt_tokens: torch.Tensor,
        prompt_token_lengths: torch.Tensor,
        prompt_mel: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        n_timesteps: int = 10,
    ) -> torch.Tensor:
        assert speech_tokens.shape[0] == prompt_tokens.shape[0], (
            f"flow batch size mismatch: speech_tokens={speech_tokens.shape[0]} "
            f"prompt_tokens={prompt_tokens.shape[0]}"
        )
        speaker_embeddings = F.normalize(speaker_embeddings, dim=1)
        speaker_embeddings = self.speaker_embedding_projection(speaker_embeddings)
        prompt_row_lengths = prompt_token_lengths.tolist()
        generated_row_lengths = token_lengths.tolist()
        combined_tokens = pad_sequence(
            [
                torch.cat(
                    [prompt_tokens[i, :prompt_length], speech_tokens[i, :token_length]]
                )
                for i, (prompt_length, token_length) in enumerate(
                    zip(prompt_row_lengths, generated_row_lengths, strict=True)
                )
            ],
            batch_first=True,
        )
        combined_token_lengths = prompt_token_lengths + token_lengths
        token_mask = (
            (~make_pad_mask(combined_token_lengths))
            .unsqueeze(-1)
            .to(speaker_embeddings)
        )
        embedded_tokens = (
            self.input_embedding(torch.clamp(combined_tokens, min=0)) * token_mask
        )
        hidden_states, _ = self.encoder.forward(embedded_tokens, combined_token_lengths)
        frame_mask = (
            ~make_pad_mask(
                combined_token_lengths * self.up_rate, hidden_states.shape[1]
            )
        ).to(hidden_states)
        hidden_states = self.encoder_proj(hidden_states) * frame_mask.unsqueeze(-1)
        mel_conditioning = torch.zeros_like(hidden_states)
        for i, prompt_length in enumerate(prompt_row_lengths):
            prompt_frames = prompt_length * self.up_rate
            mel_conditioning[i, :prompt_frames] = prompt_mel[i, :prompt_frames]
        mel_conditioning = mel_conditioning.transpose(1, 2).contiguous()
        packed_valid_frames = (
            2
            * sum(
                min(
                    (prompt_length + token_length) * self.up_rate,
                    hidden_states.shape[1],
                )
                for prompt_length, token_length in zip(
                    prompt_row_lengths, generated_row_lengths, strict=True
                )
            )
            if len(generated_row_lengths) > 1
            else None
        )
        predicted_mel, _ = self.decoder.forward(
            mu=hidden_states.transpose(1, 2).contiguous(),
            mask=frame_mask.unsqueeze(1),
            speaker_embeddings=speaker_embeddings,
            mel_conditioning=mel_conditioning,
            n_timesteps=n_timesteps,
            packed_valid_frames=packed_valid_frames,
        )
        generated = [
            predicted_mel[
                i,
                :,
                prompt_length
                * self.up_rate : (prompt_length + token_length)
                * self.up_rate,
            ]
            for i, (prompt_length, token_length) in enumerate(
                zip(prompt_row_lengths, generated_row_lengths, strict=True)
            )
        ]
        return pad_sequence(
            [row.transpose(0, 1) for row in generated], batch_first=True
        ).transpose(1, 2)

    @torch.inference_mode()
    def setup_cache(
        self,
        token_ids: torch.Tensor,
        prompt_mel: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        n_timesteps: int = 10,
    ) -> dict[str, torch.Tensor]:
        assert (
            token_ids.shape[1] - self.pre_lookahead_len
        ) * self.up_rate == prompt_mel.shape[1], (token_ids.shape, prompt_mel.shape)
        _, cache = self.inference_chunk(
            token_ids,
            speaker_embeddings,
            n_timesteps=n_timesteps,
            prompt_mel=prompt_mel,
        )
        return cache

    @torch.inference_mode()
    def inference_chunk(
        self,
        token_ids: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        is_last_chunk: bool = False,
        n_timesteps: int = 10,
        prompt_mel: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Start a stream, conditioned on prompt_mel when one is given."""
        speaker_embeddings = F.normalize(speaker_embeddings, dim=1)
        speaker_embeddings = self.speaker_embedding_projection(speaker_embeddings)
        hidden_states, conformer_state = self.encoder.forward_chunk(
            xs=self.input_embedding(token_ids),
            is_last_chunk=is_last_chunk,
            state=ConformerState(),
        )
        conformer_convolution_cache, conformer_attention_cache = (
            conformer_state.to_packed(self.encoder.up_layer.stride)
        )
        hidden_states = self.encoder_proj(hidden_states)
        mel_conditioning = (
            torch.zeros_like(hidden_states) if prompt_mel is None else prompt_mel
        )
        predicted_mel, estimator_convolution_cache, estimator_attention_cache = (
            self.decoder.forward_chunk(
                mu=hidden_states.transpose(1, 2).contiguous(),
                speaker_embeddings=speaker_embeddings,
                mel_conditioning=mel_conditioning.transpose(1, 2).contiguous(),
                n_timesteps=n_timesteps,
                temperature=1.0,
            )
        )
        return predicted_mel, {
            "conformer_convolution_cache": conformer_convolution_cache,
            "conformer_attention_cache": conformer_attention_cache,
            "estimator_convolution_cache": estimator_convolution_cache,
            "estimator_attention_cache": estimator_attention_cache,
        }

    @torch.inference_mode()
    def inference_chunks(
        self,
        token_ids: list[torch.Tensor],
        speaker_embeddings: torch.Tensor,
        caches: list[dict[str, torch.Tensor]],
        is_last_chunk: list[bool],
        n_timesteps: int = 10,
    ) -> list[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        """Decode one streaming chunk of each of several streams with one estimator pass or chunk graph replay.

        Pops each stream's estimator attention cache; the returned caches replace it.
        """
        speaker_embeddings = F.normalize(speaker_embeddings, dim=1)
        speaker_embeddings = self.speaker_embedding_projection(speaker_embeddings)
        encoder_groups: dict[tuple[int, bool, int], list[int]] = defaultdict(list)
        for row, (stream_token_ids, cache, stream_is_last_chunk) in enumerate(
            zip(token_ids, caches, is_last_chunk, strict=True)
        ):
            encoder_groups[
                (
                    stream_token_ids.shape[1],
                    stream_is_last_chunk,
                    cache["conformer_attention_cache"].shape[3],
                )
            ].append(row)
        hidden_states_by_row: dict[int, torch.Tensor] = {}
        conformer_caches: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for (_, group_is_last_chunk, _), rows in encoder_groups.items():
            conformer_state = ConformerState.from_packed(
                torch.cat([caches[row]["conformer_convolution_cache"] for row in rows]),
                torch.cat(
                    [caches[row]["conformer_attention_cache"] for row in rows], dim=1
                ),
                len(self.encoder.encoders),
                self.encoder.up_layer.stride,
            )
            hidden_states, conformer_state = self.encoder.forward_chunk(
                xs=self.input_embedding(torch.cat([token_ids[row] for row in rows])),
                is_last_chunk=group_is_last_chunk,
                state=conformer_state,
            )
            group_convolution_cache, group_attention_cache = conformer_state.to_packed(
                self.encoder.up_layer.stride
            )
            hidden_states = self.encoder_proj(hidden_states)
            for position, row in enumerate(rows):
                hidden_states_by_row[row] = hidden_states[position : position + 1]
                # note (Junnan Li): Clones keep each stream's caches at their own size and free the group tensors.
                conformer_caches[row] = (
                    group_convolution_cache[position : position + 1].clone(),
                    group_attention_cache[:, position : position + 1].clone(),
                )
        hidden_states_by_stream = [
            hidden_states_by_row[row] for row in range(len(caches))
        ]
        frame_counts = [
            hidden_states.shape[1] for hidden_states in hidden_states_by_stream
        ]
        history_counts = [
            cache["estimator_attention_cache"].shape[4] for cache in caches
        ]
        stream_count = len(caches)
        graph = self.decoder.chunk_graph(
            stream_count, max(frame_counts), max(history_counts)
        )
        if graph is None:
            window_streams = stream_count
            padded_frames = max(frame_counts)
            padded_history = max(history_counts)
        else:
            window_streams = graph.stream_count
            padded_frames = graph.frame_count
            padded_history = graph.history_capacity
        # note (Junnan Li): Rows past the real streams only fill a graph's fixed batch; they see their own zero frames and no history.
        padding_streams = window_streams - stream_count
        if (
            graph is None
            and len(set(frame_counts)) == 1
            and len(set(history_counts)) == 1
        ):
            layout = None
        else:
            layout = RaggedLayout(
                frame_counts=frame_counts + [padded_frames] * padding_streams,
                history_counts=history_counts + [0] * padding_streams,
                padded_frames=padded_frames,
                padded_history=padded_history,
            )
        mu = (
            F.pad(
                torch.cat(
                    [
                        F.pad(hidden_states, (0, 0, 0, padded_frames - frame_count))
                        for hidden_states, frame_count in zip(
                            hidden_states_by_stream, frame_counts, strict=True
                        )
                    ]
                ),
                (0, 0, 0, 0, 0, padding_streams),
            )
            .transpose(1, 2)
            .contiguous()
        )
        speaker_embeddings = F.pad(speaker_embeddings, (0, 0, 0, padding_streams))
        convolution_halves = [
            cache["estimator_convolution_cache"].chunk(2, dim=ESTIMATOR_GUIDANCE_AXIS)
            for cache in caches
        ]
        padding_convolution = torch.zeros_like(convolution_halves[0][0])
        convolution_cache = torch.cat(
            [conditioned for conditioned, _ in convolution_halves]
            + [padding_convolution] * padding_streams
            + [unconditioned for _, unconditioned in convolution_halves]
            + [padding_convolution] * padding_streams,
            dim=ESTIMATOR_GUIDANCE_AXIS,
        )
        history_shape = caches[0]["estimator_attention_cache"].shape
        if graph is None:
            attention_window = torch.empty(
                history_shape[0],
                history_shape[1],
                2 * stream_count,
                history_shape[3],
                padded_frames + padded_history,
                history_shape[5],
                dtype=caches[0]["estimator_attention_cache"].dtype,
                device=mu.device,
            )
        else:
            attention_window = graph.attention_window
        for row, (cache, history_count) in enumerate(
            zip(caches, history_counts, strict=True)
        ):
            history = cache.pop("estimator_attention_cache")
            for guidance_row, window_row in enumerate((row, window_streams + row)):
                attention_window[
                    :, :, window_row, :, padded_frames : padded_frames + history_count
                ] = history[:, :, guidance_row]
                # note (Junnan Li): Masked keys still multiply their values, so padding must be finite.
                attention_window[
                    :, :, window_row, :, padded_frames + history_count :
                ].zero_()
        if graph is None:
            predicted_mel, estimator_convolution_cache, attention_window = (
                self.decoder.forward_chunk(
                    mu=mu,
                    speaker_embeddings=speaker_embeddings,
                    mel_conditioning=torch.zeros_like(mu),
                    n_timesteps=n_timesteps,
                    temperature=1.0,
                    convolution_cache=convolution_cache,
                    attention_window=attention_window,
                    layout=layout,
                )
            )
        else:
            assert layout is not None
            predicted_mel, estimator_convolution_cache = self.decoder.replay_chunk(
                graph, mu, speaker_embeddings, convolution_cache, layout
            )
        conditioned_convolution, unconditioned_convolution = (
            estimator_convolution_cache.chunk(2, dim=ESTIMATOR_GUIDANCE_AXIS)
        )
        results: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
        for row, (frame_count, history_count) in enumerate(
            zip(frame_counts, history_counts, strict=True)
        ):
            attention_cache = attention_window.new_empty(
                attention_window.shape[0],
                attention_window.shape[1],
                2,
                attention_window.shape[3],
                frame_count + history_count,
                attention_window.shape[5],
            )
            for guidance_row, window_row in enumerate((row, window_streams + row)):
                attention_cache[:, :, guidance_row, :, :frame_count] = attention_window[
                    :, :, window_row, :, :frame_count
                ]
                attention_cache[:, :, guidance_row, :, frame_count:] = attention_window[
                    :,
                    :,
                    window_row,
                    :,
                    padded_frames : padded_frames + history_count,
                ]
            results.append(
                (
                    predicted_mel[row : row + 1, :, :frame_count],
                    {
                        "conformer_convolution_cache": conformer_caches[row][0],
                        "conformer_attention_cache": conformer_caches[row][1],
                        "estimator_convolution_cache": torch.cat(
                            (
                                conditioned_convolution.narrow(
                                    ESTIMATOR_GUIDANCE_AXIS, row, 1
                                ),
                                unconditioned_convolution.narrow(
                                    ESTIMATOR_GUIDANCE_AXIS, row, 1
                                ),
                            ),
                            dim=ESTIMATOR_GUIDANCE_AXIS,
                        ),
                        "estimator_attention_cache": attention_cache,
                    },
                )
            )
        return results
