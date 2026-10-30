# SPDX-License-Identifier: Apache-2.0
"""Slot-owned streaming state and device-graph workspaces for gesture inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang_omni.models.miburi.architecture import (
    INITIAL_CODE,
    NULL_CONDITION_ID,
    SPEECH_AUDIO_CODEBOOKS,
)
from sglang_omni.models.miburi.loading import GestureRuntime
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import DeviceGraphBackend, ReplayableGraph

# Eager frames before a batch size's capture, so lazy library handles and workspaces are created outside it.
GRAPH_WARMUP_STEPS = 2


@dataclass(frozen=True)
class SlotTensor:
    """A state tensor with one row per session on axis 0, zero for a fresh session; a guided one also holds each session's unconditional twin, one block of rows further on."""

    tensor: torch.Tensor
    guided: bool


class GestureWorkspace:
    """The LM and codec state of batch_size sessions at fixed addresses."""

    def __init__(self, gesture: GestureRuntime, batch_size: int, guided: bool) -> None:
        self.temporal = gesture.model.temporal_transformer.new_state(
            batch_size * (2 if guided else 1), gesture.device
        )
        self.codecs = tuple(
            codec.new_state(batch_size, gesture.device) for codec in gesture.codecs
        )
        self.tensors = [
            SlotTensor(tensor, guided) for tensor in self.temporal.row_tensors()
        ] + [
            SlotTensor(tensor, False)
            for state in self.codecs
            for tensor in state.row_tensors()
        ]
        self.previous_codes = torch.full(
            (batch_size, gesture.model.spec.num_codebooks, 1),
            INITIAL_CODE,
            dtype=torch.long,
            device=gesture.device,
        )


@dataclass(kw_only=True)
class CapturedFrame:
    graph: ReplayableGraph
    inputs: tuple[torch.Tensor, ...]
    outputs: tuple[torch.Tensor, ...]


class GestureBatch:
    """Gather active slots into the workspace of their count, run one frame, and scatter only those slots back."""

    def __init__(self, gesture: GestureRuntime, max_open_sessions: int) -> None:
        self.gesture = gesture
        self.max_open_sessions = max_open_sessions
        self.guided = gesture.sampling.cfg_coef != 1.0
        self.pool = GestureWorkspace(gesture, max_open_sessions, self.guided)
        self.workspaces: dict[int, GestureWorkspace] = {}
        self.graphs: dict[int, CapturedFrame] = {}
        self.random_generators: dict[int, torch.Generator] = {}
        self.free_slots = list(reversed(range(max_open_sessions)))
        self.speaker = torch.tensor([[gesture.speaker_id]], device=gesture.device)

    def open(self, random_generator: torch.Generator) -> int:
        slot = self.free_slots.pop()
        self.random_generators[slot] = random_generator
        for binding in self.pool.tensors:
            binding.tensor[slot].zero_()
            if binding.guided:
                binding.tensor[slot + self.max_open_sessions].zero_()
            else:
                pass
        self.pool.previous_codes[slot].fill_(INITIAL_CODE)
        return slot

    def close(self, slot: int) -> None:
        self.random_generators.pop(slot)
        self.free_slots.append(slot)

    def sample(
        self, logits: torch.Tensor, codebook: int, noise: torch.Tensor
    ) -> torch.Tensor:
        """Guided logits [rows, 1, codebook_size + 1] → one code per session: greedy at temperature 0, else top-p by an exponential race on the session's noise."""
        sampling = self.gesture.sampling
        spec = self.gesture.model.spec
        temperature, top_p = (
            (sampling.temporal_temperature, sampling.temporal_top_p)
            if codebook == 0
            else (sampling.depth_temperature, sampling.depth_top_p)
        )
        if self.guided:
            conditional, unconditional = logits.chunk(2)
            logits = (
                conditional
                if codebook in spec.lower_body_codebooks
                else unconditional + sampling.cfg_coef * (conditional - unconditional)
            )
        else:
            pass
        logits = logits[:, 0]
        if codebook > 0:
            # Note (Junnan Li): The released model leaves the pad code samplable on codebook 0 only.
            logits[:, spec.pad_code] = -torch.inf
        else:
            pass
        if temperature == 0:
            return logits.argmax(dim=-1)
        else:
            probabilities, indices = torch.softmax(logits / temperature, dim=-1).sort(
                dim=-1, descending=True
            )
            excluded = probabilities.cumsum(dim=-1) - probabilities > top_p
            probabilities = probabilities * (~excluded).float()
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
            selected = (probabilities / noise[:, codebook]).argmax(dim=-1, keepdim=True)
            return indices.gather(-1, selected)[:, 0]

    def forward(
        self,
        workspace: GestureWorkspace,
        condition: torch.Tensor,
        noise: torch.Tensor,
        idle_mask: torch.Tensor,
        idle_upper: torch.Tensor,
        idle_face: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One frame of the workspace's sessions: their 20 codes, then the upper, lower and face motion; the idle mask swaps in the idle codes for upper body and face."""
        gesture = self.gesture
        model = gesture.model
        spec = model.spec
        rows_per_session = 2 if self.guided else 1
        if self.guided:
            condition = torch.cat(
                (condition, torch.full_like(condition, NULL_CONDITION_ID))
            )
        else:
            pass
        speakers = self.speaker.expand(condition.shape[0], 1)
        memories = model.speech_memories(condition)
        hidden, logits = model.temporal_step(
            workspace.previous_codes.repeat(rows_per_session, 1, 1),
            memories,
            speakers,
            workspace.temporal,
        )
        codes = [self.sample(logits, 0, noise)]
        rings = model.depth_transformer.new_rings(condition.shape[0], gesture.device)
        for step in range(spec.depth_steps):
            logits = model.depth_step(
                step,
                codes[-1][:, None].repeat(rows_per_session, 1),
                hidden,
                memories,
                speakers,
                rings,
            )
            codes.append(self.sample(logits, step + 1, noise))
        sampled = torch.stack(codes, dim=1).unsqueeze(-1)
        workspace.previous_codes.copy_(sampled)
        upper_end = spec.codebooks_per_part[0]
        lower_end = upper_end + spec.codebooks_per_part[1]
        part_codes = (
            torch.where(idle_mask, idle_upper, sampled[:, :upper_end]),
            sampled[:, upper_end:lower_end],
            torch.where(idle_mask, idle_face, sampled[:, lower_end:]),
        )
        upper, lower, face = (
            codec.decode_step(part, state)
            for codec, part, state in zip(
                gesture.codecs, part_codes, workspace.codecs, strict=True
            )
        )
        return sampled, upper, lower, face

    @torch.no_grad()
    def step(
        self,
        slots: list[int],
        condition: torch.Tensor,
        idle_mask: torch.Tensor,
        idle_upper: torch.Tensor,
        idle_face: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One frame of the given slots; with a captured graph the outputs are its buffers, valid until the next step."""
        batch_size = len(slots)
        sampling = self.gesture.sampling
        spec = self.gesture.model.spec
        workspace = self.workspaces[batch_size]
        indices = torch.tensor(slots, device=self.gesture.device)
        guided_indices = torch.cat((indices, indices + self.max_open_sessions))
        for source, target in zip(self.pool.tensors, workspace.tensors, strict=True):
            torch.index_select(
                source.tensor,
                0,
                guided_indices if source.guided else indices,
                out=target.tensor,
            )
        torch.index_select(
            self.pool.previous_codes, 0, indices, out=workspace.previous_codes
        )
        noise = torch.ones(
            batch_size,
            spec.num_codebooks,
            spec.codebook_size + 1,
            device=self.gesture.device,
        )
        for row, slot in enumerate(slots):
            for codebook in range(spec.num_codebooks):
                if (
                    sampling.temporal_temperature
                    if codebook == 0
                    else sampling.depth_temperature
                ) > 0:
                    noise[row, codebook].view(1, -1).exponential_(
                        1, generator=self.random_generators[slot]
                    )
                else:
                    pass
        inputs = (condition, noise, idle_mask, idle_upper, idle_face)
        captured = self.graphs.get(batch_size)
        if captured is None:
            outputs = self.forward(workspace, *inputs)
        else:
            for static, value in zip(captured.inputs, inputs, strict=True):
                static.copy_(value)
            captured.graph.replay()
            outputs = captured.outputs
        for target, source in zip(self.pool.tensors, workspace.tensors, strict=True):
            target.tensor.index_copy_(
                0, guided_indices if target.guided else indices, source.tensor
            )
        self.pool.previous_codes.index_copy_(0, indices, workspace.previous_codes)
        return outputs

    def frame_inputs(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """Inputs of a placeholder frame: zero ids and codes, idle on, unit noise."""
        spec = self.gesture.model.spec
        device = self.gesture.device
        return (
            torch.zeros(
                batch_size,
                1 + SPEECH_AUDIO_CODEBOOKS,
                1,
                dtype=torch.long,
                device=device,
            ),
            torch.ones(
                batch_size, spec.num_codebooks, spec.codebook_size + 1, device=device
            ),
            torch.ones(batch_size, 1, 1, dtype=torch.bool, device=device),
            torch.zeros(
                batch_size,
                spec.codebooks_per_part[0],
                1,
                dtype=torch.long,
                device=device,
            ),
            torch.zeros(
                batch_size,
                spec.codebooks_per_part[2],
                1,
                dtype=torch.long,
                device=device,
            ),
        )

    @torch.no_grad()
    def warm_up(self) -> None:
        """A workspace per admitted batch size, each run once; where the device records graphs, one captured frame per batch size."""
        device = self.gesture.device
        for batch_size in range(1, self.max_open_sessions + 1):
            self.workspaces[batch_size] = GestureWorkspace(
                self.gesture, batch_size, self.guided
            )
        graph_backend = current_platform.get_device_graph_backend(device)
        if graph_backend is None:
            for batch_size, workspace in self.workspaces.items():
                self.forward(workspace, *self.frame_inputs(batch_size))
        else:
            self.capture_graphs(graph_backend)

    def capture_graphs(self, graph_backend: DeviceGraphBackend) -> None:
        device = self.gesture.device
        device_module = torch.get_device_module(device)
        capture_stream = device_module.Stream(device=device)
        graph_pool = graph_backend.graph_pool_handle()
        with device_module.device(device), current_platform.graph_capture_attention():
            capture_stream.wait_stream(device_module.current_stream(device))
            # Note (Junnan Li): Largest batch first, so smaller graphs reuse its blocks from the shared pool.
            for batch_size in range(self.max_open_sessions, 0, -1):
                workspace = self.workspaces[batch_size]
                with device_module.stream(capture_stream):
                    inputs = self.frame_inputs(batch_size)
                    for _ in range(GRAPH_WARMUP_STEPS):
                        self.forward(workspace, *inputs)
                with graph_backend.capture(
                    pool=graph_pool, stream=capture_stream, thread_local_errors=True
                ) as graph:
                    outputs = self.forward(workspace, *inputs)
                self.graphs[batch_size] = CapturedFrame(
                    graph=graph, inputs=inputs, outputs=outputs
                )
            device_module.current_stream(device).wait_stream(capture_stream)
            device_module.synchronize(device)


__all__ = ["GestureBatch", "GestureWorkspace"]
