# SPDX-License-Identifier: Apache-2.0
"""Reusable causal prefix CUDA Graphs for Fun-CosyVoice3."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from sglang_omni.models.fun_cosyvoice3.packed_dit import (
    PackedDiT,
    PackedRows,
    pack_rows,
)
from sglang_omni.models.fun_cosyvoice3.prefix_cache import (
    CONV_CONTEXT_FRAMES,
    PrefixCacheRow,
    PrefixKVPool,
    PrefixRowAttention,
    forward_prefix,
    grow_rows,
    release_rows,
)

logger = logging.getLogger(__name__)

CAPTURE_WARMUP_ITERATIONS = 1
BYTES_PER_MIB = 1024 * 1024

PrefixCudaGraphCaptureShape = tuple[int, int, int, int, Sequence[int]]
PrefixCudaGraphCaptureInputs = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
PrefixCudaGraphCaptureInputFactory = Callable[
    [tuple[int, ...]],
    PrefixCudaGraphCaptureInputs,
]


def resolve_prefix_cuda_graph_max_slack(
    chunk_frames: int,
    configured_max_slack_frames: int | None,
) -> int:
    """Resolve and validate the prefix graph slack serving policy."""
    chunk_frames = int(chunk_frames)
    if chunk_frames <= 0:
        raise ValueError(f"PackedDiT chunk size must be positive, got {chunk_frames}")
    else:
        pass
    max_slack_frames = (
        2 * chunk_frames
        if configured_max_slack_frames is None
        else int(configured_max_slack_frames)
    )
    if max_slack_frames <= 0:
        raise ValueError(
            "flow_prefix_cuda_graph_max_slack_frames must be greater than zero; "
            f"got {max_slack_frames}"
        )
    elif max_slack_frames % chunk_frames != 0:
        raise ValueError(
            "flow_prefix_cuda_graph_max_slack_frames must be a multiple of "
            f"chunk size {chunk_frames}; got {max_slack_frames}"
        )
    else:
        return max_slack_frames


@dataclass(frozen=True)
class PrefixCudaGraphEnvelope:
    """One physical prefix CUDA Graph envelope."""

    name: str
    batch_size: int
    new_frame_count: int
    max_new_frame_count: int
    max_total_frame_count: int
    capture_row_new_frames: tuple[int, ...]


def prefix_cuda_graph_envelopes_from_capture_shapes(
    capture_shapes: Sequence[PrefixCudaGraphCaptureShape],
) -> tuple[PrefixCudaGraphEnvelope, ...]:
    """Convert serializable capture specs into runtime graph envelopes."""
    return tuple(
        PrefixCudaGraphEnvelope(
            name=(
                f"B{batch_size}-N{new_frame_count}-M{max_new_frame_count}"
                f"-E{max_total_frame_count}"
            ),
            batch_size=batch_size,
            new_frame_count=new_frame_count,
            max_new_frame_count=max_new_frame_count,
            max_total_frame_count=max_total_frame_count,
            capture_row_new_frames=tuple(capture_row_new_frames),
        )
        for (
            batch_size,
            new_frame_count,
            max_new_frame_count,
            max_total_frame_count,
            capture_row_new_frames,
        ) in capture_shapes
    )


@dataclass(frozen=True)
class PrefixCudaGraphCaptureStats:
    graph_count: int
    capture_s: float
    capture_order: tuple[str, ...]
    net_capture_allocated_delta_mib: float
    net_capture_reserved_delta_mib: float
    scratch_reserved_frames: int


@dataclass
class PreparedPrefixCudaGraph:
    noise: torch.Tensor
    time_span: torch.Tensor
    mu: torch.Tensor
    speaker_embeddings: torch.Tensor
    mel_conditioning: torch.Tensor
    twin_rows: PackedRows
    attention: PrefixRowAttention
    rope: tuple[torch.Tensor, torch.Tensor]
    context: torch.Tensor
    committed_frame_counts: list[int]
    real_frame_count: int


@dataclass
class StaticPrefixCudaGraph:
    graph: torch.cuda.CUDAGraph | None
    output: torch.Tensor | None
    next_context: torch.Tensor
    noise: torch.Tensor
    time_span: torch.Tensor
    mu: torch.Tensor
    speaker_embeddings: torch.Tensor
    mel_conditioning: torch.Tensor
    twin_rows: PackedRows
    attention: PrefixRowAttention
    rope: tuple[torch.Tensor, torch.Tensor]
    context: torch.Tensor
    flow_time: torch.Tensor


@dataclass
class CapturedPrefixCudaGraph:
    envelope: PrefixCudaGraphEnvelope
    static: StaticPrefixCudaGraph
    capture_s: float


def route_prefix_cuda_graph_envelope(
    *,
    new_frame_counts: Sequence[int],
    total_frame_counts: Sequence[int],
    envelopes: Sequence[PrefixCudaGraphEnvelope],
    chunk_frames: int,
    max_slack_frames: int,
) -> PrefixCudaGraphEnvelope | None:
    """Return the smallest physical envelope compatible with real row geometry."""
    if not new_frame_counts or len(new_frame_counts) != len(total_frame_counts):
        return None
    else:
        pass
    if not all(
        frame_count > 0 and frame_count % chunk_frames == 0
        for frame_count in new_frame_counts
    ):
        return None
    else:
        pass
    batch_size = len(new_frame_counts)
    total_new_frame_count = sum(new_frame_counts)
    max_new_frame_count = max(new_frame_counts)
    max_total_frame_count = max(total_frame_counts)
    candidates = [
        envelope
        for envelope in envelopes
        if envelope.batch_size == batch_size
        and 0 <= envelope.new_frame_count - total_new_frame_count <= max_slack_frames
        and (envelope.new_frame_count - total_new_frame_count) % chunk_frames == 0
        and max_new_frame_count <= envelope.max_new_frame_count
        and max_total_frame_count <= envelope.max_total_frame_count
    ]
    if not candidates:
        return None
    else:
        return min(
            candidates,
            key=lambda envelope: (
                envelope.new_frame_count,
                envelope.max_new_frame_count,
                envelope.max_total_frame_count,
                envelope.name,
            ),
        )


class PrefixCudaGraphCache:
    """Reusable causal prefix graphs with compiled-prefix miss fallback."""

    def __init__(
        self,
        estimator: PackedDiT,
        pool: PrefixKVPool,
        *,
        device: torch.device,
        autocast_dtype: torch.dtype | None,
        cfg_rate: float,
        envelopes: tuple[PrefixCudaGraphEnvelope, ...],
        max_slack_frames: int,
        capture_warmup_iterations: int = CAPTURE_WARMUP_ITERATIONS,
    ) -> None:
        self.estimator = estimator
        self.pool = pool
        self.device = torch.device(device)
        self.autocast_dtype = autocast_dtype
        self.chunk_frames = int(estimator.chunk_size)
        self.max_slack_frames = resolve_prefix_cuda_graph_max_slack(
            self.chunk_frames, max_slack_frames
        )
        self.cfg_rate = float(cfg_rate)
        self.envelopes = envelopes
        self.capture_warmup_iterations = int(capture_warmup_iterations)
        self.entries: list[CapturedPrefixCudaGraph] = []
        self.capture_order: list[str] = []
        self.net_capture_allocated_delta_mib = 0.0
        self.net_capture_reserved_delta_mib = 0.0
        self.capture_stream: torch.cuda.Stream = torch.cuda.Stream(device=self.device)

        # This isolated CFG twin pair supplies slack rows without aliasing a request.
        free_frames_before_scratch = int(self.pool.free_frames)
        scratch_pair = (PrefixCacheRow(), PrefixCacheRow())
        if not grow_rows(
            self.pool,
            list(scratch_pair),
            [self.max_slack_frames, self.max_slack_frames],
        ):
            raise RuntimeError("prefix pool cannot reserve graph scratch pair")
        else:
            self.scratch_pair: tuple[PrefixCacheRow, PrefixCacheRow] | None = (
                scratch_pair
            )
            self.scratch_reserved_frames = free_frames_before_scratch - int(
                self.pool.free_frames
            )

    def close(self) -> None:
        if self.scratch_pair is None:
            return
        else:
            release_rows(self.pool, list(self.scratch_pair))
            self.scratch_pair = None

    def capture_stats(self) -> PrefixCudaGraphCaptureStats:
        return PrefixCudaGraphCaptureStats(
            graph_count=len(self.entries),
            capture_s=sum(entry.capture_s for entry in self.entries),
            capture_order=tuple(self.capture_order),
            net_capture_allocated_delta_mib=self.net_capture_allocated_delta_mib,
            net_capture_reserved_delta_mib=self.net_capture_reserved_delta_mib,
            scratch_reserved_frames=self.scratch_reserved_frames,
        )

    @torch.inference_mode()
    def capture(
        self,
        capture_input_factory: PrefixCudaGraphCaptureInputFactory,
    ) -> None:
        if self.entries:
            raise RuntimeError("prefix CUDA Graph cache already captured")
        else:
            capture_envelopes = sorted(
                self.envelopes,
                key=lambda envelope: (
                    envelope.new_frame_count,
                    envelope.batch_size,
                    envelope.max_new_frame_count,
                    envelope.max_total_frame_count,
                ),
                reverse=True,
            )
            # Largest-first capture improves reuse of the single shared graph pool.
            self.capture_order = [envelope.name for envelope in capture_envelopes]
            with torch.cuda.device(self.device):
                graph_pool = torch.cuda.graph_pool_handle()
                torch.cuda.synchronize(self.device)
                allocated_before = torch.cuda.memory_allocated(self.device)
                reserved_before = torch.cuda.memory_reserved(self.device)
                for envelope in capture_envelopes:
                    self.entries.append(
                        self.capture_one(
                            envelope,
                            graph_pool,
                            capture_input_factory,
                        )
                    )
                torch.cuda.synchronize(self.device)
                self.net_capture_allocated_delta_mib = (
                    torch.cuda.memory_allocated(self.device) - allocated_before
                ) / BYTES_PER_MIB
                self.net_capture_reserved_delta_mib = (
                    torch.cuda.memory_reserved(self.device) - reserved_before
                ) / BYTES_PER_MIB
            stats = self.capture_stats()
            logger.info(
                f"Fun-CosyVoice3 causal prefix CUDA Graph cache: "
                f"graphs={stats.graph_count} capture_s={stats.capture_s:.1f} "
                f"net_allocated_delta_mib={stats.net_capture_allocated_delta_mib:.2f} "
                f"net_reserved_delta_mib={stats.net_capture_reserved_delta_mib:.2f} "
                f"scratch_reserved_frames={stats.scratch_reserved_frames}"
            )

    def allocate_capture_pairs(
        self, new_frame_counts: tuple[int, ...]
    ) -> list[tuple[PrefixCacheRow, PrefixCacheRow]]:
        pairs: list[tuple[PrefixCacheRow, PrefixCacheRow]] = []
        for total_new_frame_count in new_frame_counts:
            pair = (PrefixCacheRow(), PrefixCacheRow())
            if grow_rows(
                self.pool,
                list(pair),
                [total_new_frame_count, total_new_frame_count],
            ):
                pairs.append(pair)
            else:
                release_rows(
                    self.pool,
                    [cache_row for pair in pairs for cache_row in pair],
                )
                raise RuntimeError(
                    "prefix pool cannot allocate prefix CUDA Graph capture rows"
                )
        return pairs

    def capture_one(
        self,
        envelope: PrefixCudaGraphEnvelope,
        graph_pool: tuple[int, int],
        capture_input_factory: PrefixCudaGraphCaptureInputFactory,
    ) -> CapturedPrefixCudaGraph:
        real_pairs = self.allocate_capture_pairs(envelope.capture_row_new_frames)
        try:
            capture_inputs = capture_input_factory(envelope.capture_row_new_frames)
            prepared = self.prepare(
                *capture_inputs,
                list(envelope.capture_row_new_frames),
                real_pairs,
                envelope,
            )
            static = self.build_static(prepared, envelope)
            current_stream = torch.cuda.current_stream(self.device)
            capture_stream = self.capture_stream
            capture_stream.wait_stream(current_stream)
            started = time.perf_counter()
            with (
                torch.cuda.stream(capture_stream),
                torch.autocast(
                    device_type="cuda",
                    dtype=self.autocast_dtype,
                    enabled=self.autocast_dtype is not None,
                ),
            ):
                for _ in range(self.capture_warmup_iterations):
                    self.run_static_body(static)
                torch.cuda.synchronize(self.device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    cuda_graph=graph,
                    pool=graph_pool,
                    stream=capture_stream,
                    capture_error_mode="thread_local",
                ):
                    output = self.run_static_body(static)
            current_stream.wait_stream(capture_stream)
            torch.cuda.synchronize(self.device)
            static.graph = graph
            static.output = output
            return CapturedPrefixCudaGraph(
                envelope=envelope,
                static=static,
                capture_s=time.perf_counter() - started,
            )
        finally:
            release_rows(
                self.pool,
                [cache_row for pair in real_pairs for cache_row in pair],
            )
            self.reset_scratch()

    def reset_scratch(self) -> None:
        assert self.scratch_pair is not None
        for cache_row in self.scratch_pair:
            cache_row.committed_frames = 0
            cache_row.conv_context = None

    def route(
        self,
        new_frame_counts: Sequence[int],
        total_frame_counts: Sequence[int],
    ) -> CapturedPrefixCudaGraph | None:
        envelope = route_prefix_cuda_graph_envelope(
            new_frame_counts=new_frame_counts,
            total_frame_counts=total_frame_counts,
            envelopes=self.envelopes,
            chunk_frames=self.chunk_frames,
            max_slack_frames=self.max_slack_frames,
        )
        if envelope is None:
            return None
        else:
            for entry in self.entries:
                if entry.envelope != envelope:
                    continue
                else:
                    return entry
            return None

    def prepare(
        self,
        noise: torch.Tensor,
        time_span: torch.Tensor,
        mu: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        mel_conditioning: torch.Tensor,
        new_frame_counts: list[int],
        real_pairs: Sequence[tuple[PrefixCacheRow, PrefixCacheRow]],
        envelope: PrefixCudaGraphEnvelope,
    ) -> PreparedPrefixCudaGraph:
        assert self.scratch_pair is not None
        self.reset_scratch()
        real_frame_count = sum(new_frame_counts)
        slack_frame_count = envelope.new_frame_count - real_frame_count

        prefix_frame_counts = [pair[0].committed_frames for pair in real_pairs]
        for pair, prefix_frame_count in zip(
            real_pairs, prefix_frame_counts, strict=True
        ):
            if pair[1].committed_frames != prefix_frame_count:
                raise RuntimeError(
                    "CFG prefix cache twins have different committed frames"
                )
            else:
                pass

        def append_slack(tensor: torch.Tensor) -> torch.Tensor:
            if slack_frame_count == 0:
                return tensor
            else:
                return torch.cat(
                    (
                        tensor,
                        torch.zeros(
                            tensor.shape[0],
                            slack_frame_count,
                            tensor.shape[2],
                            device=tensor.device,
                            dtype=tensor.dtype,
                        ),
                    ),
                    dim=1,
                )

        all_pairs = list(real_pairs) + [self.scratch_pair]
        logical_new_frame_counts = new_frame_counts + [slack_frame_count]
        logical_prefix_frame_counts = prefix_frame_counts + [0]
        twin_caches = [pair[0] for pair in all_pairs] + [pair[1] for pair in all_pairs]
        twin_new_frame_counts = logical_new_frame_counts * 2
        twin_prefix_frame_counts = logical_prefix_frame_counts * 2
        device = noise.device
        twin_rows = pack_rows(twin_new_frame_counts, device)
        absolute_positions = (
            twin_rows.positions
            + torch.tensor(
                twin_prefix_frame_counts,
                device=device,
            )[twin_rows.row_ids]
        )
        attention = PrefixRowAttention(
            prefix_frames=twin_prefix_frame_counts,
            new_frames=twin_new_frame_counts,
            pages=[cache_row.pages(device) for cache_row in twin_caches],
            chunk_size=self.chunk_frames,
            device=device,
        )
        if self.pool.forward is not forward_prefix:
            attention.mark_dynamic(twin_rows, absolute_positions)
        else:
            pass
        max_end_frame = max(
            prefix_frame_count + new_frame_count
            for prefix_frame_count, new_frame_count in zip(
                twin_prefix_frame_counts,
                twin_new_frame_counts,
                strict=True,
            )
        )
        angles, scale = self.estimator.dit.rotary_embed.forward_from_seq_len(
            max_end_frame
        )
        if isinstance(scale, torch.Tensor):
            raise RuntimeError(
                "prefix CUDA Graph cache does not support RoPE xpos scale"
            )
        else:
            pass
        angles = angles[:, absolute_positions]
        rope = (angles.cos(), angles.sin())
        hidden_size = int(self.estimator.dit.input_embed.proj.out_features)
        euler_step_count = len(time_span) - 1
        contexts: list[torch.Tensor] = []
        for cache_row in twin_caches:
            if cache_row.conv_context is None:
                contexts.append(
                    torch.zeros(
                        euler_step_count,
                        2,
                        CONV_CONTEXT_FRAMES,
                        hidden_size,
                        device=device,
                        dtype=speaker_embeddings.dtype,
                    )
                )
            else:
                contexts.append(cache_row.conv_context)
        context = torch.stack(contexts, dim=1)
        speaker_embeddings_with_scratch = torch.cat(
            (
                speaker_embeddings,
                torch.zeros(
                    1,
                    speaker_embeddings.shape[1],
                    device=device,
                    dtype=speaker_embeddings.dtype,
                ),
            ),
            dim=0,
        )
        return PreparedPrefixCudaGraph(
            noise=append_slack(noise),
            time_span=time_span,
            mu=append_slack(mu),
            speaker_embeddings=speaker_embeddings_with_scratch,
            mel_conditioning=append_slack(mel_conditioning),
            twin_rows=twin_rows,
            attention=attention,
            rope=rope,
            context=context,
            committed_frame_counts=list(attention.committed_frames),
            real_frame_count=real_frame_count,
        )

    def build_static(
        self,
        prepared: PreparedPrefixCudaGraph,
        envelope: PrefixCudaGraphEnvelope,
    ) -> StaticPrefixCudaGraph:
        static_rows = pack_rows(prepared.twin_rows.lengths, self.device)
        static_attention = PrefixRowAttention(
            prefix_frames=[0] * len(prepared.twin_rows.lengths),
            new_frames=list(prepared.twin_rows.lengths),
            pages=[
                torch.arange(
                    envelope.max_total_frame_count,
                    dtype=torch.int32,
                    device=self.device,
                )
                for _ in prepared.twin_rows.lengths
            ],
            chunk_size=self.chunk_frames,
            device=self.device,
        )
        static_attention.page_table = torch.zeros(
            prepared.attention.page_table.shape[0],
            envelope.max_total_frame_count,
            dtype=prepared.attention.page_table.dtype,
            device=self.device,
        )
        static_attention.slots = torch.full(
            (prepared.attention.slots.shape[0], envelope.max_new_frame_count),
            -1,
            dtype=prepared.attention.slots.dtype,
            device=self.device,
        )
        static = StaticPrefixCudaGraph(
            graph=None,
            output=None,
            next_context=torch.empty_like(prepared.context),
            noise=prepared.noise.clone(),
            time_span=prepared.time_span.clone(),
            mu=prepared.mu.clone(),
            speaker_embeddings=prepared.speaker_embeddings.clone(),
            mel_conditioning=prepared.mel_conditioning.clone(),
            twin_rows=static_rows,
            attention=static_attention,
            rope=(prepared.rope[0].clone(), prepared.rope[1].clone()),
            context=prepared.context.clone(),
            flow_time=torch.zeros(
                1,
                device=self.device,
                dtype=prepared.speaker_embeddings.dtype,
            ),
        )
        self.stage(static, prepared)
        return static

    def stage(
        self, static: StaticPrefixCudaGraph, prepared: PreparedPrefixCudaGraph
    ) -> None:
        if (
            prepared.attention.page_table.shape[1]
            > static.attention.page_table.shape[1]
        ):
            raise RuntimeError("prefix CUDA Graph page table width exceeded envelope E")
        elif prepared.attention.slots.shape[1] > static.attention.slots.shape[1]:
            raise RuntimeError("prefix CUDA Graph slots width exceeded envelope M")
        elif static.attention.max_seqlen_q != prepared.attention.max_seqlen_q:
            raise RuntimeError("prefix CUDA Graph query segment width changed")
        else:
            static.noise.copy_(prepared.noise)
            static.time_span.copy_(prepared.time_span)
            static.mu.copy_(prepared.mu)
            static.speaker_embeddings.copy_(prepared.speaker_embeddings)
            static.mel_conditioning.copy_(prepared.mel_conditioning)
            static.twin_rows.row_ids.copy_(prepared.twin_rows.row_ids)
            static.twin_rows.positions.copy_(prepared.twin_rows.positions)
            static.attention.cache_seqlens.copy_(prepared.attention.cache_seqlens)
            static.attention.cu_seqlens_q.copy_(prepared.attention.cu_seqlens_q)
            static.attention.write_index.copy_(prepared.attention.write_index)
            static.attention.slots.fill_(-1)
            static.attention.slots[:, : prepared.attention.slots.shape[1]].copy_(
                prepared.attention.slots
            )
            static.attention.tail_index.copy_(prepared.attention.tail_index)
            static.attention.page_table.zero_()
            static.attention.page_table[
                :, : prepared.attention.page_table.shape[1]
            ].copy_(prepared.attention.page_table)
            static.rope[0].copy_(prepared.rope[0])
            static.rope[1].copy_(prepared.rope[1])
            static.context.copy_(prepared.context)

    def run_static_body(self, static: StaticPrefixCudaGraph) -> torch.Tensor:
        total_new_frame_count = static.noise.shape[1]
        mu_cfg = torch.cat((static.mu, torch.zeros_like(static.mu)), dim=1)
        mel_conditioning_cfg = torch.cat(
            (static.mel_conditioning, torch.zeros_like(static.mel_conditioning)), dim=1
        )
        speaker_embeddings_cfg = torch.cat(
            (
                static.speaker_embeddings,
                torch.zeros_like(static.speaker_embeddings),
            ),
            dim=0,
        )
        speaker_embeddings_cfg = speaker_embeddings_cfg[
            static.twin_rows.row_ids
        ].unsqueeze(0)
        x = static.noise
        t = static.time_span[0]
        dt = static.time_span[1] - static.time_span[0]
        for euler_step in range(len(static.time_span) - 1):
            static.flow_time[:] = t
            vector_field, first_tail, second_tail = self.pool.forward(
                self.estimator,
                self.pool.keys[euler_step],
                self.pool.values[euler_step],
                torch.cat((x, x), dim=1),
                mu_cfg,
                speaker_embeddings_cfg,
                mel_conditioning_cfg,
                static.flow_time,
                static.twin_rows,
                static.attention,
                static.rope,
                static.context[euler_step, :, 0],
                static.context[euler_step, :, 1],
            )
            static.next_context[euler_step, :, 0].copy_(first_tail)
            static.next_context[euler_step, :, 1].copy_(second_tail)
            conditional = vector_field[:, :total_new_frame_count]
            unconditional = vector_field[:, total_new_frame_count:]
            x = x + dt * (
                (1.0 + self.cfg_rate) * conditional - self.cfg_rate * unconditional
            )
            t = t + dt
            if euler_step < len(static.time_span) - 2:
                dt = static.time_span[euler_step + 2] - t
            else:
                pass
        return x.float()

    def materialize_output(
        self, static: StaticPrefixCudaGraph, real_frame_count: int
    ) -> torch.Tensor:
        assert static.output is not None
        # Shared graph-pool workspace may be reused by the next graph replay.
        return static.output[:, :real_frame_count].clone()

    def commit_real(
        self,
        static: StaticPrefixCudaGraph,
        prepared: PreparedPrefixCudaGraph,
        real_pairs: Sequence[tuple[PrefixCacheRow, PrefixCacheRow]],
    ) -> None:
        logical_row_count = len(real_pairs) + 1
        for index, pair in enumerate(real_pairs):
            conditional_index = index
            unconditional_index = logical_row_count + index
            pair[0].committed_frames = int(
                prepared.committed_frame_counts[conditional_index]
            )
            pair[1].committed_frames = int(
                prepared.committed_frame_counts[unconditional_index]
            )
            # Request rows must not retain graph-private next-context storage.
            pair[0].conv_context = static.next_context[:, conditional_index].clone()
            pair[1].conv_context = static.next_context[:, unconditional_index].clone()
        self.reset_scratch()

    @torch.inference_mode()
    def run(
        self,
        *,
        noise: torch.Tensor,
        time_span: torch.Tensor,
        mu: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        mel_conditioning: torch.Tensor,
        new_frames: list[int],
        total_frames: list[int],
        caches: Sequence[tuple[PrefixCacheRow, PrefixCacheRow]],
    ) -> torch.Tensor | None:
        entry = self.route(new_frames, total_frames)
        if entry is None:
            # Misses are expected; replay and staging failures intentionally escape.
            return None
        else:
            with (
                torch.cuda.device(self.device),
                torch.autocast(
                    device_type="cuda",
                    dtype=self.autocast_dtype,
                    enabled=self.autocast_dtype is not None,
                ),
            ):
                prepared = self.prepare(
                    noise,
                    time_span,
                    mu,
                    speaker_embeddings,
                    mel_conditioning,
                    new_frames,
                    caches,
                    entry.envelope,
                )
                static = entry.static
                self.stage(static, prepared)
                graph = static.graph
                assert graph is not None
                graph.replay()
                generated = self.materialize_output(static, prepared.real_frame_count)
                self.commit_real(static, prepared, caches)
            return generated
