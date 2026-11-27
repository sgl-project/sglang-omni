# SPDX-License-Identifier: Apache-2.0
"""Greedy RNN-T decoding in fixed-shape steps that replay as CUDA graphs.

One step scores ``SCAN_FRAMES`` frames against the current predictor output.
A blank leaves the predictor unchanged, so the first non-blank label among
those frames is the label that frame-by-frame greedy search emits next, and
the frames before it are blanks. Up to ``max_symbols_per_step`` labels are
emitted on one frame. As in ``decode_streaming_batch``, the predictor reads
an emitted label at the start of the next step, so both decoders keep a
stream's decoder cache in the same form. The token and duration history of
``decode_streaming_batch`` is rebuilt from the emitted labels and their frames.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from sglang_omni.models.nemotron3_5_asr.decoder import Nemotron3_5ASRDecodeState
from sglang_omni.models.nemotron3_5_asr.rnnt_kernels import (
    HAS_TRITON,
    advance_rnnt_triton,
    greedy_joint_bf16,
    lstm_gate_update_triton,
)
from sglang_omni.vendor.nemotron3_5_asr.modeling_nemotron3_5_asr import (
    Nemotron3_5AsrForRNNT,
)

SCAN_FRAMES = 8
# Decoding steps per graph replay; short streaming chunks use a smaller block
# to avoid wasted no-op steps once all rows finish.
SHORT_STEPS_PER_BLOCK = 4
STEPS_PER_BLOCK = 16
BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)
FRAME_BUCKETS = (8, 16, 32, 64, 128, 256, 512)
MAX_GRAPHS = 36


class DecodeLoop:
    """Static buffers and steps for up to ``batch`` rows of up to ``steps`` frames."""

    def __init__(
        self,
        model: Nemotron3_5AsrForRNNT,
        batch: int,
        steps: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        joint_weight_bf16: torch.Tensor | None = None,
        joint_bias_bf16: torch.Tensor | None = None,
        pred_weights_bf16: tuple[torch.Tensor, ...] | None = None,
    ):
        config = model.config
        self.model = model
        self.blank = int(config.blank_token_id)
        self.max_symbols = int(model.max_symbols_per_step)
        self.batch, self.steps = batch, steps
        self.steps_per_block = SHORT_STEPS_PER_BLOCK if steps <= 16 else STEPS_PER_BLOCK
        self.capacity = steps * self.max_symbols
        width, layers = config.decoder_hidden_size, config.num_decoder_layers
        decoder_dtype = model.decoder.embedding.weight.dtype
        self.decoder_dtype = decoder_dtype
        self.use_triton_advance = HAS_TRITON and device.type == "cuda"
        self.use_bf16_joint = (
            HAS_TRITON
            and device.type == "cuda"
            and dtype == torch.bfloat16
            and joint_weight_bf16 is not None
            and joint_bias_bf16 is not None
        )
        self.joint_weight_bf16 = joint_weight_bf16
        self.joint_bias_bf16 = joint_bias_bf16
        self.pred_weights_bf16 = (
            pred_weights_bf16
            if (HAS_TRITON and device.type == "cuda" and dtype == torch.bfloat16)
            else None
        )
        long = dict(device=device, dtype=torch.long)
        self.frames = torch.zeros(batch, steps, width, device=device, dtype=dtype)
        self.hidden = torch.zeros(
            layers, batch, width, device=device, dtype=decoder_dtype
        )
        self.cell = torch.zeros_like(self.hidden)
        self.output = torch.zeros(batch, width, device=device, dtype=dtype)
        if self.pred_weights_bf16 is not None:
            bf16 = dict(device=device, dtype=torch.bfloat16)
            self.h_bf16 = torch.zeros(2, batch, width, **bf16)
            self.x_buf = torch.empty(batch, 2 * width, **bf16)
            self.g_buf = torch.empty(batch, 4 * width, **bf16)
            self.proj_out_buf = torch.empty(batch, width, **bf16)
        else:
            pass
        self.lengths = torch.zeros(batch, **long)
        self.last = torch.zeros(batch, **long)
        self.pending = torch.zeros(batch, device=device, dtype=torch.bool)
        self.pending_label = torch.zeros(batch, **long)
        self.labels = torch.zeros(batch, self.capacity, **long)
        self.stamps = torch.zeros(batch, self.capacity, **long)
        self.counts = torch.zeros(batch, **long)
        self.time = torch.zeros(batch, **long)
        self.last_frame = torch.zeros(batch, **long)
        self.run = torch.zeros(batch, **long)
        self.active = torch.zeros(batch, device=device, dtype=torch.bool)
        self.rows = torch.arange(batch, **long)[:, None]
        self.offsets = torch.arange(SCAN_FRAMES, **long)
        self.graph: torch.cuda.CUDAGraph | None = None

    def load(
        self,
        frames: torch.Tensor,
        lengths: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        output: torch.Tensor,
        pending: torch.Tensor,
        pending_label: torch.Tensor,
    ) -> None:
        rows, steps = frames.shape[:2]
        # Frames past a row's length are never read, so stale values can stay.
        self.frames[:rows, :steps].copy_(frames)
        self.lengths.zero_()
        self.lengths[:rows].copy_(lengths)
        torch.clamp_min(self.lengths - 1, 0, out=self.last)
        self.hidden.zero_()
        self.hidden[:, :rows].copy_(hidden)
        self.cell.zero_()
        self.cell[:, :rows].copy_(cell)
        if self.pred_weights_bf16 is not None:
            self.h_bf16.copy_(self.hidden)
        else:
            pass
        self.output.zero_()
        self.output[:rows].copy_(output)
        self.pending.zero_()
        self.pending[:rows].copy_(pending)
        self.pending_label.fill_(self.blank)
        self.pending_label[:rows].copy_(pending_label)
        for buffer in (self.counts, self.time, self.run):
            buffer.zero_()
        self.last_frame.fill_(-1)
        torch.gt(self.lengths, 0, out=self.active)

    def predict(self, labels: torch.Tensor):
        decoder = self.model.decoder
        output, (hidden, cell) = decoder.lstm(
            decoder.embedding(labels[:, None]), (self.hidden, self.cell)
        )
        projected = decoder.decoder_projector(output)[:, 0]
        if projected.dtype != self.output.dtype:
            projected = projected.to(self.output.dtype)
        else:
            pass
        return projected, hidden, cell

    def step(self) -> None:
        # The predictor reads the label emitted by the previous step.
        apply = self.pending & self.active
        if self.pred_weights_bf16 is not None:
            emb_w, w0, b0, w1, b1, pw, pb = self.pred_weights_bf16
            width = self.h_bf16.shape[2]
            torch.index_select(emb_w, 0, self.pending_label, out=self.x_buf[:, :width])
            self.x_buf[:, width:].copy_(self.h_bf16[0])
            torch.addmm(b0, self.x_buf, w0.t(), out=self.g_buf)
            lstm_gate_update_triton(
                self.g_buf, self.cell[0], self.hidden[0], self.h_bf16[0], apply
            )
            self.x_buf[:, :width].copy_(self.h_bf16[0])
            self.x_buf[:, width:].copy_(self.h_bf16[1])
            torch.addmm(b1, self.x_buf, w1.t(), out=self.g_buf)
            lstm_gate_update_triton(
                self.g_buf, self.cell[1], self.hidden[1], self.h_bf16[1], apply
            )
            torch.addmm(pb, self.h_bf16[1], pw.t(), out=self.proj_out_buf)
            self.output.copy_(
                torch.where(apply[:, None], self.proj_out_buf, self.output)
            )
        else:
            output, hidden, cell = self.predict(self.pending_label)
            self.hidden.copy_(torch.where(apply[None, :, None], hidden, self.hidden))
            self.cell.copy_(torch.where(apply[None, :, None], cell, self.cell))
            self.output.copy_(torch.where(apply[:, None], output, self.output))
        self.pending.logical_and_(~apply)

        if self.use_bf16_joint:
            assert (
                self.joint_weight_bf16 is not None and self.joint_bias_bf16 is not None
            )
            scan = greedy_joint_bf16(
                self.frames,
                self.output,
                self.time,
                self.lengths,
                self.joint_weight_bf16,
                self.joint_bias_bf16,
                SCAN_FRAMES,
            )
        else:
            window = self.time[:, None] + self.offsets
            frames = self.frames[self.rows, torch.minimum(window, self.last[:, None])]
            logits = self.model.joint(
                decoder_hidden_states=self.output[:, None, :],
                encoder_hidden_states=frames,
            )
            scan = logits.argmax(dim=-1)

        if self.use_triton_advance:
            advance_rnnt_triton(
                scan,
                self.time,
                self.lengths,
                self.active,
                self.counts,
                self.labels,
                self.stamps,
                self.run,
                self.last_frame,
                self.pending,
                self.pending_label,
                capacity=self.capacity,
                blank=self.blank,
                max_symbols=self.max_symbols,
                steps=SCAN_FRAMES,
            )
        else:
            window = self.time[:, None] + self.offsets
            valid = (window < self.lengths[:, None]) & self.active[:, None]
            first = torch.where(
                valid & (scan != self.blank), self.offsets, SCAN_FRAMES
            ).amin(-1)
            emit = first < SCAN_FRAMES
            label = scan.gather(1, first.clamp_max(SCAN_FRAMES - 1)[:, None])[:, 0]
            time = torch.where(emit, self.time + first, self.time + SCAN_FRAMES)
            slot = self.counts.clamp_max(self.capacity - 1)[:, None]
            self.labels.scatter_(
                1,
                slot,
                torch.where(emit, label, self.labels.gather(1, slot)[:, 0])[:, None],
            )
            self.stamps.scatter_(
                1,
                slot,
                torch.where(emit, time, self.stamps.gather(1, slot)[:, 0])[:, None],
            )
            self.counts.add_(emit)
            run = torch.where(
                emit, torch.where(time == self.last_frame, self.run + 1, 1), self.run
            )
            self.last_frame.copy_(torch.where(emit, time, self.last_frame))
            self.pending.logical_or_(emit)
            self.pending_label.copy_(torch.where(emit, label, self.pending_label))
            # Too many labels on one frame: move on without a blank.
            forced = emit & (run >= self.max_symbols)
            self.time.copy_(time + forced)
            self.run.copy_(torch.where(forced, 0, run))
            torch.lt(self.time, self.lengths, out=self.active)

    def block(self) -> None:
        for _ in range(self.steps_per_block):
            self.step()

    def capture(self, stream: torch.cuda.Stream, pool) -> None:
        stream.wait_stream(torch.cuda.current_stream(stream.device))
        with torch.cuda.stream(stream):
            self.block()  # set up cuDNN and cuBLAS outside the capture
        torch.cuda.current_stream(stream.device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream, pool=pool):
            self.block()

    def run_to_end(self) -> None:
        while bool(self.active.any()):
            if self.graph is None:
                self.block()
            else:
                self.graph.replay()


class GreedyRNNTDecoder:
    """Batched greedy RNN-T search for one serial model runner."""

    def __init__(self, model: Nemotron3_5AsrForRNNT, *, use_cuda_graphs: bool):
        self.model = model
        self.blank = int(model.config.blank_token_id)
        self.max_symbols = int(model.max_symbols_per_step)
        self.use_cuda_graphs = use_cuda_graphs
        self.loops: dict[tuple, DecodeLoop] = {}
        self.stream: torch.cuda.Stream | None = None
        self.pool = None
        self.joint_bf16: tuple[torch.Tensor, torch.Tensor] | None = None
        self.pred_bf16: tuple[torch.Tensor, ...] | None = None
        self.meta_host: torch.Tensor | None = None
        self.zeros_cache: dict[
            tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def close(self) -> None:
        self.loops.clear()
        self.zeros_cache.clear()
        self.stream = None
        self.pool = None
        self.joint_bf16 = None
        self.pred_bf16 = None
        self.meta_host = None

    def get_pred_bf16(self, device: torch.device) -> tuple[torch.Tensor, ...] | None:
        if not HAS_TRITON or device.type != "cuda":
            return None
        elif self.pred_bf16 is not None:
            return self.pred_bf16
        else:
            pass
        dec = self.model.decoder
        lstm, emb, proj = (
            getattr(dec, "lstm", None),
            getattr(dec, "embedding", None),
            getattr(dec, "decoder_projector", None),
        )
        if not (
            isinstance(lstm, torch.nn.LSTM)
            and lstm.num_layers == 2
            and lstm.input_size == lstm.hidden_size
            and lstm.bias
            and not lstm.bidirectional
            and isinstance(emb, torch.nn.Embedding)
            and isinstance(proj, torch.nn.Linear)
            and proj.in_features == proj.out_features == lstm.hidden_size
        ):
            return None
        else:
            pass

        def to_bf16(t: torch.Tensor) -> torch.Tensor:
            return t.detach().to(device=device, dtype=torch.bfloat16).contiguous()

        with torch.inference_mode(False):
            layer_wb = [
                (
                    to_bf16(
                        torch.cat(
                            [
                                getattr(lstm, f"weight_ih_l{i}"),
                                getattr(lstm, f"weight_hh_l{i}"),
                            ],
                            dim=1,
                        )
                    ),
                    to_bf16(
                        getattr(lstm, f"bias_ih_l{i}") + getattr(lstm, f"bias_hh_l{i}")
                    ),
                )
                for i in (0, 1)
            ]
            pb = (
                to_bf16(proj.bias)
                if proj.bias is not None
                else torch.zeros(lstm.hidden_size, device=device, dtype=torch.bfloat16)
            )
            self.pred_bf16 = (
                to_bf16(emb.weight),
                *layer_wb[0],
                *layer_wb[1],
                to_bf16(proj.weight),
                pb,
            )
        return self.pred_bf16

    def get_joint_bf16(
        self, device: torch.device
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not HAS_TRITON or device.type != "cuda":
            return None, None
        elif self.joint_bf16 is not None:
            return self.joint_bf16
        else:
            pass
        head = self.model.joint.head
        if (
            isinstance(head, torch.nn.Linear)
            and head.bias is not None
            and self.model.config.hidden_act == "relu"
        ):
            with torch.inference_mode(False):
                self.joint_bf16 = (
                    head.weight.detach()
                    .to(device=device, dtype=torch.bfloat16)
                    .contiguous(),
                    head.bias.detach()
                    .to(device=device, dtype=torch.bfloat16)
                    .contiguous(),
                )
            return self.joint_bf16
        else:
            return None, None

    def can_decode(
        self, state: Nemotron3_5ASRDecodeState, frames: int, limit: int | None
    ) -> bool:
        """Whether a call cannot stop early, mid-frame, on the token limit."""
        if state.symbols_at_frame:
            return False
        else:
            pass
        return limit is None or state.decoder_steps + frames * self.max_symbols < limit

    def loop_for(self, rows: int, steps: int, frames: torch.Tensor) -> DecodeLoop:
        batch = next((size for size in BATCH_BUCKETS if size >= rows), None)
        span = next((size for size in FRAME_BUCKETS if size >= steps), None)
        key = (
            (batch, span, frames.device, frames.dtype)
            if batch is not None and span is not None
            else None
        )
        if key is not None and key in self.loops:
            return self.loops[key]
        else:
            pass
        jw, jb = (
            self.get_joint_bf16(frames.device)
            if frames.dtype == torch.bfloat16
            else (None, None)
        )
        pred_bf16 = (
            self.get_pred_bf16(frames.device)
            if frames.dtype == torch.bfloat16
            else None
        )
        loop = DecodeLoop(
            self.model,
            rows if key is None else key[0],
            steps if key is None else key[1],
            frames.device,
            frames.dtype,
            joint_weight_bf16=jw,
            joint_bias_bf16=jb,
            pred_weights_bf16=pred_bf16,
        )
        if key is None:
            return loop
        else:
            pass
        if (
            self.use_cuda_graphs
            and frames.is_cuda
            and len([one for one in self.loops.values() if one.graph]) < MAX_GRAPHS
        ):
            if self.stream is None:
                self.stream = torch.cuda.Stream(frames.device)
                self.pool = torch.cuda.graph_pool_handle()
            else:
                pass
            loop.capture(self.stream, self.pool)
        else:
            pass
        self.loops[key] = loop
        return loop

    @torch.inference_mode()
    def capture_common_graphs(
        self,
        device: torch.device,
        dtype: torch.dtype,
        batch_buckets: Sequence[int] = (1, 2, 4, 8, 16, 32),
        frame_buckets: Sequence[int] = (32, 64, 128, 256, 512),
    ) -> None:
        """Pre-capture common decoder graph shapes at startup to avoid mid-run stalls."""
        if not self.use_cuda_graphs or device.type != "cuda":
            return
        else:
            pass
        dummy = torch.empty(
            (1, 1, self.model.config.decoder_hidden_size), device=device, dtype=dtype
        )
        for batch in reversed(batch_buckets):
            for span in reversed(frame_buckets):
                self.loop_for(batch, span, dummy)
        torch.cuda.synchronize(device)

    def decode(
        self,
        states: Sequence[Nemotron3_5ASRDecodeState],
        frames: torch.Tensor,
        lengths: Sequence[int],
    ) -> None:
        """Decode ``frames (B, T, H)`` (joint-projected) into each state's history."""
        device, dtype = frames.device, frames.dtype
        decoder_dtype = self.model.decoder.embedding.weight.dtype
        config = self.model.config
        layers, width = config.num_decoder_layers, config.decoder_hidden_size
        zkey = (device, decoder_dtype)
        zeros = self.zeros_cache.get(zkey)
        if zeros is None:
            zeros = (
                torch.zeros(layers, 1, width, device=device, dtype=decoder_dtype),
                torch.zeros(1, 1, width, device=device, dtype=decoder_dtype),
            )
            self.zeros_cache[zkey] = zeros
        else:
            pass
        zero_state, zero_output = zeros
        caches = [state.decoder_cache for state in states]
        hidden = torch.cat(
            [c.hidden_state if c.is_initialized else zero_state for c in caches], dim=1
        )
        cell = torch.cat(
            [c.cell_state if c.is_initialized else zero_state for c in caches], dim=1
        )
        output = torch.cat(
            [c.cache if c.is_initialized else zero_output for c in caches]
        )[:, 0]
        if output.dtype != dtype:
            output = output.to(dtype)
        else:
            pass
        rows = len(states)
        pending = [
            int(not c.is_initialized or state.tokens[-1] != self.blank)
            for c, state in zip(caches, states, strict=True)
        ]
        pending_label = [state.tokens[-1] for state in states]
        if device.type == "cuda":
            need = 3 * rows
            if self.meta_host is None or self.meta_host.numel() < need:
                cap = max(rows, 128)
                with torch.inference_mode(False):
                    self.meta_host = torch.empty(
                        3 * cap, dtype=torch.int64, pin_memory=True
                    )
            else:
                pass
            view = self.meta_host[:need].view(3, rows)
            view[0].copy_(torch.tensor(lengths, dtype=torch.int64))
            view[1].copy_(torch.tensor(pending, dtype=torch.int64))
            view[2].copy_(torch.tensor(pending_label, dtype=torch.int64))
            meta_dev = view.to(device, non_blocking=True)
            lengths_t = meta_dev[0]
            pending_t = meta_dev[1].bool()
            pending_label_t = meta_dev[2]
        else:
            lengths_t = torch.tensor(lengths, device=device, dtype=torch.long)
            pending_t = torch.tensor(pending, device=device, dtype=torch.bool)
            pending_label_t = torch.tensor(
                pending_label, device=device, dtype=torch.long
            )
        loop = self.loop_for(rows, frames.shape[1], frames)
        loop.load(
            frames,
            lengths_t,
            hidden,
            cell,
            output,
            pending_t,
            pending_label_t,
        )
        loop.run_to_end()
        hidden = loop.hidden[:, :rows].clone()
        cell = loop.cell[:, :rows].clone()
        output_out = (
            loop.output[:rows].to(decoder_dtype)
            if loop.output.dtype != decoder_dtype
            else loop.output[:rows].clone()
        )
        counts = loop.counts[:rows].tolist()
        emitted = max(counts) if counts else 0
        if emitted > 0:
            packed_hist = torch.cat(
                [loop.labels[:rows, :emitted], loop.stamps[:rows, :emitted]], dim=1
            ).tolist()
        else:
            packed_hist = [[] for _ in range(rows)]
        for row, (state, length) in enumerate(zip(states, lengths, strict=True)):
            state.encoder_frames += length
            if length == 0:
                continue
            else:
                pass
            # Views of this call's copies: no per-stream device copy.
            cache = state.decoder_cache
            cache.cache = output_out[row : row + 1, None]
            cache.hidden_state = hidden[:, row : row + 1]
            cache.cell_state = cell[:, row : row + 1]
            cache.is_initialized = True
            c_row = counts[row]
            hist_row = packed_hist[row]
            self.append_history(
                state,
                hist_row[:c_row],
                hist_row[emitted : emitted + c_row],
                length,
            )

    def append_history(
        self,
        state: Nemotron3_5ASRDecodeState,
        labels: list[int],
        stamps: list[int],
        length: int,
    ) -> None:
        """Extend ``tokens`` and ``durations`` as ``decode_streaming_batch`` does."""
        index = 0
        for frame in range(length):
            emitted = 0
            while index < len(labels) and stamps[index] == frame:
                state.tokens.append(labels[index])
                state.durations.append(0)
                emitted += 1
                index += 1
            if emitted == self.max_symbols:
                state.durations[-1] = 1
            else:
                state.tokens.append(self.blank)
                state.durations.append(1)
            state.decoder_steps += emitted + (emitted < self.max_symbols)


__all__ = ["GreedyRNNTDecoder"]
