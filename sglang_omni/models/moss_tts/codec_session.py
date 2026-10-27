# SPDX-License-Identifier: Apache-2.0
"""Persistent state slots and batched steps for the native MOSS codec."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import torch

from sglang_omni.models.moss_tts.audio_tokenizer import MossAudioTokenizerVocoder
from sglang_omni.models.moss_tts.vocoder_cuda_graph import MossVocoderCudaGraphRunner
from sglang_omni.scheduling.generation_batch_policy import build_default_cuda_graph_bs

logger = logging.getLogger(__name__)


class CodecStreamSession:
    """Persistent codec state session with slot bookkeeping.

    Live requests hold stream slots for their lifetime. Non-streaming work never
    enters the session; it uses the batched full-sequence decode path. All methods
    run on the scheduler-loop thread.
    """

    def __init__(
        self,
        codec: MossAudioTokenizerVocoder,
        *,
        stream_slots: int,
        n_vq: int,
    ) -> None:
        self.codec: MossAudioTokenizerVocoder = codec
        self.stream_slots: int = int(stream_slots)
        self.n_vq: int = int(n_vq)
        self.device: torch.device = next(codec.parameters()).device
        codec.initialize_decoder_state_pool(
            self.stream_slots, scratch_capacity=self.stream_slots
        )
        self.free_stream_slots: list[int] = list(range(self.stream_slots))
        self.stream_slots_in_use: set[int] = set()
        self.closed: bool = False
        self.cg_runner: MossVocoderCudaGraphRunner | None = None
        # Capture is attempted at most once per session; a low-VRAM skip must not re-probe per step.
        self.warmup_attempted: bool = False
        # Per-T graph-vs-eager step counts for capture-hit-rate reporting (host-side, no GPU sync).
        self.cg_graph_t: Counter[int] = Counter()
        self.cg_eager_t: Counter[int] = Counter()
        self.cg_total_steps: int = 0
        self.compact_batch_sizes: Counter[int] = Counter()

    def warmup_cuda_graph(
        self, frames: list[int], *, min_free_gb: float = 3.0
    ) -> list[int]:
        """Capture native vocoder graphs once; uncaptured shapes use eager decode."""
        self.warmup_attempted = True
        if self.closed:
            return []
        else:
            pass
        if self.cg_runner is None:
            self.cg_runner = MossVocoderCudaGraphRunner(
                self.codec,
                real_state_capacity=self.stream_slots,
                scratch_capacity=self.stream_slots,
                batch_sizes=self.graph_batch_sizes(),
                frame_sizes=frames,
                num_quantizers=self.n_vq,
                min_free_gb=min_free_gb,
            )
        else:
            pass
        try:
            self.cg_runner.warmup(frames)
        except Exception:
            self.cg_runner = None
            raise
        self.reset_slots(list(range(self.stream_slots)))
        captured = self.cg_runner.captured_frames()
        if not captured:
            self.cg_runner = None
        else:
            pass
        return captured

    def has_cuda_graph_runner(self) -> bool:
        # True only if the runner exists AND captured at least one graph.
        return bool(self.cg_runner and self.cg_runner.captured_frames())

    def captured_frames(self) -> list[int]:
        return self.cg_runner.captured_frames() if self.cg_runner else []

    def graph_batch_sizes(self) -> list[int]:
        """Return batch buckets shared by eager and CUDA graph execution."""
        return build_default_cuda_graph_bs(self.stream_slots)

    def acquire(self) -> int | None:
        if not self.free_stream_slots:
            return None
        else:
            pass
        slot = self.free_stream_slots.pop()
        self.stream_slots_in_use.add(slot)
        return slot

    def release(self, slot: int) -> None:
        if self.closed:
            return
        else:
            pass
        if slot not in self.stream_slots_in_use:
            raise RuntimeError(
                f"MOSS-Audio-Tokenizer vocoder stream slot {slot} is not leased"
            )
        else:
            pass
        self.reset_slots([slot])
        self.stream_slots_in_use.remove(slot)
        self.free_stream_slots.append(slot)

    def close(self) -> None:
        if self.closed:
            return
        else:
            pass
        if self.cg_runner is not None:
            self.log_cg_stats()
        else:
            pass
        if self.compact_batch_sizes:
            logger.info(
                f"MOSS-Audio-Tokenizer vocoder compact streaming batches: "
                f"B={dict(sorted(self.compact_batch_sizes.items()))}"
            )
        else:
            pass
        try:
            self.codec.close_decoder_state_pool()
        finally:
            self.cg_runner = None
            self.closed = True

    def log_cg_stats(self) -> None:
        graph = sum(self.cg_graph_t.values())
        eager = sum(self.cg_eager_t.values())
        total = graph + eager
        if not total:
            return
        else:
            pass
        logger.info(
            f"MOSS-Audio-Tokenizer vocoder CG stats: {graph}/{total} steps graphed "
            f"({100.0 * graph / total:.1f}%); "
            f"graph T={dict(sorted(self.cg_graph_t.items()))} "
            f"eager T={dict(sorted(self.cg_eager_t.items()))}"
        )

    def reset_slots(self, slots: list[int]) -> None:
        if not slots:
            return
        else:
            pass
        slot_ids = torch.as_tensor(slots, dtype=torch.long, device=self.device)
        with torch.no_grad():
            self.codec.reset_decoder_state_slots(slot_ids)

    def step(self, slot_codes: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        """Advance equal-length code rows and return each slot's CPU waveform."""
        if not slot_codes:
            return {}
        else:
            pass
        for slot, codes in slot_codes.items():
            if not isinstance(slot, int) or isinstance(slot, bool):
                raise TypeError(f"streaming slot id must be an int, got {slot!r}")
            else:
                pass
            if slot < 0 or slot >= self.stream_slots:
                raise ValueError(
                    f"streaming slot {slot} is outside [0, {self.stream_slots})"
                )
            else:
                pass
            if int(codes.ndim) != 2:
                raise ValueError(
                    f"streaming slot {slot} codes must have shape [NQ, T], "
                    f"got {tuple(codes.shape)}"
                )
            else:
                pass
            if int(codes.shape[0]) <= 0 or int(codes.shape[1]) <= 0:
                raise ValueError(
                    f"streaming slot {slot} codes must have positive NQ and T, "
                    f"got {tuple(codes.shape)}"
                )
            else:
                pass
        step_lengths = {int(codes.shape[1]) for codes in slot_codes.values()}
        if len(step_lengths) != 1:
            raise ValueError(
                f"streaming step requires a uniform length, got {sorted(step_lengths)}"
            )
        else:
            pass
        (step_t,) = step_lengths
        n_vq = int(next(iter(slot_codes.values())).shape[0])
        if n_vq != self.n_vq:
            raise ValueError(
                f"streaming codes must use {self.n_vq} quantizers, got {n_vq}"
            )
        else:
            pass
        if any(int(codes.shape[0]) != n_vq for codes in slot_codes.values()):
            raise ValueError("all streaming slots must use the same quantizer count")
        else:
            pass
        slots = list(slot_codes)
        # Use the same batch bucket for eager and graph execution. Changing
        # GEMM shapes on a graph miss can change BF16 PCM and live KV state.
        batch_size = next(
            size for size in self.graph_batch_sizes() if size >= len(slots)
        )
        padding = batch_size - len(slots)
        rows = [
            codes.to(device=self.device, dtype=torch.long)
            for codes in slot_codes.values()
        ]
        if padding:
            rows.extend([rows[0].new_zeros(n_vq, step_t)] * padding)
        else:
            pass
        codes_step = torch.stack(rows, dim=1)
        codes_lengths = torch.tensor(
            [step_t] * len(slots) + [0] * padding,
            dtype=torch.long,
            device=self.device,
        )
        state_slot_ids = torch.as_tensor(
            slots + list(range(self.stream_slots, self.stream_slots + padding)),
            dtype=torch.long,
            device=self.device,
        )
        exec_mask = torch.tensor(
            [True] * len(slots) + [False] * padding,
            dtype=torch.bool,
            device=self.device,
        )
        self.compact_batch_sizes[batch_size] += 1
        graphed = None
        graph_failed = False
        try:
            with torch.no_grad():
                if self.cg_runner is not None:
                    try:
                        graphed = self.cg_runner.decode_step(
                            codes_step,
                            state_slot_ids,
                            exec_mask,
                        )
                    except Exception:
                        graph_failed = True
                        raise
                else:
                    pass
                if graphed is not None:
                    audio, audio_lengths = graphed
                else:
                    audio, audio_lengths = self.codec.decode_streaming_tensors(
                        codes_step,
                        codes_lengths,
                        state_slot_ids,
                        exec_mask,
                    )
            # One batched D2H per step. A graph replay error can surface async HERE (not in
            # decode_step), so materialization stays inside the replay guard.
            audio_cpu = audio.detach().to("cpu", torch.float32)
            lengths_cpu = audio_lengths.detach().to("cpu")
        except Exception:
            # Graphed step failed (in decode_step or async on the D2H): disable the runner so future
            # steps go eager; participants abort. An eager-path error does not disable it.
            if self.cg_runner is not None and (graph_failed or graphed is not None):
                logger.exception(
                    "MOSS-Audio-Tokenizer vocoder CUDA-graph replay failed "
                    "(in decode_step or on output "
                    "materialization); disabling runner, serving eager from here"
                )
                self.cg_runner = None
            else:
                pass
            raise
        if self.cg_runner is not None:
            if graphed is not None:
                self.cg_graph_t[step_t] += 1
            else:
                self.cg_eager_t[step_t] += 1
            self.cg_total_steps += 1
            if self.cg_total_steps % 2000 == 0:
                self.log_cg_stats()
            else:
                pass
        else:
            pass
        out: dict[int, torch.Tensor] = {}
        for index, slot in enumerate(slots):
            n_samples = int(lengths_cpu[index])
            out[slot] = audio_cpu[index, :, :n_samples]
        return out


@dataclass
class CoalescedStepPlan:
    step_t: int
    slot_codes: dict[int, torch.Tensor]
