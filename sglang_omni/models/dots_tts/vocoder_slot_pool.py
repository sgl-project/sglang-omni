# SPDX-License-Identifier: Apache-2.0
"""Per-slot causal AudioVAE streaming state for dots.tts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang_omni.models.dots_tts.incremental_codec_cuda_graph import (
    DotsIncrementalCodecCudaGraphRunner,
)
from sglang_omni.utils.cuda_staging import indices_to_device

if TYPE_CHECKING:
    from dots_tts.modules.vocoder.vocoder_inference import VocoderInference
else:
    pass


def append_decoder_input_per_row(
    decoder_input: torch.Tensor,
    window: torch.Tensor,
    valid_frames: torch.Tensor,
) -> torch.Tensor:
    """Append [B, C, T] into [B, C, W] with per-row valid_frames [B].

    Same geometry as upstream _append_stream_decoder_input_tensor, but rows may
    have different ages (upstream takes a scalar valid count).
    """
    if window.dtype != decoder_input.dtype:
        window = window.to(dtype=decoder_input.dtype)
    else:
        pass
    if valid_frames.ndim != 1 or int(valid_frames.shape[0]) != int(window.shape[0]):
        raise ValueError(
            "valid_frames must have shape [batch], "
            f"got {tuple(valid_frames.shape)} for window batch {int(window.shape[0])}"
        )
    else:
        pass
    chunk_size = int(decoder_input.size(-1))
    window_size = int(window.size(-1))
    if chunk_size >= window_size:
        raise ValueError(
            "decoder window size "
            f"{window_size} must be larger than chunk_size {chunk_size}."
        )
    else:
        pass
    batch = int(window.shape[0])
    positions = torch.arange(
        window_size, device=window.device, dtype=valid_frames.dtype
    )
    clipped_valid = valid_frames.clamp(min=0, max=window_size)
    combined = torch.cat(
        [
            window,
            decoder_input.new_zeros(batch, window.size(1), chunk_size),
        ],
        dim=-1,
    )
    insert_index = clipped_valid.unsqueeze(1) + torch.arange(
        chunk_size, device=window.device, dtype=valid_frames.dtype
    ).view(1, -1)
    combined.scatter_(
        -1,
        insert_index.unsqueeze(1).expand_as(decoder_input),
        decoder_input,
    )
    new_valid = (clipped_valid + chunk_size).clamp(max=window_size)
    start = (clipped_valid + chunk_size - window_size).clamp(min=0)
    gather_index = (start.unsqueeze(1) + positions.view(1, -1)).clamp(
        max=combined.size(-1) - 1
    )
    gathered = combined.gather(
        -1,
        gather_index.unsqueeze(1).expand(batch, window.size(1), window_size),
    )
    mask = (positions.view(1, -1) < new_valid.unsqueeze(1)).to(dtype=window.dtype)
    return gathered * mask.unsqueeze(1)


class DotsVocoderSlotPool:
    """Independent causal rows over one AudioVAE; equal-T steps run as one eager batch.

    Upstream VocoderStreamState keeps scalar total_frames / emitted_frames, so
    batch_size=N is lockstep-only. This pool owns per-slot counters instead.
    """

    def __init__(
        self,
        inference: "VocoderInference",
        *,
        num_slots: int,
        chunk_size: int,
    ) -> None:
        if num_slots < 1:
            raise ValueError(f"num_slots must be >= 1, got {num_slots}")
        else:
            pass
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        else:
            pass
        self.inference = inference
        self.num_slots = int(num_slots)
        self.chunk_size = int(chunk_size)
        # note (guozhihao-224): probe shapes through the public stream-state
        # constructor so we stay aligned with upstream window/LSTM layout.
        probe = inference.init_stream_state(batch_size=1, chunk_size=self.chunk_size)
        hidden_h, hidden_c = probe.lstm_hidden
        window = probe.decoder.window
        layers, _, hidden = hidden_h.shape
        _, channels, window_size = window.shape
        self.window_size = int(window_size)
        self.lookahead = int(
            inference._decoder_stream_lookahead()
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        self.hop_size = int(inference.vocoder.hop_size)
        self.lstm_h = hidden_h.new_zeros(int(layers), self.num_slots, int(hidden))
        self.lstm_c = hidden_c.new_zeros(int(layers), self.num_slots, int(hidden))
        self.window = window.new_zeros(self.num_slots, int(channels), self.window_size)
        self.total_frames = [0] * self.num_slots
        self.emitted_frames = [0] * self.num_slots
        self.free_slots = list(reversed(range(self.num_slots)))
        self.in_use: set[int] = set()
        self.incremental_codec: DotsIncrementalCodecCudaGraphRunner | None = None

    def acquire(self) -> int:
        if not self.free_slots:
            # note (guozhihao-224): slot exhaustion is an admission failure, not a
            # cue to resize the pool under a live request.
            raise RuntimeError(
                "dots.tts streaming vocoder admission failed: ran out of slots "
                f"(num_slots={self.num_slots})"
            )
        else:
            pass
        slot = int(self.free_slots.pop())
        self.reset_slot(slot)
        self.in_use.add(slot)
        return slot

    def release(self, slot: int) -> None:
        slot = int(slot)
        if slot not in self.in_use:
            return
        else:
            pass
        self.reset_slot(slot)
        self.in_use.remove(slot)
        self.free_slots.append(slot)

    @torch.no_grad()
    def step(
        self, slot_latents: dict[int, torch.Tensor], *, final: bool = False
    ) -> dict[int, torch.Tensor]:
        """One uniform-T eager step. slot -> [1, T, C] in, slot -> wav out.

        final marks the last latents of the streams: warm rows then also decode
        the lookahead zero frames of the flush and emit all their audio.
        """
        if not slot_latents:
            return {}
        else:
            pass
        step_lengths = {int(latents.shape[1]) for latents in slot_latents.values()}
        if len(step_lengths) != 1:
            raise ValueError(
                "dots.tts streaming step requires a uniform latent length, "
                f"got {sorted(step_lengths)}"
            )
        else:
            pass
        (step_t,) = step_lengths
        if step_t < 1:
            raise ValueError("dots.tts streaming step length must be positive")
        else:
            pass
        if step_t >= self.window_size:
            raise ValueError(
                f"streaming step length {step_t} must be < window size "
                f"{self.window_size}"
            )
        else:
            pass
        slots = [int(slot) for slot in slot_latents]
        for slot in slots:
            if slot not in self.in_use:
                raise RuntimeError(
                    f"dots.tts streaming step referenced free slot {slot}"
                )
            else:
                pass
            latents = slot_latents[slot]
            if latents.ndim != 3 or int(latents.shape[0]) != 1:
                raise ValueError(
                    "slot latents must have shape [1, frames, latent_dim], "
                    f"got {tuple(latents.shape)} for slot {slot}"
                )
            else:
                pass

        slot_index = indices_to_device(slots, self.window.device)
        # note (guozhihao-224): upstream stream kernels take channel-major
        # latents [B, C, T]; Omni chunks arrive as [1, T, C].
        packed = torch.cat(
            [slot_latents[slot].transpose(1, 2).contiguous() for slot in slots],
            dim=0,
        )
        if packed.device != self.window.device:
            # note (0xtoward): host chunks move once per step through pinned memory.
            packed = packed.pin_memory().to(self.window.device, non_blocking=True)
        else:
            pass
        hidden_h = self.lstm_h.index_select(1, slot_index).contiguous()
        hidden_c = self.lstm_c.index_select(1, slot_index).contiguous()
        window = self.window.index_select(0, slot_index).contiguous()
        valid = indices_to_device(
            [min(self.total_frames[slot], self.window_size) for slot in slots],
            window.device,
        )

        inference = self.inference
        # note (guozhihao-224): call VocoderInference private eager helpers so
        # rows can age independently; stream_step's scalar counters are
        # lockstep-only. Expect breakage if upstream renames these.
        inference._validate_stream_latents(
            packed
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        decoder_input, (hidden_h, hidden_c) = (
            inference._decode_stream_latents(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
                packed, (hidden_h, hidden_c)
            )
        )
        new_window = append_decoder_input_per_row(decoder_input, window, valid)
        self.lstm_h[:, slot_index, :] = hidden_h
        self.lstm_c[:, slot_index, :] = hidden_c
        self.window[slot_index] = new_window
        if self.incremental_codec is None:
            audio_window = inference._decode_stream_window(
                new_window
            )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            out: dict[int, torch.Tensor] = {}
            for row, slot in enumerate(slots):
                self.total_frames[slot] += step_t
                out[slot] = self.slice_audio(
                    slot, audio_window[row : row + 1], final=False
                )
            return out
        else:
            return self.incremental_step(
                slots, decoder_input, new_window, step_t, final=final
            )

    def is_warm(self, slot: int) -> bool:
        """Whether the slot's decoded history covers every stage's left context."""
        assert self.incremental_codec is not None
        decoder = self.incremental_codec.decoder
        return self.total_frames[slot] - self.lookahead >= decoder.warm_history_frames

    def incremental_step(
        self,
        slots: list[int],
        decoder_input: torch.Tensor,
        new_window: torch.Tensor,
        step_t: int,
        *,
        final: bool = False,
    ) -> dict[int, torch.Tensor]:
        """Decode one step: warm rows decode only their new frames.

        Cold rows decode their whole window and record the history that later
        warm steps continue from.
        """
        assert self.incremental_codec is not None
        warm_rows = [row for row, slot in enumerate(slots) if self.is_warm(slot)]
        cold_rows = [row for row, slot in enumerate(slots) if not self.is_warm(slot)]
        out: dict[int, torch.Tensor] = {}
        if warm_rows:
            frames = decoder_input[warm_rows]
            if final:
                # note (0xtoward): the flush's lookahead zero frames ride along with
                # the last latents, saving a separate one-row decoder step.
                frames = torch.cat(
                    [
                        frames,
                        frames.new_zeros(
                            frames.shape[0], frames.shape[1], self.lookahead
                        ),
                    ],
                    dim=-1,
                )
            else:
                pass
            audio = self.incremental_codec.decode_warm(
                frames, [slots[row] for row in warm_rows]
            )
            for position, row in enumerate(warm_rows):
                slot = slots[row]
                self.total_frames[slot] += step_t
                self.emitted_frames[slot] = self.total_frames[slot] - (
                    0 if final else self.lookahead
                )
                out[slot] = audio[position : position + 1]
        else:
            pass
        if cold_rows:
            valid = [
                min(self.total_frames[slots[row]] + step_t, self.window_size)
                for row in cold_rows
            ]
            audio_window = self.incremental_codec.decode_cold(
                new_window[cold_rows],
                [slots[row] for row in cold_rows],
                [frames - self.lookahead for frames in valid],
                valid,
            )
            for position, row in enumerate(cold_rows):
                slot = slots[row]
                self.total_frames[slot] += step_t
                out[slot] = self.slice_audio(
                    slot, audio_window[position : position + 1], final=False
                )
        else:
            pass
        return out

    @torch.no_grad()
    def flush(self, slot: int) -> torch.Tensor:
        slot = int(slot)
        if slot not in self.in_use:
            raise RuntimeError(f"dots.tts streaming flush referenced free slot {slot}")
        else:
            pass
        if self.emitted_frames[slot] >= self.total_frames[slot]:
            # note (0xtoward): a final step already emitted this slot's audio.
            return self.window.new_zeros((1, 1, 0))
        elif self.incremental_codec is None:
            audio_window = self.inference._decode_stream_window(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
                self.window[slot : slot + 1]
            )
        elif self.is_warm(slot):
            audio = self.incremental_codec.flush(slot)
            self.emitted_frames[slot] = self.total_frames[slot]
            return audio
        else:
            valid = min(self.total_frames[slot], self.window_size)
            audio_window = self.incremental_codec.decode_cold(
                self.window[slot : slot + 1], [slot], [valid], [valid]
            )
        return self.slice_audio(slot, audio_window, final=True)

    def reset_slot(self, slot: int) -> None:
        self.lstm_h[:, slot].zero_()
        self.lstm_c[:, slot].zero_()
        self.window[slot].zero_()
        self.total_frames[slot] = 0
        self.emitted_frames[slot] = 0

    def slice_audio(
        self, slot: int, audio_window: torch.Tensor, *, final: bool
    ) -> torch.Tensor:
        # note (guozhihao-224): mirrors VocoderInference._slice_stream_audio_window
        # with per-slot total/emitted counters instead of the upstream scalars.
        total = self.total_frames[slot]
        emitted = self.emitted_frames[slot]
        stable_end = total if final else max(0, total - self.lookahead)
        if stable_end <= emitted:
            return audio_window.new_zeros((audio_window.size(0), 1, 0))
        else:
            pass

        valid_frames = min(total, self.window_size)
        window_start = total - valid_frames
        if emitted < window_start:
            raise RuntimeError(
                "Decoder stream window is too short for fixed-graph decoding."
            )
        else:
            pass
        local_start = emitted - window_start
        local_end = stable_end - window_start
        sample_start = local_start * self.hop_size
        sample_end = local_end * self.hop_size
        self.emitted_frames[slot] = stable_end
        return audio_window[..., sample_start:sample_end]


__all__ = ["DotsVocoderSlotPool", "append_decoder_input_per_row"]
