# SPDX-License-Identifier: Apache-2.0
"""Per-slot causal AudioVAE streaming state for dots.tts."""

from __future__ import annotations

from typing import Any

import torch


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
    if valid_frames.ndim != 1 or int(valid_frames.shape[0]) != int(window.shape[0]):
        raise ValueError(
            "valid_frames must have shape [batch], "
            f"got {tuple(valid_frames.shape)} for window batch {int(window.shape[0])}"
        )
    chunk_size = int(decoder_input.size(-1))
    window_size = int(window.size(-1))
    if chunk_size >= window_size:
        raise ValueError(
            "decoder window size "
            f"{window_size} must be larger than chunk_size {chunk_size}."
        )
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
        inference: Any,
        *,
        num_slots: int,
        chunk_size: int,
    ) -> None:
        if num_slots < 1:
            raise ValueError(f"num_slots must be >= 1, got {num_slots}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self._inference = inference
        self.num_slots = int(num_slots)
        self.chunk_size = int(chunk_size)
        # note (guozhihao-224): probe shapes through the public stream-state
        # constructor so we stay aligned with upstream window/LSTM layout.
        probe = inference.init_stream_state(batch_size=1, chunk_size=self.chunk_size)
        hidden_h, hidden_c = probe.lstm_hidden
        window = probe.decoder.window
        layers, _, hidden = hidden_h.shape
        _, channels, window_size = window.shape
        self._window_size = int(window_size)
        self._lookahead = int(inference._decoder_stream_lookahead())
        self._hop_size = int(inference.vocoder.hop_size)
        self._lstm_h = hidden_h.new_zeros(int(layers), self.num_slots, int(hidden))
        self._lstm_c = hidden_c.new_zeros(int(layers), self.num_slots, int(hidden))
        self._window = window.new_zeros(
            self.num_slots, int(channels), self._window_size
        )
        self._total_frames = [0] * self.num_slots
        self._emitted_frames = [0] * self.num_slots
        self._free_slots = list(reversed(range(self.num_slots)))
        self._in_use: set[int] = set()

    def acquire(self) -> int:
        if not self._free_slots:
            # note (guozhihao-224): slot exhaustion is an admission failure, not a
            # cue to resize the pool under a live request.
            raise RuntimeError(
                "dots.tts streaming vocoder admission failed: ran out of slots "
                f"(num_slots={self.num_slots})"
            )
        slot = int(self._free_slots.pop())
        self._reset_slot(slot)
        self._in_use.add(slot)
        return slot

    def release(self, slot: int) -> None:
        slot = int(slot)
        if slot not in self._in_use:
            return
        self._reset_slot(slot)
        self._in_use.remove(slot)
        self._free_slots.append(slot)

    @torch.no_grad()
    def step(self, slot_latents: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        """One uniform-T eager step. slot -> [1, T, C] in, slot -> wav out."""
        if not slot_latents:
            return {}
        step_lengths = {int(latents.shape[1]) for latents in slot_latents.values()}
        if len(step_lengths) != 1:
            raise ValueError(
                "dots.tts streaming step requires a uniform latent length, "
                f"got {sorted(step_lengths)}"
            )
        (step_t,) = step_lengths
        if step_t < 1:
            raise ValueError("dots.tts streaming step length must be positive")
        if step_t >= self._window_size:
            raise ValueError(
                f"streaming step length {step_t} must be < window size "
                f"{self._window_size}"
            )
        slots = [int(slot) for slot in slot_latents]
        for slot in slots:
            if slot not in self._in_use:
                raise RuntimeError(
                    f"dots.tts streaming step referenced free slot {slot}"
                )
            latents = slot_latents[slot]
            if latents.ndim != 3 or int(latents.shape[0]) != 1:
                raise ValueError(
                    "slot latents must have shape [1, frames, latent_dim], "
                    f"got {tuple(latents.shape)} for slot {slot}"
                )

        slot_index = torch.tensor(slots, device=self._window.device, dtype=torch.long)
        # note (guozhihao-224): upstream stream kernels take channel-major
        # latents [B, C, T]; Omni chunks arrive as [1, T, C].
        packed = torch.cat(
            [slot_latents[slot].transpose(1, 2).contiguous() for slot in slots],
            dim=0,
        )
        hidden_h = self._lstm_h.index_select(1, slot_index).contiguous()
        hidden_c = self._lstm_c.index_select(1, slot_index).contiguous()
        window = self._window.index_select(0, slot_index).contiguous()
        valid = torch.tensor(
            [min(self._total_frames[slot], self._window_size) for slot in slots],
            device=window.device,
            dtype=torch.int64,
        )

        inference = self._inference
        # note (guozhihao-224): call VocoderInference private eager helpers so
        # rows can age independently; stream_step's scalar counters are
        # lockstep-only. Expect breakage if upstream renames these.
        inference._validate_stream_latents(packed)
        decoder_input, (hidden_h, hidden_c) = inference._decode_stream_latents(
            packed, (hidden_h, hidden_c)
        )
        new_window = append_decoder_input_per_row(decoder_input, window, valid)
        audio_window = inference._decode_stream_window(new_window)

        self._lstm_h[:, slot_index, :] = hidden_h
        self._lstm_c[:, slot_index, :] = hidden_c
        self._window[slot_index] = new_window
        out: dict[int, torch.Tensor] = {}
        for row, slot in enumerate(slots):
            self._total_frames[slot] += step_t
            out[slot] = self._slice_audio(
                slot, audio_window[row : row + 1], final=False
            )
        return out

    @torch.no_grad()
    def flush(self, slot: int) -> torch.Tensor:
        slot = int(slot)
        if slot not in self._in_use:
            raise RuntimeError(f"dots.tts streaming flush referenced free slot {slot}")
        audio_window = self._inference._decode_stream_window(
            self._window[slot : slot + 1]
        )
        return self._slice_audio(slot, audio_window, final=True)

    def _reset_slot(self, slot: int) -> None:
        self._lstm_h[:, slot].zero_()
        self._lstm_c[:, slot].zero_()
        self._window[slot].zero_()
        self._total_frames[slot] = 0
        self._emitted_frames[slot] = 0

    def _slice_audio(
        self, slot: int, audio_window: torch.Tensor, *, final: bool
    ) -> torch.Tensor:
        # note (guozhihao-224): mirrors VocoderInference._slice_stream_audio_window
        # with per-slot total/emitted counters instead of the upstream scalars.
        total = self._total_frames[slot]
        emitted = self._emitted_frames[slot]
        stable_end = total if final else max(0, total - self._lookahead)
        if stable_end <= emitted:
            return audio_window.new_zeros((audio_window.size(0), 1, 0))

        valid_frames = min(total, self._window_size)
        window_start = total - valid_frames
        if emitted < window_start:
            raise RuntimeError(
                "Decoder stream window is too short for fixed-graph decoding."
            )
        local_start = emitted - window_start
        local_end = stable_end - window_start
        sample_start = local_start * self._hop_size
        sample_end = local_end * self._hop_size
        self._emitted_frames[slot] = stable_end
        return audio_window[..., sample_start:sample_end]


__all__ = ["DotsVocoderSlotPool", "append_decoder_input_per_row"]
