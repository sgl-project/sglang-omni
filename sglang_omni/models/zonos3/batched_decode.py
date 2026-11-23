# SPDX-License-Identifier: Apache-2.0
"""Decode Zonos3 code prefixes as right-padded batches of similar lengths, and slotted streams by their new frames only."""

from __future__ import annotations

import contextlib
import math
import threading

import torch
import torch.nn.functional as F

from sglang_omni.models.zonos3.components.codec import (
    EXTERNAL_RESAMPLE_RATIO,
    EXTERNAL_SAMPLES_PER_FRAME,
    HEAD_FRAMES_PER_CODE_FRAME,
    INTERNAL_RESAMPLE_RATIO,
    MAX_DECODER_GRAPH_FRAMES,
    TO_EXTERNAL_KERNEL_TAPS,
    Zonos3CodecDecoder,
)
from sglang_omni.models.zonos3.components.codec_decoder_kernels import (
    polyphase_resample,
)
from sglang_omni.models.zonos3.components.codec_stream import (
    Zonos3DecoderSlots,
    Zonos3DecoderStream,
)
from sglang_omni.models.zonos3.components.istft_head import (
    ISTFT_FFT_SIZES,
    ISTFT_HOP_SAMPLES,
    ISTFT_MIXER_KERNEL_SIZE,
)

# note (zhaochenyang20): On one H100 a decoder call costs about as much as 50 more padded frames, so prefixes further apart than this decode as separate batches.
MAX_STEP_PADDING_FRAMES = 48
# A resampler block holds INTERNAL_RESAMPLE_RATIO internal and EXTERNAL_RESAMPLE_RATIO external samples.
HEAD_FRAMES_PER_BLOCK = INTERNAL_RESAMPLE_RATIO // ISTFT_HOP_SAMPLES
# note (zhaochenyang20): The iSTFT half window, the mixer's half kernel and the resampler's half width reach 1184 internal samples back, so a slotted stream's tail is synthesized again from two blocks before its first unsent one.
TAIL_CONTEXT_BLOCKS = math.ceil(
    (
        max(ISTFT_FFT_SIZES) // 2
        + ISTFT_MIXER_KERNEL_SIZE // 2
        + (TO_EXTERNAL_KERNEL_TAPS - INTERNAL_RESAMPLE_RATIO) // 2
    )
    / INTERNAL_RESAMPLE_RATIO
)


def length_groups(frame_counts: list[int]) -> list[list[int]]:
    """Indices of frame_counts in the fewest groups whose lengths lie within MAX_STEP_PADDING_FRAMES."""
    groups: list[list[int]] = []
    for index in sorted(range(len(frame_counts)), key=frame_counts.__getitem__):
        if (
            groups
            and frame_counts[index] - frame_counts[groups[-1][0]]
            <= MAX_STEP_PADDING_FRAMES
        ):
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


class Zonos3BatchedDecode:
    """The vocoder's codec with its private CUDA stream and the lock that serializes its decodes.

    Waveforms leave scaled by headroom_gain, the codec registry's output headroom, so loud speech does not clip.
    """

    def __init__(
        self,
        decoder: Zonos3CodecDecoder,
        *,
        headroom_gain: float,
        slots: Zonos3DecoderSlots | None,
    ) -> None:
        self.decoder = decoder
        self.headroom_gain = headroom_gain
        self.slots = slots
        self.decode_stream: torch.cuda.Stream | None = None
        self.decode_lock: threading.Lock = threading.Lock()
        if slots is None:
            pass
        else:
            device = decoder.to_external_kernel.device
            # note (zhaochenyang20): A window after a stream's first decode starts TAIL_CONTEXT_BLOCKS blocks before its unsent samples, so it spans a whole number of frames plus those blocks, lengths the decoder's synthesis warm-up never builds; SeedTTS EN streaming at concurrency 16 on one H100 built about 250 such FFT plans while serving, 8 module loads each on the vocoder thread, and the engine's launches that overlapped them stalled.
            with torch.inference_mode():
                for frame_count in range(MAX_DECODER_GRAPH_FRAMES):
                    head_frames = (
                        frame_count * HEAD_FRAMES_PER_CODE_FRAME
                        + TAIL_CONTEXT_BLOCKS * HEAD_FRAMES_PER_BLOCK
                    )
                    polyphase_resample(
                        decoder.head.synthesize(
                            [
                                torch.zeros(
                                    (1, fft_size // 2 + 1, head_frames),
                                    dtype=torch.complex64,
                                    device=device,
                                )
                                for fft_size in ISTFT_FFT_SIZES
                            ],
                            head_frames * ISTFT_HOP_SAMPLES,
                        ),
                        decoder.to_external_kernel,
                        INTERNAL_RESAMPLE_RATIO,
                        EXTERNAL_RESAMPLE_RATIO,
                    )
            torch.cuda.synchronize(device)

    def stream_context(
        self,
    ) -> torch.cuda.StreamContext | contextlib.nullcontext[None]:
        """The private CUDA stream, which carries all of the vocoder's device work from host codes; CPU decodes run inline."""
        device = self.decoder.to_external_kernel.device
        if device.type == "cuda":
            if self.decode_stream is None:
                self.decode_stream = torch.cuda.Stream(device=device)
            else:
                pass
            # note (zhaochenyang20): The stream waits for no other: it reads only host codes and device state made ready by the startup synchronize, while the shared default stream also carries other threads' work; ordering after that stream created an event per decode, and on one H100 serving SeedTTS EN at concurrency 16 an eighth of those calls stalled 6-8 ms (up to 16 ms), mostly behind a reference encoder's pageable copy: 4% of the non-streaming and 11% of the streaming vocoder thread's time.
            return torch.cuda.stream(self.decode_stream)
        else:
            return contextlib.nullcontext()

    def decode_waveforms(
        self,
        prefixes: list[torch.Tensor],
        sample_ranges: list[tuple[int, int]],
        *,
        exact_prefix_count: int,
        decoder_streams: list[Zonos3DecoderStream | None],
    ) -> list[torch.Tensor]:
        """Host samples [start, stop) of each code prefix [codebooks, frames]; a prefix with a decoder stream runs only its new frames through the stream's slot.

        The others decode as right-padded batches of similar lengths: the first exact_prefix_count prefixes emit their last frame, so one shorter than its batch is synthesized at its own length; the rest of a batch share one synthesis whose padding reaches only their held-back boundary frame.
        """
        frame_counts = [prefix.shape[1] for prefix in prefixes]
        slotted_indices = [
            index
            for index, decoder_stream in enumerate(decoder_streams)
            if decoder_stream is not None
        ]
        prefix_indices = [
            index
            for index, decoder_stream in enumerate(decoder_streams)
            if decoder_stream is None
        ]
        device = self.decoder.to_external_kernel.device
        chunks: dict[int, torch.Tensor] = {}
        waveforms: dict[int, torch.Tensor] = {}
        with self.decode_lock, self.stream_context():
            if slotted_indices:
                assert self.slots is not None
                slotted_streams = [decoder_streams[index] for index in slotted_indices]
                self.slots.advance(
                    slotted_streams,
                    [
                        prefixes[index][:, decoder_stream.pushed_frame_count :]
                        for index, decoder_stream in zip(
                            slotted_indices, slotted_streams, strict=True
                        )
                    ],
                )
                chunks.update(
                    zip(
                        slotted_indices,
                        self.synthesize_streams(
                            slotted_streams,
                            [sample_ranges[index] for index in slotted_indices],
                        ),
                        strict=True,
                    )
                )
            else:
                pass
            for group in [
                [prefix_indices[position] for position in group]
                for group in length_groups(
                    [frame_counts[index] for index in prefix_indices]
                )
            ]:
                group_frame_count = max(frame_counts[index] for index in group)
                # Padding reaches an exact prefix's emitted samples only when it is shorter than its batch.
                alone_indices = [
                    index
                    for index in group
                    if index < exact_prefix_count
                    and frame_counts[index] < group_frame_count
                ]
                shared_indices = [
                    index for index in group if index not in alone_indices
                ]
                codes = torch.stack(
                    [
                        F.pad(
                            prefixes[index],
                            (0, group_frame_count - frame_counts[index]),
                        )
                        for index in alone_indices + shared_indices
                    ]
                )
                if device.type == "cuda":
                    # A pinned copy queues on the decode stream; a pageable one first waits for that stream to drain.
                    codes = codes.pin_memory().to(device, non_blocking=True)
                else:
                    pass
                hidden_states = self.decoder.decode_hidden_states(codes)
                states_per_frame = hidden_states.shape[-1] // group_frame_count
                for row, index in enumerate(alone_indices):
                    waveforms[index] = self.decoder.synthesize(
                        hidden_states[
                            row : row + 1,
                            :,
                            : frame_counts[index] * states_per_frame,
                        ]
                    )[0]
                waveforms.update(
                    zip(
                        shared_indices,
                        self.decoder.synthesize(hidden_states[len(alone_indices) :]),
                        strict=True,
                    )
                )
            for index in prefix_indices:
                start_sample, stop_sample = sample_ranges[index]
                chunks[index] = waveforms[index][start_sample:stop_sample]
            # One device-to-host copy serves every request of the step.
            samples = (
                torch.cat([chunks[index] for index in range(len(prefixes))])
                * self.headroom_gain
            )
            if device.type == "cuda":
                # note (zhaochenyang20): The driver queues a pageable copy behind any other thread's, and a reference encoder's waits for its whole pass; on one H100 serving SeedTTS EN at concurrency 16 a third of the vocoder's copies waited 7-8 ms (up to 17 ms) behind one, 12% of the vocoder thread's time, so the samples land in pinned memory. With the stream wait dropped too, it served 3.9% more requests per second non-streaming (median of 16 ABBA pairs) and 3.9% more streaming with first audio 9 ms earlier (4 pairs), with bit-identical audio.
                host_samples = torch.empty(
                    samples.shape, dtype=samples.dtype, pin_memory=True
                )
                host_samples.copy_(samples, non_blocking=True)
                torch.cuda.current_stream(device).synchronize()
            else:
                host_samples = samples.cpu()
        return list(
            host_samples.split(
                [
                    stop_sample - start_sample
                    for start_sample, stop_sample in sample_ranges
                ]
            )
        )

    @torch.inference_mode()
    def synthesize_streams(
        self,
        decoder_streams: list[Zonos3DecoderStream],
        sample_ranges: list[tuple[int, int]],
    ) -> list[torch.Tensor]:
        """Each stream's external samples [start, stop) from its kept spectra; start and stop lie on resampler blocks.

        A stop short of the stream's last frame leaves its held-back boundary unsent, so its window is right-padded to the longest and synthesized in one batch; a stream that sends its end is synthesized at its own length.
        """
        head = self.decoder.head
        window_first_blocks: list[int] = []
        windows: list[list[torch.Tensor]] = []
        for decoder_stream, (start_sample, stop_sample) in zip(
            decoder_streams, sample_ranges, strict=True
        ):
            assert start_sample % EXTERNAL_RESAMPLE_RATIO == 0, start_sample
            window_first_block = max(
                start_sample // EXTERNAL_RESAMPLE_RATIO - TAIL_CONTEXT_BLOCKS, 0
            )
            window_offset = (
                window_first_block * HEAD_FRAMES_PER_BLOCK
                - decoder_stream.spectra_first_head_frame
            )
            assert window_offset >= 0, window_offset
            # The kept spectra start at this call's window, so a call that fails later can be retried from the same start.
            decoder_stream.spectra = [
                spectrum[:, window_offset:] for spectrum in decoder_stream.spectra
            ]
            decoder_stream.spectra_first_head_frame = (
                window_first_block * HEAD_FRAMES_PER_BLOCK
            )
            window_first_blocks.append(window_first_block)
            windows.append(decoder_stream.spectra)
        window_head_frames = [window[0].shape[-1] for window in windows]
        batch_head_frames = max(window_head_frames)
        alone_rows = [
            row
            for row, (decoder_stream, (_, stop_sample)) in enumerate(
                zip(decoder_streams, sample_ranges, strict=True)
            )
            if stop_sample
            == decoder_stream.pushed_frame_count * EXTERNAL_SAMPLES_PER_FRAME
            and window_head_frames[row] < batch_head_frames
        ]
        shared_rows = [row for row in range(len(windows)) if row not in alone_rows]
        waveforms: dict[int, torch.Tensor] = {}
        for row_group, group_head_frames in [
            ([row], window_head_frames[row]) for row in alone_rows
        ] + [(shared_rows, batch_head_frames)]:
            if row_group:
                internal_waveform = head.synthesize(
                    [
                        torch.stack(
                            [
                                F.pad(
                                    windows[row][resolution],
                                    (0, group_head_frames - window_head_frames[row]),
                                )
                                for row in row_group
                            ]
                        )
                        for resolution in range(len(ISTFT_FFT_SIZES))
                    ],
                    group_head_frames * ISTFT_HOP_SAMPLES,
                )
                waveforms.update(
                    zip(
                        row_group,
                        polyphase_resample(
                            internal_waveform,
                            self.decoder.to_external_kernel,
                            INTERNAL_RESAMPLE_RATIO,
                            EXTERNAL_RESAMPLE_RATIO,
                        ),
                        strict=True,
                    )
                )
            else:
                pass
        chunks: list[torch.Tensor] = []
        for row, (start_sample, stop_sample) in enumerate(sample_ranges):
            window_first_sample = window_first_blocks[row] * EXTERNAL_RESAMPLE_RATIO
            chunks.append(
                waveforms[row][
                    start_sample
                    - window_first_sample : stop_sample
                    - window_first_sample
                ]
            )
        return chunks
