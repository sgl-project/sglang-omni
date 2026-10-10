# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.easymagpie_tts.codec_graphs import StreamingCodecRunner


def stream_in_chunks(runner, slot, codes, sizes):
    pieces, start = [], 0
    for index, size in enumerate(sizes):
        piece = codes[start : start + size].unsqueeze(0)
        pieces.append(runner.decode(piece, [slot], [index > 0])[0])
        start += size
    return torch.cat(pieces)


def test_slot_histories_reproduce_the_whole_utterance_decode(codec) -> None:
    runner = StreamingCodecRunner(codec, max_streams=2)
    first, second = torch.randint(0, 16, (2, 9, 4))
    slot_a, slot_b = runner.acquire(), runner.acquire()

    # Interleave two streams so each step must read only its own slot.
    audio_a = runner.decode(first[:2].unsqueeze(0), [slot_a], [False])
    audio_b = runner.decode(second[:2].unsqueeze(0), [slot_b], [False])
    rest = runner.decode(
        torch.stack([first[2:], second[2:]]), [slot_a, slot_b], [True, True]
    )

    whole = codec.decode_batch([first, second])
    torch.testing.assert_close(torch.cat([audio_a[0], rest[0]]), whole[0])
    torch.testing.assert_close(torch.cat([audio_b[0], rest[1]]), whole[1])


def test_a_reused_slot_starts_from_an_empty_history(codec) -> None:
    runner = StreamingCodecRunner(codec, max_streams=1)
    slot = runner.acquire()
    stream_in_chunks(runner, slot, torch.randint(0, 16, (5, 4)), [2, 3])
    runner.release(slot)

    codes = torch.randint(0, 16, (7, 4))
    audio = stream_in_chunks(runner, runner.acquire(), codes, [2, 5])
    torch.testing.assert_close(audio, codec.decode_batch([codes])[0])


def test_acquire_fails_once_every_slot_is_taken(codec) -> None:
    runner = StreamingCodecRunner(codec, max_streams=1)
    runner.acquire()
    with pytest.raises(RuntimeError, match="no free codec slot"):
        runner.acquire()


def test_capture_is_a_no_op_off_cuda(codec) -> None:
    runner = StreamingCodecRunner(codec, max_streams=1)
    runner.capture([2, 8], 4)
    assert runner.graphs == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_graph_replays_match_eager_steps_with_padded_batches(codec) -> None:
    codec = codec.cuda()
    eager = StreamingCodecRunner(codec, max_streams=3)
    graphed = StreamingCodecRunner(codec, max_streams=3)
    graphed.capture([2, 6], 4)
    codes = torch.randint(0, 16, (3, 8, 4))

    def run(runner):
        slots = [runner.acquire() for _ in range(3)]
        first = runner.decode(codes[:, :2], slots, [False] * 3)
        return torch.cat([first, runner.decode(codes[:, 2:], slots, [True] * 3)], 1)

    assert (2, 4) in graphed.graphs and (6, 4) in graphed.graphs
    torch.testing.assert_close(run(graphed), run(eager))
