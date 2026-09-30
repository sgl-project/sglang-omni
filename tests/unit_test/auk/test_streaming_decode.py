# SPDX-License-Identifier: Apache-2.0
"""Scheduler IO and cancellation, with a controllable incremental decoder."""

import asyncio
import threading
from unittest.mock import Mock

import torch

from sglang_omni.models.auk.payload_types import AuKState
from sglang_omni.models.auk.streaming_decode import AuKDecodeScheduler
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage


def message(request_id="s", stream=True):
    return IncomingMessage(
        request_id=request_id,
        type="new_request",
        data=StagePayload(
            request_id=request_id,
            request=OmniRequest(inputs="hello", params={"stream": stream}),
            data=AuKState(latent=torch.zeros(5, 64), gen_frames=5).to_dict(),
        ),
    )


class Decoder:
    def __init__(self):
        self.steps = 0
        self.closed = False
        self.after_chunk = lambda: None
        self.fail = None

    def iter_decode_chunks(self, latent, size):
        try:
            for index in range(3):
                if index == 1 and self.fail:
                    raise RuntimeError("decode failed")
                self.steps += 1
                yield torch.full((1, 1, 2 if index < 2 else 1), float(index))
                self.after_chunk()
        finally:
            self.closed = True


def scheduler(decoder, chunk_frames=2):
    return AuKDecodeScheduler(
        lambda payloads: payloads,
        max_batch_size=4,
        vae=decoder,
        device=torch.device("cpu"),
        chunk_frames=chunk_frames,
    )


def run(sched, batch):
    loop = asyncio.new_event_loop()
    try:
        sched._run_batch(batch, loop)
    finally:
        loop.close()
    output = []
    while not sched.outbox.empty():
        output.append(sched.outbox.get_nowait())
    return output


def test_audio_precedes_completion_without_duplicate_waveform():
    decoder = Decoder()
    sched = scheduler(decoder)

    # Outbox is populated before the generator advances to the next chunk.
    def check_emission():
        assert sched.outbox.qsize() == decoder.steps

    decoder.after_chunk = check_emission
    msg = message()
    outputs = run(sched, [msg])
    assert [item.type for item in outputs] == ["stream"] * 3 + ["result"]
    audio = torch.cat(
        [
            torch.frombuffer(
                bytearray(item.data["audio_waveform"]), dtype=torch.float32
            )
            for item in outputs[:-1]
        ]
    )
    torch.testing.assert_close(audio, torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0]))
    assert all(item.metadata == {"modality": "audio"} for item in outputs[:-1])
    assert "audio_waveform" not in outputs[-1].data.data
    assert "latent" not in msg.data.data
    assert decoder.closed


def test_abort_after_first_emission_stops_before_next_decode():
    decoder = Decoder()
    sched = scheduler(decoder)
    original_put = sched.outbox.put

    def emit_then_abort(item):
        original_put(item)
        sched.abort(item.request_id)

    sched.outbox.put = emit_then_abort
    msg = message()
    outputs = run(sched, [msg])
    assert [item.type for item in outputs] == ["stream"]
    assert decoder.steps == 1 and decoder.closed
    assert "latent" not in msg.data.data


def test_failure_errors_only_its_request_and_next_request_works():
    decoder = Decoder()
    decoder.fail = True
    sched = scheduler(decoder)
    outputs = run(sched, [message("a", False), message("b"), message("c", False)])
    assert [(o.request_id, o.type) for o in outputs] == [
        ("a", "result"),
        ("b", "stream"),
        ("b", "error"),
        ("c", "result"),
    ]
    assert decoder.closed


def test_nonstream_batch_uses_existing_batch_callable():
    sched = scheduler(Decoder())
    sched._batch_fn = Mock(side_effect=lambda payloads: payloads)
    outputs = run(sched, [message("a", False), message("b", False)])
    sched._batch_fn.assert_called_once()
    assert [o.type for o in outputs] == ["result", "result"]


def test_streaming_disabled_reports_error():
    decoder = Decoder()
    outputs = run(scheduler(decoder, 0), [message()])
    assert [o.type for o in outputs] == ["error"]
    assert "chunk_frames" in str(outputs[0].data)
    assert decoder.steps == 0


def test_chunks_follow_existing_client_pcm_protocol():
    from sglang_omni.client.audio import encode_pcm, select_audio_delta
    from sglang_omni.client.client import Client
    from sglang_omni.proto import StreamMessage

    outputs = run(scheduler(Decoder()), [message()])
    emitted = 0
    pcm = []
    for item in outputs[:-1]:
        chunk = Client._default_stream_builder(
            "s",
            StreamMessage(
                request_id="s",
                from_stage="decode",
                chunk=item.data,
                modality="audio",
            ),
        )
        assert chunk.sample_rate == 24000 and chunk.finish_reason is None
        delta, emitted = select_audio_delta(
            chunk.audio_data,
            emitted_samples=emitted,
            is_terminal=False,
        )
        pcm.append(encode_pcm(delta, chunk.sample_rate))
    assert emitted == 5
    assert b"".join(pcm) == encode_pcm(
        torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0]).numpy(), 24000
    )


def test_abort_before_dispatch_does_no_decode():
    decoder = Decoder()
    sched = scheduler(decoder)
    sched.abort("s")
    assert run(sched, [message()]) == []
    assert decoder.steps == 0


def test_stop_after_first_chunk_closes_decoder():
    decoder = Decoder()
    sched = scheduler(decoder)
    put = sched.outbox.put

    def stop_after_put(item):
        put(item)
        sched.stop()

    sched.outbox.put = stop_after_put
    assert [o.type for o in run(sched, [message()])] == ["stream"]
    assert decoder.steps == 1 and decoder.closed


def test_real_worker_thread_emits_and_completes():
    decoder = Decoder()
    sched = scheduler(decoder)
    worker = threading.Thread(target=sched.start, daemon=True)
    worker.start()
    try:
        sched.inbox.put(message())
        outputs = [sched.outbox.get(timeout=3) for _ in range(4)]
        assert [o.type for o in outputs] == ["stream"] * 3 + ["result"]
    finally:
        sched.stop()
        worker.join(timeout=3)
    assert not worker.is_alive() and decoder.closed
