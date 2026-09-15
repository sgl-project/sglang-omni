# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import torch

from sglang_omni.models.breeze_tts import stages
from sglang_omni.models.breeze_tts.request_builders import stream_output
from sglang_omni.models.qwen3_tts import streaming_vocoder
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload


class Codec:
    total_upsample = 4

    def __call__(self, codes):
        return codes[:, :1].float().repeat_interleave(4, -1)

    def chunked_decode(self, codes):
        return self(codes)


class Tokenizer:
    def __init__(self):
        self.model = SimpleNamespace(decoder=Codec())

    def get_output_sample_rate(self):
        return 24000

    def decode(self, entries):
        return [
            entry["audio_codes"][:, 0].float().repeat_interleave(4).numpy()
            for entry in entries
        ], 24000


class IncrementalCodec:
    def __init__(self, decoder):
        self.decoder = decoder

    def decode(self, codes, state):
        state.frame_position += codes.shape[-1]
        return self.decoder(codes)


def make_vocoder(monkeypatch):
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda path: path)
    monkeypatch.setattr(stages, "load_audio_tokenizer", lambda *args: Tokenizer())
    monkeypatch.setattr(
        streaming_vocoder, "Qwen3TTSIncrementalDecoder", IncrementalCodec
    )
    return stages.create_vocoder_executor("unused", device="cpu")


def chunk(codes, index):
    data = SimpleNamespace(
        is_cfg_uncond=False,
        generation=SimpleNamespace(pending_chunk=codes[0]),
        stage_payload=SimpleNamespace(
            request=OmniRequest(inputs="Hello", params={"stream": True})
        ),
    )
    message = next(stream_output("request", data, None))
    return StreamItem(
        chunk_id=index,
        data=message.data,
        from_stage="tts_engine",
        metadata=message.metadata,
    )


def test_codec_emits_early_and_flushes_short_final_chunk(monkeypatch):
    scheduler = make_vocoder(monkeypatch)
    codes = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])
    # Engine streams arrive before the terminal payload. Two frames emit
    # immediately; the one-frame tail must survive stream_done.
    for index in range(3):
        scheduler._on_chunk("request", chunk(codes[index : index + 1], index))
    assert scheduler.outbox.qsize() > 0
    payload = StagePayload(
        request_id="request",
        request=OmniRequest(inputs="Hello", params={"stream": True}),
        data={"audio_codes": codes, "sample_rate": 24000, "ref_code_len": 0},
    )
    scheduler._on_streaming_new_request("request", payload)
    scheduler._on_done("request")
    messages = []
    while not scheduler.outbox.empty():
        messages.append(scheduler.outbox.get_nowait())
    audio = np.concatenate(
        [
            np.frombuffer(m.data["audio_waveform"], dtype=np.float32)
            for m in messages
            if m.type == "stream"
        ]
    )
    np.testing.assert_array_equal(audio, np.repeat([1.0, 2.0, 3.0], 4))
    assert sum(m.type == "result" for m in messages) == 1
    assert "request" not in scheduler._stream_states


def test_codec_abort_does_not_resurrect_on_late_frames(monkeypatch):
    scheduler = make_vocoder(monkeypatch)
    scheduler._on_chunk("aborted", chunk(torch.ones(1, 4, dtype=torch.long), 0))
    scheduler.abort("aborted")
    scheduler._on_chunk("aborted", chunk(torch.ones(1, 4, dtype=torch.long), 1))
    scheduler._on_done("aborted")
    assert "aborted" not in scheduler._stream_states
    assert scheduler.outbox.empty()
