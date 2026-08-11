import threading
from types import SimpleNamespace

import torch

from sglang_omni.models.dots_tts.vocoder import DotsTTSStreamingVocoder


class _RecordingInference:
    def __init__(self) -> None:
        self.stream_calls: list[dict] = []

    def init_stream_state(self, *, batch_size: int, chunk_size: int):
        return SimpleNamespace(
            lstm_hidden=(
                torch.zeros(1, batch_size, 2),
                torch.zeros(1, batch_size, 2),
            ),
            decoder=SimpleNamespace(
                window=torch.zeros(batch_size, 5, 8),
                chunk_size=chunk_size,
                total_frames=0,
                emitted_frames=0,
            ),
        )

    def stream_step(self, latents, *, stream_state, optimize, use_compiled, **_):
        self.stream_calls.append(
            {
                "batch": int(latents.shape[0]),
                "frames": int(latents.shape[1]),
                "data_ptr": latents.data_ptr(),
                "optimize": optimize,
                "use_compiled": use_compiled,
            }
        )
        stream_state.lstm_hidden = tuple(
            hidden + 1 for hidden in stream_state.lstm_hidden
        )
        stream_state.decoder.window = stream_state.decoder.window + 1
        stream_state.decoder.total_frames += int(latents.shape[1])
        stream_state.decoder.emitted_frames += int(latents.shape[1])
        return torch.zeros(latents.shape[0], 1, 8)

    def flush(self, stream_state):
        del stream_state
        return torch.zeros(1, 4)


class _FakeCodec:
    def __init__(self) -> None:
        self.inference = _RecordingInference()
        self.lock = threading.Lock()
        self.sample_rate = 48000
        self.patch_size = 3
        self.latent_dim = 5
        self.device = "cpu"


def test_streaming_never_uses_the_compiled_stream_step() -> None:
    codec = _FakeCodec()
    vocoder = DotsTTSStreamingVocoder(codec, optimize=True, merge_steps=2)
    state = vocoder.create_stream_state("req")

    for _ in range(4):
        vocoder.ingest("req", state, torch.zeros(1, 3, 5))
        if vocoder.should_decode(state, is_final=False):
            vocoder.decode_delta("req", state, is_final=False)
    vocoder.decode_delta("req", state, is_final=True)

    assert codec.inference.stream_calls, "streaming produced no vocoder steps"
    assert all(not call["use_compiled"] for call in codec.inference.stream_calls)
    assert all(call["optimize"] for call in codec.inference.stream_calls)


def test_streaming_single_patch_reuses_input_storage() -> None:
    codec = _FakeCodec()
    vocoder = DotsTTSStreamingVocoder(codec, optimize=True, merge_steps=2)
    state = vocoder.create_stream_state("req")
    patch = torch.zeros(1, 3, 5)

    vocoder.ingest("req", state, patch)
    vocoder.decode_delta("req", state, is_final=False)

    assert codec.inference.stream_calls[0]["data_ptr"] == patch.data_ptr()


def test_streaming_batches_compatible_request_states() -> None:
    codec = _FakeCodec()
    vocoder = DotsTTSStreamingVocoder(
        codec, optimize=True, merge_steps=2, max_batch_size=4
    )
    states = {key: vocoder.create_stream_state(key) for key in ("a", "b")}
    vocoder._stream_states.update(states)
    for request_id, state in states.items():
        vocoder.ingest(request_id, state, torch.zeros(1, 3, 5))

    participants = vocoder.select_step_participants()
    plan = vocoder.build_step_plan(participants)
    outputs = vocoder.run_step(participants, plan)

    [call] = codec.inference.stream_calls
    assert call["batch"] == 2
    assert call["frames"] == 3
    assert call["optimize"] is True
    assert call["use_compiled"] is False
    assert set(outputs) == {"a", "b"}
    assert all(not state.pending for state in states.values())
    assert all(state.codec_state.decoder.total_frames == 3 for state in states.values())
    assert all(
        torch.count_nonzero(state.codec_state.decoder.window) > 0
        for state in states.values()
    )
