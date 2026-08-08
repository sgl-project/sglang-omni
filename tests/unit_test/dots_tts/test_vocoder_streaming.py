import threading

import torch

from sglang_omni.models.dots_tts.vocoder import DotsTTSStreamingVocoder


class _RecordingInference:
    def __init__(self) -> None:
        self.stream_calls: list[dict] = []

    def init_stream_state(self, *, batch_size: int, chunk_size: int):
        del batch_size, chunk_size
        return object()

    def stream_step(self, latents, *, stream_state, optimize, use_compiled, **_):
        del stream_state
        self.stream_calls.append(
            {
                "frames": int(latents.shape[1]),
                "optimize": optimize,
                "use_compiled": use_compiled,
            }
        )
        return torch.zeros(1, 8)

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
