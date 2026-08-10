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


class _RecordingGraphs:
    def __init__(self) -> None:
        self.frames: list[int] = []

    def decode(self, latents, state):
        del state
        self.frames.append(int(latents.shape[1]))
        return torch.zeros(1, 8)

    def __len__(self) -> int:
        return 1


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


def test_streaming_audio_vae_cuda_graph_is_explicit_opt_in() -> None:
    default_vocoder = DotsTTSStreamingVocoder(_FakeCodec(), optimize=True)
    graph_vocoder = DotsTTSStreamingVocoder(
        _FakeCodec(),
        optimize=True,
        enable_streaming_audio_vae_cuda_graph=True,
    )

    assert default_vocoder._cuda_graphs is None
    assert graph_vocoder._cuda_graphs is not None


def test_streaming_prefers_native_cuda_graph() -> None:
    codec = _FakeCodec()
    vocoder = DotsTTSStreamingVocoder(codec, optimize=True, merge_steps=2)
    graphs = _RecordingGraphs()
    vocoder._cuda_graphs = graphs
    state = vocoder.create_stream_state("req")

    vocoder.ingest("req", state, torch.zeros(1, 3, 5))
    vocoder.decode_delta("req", state, is_final=False)

    assert graphs.frames == [3]
    assert not codec.inference.stream_calls
