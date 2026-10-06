# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import threading
from types import SimpleNamespace

import torch
from transformers import HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel

from sglang_omni.models.higgs_tts import audio_codec


def test_higgs_codec_uses_upstream_transformers_architecture() -> None:
    assert audio_codec.HiggsAudioV2TokenizerConfig is HiggsAudioV2TokenizerConfig
    assert audio_codec.HiggsAudioV2TokenizerModel is HiggsAudioV2TokenizerModel

    config = HiggsAudioV2TokenizerConfig.from_json_file(
        audio_codec._BUNDLED_CODEC_CONFIG_PATH  # noqa: leading-underscore  # production name
    )
    with torch.device("meta"):
        model = audio_codec.HiggsAudioV2TokenizerModel(config)

    state = model.state_dict()
    assert len(state) == 527
    assert {
        key: tuple(state[key].shape)
        for key in (
            "acoustic_encoder.block.0.conv1.weight",
            "acoustic_decoder.block.0.conv_t1.weight",
            "quantizer.quantizers.0.codebook.embed",
            "semantic_model.encoder.layers.0.attention.q_proj.weight",
        )
    } == {
        "acoustic_encoder.block.0.conv1.weight": (128, 64, 16),
        "acoustic_decoder.block.0.conv_t1.weight": (1024, 512, 16),
        "quantizer.quantizers.0.codebook.embed": (1024, 64),
        "semantic_model.encoder.layers.0.attention.q_proj.weight": (768, 768),
    }
    assert config.frame_rate == 25
    assert config.num_quantizers == 8


class FakeQuantizerLayer:
    def __init__(self, offset: float) -> None:
        self.offset = offset

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        return indices.to(torch.float32).unsqueeze(-1) + self.offset


def test_capture_safe_quantizer_decode_matches_additive_rvq() -> None:
    quantizer = SimpleNamespace(
        quantizers=[
            FakeQuantizerLayer(0.25),
            FakeQuantizerLayer(0.5),
            FakeQuantizerLayer(0.75),
        ]
    )
    codes = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long)

    actual = audio_codec.capture_safe_quantizer_decode(quantizer, codes)
    expected = sum(
        layer.decode(indices) for layer, indices in zip(quantizer.quantizers, codes)
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_codec_decode_replays_matching_shape_graph() -> None:
    graph_input = torch.zeros((1, 2, 3), dtype=torch.long)
    graph_output = torch.zeros((1, 1, 1), dtype=torch.float32)

    class FakeGraph:
        def __init__(self) -> None:
            self.replays = 0

        def replay(self) -> None:
            self.replays += 1
            graph_output.fill_(float(graph_input.sum()))

    graph = FakeGraph()
    codec = object.__new__(audio_codec.HiggsAudioCodec)
    codec.decode_cuda_graphs = {
        3: audio_codec.DecodeCudaGraph(
            graph=graph,
            input_codes=graph_input,
            output_audio=graph_output,
        )
    }
    codec.decode_cuda_graph_hits = 0
    codec.decode_cuda_graph_misses = 0
    codec.decode_cuda_graph_missed_shapes = set()
    codec.decode_single_flight_lock = threading.Lock()

    output = codec.decode(torch.tensor([[1, 2], [3, 4], [5, 6]]))

    assert graph.replays == 1
    assert codec.decode_cuda_graph_hits == 1
    assert codec.decode_cuda_graph_misses == 0
    torch.testing.assert_close(output, torch.tensor([21.0]), rtol=0, atol=0)


def test_codec_decode_serializes_concurrent_graph_pool_use() -> None:
    replay_started = threading.Event()
    release_replay = threading.Event()
    batch_started = threading.Event()
    failures: list[BaseException] = []
    completions: list[str] = []

    class BlockingGraph:
        def replay(self) -> None:
            replay_started.set()
            assert release_replay.wait(timeout=1)

    codec = object.__new__(audio_codec.HiggsAudioCodec)
    codec.decode_cuda_graphs = {
        1: audio_codec.DecodeCudaGraph(
            graph=BlockingGraph(),
            input_codes=torch.zeros((1, 2, 1), dtype=torch.long),
            output_audio=torch.zeros((1, 1, 1), dtype=torch.float32),
        )
    }
    codec.decode_cuda_graph_hits = 0
    codec.decode_cuda_graph_misses = 0
    codec.decode_cuda_graph_missed_shapes = set()
    codec.decode_single_flight_lock = threading.Lock()

    def replay_graph() -> None:
        try:
            codec.decode(torch.tensor([[1, 2]], dtype=torch.long))
            completions.append("decode")
        except BaseException as exc:
            failures.append(exc)

    def replay_batch() -> None:
        try:
            batch_started.set()
            codec.decode_batch([torch.tensor([[3, 4]], dtype=torch.long)])
            completions.append("batch")
        except BaseException as exc:
            failures.append(exc)

    replay_thread = threading.Thread(target=replay_graph)
    replay_thread.start()
    assert replay_started.wait(timeout=1)
    batch_thread = threading.Thread(target=replay_batch)
    batch_thread.start()
    try:
        assert batch_started.wait(timeout=1)
        batch_thread.join(timeout=0.05)
        assert batch_thread.is_alive()
    finally:
        release_replay.set()
        replay_thread.join(timeout=1)
        batch_thread.join(timeout=1)

    assert not replay_thread.is_alive()
    assert not batch_thread.is_alive()
    assert failures == []
    assert completions == ["decode", "batch"]


def test_codec_capture_serializes_with_decode() -> None:
    capture_entered = threading.Event()
    codec = object.__new__(audio_codec.HiggsAudioCodec)
    codec.decode_single_flight_lock = threading.Lock()
    codec.capture_decode_cuda_graphs_locked = (
        lambda _frame_counts: capture_entered.set()
    )

    codec.decode_single_flight_lock.acquire()
    capture_thread = threading.Thread(
        target=lambda: codec.capture_decode_cuda_graphs((1,))
    )
    capture_thread.start()
    try:
        assert not capture_entered.wait(timeout=0.05)
    finally:
        codec.decode_single_flight_lock.release()
        capture_thread.join(timeout=1)

    assert not capture_thread.is_alive()
    assert capture_entered.is_set()


def test_codec_capture_records_through_the_platform_backend(monkeypatch) -> None:
    class FakeStream:
        def wait_stream(self, other) -> None:
            pass

        def synchronize(self) -> None:
            pass

    class FakeBackend:
        def __init__(self) -> None:
            self.captures: list[tuple[object, object, object]] = []

        @contextlib.contextmanager
        def capture(self, *, pool=None, stream=None, thread_local_errors=False):
            graph = object()
            self.captures.append((pool, stream, graph))
            yield graph

    backend = FakeBackend()
    capture_stream = FakeStream()
    pool = object()
    device_module = SimpleNamespace(
        current_stream=lambda device: FakeStream(),
        Stream=lambda device: capture_stream,
        device=lambda device: contextlib.nullcontext(),
        stream=lambda stream: contextlib.nullcontext(),
        graph_pool_handle=lambda: pool,
        synchronize=lambda device: None,
    )
    monkeypatch.setattr(
        audio_codec,
        "current_platform",
        SimpleNamespace(
            device_type="cpu", get_device_graph_backend=lambda device: backend
        ),
    )
    monkeypatch.setattr(torch, "get_device_module", lambda device: device_module)

    quantizer_decode = object()
    codec = object.__new__(audio_codec.HiggsAudioCodec)
    codec.model = SimpleNamespace(
        config=SimpleNamespace(num_quantizers=8),
        quantizer=SimpleNamespace(decode=quantizer_decode),
        decode=lambda codes: SimpleNamespace(audio_values=codes),
    )
    codec.device = torch.device("cpu")
    codec.decode_single_flight_lock = threading.Lock()

    codec.capture_decode_cuda_graphs((1, 2))

    assert [(p, s) for p, s, _ in backend.captures] == [(pool, capture_stream)] * 2
    assert {
        frame_count: graph.graph
        for frame_count, graph in codec.decode_cuda_graphs.items()
    } == {2: backend.captures[0][2], 1: backend.captures[1][2]}
    assert codec.model.quantizer.decode is quantizer_decode
