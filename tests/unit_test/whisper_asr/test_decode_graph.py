# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from torch import nn
from transformers import WhisperConfig

from sglang_omni.models.whisper_asr import decode_graph, engine_builder, sglang_model
from sglang_omni.models.whisper_asr.decode_graph import WhisperDecodeGraphRunner
from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.cuda import CUDAOmniPlatform
from sglang_omni.platforms.device_graph import XpuDeviceGraphBackend
from sglang_omni.platforms.xpu import XPUOmniPlatform
from tests.unit_test.fixtures.cuda_graph_server_args import cuda_graph_server_args

ENCODER_TOKENS = 8
DECODER_TOKENS = 16
NUM_KV_SLOTS = 160
VOCAB_SIZE = 32


class LastTokenLogits(nn.Module):

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        lm_head: nn.Embedding,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        return LogitsProcessorOutput(next_token_logits=hidden_states @ lm_head.weight.T)


class KVBuffers:

    def __init__(
        self,
        num_layers: int,
        config: WhisperConfig,
        device: str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        heads = config.decoder_attention_heads
        shape = (NUM_KV_SLOTS, heads, config.d_model // heads)
        self.key_buffers = [
            torch.randn(shape, device=device, dtype=dtype) for _ in range(num_layers)
        ]
        self.value_buffers = [
            torch.randn(shape, device=device, dtype=dtype) for _ in range(num_layers)
        ]

    def get_kv_buffer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key_buffers[layer_id], self.value_buffers[layer_id]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        location: KVWriteLoc,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self.key_buffers[layer.layer_id][location.loc] = key
        self.value_buffers[layer.layer_id][location.loc] = value

    def snapshot(self) -> list[torch.Tensor]:
        return [buffer.clone() for buffer in self.key_buffers + self.value_buffers]

    def restore(self, snapshot: list[torch.Tensor]) -> None:
        for buffer, saved in zip(
            self.key_buffers + self.value_buffers, snapshot, strict=True
        ):
            buffer.copy_(saved)


class RequestSlotAttention(AttentionBackend):

    def __init__(self, token_to_kv_pool: KVBuffers, req_to_token: torch.Tensor) -> None:
        super().__init__()
        self.token_to_kv_pool = token_to_kv_pool
        self.req_to_token = req_to_token

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        key_cache, value_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        if layer.is_cross_attention:
            pass
        else:
            self.token_to_kv_pool.set_kv_buffer(
                layer, KVWriteLoc(forward_batch.out_cache_loc), k, v
            )
        attention_outputs = []
        for row, request_pool_index in enumerate(forward_batch.req_pool_indices):
            request_slots = self.req_to_token[int(request_pool_index)].long()
            if layer.is_cross_attention:
                slots = request_slots[:ENCODER_TOKENS]
            else:
                slots = request_slots[
                    ENCODER_TOKENS : ENCODER_TOKENS + int(forward_batch.seq_lens[row])
                ]
            query = q[row].view(layer.tp_q_head_num, layer.head_dim)
            scores = torch.einsum("hd,shd->hs", query, key_cache[slots]) * layer.scaling
            attention_outputs.append(
                torch.einsum("hs,shd->hd", scores.softmax(dim=-1), value_cache[slots])
            )
        return torch.stack(attention_outputs).reshape(q.shape[0], -1)


def decoder_runner(
    monkeypatch: pytest.MonkeyPatch, device: str, dtype: torch.dtype = torch.float32
) -> SimpleNamespace:
    monkeypatch.setattr(
        sglang_model, "LogitsProcessor", lambda config: LastTokenLogits()
    )
    config = WhisperConfig(
        d_model=16,
        encoder_layers=1,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        vocab_size=VOCAB_SIZE,
        max_target_positions=DECODER_TOKENS,
        max_source_positions=ENCODER_TOKENS,
    )
    model = (
        sglang_model.WhisperForConditionalGeneration(config)
        .to(device=device, dtype=dtype)
        .eval()
    )
    num_rows = 5
    row_width = ENCODER_TOKENS + DECODER_TOKENS
    req_to_token = torch.zeros(num_rows, row_width, dtype=torch.int32, device=device)
    request_slots = torch.randperm(NUM_KV_SLOTS - 1, device=device) + 1
    for row in range(1, num_rows):
        req_to_token[row] = request_slots[(row - 1) * row_width : row * row_width]
    return SimpleNamespace(
        model=model,
        token_to_kv_pool=KVBuffers(2 * config.decoder_layers, config, device, dtype),
        req_to_token=req_to_token,
    )


def decode_batch(
    rows: list[int], decoder_lengths: list[int], runner: SimpleNamespace
) -> ForwardBatch:
    device = runner.req_to_token.device
    seq_lens = torch.tensor(decoder_lengths, device=device)
    req_pool_indices = torch.tensor(rows, device=device)
    return ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=len(rows),
        input_ids=torch.randint(VOCAB_SIZE, (len(rows),), device=device),
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        out_cache_loc=runner.req_to_token[
            req_pool_indices, ENCODER_TOKENS + seq_lens - 1
        ].long(),
        seq_lens_sum=sum(decoder_lengths),
        encoder_lens=torch.full((len(rows),), ENCODER_TOKENS, device=device),
        encoder_lens_cpu=[ENCODER_TOKENS] * len(rows),
        positions=seq_lens - 1,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )


def graph_runner(
    runner: SimpleNamespace,
    captured_request_counts: tuple[int, ...],
    allow_padding: bool = True,
) -> WhisperDecodeGraphRunner:
    return WhisperDecodeGraphRunner(
        runner.model,
        request_token_slots=runner.req_to_token,
        token_to_kv_pool=runner.token_to_kv_pool,
        graph_backend=XpuDeviceGraphBackend(),
        captured_request_counts=captured_request_counts,
        allow_padding=allow_padding,
        encoder_token_count=ENCODER_TOKENS,
        decoder_token_count=DECODER_TOKENS,
    )


def eager_logits(runner: SimpleNamespace, forward_batch: ForwardBatch) -> torch.Tensor:
    with forward_context(
        ForwardContext(
            attn_backend=RequestSlotAttention(
                runner.token_to_kv_pool, runner.req_to_token
            )
        ),
        forward_batch=forward_batch,
    ):
        return runner.model.forward(
            forward_batch.input_ids, forward_batch.positions, forward_batch
        ).next_token_logits


@torch.no_grad()
def test_a_padded_batch_decodes_each_request_through_its_own_slots_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    monkeypatch.setattr(decode_graph, "REQUESTS_PER_ATTENTION_CALL", 2)
    runner = decoder_runner(monkeypatch, "cpu")
    pool = runner.token_to_kv_pool
    rows, decoder_lengths = [1, 3, 4], [3, 9, 1]
    for row, decoder_length in zip(rows, decoder_lengths, strict=True):
        stale_slots = runner.req_to_token[row, ENCODER_TOKENS + decoder_length :].long()
        for buffer in pool.key_buffers + pool.value_buffers:
            buffer[stale_slots] = float("nan")
    forward_batch = decode_batch(rows, decoder_lengths, runner)
    start = pool.snapshot()
    expected_logits = eager_logits(runner, forward_batch)
    expected_buffers = pool.snapshot()

    pool.restore(start)
    graphs = graph_runner(runner, (1, 4))
    graphs.logits[4] = torch.empty(4, VOCAB_SIZE)
    graphs.graphs[4] = SimpleNamespace(
        replay=lambda: graphs.logits[4].copy_(graphs.decode_step(4))
    )
    replayed_logits = graphs.replay(forward_batch).next_token_logits

    torch.testing.assert_close(replayed_logits, expected_logits)
    for replayed_buffer, expected_buffer in zip(
        pool.snapshot(), expected_buffers, strict=True
    ):
        torch.testing.assert_close(
            replayed_buffer[1:], expected_buffer[1:], equal_nan=True
        )


@torch.no_grad()
def test_a_shrunken_batch_leaves_the_finished_requests_last_slot_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    monkeypatch.setattr(decode_graph, "REQUESTS_PER_ATTENTION_CALL", 2)
    runner = decoder_runner(monkeypatch, "cpu")
    pool = runner.token_to_kv_pool
    graphs = graph_runner(runner, (1, 4))
    graphs.logits[4] = torch.empty(4, VOCAB_SIZE)
    graphs.graphs[4] = SimpleNamespace(
        replay=lambda: graphs.logits[4].copy_(graphs.decode_step(4))
    )
    graphs.replay(decode_batch([1, 2, 3, 4], [3, 9, 1, 5], runner))
    before = pool.snapshot()

    graphs.replay(decode_batch([1, 2, 3], [4, 10, 2], runner))

    finished_slot = int(runner.req_to_token[4, ENCODER_TOKENS + 5 - 1])
    for buffer, saved in zip(pool.snapshot(), before, strict=True):
        assert torch.equal(buffer[finished_slot], saved[finished_slot])


@pytest.mark.parametrize(
    ("batch_size", "can_run"),
    [(1, True), (3, False), (4, True)],
    ids=["captured_one", "would_pad", "captured_four"],
)
def test_without_padding_only_a_captured_request_count_replays(
    monkeypatch: pytest.MonkeyPatch, batch_size: int, can_run: bool
) -> None:
    graphs = graph_runner(
        decoder_runner(monkeypatch, "cpu"), (1, 4), allow_padding=False
    )
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        batch_size=batch_size,
        spec_info=None,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        encoder_lens_cpu=[ENCODER_TOKENS] * batch_size,
    )
    assert graphs.can_run(forward_batch) is can_run


@pytest.mark.parametrize(
    ("forward_mode", "batch_size", "encoder_length", "capture_hidden_mode", "can_run"),
    [
        (ForwardMode.DECODE, 3, ENCODER_TOKENS, CaptureHiddenMode.NULL, True),
        (ForwardMode.EXTEND, 3, ENCODER_TOKENS, CaptureHiddenMode.NULL, False),
        (ForwardMode.DECODE, 5, ENCODER_TOKENS, CaptureHiddenMode.NULL, False),
        (ForwardMode.DECODE, 3, ENCODER_TOKENS - 1, CaptureHiddenMode.NULL, False),
        (ForwardMode.DECODE, 3, ENCODER_TOKENS, CaptureHiddenMode.LAST, False),
    ],
    ids=[
        "decode",
        "extend",
        "more_requests_than_any_graph",
        "shorter_encoder",
        "hidden_states",
    ],
)
def test_only_an_ordinary_decode_the_graphs_can_hold_replays(
    monkeypatch: pytest.MonkeyPatch,
    forward_mode: ForwardMode,
    batch_size: int,
    encoder_length: int,
    capture_hidden_mode: CaptureHiddenMode,
    can_run: bool,
) -> None:
    graphs = graph_runner(decoder_runner(monkeypatch, "cpu"), (1, 4))
    forward_batch = SimpleNamespace(
        forward_mode=forward_mode,
        batch_size=batch_size,
        spec_info=None,
        capture_hidden_mode=capture_hidden_mode,
        encoder_lens_cpu=[ENCODER_TOKENS] * (batch_size - 1) + [encoder_length],
    )
    assert graphs.can_run(forward_batch) is can_run


@pytest.mark.parametrize(
    ("platform", "selectors", "kv_cache_dtype", "captured"),
    [
        (XPUOmniPlatform(), {}, torch.float16, True),
        (
            XPUOmniPlatform(),
            {"disable_decode_cuda_graph": True, "cuda_graph_backend_decode": None},
            torch.float16,
            False,
        ),
        (
            XPUOmniPlatform(),
            {"disable_cuda_graph": True, "cuda_graph_backend_decode": None},
            torch.float16,
            False,
        ),
        (
            XPUOmniPlatform(),
            {"cuda_graph_backend_decode": "disabled"},
            torch.float16,
            False,
        ),
        (
            XPUOmniPlatform(),
            {"cuda_graph_config": {"decode": {"backend": "disabled"}}},
            torch.float16,
            False,
        ),
        (
            XPUOmniPlatform(),
            {
                "cuda_graph_config": CudaGraphConfig(
                    decode=PhaseConfig(backend="disabled")
                )
            },
            torch.float16,
            False,
        ),
        (
            XPUOmniPlatform(),
            {
                "disable_decode_cuda_graph": True,
                "cuda_graph_backend_decode": None,
                "cuda_graph_config": {"decode": {"backend": "full"}},
            },
            torch.float16,
            True,
        ),
        (XPUOmniPlatform(), {"disable_cuda_graph_padding": True}, torch.float16, True),
        (XPUOmniPlatform(), {}, torch.float8_e4m3fn, False),
        (CUDAOmniPlatform(), {}, torch.float16, False),
    ],
    ids=[
        "xpu",
        "decode_graph_disabled",
        "all_graphs_disabled",
        "decode_backend_disabled",
        "structured_decode_disabled",
        "structured_config_object_disabled",
        "structured_backend_outranks_the_flag",
        "padding_disabled",
        "quantized_kv_cache",
        "cuda",
    ],
)
def test_the_decode_graphs_follow_sglangs_decode_batch_sizes_where_the_platform_owns_them(
    monkeypatch: pytest.MonkeyPatch,
    platform: XPUOmniPlatform | CUDAOmniPlatform,
    selectors: dict[str, object],
    kv_cache_dtype: torch.dtype,
    captured: bool,
) -> None:
    runner = decoder_runner(monkeypatch, "cpu")
    captures = []
    monkeypatch.setattr(engine_builder, "current_platform", platform)
    monkeypatch.setattr(
        platform, "get_device_graph_backend", lambda device: XpuDeviceGraphBackend()
    )
    monkeypatch.setattr(
        engine_builder,
        "get_batch_sizes_to_capture",
        lambda model_runner: ([1, 2, 4, 12], []),
    )
    monkeypatch.setattr(
        WhisperDecodeGraphRunner, "capture", lambda graphs: captures.append(graphs)
    )
    monkeypatch.setattr(torch, "get_device_module", lambda device: torch.cpu)
    monkeypatch.setattr(torch.cpu, "mem_get_info", lambda device: (0, 0), raising=False)
    builder = WhisperASREngineBuilder(
        encoder_attention_backend="auto",
        max_running_requests=12,
        max_new_tokens=8,
        mem_fraction_static=0.2,
    )
    builder.encoder_token_count = ENCODER_TOKENS
    builder.decoder_context_len = DECODER_TOKENS
    builder.sglang_runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=runner.req_to_token),
        token_to_kv_pool=runner.token_to_kv_pool,
        kv_cache_dtype=kv_cache_dtype,
        dtype=torch.float16,
    )

    builder.post_cuda_graph_setup(
        runner.model,
        cuda_graph_server_args(**selectors),
    )

    if captured:
        assert captures == [runner.model.decode_graph_runner]
        assert runner.model.decode_graph_runner.captured_request_counts == (
            1,
            2,
            4,
            12,
        )
        assert runner.model.decode_graph_runner.allow_padding is not selectors.get(
            "disable_cuda_graph_padding", False
        )
    else:
        assert captures == []
        assert runner.model.decode_graph_runner is None


@pytest.mark.skipif(not current_platform.is_xpu(), reason="captures an XPU graph")
@pytest.mark.parametrize(
    ("dtype", "reference_tolerance"),
    [
        (torch.float32, {}),
        (torch.float16, {"atol": 6e-2, "rtol": 3e-2}),
        (torch.bfloat16, {"atol": 6e-2, "rtol": 3e-2}),
    ],
    ids=["float32", "float16", "bfloat16"],
)
@torch.no_grad()
def test_replay_decodes_the_loaded_batch_as_the_step_runs_eagerly(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    reference_tolerance: dict[str, float],
) -> None:
    torch.manual_seed(0)
    runner = decoder_runner(monkeypatch, "xpu", dtype)
    pool = runner.token_to_kv_pool
    graphs = graph_runner(runner, (1, 4))
    graphs.capture()

    for rows, decoder_lengths, graph_request_count in (
        ([1, 2, 3], [4, 9, 2], 4),
        ([3], [5], 1),
        ([2, 4], [12, 1], 4),
        ([1, 2, 3, 4], [3, 3, 7, 16], 4),
    ):
        forward_batch = decode_batch(rows, decoder_lengths, runner)
        start = pool.snapshot()
        replayed_logits = graphs.replay(forward_batch).next_token_logits
        replayed_buffers = pool.snapshot()
        pool.restore(start)
        eager_step_logits = graphs.decode_step(graph_request_count)[: len(rows)]
        eager_buffers = pool.snapshot()
        pool.restore(start)
        reference_logits = eager_logits(runner, forward_batch)
        pool.restore(start)

        assert replayed_logits.dtype == dtype
        torch.testing.assert_close(replayed_logits, eager_step_logits)
        for replayed_buffer, eager_buffer in zip(
            replayed_buffers, eager_buffers, strict=True
        ):
            torch.testing.assert_close(replayed_buffer[1:], eager_buffer[1:])
        torch.testing.assert_close(
            replayed_logits, reference_logits.to(dtype), **reference_tolerance
        )
