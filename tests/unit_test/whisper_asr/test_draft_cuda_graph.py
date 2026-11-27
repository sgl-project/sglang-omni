# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from torch import nn
from transformers import WhisperConfig

from sglang_omni.models.whisper_asr import sglang_model
from sglang_omni.models.whisper_asr.draft_cuda_graph import (
    DraftDecodeGraph,
    draft_capture_stream,
)
from sglang_omni.platforms import current_platform

speculative = pytest.importorskip("sglang_omni.models.whisper_asr.speculative")

NUM_DECODE_STEPS = 3
NUM_KV_SLOTS = 128
# note (Tz22z): in the draft's vocabulary of 32, tokens below 20 are text and tokens from 28 are timestamps.
TIMESTAMP_PARAMS = {
    "segment_timestamps": True,
    "timestamp_begin_id": 28,
    "no_timestamps_token_id": 27,
    "eos_token_id": 20,
    "max_initial_timestamp_index": None,
}


class LastTokenLogits(nn.Module):
    """The draft's logits with no LM-head parallelism, which needs a running server."""

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        lm_head: nn.Embedding,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        return LogitsProcessorOutput(next_token_logits=hidden_states @ lm_head.weight.T)


class KVBuffers:
    """A token-to-KV pool of one K and one V buffer per attention layer."""

    def __init__(self, num_layers: int, config: WhisperConfig, device: str) -> None:
        heads = config.decoder_attention_heads
        shape = (NUM_KV_SLOTS, heads, config.d_model // heads)
        self.key_buffers = [
            torch.randn(shape, device=device) for _ in range(num_layers)
        ]
        self.value_buffers = [
            torch.randn(shape, device=device) for _ in range(num_layers)
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
    """Attention over exactly each request's encoder slots or its first seq_len decoder slots."""

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
            encoder_length = int(forward_batch.encoder_lens[row])
            if layer.is_cross_attention:
                slots = request_slots[:encoder_length]
            else:
                slots = request_slots[
                    encoder_length : encoder_length + int(forward_batch.seq_lens[row])
                ]
            query = q[row].view(layer.tp_q_head_num, layer.head_dim)
            scores = torch.einsum("hd,shd->hs", query, key_cache[slots]) * layer.scaling
            attention_outputs.append(
                torch.einsum("hs,shd->hd", scores.softmax(dim=-1), value_cache[slots])
            )
        return torch.stack(attention_outputs).reshape(q.shape[0], -1)


def draft_runner(monkeypatch: pytest.MonkeyPatch, device: str) -> SimpleNamespace:
    """A runner of a two-layer Whisper decoder whose request rows map columns to scattered KV slots, none shared and none the padding slot 0."""
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
        vocab_size=32,
        max_target_positions=16,
        max_source_positions=8,
    )
    model = sglang_model.WhisperForConditionalGeneration(config).to(device).eval()
    num_rows = 4
    req_to_token = torch.zeros(
        num_rows,
        config.max_source_positions + config.max_target_positions,
        dtype=torch.int32,
        device=device,
    )
    request_slots = torch.randperm(NUM_KV_SLOTS - 1, device=device) + 1
    row_width = req_to_token.shape[1]
    for row in range(1, num_rows):
        req_to_token[row] = request_slots[(row - 1) * row_width : row * row_width]
    return SimpleNamespace(
        model=model,
        token_to_kv_pool=KVBuffers(2 * config.decoder_layers, config, device),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
    )


def suppression_bias(
    num_requests: int, device: str, *, timestamp_requests: tuple[int, ...] = ()
) -> torch.Tensor:
    """A Whisper-style logit bias per request that leaves a few tokens open, so a step that skips it drafts another token; timestamp_requests open four text tokens and, 50 below them, timestamp 29, so plain argmax drafts text where the timestamp rules leave only that timestamp."""
    logit_bias = torch.full((num_requests, 32), -1e9, device=device)
    for row in range(num_requests):
        if row in timestamp_requests:
            logit_bias[row, torch.randperm(20, device=device)[:4]] = 0
            logit_bias[row, 29] = -50.0
        else:
            logit_bias[row, torch.randperm(32, device=device)[:4]] = 0
    return logit_bias


def round_batch(
    rows: list[int],
    encoder_lengths: list[int],
    committed_lengths: list[int],
    device: str,
    *,
    logit_bias: torch.Tensor | None = None,
    is_greedy: bool = True,
    timestamp_requests: tuple[int, ...] = (),
) -> SimpleNamespace:
    """A decode round of one request per request pool row, each with its own lengths and the same sampling; timestamp_requests are the places in the round of the requests with segment timestamps."""
    num_requests = len(rows)
    seq_lens = torch.tensor(committed_lengths, device=device)
    return SimpleNamespace(
        req_pool_indices=torch.tensor(rows, device=device),
        encoder_lens=torch.tensor(encoder_lengths, device=device),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.cpu(),
        spec_info=SimpleNamespace(
            bonus_tokens=torch.full((num_requests,), 3, device=device)
        ),
        reqs=[
            SimpleNamespace(
                sampling_params=SimpleNamespace(
                    custom_params=(
                        TIMESTAMP_PARAMS if request in timestamp_requests else None
                    )
                ),
                output_ids=[3],
            )
            for request in range(num_requests)
        ],
        sampling_info=SimpleNamespace(
            logit_bias=logit_bias,
            is_all_greedy=is_greedy,
            top_ks=torch.full((num_requests,), 1 if is_greedy else TOP_K_ALL),
            temperatures=torch.ones(num_requests, 1),
        ),
    )


def round_worker(
    runner: SimpleNamespace,
    chain_logits: torch.Tensor,
    verified_token_ids: list[torch.Tensor],
) -> speculative.WhisperSpeculativeWorker:
    """A worker drafting NUM_DECODE_STEPS + 1 tokens whose chain forward returns chain_logits and whose verify records the token ids it checks."""
    num_requests = chain_logits.shape[0] // 2
    # note (Tz22z): the target prefers timestamp 31 over every bias, and the rules close it to a row only while the row's last two tokens are timestamps, so the tokens verify commits tell which rule state it holds for each row.
    target_logits = torch.zeros(num_requests * (NUM_DECODE_STEPS + 2), 32)
    target_logits[:, 31] = 2e9
    worker = object.__new__(speculative.WhisperSpeculativeWorker)
    worker.device = "cpu"
    worker.speculative_num_steps = NUM_DECODE_STEPS + 1
    worker.previous_token_ids = torch.zeros(4, dtype=torch.int64)
    worker.req_to_token_pool = runner.req_to_token_pool
    worker.draft_runner = runner
    worker.draft_graphs = []
    worker.timestamp_draft_graphs = []
    worker.chain_forward = lambda batch, runner, **chain: chain_logits
    worker.chain_batch = lambda batch, runner, *, token_ids, **chain: (
        verified_token_ids.append(token_ids)
    )
    worker._target_worker = SimpleNamespace(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        model_runner=None,
        forward_batch_generation=lambda batch, **verify: SimpleNamespace(
            logits_output=SimpleNamespace(next_token_logits=target_logits),
            can_run_cuda_graph=True,
        ),
    )
    return worker


@pytest.mark.parametrize(
    ("rows", "graph_batch_size", "timestamp_requests"),
    [([2], 1, ()), ([1, 2, 3], 4, ()), ([2], 1, (0,)), ([1, 2, 3], 4, (0, 2))],
    ids=[
        "lone_request",
        "three_requests_in_a_graph_of_four",
        "lone_request_with_timestamps",
        "two_of_three_requests_with_timestamps_in_a_graph_of_four",
    ],
)
@torch.no_grad()
def test_a_greedy_round_drafts_through_the_graph_steps_as_through_the_step_loop(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[int],
    graph_batch_size: int,
    timestamp_requests: tuple[int, ...],
) -> None:
    """The graph's steps draft the tokens and write the K/V of the worker's step-by-step draft loop, whose offsets and KV slots they must follow for every request, under the timestamp rules for the requests that have them and by plain argmax for the others. Verify then commits the same tokens, so it got the same rule state for every draft."""
    torch.manual_seed(0)
    runner = draft_runner(monkeypatch, "cpu")
    pool = runner.token_to_kv_pool
    reference_attention = RequestSlotAttention(
        pool, runner.req_to_token_pool.req_to_token
    )

    def forward(forward_batch: ForwardBatch) -> SimpleNamespace:
        with forward_context(
            ForwardContext(attn_backend=reference_attention),
            forward_batch=forward_batch,
        ):
            return SimpleNamespace(
                logits_output=runner.model.forward(
                    forward_batch.input_ids, forward_batch.positions, forward_batch
                )
            )

    runner.forward = forward
    # note (Haoling Pu): SGLang's decode batch puts each token at position seq_lens - 1.
    monkeypatch.setattr(
        speculative.ForwardBatch,
        "init_new",
        lambda step, model_runner, **options: ForwardBatch(
            forward_mode=step.forward_mode,
            batch_size=step.req_pool_indices.numel(),
            input_ids=step.input_ids,
            req_pool_indices=step.req_pool_indices,
            seq_lens=step.seq_lens,
            out_cache_loc=step.out_cache_loc,
            seq_lens_sum=step.seq_lens_sum,
            encoder_lens=step.encoder_lens,
            positions=step.seq_lens - 1,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        ),
    )
    num_requests = len(rows)
    logit_bias = suppression_bias(
        num_requests, "cpu", timestamp_requests=timestamp_requests
    )
    chain_logits = torch.zeros(2 * num_requests, 32)
    # note (Haoling Pu): each first draft is a token its request's bias closes, so no later draft repeats it and a step fed the wrong token writes other K/V.
    # note (Tz22z): a request with timestamps drafts timestamp 28 first, which closes text to its next draft and every timestamp to the one after, so a step that skips the rules, or reads them from the first draft's tokens again, drafts another token.
    for row in range(num_requests):
        first_draft = (
            28 if row in timestamp_requests else int((logit_bias[row] < 0).nonzero()[0])
        )
        chain_logits[2 * row + 1, first_draft] = 2e9
    batch = round_batch(
        rows,
        [5, 8, 3][:num_requests],
        [4, 9, 2][:num_requests],
        "cpu",
        logit_bias=logit_bias,
        timestamp_requests=timestamp_requests,
    )
    verified_token_ids = []
    worker = round_worker(runner, chain_logits, verified_token_ids)
    start = pool.snapshot()
    loop_result = worker.forward_decode(batch)
    loop_buffers = pool.snapshot()

    pool.restore(start)
    graph = DraftDecodeGraph(
        runner,
        NUM_DECODE_STEPS,
        graph_batch_size,
        timestamp_params=TIMESTAMP_PARAMS if timestamp_requests else None,
    )
    # note (Haoling Pu): the eager steps stand in for the replay on CPU; the CUDA test checks that the replay equals them.
    graph.replay = lambda: torch.cat(
        [graph.first_draft_token_ids[:, None], graph.run_steps()], dim=1
    )
    if timestamp_requests:
        worker.timestamp_draft_graphs = [graph]
    else:
        worker.draft_graphs = [graph]
    graph_result = worker.forward_decode(batch)

    assert verified_token_ids[1].tolist() == verified_token_ids[0].tolist()
    assert graph_result.next_token_ids.tolist() == loop_result.next_token_ids.tolist()
    assert graph_result.accept_lens.tolist() == loop_result.accept_lens.tolist()
    # note (Haoling Pu): slot 0 is the padding slot, which the graph's rows past the round write.
    for graph_buffer, loop_buffer in zip(pool.snapshot(), loop_buffers, strict=True):
        torch.testing.assert_close(graph_buffer[1:], loop_buffer[1:])


@torch.no_grad()
def test_a_round_after_a_larger_one_writes_only_its_own_requests_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rows past a round still hold the requests of a larger round before it, which must not be written."""
    torch.manual_seed(0)
    runner = draft_runner(monkeypatch, "cpu")
    pool = runner.token_to_kv_pool
    req_to_token = runner.req_to_token_pool.req_to_token
    graph = DraftDecodeGraph(runner, NUM_DECODE_STEPS, 4)
    graph.load_inputs(
        round_batch([1, 2, 3], [5, 8, 3], [4, 9, 2], "cpu"),
        torch.tensor([1, 2, 3]),
        None,
    )
    graph.load_inputs(round_batch([2], [8], [9], "cpu"), torch.tensor([4]), None)
    start = pool.snapshot()
    graph.run_steps()

    # note (Haoling Pu): the first decode step writes the column after the committed tokens and the newest one.
    written_columns = 8 + 9 + 1 + torch.arange(NUM_DECODE_STEPS)
    written_slots = set()
    for buffer, saved in zip(pool.snapshot(), start, strict=True):
        written_slots |= set(
            (buffer != saved).flatten(1).any(dim=1).nonzero().flatten().tolist()
        )
    assert written_slots - {0} == set(req_to_token[2, written_columns].tolist())


@torch.no_grad()
def test_a_request_without_timestamps_drafts_by_plain_argmax_in_a_row_a_timestamp_request_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timestamp graph's rows keep the marks and the rule tokens of the round before, and the rules run on every row, so they must reach only the rows the round marks."""
    torch.manual_seed(0)
    runner = draft_runner(monkeypatch, "cpu")
    pool = runner.token_to_kv_pool
    batch = round_batch([1, 2, 3], [5, 8, 3], [4, 9, 2], "cpu")
    first_draft_token_ids = torch.tensor([1, 2, 3])
    logit_bias = suppression_bias(3, "cpu", timestamp_requests=(0, 1, 2))
    start = pool.snapshot()
    plain_graph = DraftDecodeGraph(runner, NUM_DECODE_STEPS, 4)
    plain_graph.load_inputs(batch, first_draft_token_ids, logit_bias)
    plain_token_ids = plain_graph.run_steps()[:3]

    pool.restore(start)
    graph = DraftDecodeGraph(
        runner, NUM_DECODE_STEPS, 4, timestamp_params=TIMESTAMP_PARAMS
    )
    graph.load_inputs(batch, first_draft_token_ids, logit_bias)
    after_a_timestamp = (
        torch.full((3,), 28),
        torch.full((3,), 3),
        torch.full((3,), 28),
    )
    graph.load_timestamp_state(torch.tensor([0, 1, 2]), after_a_timestamp)
    graph.load_timestamp_state(
        torch.tensor([0]), tuple(tokens[:1] for tokens in after_a_timestamp)
    )
    token_ids = graph.run_steps()[:3]

    # note (Tz22z): after a lone timestamp the rules leave a request only timestamp 29, which plain argmax ranks below its open text tokens.
    assert token_ids[0, 0] == 29
    assert plain_token_ids[0, 0] < 20
    assert token_ids[1:].tolist() == plain_token_ids[1:].tolist()


@pytest.mark.parametrize(
    ("graph_batch_sizes", "rows", "is_greedy", "has_segment_timestamps", "used_graph"),
    [
        ((1, 2), [2], True, False, 1),
        ((1, 2), [2, 3], True, False, 2),
        ((4,), [2, 3], True, False, 4),
        ((1, 2), [1, 2, 3], True, False, None),
        ((1, 2), [2], False, False, None),
        ((1, 2), [2], True, True, 1),
        ((1, 2), [2, 3], True, True, 2),
        ((1, 2), [2], False, True, None),
    ],
    ids=[
        "lone_greedy",
        "two_requests",
        "two_requests_in_a_graph_of_four",
        "more_requests_than_any_graph",
        "sampled",
        "segment_timestamps",
        "two_requests_one_with_segment_timestamps",
        "sampled_with_segment_timestamps",
    ],
)
def test_a_greedy_round_drafts_through_the_smallest_graph_of_its_kind_that_fits(
    monkeypatch: pytest.MonkeyPatch,
    graph_batch_sizes: tuple[int, ...],
    rows: list[int],
    is_greedy: bool,
    has_segment_timestamps: bool,
    used_graph: int | None,
) -> None:
    """The draft graphs decode by argmax for a fixed number of rows, so more requests than any graph or a sampled draft must draft step by step. Only the graphs captured with the timestamp rules apply them, so a round takes those when any of its requests has segment timestamps and the plain ones otherwise."""
    eager_draft = 2
    draft_logits = torch.zeros(len(rows), 32)
    draft_logits[:, eager_draft] = 50.0
    verified_token_ids = []
    worker = round_worker(
        draft_runner(monkeypatch, "cpu"),
        draft_logits.repeat_interleave(2, dim=0),
        verified_token_ids,
    )
    worker.decode_forward = lambda batch, runner, **step: draft_logits
    timestamp_rows_loaded = []

    # note (Haoling Pu): each stand-in graph drafts 10 plus its batch size on all of its rows, so the verify input names the graph and its row count.
    def stand_in_graphs(draft_base: int) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                batch_size=graph_batch_size,
                load_inputs=lambda batch, first_draft_token_ids, logit_bias: None,
                load_timestamp_state=lambda timestamp_rows, timestamp_state: (
                    timestamp_rows_loaded.append(timestamp_rows.tolist())
                ),
                replay=lambda graph_batch_size=graph_batch_size: torch.tensor(
                    [[eager_draft] + [draft_base + graph_batch_size] * NUM_DECODE_STEPS]
                    * graph_batch_size
                ),
            )
            for graph_batch_size in graph_batch_sizes
        ]

    worker.draft_graphs = stand_in_graphs(10)
    # note (Tz22z): the timestamp graphs draft 14 plus their batch size, which no plain graph here drafts, so the verify input also names the kind.
    worker.timestamp_draft_graphs = stand_in_graphs(14)

    worker.forward_decode(
        round_batch(
            rows,
            [5] * len(rows),
            [4] * len(rows),
            "cpu",
            is_greedy=is_greedy,
            timestamp_requests=(0,) if has_segment_timestamps else (),
        )
    )

    if used_graph is None:
        later_draft = eager_draft
    else:
        later_draft = (14 if has_segment_timestamps else 10) + used_graph
    assert verified_token_ids[0][:, 1:].tolist() == [
        [eager_draft] + [later_draft] * NUM_DECODE_STEPS
    ] * len(rows)
    # note (Tz22z): the graph gets the rule tokens of the requests with timestamps only, by their place in the round.
    assert timestamp_rows_loaded == (
        [[0]] if has_segment_timestamps and used_graph is not None else []
    )


def test_a_draft_graph_refuses_torch_cuda_graphs_default_capture_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The encoder's graphs replay from their own thread with the cuBLAS workspace of torch.cuda.graph's default stream, and a draft graph sharing it hung the server at concurrency 4 (#4395)."""
    default_stream = object()
    monkeypatch.setattr(torch.cuda.graph, "default_capture_stream", default_stream)
    graph = DraftDecodeGraph(draft_runner(monkeypatch, "cpu"), NUM_DECODE_STEPS, 1)
    with pytest.raises(RuntimeError, match="default stream"):
        graph.capture((0, 0), default_stream)


def capture_draft_graphs(
    monkeypatch: pytest.MonkeyPatch,
    max_requests: int,
    streams_since_default: int | None = None,
) -> list[tuple[int, object, dict[str, int | bool | None] | None]]:
    """Run init_cuda_graphs for a request pool of max_requests rows, with a stand-in for PyTorch's 32 pool streams handed out in turn, and return each captured graph's batch size, stream and timestamp params; with streams_since_default, torch.cuda.graph's default stream was made that many pool streams earlier."""
    pool_streams = itertools.cycle([object() for _ in range(32)])
    monkeypatch.setattr(torch.cuda, "Stream", lambda: next(pool_streams))
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: (0, 0))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (0, 0))
    if streams_since_default is None:
        monkeypatch.setattr(torch.cuda.graph, "default_capture_stream", None)
    else:
        monkeypatch.setattr(
            torch.cuda.graph, "default_capture_stream", next(pool_streams)
        )
        for _ in range(streams_since_default):
            next(pool_streams)
    captured = []
    monkeypatch.setattr(
        speculative,
        "DraftDecodeGraph",
        lambda runner, num_decode_steps, batch_size, *, timestamp_params: SimpleNamespace(
            capture=lambda memory_pool, capture_stream: captured.append(
                (batch_size, capture_stream, timestamp_params)
            )
        ),
    )
    worker = object.__new__(speculative.WhisperSpeculativeWorker)
    worker._draft_worker = SimpleNamespace(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        init_cuda_graphs=lambda: None
    )
    worker._target_worker = SimpleNamespace(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        model_runner=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_generation_config=SimpleNamespace(
                    no_timestamps_token_id=TIMESTAMP_PARAMS["no_timestamps_token_id"]
                )
            ),
            model=SimpleNamespace(
                config=SimpleNamespace(eos_token_id=TIMESTAMP_PARAMS["eos_token_id"])
            ),
        )
    )
    worker.speculative_num_steps = NUM_DECODE_STEPS + 1
    worker.draft_runner = SimpleNamespace(
        decode_cuda_graph_runner=object(),
        kv_cache_dtype=torch.bfloat16,
        dtype=torch.bfloat16,
    )
    worker.req_to_token_pool = SimpleNamespace(size=max_requests)
    worker.draft_graphs = []
    worker.timestamp_draft_graphs = []
    worker.init_cuda_graphs()
    return captured


@pytest.mark.parametrize(
    "streams_since_default",
    [None, 0, 31],
    ids=["no_default_yet", "default_just_made", "default_comes_round_again"],
)
def test_the_draft_graphs_never_capture_on_torch_cuda_graphs_default_stream(
    monkeypatch: pytest.MonkeyPatch, streams_since_default: int | None
) -> None:
    """torch.cuda.Stream() hands out 32 pool streams in turn. With no default stream yet, the encoder's graphs would make it after the draft graphs capture and could be handed the draft stream; a default stream made 31 streams earlier is the next one handed out (#4397's CUDA test hit this in the accelerator job)."""
    captured = capture_draft_graphs(monkeypatch, 64, streams_since_default)
    assert torch.cuda.graph.default_capture_stream is not None
    assert all(
        stream is not torch.cuda.graph.default_capture_stream
        for _, stream, _ in captured
    )


@pytest.mark.parametrize(
    ("max_requests", "captured_batch_sizes"),
    [(8, [1, 2, 4, 8]), (10, [1, 2, 4, 8, 16]), (64, [1, 2, 4, 8, 16, 32])],
)
def test_the_draft_graphs_stop_at_the_first_size_that_holds_every_running_request(
    monkeypatch: pytest.MonkeyPatch,
    max_requests: int,
    captured_batch_sizes: list[int],
) -> None:
    """A round holds at most the request pool's rows, so a larger graph would only take memory. Every size is captured once by plain argmax and once under the timestamp rules, whose ids are the target's: its generation config names the no-timestamps token, and the timestamps start right after it."""
    captured = capture_draft_graphs(monkeypatch, max_requests)
    for graph_timestamp_params in (None, TIMESTAMP_PARAMS):
        assert (
            sorted(
                batch_size
                for batch_size, _, timestamp_params in captured
                if timestamp_params == graph_timestamp_params
            )
            == captured_batch_sizes
        )


@pytest.mark.accelerator
@pytest.mark.skipif(not current_platform.is_cuda(), reason="captures a CUDA graph")
@torch.no_grad()
def test_replay_decodes_the_loaded_round_as_the_steps_run_eagerly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Graphs of 4 and 1 requests, with and without the timestamp rules, captured on the padding row largest first in one pool, read every round's requests, lengths, first drafts, biases, timestamp rows and rule tokens from their inputs, replayed in any order."""
    torch.manual_seed(0)
    runner = draft_runner(monkeypatch, "cuda")
    pool = runner.token_to_kv_pool
    graphs = {
        (batch_size, timestamp_params is not None): DraftDecodeGraph(
            runner, NUM_DECODE_STEPS, batch_size, timestamp_params=timestamp_params
        )
        for batch_size in (4, 1)
        for timestamp_params in (TIMESTAMP_PARAMS, None)
    }
    memory_pool = torch.cuda.graph_pool_handle()
    capture_stream = draft_capture_stream()
    for graph in graphs.values():
        graph.capture(memory_pool, capture_stream)

    for (
        graph_batch_size,
        rows,
        encoder_lengths,
        committed_lengths,
        first_drafts,
        timestamp_requests,
    ) in (
        (4, [1, 2, 3], [5, 8, 3], [4, 9, 2], [7, 20, 11], ()),
        (4, [1, 2, 3], [5, 8, 3], [4, 9, 2], [28, 20, 28], (0, 2)),
        (1, [3], [6], [5], [9], ()),
        (1, [3], [6], [5], [28], (0,)),
        (4, [2], [8], [9], [5], ()),
        (4, [1, 3], [5, 3], [4, 2], [11, 28], (1,)),
    ):
        graph = graphs[graph_batch_size, bool(timestamp_requests)]
        round_size = len(rows)
        first_draft_token_ids = torch.tensor(first_drafts, device="cuda")
        graph.load_inputs(
            round_batch(rows, encoder_lengths, committed_lengths, "cuda"),
            first_draft_token_ids,
            suppression_bias(round_size, "cuda", timestamp_requests=timestamp_requests),
        )
        if timestamp_requests:
            # note (Tz22z): every request with timestamps has committed text token 3 and drafted timestamp 28 first.
            timestamp_rows = torch.tensor(timestamp_requests, device="cuda")
            graph.load_timestamp_state(
                timestamp_rows,
                (
                    first_draft_token_ids[timestamp_rows],
                    torch.full_like(timestamp_rows, 3),
                    first_draft_token_ids[timestamp_rows],
                ),
            )
        else:
            pass
        start = pool.snapshot()
        replayed_token_ids = graph.replay()[:round_size]
        replayed_buffers = pool.snapshot()
        pool.restore(start)
        eager_token_ids = graph.run_steps()[:round_size]

        assert replayed_token_ids.tolist() == [
            [first_draft] + eager_tokens
            for first_draft, eager_tokens in zip(
                first_drafts, eager_token_ids.tolist(), strict=True
            )
        ]
        # note (Tz22z): after a lone timestamp the rules leave a request only timestamp 29, so a replay that kept the capture's unmarked rows would draft text there.
        assert replayed_token_ids[list(timestamp_requests), 1].tolist() == [29] * len(
            timestamp_requests
        )
        # note (Haoling Pu): the rows past the round all write the padding slot 0, in no fixed order.
        for replayed_buffer, eager_buffer in zip(
            replayed_buffers, pool.snapshot(), strict=True
        ):
            torch.testing.assert_close(replayed_buffer[1:], eager_buffer[1:])
