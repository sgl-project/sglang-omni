# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
from sglang_omni.models.whisper_asr.draft_cuda_graph import DraftDecodeGraph
from sglang_omni.platforms import current_platform

speculative = pytest.importorskip("sglang_omni.models.whisper_asr.speculative")

NUM_DECODE_STEPS = 3
NUM_KV_SLOTS = 64


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
    """Attention over exactly a request's encoder slots or its first seq_len decoder slots."""

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
        request_slots = self.req_to_token[int(forward_batch.req_pool_indices)].long()
        encoder_length = int(forward_batch.encoder_lens)
        key_cache, value_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        if layer.is_cross_attention:
            slots = request_slots[:encoder_length]
        else:
            self.token_to_kv_pool.set_kv_buffer(
                layer, KVWriteLoc(forward_batch.out_cache_loc), k, v
            )
            slots = request_slots[
                encoder_length : encoder_length + int(forward_batch.seq_lens)
            ]
        query = q.view(layer.tp_q_head_num, layer.head_dim)
        scores = torch.einsum("hd,shd->hs", query, key_cache[slots]) * layer.scaling
        return torch.einsum(
            "hs,shd->hd", scores.softmax(dim=-1), value_cache[slots]
        ).reshape(1, -1)


def draft_runner(monkeypatch: pytest.MonkeyPatch, device: str) -> SimpleNamespace:
    """A runner of a two-layer Whisper decoder whose request rows map columns to scattered KV slots."""
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
    for row in range(1, num_rows):
        req_to_token[row] = torch.randperm(NUM_KV_SLOTS, device=device)[
            : req_to_token.shape[1]
        ]
    return SimpleNamespace(
        model=model,
        token_to_kv_pool=KVBuffers(2 * config.decoder_layers, config, device),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
    )


def suppression_bias(device: str) -> torch.Tensor:
    """A Whisper-style logit bias that leaves a few tokens open, so a step that skips it drafts another token."""
    logit_bias = torch.full((1, 32), -1e9, device=device)
    logit_bias[0, torch.randperm(32, device=device)[:4]] = 0
    return logit_bias


def round_batch(
    rows: list[int],
    encoder_length: int,
    committed_length: int,
    device: str,
    *,
    logit_bias: torch.Tensor | None = None,
    is_greedy: bool = True,
    custom_params: dict[str, int | bool | None] | None = None,
) -> SimpleNamespace:
    """A decode round of one request per request pool row, each with the same lengths and sampling."""
    num_requests = len(rows)
    seq_lens = torch.full((num_requests,), committed_length, device=device)
    return SimpleNamespace(
        req_pool_indices=torch.tensor(rows, device=device),
        encoder_lens=torch.full((num_requests,), encoder_length, device=device),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.cpu(),
        spec_info=SimpleNamespace(
            bonus_tokens=torch.full((num_requests,), 3, device=device)
        ),
        reqs=[
            SimpleNamespace(
                sampling_params=SimpleNamespace(custom_params=custom_params),
                output_ids=[3],
            )
            for _ in rows
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
    worker = object.__new__(speculative.WhisperSpeculativeWorker)
    worker.device = "cpu"
    worker.speculative_num_steps = NUM_DECODE_STEPS + 1
    worker.previous_token_ids = torch.zeros(4, dtype=torch.int64)
    worker.req_to_token_pool = runner.req_to_token_pool
    worker.draft_runner = runner
    worker.draft_graph = None
    worker.chain_forward = lambda batch, runner, **chain: chain_logits
    worker.chain_batch = lambda batch, runner, *, token_ids, **chain: (
        verified_token_ids.append(token_ids)
    )
    worker._target_worker = SimpleNamespace(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        model_runner=None,
        forward_batch_generation=lambda batch, **verify: SimpleNamespace(
            logits_output=SimpleNamespace(
                next_token_logits=torch.zeros(num_requests * (NUM_DECODE_STEPS + 2), 32)
            ),
            can_run_cuda_graph=True,
        ),
    )
    return worker


@torch.no_grad()
def test_a_lone_greedy_round_drafts_through_the_graph_steps_as_through_the_step_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The graph's steps draft the tokens and write the K/V of the worker's step-by-step draft loop, whose offsets and KV slots they must follow."""
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
            batch_size=1,
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
    logit_bias = suppression_bias("cpu")
    chain_logits = torch.zeros(2, 32)
    # note (Haoling Pu): the first draft is a token the bias closes, so no later draft repeats it and a step fed the wrong token writes other K/V.
    chain_logits[1, int((logit_bias[0] < 0).nonzero()[0])] = 2e9
    batch = round_batch([2], 5, 4, "cpu", logit_bias=logit_bias)
    verified_token_ids = []
    worker = round_worker(runner, chain_logits, verified_token_ids)
    start = pool.snapshot()
    worker.forward_decode(batch)
    loop_buffers = pool.snapshot()

    pool.restore(start)
    graph = DraftDecodeGraph(runner, NUM_DECODE_STEPS)
    # note (Haoling Pu): the eager steps stand in for the replay on CPU; the CUDA test checks that the replay equals them.
    graph.replay = lambda: torch.cat(
        [graph.first_draft_token_ids[:, None], graph.run_steps()], dim=1
    )
    worker.draft_graph = graph
    worker.forward_decode(batch)

    assert verified_token_ids[1].tolist() == verified_token_ids[0].tolist()
    for graph_buffer, loop_buffer in zip(pool.snapshot(), loop_buffers, strict=True):
        torch.testing.assert_close(graph_buffer, loop_buffer)


@pytest.mark.parametrize(
    ("rows", "is_greedy", "has_segment_timestamps", "uses_graph"),
    [
        ([2], True, False, True),
        ([2, 3], True, False, False),
        ([2], False, False, False),
        ([2], True, True, False),
    ],
    ids=["lone_greedy", "two_requests", "sampled", "segment_timestamps"],
)
def test_only_a_lone_greedy_request_without_timestamps_drafts_through_the_draft_graph(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[int],
    is_greedy: bool,
    has_segment_timestamps: bool,
    uses_graph: bool,
) -> None:
    """The draft graph decodes one request by plain argmax, so a batch, a sampled draft or timestamp rules must draft step by step."""
    eager_draft, graph_draft = 2, 9
    draft_logits = torch.zeros(len(rows), 32)
    draft_logits[:, eager_draft] = 50.0
    timestamp_params = {
        "segment_timestamps": True,
        "timestamp_begin_id": 28,
        "no_timestamps_token_id": 27,
        "eos_token_id": 20,
        "max_initial_timestamp_index": None,
    }
    verified_token_ids = []
    worker = round_worker(
        draft_runner(monkeypatch, "cpu"),
        draft_logits.repeat_interleave(2, dim=0),
        verified_token_ids,
    )
    worker.decode_forward = lambda batch, runner, **step: draft_logits
    worker.draft_graph = SimpleNamespace(
        load_inputs=lambda batch, first_draft_token_ids, logit_bias: None,
        replay=lambda: torch.tensor([[eager_draft] + [graph_draft] * NUM_DECODE_STEPS]),
    )

    worker.forward_decode(
        round_batch(
            rows,
            5,
            4,
            "cpu",
            is_greedy=is_greedy,
            custom_params=timestamp_params if has_segment_timestamps else None,
        )
    )

    later_draft = graph_draft if uses_graph else eager_draft
    assert verified_token_ids[0][:, 1:].tolist() == [
        [eager_draft] + [later_draft] * NUM_DECODE_STEPS
    ] * len(rows)


@pytest.mark.accelerator
@pytest.mark.skipif(not current_platform.is_cuda(), reason="captures a CUDA graph")
@torch.no_grad()
def test_replay_decodes_the_loaded_round_as_the_steps_run_eagerly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The graph captured on the padding row reads every round's request, lengths, first draft and bias from its inputs."""
    torch.manual_seed(0)
    runner = draft_runner(monkeypatch, "cuda")
    pool = runner.token_to_kv_pool
    graph = DraftDecodeGraph(runner, NUM_DECODE_STEPS)
    graph.capture()

    for row, encoder_length, committed_length, first_draft in (
        (2, 5, 4, 7),
        (3, 8, 9, 20),
    ):
        logit_bias = suppression_bias("cuda")
        graph.load_inputs(
            round_batch([row], encoder_length, committed_length, "cuda"),
            torch.tensor([first_draft], device="cuda"),
            logit_bias,
        )
        start = pool.snapshot()
        replayed_token_ids = graph.replay()
        replayed_buffers = pool.snapshot()
        pool.restore(start)
        eager_token_ids = graph.run_steps()

        assert replayed_token_ids.tolist() == [
            [first_draft] + eager_token_ids[0].tolist()
        ]
        for replayed_buffer, eager_buffer in zip(
            replayed_buffers, pool.snapshot(), strict=True
        ):
            torch.testing.assert_close(replayed_buffer, eager_buffer)
