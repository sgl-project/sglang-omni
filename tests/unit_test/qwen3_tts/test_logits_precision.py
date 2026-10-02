# SPDX-License-Identifier: Apache-2.0
"""Talker logits retain sampling precision without widening hidden states."""

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties

from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker


def _talker():
    talker = Qwen3TTSTalker.__new__(Qwen3TTSTalker)
    talker.model = lambda **kwargs: kwargs["input_embeds"]
    # Keep real forward/last-token selection; replace only the model and head.
    talker.codec_head = lambda hidden: (hidden + 0, None)
    return talker


def _forward(talker, inputs, *, extend_lengths=None):
    batch = SimpleNamespace(
        mrope_positions=None,
        forward_mode=SimpleNamespace(is_extend=lambda: extend_lengths is not None),
        extend_seq_lens=extend_lengths,
    )
    return talker.forward(
        input_ids=torch.zeros(inputs.shape[0], dtype=torch.long, device=inputs.device),
        positions=torch.arange(inputs.shape[0], device=inputs.device),
        forward_batch=batch,
        input_embeds=inputs,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("is_extend", [False, True], ids=["decode", "extend"])
def test_talker_penalty_preserves_token_order(dtype, is_extend):
    # Each half format needs its own close race. In both penalized rows the
    # correct winner is token 1; a half-precision writeback produces a tie.
    if dtype == torch.float16:
        positive, negative = [15.234375, 16.0], [-16.0, -15.234375]
        repeated_token, control_winner = 1, 1
    else:
        positive, negative = [16.0, 15.25], [-15.25, -16.0]
        repeated_token, control_winner = 0, 0
    # Rows: positive penalty, negative penalty, penalty=1, empty history.
    head_logits = torch.tensor([positive, negative, positive, positive], dtype=dtype)
    scales = torch.ones(4, 2, dtype=torch.float32)
    scales[:2, repeated_token] = 1.05
    # A repeated token with penalty=1 and a request with no repeated tokens
    # both supply identity scaling to the shared SGLang penalty function.
    lengths = torch.tensor([1, 3, 2, 4]) if is_extend else None
    inputs = head_logits.clone()
    if is_extend:
        inputs = torch.full((10, 2), -50.0, dtype=dtype)
        inputs[[0, 3, 5, 9]] = head_logits
    original_inputs = inputs.clone()
    result = _forward(_talker(), inputs, extend_lengths=lengths)
    shaped = result.next_token_logits.clone()
    with torch._dynamo.config.patch(disable=True):
        apply_scaling_penalties(shaped, scales)

    # Check the observable regression first: old BF16/FP16 code fails here,
    # rather than merely failing a check for the new output dtype.
    assert shaped.argmax(dim=-1).tolist() == [1, 1, control_winner, control_winner]
    assert result.next_token_logits.dtype == torch.float32
    assert result.hidden_states.dtype == dtype
    torch.testing.assert_close(result.next_token_logits, head_logits.float())
    torch.testing.assert_close(result.hidden_states, head_logits)
    torch.testing.assert_close(inputs, original_inputs)
    reference = head_logits.double()
    for row in range(2):
        value = reference[row, repeated_token].item()
        reference[row, repeated_token] = value / 1.05 if value >= 0 else value * 1.05
    torch.testing.assert_close(shaped, reference.float(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(shaped[2:], head_logits[2:].float())


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graphs")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("batch_size", [1, 8])
def test_talker_fp32_logits_refresh_on_cuda_graph_replay(dtype, batch_size):
    talker = _talker()
    inputs = torch.tensor([2.0, 1.0], device="cuda", dtype=dtype).repeat(batch_size, 1)
    # Allocate graph inputs once; only their contents change between replays.
    batch = SimpleNamespace(
        mrope_positions=None,
        forward_mode=SimpleNamespace(is_extend=lambda: False),
    )
    ids = torch.zeros(batch_size, device="cuda", dtype=torch.long)
    positions = torch.arange(batch_size, device="cuda")

    def run():
        return talker.forward(ids, positions, batch, input_embeds=inputs)

    stream = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        for _ in range(2):
            run()
    current.wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = run()
    graph.replay()
    torch.cuda.synchronize()
    logits = captured.next_token_logits
    pointer = logits.data_ptr()
    assert logits.argmax(dim=-1).tolist() == [0] * batch_size
    assert logits.dtype == torch.float32

    for values, winner in [([1.0, 3.0], 1), ([4.0, -2.0], 0)]:
        inputs.copy_(torch.tensor(values, device="cuda", dtype=dtype))
        graph.replay()
        torch.cuda.synchronize()
        assert logits.argmax(dim=-1).tolist() == [winner] * batch_size
        assert captured.next_token_logits.data_ptr() == pointer
        assert logits.dtype == torch.float32
        assert captured.hidden_states.dtype == dtype
        torch.testing.assert_close(logits, inputs.float())
        torch.testing.assert_close(captured.hidden_states, inputs)
