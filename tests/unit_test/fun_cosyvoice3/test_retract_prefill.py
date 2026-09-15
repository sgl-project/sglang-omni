# SPDX-License-Identifier: Apache-2.0
"""Replay retained speech tokens when a CosyVoice request re-enters prefill."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3.model_runner import FunCosyVoice3ModelRunner


class _TinyModel(torch.nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.speech_embedding = torch.nn.Embedding(64, 4, dtype=dtype)
        with torch.no_grad():
            self.speech_embedding.weight.copy_(torch.arange(256).reshape(64, 4) + 1000)


def _runner(dtype: torch.dtype = torch.float32) -> FunCosyVoice3ModelRunner:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner.model = _TinyModel(dtype)
    return runner


def _request(
    prefix: int, extend: int, output_ids: list[int], *, offset: float = 0
) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(
            req=SimpleNamespace(
                rid="replay",
                # Prepared prompt IDs are hashes, outside the speech vocabulary.
                origin_input_ids=[2**40 + i for i in range(4)],
                output_ids=list(output_ids),
                prefix_indices=list(range(prefix)),
                extend_range=SimpleNamespace(length=extend),
                # prepare_for_extend clears this before the model runner executes.
                is_retracted=False,
            ),
            prompt_input_embeds=torch.arange(16, dtype=torch.float32).reshape(4, 4)
            + offset,
            # Deliberately incomplete: recovery must use req.output_ids,
            # not the separate output_codes collected for downstream audio.
            output_codes=[torch.tensor([token]) for token in output_ids[:-1]],
            stream_code_seen=8,
            stream_prompt_sent=True,
        )
    )


def _batch(*requests: SimpleNamespace) -> SimpleNamespace:
    # prepare_for_extend slices get_fill_ids()[len(req.prefix_indices):];
    # its length invariant makes the endpoint prefix + extend_range.length.
    ids = []
    for request in requests:
        req = request.data.req
        start = len(req.prefix_indices)
        ids.extend(
            (req.origin_input_ids + req.output_ids)[
                start : start + req.extend_range.length
            ]
        )
    return SimpleNamespace(input_ids=torch.tensor(ids, dtype=torch.long))


@pytest.mark.parametrize(
    "prefix,extend,output_ids",
    [
        (0, 4, []),  # Fresh prompt.
        (2, 2, []),  # Ordinary prompt prefix hit.
        (0, 7, [7, 8, 9]),  # Full replay, including the last sampled token.
        (2, 5, [7, 8, 9]),  # Replay crosses the prompt/generated boundary.
        (4, 3, [7, 8, 9]),  # Cached prefix ends at the prompt boundary.
        (5, 2, [7, 8, 9]),  # Cached prefix includes generated speech tokens.
        (3, 2, [7, 8, 9]),  # Partial extend across the boundary.
        (5, 1, [7, 8, 9]),  # Partial extend wholly within generated tokens.
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prefill_reconstructs_the_requested_interval(
    prefix: int, extend: int, output_ids: list[int], dtype: torch.dtype
) -> None:
    runner = _runner(dtype)
    request = _request(prefix, extend, output_ids)
    batch = _batch(request)
    prompt = request.data.prompt_input_embeds.to(dtype=dtype)
    # FunCosyVoice3SGLangModel.forward uses speech_embedding for decode inputs.
    speech = runner.model.speech_embedding(torch.tensor(output_ids, dtype=torch.long))
    expected = torch.cat([prompt, speech])[prefix : prefix + extend]

    # The runner's execution bridge does not disable autograd. Replay must not
    # retain a graph for the trainable embedding layer during inference.
    assert runner.model.speech_embedding.weight.requires_grad
    with torch.enable_grad():
        actual = runner._build_prefill_input_embeds(batch, [request])

    assert actual.shape[0] == batch.input_ids.numel() == extend
    assert actual.dtype == dtype
    assert not actual.requires_grad
    assert actual.grad_fn is None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_prefill_preserves_mixed_batch_order() -> None:
    runner = _runner()
    fresh = _request(1, 3, [], offset=100)
    retracted = _request(3, 4, [7, 8, 9])
    expected = torch.cat(
        [
            fresh.data.prompt_input_embeds[1:],
            retracted.data.prompt_input_embeds[3:],
            runner.model.speech_embedding(torch.tensor([7, 8, 9])),
        ]
    )

    actual = runner._build_prefill_input_embeds(
        _batch(fresh, retracted), [fresh, retracted]
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_prefill_rejects_missing_retained_history() -> None:
    request = _request(2, 5, [7])
    batch = SimpleNamespace(input_ids=torch.zeros(5, dtype=torch.long))

    with pytest.raises(
        RuntimeError,
        match=r"row mismatch for replay: have 3 rows, need 5 .*generated=1",
    ):
        _runner()._build_prefill_input_embeds(batch, [request])


def test_repeated_prefill_does_not_consume_or_emit_history() -> None:
    runner = _runner()
    request = _request(0, 7, [7, 8, 9])
    data = request.data
    prompt = data.prompt_input_embeds.clone()
    codes = [code.clone() for code in data.output_codes]
    batch = _batch(request)
    first = runner._build_prefill_input_embeds(batch, [request])
    second = runner._build_prefill_input_embeds(batch, [request])

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(data.prompt_input_embeds, prompt, rtol=0, atol=0)
    assert data.req.output_ids == [7, 8, 9]
    assert len(data.output_codes) == len(codes)
    for observed, expected in zip(data.output_codes, codes):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert data.stream_code_seen == 8
    assert data.stream_prompt_sent is True
