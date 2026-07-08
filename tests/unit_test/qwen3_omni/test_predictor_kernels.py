from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sglang_omni.models.qwen3_omni.components.predictor_kernels import (
    sample_code_and_stage_,
    stage_initial_predictor_input_,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Qwen3-Omni predictor kernels require CUDA"
)


def _skip_without_triton(launched: bool) -> None:
    if not launched:
        pytest.skip("Triton predictor kernels are unavailable in this environment")


def test_stage_initial_predictor_input_matches_torch_for_strided_views() -> None:
    device = torch.device("cuda")
    batch_size, seq_len, num_groups, vocab_size, hidden_size = 3, 2, 4, 11, 7

    layer0_codes = torch.tensor(
        [[1, 3], [4, 6], [8, 2]],
        device=device,
        dtype=torch.long,
    )
    talker_hidden = torch.randn(batch_size, seq_len, hidden_size, device=device)
    codec_weight = torch.randn(vocab_size, hidden_size, device=device)
    predictor_input = torch.full(
        (batch_size, num_groups + 1, hidden_size),
        -99.0,
        device=device,
    )
    result_codes = torch.full(
        (batch_size, num_groups, seq_len),
        -1,
        device=device,
        dtype=torch.long,
    )
    summed_embeddings = torch.full(
        (batch_size, seq_len, hidden_size),
        -99.0,
        device=device,
    )

    pos = 1
    launched = stage_initial_predictor_input_(
        layer0_code=layer0_codes[:, pos],
        talker_hidden=talker_hidden[:, pos, :],
        codec_weight=codec_weight,
        predictor_input=predictor_input,
        pos_codes=result_codes[:, :, pos],
        pos_summed=summed_embeddings[:, pos, :],
    )
    _skip_without_triton(launched)
    torch.cuda.synchronize()

    expected_embed = codec_weight[layer0_codes[:, pos]]
    assert torch.equal(result_codes[:, 0, pos], layer0_codes[:, pos])
    torch.testing.assert_close(predictor_input[:, 0, :], talker_hidden[:, pos, :])
    torch.testing.assert_close(predictor_input[:, 1, :], expected_embed)
    torch.testing.assert_close(summed_embeddings[:, pos, :], expected_embed)


def test_sample_code_and_stage_matches_torch_argmax_and_embedding() -> None:
    device = torch.device("cuda")
    batch_size, num_groups, vocab_size, hidden_size = 3, 5, 9, 6

    logits = torch.full((batch_size, 1, vocab_size), -10.0, device=device)
    logits[0, 0, 2] = 4.0
    logits[0, 0, 3] = 4.0
    logits[1, 0, 7] = 5.0
    logits[2, 0, 0] = 6.0
    embedding_weight = torch.randn(vocab_size, hidden_size, device=device)
    predictor_input = torch.zeros(
        batch_size,
        num_groups + 1,
        hidden_size,
        device=device,
    )
    pos_codes = torch.full(
        (batch_size, num_groups),
        -1,
        device=device,
        dtype=torch.long,
    )
    pos_summed = torch.randn(batch_size, hidden_size, device=device)
    expected_sum = pos_summed.clone()
    layer_idx = 2

    launched = sample_code_and_stage_(
        logits=logits,
        embedding_weight=embedding_weight,
        predictor_input=predictor_input,
        pos_codes=pos_codes,
        pos_summed=pos_summed,
        layer_idx=layer_idx,
    )
    _skip_without_triton(launched)
    torch.cuda.synchronize()

    expected_codes = torch.argmax(logits[:, -1, :], dim=-1)
    expected_embed = embedding_weight[expected_codes]
    assert torch.equal(pos_codes[:, layer_idx + 1], expected_codes)
    torch.testing.assert_close(predictor_input[:, layer_idx + 2, :], expected_embed)
    torch.testing.assert_close(pos_summed, expected_sum + expected_embed)
