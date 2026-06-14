from types import SimpleNamespace

import torch

from sglang_omni.models.zonos2_tts.model_runner import Zonos2TTSModelRunner
from sglang_omni.models.zonos2_tts.state_pool import Zonos2TTSDecodeStatePool


def _dummy_model(max_rows: int = 4, hidden_size: int = 3, n_codebooks: int = 2):
    model = SimpleNamespace(
        _decode_input_embedding=torch.nn.Embedding(max_rows, hidden_size),
        hidden_size=hidden_size,
        n_codebooks=n_codebooks,
    )
    model._decode_input_embedding.weight.requires_grad_(False)
    model._state_pool = Zonos2TTSDecodeStatePool(model)
    return model


def test_state_pool_recent_history_preserves_ring_order() -> None:
    model = _dummy_model(n_codebooks=2)
    pool = model._state_pool
    row = pool.acquire_row("rid")
    pool.ensure_history_capacity(3)

    for values in ([1, 11], [2, 12], [3, 13], [4, 14]):
        pool.update_history(
            torch.tensor([row], dtype=torch.long),
            torch.tensor([values], dtype=torch.long),
        )

    recent = pool.recent_history(
        row,
        window=3,
        n_codebooks=2,
        device=torch.device("cpu"),
    )

    assert recent is not None
    assert recent.tolist() == [[2, 3, 4], [12, 13, 14]]
    assert pool.history_length(row) == 3


def test_state_pool_update_history_can_use_host_row_indices() -> None:
    model = _dummy_model(n_codebooks=2)
    pool = model._state_pool
    row = pool.acquire_row("rid")
    pool.ensure_history_capacity(2)

    pool.update_history(
        torch.tensor([row], dtype=torch.long),
        torch.tensor([[5, 15]], dtype=torch.long),
        row_indices=[row],
    )

    assert pool.history_length(row) == 1
    assert pool._history_pos_host[row] == 1


def test_state_pool_rebuild_history_after_refill() -> None:
    model = _dummy_model(n_codebooks=2)
    pool = model._state_pool
    row = pool.acquire_row("rid")
    pool.ensure_history_capacity(4)

    output_rows = [
        torch.tensor([1, 11, 100], dtype=torch.long),
        torch.tensor([2, 12, 100], dtype=torch.long),
    ]
    assert pool.reset_for_refill("rid", output_rows)

    recent = pool.recent_history(
        row,
        window=4,
        n_codebooks=2,
        device=torch.device("cpu"),
    )

    assert recent is not None
    assert recent.tolist() == [[1, 2], [11, 12]]
    assert pool.history_length(row) == 2


def test_decode_input_embedding_reads_feedback_from_state_pool() -> None:
    model = _dummy_model(max_rows=4, hidden_size=3, n_codebooks=2)
    runner = object.__new__(Zonos2TTSModelRunner)
    runner.model = model
    row = model._state_pool.acquire_row("rid")
    model._state_pool.feedback_embeds[row].copy_(torch.tensor([1.0, 2.0, 3.0]))

    forward_batch = SimpleNamespace(input_ids=torch.zeros(1, dtype=torch.long))
    request = SimpleNamespace(
        request_id="rid",
        data=SimpleNamespace(pending_feedback_queue=[]),
    )

    runner._write_decode_input_embedding(forward_batch, [request])

    torch.testing.assert_close(
        model._decode_input_embedding.weight[0],
        torch.tensor([1.0, 2.0, 3.0]),
    )
    assert forward_batch.input_ids.tolist() == [0]
    assert forward_batch.zonos2_pool_rows == [row]
