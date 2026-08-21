# SPDX-License-Identifier: Apache-2.0
"""Runner-side gates for the ZONOS2 tail CUDA graph on heterogeneous batches.

The runner used to replay only when every row carried the exact captured
sampling params, so any real serving mix fell back to eager. It now asks the
model for a graph keyed on the batch's host branches, and feeds the same
rep-window the eager branch would have built.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.zonos2.model_runner import Zonos2ModelRunner
from sglang_omni.models.zonos2.sampler import sample_tts
from sglang_omni.models.zonos2.state_pool import Zonos2DecodeStatePool

N_CODEBOOKS = 9
CODEBOOK_SIZE = 1024
AUDIO_VOCAB = CODEBOOK_SIZE + 2
DIM = 16
WINDOW = 50


_HAS_CUDA = torch.cuda.is_available()

pytestmark = pytest.mark.skipif(not _HAS_CUDA, reason="tail graph replay requires CUDA")


def _params(**overrides) -> SimpleNamespace:
    base = dict(
        temperature=1.15,
        top_k=106,
        top_p=0.0,
        min_p=0.18,
        repetition_penalty=1.2,
        repetition_window=WINDOW,
        repetition_codebooks=8,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _requests(param_list: list[SimpleNamespace]) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(request_id=f"r{i}", data=SimpleNamespace(params=p))
        for i, p in enumerate(param_list)
    ]


class _StubModel:
    """Minimal model surface for _collect_launch; records which tail ran."""

    def __init__(self, device: torch.device) -> None:
        self.n_codebooks = N_CODEBOOKS
        self.audio_vocab = AUDIO_VOCAB
        self.config = SimpleNamespace(
            eoa_id=CODEBOOK_SIZE, text_vocab=32, codebook_size=CODEBOOK_SIZE, dim=DIM
        )
        self._decode_input_embedding = SimpleNamespace(
            weight=torch.zeros(8, DIM, device=device)
        )
        self._decode_state_pool = Zonos2DecodeStatePool(self)
        self.tail_graph_calls: list[dict] = []
        self.eager_calls = 0
        self.graph_calls = 0

    def tail_graph(self, batch_size, **kwargs):
        self.tail_graph_calls.append(dict(batch_size=batch_size, **kwargs))
        return "graph-handle"

    def run_tail_graph(self, graph, hidden, *tensors):
        assert graph == "graph-handle"
        self.graph_calls += 1
        bs = hidden.shape[0]
        device = hidden.device
        codes = torch.zeros(bs, N_CODEBOOKS, device=device, dtype=torch.long)
        keys = torch.zeros(bs, device=device, dtype=torch.long)
        feedback = torch.zeros(bs, DIM, device=device)
        return codes, keys, feedback

    def compute_logits(self, hidden):
        self.eager_calls += 1
        bs = hidden.shape[0]
        return torch.zeros(bs, N_CODEBOOKS, AUDIO_VOCAB, device=hidden.device)

    def embed_frames(self, rows):
        return torch.zeros(rows.shape[0], DIM, device=rows.device)


def _runner(model: _StubModel, *, frame_graph: bool = True) -> Zonos2ModelRunner:
    runner = object.__new__(Zonos2ModelRunner)
    runner.model = model
    runner._frame_graph = frame_graph
    runner._sampler = sample_tts
    return runner


def _launch(runner, requests, device):
    hidden = torch.zeros(len(requests), DIM, device=device)
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=hidden), next_token_ids=None
    )
    return runner._collect_launch(
        result, None, SimpleNamespace(output_ids=None), requests, is_prefill=False
    )


def test_heterogeneous_batch_takes_the_graph_path():
    """Rows with different temperature/top_k/top_p no longer force eager."""
    device = torch.device("cpu")
    model = _StubModel(device)
    runner = _runner(model)
    requests = _requests(
        [
            _params(temperature=1.15, top_k=100, top_p=0.0, min_p=0.18),
            _params(temperature=0.8, top_k=40, top_p=0.9, min_p=0.0),
            _params(temperature=0.0, top_k=0, top_p=0.0, min_p=0.0),
        ]
    )

    _launch(runner, requests, device)

    assert model.graph_calls == 1
    assert model.eager_calls == 0
    assert model.tail_graph_calls == [
        dict(
            batch_size=3,
            top_k_max=100,
            any_top_p=True,
            any_min_p=True,
            window=WINDOW,
        )
    ]


def test_runner_falls_back_to_eager_when_the_model_declines():
    device = torch.device("cpu")
    model = _StubModel(device)
    model.tail_graph = lambda batch_size, **kwargs: None
    runner = _runner(model)

    _launch(runner, _requests([_params(), _params(top_k=40)]), device)

    assert model.eager_calls == 1
    assert model.graph_calls == 0


def test_frame_graph_disabled_keeps_the_eager_tail():
    device = torch.device("cpu")
    model = _StubModel(device)
    runner = _runner(model, frame_graph=False)

    _launch(runner, _requests([_params(), _params()]), device)

    assert model.eager_calls == 1
    assert not model.tail_graph_calls


@pytest.mark.parametrize("penalty", [1.0, 1.2])
def test_rep_window_ring_matches_the_eager_window(penalty: float):
    """The graph always gets a window tensor; at penalty 1.0 eager passes None,
    whose bit-exact equivalent is an all -1 (no-op) window."""
    device = torch.device("cpu")
    model = _StubModel(device)
    runner = _runner(model)
    pool = model._decode_state_pool
    row_t = torch.tensor([0, 1], device=device, dtype=torch.long)
    pool.rep_hist[row_t] = torch.randint(
        -1, CODEBOOK_SIZE + 2, (2, pool.rep_ring, N_CODEBOOKS), dtype=torch.int64
    )
    params = _params(repetition_penalty=penalty)

    ring = runner._rep_window_ring(row_t, N_CODEBOOKS, WINDOW, CODEBOOK_SIZE, params)
    eager = runner._rep_window(row_t, N_CODEBOOKS, CODEBOOK_SIZE, params)

    assert ring.shape == (2, N_CODEBOOKS, WINDOW)
    if eager is None:
        assert bool((ring == -1).all()), "no-op window must be all -1"
    else:
        assert torch.equal(ring, eager)


def test_rep_window_ring_honours_the_live_batch_codebooks():
    """The window must follow the in-flight params, not a captured default."""
    device = torch.device("cpu")
    model = _StubModel(device)
    runner = _runner(model)
    pool = model._decode_state_pool
    row_t = torch.tensor([0], device=device, dtype=torch.long)
    pool.rep_hist[row_t] = torch.zeros(1, pool.rep_ring, N_CODEBOOKS, dtype=torch.int64)

    ring = runner._rep_window_ring(
        row_t, N_CODEBOOKS, WINDOW, CODEBOOK_SIZE, _params(repetition_codebooks=2)
    )

    assert bool((ring[:, :2] == 0).all())
    assert bool((ring[:, 2:] == -1).all())


def test_params_match_guard_is_gone():
    """The all-or-nothing guard must not survive as a live code path."""
    assert not hasattr(Zonos2ModelRunner, "_params_match")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
