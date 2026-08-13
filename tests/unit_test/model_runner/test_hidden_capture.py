# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from sglang_omni.model_runner._hidden_capture import install_hidden_capture_hooks


class _FakeDecoderLayer(nn.Module):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: object,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del positions, forward_batch, kwargs
        return hidden_states, residual


class _FakeTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 3)
        self.layers = nn.ModuleList([_FakeDecoderLayer(), _FakeDecoderLayer()])
        self.start_layer = 0
        self.end_layer = 2
        self.layers_to_capture = [99]

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens


def _top_level_model() -> tuple[SimpleNamespace, _FakeTextModel]:
    text_model = _FakeTextModel()
    outer = SimpleNamespace(model=text_model)
    return SimpleNamespace(thinker=outer), text_model


def _run_layer(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
) -> None:
    layer(
        torch.arange(hidden_states.shape[0]),
        hidden_states,
        SimpleNamespace(),
        residual,
    )


def test_static_capture_records_exact_layer_inputs_in_registered_buffers() -> None:
    model, text_model = _top_level_model()

    install_hidden_capture_hooks(model, [0, 1], max_tokens=4)
    capture = model._omni_aux_hidden_capture
    first_pointers = [buffer.data_ptr() for buffer in capture.buffers]
    embedding_weight = text_model.get_input_embeddings().weight
    assert [tuple(buffer.shape) for buffer in capture.buffers] == [(4, 3), (4, 3)]
    assert all(
        buffer.dtype == embedding_weight.dtype
        and buffer.device == embedding_weight.device
        for buffer in capture.buffers
    )

    layer0_hidden = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    layer1_hidden = torch.arange(6, 12, dtype=torch.float32).reshape(2, 3)
    layer1_residual = torch.full((2, 3), 0.5)
    _run_layer(text_model.layers[0], layer0_hidden, None)
    _run_layer(text_model.layers[1], layer1_hidden, layer1_residual)

    views = capture.views(2)
    torch.testing.assert_close(views[0], layer0_hidden)
    torch.testing.assert_close(views[1], layer1_hidden + layer1_residual)
    assert [buffer.data_ptr() for buffer in capture.buffers] == first_pointers
    assert text_model.layers_to_capture == []
    assert not any("omni_aux_hidden" in key for key in text_model.state_dict())


def test_static_capture_refreshes_same_addresses_on_the_next_forward() -> None:
    model, text_model = _top_level_model()
    install_hidden_capture_hooks(model, [0], max_tokens=4)
    capture = model._omni_aux_hidden_capture
    pointer = capture.buffers[0].data_ptr()

    first = torch.ones(2, 3)
    second = torch.full((3, 3), 7.0)
    _run_layer(text_model.layers[0], first, None)
    torch.testing.assert_close(capture.views(2)[0], first)
    _run_layer(text_model.layers[0], second, None)

    assert capture.buffers[0].data_ptr() == pointer
    torch.testing.assert_close(capture.views(3)[0], second)


def test_static_capture_rejects_rows_beyond_capacity() -> None:
    model, text_model = _top_level_model()
    install_hidden_capture_hooks(model, [0], max_tokens=2)
    capture = model._omni_aux_hidden_capture

    with pytest.raises(RuntimeError, match="3 rows.*capacity is 2"):
        _run_layer(text_model.layers[0], torch.ones(3, 3), None)

    with pytest.raises(RuntimeError, match="3 rows.*capacity is 2"):
        capture.views(3)


def test_static_capture_views_rows_refreshed_by_graph_replay() -> None:
    model, text_model = _top_level_model()
    install_hidden_capture_hooks(model, [0], max_tokens=4)
    capture = model._omni_aux_hidden_capture

    # Graph buckets are captured largest-to-smallest, so the final Python hook
    # invocation may be for fewer rows than a later replay.
    _run_layer(text_model.layers[0], torch.zeros(1, 3), None)
    replayed = torch.arange(9, dtype=torch.float32).reshape(3, 3)

    # CUDA graph replay executes the captured device copy without rerunning the
    # Python hook.
    capture.buffers[0][:3].copy_(replayed)

    torch.testing.assert_close(capture.views(3)[0], replayed)


def test_static_capture_rejects_layer_input_dtype_mismatch() -> None:
    model, text_model = _top_level_model()
    install_hidden_capture_hooks(model, [0], max_tokens=4)

    with pytest.raises(RuntimeError, match="expected torch.float32 layer input"):
        _run_layer(text_model.layers[0], torch.ones(2, 3, dtype=torch.float64), None)
