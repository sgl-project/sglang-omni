# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sglang_omni.utils.graph_padding import pad_rows, select_padded_graph


def test_select_padded_graph_uses_smallest_batch_then_capacity() -> None:
    graphs = {(8, 32): "8x32", (8, 16): "8x16", (16, 16): "16x16"}

    assert select_padded_graph(graphs, 5, 12) == ("8x16", 3)
    assert select_padded_graph(graphs, 8, 12) == ("16x16", 8)
    assert select_padded_graph(graphs, 16, 12) == (None, 0)


def test_select_padded_graph_can_replace_skipped_positional_capture() -> None:
    graphs = {(8, 16): "positional"}
    gather_twins = {(8, 16): "gather"}

    assert select_padded_graph(graphs, 5, 12, skip_batch=8, extra=gather_twins) == (
        "gather",
        3,
    )


def test_pad_rows_preserves_tensor_properties_and_fill_value() -> None:
    source = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)

    padded = pad_rows(source, 2, fill_value=9)

    assert padded.dtype == source.dtype
    assert padded.device == source.device
    assert padded.tolist() == [[1, 2], [3, 4], [9, 9], [9, 9]]
    assert pad_rows(source, 0) is source
