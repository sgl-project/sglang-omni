# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.easymagpie_tts.engine_builder import (
    EasyMagpieTTSEngineBuilder,
    decode_graph_batch_sizes,
)
from sglang_omni.models.easymagpie_tts.payload_types import MAX_TEXT_TOKENS
from sglang_omni.models.easymagpie_tts.speakers import SPEAKER_SUBDIR


@pytest.mark.parametrize(
    ("max_batch", "sizes"),
    [
        (1, [1]),
        (6, [1, 2, 4, 6]),
        (8, [1, 2, 4, 8]),
        (128, [1, 2, 4, 8, 16, 32, 64, 128]),
    ],
)
def test_decode_graph_sizes_end_at_the_running_limit(max_batch, sizes) -> None:
    assert decode_graph_batch_sizes(max_batch) == sizes


def test_decode_graphs_are_on_by_default_and_prefill_graphs_off() -> None:
    builder = EasyMagpieTTSEngineBuilder(max_running_requests=8)
    defaults = builder.generation_defaults(dtype="float16")
    assert defaults["disable_cuda_graph"] is False
    assert defaults["disable_prefill_cuda_graph"] is True
    assert defaults["cuda_graph_max_bs"] == 8
    assert defaults["cuda_graph_bs"] == [1, 2, 4, 8]
    eager = EasyMagpieTTSEngineBuilder(cuda_graph=False).generation_defaults(
        dtype="float16"
    )
    assert eager["disable_cuda_graph"] is True


def test_graph_buckets_follow_a_stage_running_limit_override() -> None:
    builder = EasyMagpieTTSEngineBuilder()
    overrides = {"max_running_requests": 4}
    builder.adjust_overrides(overrides)
    assert overrides["cuda_graph_max_bs"] == 4
    assert overrides["cuda_graph_bs"] == [1, 2, 4]
    with pytest.raises(ValueError, match="tp_size"):
        builder.adjust_overrides({"tp_size": 2})


def test_setup_model_sizes_decode_state_and_loads_voices(talker, tmp_path) -> None:
    voices = tmp_path / SPEAKER_SUBDIR
    voices.mkdir()
    torch.save(torch.ones(3, 8), voices / "eng.pt")
    torch.save({"speaker_encoding": torch.zeros(2, 8)}, voices / "alt.pt")
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            model=talker, req_to_token_pool=SimpleNamespace(size=9)
        )
    )
    EasyMagpieTTSEngineBuilder(max_running_requests=6).setup_model(
        model_worker=worker,
        checkpoint_dir=str(tmp_path),
        device="cpu",
        gpu_id=0,
        server_args=SimpleNamespace(max_running_requests=3),
    )
    state = talker.decode_state
    assert (state.max_batch, state.num_slots) == (6, 9)
    assert state.text.shape[1] == MAX_TEXT_TOKENS
    speakers = talker.speaker_table
    assert speakers.spans == {"alt": (0, 2), "eng": (2, 3)}
    assert speakers.rows.dtype == next(talker.parameters()).dtype
    torch.testing.assert_close(
        speakers.rows.sum(dim=1), torch.tensor([0.0] * 2 + [8.0] * 3)
    )


def test_async_decode_is_on_from_batch_one_by_default() -> None:
    kwargs = EasyMagpieTTSEngineBuilder().extra_scheduler_kwargs()
    assert kwargs["enable_async_decode"] is True
    assert kwargs["async_decode_min_batch_size"] == 1
    sync = EasyMagpieTTSEngineBuilder(enable_async_decode=False)
    assert sync.extra_scheduler_kwargs()["enable_async_decode"] is False
