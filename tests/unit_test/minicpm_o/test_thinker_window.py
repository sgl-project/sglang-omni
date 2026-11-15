# SPDX-License-Identifier: Apache-2.0
"""The thinker history window: which units the checkpoint's basic window drops, and the in-place cut of an SGLang session."""

import random
from array import array
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from pydantic import ValidationError
from sglang.srt.managers.io_struct import SessionParams, TokenizedGenerateReqInput
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    NoOpMHATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session

from sglang_omni.audio.vad import StatelessVAD
from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.minicpm_o.engine_builder import check_window_kv_cache
from sglang_omni.models.minicpm_o.native_config import (
    MiniCPMODuplexPipelineConfig,
    MiniCPMODuplexSampling,
    MiniCPMODuplexWindow,
)
from sglang_omni.models.minicpm_o.session_adapters import ThinkerAdapter
from sglang_omni.models.minicpm_o.special_tokens import REQUIRED_SPECIAL_TOKENS
from sglang_omni.models.minicpm_o.thinker_state import MiniCPMOThinkerSessionState
from sglang_omni.proto.session import SessionIdentity
from sglang_omni.scheduling.sglang_backend.history_cut import (
    HistoryCut,
    cut_retained_kv,
    cut_token_ids,
    retained_history_length,
)
from tests.unit_test.fixtures.accelerator import require_cuda

IDENTITY = SessionIdentity("window")
HEAD_DIM = 128
ROPE_THETA = 1_000_000.0


def make_adapter(window: MiniCPMODuplexWindow) -> ThinkerAdapter:
    tokenizer = Mock(unk_token_id=0, bad_token_ids=[])
    tokenizer.convert_tokens_to_ids.side_effect = dict(
        zip(REQUIRED_SPECIAL_TOKENS, range(1, len(REQUIRED_SPECIAL_TOKENS) + 1))
    ).__getitem__
    # note (0xtoward): SGLang's RotaryEmbedding imports a vLLM kernel when no GPU is visible, so the tests build its cos_sin_cache rows.
    angles = torch.outer(
        torch.arange(8192, dtype=torch.float32),
        1.0
        / ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM),
    )
    adapter = ThinkerAdapter(
        tokenizer,
        64,
        window=window,
        rotary_embedding=SimpleNamespace(
            head_size=HEAD_DIM,
            rotary_dim=HEAD_DIM,
            is_neox_style=True,
            cos_sin_cache=torch.cat((angles.cos(), angles.sin()), dim=-1),
        ),
    )
    adapter.states[IDENTITY] = MiniCPMOThinkerSessionState(
        sampling=MiniCPMODuplexSampling(), speech_detector=Mock(spec=StatelessVAD)
    )
    return adapter


def checkpoint_kept_units(
    prefix_length: int, unit_lengths: list[int], high: int, low: int
) -> list[list[int]]:
    """Units StreamDecoder.enforce_window in the checkpoint keeps after each unit ends."""
    kept: list[tuple[int, int]] = []
    cache_length = prefix_length
    kept_after_each_unit = []
    for unit_index, unit_length in enumerate(unit_lengths):
        kept.append((unit_index, unit_length))
        cache_length += unit_length
        if cache_length > high:
            while cache_length > low and kept:
                cache_length -= kept.pop(0)[1]
        else:
            pass
        kept_after_each_unit.append([index for index, _ in kept])
    return kept_after_each_unit


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_window_keeps_the_units_the_checkpoint_window_keeps(seed: int) -> None:
    """A unit's request starts with the last unit's </unit>, so the next <unit> lands where the checkpoint checks its cache."""
    window = MiniCPMODuplexWindow(
        basic_window_high_tokens=256, basic_window_low_tokens=192
    )
    adapter = make_adapter(window)
    state = adapter.states[IDENTITY]
    random_source = random.Random(seed)
    prefix_length = 17
    unit_lengths = [random_source.randint(13, 160) for _ in range(60)]
    expected = checkpoint_kept_units(prefix_length, unit_lengths, 256, 192)
    kept: list[int] = []
    for unit_index in range(len(unit_lengths) + 1):
        unit_token_index = prefix_length if unit_index == 0 else 1
        history_length = (
            0
            if unit_index == 0
            else prefix_length + sum(unit_lengths[index] for index in kept) - 1
        )
        cut = adapter.plan_history_cut(
            IDENTITY,
            SimpleNamespace(unit_token_index=unit_token_index),
            history_length,
            history_capacity=4096,
        )
        if cut is not None:
            starts = [prefix_length]
            for index in kept[:-1]:
                starts.append(starts[-1] + unit_lengths[index])
            kept = [
                index
                for index, start in zip(kept, starts)
                if not cut.drops(start, start + unit_lengths[index])
            ]
            history_length -= cut.length
        else:
            pass
        if unit_index > 0:
            assert kept == expected[unit_index - 1]
        else:
            pass
        if unit_index < len(unit_lengths):
            state.unit_starts.append(history_length + unit_token_index)
            kept.append(unit_index)
        else:
            pass


def test_window_cuts_early_when_the_next_unit_would_not_fit() -> None:
    adapter = make_adapter(MiniCPMODuplexWindow())
    adapter.states[IDENTITY].unit_starts = [10, 110, 210, 310]
    request = SimpleNamespace(unit_token_index=1)

    assert adapter.plan_history_cut(IDENTITY, request, 409, 500) is None
    cut = adapter.plan_history_cut(IDENTITY, request, 409, 250)

    assert cut == HistoryCut(((10, 210),))
    assert adapter.states[IDENTITY].unit_starts == [10, 110]


def test_window_key_shift_rounds_the_exact_rotation_once() -> None:
    adapter = make_adapter(MiniCPMODuplexWindow())
    keys = torch.randn(64, 2, HEAD_DIM, generator=torch.Generator().manual_seed(0))
    distances = torch.randint(1, 8000, (64,))
    angles = distances.to(torch.float64)[:, None, None] * ROPE_THETA ** (
        -torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM
    )
    first, second = keys.to(torch.bfloat16).to(torch.float64).chunk(2, dim=-1)
    exact = torch.cat(
        (
            first * angles.cos() + second * angles.sin(),
            second * angles.cos() - first * angles.sin(),
        ),
        dim=-1,
    )

    shifted = adapter.shift_key_positions(keys.to(torch.bfloat16), distances)

    assert shifted.dtype == torch.bfloat16
    error = (shifted.to(torch.float64) - exact).norm() / exact.norm()
    assert error <= 2**-8


def tokenized(rid: str, input_ids: list[int]) -> TokenizedGenerateReqInput:
    return TokenizedGenerateReqInput(
        rid=rid,
        input_text=None,
        input_ids=array("q", input_ids),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=3),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        session_params=SessionParams(id="session"),
    )


def test_cut_token_ids_follow_the_sglang_streaming_session() -> None:
    """The bridge's history length and cut match what Session.create_req puts ahead of the next request."""
    session = Session(0, "session", streaming=True)
    first = session.create_req(tokenized("a", list(range(100, 120))), None, 1000)
    first.output_ids = [200, 201, 202, 203]
    first.full_untruncated_fill_ids = array("q", [*range(100, 120), 200, 201, 202])
    session.finish_req(first)

    assert retained_history_length(session) == 23
    cut_token_ids(session, HistoryCut(((2, 5), (10, 12))))
    second = session.create_req(tokenized("b", [300, 301]), None, 1000)

    kept = [
        *range(100, 102),
        *range(105, 110),
        *range(112, 120),
        200,
        201,
        202,
    ]
    assert list(second.origin_input_ids) == [*kept, 300, 301]
    assert len(second.origin_input_ids) - 2 == 23 - 5


def make_kv_cache(dtype: torch.dtype, device: str, **options) -> MHATokenToKVPool:
    return MHATokenToKVPool(
        size=256,
        page_size=1,
        dtype=dtype,
        head_num=2,
        head_dim=HEAD_DIM,
        layer_num=2,
        device=device,
        enable_memory_saver=False,
        **options,
    )


def storage(buffer: torch.Tensor) -> torch.Tensor:
    return buffer.view(torch.uint8) if buffer.dtype.itemsize == 1 else buffer


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.accelerator),
    ],
)
def test_cut_moves_the_kept_kv_down_the_session_row(
    dtype: torch.dtype, device: str
) -> None:
    if device == "cuda":
        require_cuda()
    else:
        pass
    kv_cache = make_kv_cache(dtype, device)
    request_rows = ReqToTokenPool(
        size=1, max_context_len=128, device=device, enable_memory_saver=False
    )
    generator = torch.Generator().manual_seed(0)
    slots = torch.randperm(255, generator=generator)[:104] + 1
    request_rows.req_to_token[0, :104] = slots.to(torch.int32).to(device)
    for layer_id in range(2):
        for buffer in (
            kv_cache.get_key_buffer(layer_id),
            kv_cache.get_value_buffer(layer_id),
        ):
            storage(buffer).copy_(
                storage(torch.randn(buffer.shape, generator=generator).to(dtype)).to(
                    device
                )
            )
    before = [
        (
            storage(kv_cache.get_key_buffer(layer)).clone(),
            storage(kv_cache.get_value_buffer(layer)).clone(),
        )
        for layer in range(2)
    ]
    kv = SimpleNamespace(
        req_pool_idx=0,
        cache_protected_len=0,
        kv_committed_len=100,
        kv_allocated_len=104,
    )
    rows = SimpleNamespace(
        req_to_token_pool=request_rows, page_size=1, free_kv_row=Mock()
    )
    shifts = []

    def record_shift(keys: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
        shifts.append(distances.cpu())
        return keys

    cut_retained_kv(rows, kv_cache, kv, HistoryCut(((10, 30), (50, 60))), record_shift)

    kept = [*range(10), *range(30, 50), *range(60, 100)]
    old_slots = slots[kept].to(torch.int64).to(device)
    new_slots = request_rows.req_to_token[0, :70].to(torch.int64)
    assert torch.equal(request_rows.req_to_token[0, :104].cpu(), slots.to(torch.int32))
    for layer_id, (keys, values) in enumerate(before):
        assert torch.equal(
            storage(kv_cache.get_key_buffer(layer_id))[new_slots], keys[old_slots]
        )
        assert torch.equal(
            storage(kv_cache.get_value_buffer(layer_id))[new_slots], values[old_slots]
        )
    assert all(
        torch.equal(distances, torch.tensor([20] * 20 + [30] * 40))
        for distances in shifts
    )
    rows.free_kv_row.assert_called_once_with(kv, [(70, 104)])
    assert (kv.kv_committed_len, kv.kv_allocated_len) == (70, 70)


@pytest.mark.parametrize(
    ("kv_cache", "refused"),
    [
        (make_kv_cache(torch.bfloat16, "cpu"), False),
        (make_kv_cache(torch.float8_e4m3fn, "cpu"), False),
        (make_kv_cache(torch.bfloat16, "cpu", kv_cache_layout="hnd"), True),
        (
            NoOpMHATokenToKVPool(
                size=16,
                page_size=1,
                dtype=torch.bfloat16,
                head_num=2,
                head_dim=HEAD_DIM,
                layer_num=1,
                device="cpu",
                enable_memory_saver=False,
            ),
            True,
        ),
    ],
    ids=["bf16", "fp8", "hnd", "subclass"],
)
def test_window_refuses_a_kv_cache_whose_rows_it_cannot_move(
    kv_cache: MHATokenToKVPool, refused: bool
) -> None:
    check_window_kv_cache(MiniCPMODuplexWindow(sliding_window_mode="off"), kv_cache)
    if refused:
        with pytest.raises(ValueError, match="window.sliding_window_mode off"):
            check_window_kv_cache(MiniCPMODuplexWindow(), kv_cache)
    else:
        check_window_kv_cache(MiniCPMODuplexWindow(), kv_cache)


def test_window_settings_reach_the_thinker_stage() -> None:
    manager = ConfigManager(MiniCPMODuplexPipelineConfig(model_path="unused"))
    config = manager.merge_config(
        manager.parse_extra_args(
            [
                "--window.sliding_window_mode",
                "off",
                "--window.basic_window_high_tokens",
                "7000",
                "--window.basic_window_low_tokens",
                "5000",
            ]
        )
    )

    kwargs = config.stage_factory_kwargs("thinker")

    assert kwargs["sliding_window_mode"] == "off"
    assert (kwargs["basic_window_high_tokens"], kwargs["basic_window_low_tokens"]) == (
        7000,
        5000,
    )
    with pytest.raises(ValidationError, match="must be below"):
        MiniCPMODuplexWindow(
            basic_window_high_tokens=6000, basic_window_low_tokens=6000
        )
