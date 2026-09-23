# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for LLaDA2-Uni classifier-free guidance."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace as NS
from unittest.mock import create_autospec

import pytest
import torch
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferIndicesUpdaterPrefill,
)

import sglang_omni.models.llada2_uni.cfg_attention_backend as cfg_attention_backend
from sglang_omni.models.llada2_uni.low_confidence_cfg import LowConfidenceCFG
from sglang_omni.scheduling.dllm_scheduler import DllmScheduler


@dataclass
class RequestStub:
    rid: str
    dllm_phase: str = "decode"
    _is_uncond: bool = False
    _is_uncond_img: bool = False
    _cfg_group_rid: str | None = None
    _dllm_left_pad_len: int = 0

    def is_dllm_prefill(self) -> bool:
        return self.dllm_phase == "prefill"


class RaggedWrapperStub:
    is_cuda_graph_enabled = False

    def __init__(self) -> None:
        self._custom_mask_buf = None
        self._mask_indptr_buf = None
        self.plan = None

    def begin_forward(self, *args, **kwargs) -> None:
        self.plan = (args, kwargs)


def make_config(*, fdfo: bool = False) -> DllmConfig:
    return DllmConfig(
        algorithm="LowConfidenceCFG",
        algorithm_config={"threshold": 1.0, "image_token_offset": 3},
        block_size=4,
        mask_id=9,
        max_running_requests=3,
        first_done_first_out_mode=fdfo,
    )


def make_cfg_group(size: int) -> list[RequestStub]:
    requests = [RequestStub("cond")]
    for index in range(1, size):
        requests.append(
            RequestStub(
                f"cond-u{index}",
                _is_uncond=True,
                _is_uncond_img=index == 2,
            )
        )
    if size > 1:
        for request in requests:
            request._cfg_group_rid = "cond"
    return requests


@pytest.mark.parametrize("size", [1, 2, 3])
@pytest.mark.parametrize("threshold,rescale", [(1.0, 0.0), (0.2, 0.7)])
def test_guidance_updates_all_cfg_branches(
    size: int, threshold: float, rescale: float
) -> None:
    algorithm = LowConfidenceCFG(make_config())
    algorithm.threshold = threshold
    requests = make_cfg_group(size)
    requests[0]._task_kind = "edit" if size == 3 else "t2i"
    requests[0]._dllm_steps = 3
    requests[0]._cfg_scale = 2.0
    requests[0]._cfg_image_scale = 1.5
    requests[0]._cfg_rescale = rescale

    input_ids = torch.tensor([1, 9, 9, 9] * size)
    if size > 1:
        input_ids[4] = 9
    branch_logits = torch.tensor(
        [
            [100.0, 1.0, 1.0, 2.0, 1.0, 0.0],
            [100.0, 1.0, 1.0, 3.0, 0.0, 1.0],
            [100.0, 1.0, 1.0, 5.0, 2.0, 0.0],
        ]
    )[:size]
    guided_logits = branch_logits[0].clone()
    if size >= 2:
        guided_logits = branch_logits[1] + 2.0 * (branch_logits[0] - branch_logits[1])
    if size == 3:
        guided_logits += 1.5 * (branch_logits[1] - branch_logits[2])
    if size >= 2 and rescale:
        normalized = guided_logits * (
            branch_logits[0].std() / (guided_logits.std() + 1e-6)
        )
        guided_logits = rescale * normalized + (1 - rescale) * guided_logits
    guided_logits[:3] = -torch.inf

    def forward(batch, **_kwargs):
        return NS(
            logits_output=NS(
                full_logits=branch_logits.repeat_interleave(4, dim=0).clone()
            ),
            can_run_graph=False,
        )

    result = algorithm.run(
        NS(forward=forward),
        NS(input_ids=input_ids, batch_size=size, reqs=requests),
    )

    assert result[2:] == (None, None, False)
    assert len(result[1]) == size
    expected_id = guided_logits.argmax().item()
    assert all(row.tolist() == [expected_id] * 3 for row in result[1])
    if size > 1:
        assert input_ids[4].item() == 9


def test_cfg_prefill_only_builds_kv_state() -> None:
    algorithm = LowConfidenceCFG(make_config())
    requests = make_cfg_group(2)
    for request in requests:
        request.dllm_phase = "prefill"
    input_ids = torch.tensor([1, 2, 3, 4, 9, 9, 3, 4])
    original = input_ids.clone()
    calls = 0

    def forward(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return NS(logits_output=None, can_run_graph=False)

    result = algorithm.run(
        NS(forward=forward),
        NS(reqs=requests, batch_size=2, input_ids=input_ids),
    )

    assert result == (None, [], None, None, False)
    assert calls == 1
    torch.testing.assert_close(input_ids, original)


def test_invalid_cfg_batch_is_rejected() -> None:
    with pytest.raises(ValueError, match="FDFO"):
        LowConfidenceCFG(make_config(fdfo=True))

    algorithm = LowConfidenceCFG(make_config())
    requests = make_cfg_group(2)
    requests[1]._cfg_group_rid = "other"
    with pytest.raises(RuntimeError, match="Malformed CFG"):
        algorithm.run(None, NS(reqs=requests, batch_size=2))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_cfg_scale_is_rejected(value: float) -> None:
    algorithm = LowConfidenceCFG(make_config())
    requests = make_cfg_group(2)
    requests[0]._cfg_scale = value
    with pytest.raises(ValueError, match="_cfg_scale must be finite"):
        algorithm.run(None, NS(reqs=requests, batch_size=2))


def test_scheduler_applies_left_padding_metadata() -> None:
    scheduler = object.__new__(DllmScheduler)
    requests = make_cfg_group(2)
    requests[1]._dllm_left_pad_len = 6
    forward_batch = NS(
        positions=torch.arange(4, 8).repeat(2),
        extend_seq_lens_cpu=[4, 4],
        forward_mode=NS(is_extend=lambda: True),
    )

    scheduler.apply_cfg_padding_metadata(forward_batch, NS(reqs=requests))

    assert forward_batch.dllm_left_pad_lens_cpu == [0, 6]
    assert forward_batch.positions[4:].tolist() == [0, 0, 0, 1]


def test_attention_masks_local_and_cached_padding() -> None:
    backend = object.__new__(cfg_attention_backend.LLaDA2CFGFlashInferAttnBackend)
    backend._cfg_prefill_wrapper_ragged = RaggedWrapperStub()
    backend.prefill_wrappers_paged = [object()]
    backend.prefill_split_tile_size = None
    begin_forward = create_autospec(
        FlashInferIndicesUpdaterPrefill, instance=True
    ).call_begin_forward
    backend.indices_updater_prefill = NS(
        call_begin_forward=begin_forward,
        kv_indptr=[None],
        qo_indptr=[None],
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim=2,
        q_data_type=torch.float32,
        data_type=torch.float32,
        prefill_wrapper_ragged=RaggedWrapperStub(),
    )
    batch = NS(
        dllm_left_pad_lens_cpu=[0, 6],
        extend_prefix_lens_cpu=[4, 4],
        extend_seq_lens_cpu=[4, 4],
        forward_mode=NS(is_dllm_extend=lambda: True),
        seq_lens=torch.tensor([8, 8]),
        extend_prefix_lens=torch.tensor([4, 4]),
        req_pool_indices=torch.tensor([0, 1]),
    )

    backend.init_forward_metadata(batch)

    args = begin_forward.call_args.args
    assert args[3].tolist() == [4, 0]
    assert args[7].tolist() == [0, 4]
    mask = backend._cfg_prefill_wrapper_ragged.plan[1]["custom_mask"].reshape(2, 4, 4)
    assert mask[0].all()
    assert not mask[1, 2:, :2].any()
    assert mask[1, 0, 0] and mask[1, 1, 1]
    assert mask[1, :, 2:].all()

    batch.extend_prefix_lens_cpu = [8, 8]
    batch.seq_lens += 4
    batch.extend_prefix_lens += 4
    backend.init_forward_metadata(batch)
    assert not backend._cfg_local_left_pad_active
    assert begin_forward.call_args.args[3].tolist() == [8, 2]


@pytest.mark.parametrize("cached", [False, True])
def test_attention_writes_kv_for_masked_extend(
    monkeypatch: pytest.MonkeyPatch, cached: bool
) -> None:
    backend = object.__new__(cfg_attention_backend.LLaDA2CFGFlashInferAttnBackend)
    backend._cfg_local_left_pad_active = True
    backend._cfg_has_cached_prefix = cached
    backend.forward_metadata = NS(swa_out_cache_loc=None)
    current = torch.ones(4, 1, 2)
    prefix = torch.full_like(current, 2)
    backend._cfg_prefill_wrapper_ragged = NS(
        forward=lambda *_args, **_kwargs: current,
        forward_return_lse=lambda *_args, **_kwargs: (current, torch.zeros(4, 1)),
    )
    backend.prefill_wrappers_paged = [
        NS(
            forward_return_lse=lambda *_args, **_kwargs: (
                prefix,
                torch.zeros(4, 1),
            )
        )
    ]
    monkeypatch.setattr(
        cfg_attention_backend,
        "merge_state",
        lambda output, output_lse, cached_output, _cached_lse: (
            (output + cached_output) / 2,
            output_lse,
        ),
    )
    writes = []
    backend.token_to_kv_pool = NS(
        get_kv_buffer=lambda _layer_id: object(),
        set_kv_buffer=lambda *args: writes.append(args),
    )
    backend._kv_write_scales = lambda _layer: (None, None)
    layer = NS(
        tp_q_head_num=1,
        tp_k_head_num=1,
        tp_v_head_num=1,
        head_dim=2,
        scaling=1.0,
        logit_cap=0.0,
        layer_id=0,
        k_scale_float=None,
        v_scale_float=None,
        is_cross_attention=False,
    )
    batch = NS(out_cache_loc=torch.arange(4))

    result = backend.forward_extend(current, current, current, layer, batch)

    expected = current if not cached else (current + prefix) / 2
    torch.testing.assert_close(result, expected.view(4, 2))
    assert len(writes) == 1
    assert torch.equal(writes[0][1].loc, batch.out_cache_loc)
