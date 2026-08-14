# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest
import torch

try:
    _has_sglang = importlib.util.find_spec("sglang") is not None
except ValueError:
    _has_sglang = False

if not _has_sglang:
    module_names = (
        "sglang",
        "sglang.srt",
        "sglang.srt.layers",
        "sglang.srt.layers.attention",
        "sglang.srt.layers.attention.attention_registry",
        "sglang.srt.layers.attention.flashinfer_backend",
        "sglang.srt.mem_cache",
        "sglang.srt.mem_cache.memory_pool",
    )
    for module_name in module_names:
        sys.modules.setdefault(module_name, ModuleType(module_name))

    class _FlashInferAttnBackend:
        def init_cuda_graph_state(self, *args, **kwargs) -> None:
            pass

        def init_forward_metadata(self, forward_batch) -> None:
            self.base_eager_batch = forward_batch

        def init_forward_metadata_out_graph(
            self, forward_batch, in_capture: bool = False
        ) -> None:
            self.base_graph_batch = forward_batch

    @dataclass
    class _PrefillMetadata:
        prefill_wrappers: list
        use_ragged: bool
        extend_no_prefix: bool

    flashinfer_backend = sys.modules["sglang.srt.layers.attention.flashinfer_backend"]
    flashinfer_backend.FlashInferAttnBackend = _FlashInferAttnBackend
    flashinfer_backend.PrefillMetadata = _PrefillMetadata
    flashinfer_backend._safe_merge_state = lambda *args: (args[0], args[1])
    sys.modules["sglang.srt.layers.attention.attention_registry"].ATTENTION_BACKENDS = (
        {}
    )
    sys.modules["sglang.srt.mem_cache.memory_pool"].KVWriteLoc = tuple

if "flashinfer" not in sys.modules and importlib.util.find_spec("flashinfer") is None:
    flashinfer = ModuleType("flashinfer")
    flashinfer.BatchPrefillWithRaggedKVCacheWrapper = object
    sys.modules["flashinfer"] = flashinfer

from sglang_omni.models.llada2_uni.cfg_attention_backend import (
    LLaDA2CFGFlashInferAttnBackend,
)


class _DllmExtendMode:
    @staticmethod
    def is_dllm_extend() -> bool:
        return True


def test_cfg_attention_separates_cached_and_in_query_left_padding() -> None:
    forward_batch = SimpleNamespace(
        batch_size=3,
        dllm_left_pad_lens_cpu=(0, 32, 40),
        extend_prefix_lens_cpu=[32, 32, 32],
        extend_seq_lens_cpu=[32, 32, 32],
    )

    lengths = LLaDA2CFGFlashInferAttnBackend._get_cfg_attention_lengths(forward_batch)

    assert lengths.local_left_pad_lengths == (0, 0, 8)
    assert lengths.cached_left_pad_lengths == (0, 32, 32)
    assert lengths.paged_kernel_num_tokens == 32


@pytest.mark.parametrize("query_length", [0, -1])
def test_cfg_attention_rejects_non_positive_active_block(query_length: int) -> None:
    forward_batch = SimpleNamespace(
        batch_size=2,
        dllm_left_pad_lens_cpu=(0, 1),
        extend_prefix_lens_cpu=[32, 32],
        extend_seq_lens_cpu=[32, query_length],
    )

    with pytest.raises(RuntimeError, match="empty active block"):
        LLaDA2CFGFlashInferAttnBackend._get_cfg_attention_lengths(forward_batch)


def test_cfg_cuda_graph_replay_uses_real_prefix_and_pads_capture_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_wrapper = object()
    ragged_wrapper = object()
    observed: dict[str, object] = {}

    def call_begin_forward(*args, **kwargs) -> None:
        observed["args"] = args
        observed["kwargs"] = kwargs

    backend = object.__new__(LLaDA2CFGFlashInferAttnBackend)
    backend._cfg_runtime_forward_batch = SimpleNamespace(
        batch_size=3,
        extend_prefix_lens=torch.tensor([64, 64, 64], dtype=torch.int32),
        extend_prefix_lens_cpu=[64, 64, 64],
        extend_seq_lens_cpu=[32, 32, 32],
        dllm_left_pad_lens=torch.tensor([0, 32, 64], dtype=torch.int32),
        dllm_left_pad_lens_cpu=(0, 32, 64),
    )
    backend.prefill_cuda_graph_metadata = {4: [capture_wrapper]}
    backend.indices_updater_prefill = SimpleNamespace(
        prefill_wrapper_ragged=ragged_wrapper,
        kv_indptr=[object()],
        qo_indptr=[object()],
        call_begin_forward=call_begin_forward,
    )
    backend.num_wrappers = 1
    backend.use_paged = False
    backend.prefill_split_tile_size = None
    backend._cfg_cuda_graph_prefix_lens = torch.empty(4, dtype=torch.int32)
    backend._cfg_cuda_graph_cached_left_pad_lens = torch.empty(4, dtype=torch.int32)
    backend._cfg_cuda_graph_paged_kernel_lens = torch.empty(4, dtype=torch.int32)

    def reject_tensor_item(_self):
        raise AssertionError("CFG metadata replay must not call Tensor.item()")

    replay_batch = SimpleNamespace(
        batch_size=4,
        forward_mode=_DllmExtendMode(),
        req_pool_indices=torch.tensor([0, 1, 2, 0], dtype=torch.int32),
        seq_lens=torch.tensor([96, 96, 96, 32], dtype=torch.int32),
    )
    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "item", reject_tensor_item)
        backend.init_forward_metadata_out_graph(replay_batch)

    args = observed["args"]
    assert isinstance(args, tuple)
    assert args[0] is ragged_wrapper
    assert args[1] is capture_wrapper
    assert args[2].tolist() == [0, 1, 2, 0]
    assert args[3].tolist() == [64, 32, 0, 0]
    assert args[4] == 96
    assert args[5].tolist() == [96, 96, 96, 32]
    assert args[6].tolist() == [64, 64, 64, 0]
    assert args[7].tolist() == [0, 32, 64, 0]
    assert observed["kwargs"] == {"fixed_split_size": None}


def test_cfg_cuda_graph_state_allocates_capture_sized_padding_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = LLaDA2CFGFlashInferAttnBackend.__mro__[1]
    monkeypatch.setattr(base, "init_cuda_graph_state", lambda *args, **kwargs: None)
    backend = object.__new__(LLaDA2CFGFlashInferAttnBackend)
    backend.skip_prefill = False
    backend.is_dllm_model = True
    backend.num_wrappers = 1
    backend.use_paged = False
    backend.workspace_buffer = torch.empty(1)

    backend.init_cuda_graph_state(max_bs=4, max_num_tokens=128)

    for attribute in (
        "_cfg_cuda_graph_prefix_lens",
        "_cfg_cuda_graph_cached_left_pad_lens",
        "_cfg_cuda_graph_paged_kernel_lens",
    ):
        buffer = getattr(backend, attribute)
        assert buffer.shape == (4,)
        assert buffer.dtype == torch.int32
