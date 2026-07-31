# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner.sglang_execution import SGLangExecutionBridge


class _FutureMap:
    def __init__(self) -> None:
        self.stashed = None
        self.published = None

    def stash(self, indices, payload) -> None:
        self.stashed = (indices, payload)

    def publish(self, indices, seq_lens) -> None:
        self.published = (indices, seq_lens)


class _SpecAlgorithm:
    def __init__(self, future_map: _FutureMap, *, is_none: bool = True) -> None:
        self.future_map = future_map
        self._is_none = is_none

    def create_future_map(self, device, req_to_token_pool, needs_cpu_seq_lens):
        assert device == torch.device("cpu")
        assert req_to_token_pool == "pool"
        assert needs_cpu_seq_lens is True
        return self.future_map

    def is_none(self) -> bool:
        return self._is_none


def _make_bridge() -> tuple[SGLangExecutionBridge, _FutureMap]:
    future_map = _FutureMap()
    worker = SimpleNamespace(model_runner=SimpleNamespace())
    bridge = SGLangExecutionBridge(
        device=torch.device("cpu"),
        worker=worker,
        req_to_token_pool="pool",
        spec_algorithm=_SpecAlgorithm(future_map),
    )
    return bridge, future_map


def test_publish_next_tokens_uses_future_map_and_retires_input_ids() -> None:
    bridge, future_map = _make_bridge()
    batch = SimpleNamespace(
        req_pool_indices=torch.tensor([4, 7]),
        seq_lens=torch.tensor([12, 19]),
        input_ids=torch.tensor([1, 2]),
    )
    next_token_ids = torch.tensor([31, 32])

    bridge.publish_next_tokens(batch, next_token_ids)

    stash_indices, payload = future_map.stashed
    assert torch.equal(stash_indices, batch.req_pool_indices)
    assert torch.equal(payload.bonus_tokens, next_token_ids)
    # new_seq_lens_buf has no non-spec reader; the bridge must not publish.
    assert future_map.published is None
    assert batch.input_ids is None


def test_forward_context_resolves_inputs_without_copying_sampling_info(
    monkeypatch,
) -> None:
    bridge, _ = _make_bridge()
    original_sampling_info = SimpleNamespace(
        copy_for_forward=lambda: "forward-only-sampling-info"
    )
    batch = SimpleNamespace(
        sampling_info=original_sampling_info,
        mix_running_indices=None,
    )
    seen = []

    def resolve_forward_inputs(actual_batch, future_map) -> None:
        seen.append((actual_batch, future_map))

    monkeypatch.setattr(
        "sglang.srt.managers.overlap_utils.resolve_forward_inputs",
        resolve_forward_inputs,
    )

    with bridge.forward_context(batch):
        assert batch.sampling_info is original_sampling_info

    assert seen == [(batch, bridge.future_map)]
    assert batch.sampling_info is original_sampling_info


def test_forward_context_isolates_sampling_info_for_lookahead(monkeypatch) -> None:
    bridge, _ = _make_bridge()
    original_sampling_info = SimpleNamespace(
        copy_for_forward=lambda: "forward-only-sampling-info"
    )
    batch = SimpleNamespace(
        sampling_info=original_sampling_info,
        mix_running_indices=None,
    )
    monkeypatch.setattr(
        "sglang.srt.managers.overlap_utils.resolve_forward_inputs",
        lambda *_args: None,
    )

    with bridge.forward_context(batch, isolate_sampling=True):
        assert batch.sampling_info == "forward-only-sampling-info"

    assert batch.sampling_info is original_sampling_info


def test_forward_context_resolves_mixed_prefill(monkeypatch) -> None:
    bridge, _ = _make_bridge()
    batch = SimpleNamespace(
        sampling_info=None,
        mix_running_indices=torch.tensor([1]),
    )
    resolved = []
    monkeypatch.setattr(
        "sglang.srt.managers.overlap_utils.resolve_forward_inputs",
        lambda *_args: resolved.append(True),
    )

    with bridge.forward_context(batch):
        pass

    assert resolved == [True]


def test_execution_bridge_rejects_speculative_decoding() -> None:
    future_map = _FutureMap()
    worker = SimpleNamespace(model_runner=SimpleNamespace())

    with pytest.raises(NotImplementedError, match="speculative decoding"):
        SGLangExecutionBridge(
            device=torch.device("cpu"),
            worker=worker,
            req_to_token_pool="pool",
            spec_algorithm=_SpecAlgorithm(future_map, is_none=False),
        )
