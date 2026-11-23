# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_attn_backend,
)
from sglang.srt.model_executor.model_runner import ModelRunner as NativeModelRunner
from sglang.srt.model_executor.model_runner import ModelRunnerOutput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from sglang_omni.model_runner.sglang_execution import SGLangExecutionBridge
from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner


class FutureMap:
    def __init__(self) -> None:
        self.stashed = None
        self.published = None

    def stash(self, indices, payload) -> None:
        self.stashed = (indices, payload)

    def publish(self, indices, seq_lens) -> None:
        self.published = (indices, seq_lens)


def test_plain_decode_selects_its_backend_inside_an_existing_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller's verify context must not leak into ordinary decode."""
    target_runner = object.__new__(SGLModelRunner)
    plain_runner = object.__new__(SGLModelRunner)
    plain_runner.attn_backend = SimpleNamespace()
    target_runner.plain_decode_runner = plain_runner
    plain_runner.plain_decode_runner = None
    outer_backend = SimpleNamespace()
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        attn_backend=outer_backend,
        moe_num_token_non_padded=None,
    )
    expected_output = ModelRunnerOutput(
        logits_output=LogitsProcessorOutput(next_token_logits=torch.zeros(1, 2)),
        can_run_graph=False,
    )

    def native_forward(
        runner: NativeModelRunner,
        forward_batch: SimpleNamespace,
        pp_proxy_tensors: None,
        reinit_attn_backend: bool,
        split_forward_count: int,
    ) -> ModelRunnerOutput:
        assert runner is plain_runner
        assert forward_batch is batch
        assert pp_proxy_tensors is None
        assert not reinit_attn_backend and split_forward_count == 1
        assert get_attn_backend() is plain_runner.attn_backend
        assert batch.attn_backend is plain_runner.attn_backend
        return expected_output

    monkeypatch.setattr(NativeModelRunner, "_forward_raw", native_forward)
    with forward_context(ForwardContext(attn_backend=outer_backend)):
        assert (
            target_runner._forward_raw(batch, None) is expected_output
        )  # noqa: leading-underscore  # native forward interface
        assert get_attn_backend() is outer_backend


class SpecAlgorithm:
    def __init__(self, *, is_none: bool = True) -> None:
        self.is_none_value = is_none

    def is_none(self) -> bool:
        return self.is_none_value


def make_bridge(device: str = "cpu") -> tuple[SGLangExecutionBridge, FutureMap]:
    future_map = FutureMap()
    worker = SimpleNamespace(model_runner=SimpleNamespace())
    bridge = SGLangExecutionBridge(
        device=torch.device(device),
        worker=worker,
        spec_algorithm=SpecAlgorithm(),
        future_map=future_map,
    )
    return bridge, future_map


def test_publish_next_tokens_uses_future_map_and_retires_input_ids() -> None:
    bridge, future_map = make_bridge()
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
    bridge, _ = make_bridge()
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
    bridge, _ = make_bridge()
    original_sampling_info = SimpleNamespace(
        copy_for_forward=lambda: "forward-only-sampling-info"
    )
    batch = SimpleNamespace(
        sampling_info=original_sampling_info,
        mix_running_indices=None,
    )
    monkeypatch.setattr(
        "sglang.srt.managers.overlap_utils.resolve_forward_inputs",
        lambda *args: None,
    )

    with bridge.forward_context(batch, isolate_sampling=True):
        assert batch.sampling_info == "forward-only-sampling-info"

    assert batch.sampling_info is original_sampling_info


def test_execution_bridge_rejects_speculative_decoding() -> None:
    future_map = FutureMap()
    worker = SimpleNamespace(model_runner=SimpleNamespace())

    with pytest.raises(NotImplementedError, match="speculative decoding"):
        SGLangExecutionBridge(
            device=torch.device("cpu"),
            worker=worker,
            spec_algorithm=SpecAlgorithm(is_none=False),
            future_map=future_map,
        )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_record_completion_alternates_two_events() -> None:
    bridge, _ = make_bridge("cuda")

    events = [bridge.record_completion() for _ in range(4)]

    assert events[0] is events[2] and events[1] is events[3]
    assert events[0] is not events[1]
