# SPDX-License-Identifier: Apache-2.0
"""SGLang output processor tensor slicing tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

OUTPUT_PROCESSOR_PATH = Path(
    "sglang_omni/scheduling/sglang_backend/output_processor.py"
)


def _load_output_processor_cls():
    module_name = "_sglang_output_processor_under_test"
    spec = importlib.util.spec_from_file_location(module_name, OUTPUT_PROCESSOR_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    sys.modules.pop(module_name, None)
    return module.SGLangOutputProcessor


def _scheduler_output(*extend_lens):
    from sglang_omni.scheduling.types import SchedulerOutput, SchedulerRequest

    return SchedulerOutput(
        requests=[
            SchedulerRequest(request_id=f"req-{idx}")
            for idx, _ in enumerate(extend_lens)
        ],
        batch_data=SimpleNamespace(
            reqs=[
                SimpleNamespace(extend_input_len=extend_len)
                for extend_len in extend_lens
            ]
        ),
    )


def test_slice_single_request_full_prefill_hidden_states_keeps_sequence() -> None:
    torch = pytest.importorskip("torch")
    SGLangOutputProcessor = _load_output_processor_cls()

    tensor = torch.arange(6).reshape(3, 2)

    sliced = SGLangOutputProcessor._slice_per_request_tensor(
        tensor,
        request_index=0,
        scheduler_output=_scheduler_output(3),
    )

    assert torch.equal(sliced, tensor)


def test_slice_single_request_batched_row_hidden_states_keeps_old_behavior() -> None:
    torch = pytest.importorskip("torch")
    SGLangOutputProcessor = _load_output_processor_cls()

    tensor = torch.tensor([[10, 20]])

    sliced = SGLangOutputProcessor._slice_per_request_tensor(
        tensor,
        request_index=0,
        scheduler_output=_scheduler_output(1),
    )

    assert torch.equal(sliced, tensor[0])


def test_slice_multi_request_batched_rows_take_request_row() -> None:
    torch = pytest.importorskip("torch")
    SGLangOutputProcessor = _load_output_processor_cls()

    tensor = torch.tensor([[10, 20], [30, 40]])

    sliced = SGLangOutputProcessor._slice_per_request_tensor(
        tensor,
        request_index=1,
        scheduler_output=_scheduler_output(2, 3),
    )

    assert torch.equal(sliced, tensor[1])
