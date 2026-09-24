# SPDX-License-Identifier: Apache-2.0
import sys
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_asr.encoder_cuda_graph import (
    EncoderGraphUnrecoverableError,
    NpuGraphCaptureAttention,
    NpuGraphCaptureContext,
    Qwen3ASREncoderLayerStackGraphRunner,
)


class _NpuBackend:
    supports_graph_task_update = True

    def replay(self, graph):
        graph.replay()


class _NpuDeviceModule:
    def set_device(self, device):
        pass

    def current_stream(self, device=None):
        return "encoder-compute"

    def stream(self, stream):
        return nullcontext()


def _npu_runner():
    runner = object.__new__(Qwen3ASREncoderLayerStackGraphRunner)
    runner.max_seqlen = 8
    runner.buckets = (8,)
    runner.failed = set()
    runner.graphs = {}
    runner.device = SimpleNamespace(type="npu", index=0)
    runner.graph_backend = _NpuBackend()
    runner.device_module = _NpuDeviceModule()
    runner.npu_update_stream = SimpleNamespace(wait_stream=lambda stream: None)
    return runner


@pytest.mark.parametrize("supports_graph_task_update", [True, False])
def test_runner_owns_update_stream_on_model_device(
    monkeypatch, supports_graph_task_update
):
    created = []

    def create_stream(*, device):
        stream = object()
        created.append((device, stream))
        return stream

    device_module = SimpleNamespace(Stream=create_stream)
    devices = []

    def get_device_module(device):
        devices.append(device)
        return device_module

    monkeypatch.setattr(torch, "get_device_module", get_device_module)
    runners = []
    for device in (torch.device("meta", 0), torch.device("meta", 1)):
        tower = SimpleNamespace(
            parameters=lambda device=device: iter(
                [SimpleNamespace(device=device, dtype=torch.float32)]
            ),
            config=SimpleNamespace(n_window=50, n_window_infer=100),
        )
        runners.append(
            Qwen3ASREncoderLayerStackGraphRunner(
                tower,
                buckets=(128,),
                max_batch_size=8,
                graph_backend=SimpleNamespace(
                    supports_graph_task_update=supports_graph_task_update
                ),
            )
        )
    assert devices == [torch.device("meta", 0), torch.device("meta", 1)]
    if supports_graph_task_update:
        assert [device for device, _ in created] == devices
        assert runners[0].npu_update_stream is created[0][1]
        assert runners[1].npu_update_stream is created[1][1]
        assert runners[0].npu_update_stream is not runners[1].npu_update_stream
    else:
        assert created == []
        assert all(runner.npu_update_stream is None for runner in runners)


def test_npu_replay_updates_window_boundaries_for_a_reused_bucket():
    captured = []
    operations = []
    update_submitted = threading.Event()
    host_threads = {}
    runner = _npu_runner()
    runner.buckets = (8, 16)
    runner.plan = lambda total, windows: (8, [8 - total])
    runner.npu_update_stream.wait_stream = lambda stream: operations.append(
        ("wait", stream)
    )

    runner.capture_all()
    assert runner.graphs == {}

    def apply(_device_module, _update_stream, boundaries):
        host_threads["update"] = threading.get_ident()
        operations.append(("update", boundaries))
        update_submitted.set()

    def replay():
        host_threads["replay"] = threading.get_ident()
        operations.append(("replay", 8))
        assert update_submitted.wait(timeout=1)
        update_submitted.clear()

    def capture(bucket_size, *, window_lens=None):
        captured.append((bucket_size, window_lens))
        task = SimpleNamespace(apply=apply)
        return SimpleNamespace(
            hidden_states=torch.zeros(bucket_size, 2),
            graph=SimpleNamespace(replay=replay),
            output=torch.zeros(bucket_size, 2),
            npu_update_tasks=(task,),
        )

    runner.capture = capture

    assert runner.run(torch.ones(4, 2), [4]) is not None
    assert runner.run(torch.ones(3, 2), [3]) is not None
    assert captured == [(8, (4, 4))]
    assert operations.count(("wait", "encoder-compute")) == 2
    assert operations.count(("replay", 8)) == 2
    assert [op for op in operations if op[0] == "update"] == [
        ("update", [4, 8]),
        ("update", [3, 8]),
    ]
    assert host_threads["update"] != host_threads["replay"]


def test_npu_pre_capture_failure_leaves_the_bucket_eager():
    runner = _npu_runner()
    runner.plan = lambda total, windows: (8, [8 - total])
    capture_calls = []

    def capture(bucket_size, *, window_lens=None):
        capture_calls.append((bucket_size, window_lens))
        raise torch.OutOfMemoryError("simulated warmup failure")

    runner.capture = capture

    assert runner.run(torch.ones(4, 2), [4]) is None
    assert runner.run(torch.ones(4, 2), [4]) is None
    assert capture_calls == [(8, (4, 4))]
    assert runner.failed == {8}


def test_npu_attention_capture_restores_partial_setup():
    runner = _npu_runner()
    original = torch.nn.Identity()
    attention = torch.nn.Module()
    attention.qkv_backend_name = "ascend_attn"
    attention.qkv_backend = original
    runner.tower = SimpleNamespace(
        layers=[
            SimpleNamespace(self_attn=attention),
            SimpleNamespace(self_attn=SimpleNamespace(qkv_backend_name="unsupported")),
        ]
    )

    with pytest.raises(RuntimeError, match="ascend_attn"):
        with runner.capture_npu_attention_tasks(NpuGraphCaptureContext(tasks=[])):
            pytest.fail("capture body must not run after partial setup fails")
    assert attention.qkv_backend is original


def test_npu_attention_capture_registers_and_applies_an_explicit_fia_task(monkeypatch):
    calls = []

    def operation(**kwargs):
        calls.append(("operation", kwargs))

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            _npu_fused_infer_attention_score_get_max_workspace=lambda **kwargs: (
                torch.empty(16)
            ),
            npu_fused_infer_attention_score=SimpleNamespace(out=operation),
        ),
    )

    class Event:
        def wait(self, stream):
            calls.append(("wait", stream))

        def reset(self, stream):
            calls.append(("reset", stream))

        def record(self, stream):
            calls.append(("record", stream))

    class DeviceModule:
        def current_stream(self):
            return "capture-stream"

        def ExternalEvent(self):
            return Event()

        def graph_task_group_begin(self, stream):
            calls.append(("begin", stream))

        def graph_task_group_end(self, stream):
            calls.append(("capture_end", stream))
            return "fia-handle"

        def graph_task_update_begin(self, stream, handle):
            calls.append(("update_begin", stream, handle))

        def graph_task_update_end(self, stream):
            calls.append(("update_end", stream))

    context = NpuGraphCaptureContext(tasks=[])
    device_module = DeviceModule()
    attention = NpuGraphCaptureAttention(context, device_module)
    metadata = SimpleNamespace(cu_seqlens=torch.tensor([0, 3, 8], dtype=torch.int32))

    output = attention(
        torch.zeros(8, 2, 4),
        torch.zeros(8, 2, 4),
        torch.zeros(8, 2, 4),
        forward_metadata=metadata,
    )

    assert output.shape == (8, 2, 4)
    assert len(context.tasks) == 1
    task = context.tasks[0]
    assert task.handle == "fia-handle"
    capture_kwargs = calls[-2][1]
    assert capture_kwargs["actual_seq_lengths"] == [3, 8]
    assert capture_kwargs["actual_seq_lengths_kv"] == [3, 8]

    calls.clear()
    task.apply(device_module, "update-stream", [2, 8])

    assert [call[0] for call in calls] == [
        "update_begin",
        "operation",
        "update_end",
        "record",
    ]
    assert calls[0] == ("update_begin", "update-stream", "fia-handle")
    update_kwargs = calls[1][1]
    assert update_kwargs["actual_seq_lengths"] == [2, 8]
    assert update_kwargs["actual_seq_lengths_kv"] == [2, 8]
    assert calls[2:] == [("update_end", "update-stream"), ("record", "update-stream")]


def _capture_runner(backend):
    backend.supports_graph_task_update = True

    class DeviceModule:
        pool = object()
        graph_pool_handle_calls = 0

        def graph_pool_handle(self):
            self.graph_pool_handle_calls += 1
            return self.pool

        def Stream(self, device):
            return SimpleNamespace(wait_stream=lambda other: None)

        def current_stream(self, device):
            return SimpleNamespace(wait_stream=lambda other: None)

        def stream(self, side):
            return nullcontext()

        def synchronize(self, device):
            pass

    def identity(hidden_states):
        return hidden_states, None

    class IdentityNorm:
        normalized_shape = (2,)

        def __call__(self, hidden_states):
            return hidden_states

    runner = object.__new__(Qwen3ASREncoderLayerStackGraphRunner)
    runner.tower = SimpleNamespace(
        layers=[],
        ln_post=IdentityNorm(),
        proj1=identity,
        act=lambda hidden_states: hidden_states,
        proj2=identity,
    )
    runner.device = torch.device("cpu")
    runner.dtype = torch.float32
    runner.device_module = DeviceModule()
    runner.max_seqlen = 8
    runner.buckets = (8,)
    runner.graph_backend = backend
    runner.graph_pool = None
    runner.graphs = {}
    runner.failed = set()
    runner.npu_update_stream = object()
    runner.plan = lambda total, windows: (8, [8 - total])
    return runner


def test_npu_capture_context_failure_is_terminal():
    class CaptureContext:
        def __enter__(self):
            raise RuntimeError("simulated capture context failure")

        def __exit__(self, *args):
            return False

    runner = _capture_runner(SimpleNamespace(capture=lambda **kwargs: CaptureContext()))

    with pytest.raises(EncoderGraphUnrecoverableError, match="capture failed"):
        runner.run(torch.ones(4, 2), [4])


def test_npu_captures_share_one_graph_pool():
    pools = []

    class Backend:
        def capture(self, *, pool=None, stream=None, thread_local_errors=False):
            pools.append(pool)
            return nullcontext(SimpleNamespace())

    runner = _capture_runner(Backend())

    runner.capture(8, window_lens=(4, 4))
    runner.capture(8, window_lens=(2, 2, 4))

    assert runner.device_module.graph_pool_handle_calls == 1
    assert pools == [runner.device_module.pool, runner.device_module.pool]
