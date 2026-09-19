# SPDX-License-Identifier: Apache-2.0
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem

from sglang_omni.models.qwen3_asr import sglang_model
from sglang_omni.models.qwen3_asr.encoder_cuda_graph import (
    Qwen3ASREncoderLayerStackGraphRunner,
    _NpuGraphCaptureState,
    _NpuGraphUpdateTask,
    _NpuUpdatableAttentionBackend,
    build_buckets,
    window_lens_from_token_counts,
)
from sglang_omni.models.qwen3_asr.encoder_errors import EncoderGraphUnrecoverableError
from sglang_omni.platforms import current_platform


def test_build_buckets_rejects_bad_limits():
    with pytest.raises(ValueError):
        build_buckets(0, 780)


@pytest.mark.parametrize("is_npu", [True, False])
def test_runner_owns_update_stream_on_model_device(monkeypatch, is_npu):
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
    monkeypatch.setattr(current_platform, "is_npu", lambda: is_npu)
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
                tower, buckets=(128,), max_batch_size=8, graph_backend=object()
            )
        )
    assert devices == [torch.device("meta", 0), torch.device("meta", 1)]
    if is_npu:
        assert [device for device, _ in created] == devices
        assert runners[0]._npu_update_stream is created[0][1]
        assert runners[1]._npu_update_stream is created[1][1]
        assert runners[0]._npu_update_stream is not runners[1]._npu_update_stream
    else:
        assert created == []
        assert all(runner._npu_update_stream is None for runner in runners)


def _plan_only_runner(max_batch=8, max_tokens_per_clip=780):
    r = object.__new__(Qwen3ASREncoderLayerStackGraphRunner)
    r._max_seqlen = 104
    r._max_windows_for = lambda b: max_batch + b // 104 + 1
    raw = build_buckets(max_batch, max_tokens_per_clip)
    r._buckets = raw[:-1] + (raw[-1] + r._max_windows_for(raw[-1]),)
    return r


def _npu_runner():
    runner_device = SimpleNamespace(type="npu", index=0)

    class Backend:
        def replay(self, graph):
            graph.replay()

    class DeviceModule:
        def current_stream(self):
            return "encoder-compute"

        def stream(self, stream):
            return nullcontext()

    r = object.__new__(Qwen3ASREncoderLayerStackGraphRunner)
    r._is_npu = True
    r._max_seqlen = 8
    r._buckets = (8,)
    r._failed = set()
    r._graphs = {}
    r._capture_failed = False
    r._submission_failed = False
    r._device = runner_device
    r._graph_backend = Backend()
    r._device_module = DeviceModule()
    r._npu_update_stream = SimpleNamespace(wait_stream=lambda stream: None)
    return r


@pytest.mark.parametrize("total,windows", [(65, 1), (260, 4), (6240, 64), (104, 1)])
def test_plan_invariants(total, windows):
    r = _plan_only_runner()
    bucket_size, dummies = r._plan(total, windows)
    assert total + sum(dummies) == bucket_size
    assert all(1 <= d <= r._max_seqlen for d in dummies)
    assert windows + len(dummies) == r._max_windows_for(bucket_size)
    assert r._plan(r._buckets[-1] + 1, 1) is None


def test_get_audio_feature_routing(monkeypatch):
    monkeypatch.setattr(sglang_model, "eager_preamble", lambda *a: torch.zeros(65, 4))
    tower = torch.nn.Linear(4, 4)
    tower.dtype = torch.float32
    tower.forward = lambda feats, feature_lens: SimpleNamespace(
        last_hidden_state=torch.full((1, 65, 8), 7.0)
    )
    model = object.__new__(sglang_model.Qwen3ASRForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.audio_tower = tower
    item = MultimodalDataItem(
        modality=Modality.AUDIO,
        feature=torch.zeros(1, 128, 500),
        model_specific_data={"feature_attention_mask": None, "num_audio_tokens": 65},
    )
    get = sglang_model.Qwen3ASRForConditionalGeneration.get_audio_feature

    model._encoder_graph_runner = SimpleNamespace(
        tokens_per_window=104, run=lambda h, w: torch.ones(65, 8)
    )
    assert torch.equal(get(model, [item]), torch.ones(1, 65, 8))

    model._encoder_graph_runner = SimpleNamespace(
        tokens_per_window=104, run=lambda h, w: None
    )
    assert torch.equal(get(model, [item]), torch.full((1, 65, 8), 7.0))


def test_layer_stack_forwards_precomputed_attention_metadata():
    seen = {}

    class Attention:
        def __call__(self, **kwargs):
            seen.update(kwargs)
            return kwargs["x"]

    def identity_linear(hidden_states):
        return hidden_states, None

    layer = SimpleNamespace(
        self_attn_layer_norm=lambda hidden_states: hidden_states,
        self_attn=Attention(),
        final_layer_norm=lambda hidden_states: hidden_states,
        fc1=identity_linear,
        activation_fn=lambda hidden_states: hidden_states,
        fc2=identity_linear,
    )
    tower = SimpleNamespace(
        layers=[layer],
        ln_post=lambda hidden_states: hidden_states,
        proj1=identity_linear,
        act=lambda hidden_states: hidden_states,
        proj2=identity_linear,
    )
    runner = object.__new__(Qwen3ASREncoderLayerStackGraphRunner)
    runner._tower = tower
    runner._max_seqlen = 104
    hidden_states = torch.zeros(8, 4)
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
    attention_metadata = object()

    output = runner._layer_stack(hidden_states, cu_seqlens, attention_metadata)

    assert torch.equal(output, hidden_states)
    assert seen["cu_seqlens"] is cu_seqlens
    assert seen["max_seqlen"] == 104
    assert seen["forward_metadata"] is attention_metadata


def test_npu_capture_all_defers_until_real_window_signature():
    runner = _npu_runner()
    runner._buckets = (128, 256)

    runner.capture_all()

    assert runner._graphs == {}
    assert runner._failed == set()


def test_npu_capture_materializes_sequence_boundaries_on_host():
    runner = object.__new__(Qwen3ASREncoderLayerStackGraphRunner)
    runner._is_npu = True
    runner._device = torch.device("meta")

    cu_seqlens = runner._make_cu_seqlens([4, 3, 1], device="cpu")

    assert cu_seqlens.device.type == "cpu"
    assert cu_seqlens.dtype == torch.int32
    assert cu_seqlens.tolist() == [0, 4, 7, 8]


def test_npu_replay_updates_window_boundaries_for_bucket_graph():
    captured = []
    operations = []
    runner = _npu_runner()
    runner._plan = lambda total, windows: (8, [8 - total])
    runner._npu_update_stream.wait_stream = lambda stream: operations.append(
        ("wait", stream)
    )

    def capture(bucket_size, *, window_lens=None):
        captured.append((bucket_size, window_lens))

        def apply(device_module, update_stream, boundaries):
            assert device_module is runner._device_module
            assert update_stream is runner._npu_update_stream
            operations.append(("update", boundaries))

        task = SimpleNamespace(apply=apply)
        return SimpleNamespace(
            hidden_states=torch.zeros(bucket_size, 2),
            cu_seqlens=torch.tensor([0, 4, 8], dtype=torch.int32),
            attention_metadata=None,
            graph=SimpleNamespace(
                replay=lambda: operations.append(("replay", bucket_size))
            ),
            output=torch.zeros(bucket_size, 2),
            npu_update_tasks=(task,),
        )

    runner._capture = capture
    hidden_states = torch.ones(4, 2)

    assert runner.run(hidden_states, [4]) is not None
    assert runner.run(hidden_states, [4]) is not None
    assert captured == [(8, (4, 4))]

    assert runner.run(torch.ones(3, 2), [3]) is not None
    assert captured == [(8, (4, 4))]
    assert operations == [
        ("wait", "encoder-compute"),
        ("replay", 8),
        ("update", [4, 8]),
        ("wait", "encoder-compute"),
        ("replay", 8),
        ("update", [4, 8]),
        ("wait", "encoder-compute"),
        ("replay", 8),
        ("update", [3, 8]),
    ]
    assert runner._failed == set()


def test_npu_graph_update_task_rebinds_fia_lengths_on_runner_stream():
    calls = []

    class DeviceModule:
        def graph_task_update_begin(self, stream, handle):
            calls.append(("begin", stream, handle))

        def graph_task_update_end(self, stream):
            calls.append(("end", stream))

    event = SimpleNamespace(record=lambda stream: calls.append(("event", stream)))
    operation = lambda **kwargs: calls.append(("operation", kwargs))
    task = _NpuGraphUpdateTask(
        operation=operation,
        kwargs={"query": "static-query"},
        handle="fia-handle",
        event=event,
    )

    task.apply(DeviceModule(), "update-stream", [3, 8])

    assert calls[0] == ("begin", "update-stream", "fia-handle")
    assert calls[1] == (
        "operation",
        {
            "query": "static-query",
            "actual_seq_lengths": [3, 8],
            "actual_seq_lengths_kv": [3, 8],
        },
    )
    assert calls[2:] == [("end", "update-stream"), ("event", "update-stream")]


@pytest.mark.parametrize("failure", ["replay", "begin", "operation", "end", "signal"])
def test_npu_submission_failure_is_terminal(failure):
    calls = []
    runner = _npu_runner()
    runner._plan = lambda total, windows: (8, [8 - total])

    def call(name):
        calls.append(name)
        if name == failure:
            raise torch.OutOfMemoryError("injected submission failure")

    runner._device_module.graph_task_update_begin = lambda *args: call("begin")
    runner._device_module.graph_task_update_end = lambda *args: call("end")
    runner._npu_update_stream.wait_stream = lambda stream: call("wait")
    task = _NpuGraphUpdateTask(
        operation=lambda **kwargs: call("operation"),
        kwargs={},
        handle=object(),
        event=SimpleNamespace(record=lambda stream: call("signal")),
    )
    runner._graphs[8] = SimpleNamespace(
        hidden_states=torch.zeros(8, 2),
        graph=SimpleNamespace(replay=lambda: call("replay")),
        output=torch.ones(8, 2),
        npu_update_tasks=(task,),
    )
    with pytest.raises(EncoderGraphUnrecoverableError, match="restart"):
        runner.run(torch.ones(3, 2), [3])
    expected = ["wait", "replay", "begin", "operation", "end", "signal"]
    assert calls == expected[: expected.index(failure) + 1]
    before_retry = list(calls)
    with pytest.raises(EncoderGraphUnrecoverableError, match="previously failed"):
        runner.run(torch.ones(3, 2), [3])
    assert calls == before_retry


def test_npu_attention_capture_restores_partial_setup():
    runner = _npu_runner()
    original = torch.nn.Identity()
    attention = torch.nn.Module()
    attention.qkv_backend_name = "ascend_attn"
    attention.qkv_backend = original
    runner._tower = SimpleNamespace(
        layers=[
            SimpleNamespace(self_attn=attention),
            SimpleNamespace(self_attn=SimpleNamespace(qkv_backend_name="unsupported")),
        ]
    )
    with pytest.raises(RuntimeError, match="ascend_attn"):
        with runner._capture_npu_attention_tasks(_NpuGraphCaptureState(tasks=[])):
            pytest.fail("capture body must not run after partial setup fails")
    assert attention.qkv_backend is original


def test_npu_attention_backend_captures_one_explicit_fia_task(
    monkeypatch,
):
    calls = []

    def operation(**kwargs):
        calls.append(kwargs)

    fake_torch_npu = SimpleNamespace(
        _npu_fused_infer_attention_score_get_max_workspace=lambda **kwargs: torch.empty(
            16
        ),
        npu_fused_infer_attention_score=SimpleNamespace(out=operation),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)

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
            calls.append(("end", stream))
            return "fia-handle"

    state = _NpuGraphCaptureState(tasks=[])
    backend = _NpuUpdatableAttentionBackend(state, DeviceModule())
    metadata = SimpleNamespace(cu_seqlens=torch.tensor([0, 3, 8], dtype=torch.int32))

    output = backend(
        torch.zeros(8, 2, 4),
        torch.zeros(8, 2, 4),
        torch.zeros(8, 2, 4),
        forward_metadata=metadata,
    )

    assert output.shape == (8, 2, 4)
    assert len(state.tasks) == 1
    assert state.tasks[0].handle == "fia-handle"
    assert calls[-2]["actual_seq_lengths"] == [3, 8]
    assert calls[-2]["actual_seq_lengths_kv"] == [3, 8]


def test_npu_attention_capture_restores_sglang_backends():
    original = torch.nn.Identity()
    attention = torch.nn.Module()
    attention.qkv_backend_name = "ascend_attn"
    attention.qkv_backend = original
    runner = object.__new__(Qwen3ASREncoderLayerStackGraphRunner)
    runner._tower = SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])
    runner._device_module = object()
    state = _NpuGraphCaptureState(tasks=[])

    with runner._capture_npu_attention_tasks(state):
        assert isinstance(attention.qkv_backend, _NpuUpdatableAttentionBackend)

    assert attention.qkv_backend is original


def test_npu_capture_failure_is_terminal():
    runner = _npu_runner()
    runner._plan = lambda total, windows: (8, [8 - total])

    def capture(bucket_size, *, window_lens=None):
        raise RuntimeError("simulated capture failure")

    runner._capture = capture
    hidden_states = torch.ones(4, 2)

    with pytest.raises(RuntimeError, match="restart the encoder process"):
        runner.run(hidden_states, [4])
    assert runner._capture_failed is True
    with pytest.raises(RuntimeError, match="previously failed"):
        runner.run(hidden_states, [4])


def test_npu_captures_share_one_graph_pool():
    pools = []

    class Backend:
        def capture(
            self,
            *,
            pool=None,
            thread_local_errors=False,
        ):
            pools.append(pool)
            return nullcontext(SimpleNamespace())

    class DeviceModule:
        def __init__(self):
            self.pool = object()
            self.graph_pool_handle_calls = 0

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
    runner._tower = SimpleNamespace(
        layers=[],
        ln_post=IdentityNorm(),
        proj1=identity,
        act=lambda hidden_states: hidden_states,
        proj2=identity,
    )
    runner._device = torch.device("cpu")
    runner._dtype = torch.float32
    runner._device_module = DeviceModule()
    runner._is_npu = True
    runner._max_seqlen = 8
    runner._graph_backend = Backend()
    runner._graph_pool = None
    runner._capture_failed = False
    runner._npu_update_stream = object()

    runner._capture(8, window_lens=(4, 4))
    runner._capture(8, window_lens=(2, 2, 4))

    assert runner._device_module.graph_pool_handle_calls == 1
    assert pools == [runner._device_module.pool, runner._device_module.pool]


@pytest.fixture
def asr_server_args():
    from sglang.srt.runtime_context import get_context

    if current_platform.is_rocm():
        mm_attention_backend = "aiter_attn"
    elif current_platform.is_npu():
        mm_attention_backend = "ascend_attn"
    else:
        mm_attention_backend = "triton_attn"
    with get_context().override_server_args(
        model_path="Qwen/Qwen3-ASR-1.7B", mm_attention_backend=mm_attention_backend
    ):
        yield


@pytest.mark.accelerator
@pytest.mark.skipif(
    current_platform.get_device_graph_backend(
        SimpleNamespace(type=current_platform.device_type)
    )
    is None,
    reason="requires an accelerator whose platform names a graph backend",
)
def test_graph_matches_eager_tower(asr_server_args):
    from sglang.srt.configs.qwen3_omni import Qwen3OmniMoeAudioEncoderConfig
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.distributed.parallel_state import (
        get_default_distributed_backend,
        model_parallel_is_initialized,
    )

    from sglang_omni.models.qwen3_asr.audio_lengths import qwen3_asr_num_audio_tokens
    from sglang_omni.models.qwen3_asr.encoder_cuda_graph import eager_preamble

    device = current_platform.device_type
    if not model_parallel_is_initialized():
        init_distributed_environment(
            backend=get_default_distributed_backend(device),
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="tcp://127.0.0.1:29601",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
    from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeAudioEncoder

    cfg = Qwen3OmniMoeAudioEncoderConfig(
        d_model=64,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=128,
        output_dim=32,
        num_mel_bins=128,
        n_window=50,
        n_window_infer=800,
    )
    torch.manual_seed(0)
    tower = Qwen3OmniMoeAudioEncoder(cfg).to(device).eval().to(torch.bfloat16)
    tower_device = next(tower.parameters()).device
    with torch.no_grad():
        for name, prm in tower.named_parameters():
            if prm.dim() >= 2:
                prm.normal_(0.0, 0.02)
            elif "weight" in name:
                prm.fill_(1.0)
            else:
                prm.zero_()

    runner = Qwen3ASREncoderLayerStackGraphRunner(
        tower,
        buckets=build_buckets(4, 104),
        max_batch_size=4,
        graph_backend=current_platform.get_device_graph_backend(tower_device),
    )
    runner.capture_all()
    if not runner._is_npu:
        assert runner._graphs and not runner._failed
        pools = [entry.graph.pool() for entry in runner._graphs.values()]
        assert len(set(pools)) == len(pools)

    def check(frame_lens):
        feats = (torch.randn(128, sum(frame_lens), device=device) * 0.05).to(
            torch.bfloat16
        )
        lens = torch.tensor(frame_lens, device=device, dtype=torch.long)
        with torch.no_grad():
            ref = tower(feats, feature_lens=lens).last_hidden_state.squeeze(0)
        wl = window_lens_from_token_counts(
            [qwen3_asr_num_audio_tokens(f) for f in frame_lens],
            tokens_per_window=runner.tokens_per_window,
        )
        out = runner.run(eager_preamble(tower, feats, lens), wl)
        assert out is not None
        diff = (out.float() - ref.float()).abs().max().item()
        assert diff < 3e-2, f"{frame_lens}: max|diff|={diff}"

    # [500] and [450] share a bucket, so the second replay also checks that
    # stale rows from the first never leak into the output. The descending pass
    # is the order a long clip followed by a short one produces, which no shared
    # graph memory may assume away.
    sequence = ([500], [450], [300, 500, 120], [800, 800, 800, 800])
    check(sequence[0])
    graph_count = len(runner._graphs)
    check(sequence[1])
    assert len(runner._graphs) == graph_count
    for frame_lens in sequence[2:]:
        check(frame_lens)
    for frame_lens in reversed(sequence):
        check(frame_lens)
    assert runner._graphs and not runner._failed

    assert (
        runner.run(
            torch.zeros(9999, 64, device=device, dtype=torch.bfloat16),
            [104] * 96 + [15],
        )
        is None
    )


def test_init_encoder_graphs_declines_a_device_that_cannot_capture():
    """A CPU tower must not build a runner that fails every bucket in turn.

    __new__ skips the checkpoint work; only the tower's device decides this.
    """
    model = sglang_model.Qwen3ASRForConditionalGeneration.__new__(
        sglang_model.Qwen3ASRForConditionalGeneration
    )
    model.audio_tower = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]))
    model._encoder_graph_runner = "untouched"

    sglang_model.Qwen3ASRForConditionalGeneration.init_encoder_graphs(
        model, max_batch_size=4, max_tokens_per_clip=780
    )

    assert model._encoder_graph_runner == "untouched"
