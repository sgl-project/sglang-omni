# SPDX-License-Identifier: Apache-2.0
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem

from sglang_omni.models.qwen3_asr import sglang_model
from sglang_omni.models.qwen3_asr.encoder_cuda_graph import (
    Qwen3ASREncoderLayerStackGraphRunner,
    build_buckets,
    eager_preamble,
    plan_mel_chunks,
    planned_preamble,
    window_lens_from_token_counts,
)
from sglang_omni.models.qwen3_asr.encoder_npu_graph import Qwen3ASREncoderNpuGraphRunner
from sglang_omni.platforms import current_platform


def test_build_buckets_rejects_bad_limits():
    with pytest.raises(ValueError):
        build_buckets(0, 780)


def plan_only_runner(max_batch=8, max_tokens_per_clip=780):
    r = object.__new__(Qwen3ASREncoderLayerStackGraphRunner)
    r.max_seqlen = 104
    r.max_windows_for = lambda b: max_batch + b // 104 + 1
    raw = build_buckets(max_batch, max_tokens_per_clip)
    r.buckets = raw[:-1] + (raw[-1] + r.max_windows_for(raw[-1]),)
    return r


@pytest.mark.parametrize("total,windows", [(65, 1), (260, 4), (6240, 64), (104, 1)])
def test_plan_invariants(total, windows):
    r = plan_only_runner()
    bucket_size, dummies = r.plan(total, windows)
    assert total + sum(dummies) == bucket_size
    assert all(1 <= d <= r.max_seqlen for d in dummies)
    assert windows + len(dummies) == r.max_windows_for(bucket_size)
    assert r.plan(r.buckets[-1] + 1, 1) is None


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

    model.encoder_graph_runner = SimpleNamespace(
        tokens_per_window=104,
        plan=lambda *args: (104, []),
        failed=set(),
        run=lambda h, w: torch.ones(65, 8),
    )
    assert torch.equal(get(model, [item]), torch.ones(1, 65, 8))

    model.encoder_graph_runner = SimpleNamespace(
        tokens_per_window=104,
        plan=lambda *args: (104, []),
        failed=set(),
        run=lambda h, w: None,
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
    runner.tower = tower
    runner.max_seqlen = 104
    hidden_states = torch.zeros(8, 4)
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
    attention_metadata = object()
    runner.capture_attention_metadata = attention_metadata

    output = runner.layer_stack(hidden_states, cu_seqlens)

    assert torch.equal(output, hidden_states)
    assert seen["cu_seqlens"] is cu_seqlens
    assert seen["max_seqlen"] == 104
    assert seen["forward_metadata"] is attention_metadata


@pytest.fixture
def asr_server_args():
    from sglang.srt.runtime_context import get_context, get_parallel

    if current_platform.is_rocm():
        mm_attention_backend = "aiter_attn"
    elif current_platform.is_npu():
        mm_attention_backend = "ascend_attn"
    else:
        mm_attention_backend = "triton_attn"
    with (
        get_context().override_server_args(
            model_path="Qwen/Qwen3-ASR-1.7B", mm_attention_backend=mm_attention_backend
        ),
        get_parallel().override(tp_rank=0, attn_tp_rank=0),
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
        initialize_model_parallel()
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

    runner_cls = (
        Qwen3ASREncoderNpuGraphRunner
        if tower_device.type == "npu"
        else Qwen3ASREncoderLayerStackGraphRunner
    )
    runner = runner_cls(
        tower,
        buckets=build_buckets(4, 104),
        max_batch_size=4,
        graph_backend=current_platform.get_device_graph_backend(tower_device),
    )
    runner.capture_all()
    if tower_device.type != "npu":
        assert runner.graphs and not runner.failed
        pools = [entry.graph.pool() for entry in runner.graphs.values()]
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
        eager_hidden = eager_preamble(tower, feats, lens)
        features = list(feats.split(frame_lens, dim=1))
        host_plan = plan_mel_chunks(
            [feature.unsqueeze(0).cpu() for feature in features],
            frame_lens,
            chunk_width=tower.n_window * 2,
            dtype=torch.bfloat16,
        )
        planned_hidden = planned_preamble(tower, host_plan, tower_device)
        torch.testing.assert_close(planned_hidden, eager_hidden, rtol=0, atol=0)
        out = runner.run(planned_hidden, wl)
        assert out is not None
        diff = (out.float() - ref.float()).abs().max().item()
        assert diff < 3e-2, f"{frame_lens}: max|diff|={diff}"

    # [500] and [450] share a bucket, so the second replay also checks that
    # stale rows from the first never leak into the output. The descending pass
    # is the order a long clip followed by a short one produces, which no shared
    # graph memory may assume away.
    sequence = ([500], [450], [300, 500, 120], [800, 800, 800, 800])
    check(sequence[0])
    if tower_device.type == "npu":
        entry = next(iter(runner.graphs.values()))
        assert entry.cu_seqlens[1].item() != qwen3_asr_num_audio_tokens(sequence[0][0])
    graph_count = len(runner.graphs)
    check(sequence[1])
    assert len(runner.graphs) == graph_count
    for frame_lens in sequence[2:]:
        check(frame_lens)
    for frame_lens in reversed(sequence):
        check(frame_lens)
    assert runner.graphs and not runner.failed

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
    model.encoder_graph_runner = "untouched"

    sglang_model.Qwen3ASRForConditionalGeneration.init_encoder_graphs(
        model, max_batch_size=4, max_tokens_per_clip=780
    )

    assert model.encoder_graph_runner == "untouched"


@pytest.mark.parametrize("update_fails", [False, True])
def test_npu_replay_updates_reused_bucket_without_waiting_on_replay(update_fails):
    runner = object.__new__(Qwen3ASREncoderNpuGraphRunner)
    runner.device = torch.device("cpu")
    runner.max_seqlen = 8
    runner.max_windows_for = lambda bucket: 2
    runner.graphs = {}
    runner.failed = set()
    runner.plan = lambda total, windows: (8, [8 - total])
    caller = threading.get_ident()
    waits = []
    updates = []
    captures = []
    runner.update_tasks = {}
    replay_started = threading.Event()
    runner.update_stream = SimpleNamespace(
        wait_stream=lambda stream: waits.append(threading.get_ident())
    )
    runner.device_module = SimpleNamespace(
        current_stream=lambda: "compute", stream=lambda stream: nullcontext()
    )

    def apply(device_module, stream, boundaries):
        # Force update to start after replay: the dependency must already exist.
        assert replay_started.wait(timeout=2)
        updates.append((threading.get_ident(), boundaries))
        if update_fails:
            raise RuntimeError("update failed")

    def replay():
        assert waits == [caller] * (len(updates) + 1)
        replay_started.set()

    def capture(bucket_size):
        cu, metadata = runner.capture_boundaries(bucket_size)
        captures.append(cu.tolist())
        runner.update_tasks[bucket_size] = (SimpleNamespace(apply=apply),)
        return SimpleNamespace(
            hidden_states=torch.zeros(8, 2),
            output=torch.zeros(8, 2),
            graph=SimpleNamespace(replay=replay),
        )

    runner.capture = capture

    def initialize_worker():
        assert replay_started.wait(timeout=2)

    with ThreadPoolExecutor(max_workers=1, initializer=initialize_worker) as executor:
        runner.update_executor = executor
        runner.capture_all()
        assert not runner.graphs
        for length in (3, 2):
            replay_started.clear()
            if update_fails:
                with pytest.raises(RuntimeError, match="update failed"):
                    runner.run(torch.ones(length, 2), [length])
            else:
                assert runner.run(torch.ones(length, 2), [length]) is not None
    assert captures == [[0, 4, 8]]
    assert [bounds for _, bounds in updates] == [[3, 8], [2, 8]]
    assert updates[0][0] == updates[1][0] != caller


def test_npu_attention_capture_restores_partial_setup():
    runner = object.__new__(Qwen3ASREncoderNpuGraphRunner)
    runner.device_module = SimpleNamespace()
    runner.graph_pool = object()
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
        runner.record_graph(8, lambda: pytest.fail("partial setup must abort capture"))
    assert attention.qkv_backend is original


def test_npu_attention_capture_registers_and_applies_an_explicit_fia_task(monkeypatch):
    calls = []

    def operation(**kwargs):
        calls.append(("operation", kwargs))

    monkeypatch.setattr(
        torch.ops.npu,
        "_npu_fused_infer_attention_score_get_max_workspace",
        lambda **kwargs: torch.empty(16),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.npu,
        "npu_fused_infer_attention_score",
        SimpleNamespace(out=operation),
        raising=False,
    )

    event = SimpleNamespace(
        wait=lambda stream: calls.append(("wait", stream)),
        reset=lambda stream: calls.append(("reset", stream)),
        record=lambda stream: calls.append(("record", stream)),
    )
    device_module = SimpleNamespace(
        current_stream=lambda: "capture-stream",
        ExternalEvent=lambda: event,
        graph_task_group_begin=lambda stream: calls.append(("begin", stream)),
        graph_task_group_end=lambda stream: "fia-handle",
        graph_task_update_begin=lambda stream, handle: calls.append(
            ("update_begin", stream, handle)
        ),
        graph_task_update_end=lambda stream: calls.append(("update_end", stream)),
    )

    runner = object.__new__(Qwen3ASREncoderNpuGraphRunner)
    runner.device = torch.device("cpu")
    runner.device_module = device_module
    runner.graph_pool = object()
    runner.update_tasks = {}
    runner.graph_backend = SimpleNamespace(
        capture=lambda **kwargs: nullcontext("graph")
    )
    original = torch.nn.Identity()
    attention = SimpleNamespace(qkv_backend_name="ascend_attn", qkv_backend=original)
    runner.tower = SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])
    runner.max_seqlen = 8
    runner.max_windows_for = lambda bucket: 2
    cu, metadata = runner.capture_boundaries(8)
    assert cu.device.type == "cpu"

    graph, output = runner.record_graph(
        8,
        lambda: attention.qkv_backend(
            torch.zeros(8, 2, 4),
            torch.zeros(8, 2, 4),
            torch.zeros(8, 2, 4),
            forward_metadata=metadata,
        ),
    )

    assert graph == "graph" and attention.qkv_backend is original
    assert output.shape == (8, 2, 4)
    (task,) = runner.update_tasks[8]
    assert task.handle == "fia-handle"
    assert [call[0] for call in calls] == ["wait", "reset", "begin", "operation"]
    capture_kwargs = calls[-1][1]
    assert capture_kwargs["actual_seq_lengths"] == [4, 8]
    assert capture_kwargs["actual_seq_lengths_kv"] == [4, 8]

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


@pytest.mark.parametrize("frame_counts", [[1], [99, 100, 101], [200, 307]])
@pytest.mark.parametrize("convolution_batch_size", [1, 16])
def test_host_planned_preamble_matches_eager(frame_counts, convolution_batch_size):
    class Projection(torch.nn.Linear):
        def forward(self, hidden_states):
            return super().forward(hidden_states), None

    torch.manual_seed(7)
    tower = SimpleNamespace(
        n_window=50,
        conv_chunksize=convolution_batch_size,
        conv2d1=torch.nn.Conv2d(1, 2, 3, stride=2, padding=1),
        conv2d2=torch.nn.Conv2d(2, 2, 3, stride=2, padding=1),
        conv2d3=torch.nn.Conv2d(2, 2, 3, stride=2, padding=1),
        conv_out=Projection(32, 8),
        positional_embedding=SimpleNamespace(positional_embedding=torch.randn(128, 8)),
    )
    features = [torch.randn(1, 128, frames + 11) for frames in frame_counts]
    packed_features = torch.cat(
        [feature[0, :, :frames] for feature, frames in zip(features, frame_counts)],
        dim=1,
    )
    plan = plan_mel_chunks(features, frame_counts, chunk_width=100, dtype=torch.float32)
    with torch.no_grad():
        expected = eager_preamble(tower, packed_features, torch.tensor(frame_counts))
        actual = planned_preamble(tower, plan, torch.device("cpu"))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("routing", ["uncovered", "failed", "success", "rejected"])
def test_host_plan_checks_eligibility_before_frontend(monkeypatch, routing):
    """Known missing graphs must fall back before running the convolution frontend."""
    tower = torch.nn.Linear(4, 4)
    tower.dtype = torch.float32
    tower.n_window = 50
    model = object.__new__(sglang_model.Qwen3ASRForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.audio_tower = tower
    item = MultimodalDataItem(
        modality=Modality.AUDIO,
        feature=torch.zeros(1, 128, 500),
        model_specific_data={
            "feature_attention_mask": torch.ones(1, 500, dtype=torch.long),
            "num_audio_tokens": 65,
        },
    )
    preamble = Mock(return_value=torch.zeros(65, 4))
    monkeypatch.setattr(sglang_model, "planned_preamble", preamble)
    runner = SimpleNamespace(
        tokens_per_window=104,
        plan=Mock(return_value=None if routing == "uncovered" else (104, [])),
        failed={104} if routing == "failed" else set(),
        run=Mock(return_value=None if routing == "rejected" else torch.ones(65, 8)),
    )
    model.encoder_graph_runner = runner
    result = model.planned_audio_feature([item], torch.device("cpu"))
    if routing == "success":
        torch.testing.assert_close(result, torch.ones(1, 65, 8))
    else:
        assert result is None
    assert preamble.call_count == int(routing in ("success", "rejected"))
    assert runner.run.call_count == preamble.call_count
    if routing in ("uncovered", "failed"):
        tower.forward = Mock(
            return_value=SimpleNamespace(last_hidden_state=torch.full((1, 65, 8), 7.0))
        )
        eager_frontend = Mock(
            side_effect=AssertionError(
                "fallback must run the frontend only inside the tower"
            )
        )
        monkeypatch.setattr(sglang_model, "eager_preamble", eager_frontend)
        torch.testing.assert_close(
            model.get_audio_feature([item]), torch.full((1, 65, 8), 7.0)
        )
        tower.forward.assert_called_once()
        assert preamble.call_count == 0
