# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_asr import sglang_model
from sglang_omni.models.qwen3_asr.encoder_cuda_graph import (
    Qwen3ASREncoderLayerStackGraphRunner,
    build_buckets,
    window_lens_from_token_counts,
)
from sglang_omni.platforms import current_platform


def test_build_buckets_rejects_bad_limits():
    with pytest.raises(ValueError):
        build_buckets(0, 780)


def _plan_only_runner(max_batch=8, max_tokens_per_clip=780):
    r = object.__new__(Qwen3ASREncoderLayerStackGraphRunner)
    r._max_seqlen = 104
    r._max_windows_for = lambda b: max_batch + b // 104 + 1
    raw = build_buckets(max_batch, max_tokens_per_clip)
    r._buckets = raw[:-1] + (raw[-1] + r._max_windows_for(raw[-1]),)
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
    item = SimpleNamespace(
        feature=torch.zeros(1, 128, 500),
        feature_attention_mask=None,
        model_specific_data={"num_audio_tokens": 65},
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
    runner._capture_attention_metadata = attention_metadata

    output = runner._layer_stack(hidden_states, cu_seqlens)

    assert torch.equal(output, hidden_states)
    assert seen["cu_seqlens"] is cu_seqlens
    assert seen["max_seqlen"] == 104
    assert seen["forward_metadata"] is attention_metadata


@pytest.fixture
def asr_server_args():
    from sglang.srt.runtime_context import get_context

    mm_attention_backend = "aiter_attn" if current_platform.is_rocm() else "triton_attn"
    with get_context().override_server_args(
        model_path="Qwen/Qwen3-ASR-1.7B", mm_attention_backend=mm_attention_backend
    ):
        yield


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_graph_matches_eager_tower(asr_server_args):
    from sglang.srt.configs.qwen3_omni import Qwen3OmniMoeAudioEncoderConfig
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.distributed.parallel_state import model_parallel_is_initialized

    from sglang_omni.models.qwen3_asr.audio_lengths import qwen3_asr_num_audio_tokens
    from sglang_omni.models.qwen3_asr.encoder_cuda_graph import eager_preamble

    if not model_parallel_is_initialized():
        init_distributed_environment(
            backend="nccl",
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
    tower = Qwen3OmniMoeAudioEncoder(cfg).cuda().eval().to(torch.bfloat16)
    with torch.no_grad():
        for name, prm in tower.named_parameters():
            if prm.dim() >= 2:
                prm.normal_(0.0, 0.02)
            elif "weight" in name:
                prm.fill_(1.0)
            else:
                prm.zero_()

    runner = Qwen3ASREncoderLayerStackGraphRunner(
        tower, buckets=build_buckets(4, 104), max_batch_size=4
    )
    runner.capture_all()
    assert runner._graphs and not runner._failed

    # [500] and [450] share a bucket, so the second replay also checks that
    # stale rows from the first never leak into the output.
    for frame_lens in ([500], [450], [300, 500, 120], [800, 800, 800, 800]):
        feats = (torch.randn(128, sum(frame_lens), device="cuda") * 0.05).to(
            torch.bfloat16
        )
        lens = torch.tensor(frame_lens, device="cuda", dtype=torch.long)
        with torch.no_grad():
            ref = tower(feats, feature_lens=lens).last_hidden_state.squeeze(0)
        wl = window_lens_from_token_counts(
            [qwen3_asr_num_audio_tokens(f) for f in frame_lens],
            tokens_per_window=runner.tokens_per_window,
        )
        out = runner.run(eager_preamble(tower, feats, lens), wl)
        diff = (out.float() - ref.float()).abs().max().item()
        assert diff < 3e-2, f"{frame_lens}: max|diff|={diff}"

    assert (
        runner.run(
            torch.zeros(9999, 64, device="cuda", dtype=torch.bfloat16),
            [104] * 96 + [15],
        )
        is None
    )
