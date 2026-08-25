# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from torch.nn import functional as F

from sglang_omni.models.minimax_music3.acoustic import (
    MiniMaxMusic3AcousticScheduler,
    _resolve_acoustic_dtype,
)
from sglang_omni.models.minimax_music3.chunking import chunk_windows
from sglang_omni.models.minimax_music3.config import (
    MiniMaxMusic3DualGPUPipelineConfig,
    MiniMaxMusic3PipelineConfig,
    MiniMaxMusic3SingleGPUPipelineConfig,
)
from sglang_omni.models.minimax_music3.dav import remove_weight_norm
from sglang_omni.models.minimax_music3.dit import (
    Attention,
    MiniMaxMusic3DIT,
    RotaryEmbedding,
    _apply_rope,
    _resolve_attention_backend,
)
from sglang_omni.models.minimax_music3.model_runner import _HiddenFrameBuffer
from sglang_omni.models.minimax_music3.rvq_cuda_graph import RVQDepthCudaGraphRunner
from sglang_omni.models.minimax_music3.rvq_decoder import sample_topk_seeded
from sglang_omni.pipeline.stage.stream_queue import StreamItem


@pytest.mark.parametrize(
    ("frames", "expected"),
    [
        (0, []),
        (200, [(0, 0, 200, True)]),
        (201, [(0, 0, 200, False), (1, 100, 201, True)]),
        (
            301,
            [
                (0, 0, 200, False),
                (1, 100, 300, False),
                (2, 200, 301, True),
            ],
        ),
    ],
)
def test_chunk_windows_cover_boundaries(
    frames: int, expected: list[tuple[int, int, int, bool]]
) -> None:
    actual = [
        (window.index, window.start, window.end, window.is_last)
        for window in chunk_windows(frames)
    ]
    assert actual == expected


def test_hidden_frame_buffer_uses_absolute_indexes_after_discard() -> None:
    buffer = _HiddenFrameBuffer()
    for frame in range(201):
        buffer.append(torch.full((2,), frame, dtype=torch.float32))

    first = buffer.concatenate(0, 200)
    buffer.discard_before(100)

    assert first.shape == (2, 200)
    assert buffer.base_frame == 100
    assert buffer.end_frame == 201
    assert buffer.concatenate(100, 201)[0].tolist() == list(range(100, 201))
    with pytest.raises(RuntimeError, match=r"buffer range \[0, 10\) is unavailable"):
        buffer.concatenate(0, 10)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sample_topk_seeded_is_invariant_to_batch_composition() -> None:
    # A request's codes must not depend on which other requests share its
    # decode batch, so a row sampled alone must match the same row sampled
    # inside a larger batch with the same seed and position.
    torch.manual_seed(0)
    logits = torch.randn(3, 512, device="cuda")
    seeds = torch.tensor([11, 22, 33], device="cuda")
    positions = torch.tensor([7, 7, 7], device="cuda")

    batched = sample_topk_seeded(logits, seeds, positions)
    alone = torch.cat(
        [
            sample_topk_seeded(
                logits[index : index + 1],
                seeds[index : index + 1],
                positions[index : index + 1],
            )
            for index in range(3)
        ]
    )

    assert batched.shape == (3,)
    assert torch.equal(batched, alone)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sample_topk_seeded_advances_with_the_draw_position() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 512, device="cuda")
    seed = torch.tensor([11], device="cuda")

    draws = {
        int(sample_topk_seeded(logits, seed, torch.tensor([position], device="cuda")))
        for position in range(16)
    }

    assert len(draws) > 1


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("float32", torch.float32),
        ("torch.bfloat16", torch.bfloat16),
        (torch.float32, torch.float32),
    ],
)
def test_resolve_acoustic_dtype(
    value: str | torch.dtype, expected: torch.dtype
) -> None:
    assert _resolve_acoustic_dtype(value) is expected


@pytest.mark.parametrize("value", ["float16", "int8", None])
def test_resolve_acoustic_dtype_rejects_unsupported_values(value: object) -> None:
    with pytest.raises(ValueError, match="float32.*bfloat16|bfloat16.*float32"):
        _resolve_acoustic_dtype(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("auto", None),
        ("torch_sdpa", AttentionBackendEnum.TORCH_SDPA),
        ("FA", AttentionBackendEnum.FA),
        ("sage_attn", AttentionBackendEnum.SAGE_ATTN),
    ],
)
def test_resolve_attention_backend(
    value: str, expected: AttentionBackendEnum | None
) -> None:
    assert _resolve_attention_backend(value) is expected


def test_resolve_attention_backend_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="attention_backend"):
        _resolve_attention_backend("flashinfer")


def test_minimax_music3_explicit_placements_ignore_the_machine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The named topologies mean the same thing on any machine.

    Only the no-flag default adapts; a config file or a variant that says
    two-GPU has to stay two-GPU even on a one-GPU box, or the file would not be
    describing anything.
    """
    for visible in ("6", "6,7"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
        dual = MiniMaxMusic3DualGPUPipelineConfig(model_path="/models/minimax")
        single = MiniMaxMusic3SingleGPUPipelineConfig(model_path="/models/minimax")

        dual_stages = {stage.name: stage for stage in dual.stages}
        single_stages = {stage.name: stage for stage in single.stages}
        assert dual_stages["minimax_music3_ar"].gpu == 0
        assert dual_stages["minimax_music3_ar"].factory.max_concurrency == 16
        assert dual_stages["dit_dav"].gpu == 1
        assert dual_stages["dit_dav"].factory.dtype == "float32"
        assert dual_stages["dit_dav"].factory.breakable_cuda_graph is False
        assert dual.placement.require_memory_fraction_for_colocation
        assert single_stages["minimax_music3_ar"].gpu == 0
        assert single_stages["minimax_music3_ar"].factory.max_concurrency == 16
        assert single_stages["dit_dav"].gpu == 0
        assert single_stages["dit_dav"].factory.dtype == "float32"
        assert single_stages["dit_dav"].factory.breakable_cuda_graph is False
        assert not single.placement.require_memory_fraction_for_colocation


@pytest.mark.parametrize(
    ("visible", "acoustic_gpu", "acoustic_dtype", "colocation_check"),
    [
        # Colocated is float32 like the two-GPU layout: the bfloat16 DIT solve
        # measured 6.88 LTAS L1 and 9.3 dB of loudness from the d6169737
        # baseline, which is audible, so both layouts serve one arithmetic.
        ("6", 0, "float32", False),
        ("6,7", 1, "float32", True),
        ("0,1,2,3", 1, "float32", True),
    ],
)
def test_minimax_music3_default_follows_the_visible_gpus(
    monkeypatch: pytest.MonkeyPatch,
    visible: str,
    acoustic_gpu: int,
    acoustic_dtype: str,
    colocation_check: bool,
) -> None:
    """Starting the server should need a model path and nothing else.

    The two-GPU layout cannot run on a one-GPU box, so defaulting to it forced
    every single-GPU user to pass a config file to say so. The default now
    matches the machine, which is the topology this model is usually served in.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    config = MiniMaxMusic3PipelineConfig(model_path="/models/minimax")

    stages = {stage.name: stage for stage in config.stages}
    assert stages["minimax_music3_ar"].gpu == 0
    assert stages["dit_dav"].gpu == acoustic_gpu
    assert stages["dit_dav"].factory.dtype == acoustic_dtype
    assert stages["dit_dav"].factory.compile_acoustic is True
    assert config.placement.require_memory_fraction_for_colocation is colocation_check
    assert MiniMaxMusic3PipelineConfig.stage_config_cls(
        "minimax_music3_ar"
    ).engine_stage


def test_native_attention_preserves_checkpoint_state_dict_keys() -> None:
    with torch.device("meta"):
        model = MiniMaxMusic3DIT(
            compute_dtype=torch.float32,
            attention_backend="torch_sdpa",
        )

    keys = set(model.state_dict())
    assert "diffusion_transformer.transformer.layers.0.self_attn.to_qkv.weight" in keys
    assert "diffusion_transformer.transformer.layers.35.self_attn.to_out.weight" in keys
    assert not any(".backend." in key for key in keys)


def test_dav_weight_norm_folding_preserves_output() -> None:
    torch.manual_seed(31)
    convolution = torch.nn.utils.weight_norm(
        torch.nn.Conv1d(4, 6, kernel_size=3, padding=1)
    ).eval()
    sample = torch.randn((2, 4, 11))
    expected = convolution(sample)

    assert remove_weight_norm(convolution) == 1
    actual = convolution(sample)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert remove_weight_norm(convolution) == 0


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_native_sdpa_matches_reference_without_diffusion_server_args() -> None:
    torch.manual_seed(17)
    module = (
        Attention(
            128,
            dim_heads=64,
            compute_dtype=torch.float32,
            attention_backend="torch_sdpa",
        )
        .cuda()
        .eval()
    )
    rotary = RotaryEmbedding(32).cuda()
    freqs, _ = rotary.forward_from_seq_len(19)
    x = torch.randn((2, 19, 128), device="cuda", dtype=torch.float32)

    q, k, v = module.to_qkv(x).chunk(3, dim=-1)
    rope_cos, rope_sin = freqs.cos(), freqs.sin()
    q = _apply_rope(q.view(2, 19, 2, 64).transpose(1, 2), rope_cos, rope_sin)
    k = _apply_rope(k.view(2, 19, 2, 64).transpose(1, 2), rope_cos, rope_sin)
    v = v.view(2, 19, 2, 64).transpose(1, 2)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    expected = module.to_out(expected.transpose(1, 2).contiguous().view(2, 19, 128))

    with set_forward_context(0, None):
        actual = module(x, rope_cos, rope_sin)

    assert module.backend.backend is AttentionBackendEnum.TORCH_SDPA
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rvq_cuda_graph_replays_per_bucket_and_clones_outputs() -> None:
    def depth_forward(
        hidden: torch.Tensor,
        c0: torch.Tensor,
        seeds: torch.Tensor,
        positions: torch.Tensor,
        forced: torch.Tensor,
        replay: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sampled = torch.stack((c0, seeds + positions), dim=1)
        return (
            torch.where(replay, forced[:, :2], sampled),
            hidden * 1.25,
            forced[:, :7],
        )

    runner = RVQDepthCudaGraphRunner(
        forward=depth_forward,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        hidden_size=16,
        num_codebooks=8,
        buckets=[1, 2, 4],
    )
    assert runner.max_batch_size == 4

    # A batch smaller than a bucket pads into it and must read back only its
    # own rows, matching what the eager path produces for those rows.
    hidden = torch.randn((3, 16), device="cuda", dtype=torch.bfloat16)
    c0 = torch.tensor((11, 12, 13), device="cuda")
    seeds = torch.tensor((5, 6, 7), device="cuda")
    positions = torch.tensor((1, 2, 3), device="cuda")

    forced = torch.zeros((3, 8), dtype=torch.long, device="cuda")
    no_replay = torch.zeros((), dtype=torch.bool, device="cuda")
    expected_codes, expected_hidden, _ = depth_forward(
        hidden, c0, seeds, positions, forced, no_replay
    )
    actual_codes, actual_hidden, _ = runner(
        hidden, c0, seeds, positions, forced, no_replay
    )
    torch.cuda.synchronize()
    assert actual_codes.shape[0] == 3
    assert torch.equal(actual_codes, expected_codes)
    assert torch.equal(actual_hidden, expected_hidden)

    # Outputs are cloned off the static buffers, so a later replay must not
    # rewrite what an earlier one returned.
    preserved_codes = actual_codes.clone()
    runner(hidden + 1, c0 + 1, seeds, positions, forced, no_replay)
    torch.cuda.synchronize()
    assert torch.equal(actual_codes, preserved_codes)


@pytest.mark.accelerator
def test_rvq_cuda_graph_declines_a_batch_larger_than_every_bucket() -> None:
    def depth_forward(hidden, c0, seeds, positions, forced, replay):
        del c0, seeds, positions, replay
        zeros = torch.zeros((hidden.shape[0], 8), device=hidden.device)
        return zeros, hidden, forced[:, :7]

    runner = RVQDepthCudaGraphRunner(
        forward=depth_forward,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        hidden_size=16,
        num_codebooks=8,
        buckets=[1, 2],
    )

    oversized = torch.zeros((3, 16), device="cuda", dtype=torch.bfloat16)
    zeros = torch.zeros((3,), dtype=torch.long, device="cuda")
    forced = torch.zeros((3, 8), dtype=torch.long, device="cuda")
    replay = torch.zeros((), dtype=torch.bool, device="cuda")
    assert runner(oversized, zeros, zeros, zeros, forced, replay) is None


class _TinyCacheBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(16, 16)

    def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x + self.proj(x) * scale


class _TinyCacheTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_TinyCacheBlock() for _ in range(4)])

    def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, scale)
        return x


class _TinyCacheDiffusion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = _TinyCacheTransformer()


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cache_dit_block_adapter_runs_hidden_only_pattern() -> None:
    model = MiniMaxMusic3DIT.__new__(MiniMaxMusic3DIT)
    torch.nn.Module.__init__(model)
    model.diffusion_transformer = _TinyCacheDiffusion().cuda().eval()
    model.enable_cache_dit(
        num_steps=4,
        fn_compute_blocks=1,
        bn_compute_blocks=1,
        max_warmup_steps=1,
        residual_diff_threshold=0.08,
        max_continuous_cached_steps=1,
    )

    x = torch.randn((2, 11, 16), device="cuda")
    for step in range(4):
        scale = torch.tensor(0.1 + step * 0.01, device="cuda")
        x = model.diffusion_transformer.transformer(x, scale)

    assert x.shape == (2, 11, 16)
    assert torch.isfinite(x).all()


class _ZeroDiffusionTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def _transformer(
        self, x: torch.Tensor, _t: torch.Tensor, _condition: torch.Tensor
    ) -> torch.Tensor:
        self.calls += 1
        return torch.zeros_like(x)


def _tiny_dit() -> MiniMaxMusic3DIT:
    dit = MiniMaxMusic3DIT.__new__(MiniMaxMusic3DIT)
    torch.nn.Module.__init__(dit)
    dit.diffusion_transformer = _ZeroDiffusionTransformer()
    dit._bcg_runner = None
    return dit


def test_dit_step_count_is_configurable_without_changing_shape() -> None:
    dit = _tiny_dit()
    align = torch.zeros((1, 2048, 3))

    latent = dit(
        align,
        generator=torch.Generator().manual_seed(1),
        num_steps=4,
        cfg_scale=1.7,
    )

    assert latent.shape == (1, 128, 3)
    assert dit.diffusion_transformer.calls == 4


def test_dit_aligned_mel_length_matches_full_window_contract() -> None:
    dit = _tiny_dit()
    dit.sr_input = 24_000
    dit.sr_output = 44_100
    dit.hop_size_input = 960
    dit.hop_size_output = 512

    assert dit.aligned_mel_length(200) == 689
    assert dit.aligned_mel_length(100) == 344


def test_dit_abort_stops_before_the_next_euler_step() -> None:
    dit = _tiny_dit()
    checks = iter((False, False, True))

    with pytest.raises(InterruptedError, match="aborted"):
        dit(
            torch.zeros((1, 2048, 3)),
            generator=torch.Generator().manual_seed(1),
            num_steps=5,
            should_abort=lambda: next(checks),
        )

    assert dit.diffusion_transformer.calls == 2


def test_acoustic_scheduler_rejects_malformed_hidden_before_decode() -> None:
    scheduler = MiniMaxMusic3AcousticScheduler(decoder=object())  # type: ignore[arg-type]
    item = StreamItem(
        chunk_id=0,
        data=torch.zeros((1, 200, 16)),
        from_stage="minimax_music3_ar",
        metadata={
            "chunk_idx": 0,
            "start_frame": 0,
            "end_frame": 200,
            "seed": 0,
            "is_final": True,
        },
    )

    with pytest.raises(ValueError, match="hidden must be"):
        scheduler.on_stream_chunk("req", item)


class _FakeAcousticDecoder:
    def __init__(self) -> None:
        self.hidden_shape: tuple[int, ...] | None = None

    def decode_with_state(
        self, hidden: torch.Tensor, **_kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.hidden_shape = tuple(hidden.shape)
        return torch.zeros((2, 16)), torch.zeros((1, 128, 1)), torch.zeros((1, 2048, 1))


def test_acoustic_scheduler_accepts_the_relay_tensor_shape() -> None:
    decoder = _FakeAcousticDecoder()
    scheduler = MiniMaxMusic3AcousticScheduler(decoder=decoder)  # type: ignore[arg-type]
    item = StreamItem(
        chunk_id=0,
        data=torch.zeros((1, 1, 32768)),
        from_stage="minimax_music3_ar",
        metadata={
            "chunk_idx": 0,
            "start_frame": 0,
            "end_frame": 1,
            "seed": 0,
            "is_final": True,
        },
    )

    assert scheduler.on_stream_chunk("req", item) == []
    assert decoder.hidden_shape == (1, 32768)


def test_backbone_config_rewrite_does_not_write_through_a_symlink(
    tmp_path: Path,
) -> None:
    """A Hub snapshot symlinks config.json into the shared blob store.

    Writing through the link would rewrite a blob whose filename is its own
    content hash, corrupting it for every other snapshot that shares it.
    """
    from sglang_omni.models.minimax_music3.engine_builder import (
        MiniMaxMusic3EngineBuilder,
    )

    blob = tmp_path / "blob"
    blob.write_text(json.dumps({"model_type": "mixtral", "hidden_size": 4096}))
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    config_path = snapshot / "config.json"
    config_path.symlink_to(blob)

    MiniMaxMusic3EngineBuilder._normalize_backbone_config(config_path)

    assert json.loads(blob.read_text())["model_type"] == "mixtral"
    assert not config_path.is_symlink()
    assert json.loads(config_path.read_text())["model_type"] == "qwen3"
