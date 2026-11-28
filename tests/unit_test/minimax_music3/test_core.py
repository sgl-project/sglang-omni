# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from torch import nn
from torch.nn import functional as F

from sglang_omni.config import resolve_stage_factory_args
from sglang_omni.config.path import ConfigPath
from sglang_omni.models.minimax_music3 import model_runner as music3_runner
from sglang_omni.models.minimax_music3 import sglang_model as music3_model
from sglang_omni.models.minimax_music3 import stages as music3_stages
from sglang_omni.models.minimax_music3.acoustic import (
    MiniMaxMusic3AcousticScheduler,
    resolve_acoustic_dtype,
)
from sglang_omni.models.minimax_music3.chunking import chunk_windows
from sglang_omni.models.minimax_music3.config import (
    MiniMaxMusic3DualGPUPipelineConfig,
    MiniMaxMusic3PipelineConfig,
    MiniMaxMusic3SingleGPUPipelineConfig,
)
from sglang_omni.models.minimax_music3.constants import AR_CHUNK_FRAMES
from sglang_omni.models.minimax_music3.dav import remove_weight_norm
from sglang_omni.models.minimax_music3.dit import (
    Attention,
    MiniMaxMusic3DIT,
    RotaryEmbedding,
    apply_rope,
    resolve_attention_backend,
)
from sglang_omni.models.minimax_music3.engine_builder import MiniMaxMusic3EngineBuilder
from sglang_omni.models.minimax_music3.prompt import (
    AUDIO_CODE_OFFSET,
    SPECIAL_TOKEN_IDS,
)
from sglang_omni.models.minimax_music3.rvq_cuda_graph import RVQDepthCudaGraphRunner
from sglang_omni.models.minimax_music3.rvq_decoder import (
    RVQDepthDecoder,
    sample_topk_seeded,
)
from sglang_omni.models.minimax_music3.serial_offload import SerialOffloadCoordinator
from sglang_omni.models.minimax_music3.sglang_request_builder import build_stream_output
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import DeviceGraphBackend
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.types import SchedulerRequest
from sglang_omni.scheduling.weight_cache import checkpoint_identity


def get_test_device_if_eligible() -> torch.device:
    if current_platform.device_type not in ("cuda", "musa", "xpu"):
        pytest.skip("Requires CUDA/MUSA/XPU device")
    return current_platform.get_device(0)


def rvq_graph_target() -> tuple[torch.device, DeviceGraphBackend]:
    """The device and graph backend the RVQ capture tests run on."""
    device = current_platform.get_device(0)
    backend = current_platform.get_device_graph_backend(device)
    if backend is None:
        pytest.skip(f"{device.type} records no model-owned graphs")
    return (device, backend)


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
    buffer = music3_runner.HiddenFrameBuffer()
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
def test_sample_topk_seeded_is_invariant_to_batch_composition() -> None:
    # A request's codes must not depend on which other requests share its
    # decode batch, so a row sampled alone must match the same row sampled
    # inside a larger batch with the same seed and position.
    device = get_test_device_if_eligible()
    torch.manual_seed(0)
    logits = torch.randn(3, 512, device=device)
    seeds = torch.tensor([11, 22, 33], device=device)
    positions = torch.tensor([7, 7, 7], device=device)

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
def test_sample_topk_seeded_advances_with_the_draw_position() -> None:
    device = get_test_device_if_eligible()
    torch.manual_seed(0)
    logits = torch.randn(1, 512, device=device)
    seed = torch.tensor([11], device=device)

    draws = {
        int(sample_topk_seeded(logits, seed, torch.tensor([position], device=device)))
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
    assert resolve_acoustic_dtype(value) is expected


@pytest.mark.parametrize("value", ["float16", None])
def test_resolve_acoustic_dtype_rejects_unsupported_values(value: object) -> None:
    with pytest.raises(ValueError, match="float32.*bfloat16|bfloat16.*float32"):
        resolve_acoustic_dtype(value)  # type: ignore[arg-type]


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
    assert resolve_attention_backend(value) is expected


def test_resolve_attention_backend_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="attention_backend"):
        resolve_attention_backend("flashinfer")


def test_minimax_music3_explicit_placements_ignore_the_machine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The named topologies mean the same thing on any machine.

    Only the no-flag default adapts; a config file or a variant that says
    two-GPU has to stay two-GPU even on a one-GPU box, or the file would not be
    describing anything.
    """
    for gpu_count in (1, 2):
        monkeypatch.setattr(
            "sglang_omni.models.minimax_music3.config.visible_gpu_count",
            lambda count=gpu_count: count,
        )
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
    ("gpu_count", "acoustic_gpu", "acoustic_dtype", "colocation_check"),
    [
        # Colocated is float32 like the two-GPU layout: the bfloat16 DIT solve
        # measured 6.88 LTAS L1 and 9.3 dB of loudness from the d6169737
        # baseline, which is audible, so both layouts serve one arithmetic.
        (1, 0, "float32", False),
        (2, 1, "float32", True),
        (4, 1, "float32", True),
    ],
)
def test_minimax_music3_default_follows_the_visible_gpus(
    monkeypatch: pytest.MonkeyPatch,
    gpu_count: int,
    acoustic_gpu: int,
    acoustic_dtype: str,
    colocation_check: bool,
) -> None:
    """Starting the server should need a model path and nothing else.

    The two-GPU layout cannot run on a one-GPU box, so defaulting to it forced
    every single-GPU user to pass a config file to say so. The default now
    matches the machine, which is the topology this model is usually served in.
    """
    monkeypatch.setattr(
        "sglang_omni.models.minimax_music3.config.visible_gpu_count",
        lambda: gpu_count,
    )
    config = MiniMaxMusic3PipelineConfig(model_path="/models/minimax")

    stages = {stage.name: stage for stage in config.stages}
    assert stages["minimax_music3_ar"].gpu == 0
    assert stages["dit_dav"].gpu == acoustic_gpu
    assert stages["dit_dav"].factory.dtype == acoustic_dtype
    assert stages["dit_dav"].factory.compile_acoustic is True
    assert stages["dit_dav"].factory.fp32_flex_attention is None
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


def test_auto_attention_backend_accepts_the_platform_fallback(monkeypatch) -> None:
    from sglang.multimodal_gen.runtime.layers.attention import selector
    from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPABackend

    monkeypatch.setattr(selector, "_cached_get_attn_backend", lambda *_: SDPABackend)
    with torch.device("meta"):
        model = MiniMaxMusic3DIT(
            compute_dtype=torch.float32,
            attention_backend="auto",
        )

    attention = model.diffusion_transformer.transformer.layers[0].self_attn
    assert attention.backend.backend is AttentionBackendEnum.TORCH_SDPA


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
def test_native_sdpa_matches_reference_without_diffusion_server_args() -> None:
    device = get_test_device_if_eligible()
    torch.manual_seed(17)
    module = (
        Attention(
            128,
            dim_heads=64,
            compute_dtype=torch.float32,
            attention_backend="torch_sdpa",
            fp32_flex_attention=False,
        )
        .to(device)
        .eval()
    )
    rotary = RotaryEmbedding(32).to(device)
    freqs, _ = rotary.forward_from_seq_len(19)
    x = torch.randn((2, 19, 128), device=device, dtype=torch.float32)

    q, k, v = module.to_qkv(x).chunk(3, dim=-1)
    rope_cos, rope_sin = freqs.cos(), freqs.sin()
    q = apply_rope(q.view(2, 19, 2, 64).transpose(1, 2), rope_cos, rope_sin)
    k = apply_rope(k.view(2, 19, 2, 64).transpose(1, 2), rope_cos, rope_sin)
    v = v.view(2, 19, 2, 64).transpose(1, 2)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    expected = module.to_out(expected.transpose(1, 2).contiguous().view(2, 19, 128))

    with set_forward_context(0, None):
        actual = module(x, rope_cos, rope_sin)

    assert module.backend.backend is AttentionBackendEnum.TORCH_SDPA
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


@pytest.mark.accelerator
def test_rvq_cuda_graph_replays_per_bucket_and_clones_outputs() -> None:
    device, backend = rvq_graph_target()
    device_module = torch.get_device_module(device)

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
        backend=backend,
        device=device,
        dtype=torch.bfloat16,
        hidden_size=16,
        num_codebooks=8,
        buckets=[1, 2, 4],
    )
    assert runner.max_batch_size == 4

    # A batch smaller than a bucket pads into it and must read back only its
    # own rows, matching what the eager path produces for those rows.
    hidden = torch.randn((3, 16), device=device, dtype=torch.bfloat16)
    c0 = torch.tensor((11, 12, 13), device=device)
    seeds = torch.tensor((5, 6, 7), device=device)
    positions = torch.tensor((1, 2, 3), device=device)

    forced = torch.zeros((3, 8), dtype=torch.long, device=device)
    no_replay = torch.zeros((), dtype=torch.bool, device=device)
    expected_codes, expected_hidden, _ = depth_forward(
        hidden, c0, seeds, positions, forced, no_replay
    )
    actual_codes, actual_hidden, _ = runner(
        hidden, c0, seeds, positions, forced, no_replay
    )
    device_module.synchronize()
    assert actual_codes.shape[0] == 3
    assert torch.equal(actual_codes, expected_codes)
    assert torch.equal(actual_hidden, expected_hidden)

    # Outputs are cloned off the static buffers, so a later replay must not
    # rewrite what an earlier one returned.
    preserved_codes = actual_codes.clone()
    runner(hidden + 1, c0 + 1, seeds, positions, forced, no_replay)
    device_module.synchronize()
    assert torch.equal(actual_codes, preserved_codes)


@pytest.mark.accelerator
def test_rvq_cuda_graph_declines_a_batch_larger_than_every_bucket() -> None:
    device, backend = rvq_graph_target()

    def depth_forward(hidden, c0, seeds, positions, forced, replay):
        del c0, seeds, positions, replay
        zeros = torch.zeros((hidden.shape[0], 8), device=hidden.device)
        return zeros, hidden, forced[:, :7]

    runner = RVQDepthCudaGraphRunner(
        forward=depth_forward,
        backend=backend,
        device=device,
        dtype=torch.bfloat16,
        hidden_size=16,
        num_codebooks=8,
        buckets=[1, 2],
    )

    oversized = torch.zeros((3, 16), device=device, dtype=torch.bfloat16)
    zeros = torch.zeros((3,), dtype=torch.long, device=device)
    forced = torch.zeros((3, 8), dtype=torch.long, device=device)
    replay = torch.zeros((), dtype=torch.bool, device=device)
    assert runner(oversized, zeros, zeros, zeros, forced, replay) is None


class TinyCacheBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(16, 16)

    def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x + self.proj(x) * scale


class TinyCacheTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyCacheBlock() for _ in range(4)])

    def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, scale)
        return x


class TinyCacheDiffusion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = TinyCacheTransformer()


@pytest.mark.accelerator
def test_cache_dit_block_adapter_runs_hidden_only_pattern() -> None:
    device = get_test_device_if_eligible()
    model = MiniMaxMusic3DIT.__new__(MiniMaxMusic3DIT)
    torch.nn.Module.__init__(model)
    model.diffusion_transformer = TinyCacheDiffusion().to(device).eval()
    model.enable_cache_dit(
        num_steps=4,
        fn_compute_blocks=1,
        bn_compute_blocks=1,
        max_warmup_steps=1,
        residual_diff_threshold=0.08,
        max_continuous_cached_steps=1,
    )

    x = torch.randn((2, 11, 16), device=device)
    for step in range(4):
        scale = torch.tensor(0.1 + step * 0.01, device=device)
        x = model.diffusion_transformer.transformer(x, scale)

    assert x.shape == (2, 11, 16)
    assert torch.isfinite(x).all()


class ZeroDiffusionTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def _transformer(  # noqa: leading-underscore  # production name
        self, x: torch.Tensor, _t: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        self.calls += 1
        return torch.zeros_like(x)


def tiny_dit() -> MiniMaxMusic3DIT:
    dit = MiniMaxMusic3DIT.__new__(MiniMaxMusic3DIT)
    torch.nn.Module.__init__(dit)
    dit.diffusion_transformer = ZeroDiffusionTransformer()
    dit.bcg_runner = None
    return dit


def test_dit_step_count_is_configurable_without_changing_shape() -> None:
    dit = tiny_dit()
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
    dit = tiny_dit()
    dit.sr_input = 24_000
    dit.sr_output = 44_100
    dit.hop_size_input = 960
    dit.hop_size_output = 512

    assert dit.aligned_mel_length(200) == 689
    assert dit.aligned_mel_length(100) == 344


def test_dit_abort_stops_before_the_next_euler_step() -> None:
    dit = tiny_dit()
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


class FakeAcousticDecoder:
    serial_offload = False

    def __init__(self) -> None:
        self.hidden_shape: tuple[int, ...] | None = None

    def decode_with_state(
        self, hidden: torch.Tensor, **_kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.hidden_shape = tuple(hidden.shape)
        return torch.zeros((2, 16)), torch.zeros((1, 128, 1)), torch.zeros((1, 2048, 1))


def test_acoustic_scheduler_accepts_the_relay_tensor_shape() -> None:
    decoder = FakeAcousticDecoder()
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


def test_engine_builder_selects_native_backbone_without_mutating_legacy_config(
    tmp_path: Path,
) -> None:
    """The native language model is loadable without changing the legacy copy."""
    root = tmp_path / "checkpoint"
    language_model_dir = root / "language_model"
    language_model_dir.mkdir(parents=True)
    (language_model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "hidden_size": 4096})
    )
    (root / "flowmatching_vae.pth").touch()

    legacy_dir = root / "qwen_7B" / "qwen_7B"
    legacy_dir.mkdir(parents=True)
    blob = tmp_path / "config-blob"
    original_config = b'{"model_type":"mixtral","hidden_size":4096}'
    blob.write_bytes(original_config)
    legacy_config_path = legacy_dir / "config.json"
    legacy_config_path.symlink_to(blob)

    builder = MiniMaxMusic3EngineBuilder(enable_async_decode=True)
    resolved = Path(builder.resolve_checkpoint(str(root)))
    load_weights = Qwen3ForCausalLM.load_weights

    builder.pre_infra_setup(str(resolved))
    assert resolved == language_model_dir
    assert builder.model_arch_override is None
    assert legacy_config_path.is_symlink()
    assert legacy_config_path.resolve() == blob
    assert legacy_config_path.read_bytes() == original_config
    assert not (legacy_dir / "config.json.bak").exists()
    assert Qwen3ForCausalLM.load_weights is load_weights


def test_checkpoint_requires_native_qwen3_backbone(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    legacy_dir = root / "qwen_7B" / "qwen_7B"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "config.json").write_text(
        json.dumps({"model_type": "mixtral", "hidden_size": 4096})
    )
    (root / "flowmatching_vae.pth").touch()

    with pytest.raises(ValueError, match="language_model/config.json"):
        MiniMaxMusic3EngineBuilder(enable_async_decode=True).resolve_checkpoint(
            str(root)
        )


def test_checkpoint_rejects_non_qwen3_native_config(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    language_model_dir = root / "language_model"
    language_model_dir.mkdir(parents=True)
    (language_model_dir / "config.json").write_text(
        json.dumps({"model_type": "mixtral", "hidden_size": 4096})
    )
    legacy_dir = root / "qwen_7B" / "qwen_7B"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "config.json").write_text(
        json.dumps({"model_type": "mixtral", "hidden_size": 4096})
    )
    (root / "flowmatching_vae.pth").touch()

    with pytest.raises(ValueError, match="model_type='qwen3'"):
        MiniMaxMusic3EngineBuilder(enable_async_decode=True).resolve_checkpoint(
            str(root)
        )


def test_checkpoint_rejects_hidden_size_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    language_model_dir = root / "language_model"
    language_model_dir.mkdir(parents=True)
    (language_model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "hidden_size": 4096})
    )
    legacy_dir = root / "qwen_7B" / "qwen_7B"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "config.json").write_text(
        json.dumps({"model_type": "mixtral", "hidden_size": 3584})
    )
    (root / "flowmatching_vae.pth").touch()

    with pytest.raises(ValueError, match="hidden_size mismatch"):
        MiniMaxMusic3EngineBuilder(enable_async_decode=True).resolve_checkpoint(
            str(root)
        )


def test_serial_offload_identity_includes_native_backbone_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "checkpoint"
    native_config = root / "language_model" / "config.json"
    native_config.parent.mkdir(parents=True)
    native_config.write_text(json.dumps({"model_type": "qwen3", "hidden_size": 4096}))
    audio_config = root / "qwen_7B" / "qwen_7B" / "config.json"
    audio_config.parent.mkdir(parents=True)
    audio_config.write_text(json.dumps({"model_type": "mixtral", "hidden_size": 4096}))
    (audio_config.parent / "model.safetensors").write_bytes(b"weights")

    builder = MiniMaxMusic3EngineBuilder(
        enable_serial_offload=True,
        serial_offload_source="ram",
        enable_async_decode=True,
    )
    builder.checkpoint_root = str(root)
    coordinator = SerialOffloadCoordinator()
    checkpoint_paths: list[Path] = []

    def record_checkpoint_identity(paths: list[Path]) -> dict[str, str]:
        checkpoint_paths.extend(paths)
        return checkpoint_identity(paths)

    monkeypatch.setattr(
        "sglang_omni.models.minimax_music3.engine_builder.get_coordinator",
        lambda: coordinator,
    )
    monkeypatch.setattr(
        "sglang_omni.models.minimax_music3.engine_builder.checkpoint_identity",
        record_checkpoint_identity,
    )

    builder.setup_model_resources(
        torch.nn.Linear(1, 1),
        None,  # type: ignore[arg-type]
        generation_cuda_graph_enabled=False,
    )

    assert coordinator.ar_residency is not None
    assert native_config in checkpoint_paths


class TinyMusic3Model(nn.Module):
    """The real MiniMax audio modules on a one-layer stand-in for the Qwen3 backbone."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        torch.manual_seed(0)
        hidden_size, audio_vocab_size, num_codebooks = 16, 64, 8
        self.num_codebooks = num_codebooks
        self.audio_vocab_size = audio_vocab_size
        self.mel_token_offset = AUDIO_CODE_OFFSET
        self.stop_mel_token = SPECIAL_TOKEN_IDS["<|audio_end|>"]
        self.frame_embedding_scale = num_codebooks**-0.5
        self.graph_feedback_buffer = None
        self.rvq_depth_graph = None
        self.register_buffer(
            "c0_logit_ids", music3_model.build_c0_logit_ids(self.stop_mel_token, device)
        )
        self.register_buffer(
            "audio_embedding_offsets",
            (torch.arange(num_codebooks - 1) * audio_vocab_size).unsqueeze(0),
        )
        vocab_size = int(self.c0_logit_ids[-1]) + 1
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.audio_embeddings = nn.Embedding(
            audio_vocab_size * (num_codebooks - 1), hidden_size
        )
        self.rvq_decoder = RVQDepthDecoder(
            hidden_size=hidden_size,
            num_layers=1,
            num_heads=2,
            intermediate_size=32,
            audio_vocab_size=audio_vocab_size,
            num_codebooks=num_codebooks,
        )
        self.backbone = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.to(device).eval()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def step(
        self,
        input_embeds: torch.Tensor,
        requests: list[SchedulerRequest],
        stop_frames: dict[int, int],
    ) -> SimpleNamespace:
        """One forward; a pair whose request is listed in stop_frames samples stop there."""
        rows = torch.arange(len(requests), device=input_embeds.device) % 2
        hidden = torch.tanh(self.backbone(input_embeds) + 0.25 * rows[:, None])
        logits = self.lm_head(hidden).float()
        for pair, cond in enumerate(requests[0::2]):
            ar_state = cond.data.ar_state
            frame = ar_state.sampling_position // self.num_codebooks
            if stop_frames.get(ar_state.seed) == frame:
                logits[2 * pair : 2 * pair + 2, self.stop_mel_token] = 1e4
            else:
                pass
        return SimpleNamespace(
            logits_output=SimpleNamespace(
                hidden_states=hidden, next_token_logits=logits
            ),
            next_token_ids=None,
        )


class FinishFlag:
    """The Req fields the runner reads; set by the stand-in output step."""

    def __init__(self) -> None:
        self.done = False
        self.is_retracted = False
        self.to_finish = None

    def finished(self) -> bool:
        return self.done


def seeded_gumbel_sample(
    logits: torch.Tensor, seeds: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """CPU stand-in for the seeded sampler: each row depends only on its own seed and position."""
    drawn = []
    for row, seed, position in zip(logits.float(), seeds.tolist(), positions.tolist()):
        generator = torch.Generator().manual_seed(seed * 1_000_003 + position)
        noise = torch.rand(row.shape, generator=generator).to(row.device)
        drawn.append(torch.argmax(row - torch.log(-torch.log(noise))))
    return torch.stack(drawn)


def music3_runner_for(
    model: TinyMusic3Model,
) -> music3_runner.MiniMaxMusic3ModelRunner:
    runner = music3_runner.MiniMaxMusic3ModelRunner.__new__(
        music3_runner.MiniMaxMusic3ModelRunner
    )
    runner.model = model
    runner.forced_codes_dir = None
    runner.dump_dir = None
    runner.request_data = {}
    runner.serial_offload = SimpleNamespace(
        enabled=False, begin_dit_handoff=lambda request_id: None
    )
    runner.host_staging_buffers = []
    runner.staging_slot = 0
    if model.c0_logit_ids.device.type == "cpu":
        runner.next_host_staging = lambda shape, dtype: torch.empty(shape, dtype=dtype)
    else:
        pass
    return runner


def music3_pairs(limits: list[int]) -> list[list[SchedulerRequest]]:
    pairs = []
    for seed, limit in enumerate(limits):
        uncond = SimpleNamespace(req=FinishFlag(), ar_state=None, cfg_uncond=None)
        cond = SimpleNamespace(
            req=FinishFlag(),
            cfg_uncond=uncond,
            ar_state=music3_runner.MiniMaxMusic3ARState(
                sampling_seed=1000 + seed, seed=seed, decode_limit=limit
            ),
            minimax_state=SimpleNamespace(),
            prompt_tokens=16,
        )
        pairs.append(
            [
                SchedulerRequest(request_id=f"song{seed}", data=cond),
                SchedulerRequest(request_id=f"song{seed}-uncond", data=uncond),
            ]
        )
    return pairs


def process_output(
    runner: music3_runner.MiniMaxMusic3ModelRunner,
    requests: list[SchedulerRequest],
    token_ids: list[int],
    emitted: dict[str, list[int]],
    streamed: dict[str, list[OutgoingMessage]],
) -> None:
    """What output processing does with one step: record ids, finish on stop or
    length, stream the queued windows, and flush the tail of a finished pair."""
    for pair, cond in enumerate(requests[0::2]):
        if cond.data.req.finished():
            continue
        else:
            pass
        token = token_ids[2 * pair]
        emitted.setdefault(cond.request_id, []).append(token)
        ar_state = cond.data.ar_state
        done = (
            token == SPECIAL_TOKEN_IDS["<|audio_end|>"]
            or ar_state.generated_frames >= ar_state.decode_limit
        )
        if done:
            cond.data.req.done = True
            requests[2 * pair + 1].data.req.done = True
        else:
            pass
        messages = streamed.setdefault(cond.request_id, [])
        messages.extend(build_stream_output(cond.request_id, cond.data, None))
        if not done and ar_state.generated_frames == AR_CHUNK_FRAMES + 1:
            assert len(messages) == 1
            assert (
                messages[0].metadata["chunk_idx"],
                messages[0].metadata["start_frame"],
                messages[0].metadata["end_frame"],
                messages[0].metadata["is_final"],
            ) == (0, 0, AR_CHUNK_FRAMES, False)
        else:
            pass
        if done:
            runner.on_request_finished(cond.request_id, cond.data)
            messages.extend(build_stream_output.flush(cond.request_id, cond.data))
        else:
            pass


def launch_event(model: TinyMusic3Model):
    """The completion event the scheduler waits on before resolve, None on CPU."""
    device = model.c0_logit_ids.device
    if device.type == "cpu":
        return None
    else:
        event = torch.get_device_module(device).Event()
        event.record()
        return event


def generate_music3(
    model: TinyMusic3Model,
    pairs: list[list[SchedulerRequest]],
    stop_frames: dict[int, int],
    *,
    lookahead: bool,
) -> tuple[dict[str, list[int]], dict[str, list[OutgoingMessage]]]:
    """Prefill, then decode in scheduler order: with lookahead, step N+1 launches
    from the batch as it stood before step N resolved. Returns the token ids and
    the stream messages of every song."""
    runner = music3_runner_for(model)
    emitted: dict[str, list[int]] = {}
    streamed: dict[str, list[OutgoingMessage]] = {}
    batch = [request for pair in pairs for request in pair]
    prompts = torch.stack(
        [
            torch.randn(16, generator=torch.Generator().manual_seed(row // 2))
            for row in range(len(batch))
        ]
    ).to(model.c0_logit_ids.device)
    result = model.step(prompts, batch, stop_frames)
    runner.advance(result, batch, emit=False)
    process_output(runner, batch, result.next_token_ids.tolist(), emitted, streamed)
    pending = None
    while True:
        batch = [r for pair in pairs if not pair[0].data.req.finished() for r in pair]
        if batch:
            forward_batch = SimpleNamespace(input_embeds=None)
            runner.before_decode(forward_batch, None, batch, is_lookahead=lookahead)
            result = model.step(forward_batch.input_embeds, batch, stop_frames)
        else:
            pass
        if not lookahead:
            if not batch:
                return emitted, streamed
            else:
                pass
            runner.post_decode(result, None, None, batch)
            process_output(
                runner, batch, result.next_token_ids.tolist(), emitted, streamed
            )
            continue
        else:
            pass
        launched = None
        if batch:
            frame = runner.post_decode_launch(result, None, batch)
            launched = (frame, result, batch, launch_event(model))
        else:
            pass
        if pending is not None:
            frame, previous, previous_batch, event = pending
            if event is not None:
                event.synchronize()
            else:
                pass
            runner.post_decode_resolve(frame, previous, None, None, previous_batch)
            process_output(
                runner,
                previous_batch,
                previous.next_token_ids.tolist(),
                emitted,
                streamed,
            )
        else:
            pass
        if launched is None:
            return emitted, streamed
        else:
            pending = launched


def assert_windows_cover_the_song(messages: list[OutgoingMessage], frames: int) -> None:
    """The stream carries exactly the song's windows: the non-final ones emitted
    during decode, then the terminal tail from the flush."""
    assert messages
    got = [
        (
            message.metadata["chunk_idx"],
            message.metadata["start_frame"],
            message.metadata["end_frame"],
            message.metadata["is_final"],
        )
        for message in messages
    ]
    assert got == [
        (window.index, window.start, window.end, window.is_last)
        for window in chunk_windows(frames)
    ]
    assert all(
        message.data.shape[1]
        == message.metadata["end_frame"] - message.metadata["start_frame"]
        for message in messages
    )


def assert_lookahead_matches_sync(
    device: torch.device, limits: list[int], stop_frames: dict[int, int]
) -> dict[str, int]:
    """Lookahead keeps a just-finished pair in one extra batch. The stand-in
    backbone is not batch-invariant, so hidden states agree to rounding here;
    served audio stays bitwise equal under the deterministic default.
    Returns the frame count of every song."""
    sync_pairs, lookahead_pairs = music3_pairs(limits), music3_pairs(limits)
    model = TinyMusic3Model(device)
    with torch.inference_mode():
        sync_ids, sync_stream = generate_music3(
            model, sync_pairs, stop_frames, lookahead=False
        )
        lookahead_ids, lookahead_stream = generate_music3(
            model, lookahead_pairs, stop_frames, lookahead=True
        )

    assert lookahead_ids == sync_ids
    frames = {}
    for sync_pair, lookahead_pair in zip(sync_pairs, lookahead_pairs, strict=True):
        request_id = sync_pair[0].request_id
        expected, got = sync_pair[0].data.ar_state, lookahead_pair[0].data.ar_state
        assert (got.generated_frames, got.finish_reason) == (
            expected.generated_frames,
            expected.finish_reason,
        ), request_id
        frames[request_id] = expected.generated_frames
        assert_windows_cover_the_song(sync_stream[request_id], frames[request_id])
        assert_windows_cover_the_song(lookahead_stream[request_id], frames[request_id])
        for a, b in zip(
            sync_stream[request_id], lookahead_stream[request_id], strict=True
        ):
            assert a.metadata == b.metadata, request_id
            assert torch.allclose(a.data, b.data, rtol=1e-5, atol=1e-6), request_id
    for seed, limit in enumerate(limits):
        stops = seed in stop_frames and stop_frames[seed] <= limit
        assert sync_pairs[seed][0].data.ar_state.finish_reason == (
            "stop" if stops else "length"
        )
    return frames


def cpu_window_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    """The window path waits on the device stream; a CPU stream has nothing to wait for."""
    monkeypatch.setattr(
        torch.cpu.Stream, "synchronize", lambda self: None, raising=False
    )


@pytest.mark.parametrize(
    ("limits", "stop_frames"),
    [
        ([12], {}),
        ([AR_CHUNK_FRAMES + 3], {}),
        ([12, 9, 30], {1: 5}),
        ([30, 30, 30, 30], {0: 3, 3: 17}),
    ],
)
def test_async_decode_commits_the_frames_sync_decode_commits(
    monkeypatch: pytest.MonkeyPatch, limits: list[int], stop_frames: dict[int, int]
) -> None:
    """Launching frame N+1 before frame N resolves must not change any song: same
    token ids, frame counts, finish reasons and streamed windows, including the
    pair that stops or hits its length while a lookahead frame is already in flight."""
    monkeypatch.setattr(music3_runner, "sample_topk_seeded", seeded_gumbel_sample)
    monkeypatch.setattr(music3_model, "sample_topk_seeded", seeded_gumbel_sample)
    cpu_window_sync(monkeypatch)
    assert_lookahead_matches_sync(torch.device("cpu"), limits, stop_frames)


@pytest.mark.accelerator
def test_async_decode_matches_sync_with_the_seeded_sampler_and_windows() -> None:
    """The same parity with the real seeded sampler, across the 200-frame window
    and its hop, so non-final windows stream while a frame is in flight."""
    device = get_test_device_if_eligible()
    frames = assert_lookahead_matches_sync(device, [260, 230, 120], {1: 215, 2: 40})
    assert any(count > AR_CHUNK_FRAMES for count in frames.values())


@torch.inference_mode()
def test_async_decode_skips_a_pair_aborted_before_its_frame_resolves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deferred abort only sets to_finish, so the pair is not finished yet when
    its launched frame resolves. That frame must commit nothing for the cancelled
    pair and queue it no window, while the other pair commits the same frame and
    queues its first window."""
    monkeypatch.setattr(music3_runner, "sample_topk_seeded", seeded_gumbel_sample)
    monkeypatch.setattr(music3_model, "sample_topk_seeded", seeded_gumbel_sample)
    cpu_window_sync(monkeypatch)
    model = TinyMusic3Model(torch.device("cpu"))
    runner = music3_runner_for(model)
    pairs = music3_pairs([2 * AR_CHUNK_FRAMES, 2 * AR_CHUNK_FRAMES])
    batch = [request for pair in pairs for request in pair]

    def step(input_embeds: torch.Tensor) -> SimpleNamespace:
        result = model.step(input_embeds, batch, {})
        result.logits_output.next_token_logits[:, model.stop_mel_token] = -1e4
        return result

    runner.advance(step(torch.randn(len(batch), 16)), batch, emit=False)
    for _ in range(AR_CHUNK_FRAMES):
        forward_batch = SimpleNamespace(input_embeds=None)
        runner.before_decode(forward_batch, None, batch)
        runner.post_decode(step(forward_batch.input_embeds), None, None, batch)
    forward_batch = SimpleNamespace(input_embeds=None)
    runner.before_decode(forward_batch, None, batch, is_lookahead=True)
    result = step(forward_batch.input_embeds)
    frame = runner.post_decode_launch(result, None, batch)
    for request in pairs[0]:
        request.data.req.to_finish = FINISH_ABORT()

    runner.post_decode_resolve(frame, result, None, None, batch)

    cancelled, survivor = pairs[0][0].data.ar_state, pairs[1][0].data.ar_state
    assert not pairs[0][0].data.req.finished()
    assert cancelled.generated_frames == AR_CHUNK_FRAMES
    assert cancelled.pending_chunks == []
    assert survivor.generated_frames == AR_CHUNK_FRAMES + 1
    assert [metadata["chunk_idx"] for _, metadata in survivor.pending_chunks] == [0]


def test_reference_replay_keeps_decode_synchronous() -> None:
    """Replay consumes one reference frame per launch, so it never takes the lookahead."""
    runner = music3_runner_for(TinyMusic3Model(torch.device("cpu")))
    batch = SimpleNamespace(reqs=[])
    assert runner.lookahead_eligible(batch)
    runner.forced_codes_dir = "/references"
    assert not runner.lookahead_eligible(batch)


def test_async_decode_switch_reaches_the_scheduler_and_offload_stays_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parsed override travels through create_ar_executor and the builder to
    the scheduler; the engine build is stubbed."""
    monkeypatch.setattr(torch.backends.cudnn, "enabled", torch.backends.cudnn.enabled)
    monkeypatch.setattr(torch.backends.cuda, "enable_cudnn_sdp", lambda enabled: None)

    def build(builder, model_path, **kwargs) -> SimpleNamespace:
        builder.device = "cpu"
        return SimpleNamespace(**builder.extra_scheduler_kwargs())

    monkeypatch.setattr(MiniMaxMusic3EngineBuilder, "build", build)

    def executor_args(config) -> dict:
        stages = {stage.name: stage for stage in config.stages}
        return resolve_stage_factory_args(stages["minimax_music3_ar"], config, gpu_id=0)

    config = MiniMaxMusic3SingleGPUPipelineConfig(model_path="/models/minimax")
    values = config.model_dump()
    ConfigPath.parse(
        "stages.minimax_music3_ar.factory.enable_async_decode", type(config)
    ).write(values, False)
    disabled = executor_args(type(config)(**values))

    assert disabled["enable_async_decode"] is False
    create = music3_stages.create_ar_executor
    assert create(**disabled).enable_async_decode is False
    assert create(**executor_args(config)).enable_async_decode is True
    assert not create("/models/minimax", enable_serial_offload=True).enable_async_decode
