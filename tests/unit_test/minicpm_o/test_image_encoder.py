# SPDX-License-Identifier: Apache-2.0
"""Tests for the srt-based MiniCPM-o image encoder.

The golden-parity test compares the srt-module encoder against the
checkpoint's remote-code path (``modeling_navit_siglip.SiglipVisionTransformer``
+ ``modeling_minicpmo.Resampler``) on the real checkpoint weights, so it needs
a full checkpoint (weights included) and a CUDA device for the srt vision
attention. Set ``MINICPMO_CHECKPOINT`` or place ``MiniCPM-o-4_6``/
``MiniCPM-o-4_5`` in the repo root; the test skips otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from transformers import PretrainedConfig

from sglang_omni.models.minicpm_o.components.image_encoder import (
    _init_sglang_tp,
    _vision_config_object,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def tp_context(monkeypatch):
    import sglang.srt.layers.dp_attention as dp
    from sglang.srt import server_args
    from sglang.srt.distributed import parallel_state

    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setattr(
        parallel_state, "model_parallel_is_initialized", Mock(return_value=True)
    )
    monkeypatch.setattr(
        parallel_state, "get_tensor_model_parallel_world_size", Mock(return_value=1)
    )
    monkeypatch.setattr(parallel_state, "init_distributed_environment", Mock())
    monkeypatch.setattr(parallel_state, "initialize_model_parallel", Mock())
    monkeypatch.setattr(server_args, "ServerArgs", Mock())
    monkeypatch.setattr(server_args, "set_global_server_args_for_scheduler", Mock())
    monkeypatch.setattr(dp, "_ATTN_TP_SIZE", None, raising=False)
    monkeypatch.setattr(dp, "_ATTN_TP_RANK", None, raising=False)
    return dp, parallel_state, server_args


@pytest.mark.parametrize("legacy_tp_size", [None, 1, 8])
def test_init_sglang_tp_reuses_initialized_context(
    tp_context, monkeypatch, legacy_tp_size
):
    dp, parallel_state, server_args = tp_context
    if legacy_tp_size is not None:
        monkeypatch.setattr(dp, "_ATTN_TP_SIZE", legacy_tp_size, raising=False)
    else:
        monkeypatch.delattr(dp, "_ATTN_TP_SIZE")

    _init_sglang_tp()

    parallel_state.get_tensor_model_parallel_world_size.assert_called_once_with()
    parallel_state.init_distributed_environment.assert_not_called()
    parallel_state.initialize_model_parallel.assert_not_called()
    server_args.ServerArgs.assert_not_called()
    server_args.set_global_server_args_for_scheduler.assert_not_called()
    assert getattr(dp, "_ATTN_TP_SIZE", None) == legacy_tp_size
    assert dp._ATTN_TP_RANK is None


def test_init_sglang_tp_rejects_actual_multi_rank_context(tp_context, monkeypatch):
    dp, parallel_state, server_args = tp_context
    parallel_state.get_tensor_model_parallel_world_size.return_value = 2
    monkeypatch.setattr(dp, "_ATTN_TP_SIZE", 1, raising=False)

    with pytest.raises(RuntimeError, match="already initialized tp_size=2"):
        _init_sglang_tp()

    server_args.ServerArgs.assert_not_called()
    server_args.set_global_server_args_for_scheduler.assert_not_called()
    parallel_state.init_distributed_environment.assert_not_called()
    parallel_state.initialize_model_parallel.assert_not_called()


def test_init_sglang_tp_initializes_standalone_context(tp_context):
    dp, parallel_state, server_args = tp_context
    parallel_state.model_parallel_is_initialized.return_value = False

    _init_sglang_tp()

    server_args.ServerArgs.assert_called_once_with(model_path="dummy")
    server_args.set_global_server_args_for_scheduler.assert_called_once_with(
        server_args.ServerArgs.return_value
    )
    parallel_state.init_distributed_environment.assert_called_once_with(
        backend="nccl", world_size=1, rank=0, local_rank=0
    )
    parallel_state.initialize_model_parallel.assert_called_once_with(
        tensor_model_parallel_size=1
    )
    assert dp._ATTN_TP_SIZE == 1
    assert dp._ATTN_TP_RANK == 0


def _checkpoint_dir() -> Path | None:
    env = os.environ.get("MINICPMO_CHECKPOINT")
    candidates = [Path(env)] if env else []
    candidates += [REPO_ROOT / "MiniCPM-o-4_6", REPO_ROOT / "MiniCPM-o-4_5"]
    for path in candidates:
        if (path / "model.safetensors.index.json").exists() and (
            path / "modeling_navit_siglip.py"
        ).exists():
            return path
    return None


def test_patch_attn_mask_vectorization_matches_loop() -> None:
    patch_counts = torch.tensor([6, 1, 4])
    max_patches = int(patch_counts.max())
    ref = torch.zeros(3, 1, max_patches, dtype=torch.bool)
    for i in range(3):
        ref[i, 0, : patch_counts[i]] = True
    got = (torch.arange(max_patches)[None, :] < patch_counts[:, None]).unsqueeze(1)
    assert torch.equal(got, ref)


def test_vision_config_object_preserves_config_instances() -> None:
    vision_config = PretrainedConfig(hidden_size=128, patch_size=14)
    config = PretrainedConfig(vision_config=vision_config)
    assert _vision_config_object(config) is vision_config


def test_vision_config_object_converts_shim_dict() -> None:
    config = PretrainedConfig(vision_config={"hidden_size": 128, "patch_size": 14})
    vision_config = _vision_config_object(config)
    assert isinstance(vision_config, PretrainedConfig)
    assert vision_config.hidden_size == 128
    assert vision_config.patch_size == 14


def _build_remote_encoder(checkpoint: Path, device: torch.device, dtype: torch.dtype):
    """The pre-srt remote-code path this component replaced, as golden."""
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    from sglang_omni.models.weight_loader import load_module

    model_dir = str(checkpoint)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    siglip_cls = get_class_from_dynamic_module(
        "modeling_navit_siglip.SiglipVisionTransformer", model_dir
    )
    resampler_cls = get_class_from_dynamic_module(
        "modeling_minicpmo.Resampler", model_dir
    )
    vision_config = config.vision_config
    vision_config._attn_implementation = "eager"
    vpm = siglip_cls(vision_config)
    if getattr(config, "drop_vision_last_layer", False):
        vpm.encoder.layers = vpm.encoder.layers[:-1]
    vpm = load_module(vpm, model_dir, prefix=("vpm.",), dtype=dtype, device=str(device))

    embed_dim = config.hidden_size
    resampler = resampler_cls(
        num_queries=config.query_num,
        embed_dim=embed_dim,
        num_heads=embed_dim // 128,
        kv_dim=vision_config.hidden_size,
        adaptive=True,
    )
    resampler = load_module(
        resampler, model_dir, prefix=("resampler.",), dtype=dtype, device=str(device)
    )
    resampler._set_2d_pos_cache(resampler.max_size, device=str(device))
    return config, vpm.eval(), resampler.eval()


def _remote_forward(vpm, resampler, pixel_values, tgt_sizes, device, dtype):
    from torch.nn.utils.rnn import pad_sequence

    tgt_sizes = tgt_sizes.to(device, dtype=torch.int32)
    all_pixel_values = [
        v.to(device, dtype=dtype).flatten(end_dim=1).permute(1, 0) for v in pixel_values
    ]
    all_pixel_values = pad_sequence(
        all_pixel_values, batch_first=True, padding_value=0.0
    )
    B, L, _ = all_pixel_values.shape
    all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)
    patch_counts = tgt_sizes[:, 0] * tgt_sizes[:, 1]
    max_patches = int(patch_counts.max().item())
    patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
    for i in range(B):
        patch_attn_mask[i, 0, : patch_counts[i]] = True
    vision_embedding = vpm(
        all_pixel_values,
        patch_attention_mask=patch_attn_mask,
        tgt_sizes=tgt_sizes,
    ).last_hidden_state
    return resampler(vision_embedding, tgt_sizes)


def test_golden_parity_vs_remote_code() -> None:
    checkpoint = _checkpoint_dir()
    if checkpoint is None:
        pytest.skip("no MiniCPM-o checkpoint with weights")
    if not torch.cuda.is_available():
        pytest.skip("srt vision attention requires CUDA")

    from sglang_omni.models.minicpm_o.components.image_encoder import (
        MiniCPMOImageEncoder,
    )

    # srt VisionAttention's flash-attn backend only supports fp16/bf16, so the
    # srt encoder cannot run an fp32 bitwise-parity pass. Instead, both the
    # srt path and the remote-code path run in bf16 against an fp32
    # remote-code golden, and the srt path's error must stay within the
    # remote path's own bf16 rounding error (plus slack) — i.e. the module
    # swap adds no error beyond dtype noise.
    device = torch.device("cuda")

    torch.manual_seed(0)
    config, vpm32, resampler32 = _build_remote_encoder(
        checkpoint, device, torch.float32
    )
    patch = config.vision_config.patch_size
    # Variable-resolution slices (h, w) in patch units, incl. a 1-patch-high one.
    tgt_sizes = torch.tensor([[8, 12], [3, 5], [1, 9]], dtype=torch.int32)
    pixel_values = [
        torch.randn(3, patch, int(h * w) * patch) for h, w in tgt_sizes.tolist()
    ]
    with torch.no_grad():
        golden = _remote_forward(
            vpm32, resampler32, pixel_values, tgt_sizes, device, torch.float32
        ).float()
    del vpm32, resampler32
    torch.cuda.empty_cache()

    config, vpm16, resampler16 = _build_remote_encoder(
        checkpoint, device, torch.bfloat16
    )
    with torch.no_grad():
        remote_bf16 = _remote_forward(
            vpm16, resampler16, pixel_values, tgt_sizes, device, torch.bfloat16
        ).float()
    del vpm16, resampler16
    torch.cuda.empty_cache()

    native = MiniCPMOImageEncoder(str(checkpoint), device="cuda", dtype="bfloat16")
    with torch.no_grad():
        got = (
            native(pixel_values=pixel_values, tgt_sizes=tgt_sizes)["image_embeds"]
            .float()
            .view(golden.shape)
        )

    remote_err = (remote_bf16 - golden).abs()
    native_err = (got - golden).abs()
    assert native_err.mean() <= remote_err.mean() * 1.5, (
        f"srt path error {native_err.mean():.6f} exceeds remote bf16 "
        f"rounding error {remote_err.mean():.6f}"
    )
    cos = torch.nn.functional.cosine_similarity(got, golden, dim=-1)
    remote_cos = torch.nn.functional.cosine_similarity(remote_bf16, golden, dim=-1)
    assert cos.min() >= remote_cos.min() - 0.01, (
        f"srt path cos_min {cos.min():.6f} below remote bf16 "
        f"cos_min {remote_cos.min():.6f}"
    )
