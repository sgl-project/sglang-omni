# SPDX-License-Identifier: Apache-2.0
"""Tests for the srt-based MiniCPM-o image encoder.

The golden-parity test compares the srt-module encoder against the
checkpoint's remote-code path (modeling_navit_siglip.SiglipVisionTransformer
+ modeling_minicpmo.Resampler) on the real checkpoint weights, so it needs
a full checkpoint (weights included) and a CUDA device for the srt vision
attention. Set MINICPMO_CHECKPOINT or place MiniCPM-o-4_6 / MiniCPM-o-4_5
in the repo root; the test skips otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest
import torch
from sglang.srt import runtime_context
from sglang.srt.distributed import parallel_state
from sglang.srt.runtime_context import ParallelContext, RuntimeContext, SpawnRanks
from sglang.srt.server_args import ServerArgs

from sglang_omni.models.minicpm_o.components import image_encoder
from sglang_omni.models.minicpm_o.components.image_encoder import MiniCPMOImageEncoder
from sglang_omni.platforms.cuda import CUDAOmniPlatform
from sglang_omni.platforms.interface import OmniPlatform
from sglang_omni.platforms.xpu import XPUOmniPlatform
from sglang_omni.utils.port_claim import release_tcp_port

REPO_ROOT = Path(__file__).resolve().parents[3]


def checkpoint_dir() -> Path | None:
    env = os.environ.get("MINICPMO_CHECKPOINT")
    candidates = [Path(env)] if env else []
    candidates += [REPO_ROOT / "MiniCPM-o-4_6", REPO_ROOT / "MiniCPM-o-4_5"]
    for path in candidates:
        if (path / "model.safetensors.index.json").exists() and (
            path / "modeling_navit_siglip.py"
        ).exists():
            return path
    return None


def test_padding_does_not_change_image_embeddings() -> None:
    batch_size = 2
    encoder = object.__new__(MiniCPMOImageEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.device = torch.device("cpu")
    encoder.dtype = torch.float32
    encoder.vision_batch_size = batch_size

    def run_vpm(pixel_values, patch_attn_mask, tgt_sizes, patch_counts_cpu):
        features = pixel_values.mean(dim=1)
        pooled = (features * patch_attn_mask).sum(dim=-1) / patch_attn_mask.sum(dim=-1)
        return pooled.unsqueeze(-1)

    encoder.run_vpm = run_vpm
    encoder.resampler = lambda features, tgt_sizes: features
    tgt_sizes = torch.tensor([[1, 6], [1, 1], [1, 4]], dtype=torch.int32)
    pixel_values = [
        torch.full((3, 1, count), float(i + 1)) for i, count in enumerate([6, 1, 4])
    ]

    batched = encoder(pixel_values=pixel_values, tgt_sizes=tgt_sizes)["image_embeds"]
    individual = torch.cat(
        [
            encoder(pixel_values=[pixels], tgt_sizes=tgt_sizes[i : i + 1])[
                "image_embeds"
            ]
            for i, pixels in enumerate(pixel_values)
        ]
    )

    torch.testing.assert_close(batched, individual)
    torch.testing.assert_close(batched[:, 0], torch.tensor([1.0, 2.0, 3.0]))


class DistributedRecorder:
    """Stands in for SGLang's process-group setup and records what it was asked to build."""

    def __init__(self) -> None:
        self.called_functions: list[str] = []
        self.distributed_arguments: dict[str, int | str] = {}
        self.server_args_at_init: ServerArgs | None = None

    def init_distributed_environment(
        self,
        world_size: int = -1,
        rank: int = -1,
        distributed_init_method: str = "env://",
        local_rank: int = -1,
        backend: str = "nccl",
        timeout: int | None = None,
        moe_a2a_backend: str | None = None,
        recovered_rank: bool = False,
        max_world_size: int | None = None,
    ) -> None:
        self.called_functions.append("init_distributed_environment")
        self.distributed_arguments = {
            "world_size": world_size,
            "rank": rank,
            "local_rank": local_rank,
            "backend": backend,
        }
        self.server_args_at_init = runtime_context.get_server_args()

    def initialize_model_parallel(
        self,
        backend: str | None = None,
        duplicate_tp_group: bool = False,
        enable_symm_mem: bool = False,
        recovered_rank: bool = False,
        rank_offset: int = 0,
        max_world_size: int | None = None,
    ) -> None:
        self.called_functions.append("initialize_model_parallel")


@pytest.fixture
def distributed_recorder(monkeypatch: pytest.MonkeyPatch) -> DistributedRecorder:
    recorder = DistributedRecorder()
    monkeypatch.setattr(
        parallel_state,
        "init_distributed_environment",
        recorder.init_distributed_environment,
    )
    monkeypatch.setattr(
        parallel_state, "initialize_model_parallel", recorder.initialize_model_parallel
    )
    return recorder


def thinker_runtime_context(tensor_parallel_size: int) -> RuntimeContext:
    return runtime_context.create_context(
        ServerArgs(model_path="dummy", tp_size=tensor_parallel_size),
        role="scheduler",
        ranks=SpawnRanks(world_rank=0),
    )


@pytest.mark.parametrize(
    ("platform", "distributed_backend"),
    [(XPUOmniPlatform(), "xccl"), (CUDAOmniPlatform(), "nccl")],
    ids=["xpu", "cuda"],
)
@pytest.mark.parametrize("is_runtime_context_published", [False, True])
def test_standalone_tensor_parallel_init_builds_one_rank_on_the_platform_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    distributed_recorder: DistributedRecorder,
    platform: OmniPlatform,
    distributed_backend: str,
    is_runtime_context_published: bool,
) -> None:
    """The encoder hardcoded nccl, which XPU cannot initialize; it must also keep a runtime context it did not publish."""
    monkeypatch.setattr(image_encoder, "current_platform", platform)
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: False)
    monkeypatch.setenv("SGLANG_OMNI_PORT_CLAIM_DIR", str(tmp_path))
    scoped_runtime_context = (
        thinker_runtime_context(tensor_parallel_size=1)
        if is_runtime_context_published
        else RuntimeContext(parallel=ParallelContext())
    )
    published_server_args = (
        scoped_runtime_context.server_args if is_runtime_context_published else None
    )

    with (
        mock.patch.dict(os.environ),
        runtime_context.use_context(scoped_runtime_context),
    ):
        os.environ.pop("MASTER_ADDR", None)
        os.environ.pop("MASTER_PORT", None)
        image_encoder.init_sglang_tp()
        master_address = os.environ["MASTER_ADDR"]
        master_port = int(os.environ["MASTER_PORT"])

    assert distributed_recorder.called_functions == [
        "init_distributed_environment",
        "initialize_model_parallel",
    ]
    assert distributed_recorder.distributed_arguments == {
        "world_size": 1,
        "rank": 0,
        "local_rank": 0,
        "backend": distributed_backend,
    }
    if is_runtime_context_published:
        assert distributed_recorder.server_args_at_init is published_server_args
    else:
        assert distributed_recorder.server_args_at_init.tp_size == 1
    assert master_address == "127.0.0.1"
    assert (tmp_path / str(master_port)).exists()
    release_tcp_port(master_port)


@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
def test_an_initialized_tensor_parallel_group_is_reused_only_at_size_one(
    monkeypatch: pytest.MonkeyPatch,
    distributed_recorder: DistributedRecorder,
    tensor_parallel_size: int,
) -> None:
    """The encoder shares the thinker's process group, so it must not rebuild it, and its replicated weights cannot join a wider one."""
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: True)

    with runtime_context.use_context(
        thinker_runtime_context(tensor_parallel_size=tensor_parallel_size)
    ):
        if tensor_parallel_size == 1:
            image_encoder.init_sglang_tp()
        else:
            with pytest.raises(RuntimeError, match="requires tp_size=1.*tp_size=2"):
                image_encoder.init_sglang_tp()

    assert distributed_recorder.called_functions == []


def build_remote_encoder(checkpoint: Path, device: torch.device, dtype: torch.dtype):
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
    vision_config._attn_implementation = (
        "eager"  # noqa: leading-underscore  # production name
    )
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
    resampler._set_2d_pos_cache(
        resampler.max_size, device=str(device)
    )  # noqa: leading-underscore  # upstream name
    return config, vpm.eval(), resampler.eval()


def remote_forward(vpm, resampler, pixel_values, tgt_sizes, device, dtype):
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
    checkpoint = checkpoint_dir()
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
    config, vpm32, resampler32 = build_remote_encoder(checkpoint, device, torch.float32)
    patch = config.vision_config.patch_size
    # Variable-resolution slices (h, w) in patch units, incl. a 1-patch-high one.
    tgt_sizes = torch.tensor([[8, 12], [3, 5], [1, 9]], dtype=torch.int32)
    pixel_values = [
        torch.randn(3, patch, int(h * w) * patch) for h, w in tgt_sizes.tolist()
    ]
    with torch.no_grad():
        golden = remote_forward(
            vpm32, resampler32, pixel_values, tgt_sizes, device, torch.float32
        ).float()
    del vpm32, resampler32
    torch.cuda.empty_cache()

    config, vpm16, resampler16 = build_remote_encoder(
        checkpoint, device, torch.bfloat16
    )
    with torch.no_grad():
        remote_bf16 = remote_forward(
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
