# SPDX-License-Identifier: Apache-2.0
"""Opt-in single-rank SGLang decoder equivalence test."""

import os
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp


def _native_worker(rank, directory, cfg, dtype):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    from sglang_omni.models.llada2_uni.components.decoder_model import (
        ZImageTransformer2DModelWrapper,
    )
    from sglang_omni.models.llada2_uni.components.decoder_runtime import (
        initialize_decoder_runtime,
    )

    path = Path(directory)
    with initialize_decoder_runtime(
        directory,
        gpu_id=0,
        dtype=dtype,
        attention_backend="torch_sdpa",
        dist_timeout=120,
    ) as runtime:
        with runtime.compute_context(), torch.inference_mode():
            inputs = torch.load(
                path / "inputs.pt", weights_only=True, map_location=runtime.device
            )
            model = ZImageTransformer2DModelWrapper(
                directory,
                cfg,
                runtime.device,
                dtype,
                backend="sglang",
                runtime=runtime,
            )
            output = torch.stack(
                model(
                    list(inputs["x"].to(dtype).unbind(0)),
                    inputs["t"],
                    list(inputs["cap"].to(dtype).unbind(0)),
                    return_dict=False,
                )[0]
            )
            torch.save(output.cpu(), path / "native.pt")
    assert not torch.distributed.is_initialized()


@pytest.mark.skipif(
    os.environ.get("LLADA_DECODER_GPU_TEST") != "1",
    reason="opt-in real native GPU test",
)
def test_sglang_backend_matches_diffusers(tmp_path, monkeypatch):
    from diffusers.models.transformers.transformer_z_image import (
        ZImageTransformer2DModel,
    )
    from safetensors.torch import save_file

    from sglang_omni.models.llada2_uni.components.decoder_model import decoder_config

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    dtype = torch.bfloat16
    cfg = decoder_config(
        {
            "dim": 768,
            "n_layers": 1,
            "n_refiner_layers": 1,
            "n_heads": 6,
            "n_kv_heads": 6,
            "cap_feat_dim": 16,
            "axes_dims": (32, 48, 48),
            "axes_lens": (128, 64, 64),
        }
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        reference = ZImageTransformer2DModel(**cfg).eval()
        torch.nn.init.normal_(reference.x_pad_token, std=0.01)
        torch.nn.init.normal_(reference.cap_pad_token, std=0.01)
        inputs = {
            "x": torch.randn(2, 16, 1, 16, 16),
            "cap": torch.randn(2, 32, 16),
            "t": torch.tensor([0.125, 0.875]),
        }
    save_file(
        {
            name.replace("cap_embedder.", "semantic_embedder."): value.contiguous()
            for name, value in reference.state_dict().items()
        },
        str(tmp_path / "model.safetensors"),
    )
    torch.save(inputs, tmp_path / "inputs.pt")
    with torch.inference_mode():
        expected = torch.stack(
            reference(
                x=list(inputs["x"].unbind(0)),
                t=inputs["t"],
                cap_feats=list(inputs["cap"].unbind(0)),
                return_dict=False,
            )[0]
        )
    del reference

    mp.spawn(
        _native_worker,
        args=(str(tmp_path), cfg, dtype),
        nprocs=1,
        join=True,
    )
    actual = torch.load(tmp_path / "native.pt", weights_only=True)
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)
