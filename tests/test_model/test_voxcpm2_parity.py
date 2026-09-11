# SPDX-License-Identifier: Apache-2.0
"""Opt-in checkpoint parity for the VoxCPM2 ports against OpenBMB/VoxCPM.

The AR stacks run inside SGLang and are covered end to end; what this pins is
the hand-ported half - AudioVAE, local encoder, local DiT and the flow-matching
sampler - where a transcription slip produces plausible noise rather than an
error. Run it with the upstream package checked out:

    VOXCPM2_UPSTREAM_SOURCE=/path/to/VoxCPM \\
    VOXCPM2_PARITY_CHECKPOINT=openbmb/VoxCPM2 \\
    pytest tests/test_model/test_voxcpm2_parity.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.accelerator

_ATOL = 2e-3
_RTOL = 2e-3


def _assert_close(ours: torch.Tensor, theirs: torch.Tensor, what: str) -> None:
    ours = ours.detach().float().cpu()
    theirs = theirs.detach().float().cpu()
    assert ours.shape == theirs.shape, f"{what}: {ours.shape} vs {theirs.shape}"
    torch.testing.assert_close(ours, theirs, atol=_ATOL, rtol=_RTOL, msg=what)


@pytest.fixture(scope="module")
def parity_models():
    source = os.environ.get("VOXCPM2_UPSTREAM_SOURCE")
    checkpoint = os.environ.get("VOXCPM2_PARITY_CHECKPOINT")
    if not source or not checkpoint:
        pytest.skip(
            "Set VOXCPM2_UPSTREAM_SOURCE and VOXCPM2_PARITY_CHECKPOINT "
            "for real checkpoint parity"
        )
    if not torch.cuda.is_available():
        pytest.skip("VoxCPM2 checkpoint parity requires CUDA")

    sys.path.insert(0, str(Path(source) / "src"))
    from voxcpm.model.voxcpm2 import VoxCPM2Model

    from sglang_omni.models.voxcpm2.hf_config import load_voxcpm2_config
    from sglang_omni.models.voxcpm2.stages import _load_audio_vae
    from sglang_omni.utils.checkpoint import resolve_checkpoint

    checkpoint = resolve_checkpoint(checkpoint)
    config = load_voxcpm2_config(checkpoint)
    upstream = VoxCPM2Model.from_local(checkpoint, device="cuda:0")
    upstream.eval()

    ours = {
        "audio_vae": _load_audio_vae(checkpoint, config, device="cuda:0"),
        "config": config,
        "checkpoint": checkpoint,
    }
    return upstream, ours


def test_audio_vae_encode_matches_upstream(parity_models):
    upstream, ours = parity_models
    sample_rate = ours["config"].sample_rate
    waveform = torch.randn(1, 1, sample_rate, device="cuda:0")

    theirs = upstream.audio_vae.encode(waveform, sample_rate)
    mine = ours["audio_vae"].encode(waveform, sample_rate)
    _assert_close(mine, theirs, "AudioVAE encode")


def test_audio_vae_decode_matches_upstream(parity_models):
    upstream, ours = parity_models
    latent_dim = ours["config"].latent_dim
    latents = torch.randn(1, latent_dim, 32, device="cuda:0")

    theirs = upstream.audio_vae.decode(latents)
    mine = ours["audio_vae"].decode(latents)
    _assert_close(mine, theirs, "AudioVAE decode")


def test_local_encoder_matches_upstream(parity_models):
    upstream, ours = parity_models
    config = ours["config"]
    patches = torch.randn(1, 3, config.patch_size, config.feat_dim, device="cuda:0").to(
        next(upstream.feat_encoder.parameters()).dtype
    )

    theirs = upstream.feat_encoder(patches)
    mine = _our_engine_module(ours, "feat_encoder")(patches)
    _assert_close(mine, theirs, "local encoder")


def test_local_dit_matches_upstream(parity_models):
    upstream, ours = parity_models
    config = ours["config"]
    dtype = next(upstream.feat_decoder.parameters()).dtype
    dit_hidden = config.dit["hidden_dim"] * 2

    x = torch.randn(1, config.feat_dim, config.patch_size, device="cuda:0", dtype=dtype)
    mu = torch.randn(1, dit_hidden, device="cuda:0", dtype=dtype)
    cond = torch.randn_like(x)
    t = torch.rand(1, device="cuda:0", dtype=dtype)
    dt = torch.zeros_like(t)

    theirs = upstream.feat_decoder.estimator(x, mu, t, cond, dt)
    mine = _our_engine_module(ours, "feat_decoder").estimator(x, mu, t, cond, dt)
    _assert_close(mine, theirs, "local DiT")


def test_flow_sampler_matches_upstream(parity_models):
    """Same seed, same recipe: the euler path must land on the same latents."""
    upstream, ours = parity_models
    config = ours["config"]
    dtype = next(upstream.feat_decoder.parameters()).dtype
    dit_hidden = config.dit["hidden_dim"] * 2

    mu = torch.randn(1, dit_hidden, device="cuda:0", dtype=dtype)
    cond = torch.randn(
        1, config.feat_dim, config.patch_size, device="cuda:0", dtype=dtype
    )

    torch.manual_seed(1234)
    theirs = upstream.feat_decoder(
        mu=mu, n_timesteps=10, patch_size=config.patch_size, cond=cond, cfg_value=2.0
    )
    torch.manual_seed(1234)
    mine = _our_engine_module(ours, "feat_decoder")(
        mu=mu, n_timesteps=10, patch_size=config.patch_size, cond=cond, cfg_value=2.0
    )
    _assert_close(mine, theirs, "flow-matching sampler")


def _our_engine_module(ours: dict, name: str):
    """Build the AR model's ported submodules without standing up SGLang."""
    cached = ours.get(f"_{name}")
    if cached is not None:
        return cached

    from sglang_omni.models.weight_loader import load_module

    config = ours["config"]
    if name == "feat_encoder":
        from sglang_omni.models.voxcpm2.components.local_encoder import VoxCPMLocEnc
        from sglang_omni.models.voxcpm2.sglang_model import _local_config

        module = VoxCPMLocEnc(
            _local_config(_lm_namespace(config), config.encoder),
            input_dim=config.feat_dim,
        )
    else:
        from sglang_omni.models.voxcpm2.components.cfm import CfmConfig, UnifiedCFM
        from sglang_omni.models.voxcpm2.components.local_dit import VoxCPMLocDiT
        from sglang_omni.models.voxcpm2.sglang_model import _local_config

        module = UnifiedCFM(
            in_channels=config.feat_dim,
            cfm_params=CfmConfig(**config.cfm),
            estimator=VoxCPMLocDiT(
                _local_config(_lm_namespace(config), config.dit),
                in_channels=config.feat_dim,
            ),
            mean_mode=config.dit_mean_mode,
        )

    module = load_module(module, ours["checkpoint"], prefix=f"{name}.", device="cuda:0")
    ours[f"_{name}"] = module
    return module


def _lm_namespace(config):
    from types import SimpleNamespace

    return SimpleNamespace(**config.lm)
